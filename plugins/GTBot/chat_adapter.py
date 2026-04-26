from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator, Iterator
from types import MethodType
from typing import Any, Callable, TypedDict, cast

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables.config import run_in_executor
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _construct_lc_result_from_responses_api,
    _handle_openai_api_error,
    _handle_openai_bad_request,
    _is_pydantic_class,
    _convert_responses_chunk_to_generation_chunk,
)
from pydantic import SecretStr
from requests.exceptions import HTTPError


SUPPORTED_CHAT_PROVIDER_TYPES = frozenset(
    {"openai_compatible", "openai_responses", "anthropic", "gemini", "dashscope"}
)
_RAW_RESPONSE_BODY_TEXT_LIMIT = 12_000
_RAW_RESPONSE_COLLECTION_LIMIT = 50
_RAW_RESPONSE_MAX_DEPTH = 4
_REDACTED_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"}
)

RetryWrappedCallable = Callable[..., Any]
RetryDecorator = Callable[[RetryWrappedCallable], RetryWrappedCallable]
RetryFactory = Callable[[Any], RetryDecorator]


class RawResponsePayload(TypedDict):
    """描述兼容层写入消息对象中的原始响应摘要。

    该结构面向项目内部消费，统一封装 OpenAI 家族请求返回的关键原始信息，
    方便日志、排障与后续插件读取，而不要求调用方理解底层 SDK 的原始响应对象。
    """

    provider_type: str
    api_style: str
    status_code: int | None
    headers: dict[str, str]
    body_json: dict[str, Any] | list[Any] | str | int | float | bool | None
    body_text: str | None
    request_id: str | None


def _safe_read_mapping_value(payload: Any, key: str) -> Any:
    """安全读取 DashScope 响应中的字段，避免缺失属性被错误包装成 `KeyError`。

    DashScope SDK 的响应对象同时实现了映射访问和属性访问，但缺失属性时会从
    `__getattr__` 直接抛出 `KeyError`，与常见 Python 对象的 `AttributeError`
    约定不一致。这里统一走映射读取并在失败时返回 `None`，避免异常路径再次崩溃。

    Args:
        payload: DashScope SDK 返回的响应对象或普通映射。
        key: 待读取字段名。

    Returns:
        成功时返回字段值，缺失或读取失败时返回 `None`。
    """

    try:
        if isinstance(payload, dict):
            return payload.get(key)
        return payload[key]
    except Exception:
        return None


def _safe_check_tongyi_response(resp: Any) -> Any:
    """校验 Tongyi/DashScope 响应，并在错误场景下抛出兼容 `requests` 的异常。

    `langchain_community` 当前会把 DashScope 的响应对象直接作为 `HTTPError`
    的 `response` 参数传给 `requests`，而该对象缺少 `request` 属性且会错误地
    抛出 `KeyError`，导致真实错误信息被覆盖。本函数保留原有 200/400/401/5xx
    的语义，但不再把原始响应对象传入 `HTTPError` 构造器。

    Args:
        resp: DashScope SDK 返回的响应对象。

    Returns:
        当状态码为 200 时，原样返回响应对象。

    Raises:
        ValueError: 当 DashScope 返回 400 或 401 类客户端错误时抛出。
        HTTPError: 当 DashScope 返回其他非 200 状态码时抛出，以保留上游重试语义。
    """

    status_code = _safe_read_mapping_value(resp, "status_code")
    request_id = _safe_read_mapping_value(resp, "request_id")
    error_code = _safe_read_mapping_value(resp, "code")
    message = _safe_read_mapping_value(resp, "message")

    if status_code == 200:
        return resp
    if status_code in {400, 401}:
        raise ValueError(
            f"request_id: {request_id} \n "
            f"status_code: {status_code} \n "
            f"code: {error_code} \n message: {message}"
        )
    raise HTTPError(
        f"request_id: {request_id} \n "
        f"status_code: {status_code} \n "
        f"code: {error_code} \n message: {message}"
    )


def _patch_tongyi_error_handling(model: BaseChatModel, tongyi_mod: Any) -> BaseChatModel:
    """为 `ChatTongyi` 实例注入安全的错误处理逻辑。

    该补丁仅替换 Tongyi 的重试入口，不改变正常成功响应和流式增量处理逻辑。
    这样可以在不修改第三方包源码的前提下，修复 DashScope 错误响应触发
    `KeyError('request')` 的兼容性问题。

    Args:
        model: 已实例化的 `ChatTongyi` 模型对象。
        tongyi_mod: `langchain_community.chat_models.tongyi` 模块。

    Returns:
        已注入安全错误处理逻辑的模型对象；若当前实例不具备对应方法，则原样返回。
    """

    if getattr(model, "_gtbot_safe_tongyi_error_patched", False):
        return model

    retry_factory = getattr(tongyi_mod, "_create_retry_decorator", None)
    if not callable(retry_factory):
        return model
    typed_retry_factory = cast(RetryFactory, retry_factory)

    def completion_with_retry(self: Any, **kwargs: Any) -> Any:
        """使用安全错误包装执行 Tongyi 非流式请求，并保留原始重试策略。"""

        retry_decorator = typed_retry_factory(self)

        @retry_decorator
        def _completion_with_retry(**_kwargs: Any) -> Any:
            resp = self.client.call(**_kwargs)
            return _safe_check_tongyi_response(resp)

        return _completion_with_retry(**kwargs)

    def stream_completion_with_retry(self: Any, **kwargs: Any) -> Any:
        """使用安全错误包装执行 Tongyi 流式请求，并保留原始 delta 计算逻辑。"""

        retry_decorator = typed_retry_factory(self)

        @retry_decorator
        def _stream_completion_with_retry(**_kwargs: Any) -> Any:
            responses = self.client.call(**_kwargs)
            prev_resp = None

            for resp in responses:
                if _kwargs.get("stream") and not _kwargs.get("incremental_output", False):
                    resp_copy = json.loads(json.dumps(resp))
                    if resp_copy.get("output") and resp_copy["output"].get("choices"):
                        choice = resp_copy["output"]["choices"][0]
                        message = choice["message"]
                        if isinstance(message.get("content"), list):
                            content_text = "".join(
                                item.get("text", "") for item in message["content"] if isinstance(item, dict)
                            )
                            message["content"] = content_text
                        resp = resp_copy
                    if prev_resp is None:
                        delta_resp = resp
                    else:
                        delta_resp = self.subtract_client_response(resp, prev_resp)
                    prev_resp = resp
                    yield _safe_check_tongyi_response(delta_resp)
                else:
                    yield _safe_check_tongyi_response(resp)

        return _stream_completion_with_retry(**kwargs)

    setattr(model, "completion_with_retry", MethodType(completion_with_retry, model))
    setattr(model, "stream_completion_with_retry", MethodType(stream_completion_with_retry, model))
    setattr(model, "_gtbot_safe_tongyi_error_patched", True)
    return model


def _is_dashscope_multimodal_only_chat_model(model_id: str) -> bool:
    """判断模型是否必须走 DashScope 多模态兼容接口。

    当前 `qwen3.5-*` 系列不适合复用 `ChatTongyi` 文本生成路径，
    需要在创建阶段直接阻止错误路由，避免运行时才暴露不兼容问题。

    Args:
        model_id: 归一化前的上游模型名称。

    Returns:
        若模型属于仅支持多模态兼容路径的 DashScope 系列，则返回 `True`。
    """

    normalized_model_id = str(model_id or "").strip().lower()
    return normalized_model_id.startswith("qwen3.5-")


def normalize_chat_provider_type(provider_type: str | None) -> str:
    """将提供商类型归一化为兼容层支持的固定枚举。

    空值会退回到 `openai_compatible`，其余值必须命中兼容层白名单，
    以确保后续工厂分发逻辑保持稳定且可预测。

    Args:
        provider_type: 原始配置中的提供商类型字符串。

    Returns:
        归一化后的提供商类型。

    Raises:
        ValueError: 当传入值不在兼容层支持范围内时抛出。
    """

    normalized = str(provider_type or "openai_compatible").strip() or "openai_compatible"
    if normalized not in SUPPORTED_CHAT_PROVIDER_TYPES:
        supported = "/".join(sorted(SUPPORTED_CHAT_PROVIDER_TYPES))
        raise ValueError(f"provider_type must be one of {supported}")
    return normalized


def _truncate_text(value: str, *, limit: int = _RAW_RESPONSE_BODY_TEXT_LIMIT) -> str:
    """按长度截断原始文本，避免消息对象过大。"""

    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated, total={len(text)})"


def _sanitize_headers(headers: Any) -> dict[str, str]:
    """脱敏并归一化原始响应头。

    兼容层只保留字符串键值，并屏蔽常见认证头，避免消息对象中出现敏感信息。

    Args:
        headers: SDK 原始返回中的 headers 容器。

    Returns:
        可安全写入消息对象的响应头字典。
    """

    sanitized: dict[str, str] = {}
    if headers is None:
        return sanitized
    try:
        header_items = dict(headers).items()
    except Exception:
        return sanitized

    for key, value in header_items:
        header_name = str(key or "").strip()
        if not header_name:
            continue
        if header_name.lower() in _REDACTED_HEADERS:
            sanitized[header_name] = "***"
        else:
            sanitized[header_name] = _truncate_text(str(value or ""), limit=1024)
    return sanitized


def _truncate_json_value(value: Any, *, depth: int = 0) -> Any:
    """递归裁剪 JSON 结构，限制原始响应体积。

    该函数优先保留结构信息，而不是保留完整文本。过深的嵌套和过长的集合
    会被摘要化，以便排障时仍能看清整体形态。

    Args:
        value: 待裁剪的 JSON 值。
        depth: 当前递归深度。

    Returns:
        裁剪后的 JSON 兼容值。
    """

    if depth >= _RAW_RESPONSE_MAX_DEPTH:
        return "<truncated-depth>"
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, list):
        trimmed_list = [_truncate_json_value(item, depth=depth + 1) for item in value[:_RAW_RESPONSE_COLLECTION_LIMIT]]
        if len(value) > _RAW_RESPONSE_COLLECTION_LIMIT:
            trimmed_list.append(f"<truncated-items total={len(value)}>")
        return trimmed_list
    if isinstance(value, dict):
        trimmed_dict: dict[str, Any] = {}
        items = list(value.items())[:_RAW_RESPONSE_COLLECTION_LIMIT]
        for key, item in items:
            trimmed_dict[str(key)] = _truncate_json_value(item, depth=depth + 1)
        if len(value) > _RAW_RESPONSE_COLLECTION_LIMIT:
            trimmed_dict["__truncated_items__"] = len(value)
        return trimmed_dict
    return value


def _extract_request_id(*, headers: dict[str, str], body_json: Any) -> str | None:
    """从响应头或响应体中提取请求标识。

    Args:
        headers: 已脱敏后的响应头。
        body_json: 已裁剪后的响应体 JSON。

    Returns:
        最可用的请求 ID；无法确定时返回 `None`。
    """

    for key in ("x-request-id", "request-id", "openai-request-id"):
        if headers.get(key):
            return headers[key]
        for header_key, header_value in headers.items():
            if header_key.lower() == key:
                return header_value
    if isinstance(body_json, dict):
        request_id = body_json.get("id")
        if isinstance(request_id, str) and request_id.strip():
            return request_id.strip()
    return None


def _build_raw_response_payload(
    *,
    provider_type: str,
    api_style: str,
    raw_response: Any,
    parsed_response: Any,
) -> RawResponsePayload:
    """构造可安全挂载到消息对象上的原始响应摘要。

    Args:
        provider_type: 当前兼容层提供商类型。
        api_style: OpenAI 请求风格，值为 `chat_completions` 或 `responses`。
        raw_response: SDK 的 `with_raw_response` 返回对象。
        parsed_response: 已解析的 SDK 响应对象。

    Returns:
        统一格式的原始响应摘要。
    """

    http_response = getattr(raw_response, "http_response", None)
    headers = _sanitize_headers(getattr(raw_response, "headers", None))
    if not headers and http_response is not None:
        headers = _sanitize_headers(getattr(http_response, "headers", None))

    status_code = getattr(http_response, "status_code", None)
    body_json: Any = None
    body_text: str | None = None

    if http_response is not None:
        try:
            body_json = _truncate_json_value(http_response.json())
        except Exception:
            try:
                body_text = _truncate_text(getattr(http_response, "text", ""))
            except Exception:
                body_text = None

    if body_json is None and parsed_response is not None:
        if hasattr(parsed_response, "model_dump"):
            try:
                body_json = _truncate_json_value(parsed_response.model_dump(mode="json", exclude_none=False))
            except Exception:
                body_json = None
        elif isinstance(parsed_response, (dict, list, str, int, float, bool)):
            body_json = _truncate_json_value(parsed_response)

    request_id = _extract_request_id(headers=headers, body_json=body_json)
    return RawResponsePayload(
        provider_type=provider_type,
        api_style=api_style,
        status_code=int(status_code) if isinstance(status_code, int) else None,
        headers=headers,
        body_json=body_json,
        body_text=body_text,
        request_id=request_id,
    )


def _attach_raw_response_to_result(
    result: ChatResult,
    *,
    raw_payload: RawResponsePayload | None,
    available: bool,
) -> ChatResult:
    """将原始响应摘要写入 LangChain `ChatResult` 中的消息对象。

    Args:
        result: LangChain 生成结果。
        raw_payload: 兼容层构造的原始响应摘要。
        available: 当前消息是否携带可用原始响应。

    Returns:
        原对象本身，便于在调用链中直接返回。
    """

    for generation in result.generations:
        message = getattr(generation, "message", None)
        if not isinstance(message, AIMessage):
            continue
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        additional_kwargs["raw_response"] = raw_payload if available else None
        message.additional_kwargs = additional_kwargs

        response_metadata = dict(getattr(message, "response_metadata", {}) or {})
        response_metadata["raw_response_available"] = available
        message.response_metadata = response_metadata
    return result


def _attach_raw_response_to_generation_chunk(
    generation_chunk: ChatGenerationChunk,
    *,
    raw_payload: RawResponsePayload,
    available: bool,
) -> ChatGenerationChunk:
    """将原始响应摘要挂到流式 chunk 的消息对象上。

    该函数用于流式路径的最终 chunk。当前只在消息对象支持 `additional_kwargs`
    与 `response_metadata` 时注入，保持其他 chunk 结构不变。

    Args:
        generation_chunk: LangChain 流式生成 chunk。
        raw_payload: 兼容层构造的原始响应摘要。
        available: 当前 chunk 是否携带可用原始响应。

    Returns:
        原 chunk 本身，便于在流式迭代中直接返回。
    """

    message = getattr(generation_chunk, "message", None)
    if not isinstance(message, AIMessageChunk):
        return generation_chunk
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs["raw_response"] = raw_payload if available else None
    message.additional_kwargs = additional_kwargs

    response_metadata = dict(getattr(message, "response_metadata", {}) or {})
    response_metadata["raw_response_available"] = available
    message.response_metadata = response_metadata
    return generation_chunk


class RawCaptureChatOpenAI(ChatOpenAI):
    """为 OpenAI 家族模型补充原始响应采集能力的兼容包装。

    该类保持 `ChatOpenAI` 的 LangChain 使用方式不变，只在同步与异步生成主路径中
    截获 `with_raw_response` 返回对象，并把脱敏后的原始响应摘要写回 `AIMessage`。
    不修改消息主内容与工具调用结构，因此可直接复用现有 GTBot 上层逻辑。
    """

    provider_type: str = "openai_compatible"
    """当前实例对应的兼容层提供商类型。"""
    _gtbot_raw_response_available: bool = True
    """标记当前模型支持原始响应采集，供运行时日志诊断使用。"""

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步调用 OpenAI 模型并注入原始响应摘要。

        Args:
            messages: LangChain 消息列表。
            stop: 可选的停止词列表。
            run_manager: LangChain 回调管理器，当前实现不直接消费。
            **kwargs: 透传给底层模型调用的额外参数。

        Returns:
            LangChain `ChatResult`，其中首条 `AIMessage` 会额外带上原始响应摘要。

        Raises:
            Exception: 透传底层 OpenAI SDK 或 LangChain 显式抛出的异常。
        """

        del run_manager
        self._ensure_sync_client_available()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                raw_response = self.root_client.chat.completions.with_raw_response.parse(**payload)
                response = raw_response.parse()
                result = self._create_chat_result(response, generation_info)
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="chat_completions",
                    raw_response=raw_response,
                    parsed_response=response,
                )
                return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

            if self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = self.root_client.responses.with_raw_response.parse(**payload)
                else:
                    raw_response = self.root_client.responses.with_raw_response.create(**payload)
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                result = _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="responses",
                    raw_response=raw_response,
                    parsed_response=response,
                )
                return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

            raw_response = self.client.with_raw_response.create(**payload)
            response = raw_response.parse()
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise
        except Exception as exc:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                exc.response = raw_response.http_response  # type: ignore[attr-defined]
            raise

        if self.include_response_headers and raw_response is not None and hasattr(raw_response, "headers"):
            generation_info = {"headers": dict(raw_response.headers)}
        result = self._create_chat_result(response, generation_info)
        raw_payload = _build_raw_response_payload(
            provider_type=self.provider_type,
            api_style="chat_completions",
            raw_response=raw_response,
            parsed_response=response,
        )
        return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步调用 OpenAI 模型并注入原始响应摘要。

        Args:
            messages: LangChain 消息列表。
            stop: 可选的停止词列表。
            run_manager: LangChain 回调管理器，当前实现不直接消费。
            **kwargs: 透传给底层模型调用的额外参数。

        Returns:
            LangChain `ChatResult`，其中首条 `AIMessage` 会额外带上原始响应摘要。

        Raises:
            Exception: 透传底层 OpenAI SDK 或 LangChain 显式抛出的异常。
        """

        del run_manager
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                raw_response = await self.root_async_client.chat.completions.with_raw_response.parse(**payload)
                response = raw_response.parse()
                result = await run_in_executor(None, self._create_chat_result, response, generation_info)
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="chat_completions",
                    raw_response=raw_response,
                    parsed_response=response,
                )
                return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

            if self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = await self.root_async_client.responses.with_raw_response.parse(**payload)
                else:
                    raw_response = await self.root_async_client.responses.with_raw_response.create(**payload)
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                result = _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="responses",
                    raw_response=raw_response,
                    parsed_response=response,
                )
                return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

            raw_response = await self.async_client.with_raw_response.create(**payload)
            response = raw_response.parse()
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise
        except Exception as exc:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                exc.response = raw_response.http_response  # type: ignore[attr-defined]
            raise

        if self.include_response_headers and raw_response is not None and hasattr(raw_response, "headers"):
            generation_info = {"headers": dict(raw_response.headers)}
        result = await run_in_executor(None, self._create_chat_result, response, generation_info)
        raw_payload = _build_raw_response_payload(
            provider_type=self.provider_type,
            api_style="chat_completions",
            raw_response=raw_response,
            parsed_response=response,
        )
        return _attach_raw_response_to_result(result, raw_payload=raw_payload, available=True)

    def _stream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """在流式模式下为最终 chunk 注入原始响应摘要。"""

        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            yield from self._stream_responses_with_raw_capture(*args, **kwargs)
            return
        yield from self._stream_chat_completions_with_raw_capture(*args, **kwargs)

    async def _astream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """在异步流式模式下为最终 chunk 注入原始响应摘要。"""

        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in self._astream_responses_with_raw_capture(*args, **kwargs):
                yield chunk
            return
        async for chunk in self._astream_chat_completions_with_raw_capture(*args, **kwargs):
            yield chunk

    def _stream_responses_with_raw_capture(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """为 Responses API 流式完成事件注入原始响应摘要。"""

        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        raw_context_manager = None
        try:
            raw_context_manager = self.root_client.with_raw_response.responses.create(**payload)
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
            original_schema_obj = kwargs.get("response_format")

            with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                for chunk in response:
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if getattr(generation_chunk.message, "chunk_position", None) == "last":
                            raw_payload = _build_raw_response_payload(
                                provider_type=self.provider_type,
                                api_style="responses",
                                raw_response=raw_context_manager,
                                parsed_response=getattr(chunk, "response", None),
                            )
                            generation_chunk = _attach_raw_response_to_generation_chunk(
                                generation_chunk,
                                raw_payload=raw_payload,
                                available=True,
                            )
                        if run_manager:
                            run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise

    async def _astream_responses_with_raw_capture(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """为 Responses API 异步流式完成事件注入原始响应摘要。"""

        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        raw_context_manager = None
        try:
            raw_context_manager = await self.root_async_client.with_raw_response.responses.create(**payload)
            context_manager = raw_context_manager.parse()
            headers = {"headers": dict(raw_context_manager.headers)}
            original_schema_obj = kwargs.get("response_format")

            async with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                async for chunk in response:
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if getattr(generation_chunk.message, "chunk_position", None) == "last":
                            raw_payload = _build_raw_response_payload(
                                provider_type=self.provider_type,
                                api_style="responses",
                                raw_response=raw_context_manager,
                                parsed_response=getattr(chunk, "response", None),
                            )
                            generation_chunk = _attach_raw_response_to_generation_chunk(
                                generation_chunk,
                                raw_payload=raw_payload,
                                available=True,
                            )
                        if run_manager:
                            await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise

    def _stream_chat_completions_with_raw_capture(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """为 Chat Completions 流式最终 completion 注入原始响应摘要。"""

        self._ensure_sync_client_available()
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type = AIMessageChunk
        base_generation_info: dict[str, Any] = {}
        raw_response = None
        response = None

        try:
            if "response_format" in payload:
                payload.pop("stream")
                response_stream = self.root_client.beta.chat.completions.stream(**payload)
                context_manager = response_stream
            else:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                context_manager = response
                if hasattr(raw_response, "headers"):
                    base_generation_info = {"headers": dict(raw_response.headers)}
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk, logprobs=logprobs)
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise
        if hasattr(response, "get_final_completion"):
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(final_completion)
            if raw_response is not None:
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="chat_completions",
                    raw_response=raw_response,
                    parsed_response=final_completion,
                )
                generation_chunk = _attach_raw_response_to_generation_chunk(
                    generation_chunk,
                    raw_payload=raw_payload,
                    available=True,
                )
            if run_manager:
                run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
            yield generation_chunk

    async def _astream_chat_completions_with_raw_capture(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """为 Chat Completions 异步流式最终 completion 注入原始响应摘要。"""

        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type = AIMessageChunk
        base_generation_info: dict[str, Any] = {}
        raw_response = None
        response = None

        try:
            if "response_format" in payload:
                payload.pop("stream")
                response_stream = self.root_async_client.beta.chat.completions.stream(**payload)
                context_manager = response_stream
            else:
                raw_response = await self.async_client.with_raw_response.create(**payload)
                response = raw_response.parse()
                context_manager = response
                if hasattr(raw_response, "headers"):
                    base_generation_info = {"headers": dict(raw_response.headers)}
            async with context_manager as response:
                is_first_chunk = True
                async for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as exc:
            _handle_openai_bad_request(exc)
            raise
        except openai.APIError as exc:
            _handle_openai_api_error(exc)
            raise
        if hasattr(response, "get_final_completion"):
            final_completion = await response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(final_completion)
            if raw_response is not None:
                raw_payload = _build_raw_response_payload(
                    provider_type=self.provider_type,
                    api_style="chat_completions",
                    raw_response=raw_response,
                    parsed_response=final_completion,
                )
                generation_chunk = _attach_raw_response_to_generation_chunk(
                    generation_chunk,
                    raw_payload=raw_payload,
                    available=True,
                )
            if run_manager:
                await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
            yield generation_chunk


def _mark_non_openai_raw_response_capability(model: BaseChatModel) -> BaseChatModel:
    """为非 OpenAI provider 标记统一的 raw-response 能力状态。

    该标记只作为运行时可读约定，第一版不强行包裹非 OpenAI provider 的消息结果，
    因此当前实现主要用于让上层有统一的能力探测入口。

    Args:
        model: 已创建好的 LangChain 模型对象。

    Returns:
        原模型对象本身。
    """

    setattr(model, "_gtbot_raw_response_available", False)
    return model


def build_chat_adapter_model(
    *,
    provider_type: str,
    model_id: str,
    base_url: str,
    api_key: str,
    streaming: bool,
    model_parameters: dict[str, Any] | None = None,
) -> Any:
    """按统一兼容层规则创建聊天模型对象。

    该工厂收口项目内所有聊天模型创建逻辑，统一处理 provider 分发、OpenAI
    原始响应采集，以及 DashScope 等 provider 的参数兼容细节。

    Args:
        provider_type: 已配置的提供商类型。
        model_id: 上游模型 ID。
        base_url: 提供商基础地址。
        api_key: 提供商 API 密钥。
        streaming: 是否启用流式输出。
        model_parameters: 归一化后的模型参数。

    Returns:
        可直接用于 LangChain `create_agent()` 与 `invoke/ainvoke` 的模型对象。

    Raises:
        ValueError: 当关键配置缺失或 provider 不支持当前模型时抛出。
        RuntimeError: 当所需 provider 依赖不可用时抛出。
    """

    normalized_provider_type = normalize_chat_provider_type(provider_type)
    normalized_model_id = str(model_id or "").strip()
    normalized_base_url = str(base_url or "").strip()
    normalized_api_key = str(api_key or "").strip()
    model_kwargs = dict(model_parameters or {})

    if not normalized_model_id:
        raise ValueError("missing chat model_id")

    if normalized_provider_type in {"openai_compatible", "openai_responses", "anthropic"} and not normalized_base_url:
        raise ValueError(f"provider_type={normalized_provider_type} requires non-empty base_url")

    if normalized_provider_type == "openai_compatible":
        return RawCaptureChatOpenAI(
            model=normalized_model_id,
            base_url=normalized_base_url,
            api_key=SecretStr(normalized_api_key),
            provider_type=normalized_provider_type,
            streaming=streaming,
            extra_body=model_kwargs,
        )

    if normalized_provider_type == "openai_responses":
        output_version = model_kwargs.pop("output_version", None)
        init_kwargs: dict[str, Any] = {
            "model": normalized_model_id,
            "base_url": normalized_base_url,
            "api_key": SecretStr(normalized_api_key),
            "provider_type": normalized_provider_type,
            "streaming": streaming,
            "extra_body": model_kwargs,
            "use_responses_api": True,
        }
        if output_version is not None:
            init_kwargs["output_version"] = output_version
        return RawCaptureChatOpenAI(**init_kwargs)

    if normalized_provider_type == "anthropic":
        try:
            anthropic_mod = importlib.import_module("langchain_anthropic")
        except ImportError as exc:
            raise RuntimeError(
                "provider_type=anthropic requires installing langchain-anthropic"
            ) from exc

        chat_cls = getattr(anthropic_mod, "ChatAnthropic", None)
        if chat_cls is None:
            raise RuntimeError("langchain_anthropic.ChatAnthropic is unavailable")
        return _mark_non_openai_raw_response_capability(
            cast(
                BaseChatModel,
                chat_cls(
                    model=normalized_model_id,
                    base_url=normalized_base_url,
                    api_key=SecretStr(normalized_api_key),
                    streaming=streaming,
                    model_kwargs=model_kwargs,
                ),
            )
        )

    if normalized_provider_type == "gemini":
        module_candidates = [
            ("langchain_google_genai", "ChatGoogleGenerativeAI"),
            ("langchain_google_vertexai", "ChatVertexAI"),
        ]
        last_error: BaseException | None = None
        for module_name, class_name in module_candidates:
            try:
                provider_mod = importlib.import_module(module_name)
            except ImportError as exc:
                last_error = exc
                continue

            chat_cls = getattr(provider_mod, class_name, None)
            if chat_cls is None:
                continue
            return _mark_non_openai_raw_response_capability(
                cast(
                    BaseChatModel,
                    chat_cls(
                        model=normalized_model_id,
                        google_api_key=SecretStr(normalized_api_key),
                        streaming=streaming,
                        model_kwargs=model_kwargs,
                    ),
                )
            )

        raise RuntimeError(
            "provider_type=gemini requires installing langchain-google-genai or langchain-google-vertexai"
        ) from last_error

    if normalized_provider_type == "dashscope":
        if _is_dashscope_multimodal_only_chat_model(normalized_model_id):
            raise ValueError(
                "provider_type=dashscope does not support "
                f"model={normalized_model_id} via ChatTongyi text-generation path. "
                "Qwen3.5 series uses DashScope multimodal API; "
                "please switch this provider to openai_compatible and use "
                "base_url=https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        try:
            tongyi_mod = importlib.import_module("langchain_community.chat_models.tongyi")
        except ImportError as exc:
            raise RuntimeError(
                "provider_type=dashscope requires langchain-community and dashscope"
            ) from exc

        chat_cls = getattr(tongyi_mod, "ChatTongyi", None)
        if chat_cls is None:
            raise RuntimeError("langchain_community.chat_models.tongyi.ChatTongyi is unavailable")

        dashscope_kwargs = dict(model_kwargs)
        if normalized_base_url:
            dashscope_kwargs.pop("base_url", None)
            dashscope_kwargs.setdefault("base_address", normalized_base_url)
        model = cast(
            BaseChatModel,
            chat_cls(
                model=normalized_model_id,
                api_key=normalized_api_key or None,
                streaming=streaming,
                model_kwargs=dashscope_kwargs,
            ),
        )
        model = _patch_tongyi_error_handling(model, tongyi_mod)
        return _mark_non_openai_raw_response_capability(model)

    raise RuntimeError(f"unsupported provider_type: {normalized_provider_type}")
