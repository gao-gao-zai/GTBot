from __future__ import annotations

import importlib
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


SUPPORTED_CHAT_PROVIDER_TYPES = frozenset(
    {"openai_compatible", "openai_responses", "anthropic", "gemini", "dashscope"}
)


def _is_dashscope_multimodal_only_chat_model(model_id: str) -> bool:
    normalized_model_id = str(model_id or "").strip().lower()
    return normalized_model_id.startswith("qwen3.5-")


def normalize_chat_provider_type(provider_type: str | None) -> str:
    normalized = str(provider_type or "openai_compatible").strip() or "openai_compatible"
    if normalized not in SUPPORTED_CHAT_PROVIDER_TYPES:
        supported = "/".join(sorted(SUPPORTED_CHAT_PROVIDER_TYPES))
        raise ValueError(f"provider_type must be one of {supported}")
    return normalized


def build_chat_model(
    *,
    provider_type: str,
    model_id: str,
    base_url: str,
    api_key: str,
    streaming: bool,
    model_parameters: dict[str, Any] | None = None,
) -> Any:
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
        return ChatOpenAI(
            model=normalized_model_id,
            base_url=normalized_base_url,
            api_key=SecretStr(normalized_api_key),
            streaming=streaming,
            extra_body=model_kwargs,
        )

    if normalized_provider_type == "openai_responses":
        return ChatOpenAI(
            model=normalized_model_id,
            base_url=normalized_base_url,
            api_key=SecretStr(normalized_api_key),
            streaming=streaming,
            extra_body=model_kwargs,
            model_kwargs={"use_responses_api": True},
        )

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
        return chat_cls(
            model=normalized_model_id,
            base_url=normalized_base_url,
            api_key=SecretStr(normalized_api_key),
            streaming=streaming,
            model_kwargs=model_kwargs,
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
            return chat_cls(
                model=normalized_model_id,
                google_api_key=SecretStr(normalized_api_key),
                streaming=streaming,
                model_kwargs=model_kwargs,
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
            # ChatTongyi -> dashscope SDK expects `base_address`, not `base_url`.
            dashscope_kwargs.pop("base_url", None)
            dashscope_kwargs.setdefault("base_address", normalized_base_url)
        return chat_cls(
            model=normalized_model_id,
            api_key=normalized_api_key or None,
            streaming=streaming,
            model_kwargs=dashscope_kwargs,
        )

    raise RuntimeError(f"unsupported provider_type: {normalized_provider_type}")
