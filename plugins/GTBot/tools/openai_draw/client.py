from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from typing import Any

import httpx

from .config import OpenAIDrawPluginConfig


class OpenAIDrawClientError(RuntimeError):
    """表示 OpenAI 绘图网关请求失败或响应不可用。"""


@dataclass(frozen=True, slots=True)
class OpenAIImageResult:
    """描述单次文生图请求返回的图片结果。

    Attributes:
        b64_json: 图片的 Base64 编码数据，适合直接落盘。
        url: 若网关返回远程地址，则保存为该字段并由上层下载。
        revised_prompt: 网关可能返回的修订提示词，供诊断或日志使用。
    """

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class OpenAIDrawResponse:
    """描述单次文生图接口的结构化响应。

    Attributes:
        created: 服务端返回的生成时间戳。
        data: 图片结果列表，当前插件仅消费第一张图。
        raw_payload: 原始 JSON 响应，便于必要时排查问题。
    """

    created: int | None
    data: list[OpenAIImageResult]
    raw_payload: dict[str, Any]


class OpenAIDrawClient:
    """封装标准 OpenAI Images API 的最小调用逻辑。

    该客户端只负责组织请求、解析响应和提炼错误信息，不负责任务排队、
    图片落盘或消息发送，从而让调用方可以在后台任务中自由编排流程。
    """

    def __init__(self, cfg: OpenAIDrawPluginConfig) -> None:
        """初始化绘图客户端。

        Args:
            cfg: 当前插件配置，用于确定地址、鉴权和超时参数。
        """

        self._cfg = cfg

    async def generate_image(
        self,
        *,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
    ) -> OpenAIDrawResponse:
        """调用标准 OpenAI 文生图接口生成图片。

        Args:
            prompt: 文生图提示词。
            size: 请求的图片尺寸，例如 `1024x1024`。
            quality: 图片质量参数。
            background: 背景参数。
            output_format: 输出图片格式。

        Returns:
            结构化的图片生成响应。

        Raises:
            OpenAIDrawClientError: 当缺少必要配置、HTTP 请求失败或响应格式非法时抛出。
        """

        api_key = str(self._cfg.api_key or "").strip()
        model = str(self._cfg.model or "").strip()
        if not api_key:
            raise OpenAIDrawClientError("配置缺少 api_key")
        if not model:
            raise OpenAIDrawClientError("配置缺少 model")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
        if self._cfg.allow_compat_mode:
            payload["n"] = 1

        # GPT Image 模型通常直接返回 base64 图像；兼容网关或旧模型仍可能需要显式指定。
        payload["response_format"] = "b64_json"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = await self._post_once(
            url=self._cfg.image_generation_url,
            headers=headers,
            json=payload,
        )

        if response.is_error:
            raise OpenAIDrawClientError(self._extract_error_message(response))

        try:
            parsed = response.json()
        except ValueError as exc:
            raise OpenAIDrawClientError("绘图网关返回了无法解析的 JSON") from exc
        if not isinstance(parsed, dict):
            raise OpenAIDrawClientError("绘图网关返回格式非法")

        raw_data = parsed.get("data")
        if not isinstance(raw_data, list) or not raw_data:
            raise OpenAIDrawClientError("绘图网关返回中缺少 data[0]")

        results: list[OpenAIImageResult] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            results.append(
                OpenAIImageResult(
                    b64_json=self._normalize_optional_string(item.get("b64_json")),
                    url=self._normalize_optional_string(item.get("url")),
                    revised_prompt=self._normalize_optional_string(item.get("revised_prompt")),
                )
            )

        if not results:
            raise OpenAIDrawClientError("绘图网关返回的图片结果为空")
        if not results[0].b64_json and not results[0].url:
            raise OpenAIDrawClientError("绘图网关未返回可用图片内容")

        created = parsed.get("created")
        created_value = int(created) if isinstance(created, int) else None
        return OpenAIDrawResponse(created=created_value, data=results, raw_payload=parsed)

    async def edit_image(
        self,
        *,
        prompt: str,
        images: list[tuple[str, bytes]],
        mask: tuple[str, bytes] | None,
        size: str,
        quality: str,
        background: str,
        input_fidelity: str,
        output_format: str,
    ) -> OpenAIDrawResponse:
        """调用标准 OpenAI 编辑图接口生成图片。

        当前实现采用 `multipart/form-data` 上传图片二进制，兼容官方 OpenAI
        以及常见的 OpenAI 兼容网关。若传入多张图片，将按接口语义共同参与编辑；
        若提供 `mask`，则只会应用在第一张图片上。

        Args:
            prompt: 编辑提示词。
            images: 输入图片列表，每项为 `(文件名, 图片字节)`。
            mask: 可选遮罩图片，格式为 `(文件名, 图片字节)`。
            size: 目标输出尺寸。
            quality: 输出质量参数。
            background: 输出背景参数。
            input_fidelity: 输入保真度，当前支持 `low` 和 `high`。
            output_format: 输出图片格式。

        Returns:
            结构化的图片编辑响应。

        Raises:
            OpenAIDrawClientError: 当缺少必要配置、输入图片为空、HTTP 请求失败或响应格式非法时抛出。
        """

        api_key = str(self._cfg.api_key or "").strip()
        model = str(self._cfg.model or "").strip()
        if not api_key:
            raise OpenAIDrawClientError("配置缺少 api_key")
        if not model:
            raise OpenAIDrawClientError("配置缺少 model")
        if not images:
            raise OpenAIDrawClientError("编辑图请求缺少输入图片")

        data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "input_fidelity": input_fidelity,
            "output_format": output_format,
            "response_format": "b64_json",
        }
        if self._cfg.allow_compat_mode:
            data["n"] = "1"

        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for file_name, image_bytes in images:
            files.append(
                (
                    "image",
                    (
                        file_name,
                        image_bytes,
                        self._guess_content_type(file_name),
                    ),
                )
            )
        if mask is not None:
            mask_name, mask_bytes = mask
            files.append(
                (
                    "mask",
                    (
                        mask_name,
                        mask_bytes,
                        self._guess_content_type(mask_name),
                    ),
                )
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        response = await self._post_once(
            url=self._cfg.image_edit_url,
            headers=headers,
            data={key: str(value) for key, value in data.items()},
            files=files,
        )

        if response.is_error:
            raise OpenAIDrawClientError(self._extract_error_message(response))

        try:
            parsed = response.json()
        except ValueError as exc:
            raise OpenAIDrawClientError("编辑图网关返回了无法解析的 JSON") from exc
        if not isinstance(parsed, dict):
            raise OpenAIDrawClientError("编辑图网关返回格式非法")

        raw_data = parsed.get("data")
        if not isinstance(raw_data, list) or not raw_data:
            raise OpenAIDrawClientError("编辑图网关返回中缺少 data[0]")

        results: list[OpenAIImageResult] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            results.append(
                OpenAIImageResult(
                    b64_json=self._normalize_optional_string(item.get("b64_json")),
                    url=self._normalize_optional_string(item.get("url")),
                    revised_prompt=self._normalize_optional_string(item.get("revised_prompt")),
                )
            )

        if not results:
            raise OpenAIDrawClientError("编辑图网关返回的图片结果为空")
        if not results[0].b64_json and not results[0].url:
            raise OpenAIDrawClientError("编辑图网关未返回可用图片内容")

        created = parsed.get("created")
        created_value = int(created) if isinstance(created, int) else None
        return OpenAIDrawResponse(created=created_value, data=results, raw_payload=parsed)

    async def _post_once(
        self,
        *,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> httpx.Response:
        """对上游网关执行单次 POST 请求。

        该方法不会自动重试，保持一次工具调用只对应一次真实上游请求。
        当请求建立阶段或传输阶段失败时，会补充异常类型和 URL，便于排查。

        Args:
            url: 目标请求地址。
            headers: HTTP 请求头。
            json: 可选 JSON 请求体。
            data: 可选表单字段。
            files: 可选 multipart 文件列表。

        Returns:
            成功返回的 HTTP 响应对象。

        Raises:
            OpenAIDrawClientError: 当请求失败时抛出包含异常类型的明确错误。
        """
        try:
            async with httpx.AsyncClient(timeout=float(self._cfg.timeout_sec), follow_redirects=True) as client:
                return await client.post(
                    url,
                    headers=headers,
                    json=json,
                    data=data,
                    files=files,
                )
        except httpx.HTTPError as exc:
            raise OpenAIDrawClientError(self._format_request_error(exc, url=url)) from exc

    def _extract_error_message(self, response: httpx.Response) -> str:
        """从错误响应中提取适合发给用户的简短提示。

        Args:
            response: HTTP 错误响应对象。

        Returns:
            经过截断和清洗后的错误信息文本。
        """

        detail = f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip().replace("\n", " ")
            if text:
                return f"{detail}: {text[:240]}"
            return detail

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = self._normalize_optional_string(error.get("message"))
                if message:
                    return f"{detail}: {message[:240]}"
            message = self._normalize_optional_string(payload.get("message"))
            if message:
                return f"{detail}: {message[:240]}"
        return detail

    def _format_request_error(self, exc: httpx.HTTPError, *, url: str) -> str:
        """将网络层异常格式化为便于排查的错误信息。

        Args:
            exc: `httpx` 抛出的网络层异常。
            url: 发生错误的请求地址。

        Returns:
            适合直接展示给用户和写入日志的简短错误文本。
        """

        message = str(exc).strip()
        if not message:
            message = repr(exc)
        return f"请求绘图网关失败: {type(exc).__name__}: {message} url={url}"

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        """将可选值规范化为非空字符串。

        Args:
            value: 待转换的值。

        Returns:
            非空字符串；若为空值则返回 `None`。
        """

        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    @staticmethod
    def _guess_content_type(file_name: str) -> str:
        """根据文件名推断上传图片的 MIME 类型。

        Args:
            file_name: 输入文件名。

        Returns:
            适合作为 multipart 上传内容类型的 MIME 字符串。
        """

        mime_type, _ = mimetypes.guess_type(file_name)
        if isinstance(mime_type, str) and mime_type.startswith("image/"):
            return mime_type
        return "image/png"
