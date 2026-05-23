from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain.tools import tool

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.services.file_registry import resolve_file_ref

_DEFAULT_MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024


def _current_chat_model_supports_vision() -> bool:
    """判断当前聊天模型是否声明支持视觉输入。

    该插件不在注册阶段隐藏工具，而是在工具调用时把能力不足反馈给 Agent，方便
    Agent 根据错误结果改用文字说明、追问用户或停止识图链路。

    Returns:
        当前聊天模型声明支持视觉输入时返回 `True`，否则返回 `False`。
    """

    chat_model = total_config.processed_configuration.current_config_group.chat_model
    return bool(getattr(chat_model, "supports_vision", False))


def _build_unsupported_vision_result() -> tuple[str, dict[str, object]]:
    """构造非视觉模型调用本工具时返回给 Agent 的错误结果。

    Returns:
        符合 `content_and_artifact` 格式的错误结果。第一项是 Agent 可读的错误文本，
        第二项是结构化错误 artifact，便于日志或后续中间件识别。
    """

    message = "错误：当前聊天模型配置未声明支持视觉输入，无法直接读取图片。"
    return message, {
        "type": "error",
        "code": "vision_not_supported",
        "message": message,
    }


def _guess_image_mime_type(path: Path, configured_mime_type: str | None) -> str:
    """推断图片 MIME 类型。

    优先使用文件映射中记录的 MIME 类型；缺失时根据文件名推断。若仍无法识别，
    回退为 `image/png`，以保持 data URL 可被常见多模态模型识别。

    Args:
        path: 图片本地路径。
        configured_mime_type: 文件映射中已有的 MIME 类型。

    Returns:
        可用于 data URL 的图片 MIME 类型。
    """

    if configured_mime_type and configured_mime_type.startswith("image/"):
        return configured_mime_type
    guessed_mime_type, _ = mimetypes.guess_type(str(path))
    if guessed_mime_type and guessed_mime_type.startswith("image/"):
        return guessed_mime_type
    return "image/png"


def _read_image_bytes(path: Path, *, max_size_bytes: int) -> bytes:
    """读取图片字节并进行大小限制。

    该工具会把图片内联为 base64 data URL 返回给模型，因此需要在读取前后都做
    大小校验，避免单次工具结果过大导致模型请求失败或上下文膨胀。

    Args:
        path: 图片本地路径。
        max_size_bytes: 允许读取的最大字节数。

    Returns:
        图片原始字节。

    Raises:
        ValueError: 当 `max_size_bytes` 非正数或图片超过限制时抛出。
    """

    if int(max_size_bytes) <= 0:
        raise ValueError("max_size_bytes 必须大于 0")

    stat_size = int(path.stat().st_size)
    if stat_size > int(max_size_bytes):
        raise ValueError(f"图片大小超过限制: {stat_size} > {int(max_size_bytes)}")

    image_bytes = path.read_bytes()
    if len(image_bytes) > int(max_size_bytes):
        raise ValueError(f"图片大小超过限制: {len(image_bytes)} > {int(max_size_bytes)}")
    return image_bytes


@tool("get_image_for_model", response_format="content_and_artifact")
def get_image_for_model(
    file_ref: str,
    max_size_bytes: int = _DEFAULT_MAX_IMAGE_SIZE_BYTES,
) -> tuple[str, dict[str, object]]:
    """根据 `gfid:` 或 `gf:` 图片引用，把图片本体交给你直接查看。

    当聊天记录里出现图片引用，且你需要亲自看图才能回答时，传入对应的 `file_ref`
    调用本工具。工具成功后，图片会作为多模态内容出现在工具结果里；请查看图片本体，
    再根据用户问题回答、判断、总结或继续后续工具流程。

    输入必须是聊天上下文中出现过的 `gfid:` 或 `gf:` 图片引用。工具返回的文件名、
    MIME 和大小只是辅助信息，核心结果是图片本体。

    Args:
        file_ref: 要查看的图片引用，必须形如 `gfid:...` 或 `gf:...`。
        max_size_bytes: 图片大小上限，通常保持默认即可。

    Returns:
        如果可用，返回图片内容供模型直接查看；如果不可用，返回明确错误信息。

    Raises:
        ValueError: 当图片引用无效、目标不是图片或图片过大时抛出。
        FileNotFoundError: 当图片文件不存在时抛出。
    """

    if not _current_chat_model_supports_vision():
        return _build_unsupported_vision_result()

    normalized_file_ref = str(file_ref or "").strip()
    if not normalized_file_ref:
        raise ValueError("file_ref 不能为空")

    handle = resolve_file_ref(normalized_file_ref)
    if handle.mime_type and not str(handle.mime_type).startswith("image/"):
        raise ValueError(f"file_ref 对应文件不是图片: {handle.mime_type}")

    image_path = handle.local_path
    image_bytes = _read_image_bytes(image_path, max_size_bytes=int(max_size_bytes))
    mime_type = _guess_image_mime_type(image_path, handle.mime_type)
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    content = (
        f"已读取图片 {normalized_file_ref}，文件名={image_path.name}，"
        f"MIME={mime_type}，大小={len(image_bytes)} 字节。"
        "如果当前模型支持多模态工具结果，请直接查看 artifact.image_url 中的图片内容。"
    )
    artifact: dict[str, object] = {
        "type": "image",
        "file_ref": normalized_file_ref,
        "mime_type": mime_type,
        "size_bytes": len(image_bytes),
        "image_url": {"url": image_data_url},
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            }
        ],
    }
    return content, artifact
