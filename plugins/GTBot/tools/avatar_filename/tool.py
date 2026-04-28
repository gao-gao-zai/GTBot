from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path

import httpx
from langchain.tools import ToolRuntime, tool

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.services.file_registry import register_local_file

_AVATAR_FILE_REF_TTL_SEC = 7 * 24 * 60 * 60


def _avatar_cache_dir() -> Path:
    """返回头像缓存目录并确保其存在。

    头像文件统一保存到 GTBot 配置指定的数据目录下的 `avatar_filename`
    子目录，而不是依赖当前工作目录。这样即使机器人从不同启动目录运行，
    缓存位置也保持稳定，后续注册到文件映射系统时也能得到稳定的物理路径。

    Returns:
        头像缓存目录的绝对路径。
    """

    data_dir = total_config.get_data_dir_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "avatar_filename"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _normalize_positive_int(value: int | None, *, name: str) -> int:
    """将输入参数规范化为正整数。

    该函数用于统一校验用户或群号类参数，避免主工具函数中重复拼接同样的
    判空与正数判断逻辑。若调用方传入了字符串数字，`int()` 也会被接受。

    Args:
        value: 待校验的数字输入。
        name: 当前参数名，用于构造错误提示。

    Returns:
        规范化后的正整数值。

    Raises:
        ValueError: 当输入为空、无法转成整数或不大于 0 时抛出。
    """

    if value is None:
        raise ValueError(f"{name} 不能为空")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return normalized


def _user_avatar_url(user_id: int) -> str:
    """构造指定 QQ 用户头像的下载地址。

    Args:
        user_id: 目标用户 QQ 号。

    Returns:
        基于 QQ 官方头像服务的下载 URL。
    """

    return f"https://q1.qlogo.cn/g?b=qq&nk={int(user_id)}&s=640"


def _group_avatar_url(group_id: int) -> str:
    """构造指定 QQ 群头像的下载地址。

    Args:
        group_id: 目标群号。

    Returns:
        基于 QQ 群头像服务的下载 URL。
    """

    group = int(group_id)
    return f"https://p.qlogo.cn/gh/{group}/{group}/640"


def _guess_extension(*, content_type: str | None, fallback: str = ".jpg") -> str:
    """根据响应头推断头像文件后缀。

    Args:
        content_type: HTTP 响应中的内容类型。
        fallback: 无法识别时使用的兜底后缀。

    Returns:
        适合保存到本地文件名中的扩展名。
    """

    if isinstance(content_type, str):
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if isinstance(ext, str) and ext:
            return ext
    return fallback


async def _download_avatar_bytes(url: str) -> tuple[bytes, str | None]:
    """下载头像字节并返回内容类型。

    Args:
        url: 头像下载地址。

    Returns:
        一个二元组，包含头像字节与响应头中的内容类型。

    Raises:
        RuntimeError: 当头像下载失败时抛出。
    """

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"下载头像失败: {type(exc).__name__}: {exc!s}") from exc

    return response.content, response.headers.get("content-type")


def _save_avatar_file(*, prefix: str, target_id: int, avatar_bytes: bytes, content_type: str | None) -> Path:
    """将头像内容保存到本地缓存目录。

    文件名附带目标 ID 与内容哈希前缀，便于在重复生成时保留稳定可识别的文件名，
    同时避免不同头像内容之间的覆盖冲突。写入时先落到临时文件再原子替换，减少
    并发读取时拿到半写入文件的风险。

    Args:
        prefix: 文件名前缀，通常为 `user_avatar` 或 `group_avatar`。
        target_id: 目标用户或群组 ID。
        avatar_bytes: 头像原始字节内容。
        content_type: HTTP 内容类型，用于推断扩展名。

    Returns:
        保存后的本地文件路径。
    """

    cache_dir = _avatar_cache_dir()
    digest = hashlib.sha1(avatar_bytes).hexdigest()[:12]
    suffix = _guess_extension(content_type=content_type)
    target = cache_dir / f"{prefix}_{int(target_id)}_{digest}{suffix}"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(avatar_bytes)
    tmp.replace(target)
    return target


@tool("get_user_avatar_filename")
async def get_user_avatar_filename(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int | None = None,
) -> str:
    """下载并注册指定用户头像，返回稳定 `file_id`。

    该工具会优先从当前会话上下文补齐默认用户号，然后从 QQ 官方头像服务下载头像，
    把头像保存到 GTBot 数据目录下的缓存文件中，最后注册到统一文件映射系统。
    返回值不再是裸路径，而是供其他 Agent tool 继续消费的稳定 `gfid:...` 句柄。

    Args:
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        user_id: 目标用户 QQ 号；未传入时默认使用当前会话用户。

    Returns:
        可直接传给其他图片工具继续使用的稳定 `file_id`。

    Raises:
        ValueError: 当运行时上下文缺失或用户 ID 非法时抛出。
        RuntimeError: 当头像下载失败时抛出。
    """

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    target_user_id = (
        int(getattr(ctx, "user_id", 0) or 0)
        if user_id is None
        else _normalize_positive_int(user_id, name="user_id")
    )
    if target_user_id <= 0:
        raise ValueError("运行时上下文缺少可用的 user_id")

    url = _user_avatar_url(target_user_id)
    avatar_bytes, content_type = await _download_avatar_bytes(url)
    saved = _save_avatar_file(
        prefix="user_avatar",
        target_id=target_user_id,
        avatar_bytes=avatar_bytes,
        content_type=content_type,
    )
    return register_local_file(
        saved,
        kind="avatar",
        source_type="avatar_download",
        session_id=str(getattr(ctx, "session_id", "") or "").strip() or None,
        group_id=int(getattr(ctx, "group_id", 0) or 0) or None,
        user_id=target_user_id,
        mime_type=content_type,
        original_name=saved.name,
        extra={"avatar_type": "user", "target_user_id": target_user_id},
        expires_at=float(time.time()) + float(_AVATAR_FILE_REF_TTL_SEC),
    )


@tool("get_group_avatar_filename")
async def get_group_avatar_filename(
    runtime: ToolRuntime[GroupChatContext],
    group_id: int | None = None,
) -> str:
    """下载并注册指定群头像，返回稳定 `file_id`。

    该工具会优先从当前会话上下文补齐默认群号，然后从 QQ 群头像服务下载头像，
    保存到 GTBot 数据目录后注册进统一文件映射系统。返回值同样是稳定 `file_id`，
    以便识图、改图、收藏等后续工具直接复用。

    Args:
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        group_id: 目标群号；未传入时默认使用当前会话群号。

    Returns:
        可直接传给其他图片工具继续使用的稳定 `file_id`。

    Raises:
        ValueError: 当运行时上下文缺失或群号非法时抛出。
        RuntimeError: 当头像下载失败时抛出。
    """

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    target_group_id = (
        int(getattr(ctx, "group_id", 0) or 0)
        if group_id is None
        else _normalize_positive_int(group_id, name="group_id")
    )
    if target_group_id <= 0:
        raise ValueError("运行时上下文缺少可用的 group_id")

    url = _group_avatar_url(target_group_id)
    avatar_bytes, content_type = await _download_avatar_bytes(url)
    saved = _save_avatar_file(
        prefix="group_avatar",
        target_id=target_group_id,
        avatar_bytes=avatar_bytes,
        content_type=content_type,
    )
    return register_local_file(
        saved,
        kind="avatar",
        source_type="avatar_download",
        session_id=str(getattr(ctx, "session_id", "") or "").strip() or None,
        group_id=target_group_id,
        user_id=int(getattr(ctx, "user_id", 0) or 0) or None,
        mime_type=content_type,
        original_name=saved.name,
        extra={"avatar_type": "group", "target_group_id": target_group_id},
        expires_at=float(time.time()) + float(_AVATAR_FILE_REF_TTL_SEC),
    )
