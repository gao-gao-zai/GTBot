from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext


def _avatar_cache_dir() -> Path:
    """返回头像缓存目录并确保其存在。

    头像文件统一保存到项目根目录下的 `data/avatar_filename` 子目录。
    这样返回值可以使用更短的相对路径，同时仍然足够稳定，便于后续工具继续引用。

    Returns:
        头像缓存目录的绝对路径。
    """

    cache_dir = Path.cwd() / "data" / "avatar_filename"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _display_path(path: Path) -> str:
    """将缓存文件路径转换为尽量短且稳定的项目内相对路径。

    Args:
        path: 已保存头像文件的绝对路径。

    Returns:
        适合返回给 AI 的相对路径字符串；若无法相对化则回退为原始路径。
    """

    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _normalize_positive_int(value: int | None, *, name: str) -> int:
    """将输入值规范化为正整数。

    Args:
        value: 待校验的数字输入。
        name: 当前参数名，用于构造错误提示。

    Returns:
        规范化后的正整数值。

    Raises:
        ValueError: 当输入为空、不是整数或不大于 0 时抛出。
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
        基于 QQ 头像服务的头像 URL。
    """

    return f"https://q1.qlogo.cn/g?b=qq&nk={int(user_id)}&s=640"


def _group_avatar_url(group_id: int) -> str:
    """构造指定群头像的下载地址。

    Args:
        group_id: 目标群号。

    Returns:
        基于 QQ 群头像服务的头像 URL。
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

    文件名采用稳定且可预测的格式，并附加内容哈希前缀，
    这样可以在重复调用时覆盖旧头像，也便于调用方直接拿到真实文件名使用。

    Args:
        prefix: 文件名前缀，通常为 `user_avatar` 或 `group_avatar`。
        target_id: 目标用户或群组 ID。
        avatar_bytes: 头像原始字节内容。
        content_type: 响应内容类型，用于推断扩展名。

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


async def _fetch_user_display_name(*, ctx: GroupChatContext, user_id: int) -> str:
    """通过宿主缓存链路获取用户显示名。

    Args:
        ctx: 当前运行时上下文。
        user_id: 目标用户 QQ 号。

    Returns:
        优先使用昵称，否则回退为空字符串。

    Raises:
        ValueError: 当上下文缺少 `bot` 或 `cache` 时抛出。
    """

    if ctx.bot is None or ctx.cache is None:
        raise ValueError("运行时上下文缺少 bot/cache")
    info = await ctx.cache.get_stranger_info(ctx.bot, user_id)
    return str(getattr(info, "nickname", "") or "").strip()


async def _fetch_group_display_name(*, ctx: GroupChatContext, group_id: int) -> str:
    """通过宿主缓存链路获取群名称。

    Args:
        ctx: 当前运行时上下文。
        group_id: 目标群号。

    Returns:
        群名称；若无法解析则返回空字符串。

    Raises:
        ValueError: 当上下文缺少 `bot` 或 `cache` 时抛出。
    """

    if ctx.bot is None or ctx.cache is None:
        raise ValueError("运行时上下文缺少 bot/cache")
    info = await ctx.cache.get_group_info(ctx.bot, group_id)
    return str(getattr(info, "group_name", "") or "").strip()


@tool("get_user_avatar_filename")
async def get_user_avatar_filename(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int | None = None,
) -> str:
    """下载并返回指定用户头像的本地文件名。

    该工具会优先校验目标用户并获取昵称，然后从 QQ 官方头像服务下载头像，
    最终把头像保存到 GTBot 数据目录下的缓存文件中，并把文件名与路径返回给 AI。
    若未传入 `user_id`，则默认使用当前会话用户。

    Args:
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        user_id: 目标用户 QQ 号；未传时默认取当前会话用户。

    Returns:
        项目目录下可直接识别的头像相对路径。

    Raises:
        ValueError: 当运行时上下文缺失或用户 ID 非法时抛出。
        RuntimeError: 当头像下载失败时抛出。
    """

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    target_user_id = int(getattr(ctx, "user_id", 0) or 0) if user_id is None else _normalize_positive_int(user_id, name="user_id")
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

    return _display_path(saved)


@tool("get_group_avatar_filename")
async def get_group_avatar_filename(
    runtime: ToolRuntime[GroupChatContext],
    group_id: int | None = None,
) -> str:
    """下载并返回指定群头像的本地文件名。

    该工具会优先校验目标群并获取群名，然后从 QQ 群头像服务下载头像，
    最终把头像保存到 GTBot 数据目录下的缓存文件中，并把文件名与路径返回给 AI。
    若未传入 `group_id`，则默认使用当前会话群号。

    Args:
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        group_id: 目标群号；未传时默认取当前会话群号。

    Returns:
        项目目录下可直接识别的群头像相对路径。

    Raises:
        ValueError: 当运行时上下文缺失或群号非法时抛出。
        RuntimeError: 当头像下载失败时抛出。
    """

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    target_group_id = int(getattr(ctx, "group_id", 0) or 0) if group_id is None else _normalize_positive_int(group_id, name="group_id")
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

    return _display_path(saved)
