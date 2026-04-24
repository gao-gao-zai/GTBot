from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext

from .config import get_openai_draw_plugin_config

_manager_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".manager")

OpenAIDrawJobSpec = cast(Any, getattr(_manager_mod, "OpenAIDrawJobSpec"))
OpenAIInputImage = cast(Any, getattr(_manager_mod, "OpenAIInputImage"))
get_openai_draw_queue_manager = cast(Callable[[], Any], getattr(_manager_mod, "get_openai_draw_queue_manager"))


@dataclass(frozen=True, slots=True)
class ResolvedInputImages:
    """描述从当前消息上下文中提取出的编辑图输入。

    Attributes:
        images: 作为编辑原图上传的图片列表。
        mask_image: 可选遮罩图；当前约定为同一消息中的第二张图。
    """

    images: tuple[Any, ...]
    mask_image: Any | None = None


@dataclass(frozen=True, slots=True)
class RawImageRef:
    """描述从消息段中直接提取出的图片引用。

    Attributes:
        file: OneBot 图片段中的 `file` 字段，可能是文件名或本地路径。
        url: OneBot 图片段中的 `url` 字段，可直接用于下载图片。
    """

    file: str = ""
    url: str = ""


def _normalize_size(*, value: str | None, default: str, allowed_sizes: list[str]) -> str:
    """规范化并校验绘图尺寸参数。

    Args:
        value: 调用方传入的尺寸。
        default: 插件默认尺寸。
        allowed_sizes: 配置允许的尺寸列表。

    Returns:
        规范化后的尺寸字符串。

    Raises:
        ValueError: 当尺寸不在允许列表中时抛出。
    """

    normalized = str(value or default).strip().lower()
    if normalized not in allowed_sizes:
        raise ValueError(f"size 仅支持: {', '.join(allowed_sizes)}")
    return normalized


def _normalize_option(*, value: str | None, default: str, allowed: set[str], name: str) -> str:
    """规范化可选字符串参数并限制可接受范围。

    Args:
        value: 原始输入值。
        default: 未传参时使用的默认值。
        allowed: 合法取值集合。
        name: 当前参数名，用于构造错误提示。

    Returns:
        规范化后的参数值。

    Raises:
        ValueError: 当值不在允许集合中时抛出。
    """

    normalized = str(value or default).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{name} 仅支持: {', '.join(sorted(allowed))}")
    return normalized


def _collect_image_refs_from_raw_messages(raw_messages: list[Any]) -> list[RawImageRef]:
    """从原始消息中提取图片段里的 `file` 和 `url` 引用。

    该方法只解析结构化消息段，不依赖 CQ 字符串反解析。
    这样可以优先复用消息事件里已经带上的直链 URL，避免某些 OneBot 实现中
    `get_image(file=...)` 无法再次命中文件缓存的问题。

    Args:
        raw_messages: 当前请求关联的原始消息列表。

    Returns:
        按出现顺序提取出的图片引用列表，重复项会被去重。
    """

    refs: list[RawImageRef] = []

    def _append_ref(file_value: Any, url_value: Any) -> None:
        file_text = str(file_value or "").strip()
        url_text = str(url_value or "").strip()
        if not file_text and not url_text:
            return
        ref = RawImageRef(file=file_text, url=url_text)
        if ref not in refs:
            refs.append(ref)

    def _extract_from_message_value(message_value: Any) -> None:
        if isinstance(message_value, list):
            iterable = message_value
        elif message_value is None or isinstance(message_value, (str, bytes, dict)):
            return
        else:
            try:
                iterable = list(message_value)
            except TypeError:
                return

        for segment in iterable:
            if isinstance(segment, dict):
                if str(segment.get("type") or "").strip() != "image":
                    continue
                data = segment.get("data")
                if not isinstance(data, dict):
                    continue
                _append_ref(data.get("file"), data.get("url"))
                continue

            if getattr(segment, "type", "") != "image":
                continue
            data = getattr(segment, "data", None)
            if not isinstance(data, dict):
                continue
            _append_ref(data.get("file"), data.get("url"))

    for raw_message in raw_messages:
        if isinstance(raw_message, dict):
            _extract_from_message_value(raw_message.get("message"))
            continue
        _extract_from_message_value(getattr(raw_message, "message", None))

    return refs


async def _resolve_input_images_from_messages(
    *,
    bot: Any,
    raw_messages: list[Any],
    max_size_bytes: int,
    max_count: int,
) -> ResolvedInputImages:
    """从 GTBot 上下文消息中提取编辑图所需的输入图片。

    当前约定：
    - 第 1 张图作为编辑原图。
    - 第 2 张图若存在，则作为 `mask` 上传。
    - 其余图片暂不消费，避免在 v1 中引入过多分支。

    Args:
        bot: 当前 OneBot Bot 实例。
        raw_messages: 原始消息列表。
        max_size_bytes: 单张输入图片允许的最大体积。
        max_count: 最多扫描的图片数量上限。

    Returns:
        包含原图和可选遮罩图的结构化结果。

    Raises:
        ValueError: 当当前上下文中找不到图片时抛出。
    """

    from plugins.GTBot.tools.vlm_image.tool import _call_onebot_get_image, _resolve_image_bytes_from_onebot_data

    image_refs = _collect_image_refs_from_raw_messages(raw_messages)
    if not image_refs:
        raise ValueError("当前消息中没有可用于编辑的图片")

    resolved: list[Any] = []
    for ref in image_refs[: min(2, max(1, int(max_count)))]:
        image_path = None
        image_bytes: bytes | None = None

        direct_file = Path(ref.file) if ref.file else None
        if direct_file is not None and direct_file.exists():
            image_path = direct_file
            image_bytes = direct_file.read_bytes()
            if len(image_bytes) > int(max_size_bytes):
                raise ValueError(f"输入图片过大: > {int(max_size_bytes)} bytes")
        elif ref.url:
            payload = {
                "file": ref.file,
                "url": ref.url,
            }
            image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
                payload,
                max_size_bytes=int(max_size_bytes),
            )
        elif ref.file:
            payload = await _call_onebot_get_image(bot, ref.file)
            image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
                payload,
                max_size_bytes=int(max_size_bytes),
            )
        else:
            continue

        source_name = ref.file or (str(image_path.name) if image_path is not None else "image.png")
        suffix = ""
        if image_path is not None:
            suffix = Path(str(image_path)).suffix
        if not suffix:
            suffix = Path(source_name).suffix or ".png"
        resolved.append(
            OpenAIInputImage(
                file_name=f"{Path(source_name).stem or 'image'}{suffix}",
                image_bytes=image_bytes,
            )
        )

    if not resolved:
        raise ValueError("当前消息中没有可用于编辑的图片")

    return ResolvedInputImages(
        images=(resolved[0],),
        mask_image=resolved[1] if len(resolved) > 1 else None,
    )


@tool("openai_draw_image")
async def openai_draw_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交一条 OpenAI 文生图任务，并在完成后回发结果到当前会话。

    该工具不会阻塞等待出图完成，而是将任务放入后台队列中异步执行。
    这样可以让 Agent 快速继续对话，同时复用 GTBot 现有的群聊和私聊发送链路。

    Args:
        prompt: 文生图提示词，不能为空。
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        size: 目标图片尺寸，未传时使用配置默认值。
        quality: 图片质量参数，未传时使用配置默认值。
        background: 背景参数，未传时使用配置默认值。
        target_user_id: 结果接收者 QQ 号，未传时默认发给当前用户。

    Returns:
        队列受理结果文本，包含任务 ID 和当前排队状态。

    Raises:
        ValueError: 当提示词为空、运行时上下文缺失或参数非法时抛出。
        RuntimeError: 当绘图队列已满时抛出。
    """

    text = str(prompt or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    cfg = get_openai_draw_plugin_config()
    if not cfg.enabled:
        raise ValueError("绘图插件当前已禁用")

    normalized_size = _normalize_size(
        value=size,
        default=cfg.default_size,
        allowed_sizes=list(cfg.allowed_sizes),
    )
    normalized_quality = _normalize_option(
        value=quality,
        default=cfg.default_quality,
        allowed={"auto", "low", "medium", "high", "standard", "hd"},
        name="quality",
    )
    normalized_background = _normalize_option(
        value=background,
        default=cfg.default_background,
        allowed={"auto", "transparent", "opaque"},
        name="background",
    )

    chat_type = str(getattr(ctx, "chat_type", "group") or "group")
    session_id = str(getattr(ctx, "session_id", "") or "").strip()
    group_id = int(getattr(ctx, "group_id", 0) or 0)
    requester_user_id = int(getattr(ctx, "user_id", 0) or 0)
    if requester_user_id <= 0:
        raise ValueError("运行时上下文缺少 user_id")

    if not session_id:
        if chat_type == "private":
            session_id = f"private:{requester_user_id}"
        elif group_id > 0:
            session_id = f"group:{group_id}"
        else:
            raise ValueError("运行时上下文缺少 session_id")

    target = requester_user_id
    if target_user_id is not None and int(target_user_id) > 0:
        target = int(target_user_id)
    if chat_type == "private" and target != requester_user_id:
        raise ValueError("私聊会话中 target_user_id 只能是当前用户")

    bot = getattr(ctx, "bot", None)
    message_manager = getattr(ctx, "message_manager", None)
    cache = getattr(ctx, "cache", None)
    if bot is None or message_manager is None or cache is None:
        raise ValueError("运行时上下文缺少 bot/message_manager/cache")

    manager = get_openai_draw_queue_manager()
    spec = OpenAIDrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=text,
        size=normalized_size,
        quality=normalized_quality,
        background=normalized_background,
        output_format=str(cfg.default_output_format),
        group_id=group_id if group_id > 0 else None,
        requester_user_id=requester_user_id,
        target_user_id=target,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
    )

    state = await manager.submit(spec)
    snap = await manager.snapshot()
    queued_count = int(snap.get("queued_count") or 0)
    running_count = int(snap.get("running_count") or 0)

    return (
        f"已提交绘图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"running={running_count} queued={queued_count}"
    )


@tool("openai_edit_image")
async def openai_edit_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    input_fidelity: str | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交一条 OpenAI 编辑图任务，并在完成后回发结果到当前会话。

    该工具会自动从当前上下文消息里提取图片作为编辑输入。
    当前约定同一条消息中的第一张图为原图、第二张图为可选 mask，便于用户直接通过发图或回复图片触发改图。

    Args:
        prompt: 编辑图提示词，不能为空。
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        size: 目标图片尺寸，未传时使用配置默认值。
        quality: 图片质量参数，未传时使用配置默认值。
        background: 背景参数，未传时使用配置默认值。
        input_fidelity: 输入保真度，支持 `low` 和 `high`。
        target_user_id: 结果接收者 QQ 号，未传时默认发给当前用户。

    Returns:
        队列受理结果文本，包含任务 ID 和当前排队状态。

    Raises:
        ValueError: 当提示词为空、运行时上下文缺失、参数非法或当前消息不含图片时抛出。
        RuntimeError: 当绘图队列已满时抛出。
    """

    text = str(prompt or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行时上下文")

    cfg = get_openai_draw_plugin_config()
    if not cfg.enabled:
        raise ValueError("绘图插件当前已禁用")

    normalized_size = _normalize_size(
        value=size,
        default=cfg.default_size,
        allowed_sizes=list(cfg.allowed_sizes),
    )
    normalized_quality = _normalize_option(
        value=quality,
        default=cfg.default_quality,
        allowed={"auto", "low", "medium", "high", "standard", "hd"},
        name="quality",
    )
    normalized_background = _normalize_option(
        value=background,
        default=cfg.default_background,
        allowed={"auto", "transparent", "opaque"},
        name="background",
    )
    normalized_input_fidelity = _normalize_option(
        value=input_fidelity,
        default=cfg.default_input_fidelity,
        allowed={"low", "high"},
        name="input_fidelity",
    )

    chat_type = str(getattr(ctx, "chat_type", "group") or "group")
    session_id = str(getattr(ctx, "session_id", "") or "").strip()
    group_id = int(getattr(ctx, "group_id", 0) or 0)
    requester_user_id = int(getattr(ctx, "user_id", 0) or 0)
    if requester_user_id <= 0:
        raise ValueError("运行时上下文缺少 user_id")

    if not session_id:
        if chat_type == "private":
            session_id = f"private:{requester_user_id}"
        elif group_id > 0:
            session_id = f"group:{group_id}"
        else:
            raise ValueError("运行时上下文缺少 session_id")

    target = requester_user_id
    if target_user_id is not None and int(target_user_id) > 0:
        target = int(target_user_id)
    if chat_type == "private" and target != requester_user_id:
        raise ValueError("私聊会话中 target_user_id 只能是当前用户")

    bot = getattr(ctx, "bot", None)
    message_manager = getattr(ctx, "message_manager", None)
    cache = getattr(ctx, "cache", None)
    if bot is None or message_manager is None or cache is None:
        raise ValueError("运行时上下文缺少 bot/message_manager/cache")

    resolved_inputs = await _resolve_input_images_from_messages(
        bot=bot,
        raw_messages=list(getattr(ctx, "raw_messages", []) or []),
        max_size_bytes=int(cfg.max_input_image_bytes),
        max_count=int(cfg.max_input_image_count),
    )

    manager = get_openai_draw_queue_manager()
    spec = OpenAIDrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=text,
        size=normalized_size,
        quality=normalized_quality,
        background=normalized_background,
        input_fidelity=normalized_input_fidelity,
        output_format=str(cfg.default_output_format),
        mode="edit",
        input_images=tuple(resolved_inputs.images),
        mask_image=resolved_inputs.mask_image,
        group_id=group_id if group_id > 0 else None,
        requester_user_id=requester_user_id,
        target_user_id=target,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
    )

    state = await manager.submit(spec)
    snap = await manager.snapshot()
    queued_count = int(snap.get("queued_count") or 0)
    running_count = int(snap.get("running_count") or 0)

    return (
        f"已提交改图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"input_fidelity={normalized_input_fidelity} running={running_count} queued={queued_count}"
    )
