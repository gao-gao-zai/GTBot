from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, cast

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext

from .config import get_openai_draw_plugin_config

_manager_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".manager")

OpenAIDrawJobSpec = cast(Any, getattr(_manager_mod, "OpenAIDrawJobSpec"))
OpenAIInputImage = cast(Any, getattr(_manager_mod, "OpenAIInputImage"))
get_openai_draw_queue_manager = cast(Callable[[], Any], getattr(_manager_mod, "get_openai_draw_queue_manager"))


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
        raise ValueError(f"size 仅支持 {', '.join(allowed_sizes)}")
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
        raise ValueError(f"{name} 仅支持 {', '.join(sorted(allowed))}")
    return normalized


async def _resolve_input_image(
    *,
    bot: Any,
    image: str,
    max_size_bytes: int,
    parameter_name: str,
) -> Any:
    """将单个图片参数解析为可上传的输入图片对象。

    支持本地文件路径、可下载 URL，以及可交给 OneBot `get_image` 解析的图片引用名。
    解析结果会统一转换为 `OpenAIInputImage`，以便复用现有编辑图任务提交流程。

    Args:
        bot: 当前 OneBot Bot 实例。
        image: 调用方传入的图片引用字符串。
        max_size_bytes: 单张输入图片允许的最大字节数。
        parameter_name: 当前参数名，用于构造明确的错误提示。

    Returns:
        可直接提交给编辑图任务的输入图片对象。

    Raises:
        ValueError: 当参数为空或图片内容无法解析时抛出。
    """

    from plugins.GTBot.tools.vlm_image.tool import _call_onebot_get_image, _resolve_image_bytes_from_onebot_data

    image_ref = str(image or "").strip()
    if not image_ref:
        raise ValueError(f"{parameter_name} 不能为空")

    image_path = None
    image_bytes: bytes | None = None
    direct_file = Path(image_ref)
    if direct_file.exists() and direct_file.is_file():
        image_path = direct_file
        image_bytes = direct_file.read_bytes()
        if len(image_bytes) > int(max_size_bytes):
            raise ValueError(f"{parameter_name} 对应图片过大: > {int(max_size_bytes)} bytes")
    elif image_ref.startswith(("http://", "https://")):
        image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
            {"file": Path(image_ref).name or f"{parameter_name}.png", "url": image_ref},
            max_size_bytes=int(max_size_bytes),
        )
    else:
        payload = await _call_onebot_get_image(bot, image_ref)
        image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
            payload,
            max_size_bytes=int(max_size_bytes),
        )

    source_name = str(image_path.name) if image_path is not None else Path(image_ref).name or f"{parameter_name}.png"
    suffix = Path(source_name).suffix or ".png"
    return OpenAIInputImage(
        file_name=f"{Path(source_name).stem or parameter_name}{suffix}",
        image_bytes=image_bytes,
    )


async def _resolve_input_images(
    *,
    bot: Any,
    images: list[str],
    max_size_bytes: int,
    max_count: int,
) -> tuple[Any, ...]:
    """批量解析显式传入的原图参数列表。

    该函数会保留调用方传入的图片顺序，并在进入上传链路前统一校验数量上限，
    以避免编辑图任务在上游接口处才暴露过量输入错误。

    Args:
        bot: 当前 OneBot Bot 实例。
        images: 调用方传入的原图引用列表。
        max_size_bytes: 单张输入图片允许的最大字节数。
        max_count: 允许上传的最大原图数量。

    Returns:
        解析完成后的输入图片对象元组。

    Raises:
        ValueError: 当原图列表为空、超过数量上限或其中任一项非法时抛出。
    """

    normalized_images = [str(item or "").strip() for item in images if str(item or "").strip()]
    if not normalized_images:
        raise ValueError("images 不能为空")
    if len(normalized_images) > int(max_count):
        raise ValueError(f"images 数量不能超过 {int(max_count)} 张")

    resolved: list[Any] = []
    for index, image in enumerate(normalized_images, start=1):
        resolved.append(
            await _resolve_input_image(
                bot=bot,
                image=image,
                max_size_bytes=int(max_size_bytes),
                parameter_name=f"images[{index}]",
            )
        )
    return tuple(resolved)


@tool("openai_draw_image")
async def openai_draw_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交一条 OpenAI 文生图任务。

    该工具只负责启动后台绘图任务，并立即返回任务是否提交成功的提示文本。
    出图在后台异步执行，不会阻塞当前智能体继续运行；图片结果仍由现有消息链路在任务完成后另行发送。

    Args:
        prompt: 文生图提示词，不得为空。
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        size: 目标图片尺寸；未传时使用配置默认值。
        quality: 图片质量参数；未传时使用配置默认值。
        background: 背景参数；未传时使用配置默认值。
        target_user_id: 结果接收者 QQ 号；未传时默认在任务完成后发送给当前用户。

    Returns:
        任务启动结果文本，包含任务 ID 和当前排队状态。

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
        f"已启动异步绘图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"running={running_count} queued={queued_count} "
        "图片将由后台任务完成后另行发送，不会阻塞当前智能体。"
    )


@tool("openai_edit_image")
async def openai_edit_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    images: list[str],
    mask: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    input_fidelity: str | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交一条 OpenAI 编辑图任务。

    该工具只负责启动后台改图任务，并立即返回任务是否提交成功的提示文本。
    改图在后台异步执行，不会阻塞当前智能体继续运行；图片结果仍由现有消息链路在任务完成后另行发送。
    `images` 为必填原图列表，`mask` 为可选遮罩图。它们都支持本地文件路径、
    可直连下载的 URL，或可交给 OneBot `get_image` 解析的图片引用名。

    Args:
        prompt: 编辑图提示词，不得为空。
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        images: 原图引用列表，支持多张本地路径、URL 或 OneBot 图片引用名。
        mask: 可选遮罩图引用，支持本地路径、URL 或 OneBot 图片引用名。
        size: 目标图片尺寸；未传时使用配置默认值。
        quality: 图片质量参数；未传时使用配置默认值。
        background: 背景参数；未传时使用配置默认值。
        input_fidelity: 输入保真度，仅支持 `low` 或 `high`。
        target_user_id: 结果接收者 QQ 号；未传时默认在任务完成后发送给当前用户。

    Returns:
        任务启动结果文本，包含任务 ID 和当前排队状态。

    Raises:
        ValueError: 当提示词为空、运行时上下文缺失、图片参数缺失或参数非法时抛出。
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

    input_images = await _resolve_input_images(
        bot=bot,
        images=images,
        max_size_bytes=int(cfg.max_input_image_bytes),
        max_count=int(cfg.max_input_image_count),
    )
    mask_image = None
    if mask is not None and str(mask).strip():
        mask_image = await _resolve_input_image(
            bot=bot,
            image=mask,
            max_size_bytes=int(cfg.max_input_image_bytes),
            parameter_name="mask",
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
        input_images=input_images,
        mask_image=mask_image,
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
        f"已启动异步改图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"input_fidelity={normalized_input_fidelity} running={running_count} queued={queued_count} "
        "图片将由后台任务完成后另行发送，不会阻塞当前智能体。"
    )
