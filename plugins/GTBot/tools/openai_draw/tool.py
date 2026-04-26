from __future__ import annotations

import importlib
from pathlib import Path
from collections.abc import Awaitable
from typing import Any, Callable, cast

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext

from .config import get_openai_draw_plugin_config

_manager_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".manager")

OpenAIDrawJobSpec = cast(Any, getattr(_manager_mod, "OpenAIDrawJobSpec"))
OpenAIInputImage = cast(Any, getattr(_manager_mod, "OpenAIInputImage"))
get_openai_draw_queue_manager = cast(Callable[[], Any], getattr(_manager_mod, "get_openai_draw_queue_manager"))


def _normalize_size(
    *,
    value: str | None,
    default: str,
    max_width: int,
    max_height: int,
    size_multiple: int,
    max_aspect_ratio: float,
    min_pixels: int,
    max_pixels: int,
) -> str:
    """规范化并校验绘图尺寸参数。

    当前实现对齐 `gpt-image-2` 的尺寸约束：支持 `auto`，也支持任意显式
    `宽x高` 尺寸，但需要同时满足边长上限、边长倍数、长宽比与总像素范围。
    这样既能兼容官方推荐尺寸，也能让调用方在合法范围内自由指定分辨率。

    Args:
        value: 调用方传入的尺寸。
        default: 插件默认尺寸。
        max_width: 允许的最大宽度。
        max_height: 允许的最大高度。
        size_multiple: 宽高必须满足的倍数约束。
        max_aspect_ratio: 长边与短边的最大比例。
        min_pixels: 允许的最小总像素数。
        max_pixels: 允许的最大总像素数。

    Returns:
        规范化后的尺寸字符串，或保留 `auto`。

    Raises:
        ValueError: 当尺寸格式非法，或不满足任一尺寸约束时抛出。
    """

    normalized = str(value or default).strip().lower()
    if normalized == "auto":
        return normalized

    width_text, sep, height_text = normalized.partition("x")
    if sep != "x":
        raise ValueError("size 必须为 宽x高 格式或 auto")
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise ValueError("size 必须为 宽x高 格式或 auto") from exc
    if width <= 0 or height <= 0:
        raise ValueError("size 的宽高必须大于 0")
    if width > int(max_width) or height > int(max_height):
        raise ValueError(f"size 的单边不能超过 {int(max_width)}x{int(max_height)}")
    if int(size_multiple) > 1 and (width % int(size_multiple) != 0 or height % int(size_multiple) != 0):
        raise ValueError(f"size 的宽高必须都是 {int(size_multiple)} 的倍数")

    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge <= 0:
        raise ValueError("size 的宽高必须大于 0")
    if float(long_edge) / float(short_edge) > float(max_aspect_ratio):
        raise ValueError(f"size 的长边与短边比例不能超过 {float(max_aspect_ratio):g}:1")

    total_pixels = width * height
    if total_pixels < int(min_pixels) or total_pixels > int(max_pixels):
        raise ValueError(f"size 的总像素必须在 {int(min_pixels)} 到 {int(max_pixels)} 之间")
    return f"{width}x{height}"


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


def _resolve_agent_default(*, value: str | None, config_default: str, allow_auto: bool) -> str:
    """解析 Agent tool 缺省参数的最终默认值。

    当工具参数本身支持 `auto` 时，Agent 未显式传参通常意味着希望交由上游模型自动决策，
    这里优先回落到 `auto`，而不是继续使用本地配置中的固定默认值。这样可以让工具描述与
    实际行为保持一致，也避免模型因为不了解配置文件而无意中触发固定尺寸或固定质量。

    Args:
        value: Agent 显式传入的原始参数值；空字符串和 `None` 都视为未传。
        config_default: 配置文件中的默认值，当当前参数不支持 `auto` 时作为兜底值使用。
        allow_auto: 当前参数是否允许使用 `auto` 作为缺省值。

    Returns:
        传给后续规范化逻辑的默认值。若 Agent 已显式传参，则保留配置默认值以避免影响
        现有校验路径；若未传且支持 `auto`，则返回 `auto`。
    """

    if str(value or "").strip():
        return config_default
    if allow_auto:
        return "auto"
    return config_default


async def _send_tool_feedback(*, ctx: Any, text: str) -> None:
    """在工具提交成功后向当前会话发送一条即时反馈。

    Agent tool 的返回文本未必会直接暴露给最终用户，因此这里优先尝试复用
    运行时注入的会话传输层，在任务进入后台队列的同一时刻补发一条简短状态消息。
    如果当前上下文没有可用传输层，函数会安静跳过，不影响工具主流程。

    Args:
        ctx: 当前工具运行时上下文对象。
        text: 需要发送给会话的反馈文本。
    """

    transport = getattr(ctx, "transport", None)
    if transport is None:
        return

    send_feedback = getattr(transport, "send_feedback", None)
    if not callable(send_feedback):
        return
    feedback_callable = cast(Callable[..., Awaitable[object]], send_feedback)
    await feedback_callable(str(text), at_sender=False)


def _parameter_error(message: str) -> str:
    """构造统一的参数错误返回文本。

    Agent tool 的参数校验失败属于可预期调用错误，不应把异常继续抛给上层链路。
    这里统一包装成简短文本返回，便于模型收到明确反馈后自行修正参数。

    Args:
        message: 原始参数错误摘要。

    Returns:
        带有统一前缀的参数错误文本。
    """

    return f"参数错误: {str(message).strip()}"


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
        size: 目标图片尺寸。支持 `auto`，或显式传入 `宽x高`。
            当传入显式尺寸时，宽高必须都是 `16` 的倍数、不得超过配置中的
            `max_width` / `max_height`、长短边比例不得超过 `max_aspect_ratio`，
            且总像素必须落在 `min_pixels` 到 `max_pixels` 之间；未传时默认使用 `auto`。
        quality: 图片质量参数；未传时默认使用 `auto`。
        background: 背景参数；未传时默认使用 `auto`。
        target_user_id: 结果接收者 QQ 号；未传时默认在任务完成后发送给当前用户。

    Returns:
        任务启动结果文本，包含任务 ID 和当前排队状态。

    参数说明补充：
        当前默认尺寸约束对齐 `gpt-image-2`：支持 `auto`，或显式传入 `宽x高`。
        显式尺寸要求宽高都为 `16` 的倍数，单边不超过 `3840`，长短边比例不超过
        `3:1`，总像素介于 `655360` 到 `8294400` 之间。

    Raises:
        RuntimeError: 当绘图队列已满时抛出。
    """

    try:
        text = str(prompt or "").strip()
        if not text:
            return _parameter_error("prompt 不能为空")

        ctx = getattr(runtime, "context", None)
        if ctx is None:
            return _parameter_error("缺少运行时上下文")

        cfg = get_openai_draw_plugin_config()
        if not cfg.enabled:
            return _parameter_error("绘图插件当前已禁用")

        normalized_size = _normalize_size(
            value=size,
            default=_resolve_agent_default(value=size, config_default=cfg.default_size, allow_auto=True),
            max_width=int(cfg.max_width),
            max_height=int(cfg.max_height),
            size_multiple=int(cfg.size_multiple),
            max_aspect_ratio=float(cfg.max_aspect_ratio),
            min_pixels=int(cfg.min_pixels),
            max_pixels=int(cfg.max_pixels),
        )
        normalized_quality = _normalize_option(
            value=quality,
            default=_resolve_agent_default(
                value=quality,
                config_default=cfg.default_quality,
                allow_auto=True,
            ),
            allowed={"auto", "low", "medium", "high", "standard", "hd"},
            name="quality",
        )
        normalized_background = _normalize_option(
            value=background,
            default=_resolve_agent_default(
                value=background,
                config_default=cfg.default_background,
                allow_auto=True,
            ),
            allowed={"auto", "transparent", "opaque"},
            name="background",
        )

        chat_type = str(getattr(ctx, "chat_type", "group") or "group")
        session_id = str(getattr(ctx, "session_id", "") or "").strip()
        group_id = int(getattr(ctx, "group_id", 0) or 0)
        requester_user_id = int(getattr(ctx, "user_id", 0) or 0)
        if requester_user_id <= 0:
            return _parameter_error("运行时上下文缺少 user_id")

        if not session_id:
            if chat_type == "private":
                session_id = f"private:{requester_user_id}"
            elif group_id > 0:
                session_id = f"group:{group_id}"
            else:
                return _parameter_error("运行时上下文缺少 session_id")

        target = requester_user_id
        if target_user_id is not None and int(target_user_id) > 0:
            target = int(target_user_id)
        if chat_type == "private" and target != requester_user_id:
            return _parameter_error("私聊会话中 target_user_id 只能是当前用户")

        bot = getattr(ctx, "bot", None)
        message_manager = getattr(ctx, "message_manager", None)
        cache = getattr(ctx, "cache", None)
        if bot is None or message_manager is None or cache is None:
            return _parameter_error("运行时上下文缺少 bot/message_manager/cache")
    except ValueError as exc:
        return _parameter_error(str(exc))

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

    result_text = (
        f"已启动异步绘图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"running={running_count} queued={queued_count} "
        "图片将由后台任务完成后另行发送，不会阻塞当前智能体。"
    )
    await _send_tool_feedback(ctx=ctx, text=result_text)
    return result_text


@tool("openai_edit_image")
async def openai_edit_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    images: list[str],
    size: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交一条 OpenAI 编辑图任务。

    该工具只负责启动后台改图任务，并立即返回任务是否提交成功的提示文本。
    改图在后台异步执行，不会阻塞当前智能体继续运行；图片结果仍由现有消息链路在任务完成后另行发送。
    `images` 为必填原图列表，支持本地文件路径、可直连下载的 URL，
    或可交给 OneBot `get_image` 解析的图片引用名。

    Args:
        prompt: 编辑图提示词，不得为空。
        runtime: LangChain 提供的运行时上下文，内部需携带 `GroupChatContext`。
        images: 原图引用列表，支持多张本地路径、URL 或 OneBot 图片引用名。
        size: 目标图片尺寸。支持 `auto`，或显式传入 `宽x高`。
            当传入显式尺寸时，宽高必须都是 `16` 的倍数、不得超过配置中的
            `max_width` / `max_height`、长短边比例不得超过 `max_aspect_ratio`，
            且总像素必须落在 `min_pixels` 到 `max_pixels` 之间；未传时默认使用 `auto`。
        quality: 图片质量参数；未传时默认使用 `auto`。
        background: 背景参数；未传时默认使用 `auto`。
        target_user_id: 结果接收者 QQ 号；未传时默认在任务完成后发送给当前用户。

    Returns:
        任务启动结果文本，包含任务 ID 和当前排队状态。

    参数说明补充：
        当前默认尺寸约束对齐 `gpt-image-2`：支持 `auto`，或显式传入 `宽x高`。
        显式尺寸要求宽高都为 `16` 的倍数，单边不超过 `3840`，长短边比例不超过
        `3:1`，总像素介于 `655360` 到 `8294400` 之间。

    Raises:
        RuntimeError: 当绘图队列已满时抛出。
    """

    try:
        text = str(prompt or "").strip()
        if not text:
            return _parameter_error("prompt 不能为空")

        ctx = getattr(runtime, "context", None)
        if ctx is None:
            return _parameter_error("缺少运行时上下文")

        cfg = get_openai_draw_plugin_config()
        if not cfg.enabled:
            return _parameter_error("绘图插件当前已禁用")

        normalized_size = _normalize_size(
            value=size,
            default=_resolve_agent_default(value=size, config_default=cfg.default_size, allow_auto=True),
            max_width=int(cfg.max_width),
            max_height=int(cfg.max_height),
            size_multiple=int(cfg.size_multiple),
            max_aspect_ratio=float(cfg.max_aspect_ratio),
            min_pixels=int(cfg.min_pixels),
            max_pixels=int(cfg.max_pixels),
        )
        normalized_quality = _normalize_option(
            value=quality,
            default=_resolve_agent_default(
                value=quality,
                config_default=cfg.default_quality,
                allow_auto=True,
            ),
            allowed={"auto", "low", "medium", "high", "standard", "hd"},
            name="quality",
        )
        normalized_background = _normalize_option(
            value=background,
            default=_resolve_agent_default(
                value=background,
                config_default=cfg.default_background,
                allow_auto=True,
            ),
            allowed={"auto", "transparent", "opaque"},
            name="background",
        )
        chat_type = str(getattr(ctx, "chat_type", "group") or "group")
        session_id = str(getattr(ctx, "session_id", "") or "").strip()
        group_id = int(getattr(ctx, "group_id", 0) or 0)
        requester_user_id = int(getattr(ctx, "user_id", 0) or 0)
        if requester_user_id <= 0:
            return _parameter_error("运行时上下文缺少 user_id")

        if not session_id:
            if chat_type == "private":
                session_id = f"private:{requester_user_id}"
            elif group_id > 0:
                session_id = f"group:{group_id}"
            else:
                return _parameter_error("运行时上下文缺少 session_id")

        target = requester_user_id
        if target_user_id is not None and int(target_user_id) > 0:
            target = int(target_user_id)
        if chat_type == "private" and target != requester_user_id:
            return _parameter_error("私聊会话中 target_user_id 只能是当前用户")

        bot = getattr(ctx, "bot", None)
        message_manager = getattr(ctx, "message_manager", None)
        cache = getattr(ctx, "cache", None)
        if bot is None or message_manager is None or cache is None:
            return _parameter_error("运行时上下文缺少 bot/message_manager/cache")

        input_images = await _resolve_input_images(
            bot=bot,
            images=images,
            max_size_bytes=int(cfg.max_input_image_bytes),
            max_count=int(cfg.max_input_image_count),
        )
    except ValueError as exc:
        return _parameter_error(str(exc))

    manager = get_openai_draw_queue_manager()
    spec = OpenAIDrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=text,
        size=normalized_size,
        quality=normalized_quality,
        background=normalized_background,
        output_format=str(cfg.default_output_format),
        mode="edit",
        input_images=input_images,
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

    result_text = (
        f"已启动异步改图任务 job={state.job_id} size={normalized_size} "
        f"quality={normalized_quality} background={normalized_background} "
        f"running={running_count} queued={queued_count} "
        "图片将由后台任务完成后另行发送，不会阻塞当前智能体。"
    )
    await _send_tool_feedback(ctx=ctx, text=result_text)
    return result_text
