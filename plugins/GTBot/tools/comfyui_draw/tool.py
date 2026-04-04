from __future__ import annotations

import importlib
import random
from typing import Any, Callable, cast

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext

from .config import get_comfyui_draw_plugin_config

_manager_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".manager")

DrawJobSpec = cast(Any, getattr(_manager_mod, "DrawJobSpec"))
get_draw_queue_manager = cast(Callable[[], Any], getattr(_manager_mod, "get_draw_queue_manager"))


def _normalize_size(*, value: int | None, min_v: int, max_v: int, step: int, default: int, name: str) -> int:
    v = default if value is None else int(value)
    if v < int(min_v) or v > int(max_v):
        raise ValueError(f"{name} 必须在 {min_v}-{max_v} 之间")
    if step > 1 and (v % int(step) != 0):
        raise ValueError(f"{name} 必须是 {step} 的倍数")
    return v


@tool("comfyui_draw_image")
async def comfyui_draw_image(
    prompt: str,
    runtime: ToolRuntime[GroupChatContext],
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    target_user_id: int | None = None,
) -> str:
    """提交文生图任务（异步出图，完成后在群里@目标用户并发送图片）。

    Args:
        prompt: 文生图提示词。
        width: 图片宽度（像素）。不填使用插件默认值。
        height: 图片高度（像素）。不填使用插件默认值。
        seed: 随机种子。不填则自动生成。
        target_user_id: 目标用户 QQ 号。不填则默认使用当前事件的 user_id。

    Returns:
        提交结果文本（包含任务 ID / 队列状态）。

    Raises:
        ValueError: 当参数非法或缺少运行期上下文时抛出。
        RuntimeError: 当队列已满时抛出。
    """

    p = str(prompt or "").strip()
    if not p:
        raise ValueError("prompt 不能为空")

    ctx = getattr(runtime, "context", None)
    if ctx is None:
        raise ValueError("缺少运行期上下文")

    cfg = get_comfyui_draw_plugin_config()

    w = _normalize_size(
        value=width,
        min_v=int(cfg.min_width),
        max_v=int(cfg.max_width),
        step=int(cfg.size_step),
        default=int(cfg.default_width),
        name="width",
    )
    h = _normalize_size(
        value=height,
        min_v=int(cfg.min_height),
        max_v=int(cfg.max_height),
        step=int(cfg.size_step),
        default=int(cfg.default_height),
        name="height",
    )

    s = int(seed) if seed is not None else random.randint(0, 99999999999)

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
        raise ValueError("运行期上下文缺少 bot/message_manager/cache")

    manager = get_draw_queue_manager()

    spec = DrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=p,
        width=w,
        height=h,
        seed=s,
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
        f"已提交绘图任务 job={state.job_id} seed={s} size={w}x{h} "
        f"running={running_count} queued={queued_count}"
    )
