from __future__ import annotations

import asyncio
from time import time

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.services.shared import fun as Fun

from .config import get_friend_management_plugin_config
from .usage_limits import calculate_like_send_times, get_friend_management_like_limit_manager

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


@tool("delete_friend")
async def delete_friend_tool(
    user_id: int,
    runtime: ToolRuntime[GroupChatContext],
    reason: str | None = None,
) -> str:
    """主动删除指定 QQ 好友。

    该工具属于高风险操作，只在插件总开关启用时才会真正调用底层 OneBot 删除接口。
    调用前会校验目标 QQ 号是否合法，并检查目标是否位于受保护列表中；命中保护名单时
    会直接返回拒绝结果，不触发任何外部副作用。

    Args:
        user_id: 要删除的好友 QQ 号，必须为正整数。
        runtime: LangChain 工具运行时，用于读取当前会话的 Bot 与操作者信息。
        reason: 可选的审计说明，会写入日志，便于后续追踪调用原因。

    Returns:
        面向 Agent 的执行结果摘要文本。

    Raises:
        ValueError: 当 `user_id` 非正整数，或运行时缺少 `bot` 时抛出。
        RuntimeError: 当底层删除好友接口超时或显式调用失败时抛出。
    """

    cfg = get_friend_management_plugin_config()
    if not bool(cfg.enabled):
        return "friend_management plugin is disabled in config"

    target_user_id = int(user_id)
    if target_user_id <= 0:
        raise ValueError("user_id must be a positive integer")

    if cfg.is_protected(target_user_id):
        note = cfg.get_protected_note(target_user_id)
        if note:
            return f"refused to delete protected friend {target_user_id}: {note}"
        return f"refused to delete protected friend {target_user_id}"

    ctx = runtime.context
    bot = getattr(ctx, "bot", None)
    if bot is None:
        raise ValueError("runtime.context.bot is required")

    operator_user_id = int(getattr(ctx, "user_id", 0) or 0)
    group_id = int(getattr(ctx, "group_id", 0) or 0)
    action = str(cfg.api_action or "delete_friend").strip() or "delete_friend"
    detail_reason = str(reason or "").strip()

    logger.warning(
        "friend_management delete requested: target_user_id=%s operator_user_id=%s group_id=%s reason=%s",
        target_user_id,
        operator_user_id,
        group_id,
        detail_reason,
    )

    try:
        await asyncio.wait_for(
            bot.call_api(action, user_id=target_user_id),
            timeout=float(cfg.timeout_sec),
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{action} timed out for user_id={target_user_id}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{action} failed for user_id={target_user_id}: {type(exc).__name__}: {exc!s}") from exc

    if detail_reason:
        return f"deleted friend {target_user_id} successfully, reason={detail_reason}"
    return f"deleted friend {target_user_id} successfully"


@tool("send_like")
async def send_like_tool(
    user_id: int,
    runtime: ToolRuntime[GroupChatContext],
) -> str:
    """给指定用户发送点赞并默认点满。

    该工具用于向 Agent 暴露较低风险的点赞能力，是否可见由
    `expose_like_tool_to_agent` 独立控制，而不跟删好友总开关绑死。工具内部固定
    使用 `times=10`，让 Agent 不需要再参与点赞次数决策，避免出现不一致行为。

    Args:
        user_id: 要点赞的目标用户 QQ 号，必须为正整数。
        runtime: LangChain 工具运行时，用于读取当前会话绑定的 Bot 实例。

    Returns:
        点赞执行结果摘要文本。若当天额度不足 10，则只会发送剩余额度允许的数量。

    Raises:
        ValueError: 当 `user_id` 非正整数，或运行时缺少 `bot` 时抛出。
        RuntimeError: 当底层点赞接口超时或显式调用失败时抛出。
    """

    cfg = get_friend_management_plugin_config()
    target_user_id = int(user_id)
    if target_user_id <= 0:
        raise ValueError("user_id must be a positive integer")

    ctx = runtime.context
    bot = getattr(ctx, "bot", None)
    if bot is None:
        raise ValueError("runtime.context.bot is required")

    operator_user_id = int(getattr(ctx, "user_id", 0) or 0)
    group_id = int(getattr(ctx, "group_id", 0) or 0)

    logger.info(
        "friend_management like requested: target_user_id=%s operator_user_id=%s group_id=%s",
        target_user_id,
        operator_user_id,
        group_id,
    )

    now_ts = time()
    send_times = calculate_like_send_times(cfg=cfg, user_id=target_user_id, now_ts=now_ts)
    if send_times <= 0:
        limit = int(cfg.max_likes_per_user_per_day)
        return f"user {target_user_id} has reached today's like limit ({limit}/{limit})"

    try:
        await asyncio.wait_for(
            Fun.send_like(bot, target_user_id, times=send_times),
            timeout=float(cfg.timeout_sec),
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"send_like timed out for user_id={target_user_id}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"send_like failed for user_id={target_user_id}: {type(exc).__name__}: {exc!s}") from exc

    get_friend_management_like_limit_manager().record_like(
        cfg=cfg,
        user_id=target_user_id,
        count=send_times,
        now_ts=now_ts,
    )
    if send_times < 10:
        return f"sent {send_times} likes to user {target_user_id} successfully (daily limit reached)"
    return f"sent like to user {target_user_id} successfully"
