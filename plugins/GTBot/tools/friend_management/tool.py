from __future__ import annotations

import asyncio

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.GroupChatContext import GroupChatContext

from .config import get_friend_management_plugin_config

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
    """Delete a QQ friend proactively.
    Args:
        user_id: QQ user ID of the friend to delete.
        reason: Optional short reason for audit logging.
    Returns:
        Result summary text.
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

