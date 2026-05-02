from __future__ import annotations

from time import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from nonebot.adapters.onebot.v11.message import Message

from ...constants import DEFAULT_BOT_NAME_PLACEHOLDER
from ..message.segments import serialize_message_segments
from .queue_payload import QueueMessageContent, prepare_queue_messages

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from .. import cache as CacheManager
    from ..message import GroupMessageManager


async def send_group_messages_direct(
    *,
    bot: "Bot",
    group_id: int,
    message_manager: "GroupMessageManager",
    cache: "CacheManager.UserCacheManager",
    messages: Sequence[QueueMessageContent],
) -> None:
    """直接向群聊发送一组非 Agent 消息，并同步回写消息账本。

    该路径用于系统反馈、后台任务通知等不应受 Agent 队列节奏影响的消息。
    函数会保留原有消息规范化与落库行为，但不会引入额外等待或排队。

    Args:
        bot: 当前 OneBot 机器人实例。
        group_id: 目标群号。
        message_manager: 统一消息管理器。
        cache: 用户缓存管理器，用于补齐机器人昵称。
        messages: 待直接发送的消息列表。
    """

    if not messages:
        return

    prepared_messages = await prepare_queue_messages(messages, scope=f"群组 {group_id}")
    bot_user_name = await cache.get_user_name(
        bot,
        int(bot.self_id),
    ) or DEFAULT_BOT_NAME_PLACEHOLDER

    for prepared_message in prepared_messages:
        await _send_group_message_direct(
            bot=bot,
            group_id=group_id,
            message_manager=message_manager,
            message=prepared_message,
            bot_user_name=bot_user_name,
        )


async def send_private_messages_direct(
    *,
    bot: "Bot",
    user_id: int,
    session_id: str,
    message_manager: "GroupMessageManager",
    cache: "CacheManager.UserCacheManager",
    messages: Sequence[QueueMessageContent],
) -> None:
    """直接向私聊发送一组非 Agent 消息，并同步回写消息账本。

    该路径用于系统反馈、后台工具通知等非角色化输出，避免它们被 Agent 队列
    堵塞。消息仍会先复用统一规范化逻辑，再逐条直发并写入统一消息表。

    Args:
        bot: 当前 OneBot 机器人实例。
        user_id: 目标私聊用户 ID。
        session_id: 当前私聊会话 ID。
        message_manager: 统一消息管理器。
        cache: 用户缓存管理器，用于补齐机器人昵称。
        messages: 待直接发送的消息列表。
    """

    if not messages:
        return

    prepared_messages = await prepare_queue_messages(messages, scope=f"session {session_id}")
    bot_user_name = await cache.get_user_name(
        bot,
        int(bot.self_id),
    ) or DEFAULT_BOT_NAME_PLACEHOLDER

    for prepared_message in prepared_messages:
        await _send_private_message_direct(
            bot=bot,
            user_id=user_id,
            session_id=session_id,
            message_manager=message_manager,
            message=prepared_message,
            bot_user_name=bot_user_name,
        )


async def _send_group_message_direct(
    *,
    bot: "Bot",
    group_id: int,
    message_manager: "GroupMessageManager",
    message: Message,
    bot_user_name: str,
) -> None:
    """直接发送单条群聊消息并回写消息表。

    Args:
        bot: 当前 OneBot 机器人实例。
        group_id: 目标群号。
        message_manager: 统一消息管理器。
        message: 已完成规范化的单条消息。
        bot_user_name: 机器人展示名称。
    """

    result = await bot.send_group_msg(group_id=group_id, message=message)
    await message_manager.add_chat_message(
        message_id=int(result["message_id"]),
        session_id=f"group:{int(group_id)}",
        group_id=int(group_id),
        peer_user_id=int(group_id),
        sender_user_id=int(bot.self_id),
        sender_name=bot_user_name,
        content=str(message),
        serialized_segments=serialize_message_segments(message),
        send_time=time(),
        is_withdrawn=False,
    )


async def _send_private_message_direct(
    *,
    bot: "Bot",
    user_id: int,
    session_id: str,
    message_manager: "GroupMessageManager",
    message: Message,
    bot_user_name: str,
) -> None:
    """直接发送单条私聊消息并回写消息表。

    Args:
        bot: 当前 OneBot 机器人实例。
        user_id: 目标私聊用户 ID。
        session_id: 当前私聊会话 ID。
        message_manager: 统一消息管理器。
        message: 已完成规范化的单条消息。
        bot_user_name: 机器人展示名称。
    """

    result = await bot.send_private_msg(user_id=user_id, message=message)
    await message_manager.add_chat_message(
        message_id=int(result["message_id"]),
        session_id=str(session_id),
        group_id=None,
        peer_user_id=int(user_id),
        sender_user_id=int(bot.self_id),
        sender_name=bot_user_name,
        content=str(message),
        serialized_segments=serialize_message_segments(message),
        send_time=time(),
        is_withdrawn=False,
    )
