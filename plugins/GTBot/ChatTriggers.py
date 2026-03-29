from __future__ import annotations

import random
from typing import cast

from nonebot import on_message
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, PrivateMessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import Depends, EventMessage
from nonebot.rule import to_me

from . import CacheManager, Fun
from .ChatCore import (
    ChatTriggerMode,
    ChatTurn,
    _build_group_session,
    _build_private_session,
    _build_transport,
    run_chat_turn,
)
from .GroupKeywordTriggerManager import get_group_keyword_trigger_manager
from .Logger import logger
from .MassageManager import GroupMessageManager, get_message_manager

GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)
GroupChatKeywordTriggerRequest = on_message(priority=6, block=False)
PrivateChatPassiveRequest = on_message(priority=5, block=False)


async def _build_group_turn(
    event: GroupMessageEvent,
    msg: Message,
    *,
    trigger_mode: ChatTriggerMode = "group_at",
) -> ChatTurn:
    """将群聊事件转换为统一的 `ChatTurn`。

    Args:
        event: 当前群消息事件。
        msg: NoneBot 注入的原始消息对象。

    Returns:
        ChatTurn: 可直接交给聊天核心链路处理的一轮输入描述。
    """
    sender_name = event.sender.card or event.sender.nickname or ""
    input_text = await Fun.message_to_text(msg)
    return ChatTurn(
        session=_build_group_session(int(event.group_id)),
        sender_user_id=int(event.user_id),
        sender_name=sender_name,
        anchor_message_id=int(event.message_id),
        input_text=input_text,
        trigger_mode=cast(ChatTriggerMode, trigger_mode),
        source="passive",
        event=event,
        message=msg,
    )


async def _build_private_turn(event: PrivateMessageEvent, msg: Message) -> ChatTurn:
    """将私聊事件转换为统一的 `ChatTurn`。

    Args:
        event: 当前私聊消息事件。
        msg: NoneBot 注入的原始消息对象。

    Returns:
        ChatTurn: 可直接交给聊天核心链路处理的一轮输入描述。
    """
    sender_name = getattr(event.sender, "nickname", "") or ""
    input_text = await Fun.message_to_text(msg)
    return ChatTurn(
        session=_build_private_session(int(event.user_id)),
        sender_user_id=int(event.user_id),
        sender_name=sender_name,
        anchor_message_id=int(event.message_id),
        input_text=input_text,
        trigger_mode="private",
        source="passive",
        event=event,
        message=msg,
    )


def _message_mentions_bot(msg: Message, bot: Bot) -> bool:
    """判断一条群消息是否显式 @ 了机器人。"""

    bot_self_id = str(getattr(bot, "self_id", "") or "").strip()
    if not bot_self_id:
        return False

    for segment in msg:
        if getattr(segment, "type", "") != "at":
            continue
        data = getattr(segment, "data", {}) or {}
        if str(data.get("qq") or "").strip() == bot_self_id:
            return True
    return False


def _looks_like_command(text: str) -> bool:
    """判断一段文本是否更像命令而不是普通聊天。"""

    normalized = str(text).strip()
    return normalized.startswith("/") or normalized.startswith("／")


async def _extract_keyword_text(msg: Message) -> str:
    """提取用于关键词匹配的纯文本内容。"""

    plain_text = str(getattr(msg, "extract_plain_text", lambda: "")() or "").strip()
    if plain_text:
        return plain_text
    fallback_text = await Fun.message_to_text(msg)
    return str(fallback_text or "").strip()


@GroupChatProactiveRequest.handle()
async def handle_group_chat_request(
    event: GroupMessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
) -> None:
    """将群聊事件适配为统一聊天请求并交给核心链路处理。

    Args:
        event: 当前群消息事件。
        bot: OneBot 机器人实例。
        msg: NoneBot 注入的原始消息对象。
        msg_mg: 消息管理器依赖。
        cache: 用户缓存管理器依赖。
    """
    turn = await _build_group_turn(event, msg)
    transport = _build_transport(
        bot=bot,
        message_manager=msg_mg,
        cache=cache,
        turn=turn,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)


@GroupChatKeywordTriggerRequest.handle()
async def handle_group_keyword_trigger_request(
    event: MessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
) -> None:
    """处理群聊关键词触发入口，并在命中后进入统一聊天主链路。"""

    if not isinstance(event, GroupMessageEvent):
        return

    if bool(getattr(event, "to_me", False)) or _message_mentions_bot(msg, bot):
        return

    keyword_text = await _extract_keyword_text(msg)
    if not keyword_text or _looks_like_command(keyword_text):
        return

    manager = get_group_keyword_trigger_manager()
    group_id = int(event.group_id)
    if not await manager.is_group_enabled(group_id):
        return

    matched_keyword = await manager.find_matching_keyword(group_id, keyword_text)
    if matched_keyword is None:
        return

    probability = await manager.get_effective_probability(group_id)
    if probability <= 0:
        return

    if probability < 100 and random.random() * 100 >= probability:
        logger.debug(
            "群聊关键词触发命中但未通过概率判定: group_id=%s probability=%.2f keyword=%r",
            group_id,
            probability,
            matched_keyword,
        )
        return

    turn = await _build_group_turn(event, msg, trigger_mode="group_keyword")
    transport = _build_transport(
        bot=bot,
        message_manager=msg_mg,
        cache=cache,
        turn=turn,
    )
    logger.info(
        "群聊关键词触发命中: group_id=%s user_id=%s keyword=%r probability=%.2f",
        group_id,
        int(event.user_id),
        matched_keyword,
        probability,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)


@PrivateChatPassiveRequest.handle()
async def handle_private_chat_request(
    event: MessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
) -> None:
    """将私聊事件适配为统一聊天请求并交给核心链路处理。

    Args:
        event: 当前消息事件；只有私聊事件会继续处理。
        bot: OneBot 机器人实例。
        msg: NoneBot 注入的原始消息对象。
        msg_mg: 消息管理器依赖。
        cache: 用户缓存管理器依赖。
    """
    if not isinstance(event, PrivateMessageEvent):
        return

    turn = await _build_private_turn(event, msg)
    transport = _build_transport(
        bot=bot,
        message_manager=msg_mg,
        cache=cache,
        turn=turn,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)
