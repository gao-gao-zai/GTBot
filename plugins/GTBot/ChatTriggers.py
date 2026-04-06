from __future__ import annotations

import random
from typing import cast

from nonebot import on_message
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, PrivateMessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import Depends, EventMessage
from nonebot.rule import to_me

from .Logger import logger
from .services import cache as CacheManager
from .services.chat.continuation import get_continuation_manager
from .services.chat.runtime import (
    ChatTriggerMode,
    ChatTurn,
    _build_group_session,
    _build_private_session,
    _build_transport,
    run_chat_turn,
)
from .services.message import GroupMessageManager, get_message_manager
from .services.shared import fun as Fun
from .services.trigger.keyword import get_group_keyword_trigger_manager

GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)
GroupChatKeywordTriggerRequest = on_message(priority=6, block=False)
GroupChatContinuationRequest = on_message(priority=7, block=False)
PrivateChatPassiveRequest = on_message(priority=5, block=False)


async def _build_group_turn(
    event: GroupMessageEvent,
    msg: Message,
    *,
    trigger_mode: ChatTriggerMode = "group_at",
) -> ChatTurn:
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
    normalized = str(text).strip()
    return normalized.startswith("/") or normalized.startswith("？")


def _is_bot_self_message(event: GroupMessageEvent, bot: Bot) -> bool:
    return str(int(event.user_id)) == str(getattr(bot, "self_id", "") or "").strip()


async def _extract_keyword_text(msg: Message) -> str:
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
            "群聊关键词触发命中但未通过概率判定: group_id={} probability={:.2f} keyword={!r}",
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
        "群聊关键词触发命中: group_id={} user_id={} keyword={!r} probability={:.2f}",
        group_id,
        int(event.user_id),
        matched_keyword,
        probability,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)


@GroupChatContinuationRequest.handle()
async def handle_group_continuation_request(
    event: MessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
) -> None:
    if not isinstance(event, GroupMessageEvent):
        return
    if _is_bot_self_message(event, bot):
        return
    if bool(getattr(event, "to_me", False)) or _message_mentions_bot(msg, bot):
        return

    session = _build_group_session(int(event.group_id))
    continuation_manager = get_continuation_manager()
    if not await continuation_manager.has_active_window(session.session_id):
        return

    keyword_text = await _extract_keyword_text(msg)
    if not keyword_text or _looks_like_command(keyword_text):
        return

    keyword_manager = get_group_keyword_trigger_manager()
    if await keyword_manager.is_group_enabled(int(event.group_id)):
        matched_keyword = await keyword_manager.find_matching_keyword(
            int(event.group_id),
            keyword_text,
        )
        if matched_keyword is not None:
            logger.debug(
                "skip continuation because keyword trigger matched: group_id={} user_id={} keyword={!r}",
                int(event.group_id),
                int(event.user_id),
                matched_keyword,
            )
            return

    await continuation_manager.register_incoming_message(
        session_id=session.session_id,
        group_id=int(event.group_id),
        message_id=int(event.message_id),
        bot=bot,
        msg_mg=msg_mg,
        cache=cache,
    )


@PrivateChatPassiveRequest.handle()
async def handle_private_chat_request(
    event: MessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
) -> None:
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
