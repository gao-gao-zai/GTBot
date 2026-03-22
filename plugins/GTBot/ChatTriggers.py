from __future__ import annotations

from nonebot import on_message
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, PrivateMessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import Depends, EventMessage
from nonebot.rule import to_me

from . import CacheManager, Fun
from .ChatCore import (
    ChatTurn,
    _build_group_session,
    _build_private_session,
    _build_transport,
    run_chat_turn,
)
from .MassageManager import GroupMessageManager, get_message_manager

GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)
PrivateChatPassiveRequest = on_message(priority=5, block=False)


async def _build_group_turn(event: GroupMessageEvent, msg: Message) -> ChatTurn:
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
        source="passive",
        event=event,
        message=msg,
    )


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
