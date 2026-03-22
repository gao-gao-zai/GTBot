from nonebot import logger, on_message, on_notice
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import (
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    MessageEvent,
    PrivateMessageEvent,
)
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import Depends, EventMessage

from .Fun import message_to_text
from .MassageManager import GroupMessageManager, get_message_manager
from .model import GroupMessage

record_message = on_message(priority=1, block=False)
recall_message = on_notice(priority=1, block=False)


@record_message.handle()
async def handle_message(
    event: MessageEvent,
    bot: Bot,
    msg: Message = EventMessage(),
    message_manager: GroupMessageManager = Depends(get_message_manager),
) -> None:
    msg_text = await message_to_text(event.original_message)
    logger.debug(f"received message: {msg_text}")

    if isinstance(event, GroupMessageEvent):
        group_message = GroupMessage(
            message_id=event.message_id,
            group_id=event.group_id,
            user_id=event.user_id,
            user_name=event.sender.card or event.sender.nickname or "",
            content=msg_text,
            send_time=event.time,
            is_withdrawn=False,
        )
        await message_manager.add_message(group_message)
        return

    if isinstance(event, PrivateMessageEvent):
        await message_manager.add_chat_message(
            message_id=event.message_id,
            session_id=f"private:{int(event.user_id)}",
            group_id=None,
            peer_user_id=int(event.user_id),
            sender_user_id=int(event.user_id),
            sender_name=getattr(event.sender, "nickname", "") or "",
            content=msg_text,
            send_time=float(event.time),
            is_withdrawn=False,
        )


@recall_message.handle()
async def handle_group_recall(
    event: GroupRecallNoticeEvent,
    bot: Bot,
    message_manager: GroupMessageManager = Depends(get_message_manager),
) -> None:
    await message_manager.update_message(identify_by_msg_id=event.message_id, is_withdrawn=True)
