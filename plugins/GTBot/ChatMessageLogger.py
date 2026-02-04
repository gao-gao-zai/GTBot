from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends, EventMessage
from nonebot import on, on_message, get_driver, on_notice, logger
from pathlib import Path

from .MassageManager import GroupMessageManager, get_message_manager, GroupMessageManager
from .Fun import message_to_text
from .DBmodel import GroupMessage

record_message = on_message(priority=-5, block=False)
recall_message = on_notice(priority=-5, block=False)



@record_message.handle()
async def handle_group_message(event: GroupMessageEvent, bot: Bot, msg: Message = EventMessage(), 
                               message_manager:GroupMessageManager=Depends(get_message_manager)):
    """记录群消息的处理函数"""

    msg_text = await message_to_text(event.original_message) # 如果直接使用msg，nonebot会把@消息等自动消除
    logger.debug(f"收到消息: {msg_text}")
    group_message = GroupMessage(
        message_id=event.message_id,
        group_id=event.group_id,
        user_id=event.user_id,
        user_name=event.sender.card or event.sender.nickname or "",
        content=msg_text,
        send_time=event.time,
        is_withdrawn=False
    )
    await message_manager.add_message(group_message)

@recall_message.handle()
async def handle_group_recall(event: GroupRecallNoticeEvent, bot: Bot,
                              message_manager:GroupMessageManager=Depends(get_message_manager)):
    """处理群消息撤回的函数"""
    await message_manager.update_message(event.message_id, is_withdrawn=True)