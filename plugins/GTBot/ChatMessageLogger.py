from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends
from nonebot import on, on_message, get_driver
from pathlib import Path

from .MassageManager import get_message_manager
from .Fun import message_to_text
from .model import GroupMessage

record_message = on_message(priority=5, block=False)



# @record_message.handle()
# async def handle_group_message(event: GroupMessageEvent, bot: Bot, msg: Message, 
#                                message_manager=Depends(get_message_manager)):
#     """记录群消息的处理函数"""
#     msg_text = await message_to_text(msg)
#     group_message = GroupMessage(
#         message_id=event.message_id,
#         group_id=event.group_id,
#         user_id=event.user_id,
#         user_name=event.sender.nickname,
#         content=msg_text,
#         send_time=event.time,
#         is_withdrawn=False
#     )