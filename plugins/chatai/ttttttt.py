import sqlite3
from pathlib import Path
import sys
from nonebot import on_type, logger, on_notice, on_command
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment, GroupMessageEvent, PokeNotifyEvent, GroupIncreaseNoticeEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
sys.path.append(str(Path(__file__).parent))


g = on_command("test", block=True)



@g.handle()
async def _(bot: Bot, event: Event):
    user_id = event.get_user_id()
    if isinstance(event, GroupMessageEvent):
        group_user_data = await bot.get_group_member_info(group_id=event.group_id, user_id=int(user_id))
        await g.send(str(group_user_data))
        group_data = await bot.get_group_info(group_id=event.group_id)
        await g.send(str(group_data))
        
    user_data = await bot.get_stranger_info(user_id=int(user_id))
    await g.send(str(user_data))

