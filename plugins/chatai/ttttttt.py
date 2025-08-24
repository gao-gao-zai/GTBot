import sqlite3
from pathlib import Path
import sys
from nonebot import on_type, logger, on_notice, on_command
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    Message,
    MessageSegment,
    GroupMessageEvent,
    PokeNotifyEvent,
    GroupIncreaseNoticeEvent,
    GroupRecallNoticeEvent,
)
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
import traceback
sys.path.append(str(Path(__file__).parent))


g = on_command("撤回", block=True)


@g.handle()
async def _(bot: Bot, event: Event):
    if hasattr(event, "reply") and event.reply:
        try:
            await bot.delete_msg(message_id=event.reply.message_id)
        except Exception as e:
            await g.finish(f"删除失败：{e}")
    else:
        msg = event.get_plaintext()[5:].strip()
        try:
            await bot.delete_msg(message_id=int(msg))
        except Exception as e:
            await g.finish(f"删除失败：{e}")

z = on_command("赞", block=True)


@z.handle()
async def _(bot: Bot, event: Event):
    user_id = event.get_plaintext()[2:].strip()
    try:
        await bot.send_like(user_id=int(user_id))
    except Exception as e:
        await z.finish(f"赞失败：{e}")

c = on_command("戳", block=True)


@c.handle()
async def _(bot: Bot, event: GroupMessageEvent):
    user_id = event.get_plaintext()[2:].strip()
    try:
        await bot.call_api("group_poke", group_id=event.group_id, user_id=int(user_id))
    except Exception as e:
        await c.finish(f"戳失败：{e}")