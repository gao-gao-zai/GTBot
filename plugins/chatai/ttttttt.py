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

sys.path.append(str(Path(__file__).parent))


g = on_command("test", block=True)


@g.handle()
async def _(bot: Bot, event: Event):
    await g.finish(event.get_plaintext())
