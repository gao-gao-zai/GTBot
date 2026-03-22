from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from pydantic import BaseModel, ConfigDict, Field

from .model import GroupMessage

if TYPE_CHECKING:
    from .CacheManager import UserCacheManager as _UserCacheManager
    from .MassageManager import GroupMessageManager as _GroupMessageManager

    GroupMessageManagerT = _GroupMessageManager
    UserCacheManagerT = _UserCacheManager
    LongMemoryManagerT = Any
else:
    GroupMessageManagerT = Any
    UserCacheManagerT = Any
    LongMemoryManagerT = Any


class GroupChatContext(BaseModel):
    """Runtime context shared by chat handlers, tools and middleware."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: Bot
    chat_type: str = "group"
    session_id: str = ""
    group_id: int | None = None
    user_id: int
    message_id: int | None = None
    event: MessageEvent | None = None
    message: Message | None = None
    message_manager: GroupMessageManagerT
    cache: UserCacheManagerT
    long_memory: LongMemoryManagerT | None = None
    streaming_enabled: bool = False
    raw_messages: list[GroupMessage] = Field(default_factory=list)
    transport: Any | None = None
