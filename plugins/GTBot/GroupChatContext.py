from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias



from pydantic import BaseModel, ConfigDict
from nonebot.adapters.onebot.v11.message import Message
from nonebot.adapters.onebot.v11.event import GroupMessageEvent
from nonebot.adapters.onebot.v11.bot import Bot



from .model import (
    GroupAllInfo, 
    GroupInfo, 
    GroupMemberInfo, 
    StrangerInfo, 
    UserProfile, 
    GroupProfile,
    GroupMessage
)
if TYPE_CHECKING:
    from .CacheManager import UserCacheManager as _UserCacheManager
    from .MassageManager import GroupMessageManager as _GroupMessageManager
    from .services.LongMemory import LongMemoryContainer as _LongMemoryManager

    GroupMessageManagerT: TypeAlias = _GroupMessageManager
    UserCacheManagerT: TypeAlias = _UserCacheManager
    LongMemoryManagerT: TypeAlias = _LongMemoryManager
else:
    # CLI/测试场景下不初始化 NoneBot：避免导入包含 get_driver() 副作用的模块。
    # 运行时仅需要这些字段“可放任意对象”，因此降级为 Any。
    GroupMessageManagerT: TypeAlias = Any
    UserCacheManagerT: TypeAlias = Any
    LongMemoryManagerT: TypeAlias = Any


class GroupChatContext(BaseModel):
    """群聊上下文类，用于存储群组聊天的相关信息。
    
    Attributes:
        bot (Bot): OneBot 机器人实例。
        group_id (int): 群组 ID。
        user_id (int): 用户 ID。
        message_id (int): 消息 ID。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bot: Bot
    event: GroupMessageEvent
    message: Message
    group_id: int
    user_id: int
    message_id: int
    session_id: str | None = None
    message_manager: GroupMessageManagerT
    cache: UserCacheManagerT
    long_memory: LongMemoryManagerT
    streaming_enabled: bool = False
    raw_messages: list[GroupMessage] = []
    """是否启用“流式输出到群聊”。

    该字段用于让 LangChain 中间件/工具在运行时获知当前调用是否处于流式模式。
    """
