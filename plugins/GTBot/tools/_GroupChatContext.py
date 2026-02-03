from __future__ import annotations
from typing import TYPE_CHECKING



from pydantic import BaseModel, ConfigDict
from nonebot.adapters.onebot.v11.message import Message
from nonebot.adapters.onebot.v11.event import GroupMessageEvent
from nonebot.adapters.onebot.v11.bot import Bot





from .. import CacheManager
from ..MassageManager import GroupMessageManager
from ..UserProfileManager import ProfileManager
from ..model import (
    GroupAllInfo, 
    GroupInfo, 
    GroupMemberInfo, 
    StrangerInfo, 
    UserProfile, 
    GroupProfile,
    GroupMessage
)


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
    message_manager: GroupMessageManager
    cache: CacheManager.UserCacheManager
    profile_manager: ProfileManager
