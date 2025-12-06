from torch import ge
from .Logger import logger
from .ConfigManager import TotalConfiguration
from .CacheManager import user_cache_manager, UserCacheManager
from . import cache_tasks
from . import ChatMessageLogger
from . import Chat

from nonebot import get_driver
from nonebot.adapters.onebot.v11.bot import Bot


# 添加bot缓存
@get_driver().on_bot_connect
async def on_startup(bot: Bot) -> None:
    await user_cache_manager.set_bot_self_id(bot)


__all__ = ["logger", "TotalConfiguration", "user_cache_manager", "CacheManager", "cache_tasks", "ChatMessageLogger", "Chat"]