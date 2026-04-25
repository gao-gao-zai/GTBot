from __future__ import annotations

from .Logger import logger
from . import model
from .ConfigManager import TotalConfiguration
from .services.cache import UserCacheManager, user_cache_manager

from nonebot.adapters.onebot.v11.bot import Bot

def _try_register_nonebot_hooks() -> None:
    """在 NoneBot 环境下注册钩子与加载子模块。

    CLI/脚本测试时，NoneBot 往往未初始化，直接调用 `get_driver()` 会抛出异常。
    这里做成“尽力而为”的初始化：

    - NoneBot 已初始化：正常导入并注册事件。
    - NoneBot 未初始化：跳过注册，使包可被安全导入。

    Returns:
        None: 无返回值。
    """

    try:
        from nonebot import get_driver

        driver = get_driver()
    except Exception:
        return

    from . import (  # noqa: F401
        AdminHandlers,
        ChatLatencyHandlers,
        ChatAccessHandlers,
        ChatMessageLogger,
        ChatTriggers,
        GroupAutoTriggerHandlers,
        GroupKeywordTriggerHandlers,
        cache_tasks,
        group_auto_trigger_tasks,
    )



    @driver.on_bot_connect
    async def on_startup(bot: Bot) -> None:
        await user_cache_manager.set_bot_self_id(bot)


_try_register_nonebot_hooks()


__all__ = [
    "logger",
    "TotalConfiguration",
    "user_cache_manager",
    "UserCacheManager",
    "model",
]
