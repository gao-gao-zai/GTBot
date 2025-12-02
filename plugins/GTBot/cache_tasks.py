from nonebot import get_driver, get_bots
from nonebot.adapters.onebot.v11 import Bot
from nonebot_plugin_apscheduler import scheduler

from .ConfigManager import total_config
from .Logger import logger
from .UserCacheManager import user_cache_manager

try:
    driver = get_driver()
except Exception:
    driver = None


def _pick_active_bot() -> Bot | None:
    """返回第一个在线的 OneBot V11 Bot 实例。"""
    bots = get_bots()
    for bot in bots.values():
        if isinstance(bot, Bot):
            return bot
    return None


async def _refresh_cache_job() -> None:
    """定时刷新所有超时的缓存记录。"""
    bot = _pick_active_bot()
    if bot is None:
        logger.warning("未找到可用的 OneBot V11 Bot，跳过用户信息缓存刷新任务")
        return
    await user_cache_manager.refresh_due_entries(bot)


async def _cleanup_cache_job() -> None:
    """定期清理长时间未访问的缓存记录。"""
    await user_cache_manager.cleanup_expired_entries()


if driver:
    @driver.on_startup
    async def _setup_cache_jobs() -> None:
        """在 NoneBot 启动阶段注册刷新与清理任务。"""
        config = total_config.processed_configuration.config
        refresh_interval = getattr(config, "user_cache_update_interval_sec", 3600)
        expire_interval = getattr(config, "user_cache_expire_sec", 86400)

        await user_cache_manager.ensure_ready()

        scheduler.add_job(
            _refresh_cache_job,
            "interval",
            seconds=max(60, refresh_interval),
            id="gtbot_user_cache_refresh",
            replace_existing=True,
        )

        cleanup_interval = max(300, min(expire_interval, refresh_interval * 2))
        scheduler.add_job(
            _cleanup_cache_job,
            "interval",
            seconds=cleanup_interval,
            id="gtbot_user_cache_cleanup",
            replace_existing=True,
        )

        logger.info(
            "用户缓存定时任务已启动：刷新间隔 %ss，清理间隔 %ss",
            int(max(60, refresh_interval)),
            int(cleanup_interval),
        )

