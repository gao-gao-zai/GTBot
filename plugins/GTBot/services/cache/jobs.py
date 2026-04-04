from __future__ import annotations

from nonebot import get_bots
from nonebot.adapters.onebot.v11 import Bot

from ...ConfigManager import total_config
from ...Logger import logger
from . import user_cache_manager


def pick_active_bot() -> Bot | None:
    bots = get_bots()
    for bot in bots.values():
        if isinstance(bot, Bot):
            return bot
    return None


async def refresh_cache_job() -> None:
    bot = pick_active_bot()
    if bot is None:
        logger.warning("No active OneBot V11 bot available for cache refresh")
        return
    await user_cache_manager.refresh_due_entries(bot)


async def cleanup_cache_job() -> None:
    await user_cache_manager.cleanup_expired_entries()


async def setup_cache_jobs(scheduler) -> None:  # noqa: ANN001
    config = total_config.processed_configuration.config
    refresh_interval = getattr(config, "user_cache_update_interval_sec", 3600)
    expire_interval = getattr(config, "user_cache_expire_sec", 86400)

    await user_cache_manager.ensure_ready()

    scheduler.add_job(
        refresh_cache_job,
        "interval",
        seconds=max(60, refresh_interval),
        id="gtbot_user_cache_refresh",
        replace_existing=True,
    )

    cleanup_interval = max(300, min(expire_interval, refresh_interval * 2))
    scheduler.add_job(
        cleanup_cache_job,
        "interval",
        seconds=cleanup_interval,
        id="gtbot_user_cache_cleanup",
        replace_existing=True,
    )

    logger.info(
        "User cache jobs started: refresh=%ss cleanup=%ss",
        int(max(60, refresh_interval)),
        int(cleanup_interval),
    )
