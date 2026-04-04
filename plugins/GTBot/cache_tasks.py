from __future__ import annotations

from nonebot import get_driver
from nonebot_plugin_apscheduler import scheduler

from .services.cache.jobs import setup_cache_jobs

try:
    driver = get_driver()
except Exception:
    driver = None


if driver:
    @driver.on_startup
    async def _setup_cache_jobs() -> None:
        await setup_cache_jobs(scheduler)
