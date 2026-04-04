from __future__ import annotations

import random
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from .auto import GroupAutoTriggerManager, get_group_auto_trigger_manager
from ...Logger import logger

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot as _Bot

    BotT: TypeAlias = _Bot
else:
    BotT: TypeAlias = Any

try:
    from nonebot import get_driver, get_bots
    from nonebot.adapters.onebot.v11 import Bot as _RuntimeBot
    from nonebot_plugin_apscheduler import scheduler as _scheduler
except Exception:  # noqa: BLE001
    _RuntimeBot = None

    def get_bots() -> dict[str, Any]:
        return {}

    def get_driver() -> Any:
        raise RuntimeError("nonebot is not initialized")

    _scheduler = None


GROUP_AUTO_TRIGGER_SCAN_INTERVAL_SECONDS = 60.0


def _pick_active_bot() -> BotT | None:
    """返回第一个在线的 OneBot v11 Bot 实例。"""

    if _RuntimeBot is None:
        return None

    bots = get_bots()
    for bot in bots.values():
        if isinstance(bot, _RuntimeBot):
            return bot
    return None


async def run_group_auto_trigger_scan(
    *,
    bot: BotT,
    manager: GroupAutoTriggerManager | None = None,
    message_manager: Any,
    cache: Any,
    runner: Callable[..., Awaitable[None]] | None = None,
    now_ts: float | None = None,
    context_limit: int | None = None,
) -> list[int]:
    """执行一次群聊自动触发扫描并返回本轮成功触发的群号列表。

    Args:
        bot: 当前可用的 OneBot 机器人实例。
        manager: 自动触发管理器；未提供时使用全局单例。
        message_manager: 消息管理器，需支持 `get_recent_messages(...)`。
        cache: 用户缓存管理器，会透传给聊天主链路。
        runner: 实际执行自动触发的协程入口；未提供时延迟导入 `run_group_auto_chat_turn`。
        now_ts: 可选的当前时间戳，单测可固定时间。
        context_limit: 可选的最近消息数量上限；未提供时沿用聊天主链路配置。

    Returns:
        list[int]: 本轮成功进入聊天主链路的群号列表。
    """

    effective_manager = manager or get_group_auto_trigger_manager()
    scan_started_at = float(now_ts if now_ts is not None else time.time())
    triggered_groups: list[int] = []
    effective_runner = runner
    if effective_runner is None:
        from ..chat.runtime import config as chat_core_config
        from ..chat.runtime import run_group_auto_chat_turn

        effective_runner = run_group_auto_chat_turn
        if context_limit is None:
            context_limit = int(getattr(chat_core_config.chat_model, "maximum_number_of_incoming_messages", 10) or 10)

    effective_context_limit = max(1, int(context_limit or 10))

    whitelist = await effective_manager.list_entries()
    if not whitelist:
        return triggered_groups

    for group_id in whitelist:
        try:
            cooldown_seconds = await effective_manager.get_effective_cooldown_seconds(int(group_id))
            if not await effective_manager.is_cooldown_ready(
                int(group_id),
                now_ts=scan_started_at,
                cooldown_seconds=cooldown_seconds,
            ):
                continue

            probability = await effective_manager.get_effective_probability(int(group_id))
            if probability <= 0:
                continue
            if probability < 100 and random.random() * 100 >= probability:
                logger.debug(
                    "群聊自动触发命中白名单但未通过概率判定: group_id=%s probability=%.2f",
                    int(group_id),
                    probability,
                )
                continue

            recent_messages_desc = await message_manager.get_recent_messages(
                limit=effective_context_limit,
                group_id=int(group_id),
            )
            if not recent_messages_desc:
                continue

            recent_messages = list(reversed(list(recent_messages_desc)))
            latest_message = recent_messages[-1]
            trigger_meta = {
                "reason": "scheduled",
                "scheduler_run_at": scan_started_at,
                "latest_message_id": int(getattr(latest_message, "message_id", 0) or 0),
                "latest_message_user_id": int(getattr(latest_message, "user_id", 0) or 0),
                "recent_message_count": len(recent_messages),
                "has_serialized_message": bool(getattr(latest_message, "serialized_segments", None)),
                "cooldown_seconds": float(cooldown_seconds),
                "probability": float(probability),
            }

            await effective_runner(
                latest_message=latest_message,
                trigger_meta=trigger_meta,
                bot=bot,
                msg_mg=message_manager,
                cache=cache,
            )
            await effective_manager.mark_triggered(int(group_id), triggered_at=scan_started_at)
            triggered_groups.append(int(group_id))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "群聊自动触发扫描异常，已跳过当前群: group_id=%s error=%s",
                int(group_id),
                exc,
            )

    return triggered_groups


async def _group_auto_trigger_job() -> None:
    """定时执行群聊自动触发扫描。"""

    bot = _pick_active_bot()
    if bot is None:
        logger.warning("未找到可用的 OneBot V11 Bot，跳过群聊自动触发扫描")
        return

    from ..cache import get_user_cache_manager
    from ..message import get_message_manager

    message_manager = await get_message_manager()
    cache = await get_user_cache_manager()
    await run_group_auto_trigger_scan(
        bot=bot,
        manager=get_group_auto_trigger_manager(),
        message_manager=message_manager,
        cache=cache,
    )


try:
    driver = get_driver()
except Exception:
    driver = None


if driver and _scheduler is not None:
    active_scheduler = _scheduler

    @driver.on_startup
    async def _setup_group_auto_trigger_job() -> None:
        """在 NoneBot 启动时注册群聊自动触发定时任务。"""

        active_scheduler.add_job(
            _group_auto_trigger_job,
            "interval",
            seconds=GROUP_AUTO_TRIGGER_SCAN_INTERVAL_SECONDS,
            id="gtbot_group_auto_trigger_scan",
            replace_existing=True,
        )
        logger.info(
            "群聊自动触发定时任务已启动：扫描间隔 %ss",
            int(GROUP_AUTO_TRIGGER_SCAN_INTERVAL_SECONDS),
        )
