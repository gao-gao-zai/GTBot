from __future__ import annotations

from datetime import datetime
from time import time
from typing import cast
from typing import Any
from zoneinfo import ZoneInfo

from local_plugins.nonebot_plugin_gt_permission import PermissionRole, has_role

from .config import FriendManagementPluginConfig

_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _current_shanghai_day_range(now_ts: float | None = None) -> tuple[int, int]:
    """计算当前北京时间自然日的起止 Unix 时间戳。

    Args:
        now_ts: 可选的当前时间戳。未提供时使用系统当前时间。

    Returns:
        `(start_ts, end_ts)`，均为秒级 Unix 时间戳，其中 `end_ts` 为次日 0 点。
    """

    current_ts = float(time() if now_ts is None else now_ts)
    now = datetime.fromtimestamp(current_ts, tz=_SHANGHAI_TZ)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_start = datetime.fromtimestamp(day_start.timestamp() + 24 * 60 * 60, tz=_SHANGHAI_TZ)
    return int(day_start.timestamp()), int(next_day_start.timestamp())


def _extract_today_like_count(payload: Any, *, liker_user_id: int, start_ts: int, end_ts: int) -> int:
    """从 `get_profile_like_me` 返回中提取某用户今天点过的赞数。

    LLOneBot 返回的 `users` 列表里通常会聚合同一个点赞者的总点赞次数，并给出最近一次
    点赞时间 `latestTime`。这里只把“最近一次点赞仍在今天”的记录视为今天有效，再读取
    对应的 `count`。当前实现假设同一用户当日点赞会聚合在单条记录中；若上游未来调整
    数据结构，这个辅助函数可以单独替换而不影响命令主流程。

    Args:
        payload: `get_profile_like_me` 的原始返回体。
        liker_user_id: 要查询的点赞者 QQ 号。
        start_ts: 北京时间当日 0 点时间戳。
        end_ts: 次日 0 点时间戳。

    Returns:
        该用户今天给机器人点过的赞数；未命中时返回 `0`。
    """

    users = payload.get("users") if isinstance(payload, dict) else None
    if not isinstance(users, list):
        return 0

    target_user_id = int(liker_user_id)
    for item in users:
        if not isinstance(item, dict):
            continue
        try:
            uin = int(item.get("uin") or 0)
        except Exception:
            continue
        if uin != target_user_id:
            continue

        latest_time_raw = item.get("latestTime")
        try:
            latest_time = int(latest_time_raw or 0)
        except Exception:
            latest_time = 0
        if latest_time < start_ts or latest_time >= end_ts:
            return 0

        try:
            count = int(item.get("count") or 0)
        except Exception:
            count = 0
        return max(count, 0)
    return 0


async def should_bypass_zanwo_gate(*, cfg: FriendManagementPluginConfig, user_id: int) -> bool:
    """判断某个用户是否应跳过 `赞我` 门槛校验。

    Args:
        cfg: 当前好友管理插件配置。
        user_id: 命令发起人的 QQ 号。

    Returns:
        当用户命中显式豁免名单，或配置允许管理员豁免且该用户具备管理员权限时返回 `True`。
    """

    normalized_user_id = int(user_id)
    if cfg.is_zanwo_gate_exempt(normalized_user_id):
        return True
    if bool(cfg.exempt_admin_for_zanwo_gate):
        return cast(bool, await has_role(normalized_user_id, PermissionRole.ADMIN))
    return False


async def get_today_like_count_for_bot_from_user(*, bot: Any, user_id: int) -> int:
    """查询某个用户今天给机器人点过多少个赞。

    Args:
        bot: 当前 OneBot Bot 实例。
        user_id: 要查询的点赞者 QQ 号。

    Returns:
        该用户今天给机器人点过的赞数。查询失败时返回 `0`。
    """

    start_ts, end_ts = _current_shanghai_day_range()
    try:
        payload = await bot.call_api("get_profile_like_me", start=0, count=100)
    except Exception:
        return 0
    return _extract_today_like_count(payload, liker_user_id=int(user_id), start_ts=start_ts, end_ts=end_ts)
