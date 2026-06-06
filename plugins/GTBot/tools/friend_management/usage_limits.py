from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .config import FriendManagementPluginConfig, get_friend_management_plugin_config

_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class FriendManagementLikeLimitManager:
    """管理好友点赞能力的每日限额状态。

    该管理器只关心“机器人今天已经给某个用户点了多少个赞”，并按北京时间自然日做
    归档。状态采用轻量 JSON 持久化，避免把这类简单配额引入数据库或更重的依赖。
    当日期跨天时，旧日期桶会被清理，只保留当天仍然有效的计数。
    """

    def __init__(self, state_path: Path | None = None) -> None:
        """初始化点赞限额管理器。

        Args:
            state_path: 可选的状态文件路径。未提供时使用插件默认状态文件路径。
        """

        self._state_path = Path(state_path) if state_path is not None else Path(__file__).with_name("like_usage.json")

    def get_remaining_likes(self, *, cfg: FriendManagementPluginConfig, user_id: int, now_ts: float) -> int | None:
        """计算当前自然日内某个用户剩余可点赞数。

        Args:
            cfg: 当前插件配置。
            user_id: 目标用户 QQ 号。
            now_ts: 当前时间对应的 Unix 时间戳。

        Returns:
            剩余可点赞数；当配置为 `0` 表示不限额时返回 `None`。
        """

        limit = int(cfg.max_likes_per_user_per_day)
        if limit <= 0:
            return None

        state = self._load_state()
        day_key = self._day_key(now_ts=now_ts)
        self._cleanup_state(state=state, day_key=day_key)
        used = self._read_user_count(state=state, user_id=int(user_id), day_key=day_key)
        return max(limit - used, 0)

    def record_like(self, *, cfg: FriendManagementPluginConfig, user_id: int, count: int, now_ts: float) -> None:
        """记录一次实际已发送的点赞数量。

        Args:
            cfg: 当前插件配置。
            user_id: 本次被点赞的目标用户 QQ 号。
            count: 本次实际发送的点赞数量。小于等于 0 时忽略。
            now_ts: 本次发送对应的 Unix 时间戳。
        """

        limit = int(cfg.max_likes_per_user_per_day)
        normalized_count = int(count)
        if limit <= 0 or normalized_count <= 0:
            return

        state = self._load_state()
        day_key = self._day_key(now_ts=now_ts)
        self._cleanup_state(state=state, day_key=day_key)
        current = self._read_user_count(state=state, user_id=int(user_id), day_key=day_key)
        self._write_user_count(
            state=state,
            user_id=int(user_id),
            day_key=day_key,
            count=current + normalized_count,
        )
        self._save_state(state)

    def _day_key(self, *, now_ts: float) -> str:
        """按北京时间计算当前自然日键。

        Args:
            now_ts: 待转换的 Unix 时间戳。

        Returns:
            `YYYY-MM-DD` 形式的北京时间日期键。
        """

        now = datetime.fromtimestamp(float(now_ts), tz=_SHANGHAI_TZ)
        return now.strftime("%Y-%m-%d")

    def _load_state(self) -> dict[str, Any]:
        """读取并规范化本地限额状态。

        Returns:
            始终包含 `users` 顶层桶的状态字典。文件缺失或损坏时返回空状态。
        """

        if not self._state_path.exists():
            return self._empty_state()

        try:
            raw = self._state_path.read_text(encoding="utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            return self._empty_state()
        if not isinstance(parsed, dict):
            return self._empty_state()
        return self._normalize_state(parsed)

    def _save_state(self, state: dict[str, Any]) -> None:
        """将限额状态原子写回本地文件。

        Args:
            state: 待持久化的状态字典。
        """

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self._state_path)

    def _cleanup_state(self, *, state: dict[str, Any], day_key: str) -> None:
        """清理过期日期桶，只保留当前北京时间自然日的数据。

        Args:
            state: 待清理的状态字典，会原地修改。
            day_key: 当前有效的北京时间日期键。
        """

        users_bucket = self._ensure_bucket(state=state, key="users")
        cleaned_users: dict[str, dict[str, int]] = {}
        for raw_user_id, raw_bucket in users_bucket.items():
            if not isinstance(raw_bucket, dict):
                continue
            current_value = self._safe_int(raw_bucket.get(day_key))
            if current_value > 0:
                cleaned_users[str(raw_user_id)] = {day_key: current_value}
        state["users"] = cleaned_users

    def _normalize_state(self, value: dict[str, Any]) -> dict[str, Any]:
        """把外部读取到的原始 JSON 规范化为内部状态结构。

        Args:
            value: 原始状态字典。

        Returns:
            规范化后的内部状态字典。
        """

        state = self._empty_state()
        raw_users = value.get("users")
        if not isinstance(raw_users, dict):
            return state

        normalized_users: dict[str, dict[str, int]] = {}
        for raw_user_id, raw_bucket in raw_users.items():
            if not isinstance(raw_bucket, dict):
                continue
            normalized_bucket: dict[str, int] = {}
            for raw_day_key, raw_count in raw_bucket.items():
                normalized_count = self._safe_int(raw_count)
                if normalized_count > 0:
                    normalized_bucket[str(raw_day_key)] = normalized_count
            if normalized_bucket:
                normalized_users[str(raw_user_id)] = normalized_bucket
        state["users"] = normalized_users
        return state

    def _read_user_count(self, *, state: dict[str, Any], user_id: int, day_key: str) -> int:
        """读取某个用户在指定自然日内已发送的点赞数。"""

        users_bucket = self._ensure_bucket(state=state, key="users")
        user_bucket = self._ensure_bucket(state=users_bucket, key=str(user_id))
        return self._safe_int(user_bucket.get(day_key))

    def _write_user_count(self, *, state: dict[str, Any], user_id: int, day_key: str, count: int) -> None:
        """写入某个用户在指定自然日内的已发送点赞数。"""

        users_bucket = self._ensure_bucket(state=state, key="users")
        user_bucket = self._ensure_bucket(state=users_bucket, key=str(user_id))
        user_bucket[day_key] = self._safe_int(count)

    @staticmethod
    def _ensure_bucket(*, state: dict[str, Any], key: str) -> dict[str, Any]:
        """确保状态字典中存在某个子桶并返回它。"""

        bucket = state.get(key)
        if isinstance(bucket, dict):
            return bucket
        normalized: dict[str, Any] = {}
        state[key] = normalized
        return normalized

    @staticmethod
    def _safe_int(value: Any) -> int:
        """将任意外部值安全转换为非负整数。"""

        try:
            normalized = int(value)
        except Exception:
            return 0
        return normalized if normalized > 0 else 0

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        """构造空的点赞限额状态。"""

        return {"users": {}}


_friend_management_like_limit_manager: FriendManagementLikeLimitManager | None = None


def get_friend_management_like_limit_manager() -> FriendManagementLikeLimitManager:
    """返回全局共享的好友点赞限额管理器。"""

    global _friend_management_like_limit_manager
    if _friend_management_like_limit_manager is None:
        _friend_management_like_limit_manager = FriendManagementLikeLimitManager()
    return _friend_management_like_limit_manager


def calculate_like_send_times(*, cfg: FriendManagementPluginConfig, user_id: int, now_ts: float) -> int:
    """根据每日限额计算本次最多还能发送多少个赞。

    Args:
        cfg: 当前插件配置。
        user_id: 目标用户 QQ 号。
        now_ts: 当前时间对应的 Unix 时间戳。

    Returns:
        本次应发送的点赞数。返回 `0` 表示当天额度已用尽。
    """

    remaining = get_friend_management_like_limit_manager().get_remaining_likes(
        cfg=cfg,
        user_id=int(user_id),
        now_ts=now_ts,
    )
    if remaining is None:
        return 10
    return min(10, max(remaining, 0))
