from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import OpenAIDrawPluginConfig, get_openai_draw_plugin_config

_GLOBAL_WINDOWS: tuple[tuple[str, str], ...] = (
    ("per_day", "今日"),
    ("per_week", "本周"),
)
_USER_WINDOWS: tuple[tuple[str, str], ...] = (
    ("per_hour", "本小时"),
    ("per_day", "今日"),
    ("per_week", "本周"),
    ("per_month", "本月"),
)


class OpenAIDrawUsageLimitManager:
    """管理 `openai_draw` 的本地次数限制状态。

    该管理器只负责两件事：在任务提交前检查当前用户是否还能继续提交绘图，以及在任务成功
    入队后把本次提交记入本地状态文件。状态采用轻量 JSON 持久化，并按自然小时、日、周、
    月拆分键值，而不是保存完整事件流水。
    """

    def __init__(self, state_path: Path | None = None) -> None:
        """初始化次数限制管理器。

        Args:
            state_path: 可选的状态文件路径。未提供时使用插件默认状态文件路径。
        """

        cfg = get_openai_draw_plugin_config()
        self._state_path = Path(state_path) if state_path is not None else cfg.usage_counter_path

    def is_exempt_user(self, *, cfg: OpenAIDrawPluginConfig, user_id: int) -> bool:
        """判断用户是否属于豁免名单。

        Args:
            cfg: 当前插件配置。
            user_id: 待判断的用户 ID。

        Returns:
            `True` 表示该用户跳过限额检查和记账；否则返回 `False`。
        """

        exempt_user_ids = {int(item) for item in cfg.usage_limits.exempt_user_ids}
        return int(user_id) in exempt_user_ids

    def ensure_can_submit(self, *, cfg: OpenAIDrawPluginConfig, user_id: int, now_ts: float) -> None:
        """检查当前用户是否还能继续提交绘图。

        该方法只做校验，不修改状态。调用方应在任务真正成功入队后，再调用
        `record_submission` 完成本次记账。

        Args:
            cfg: 当前插件配置。
            user_id: 本次提交任务的用户 ID。
            now_ts: 当前检查对应的 Unix 时间戳。

        Raises:
            RuntimeError: 当任一全局或个人周期配额已用尽时抛出。
        """

        if not cfg.usage_limits.enabled or self.is_exempt_user(cfg=cfg, user_id=user_id):
            return

        state = self._load_state()
        period_keys = self._period_keys(now_ts=now_ts)
        self._cleanup_state(state=state, period_keys=period_keys)

        for field_name, label in _GLOBAL_WINDOWS:
            limit = int(getattr(cfg.usage_limits.global_limits, field_name))
            if limit <= 0:
                continue
            used = self._read_global_count(
                state=state,
                field_name=field_name,
                period_key=period_keys[field_name],
            )
            if used >= limit:
                raise RuntimeError(f"绘图次数已达全局{label}上限 ({used}/{limit})")

        for field_name, label in _USER_WINDOWS:
            limit = int(getattr(cfg.usage_limits.user_limits, field_name))
            if limit <= 0:
                continue
            used = self._read_user_count(
                state=state,
                user_id=int(user_id),
                field_name=field_name,
                period_key=period_keys[field_name],
            )
            if used >= limit:
                raise RuntimeError(f"你{label}的绘图次数已达上限 ({used}/{limit})")

    def record_submission(self, *, cfg: OpenAIDrawPluginConfig, user_id: int, now_ts: float) -> None:
        """记录一次成功入队的绘图提交。

        Args:
            cfg: 当前插件配置。
            user_id: 本次提交任务的用户 ID。
            now_ts: 本次提交对应的 Unix 时间戳。
        """

        if not cfg.usage_limits.enabled or self.is_exempt_user(cfg=cfg, user_id=user_id):
            return

        state = self._load_state()
        period_keys = self._period_keys(now_ts=now_ts)
        self._cleanup_state(state=state, period_keys=period_keys)

        for field_name, _label in _GLOBAL_WINDOWS:
            self._increment_global_count(
                state=state,
                field_name=field_name,
                period_key=period_keys[field_name],
            )

        for field_name, _label in _USER_WINDOWS:
            self._increment_user_count(
                state=state,
                user_id=int(user_id),
                field_name=field_name,
                period_key=period_keys[field_name],
            )

        self._save_state(state)

    def _period_keys(self, *, now_ts: float) -> dict[str, str]:
        """按服务端本地时区计算各自然周期键。

        Args:
            now_ts: 待转换的 Unix 时间戳。

        Returns:
            包含小时、日、周、月四种自然周期键的字典。
        """

        now = datetime.fromtimestamp(float(now_ts)).astimezone()
        iso_year, iso_week, _iso_weekday = now.isocalendar()
        return {
            "per_hour": now.strftime("%Y-%m-%dT%H"),
            "per_day": now.strftime("%Y-%m-%d"),
            "per_week": f"{iso_year}-W{iso_week:02d}",
            "per_month": now.strftime("%Y-%m"),
        }

    def _load_state(self) -> dict[str, Any]:
        """读取并规范化本地计数状态。

        Returns:
            规范化后的状态字典。文件缺失或损坏时返回空状态。
        """

        path = self._state_path
        if not path.exists():
            return self._empty_state()

        try:
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            return self._empty_state()
        if not isinstance(parsed, dict):
            return self._empty_state()
        return self._normalize_state(parsed)

    def _save_state(self, state: dict[str, Any]) -> None:
        """将计数状态原子写回本地文件。

        Args:
            state: 待持久化的状态字典。
        """

        path = self._state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _cleanup_state(self, *, state: dict[str, Any], period_keys: dict[str, str]) -> None:
        """清理所有已过期周期键，避免状态文件无限增长。

        Args:
            state: 待清理的状态字典，会原地修改。
            period_keys: 当前时刻对应的有效周期键集合。
        """

        global_bucket = self._ensure_bucket(state=state, key="global")
        for field_name, _label in _GLOBAL_WINDOWS:
            current_key = period_keys[field_name]
            current_value = self._safe_int(self._ensure_bucket(state=global_bucket, key=field_name).get(current_key))
            global_bucket[field_name] = {current_key: current_value} if current_value > 0 else {}

        users_bucket = self._ensure_bucket(state=state, key="users")
        cleaned_users: dict[str, dict[str, dict[str, int]]] = {}
        for raw_user_id, raw_user_state in users_bucket.items():
            if not isinstance(raw_user_state, dict):
                continue
            normalized_user_state: dict[str, dict[str, int]] = {}
            for field_name, _label in _USER_WINDOWS:
                current_key = period_keys[field_name]
                field_bucket = raw_user_state.get(field_name)
                current_value = 0
                if isinstance(field_bucket, dict):
                    current_value = self._safe_int(field_bucket.get(current_key))
                if current_value > 0:
                    normalized_user_state[field_name] = {current_key: current_value}
            if normalized_user_state:
                cleaned_users[str(raw_user_id)] = normalized_user_state
        state["users"] = cleaned_users

    def _normalize_state(self, value: dict[str, Any]) -> dict[str, Any]:
        """将外部读取到的原始状态规范化为内部结构。

        Args:
            value: 原始状态字典。

        Returns:
            始终包含 `global` 和 `users` 顶层桶的规范化状态。
        """

        state = self._empty_state()
        raw_global = value.get("global")
        if isinstance(raw_global, dict):
            for field_name, _label in _GLOBAL_WINDOWS:
                field_bucket = raw_global.get(field_name)
                if not isinstance(field_bucket, dict):
                    continue
                normalized_field: dict[str, int] = {}
                for period_key, count in field_bucket.items():
                    normalized_count = self._safe_int(count)
                    if normalized_count > 0:
                        normalized_field[str(period_key)] = normalized_count
                state["global"][field_name] = normalized_field

        raw_users = value.get("users")
        if isinstance(raw_users, dict):
            normalized_users: dict[str, dict[str, dict[str, int]]] = {}
            for raw_user_id, raw_user_state in raw_users.items():
                if not isinstance(raw_user_state, dict):
                    continue
                normalized_user_state: dict[str, dict[str, int]] = {}
                for field_name, _label in _USER_WINDOWS:
                    field_bucket = raw_user_state.get(field_name)
                    if not isinstance(field_bucket, dict):
                        continue
                    normalized_field = {}
                    for period_key, count in field_bucket.items():
                        normalized_count = self._safe_int(count)
                        if normalized_count > 0:
                            normalized_field[str(period_key)] = normalized_count
                    if normalized_field:
                        normalized_user_state[field_name] = normalized_field
                if normalized_user_state:
                    normalized_users[str(raw_user_id)] = normalized_user_state
            state["users"] = normalized_users

        return state

    def _read_global_count(self, *, state: dict[str, Any], field_name: str, period_key: str) -> int:
        """读取某个全局周期的当前计数。"""

        global_bucket = self._ensure_bucket(state=state, key="global")
        field_bucket = self._ensure_bucket(state=global_bucket, key=field_name)
        return self._safe_int(field_bucket.get(period_key))

    def _increment_global_count(self, *, state: dict[str, Any], field_name: str, period_key: str) -> None:
        """给某个全局周期计数加一。"""

        global_bucket = self._ensure_bucket(state=state, key="global")
        field_bucket = self._ensure_bucket(state=global_bucket, key=field_name)
        field_bucket[period_key] = self._read_global_count(
            state=state,
            field_name=field_name,
            period_key=period_key,
        ) + 1

    def _read_user_count(self, *, state: dict[str, Any], user_id: int, field_name: str, period_key: str) -> int:
        """读取某个用户在指定周期内的当前计数。"""

        users_bucket = self._ensure_bucket(state=state, key="users")
        user_bucket = self._ensure_bucket(state=users_bucket, key=str(user_id))
        field_bucket = self._ensure_bucket(state=user_bucket, key=field_name)
        return self._safe_int(field_bucket.get(period_key))

    def _increment_user_count(self, *, state: dict[str, Any], user_id: int, field_name: str, period_key: str) -> None:
        """给某个用户的指定周期计数加一。"""

        users_bucket = self._ensure_bucket(state=state, key="users")
        user_bucket = self._ensure_bucket(state=users_bucket, key=str(user_id))
        field_bucket = self._ensure_bucket(state=user_bucket, key=field_name)
        field_bucket[period_key] = self._read_user_count(
            state=state,
            user_id=user_id,
            field_name=field_name,
            period_key=period_key,
        ) + 1

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
        """构造空的次数限制状态。"""

        return {
            "global": {field_name: {} for field_name, _label in _GLOBAL_WINDOWS},
            "users": {},
        }


_openai_draw_usage_limit_manager: OpenAIDrawUsageLimitManager | None = None


def get_openai_draw_usage_limit_manager() -> OpenAIDrawUsageLimitManager:
    """返回全局共享的绘图次数限制管理器。"""

    global _openai_draw_usage_limit_manager
    if _openai_draw_usage_limit_manager is None:
        _openai_draw_usage_limit_manager = OpenAIDrawUsageLimitManager()
    return _openai_draw_usage_limit_manager
