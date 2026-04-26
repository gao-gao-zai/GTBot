from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


class FriendManagementPluginConfig(BaseModel):
    enabled: bool = False
    api_action: str = "delete_friend"
    timeout_sec: float = Field(default=15.0, ge=1.0, le=120.0)
    protected_friend_ids: list[int] = Field(default_factory=list)
    protected_friend_notes: dict[str, str] = Field(default_factory=dict)

    def is_protected(self, user_id: int) -> bool:
        return int(user_id) in {int(x) for x in self.protected_friend_ids}

    def get_protected_note(self, user_id: int) -> str:
        return str(self.protected_friend_notes.get(str(int(user_id)), "")).strip()


_config_cache: FriendManagementPluginConfig | None = None


def _config_path() -> Path:
    return Path(__file__).with_name("config.json")


def _example_path() -> Path:
    return Path(__file__).with_name("config.json.example")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_config() -> FriendManagementPluginConfig:
    return FriendManagementPluginConfig(
        enabled=False,
        api_action="delete_friend",
        timeout_sec=15.0,
        protected_friend_ids=[],
        protected_friend_notes={},
    )


def _ensure_default_files() -> FriendManagementPluginConfig:
    cfg = _default_config()
    payload = cfg.model_dump(mode="json")
    config_path = _config_path()
    example_path = _example_path()

    if not example_path.exists():
        _write_json(example_path, payload)
    if not config_path.exists():
        _write_json(config_path, payload)
    return cfg


def get_friend_management_plugin_config() -> FriendManagementPluginConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_cfg = _ensure_default_files()
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise TypeError("friend_management config.json must be a JSON object")
        _config_cache = FriendManagementPluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"friend_management config.json parse failed, fallback to defaults: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))
    return _config_cache


def reload_friend_management_plugin_config() -> FriendManagementPluginConfig:
    global _config_cache
    _config_cache = None
    return get_friend_management_plugin_config()
