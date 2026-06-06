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
    """描述好友管理插件的配置项。

    当前插件同时承载“删好友 Agent 工具”和“点赞能力”的配置。`enabled`
    仍只控制高风险的删好友能力；点赞工具是否暴露给 Agent 则使用独立开关，
    这样可以在不启用删好友的情况下单独开放更低风险的点赞能力。
    """

    enabled: bool = False
    expose_like_tool_to_agent: bool = False
    max_likes_per_user_per_day: int = Field(default=10, ge=0, le=1000)
    require_likes_before_zanwo: int = Field(default=0, ge=0, le=1000)
    exempt_admin_for_zanwo_gate: bool = True
    zanwo_gate_exempt_user_ids: list[int] = Field(default_factory=list)
    api_action: str = "delete_friend"
    timeout_sec: float = Field(default=15.0, ge=1.0, le=120.0)
    protected_friend_ids: list[int] = Field(default_factory=list)
    protected_friend_notes: dict[str, str] = Field(default_factory=dict)

    def is_protected(self, user_id: int) -> bool:
        """判断指定好友是否处于保护名单中。

        Args:
            user_id: 要检查的好友 QQ 号。

        Returns:
            当该好友位于保护列表中时返回 `True`，否则返回 `False`。
        """

        return int(user_id) in {int(x) for x in self.protected_friend_ids}

    def get_protected_note(self, user_id: int) -> str:
        """读取受保护好友的备注说明。

        Args:
            user_id: 要查询备注的好友 QQ 号。

        Returns:
            配置中记录的备注文本；未配置时返回空字符串。
        """

        return str(self.protected_friend_notes.get(str(int(user_id)), "")).strip()

    def is_zanwo_gate_exempt(self, user_id: int) -> bool:
        """判断某个用户是否命中 `赞我` 门槛豁免名单。

        Args:
            user_id: 要检查的用户 QQ 号。

        Returns:
            当该用户位于 `赞我` 门槛豁免名单中时返回 `True`。
        """

        return int(user_id) in {int(x) for x in self.zanwo_gate_exempt_user_ids}


_config_cache: FriendManagementPluginConfig | None = None


def _config_path() -> Path:
    """返回正式配置文件路径。"""

    return Path(__file__).with_name("config.json")


def _example_path() -> Path:
    """返回示例配置文件路径。"""

    return Path(__file__).with_name("config.json.example")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """以原子替换方式写入 JSON 配置文件。

    Args:
        path: 目标文件路径。
        data: 待写入的 JSON 对象。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_config() -> FriendManagementPluginConfig:
    """构造好友管理插件的默认配置。

    Returns:
        带有默认值的配置对象，供首次启动和配置损坏时回退使用。
    """

    return FriendManagementPluginConfig(
        enabled=False,
        expose_like_tool_to_agent=False,
        max_likes_per_user_per_day=10,
        require_likes_before_zanwo=0,
        exempt_admin_for_zanwo_gate=True,
        zanwo_gate_exempt_user_ids=[],
        api_action="delete_friend",
        timeout_sec=15.0,
        protected_friend_ids=[],
        protected_friend_notes={},
    )


def _ensure_default_files() -> FriendManagementPluginConfig:
    """确保正式配置与示例配置文件存在。

    Returns:
        默认配置对象。若磁盘上缺少配置文件，会顺手写入默认内容。
    """

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
    """读取并缓存好友管理插件配置。

    当配置文件缺失、为空或解析失败时，会回退到默认配置并把修正后的内容写回磁盘，
    以避免后续每次调用都重复进入异常分支。

    Returns:
        当前生效的好友管理插件配置对象。
    """

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
    """清空配置缓存并重新加载好友管理插件配置。

    Returns:
        重新读取后的最新配置对象。
    """

    global _config_cache
    _config_cache = None
    return get_friend_management_plugin_config()
