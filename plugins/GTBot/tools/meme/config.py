from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from nonebot import logger
from pydantic import BaseModel, Field


class MemePluginConfig(BaseModel):
    """表情包插件配置。"""

    max_meme_count: int = Field(default=200, ge=1, le=10000)
    max_meme_size_bytes: int = Field(default=2 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    max_title_chars: int = Field(default=24, ge=1, le=200)
    max_injected_memes: int = Field(default=50, ge=1, le=500)


_config_cache: MemePluginConfig | None = None


def _config_path() -> Path:
    """返回插件配置文件路径。"""

    return Path(__file__).with_name("config.json")


def _save_to_disk(*, path: Path, cfg: MemePluginConfig) -> None:
    """将配置原子写入磁盘。"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"meme config 写入失败，已忽略: {exc!s}")


def _load_from_disk(*, path: Path) -> MemePluginConfig:
    """从磁盘加载配置，不存在时写入默认值。"""

    if not path.exists():
        cfg = MemePluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        data = parsed if isinstance(parsed, dict) else {}
        return cast(MemePluginConfig, MemePluginConfig.model_validate(data))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"meme config 解析失败，将使用默认配置覆盖写入: {exc!s}")
        cfg = MemePluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg


def get_meme_plugin_config() -> MemePluginConfig:
    """获取缓存后的表情包插件配置。"""

    global _config_cache
    if _config_cache is not None:
        return _config_cache

    _config_cache = _load_from_disk(path=_config_path())
    return _config_cache


def save_meme_plugin_config(cfg: MemePluginConfig) -> None:
    """更新内存与磁盘中的表情包插件配置。"""

    global _config_cache
    _config_cache = cfg
    _save_to_disk(path=_config_path(), cfg=cfg)
