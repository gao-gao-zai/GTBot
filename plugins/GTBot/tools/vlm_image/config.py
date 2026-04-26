from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from nonebot import logger
from pydantic import BaseModel, Field


class VLMImagePluginConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = Field(default=512, ge=1, le=8192)
    timeout_sec: float = Field(default=60.0, ge=1.0, le=600.0)

    max_image_size_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    title_recommended_chars: int | None = Field(default=None, ge=1, le=200)
    title_max_chars: int = Field(default=16, ge=1, le=200)
    description_recommended_chars: int | None = Field(default=None, ge=1, le=2000)
    description_max_chars: int = Field(default=120, ge=1, le=2000)

    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    allow_override_reserved: bool = False


_config_cache: VLMImagePluginConfig | None = None


def _config_path() -> Path:
    return Path(__file__).with_name("config.json")


def _legacy_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "vlm_image_config.json"


def _maybe_migrate_legacy_config(*, target: Path) -> None:
    legacy = _legacy_config_path()
    if target.exists() or not legacy.exists():
        return

    try:
        raw = legacy.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            return
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(target)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLMImage legacy config 迁移失败，已忽略: {exc!s}")


def _load_from_disk(*, path: Path) -> VLMImagePluginConfig:
    _maybe_migrate_legacy_config(target=path)

    if not path.exists():
        cfg = VLMImagePluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        data = parsed if isinstance(parsed, dict) else {}
        return cast(VLMImagePluginConfig, VLMImagePluginConfig.model_validate(data))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLMImage config.json 解析失败，将使用默认配置并覆盖写入: {exc!s}")
        cfg = VLMImagePluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg


def _save_to_disk(*, path: Path, cfg: VLMImagePluginConfig) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLMImage config.json 写入失败，已忽略: {exc!s}")


def get_vlm_image_plugin_config() -> VLMImagePluginConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    path = _config_path()
    _config_cache = _load_from_disk(path=path)
    return _config_cache


def save_vlm_image_plugin_config(cfg: VLMImagePluginConfig) -> None:
    global _config_cache
    _config_cache = cfg
    _save_to_disk(path=_config_path(), cfg=cfg)
