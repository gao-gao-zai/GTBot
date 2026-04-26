from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from nonebot import logger
from pydantic import BaseModel, Field

BASEURL = "http://127.0.0.1:33330"


class ComfyUIDrawPluginConfig(BaseModel):
    base_url: str = BASEURL

    timeout_sec: float = Field(default=60.0, ge=1.0, le=600.0)
    poll_interval_sec: float = Field(default=1.0, ge=0.2, le=10.0)
    max_wait_sec: float = Field(default=600.0, ge=1.0, le=36000.0)

    worker_concurrency: int = Field(default=1, ge=1, le=8)
    max_queue_size: int = Field(default=10, ge=1, le=200)

    default_width: int = Field(default=1024, ge=64, le=8192)
    default_height: int = Field(default=1024, ge=64, le=8192)

    min_width: int = Field(default=256, ge=64, le=8192)
    max_width: int = Field(default=1536, ge=64, le=8192)
    min_height: int = Field(default=256, ge=64, le=8192)
    max_height: int = Field(default=1536, ge=64, le=8192)
    size_step: int = Field(default=64, ge=1, le=512)


_config_cache: ComfyUIDrawPluginConfig | None = None


def _config_path() -> Path:
    return Path(__file__).with_name("config.json")


def _load_from_disk(*, path: Path) -> ComfyUIDrawPluginConfig:
    if not path.exists():
        cfg = ComfyUIDrawPluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        data = parsed if isinstance(parsed, dict) else {}
        return cast(ComfyUIDrawPluginConfig, ComfyUIDrawPluginConfig.model_validate(data))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ComfyUIDraw config.json 解析失败，将使用默认配置并覆盖写入: {exc!s}")
        cfg = ComfyUIDrawPluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg


def _save_to_disk(*, path: Path, cfg: ComfyUIDrawPluginConfig) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ComfyUIDraw config.json 写入失败，已忽略: {exc!s}")


def get_comfyui_draw_plugin_config() -> ComfyUIDrawPluginConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    path = _config_path()
    _config_cache = _load_from_disk(path=path)
    return _config_cache


def save_comfyui_draw_plugin_config(cfg: ComfyUIDrawPluginConfig) -> None:
    global _config_cache
    _config_cache = cfg
    _save_to_disk(path=_config_path(), cfg=cfg)
