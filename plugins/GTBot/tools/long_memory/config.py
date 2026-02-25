from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nonebot import logger
from pydantic import BaseModel, Field


class LongMemoryContainerConfig(BaseModel):
    qdrant_server_url: str = "http://localhost:6333/"
    qdrant_api_key: str | None = ""
    qdrant_collection_name: str = "long_memory"

    embed_service_type: str = "openai"
    embed_service_url: str = "http://localhost:30020/v1/embeddings"
    embed_model: str = "qwen3-embedding-0.6b"
    embed_api_key: str | None = ""

    notepad_max_length: int = Field(default=20, ge=1, le=200)
    max_sessions: int | None = Field(default=1000, ge=1, le=100000)
    session_timeout_seconds: float | None = Field(default=3600, ge=0.0, le=86400.0)


class LongMemoryRerankConfig(BaseModel):
    enable: bool = False
    service_url: str = "http://localhost:30021/rerank"
    api_key: str | None = None


class LongMemoryRecallConfigModel(BaseModel):
    context_capacity: int = Field(default=80, ge=1, le=1000)
    query_max_messages: int = Field(default=12, ge=1, le=200)
    query_max_chars: int = Field(default=800, ge=1, le=20000)

    refresh_message_threshold: int = Field(default=3, ge=1, le=100)
    idle_refresh_seconds: float = Field(default=2.0, ge=0.1, le=300.0)
    max_concurrent_refreshes: int = Field(default=3, ge=1, le=50)

    event_log_max_items: int = Field(default=2, ge=1, le=100)
    event_log_total_chars: int = Field(default=300, ge=0, le=20000)
    event_log_min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    public_knowledge_max_items: int = Field(default=10, ge=1, le=200)
    public_knowledge_total_chars: int = Field(default=200, ge=0, le=20000)
    public_knowledge_min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    user_profile_max_items: int = Field(default=10, ge=1, le=200)
    user_profile_total_chars: int = Field(default=300, ge=0, le=20000)
    user_profile_min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    group_profile_max_items: int = Field(default=5, ge=1, le=200)
    group_profile_total_chars: int = Field(default=300, ge=0, le=20000)
    group_profile_min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    group_profile_stable_max_items: int = Field(default=5, ge=1, le=200)
    group_profile_stable_total_chars: int = Field(default=300, ge=0, le=20000)


class LongMemoryIngestConfigModel(BaseModel):
    processed_capacity: int = Field(default=200, ge=1, le=5000)
    pending_capacity: int = Field(default=200, ge=1, le=5000)
    flush_pending_threshold: int = Field(default=20, ge=1, le=5000)
    idle_flush_seconds: float = Field(default=60.0, ge=0.1, le=86400.0)
    flush_max_messages: int = Field(default=20, ge=1, le=5000)

    max_concurrent_flushes: int = Field(default=2, ge=1, le=100)

    model_id: str = ""
    base_url: str = ""
    api_key: str = ""
    model_parameters: dict[str, Any] = Field(default_factory=dict)

    public_knowledge_similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    public_knowledge_max_items: int = Field(default=5, ge=1, le=200)

    event_log_similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    event_log_max_items: int = Field(default=5, ge=1, le=200)

    user_profile_similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    user_profile_max_items: int = Field(default=10, ge=1, le=200)

    group_profile_min_items_threshold: int = Field(default=0, ge=0, le=200)
    group_profile_max_items: int = Field(default=20, ge=1, le=2000)


class LongMemoryPostLLMIngestConfig(BaseModel):
    recent_n: int = Field(default=20, ge=1, le=200)
    delay_seconds: float = Field(default=0.3, ge=0.0, le=10.0)


class LongMemoryCleanupLayerConfig(BaseModel):
    enable: bool = True
    expire_days: int = Field(default=14, ge=1, le=3650)


class LongMemoryCleanupConfig(BaseModel):
    min_interval_seconds: float = Field(default=86400.0, ge=0.0, le=86400.0 * 3650)

    user_profile: LongMemoryCleanupLayerConfig = Field(default_factory=LongMemoryCleanupLayerConfig)
    group_profile: LongMemoryCleanupLayerConfig = Field(default_factory=LongMemoryCleanupLayerConfig)
    event_log: LongMemoryCleanupLayerConfig = Field(default_factory=LongMemoryCleanupLayerConfig)
    public_knowledge: LongMemoryCleanupLayerConfig = Field(default_factory=LongMemoryCleanupLayerConfig)


class LongMemoryPluginConfig(BaseModel):
    auto_init: bool = True
    container: LongMemoryContainerConfig = Field(default_factory=LongMemoryContainerConfig)
    rerank: LongMemoryRerankConfig = Field(default_factory=LongMemoryRerankConfig)
    recall: LongMemoryRecallConfigModel = Field(default_factory=LongMemoryRecallConfigModel)
    ingest: LongMemoryIngestConfigModel = Field(default_factory=LongMemoryIngestConfigModel)
    post_llm_ingest: LongMemoryPostLLMIngestConfig = Field(default_factory=LongMemoryPostLLMIngestConfig)
    cleanup: LongMemoryCleanupConfig = Field(default_factory=LongMemoryCleanupConfig)


_config_cache: LongMemoryPluginConfig | None = None


def _config_path() -> Path:
    return Path(__file__).with_name("config.json")


def _normalize_legacy_data(data: dict[str, Any]) -> dict[str, Any]:
    has_legacy = bool(
        "post_llm_ingest_recent_n" in data
        or "post_llm_ingest_delay_seconds" in data
    )
    missing_post = bool(
        "post_llm_ingest" not in data
        or not isinstance(data.get("post_llm_ingest"), dict)
    )

    missing_cleanup = bool(
        "cleanup" not in data
        or not isinstance(data.get("cleanup"), dict)
    )

    if not has_legacy and not missing_post and not missing_cleanup:
        return data

    out: dict[str, Any] = dict(data)

    if "post_llm_ingest" not in out or not isinstance(out.get("post_llm_ingest"), dict):
        out["post_llm_ingest"] = {}

    if "cleanup" not in out or not isinstance(out.get("cleanup"), dict):
        out["cleanup"] = {}

    post = out.get("post_llm_ingest")
    if isinstance(post, dict):
        if "post_llm_ingest_recent_n" in out and "recent_n" not in post:
            post["recent_n"] = out.get("post_llm_ingest_recent_n")
        if "post_llm_ingest_delay_seconds" in out and "delay_seconds" not in post:
            post["delay_seconds"] = out.get("post_llm_ingest_delay_seconds")

    out.pop("post_llm_ingest_recent_n", None)
    out.pop("post_llm_ingest_delay_seconds", None)
    return out


def _load_from_disk(*, path: Path) -> LongMemoryPluginConfig:
    if not path.exists():
        cfg = LongMemoryPluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        data = parsed if isinstance(parsed, dict) else {}
        normalized = _normalize_legacy_data(data)
        cfg = LongMemoryPluginConfig.model_validate(normalized)
        if normalized is not data:
            _save_to_disk(path=path, cfg=cfg)
        return cfg
    except Exception as exc:
        logger.warning(f"LongMemory config.json 解析失败，将使用默认配置并覆盖写入: {exc!s}")
        cfg = LongMemoryPluginConfig()
        _save_to_disk(path=path, cfg=cfg)
        return cfg


def _save_to_disk(*, path: Path, cfg: LongMemoryPluginConfig) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.warning(f"LongMemory config.json 写入失败，已忽略: {exc!s}")


def get_long_memory_plugin_config() -> LongMemoryPluginConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    path = _config_path()
    _config_cache = _load_from_disk(path=path)
    return _config_cache


def save_long_memory_plugin_config(cfg: LongMemoryPluginConfig) -> None:
    global _config_cache
    _config_cache = cfg
    _save_to_disk(path=_config_path(), cfg=cfg)


def update_long_memory_plugin_config(**kwargs: Any) -> LongMemoryPluginConfig:
    cfg = get_long_memory_plugin_config()
    new_cfg = cfg.model_copy(update=dict(kwargs))
    save_long_memory_plugin_config(new_cfg)
    return new_cfg
