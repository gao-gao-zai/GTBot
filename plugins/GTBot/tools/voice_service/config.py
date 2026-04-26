from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


PLUGIN_DIR = Path(__file__).resolve().parent


class PermissionConfig(BaseModel):
    manage_mode: Literal["admin", "all"] = "admin"
    set_voice: Literal["admin", "all"] = "admin"
    synthesize: Literal["admin", "all"] = "all"
    recognize: Literal["admin", "all"] = "all"
    clone_voice: Literal["admin", "all"] = "admin"


class QQAPIConfig(BaseModel):
    enabled: bool = True
    chat_type: int = 1


class AliyunQwenConfig(BaseModel):
    voice_clone_model: str = "qwen-voice-enrollment"
    tts_custom_model: str = "qwen3-tts-vc-realtime-2026-01-15"
    tts_builtin_model: str = "qwen3-tts-flash"
    language_type: str = "Chinese"
    sample_rate: int = 24000
    response_format: Literal["wav", "pcm", "mp3", "opus"] = "wav"


class AliyunCosyVoiceConfig(BaseModel):
    enabled: bool = True
    voice_clone_model: str = "voice-enrollment"
    tts_custom_model: str = "cosyvoice-v3-flash"
    tts_builtin_model: str = "cosyvoice-v3-flash"
    sample_rate: int = 24000
    response_format: Literal["wav", "mp3", "pcm"] = "mp3"


class AliyunConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    region: str = "cn-beijing"
    asr_model: str = "qwen3-asr-flash-realtime"
    timeout_sec: float = Field(default=60.0, ge=1.0, le=600.0)
    qwen: AliyunQwenConfig = Field(default_factory=AliyunQwenConfig)
    cosyvoice: AliyunCosyVoiceConfig = Field(default_factory=AliyunCosyVoiceConfig)


class AudioConfig(BaseModel):
    ffmpeg_path: str = "ffmpeg"
    max_sample_duration_sec: int = Field(default=60, ge=1, le=3600)
    max_sample_size_mb: int = Field(default=10, ge=1, le=100)
    target_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    target_channels: int = Field(default=1, ge=1, le=2)


class StorageConfig(BaseModel):
    state_file: str = "./data/state.json"
    cache_dir: str = "./data/cache"
    temp_ttl_sec: int = Field(default=3600, ge=60, le=7 * 24 * 3600)


class VoiceServicePluginConfig(BaseModel):
    command_prefix: str = "语音"
    scope_mode: Literal["session"] = "session"
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    qq_api: QQAPIConfig = Field(default_factory=QQAPIConfig)
    aliyun: AliyunConfig = Field(default_factory=AliyunConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (PLUGIN_DIR / path).resolve()

    @property
    def state_file_path(self) -> Path:
        return self.resolve_path(self.storage.state_file)

    @property
    def cache_dir_path(self) -> Path:
        return self.resolve_path(self.storage.cache_dir)

    @property
    def aliyun_http_base(self) -> str:
        region = self.aliyun.region.lower()
        if region.startswith("cn-"):
            return "https://dashscope.aliyuncs.com/api/v1"
        if region.startswith("ap-") or "singapore" in region or region == "sg":
            return "https://dashscope-intl.aliyuncs.com/api/v1"
        if region.startswith("us-"):
            return "https://dashscope-us.aliyuncs.com/api/v1"
        return "https://dashscope.aliyuncs.com/api/v1"

    @property
    def aliyun_ws_base(self) -> str:
        region = self.aliyun.region.lower()
        if region.startswith("cn-"):
            return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        if region.startswith("ap-") or "singapore" in region or region == "sg":
            return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
        if region.startswith("us-"):
            return "wss://dashscope-us.aliyuncs.com/api-ws/v1/realtime"
        return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    @property
    def aliyun_cosyvoice_ws_base(self) -> str:
        region = self.aliyun.region.lower()
        if region.startswith("cn-"):
            return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        if region.startswith("ap-") or "singapore" in region or region == "sg":
            return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
        if region.startswith("us-"):
            return "wss://dashscope-us.aliyuncs.com/api-ws/v1/inference"
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


_config_cache: VoiceServicePluginConfig | None = None


def _config_path() -> Path:
    return PLUGIN_DIR / "config.json"


def _config_example_path() -> Path:
    return PLUGIN_DIR / "config.json.example"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_default_files() -> VoiceServicePluginConfig:
    cfg = VoiceServicePluginConfig()
    payload = cfg.model_dump(mode="json")

    config_path = _config_path()
    example_path = _config_example_path()

    if not example_path.exists():
        _write_json(example_path, payload)

    if not config_path.exists():
        _write_json(config_path, payload)

    cfg.cache_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.state_file_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def get_voice_service_plugin_config() -> VoiceServicePluginConfig:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    default_cfg = _ensure_default_files()
    path = _config_path()

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise TypeError("voice_service config.json must be a JSON object")
        _upgrade_legacy_aliyun_config(parsed)
        _config_cache = VoiceServicePluginConfig.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"voice_service config.json parse failed, fallback to defaults: {exc!s}")
        _config_cache = default_cfg
        _write_json(path, _config_cache.model_dump(mode="json"))

    _config_cache.cache_dir_path.mkdir(parents=True, exist_ok=True)
    _config_cache.state_file_path.parent.mkdir(parents=True, exist_ok=True)
    return _config_cache


def reload_voice_service_plugin_config() -> VoiceServicePluginConfig:
    global _config_cache
    _config_cache = None
    return get_voice_service_plugin_config()


def _upgrade_legacy_aliyun_config(parsed: dict[str, Any]) -> None:
    aliyun = parsed.get("aliyun")
    if not isinstance(aliyun, dict):
        return
    if "qwen" in aliyun or "cosyvoice" in aliyun:
        return

    qwen_payload = {
        "voice_clone_model": aliyun.pop("voice_clone_model", "qwen-voice-enrollment"),
        "tts_custom_model": aliyun.pop("tts_custom_model", "qwen3-tts-vc-realtime-2026-01-15"),
        "tts_builtin_model": aliyun.pop("tts_builtin_model", "qwen3-tts-flash"),
        "language_type": aliyun.pop("language_type", "Chinese"),
        "sample_rate": aliyun.pop("sample_rate", 24000),
        "response_format": aliyun.pop("response_format", "wav"),
    }
    aliyun["qwen"] = qwen_payload
    aliyun["cosyvoice"] = {
        "enabled": True,
        "voice_clone_model": "voice-enrollment",
        "tts_custom_model": "cosyvoice-v3-flash",
        "tts_builtin_model": "cosyvoice-v3-flash",
        "sample_rate": 24000,
        "response_format": "mp3",
    }
