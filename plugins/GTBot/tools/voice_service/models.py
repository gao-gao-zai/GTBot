from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


VoiceMode = Literal["qq", "aliyun_qwen", "aliyun_cosyvoice"]
VoiceType = Literal["builtin", "custom"]
DeliveryType = Literal["onebot_record", "qq_group_ai"]
SessionScopeType = Literal["group", "private"]


class SessionContext(BaseModel):
    scope_type: SessionScopeType
    scope_id: int
    session_key: str
    user_id: int
    group_id: int | None = None

    @property
    def is_group(self) -> bool:
        return self.scope_type == "group"


class VoiceItem(BaseModel):
    provider: VoiceMode
    name: str
    display_name: str
    voice_type: VoiceType
    voice_id: str | None = None
    target_model: str | None = None
    description: str = ""
    created_at: str | None = None


class AudioResult(BaseModel):
    provider: VoiceMode
    delivery: DeliveryType
    voice_name: str
    audio_path: str | None = None
    mime_type: str | None = None
    message_id: int | None = None

    def audio_file_path(self) -> Path | None:
        if not self.audio_path:
            return None
        return Path(self.audio_path)


class TranscriptResult(BaseModel):
    provider: VoiceMode
    text: str
    language: str | None = None
    emotion: str | None = None


class CloneResult(BaseModel):
    provider: Literal["aliyun_qwen", "aliyun_cosyvoice"]
    voice_name: str
    voice_id: str
    target_model: str
    auto_selected: bool = True


class ReplyVoiceMessage(BaseModel):
    reply_message_id: int
    raw_message: dict[str, Any] = Field(default_factory=dict)
    record_file: str | None = None
    record_url: str | None = None
    source_path: str | None = None
    normalized_wav_path: str | None = None
    duration_sec: float | None = None
    file_size_bytes: int | None = None

    def normalized_path(self) -> Path | None:
        if not self.normalized_wav_path:
            return None
        return Path(self.normalized_wav_path)
