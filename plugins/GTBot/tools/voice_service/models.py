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
    """描述一次语音合成结果及其后续投递方式。

    该模型同时承载“平台直发”与“本地音频文件”两类结果。当 `delivery` 为
    `onebot_record` 时，调用方通常会读取 `audio_path` 并构造 OneBot 语音消息；
    当后续已将本地音频注册进 GT 文件系统时，可通过 `audio_file_ref` 把稳定
    引用继续传递给 Agent 或其他工具，而不暴露裸路径。
    """

    provider: VoiceMode
    delivery: DeliveryType
    voice_name: str
    audio_path: str | None = None
    audio_file_ref: str | None = None
    mime_type: str | None = None
    message_id: int | None = None

    def audio_file_path(self) -> Path | None:
        """返回本地音频文件路径对象。

        Returns:
            当当前结果包含本地音频文件时返回对应 `Path`；否则返回 `None`。
        """

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
