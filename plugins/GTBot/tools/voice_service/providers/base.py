from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AudioResult, CloneResult, ReplyVoiceMessage, SessionContext, TranscriptResult, VoiceItem
from ..state import SessionVoiceState


class BaseVoiceProvider(ABC):
    @abstractmethod
    async def list_voices(self, session: SessionContext, state: SessionVoiceState) -> list[VoiceItem]:
        raise NotImplementedError

    @abstractmethod
    async def synthesize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        text: str,
        *,
        rate: float | None = None,
        pitch: float | None = None,
        volume: int | None = None,
        style_text: str | None = None,
    ) -> AudioResult:
        raise NotImplementedError

    @abstractmethod
    async def recognize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
    ) -> TranscriptResult:
        raise NotImplementedError

    @abstractmethod
    async def clone_voice(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
        preferred_name: str,
        *,
        target_model: str | None = None,
    ) -> CloneResult:
        raise NotImplementedError
