from __future__ import annotations

from typing import Any

from ..audio_utils import VoiceServiceError
from ..config import VoiceServicePluginConfig
from ..models import AudioResult, CloneResult, ReplyVoiceMessage, SessionContext, TranscriptResult, VoiceItem
from ..state import SessionVoiceState
from .base import BaseVoiceProvider


class QQVoiceProvider(BaseVoiceProvider):
    def __init__(self, cfg: VoiceServicePluginConfig, bot: Any | None) -> None:
        self.cfg = cfg
        self.bot = bot

    async def _post(self, action: str, payload: dict[str, Any]) -> Any:
        if not self.cfg.qq_api.enabled:
            raise VoiceServiceError("QQ API 模式未启用")
        if self.bot is None:
            raise VoiceServiceError("QQ 模式需要可用的 Bot 上下文")
        return await self.bot.call_api(action, **payload)

    @staticmethod
    def _unwrap_data(payload: Any) -> Any:
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    async def list_voices(self, session: SessionContext, state: SessionVoiceState) -> list[VoiceItem]:
        if session.group_id is None:
            raise VoiceServiceError("QQ 模式的音色列表仅支持群聊")

        payload = {
            "group_id": session.group_id,
            "chat_type": self.cfg.qq_api.chat_type,
        }
        result = self._unwrap_data(await self._post("get_ai_characters", payload))

        items: list[VoiceItem] = []
        if isinstance(result, list):
            source = result
        elif isinstance(result, dict):
            source = []
            for key in ("characters", "voice_list", "list", "items"):
                value = result.get(key)
                if isinstance(value, list):
                    source = value
                    break
        else:
            source = []

        for item in source:
            if not isinstance(item, dict):
                continue

            category = str(item.get("type") or "").strip()
            nested_characters = item.get("characters")
            if isinstance(nested_characters, list):
                for character in nested_characters:
                    if not isinstance(character, dict):
                        continue
                    raw_name = (
                        character.get("character_name")
                        or character.get("name")
                        or character.get("voice")
                        or character.get("character")
                        or character.get("character_id")
                    )
                    if raw_name is None:
                        continue
                    name = str(raw_name)
                    voice_id = str(
                        character.get("character_id")
                        or character.get("character")
                        or character.get("voice")
                        or name
                    )
                    description = str(character.get("desc") or character.get("description") or "")
                    if category:
                        description = f"{category} {description}".strip()
                    items.append(
                        VoiceItem(
                            provider="qq",
                            name=name,
                            display_name=name,
                            voice_type="builtin",
                            voice_id=voice_id,
                            description=description,
                        )
                    )
                continue

            raw_name = item.get("character_name") or item.get("name") or item.get("voice") or item.get("character")
            if raw_name is None:
                continue
            name = str(raw_name)
            items.append(
                VoiceItem(
                    provider="qq",
                    name=name,
                    display_name=name,
                    voice_type="builtin",
                    voice_id=str(item.get("character_id") or item.get("character") or name),
                    description=str(item.get("desc") or item.get("description") or ""),
                )
            )

        deduped: list[VoiceItem] = []
        seen_voice_ids: set[str] = set()
        for item in items:
            key = item.voice_id or item.name
            if key in seen_voice_ids:
                continue
            seen_voice_ids.add(key)
            deduped.append(item)
        return deduped

    async def synthesize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        text: str,
        *,
        style_text: str | None = None,
        rate: float | None = None,
        pitch: float | None = None,
        volume: int | None = None,
    ) -> AudioResult:
        if style_text is not None or rate is not None or pitch is not None or volume is not None:
            raise VoiceServiceError("QQ 模式不支持调节语速、语调或音量")
        if session.group_id is None:
            raise VoiceServiceError("QQ 模式仅支持群聊语音合成")
        voice_name = state.qq.current_voice
        if not voice_name:
            raise VoiceServiceError("QQ 模式尚未设置当前音色")

        payload = {
            "group_id": session.group_id,
            "chat_type": self.cfg.qq_api.chat_type,
            "character": voice_name,
            "text": text,
        }
        result = self._unwrap_data(await self._post("send_group_ai_record", payload))
        message_id = None
        if isinstance(result, dict):
            raw_message_id = result.get("message_id") or result.get("id")
            if raw_message_id is not None:
                try:
                    message_id = int(raw_message_id)
                except Exception:
                    message_id = None

        return AudioResult(
            provider="qq",
            delivery="qq_group_ai",
            voice_name=voice_name,
            message_id=message_id,
        )

    async def recognize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
    ) -> TranscriptResult:
        payload = {"message_id": reply_voice.reply_message_id}
        result = self._unwrap_data(await self._post("voice_msg_to_text", payload))

        text = ""
        if isinstance(result, str):
            text = result
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("transcript") or result.get("message") or "")

        text = text.strip()
        if not text:
            raise VoiceServiceError("QQ 语音识别未返回文本")
        return TranscriptResult(provider="qq", text=text)

    async def clone_voice(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
        preferred_name: str,
        *,
        target_model: str | None = None,
    ) -> CloneResult:
        raise VoiceServiceError("QQ 模式不支持克隆音色")
