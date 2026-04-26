from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from .config import get_voice_service_plugin_config
from .models import SessionContext, VoiceMode, VoiceType


class QQSessionState(BaseModel):
    current_voice: str | None = None


class AliyunEngineState(BaseModel):
    current_voice_name: str | None = None
    current_voice_id: str | None = None
    current_voice_type: VoiceType | None = None
    current_target_model: str | None = None


class SessionVoiceState(BaseModel):
    synth_mode: VoiceMode = "qq"
    recognize_mode: VoiceMode = "qq"
    qq: QQSessionState = Field(default_factory=QQSessionState)
    qwen: AliyunEngineState = Field(default_factory=AliyunEngineState)
    cosyvoice: AliyunEngineState = Field(default_factory=AliyunEngineState)


class CustomVoiceAlias(BaseModel):
    voice_name: str
    target_model: str | None = None


class VoiceServiceStateFile(BaseModel):
    sessions: dict[str, SessionVoiceState] = Field(default_factory=dict)
    qwen_custom_voice_aliases: dict[str, CustomVoiceAlias] = Field(default_factory=dict)
    cosyvoice_custom_voice_aliases: dict[str, CustomVoiceAlias] = Field(default_factory=dict)


class VoiceStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._cache: VoiceServiceStateFile | None = None

    async def _load(self) -> VoiceServiceStateFile:
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            self._cache = VoiceServiceStateFile()
            await self._save(self._cache)
            return self._cache

        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
            if isinstance(parsed, dict):
                sessions = parsed.get("sessions")
                if isinstance(sessions, dict):
                    for session_data in sessions.values():
                        if not isinstance(session_data, dict):
                            continue
                        legacy_mode = session_data.pop("current_mode", None)
                        if legacy_mode == "qq":
                            session_data.setdefault("synth_mode", "qq")
                            session_data.setdefault("recognize_mode", "qq")
                        elif legacy_mode == "aliyun":
                            session_data.setdefault("synth_mode", "aliyun_qwen")
                            session_data.setdefault("recognize_mode", "aliyun_qwen")
                        synth_mode = session_data.get("synth_mode")
                        recognize_mode = session_data.get("recognize_mode")
                        if synth_mode == "aliyun":
                            session_data["synth_mode"] = "aliyun_qwen"
                        if recognize_mode == "aliyun":
                            session_data["recognize_mode"] = "aliyun_qwen"
                        legacy_aliyun = session_data.pop("aliyun", None)
                        if isinstance(legacy_aliyun, dict):
                            session_data.setdefault("qwen", legacy_aliyun)
            self._cache = cast(
                VoiceServiceStateFile,
                VoiceServiceStateFile.model_validate(parsed if isinstance(parsed, dict) else {}),
            )
        except Exception:
            self._cache = VoiceServiceStateFile()
            await self._save(self._cache)
        return self._cache

    async def _save(self, state: VoiceServiceStateFile) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        await asyncio.to_thread(self._path.write_text, payload, encoding="utf-8")
        self._cache = state

    async def get(self, session: SessionContext) -> SessionVoiceState:
        async with self._lock:
            state = await self._load()
            if session.session_key not in state.sessions:
                state.sessions[session.session_key] = SessionVoiceState()
                await self._save(state)
            return cast(SessionVoiceState, state.sessions[session.session_key].model_copy(deep=True))

    async def set(self, session: SessionContext, session_state: SessionVoiceState) -> SessionVoiceState:
        async with self._lock:
            state = await self._load()
            state.sessions[session.session_key] = session_state
            await self._save(state)
            return session_state

    async def set_mode(self, session: SessionContext, mode: VoiceMode) -> SessionVoiceState:
        current = await self.get(session)
        current.synth_mode = mode
        current.recognize_mode = mode
        return await self.set(session, current)

    async def set_synth_mode(self, session: SessionContext, mode: VoiceMode) -> SessionVoiceState:
        current = await self.get(session)
        current.synth_mode = mode
        return await self.set(session, current)

    async def set_recognize_mode(self, session: SessionContext, mode: VoiceMode) -> SessionVoiceState:
        current = await self.get(session)
        current.recognize_mode = mode
        return await self.set(session, current)

    async def set_qq_voice(self, session: SessionContext, voice_name: str) -> SessionVoiceState:
        current = await self.get(session)
        current.qq.current_voice = voice_name
        return await self.set(session, current)

    async def set_qwen_voice(
        self,
        session: SessionContext,
        *,
        voice_name: str,
        voice_id: str | None,
        voice_type: VoiceType,
        target_model: str | None,
    ) -> SessionVoiceState:
        async with self._lock:
            state = await self._load()
            current = cast(
                SessionVoiceState,
                state.sessions.get(session.session_key, SessionVoiceState()).model_copy(deep=True),
            )
            current.qwen.current_voice_name = voice_name
            current.qwen.current_voice_id = voice_id
            current.qwen.current_voice_type = voice_type
            current.qwen.current_target_model = target_model
            state.sessions[session.session_key] = current
            if voice_type == "custom" and voice_id:
                state.qwen_custom_voice_aliases[voice_id] = CustomVoiceAlias(
                    voice_name=voice_name,
                    target_model=target_model,
                )
            await self._save(state)
            return current

    async def set_cosyvoice_voice(
        self,
        session: SessionContext,
        *,
        voice_name: str,
        voice_id: str | None,
        voice_type: VoiceType,
        target_model: str | None,
    ) -> SessionVoiceState:
        async with self._lock:
            state = await self._load()
            current = cast(
                SessionVoiceState,
                state.sessions.get(session.session_key, SessionVoiceState()).model_copy(deep=True),
            )
            current.cosyvoice.current_voice_name = voice_name
            current.cosyvoice.current_voice_id = voice_id
            current.cosyvoice.current_voice_type = voice_type
            current.cosyvoice.current_target_model = target_model
            state.sessions[session.session_key] = current
            if voice_type == "custom" and voice_id:
                state.cosyvoice_custom_voice_aliases[voice_id] = CustomVoiceAlias(
                    voice_name=voice_name,
                    target_model=target_model,
                )
            await self._save(state)
            return current

    async def get_qwen_custom_voice_aliases(self) -> dict[str, CustomVoiceAlias]:
        async with self._lock:
            state = await self._load()
            return {key: value.model_copy(deep=True) for key, value in state.qwen_custom_voice_aliases.items()}

    async def get_cosyvoice_custom_voice_aliases(self) -> dict[str, CustomVoiceAlias]:
        async with self._lock:
            state = await self._load()
            return {key: value.model_copy(deep=True) for key, value in state.cosyvoice_custom_voice_aliases.items()}


def build_session_context(*, user_id: int, group_id: int | None = None) -> SessionContext:
    if group_id is not None:
        return SessionContext(
            scope_type="group",
            scope_id=group_id,
            session_key=f"group:{group_id}",
            user_id=user_id,
            group_id=group_id,
        )
    return SessionContext(
        scope_type="private",
        scope_id=user_id,
        session_key=f"private:{user_id}",
        user_id=user_id,
        group_id=None,
    )


_state_store: VoiceStateStore | None = None


def get_voice_state_store() -> VoiceStateStore:
    global _state_store
    if _state_store is None:
        cfg = get_voice_service_plugin_config()
        _state_store = VoiceStateStore(cfg.state_file_path)
    return _state_store
