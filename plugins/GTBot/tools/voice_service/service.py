from __future__ import annotations

from .audio_utils import VoiceServiceError
from .config import get_voice_service_plugin_config
from .models import AudioResult, CloneResult, ReplyVoiceMessage, SessionContext, TranscriptResult, VoiceItem
from .providers import AliyunCosyVoiceProvider, AliyunVoiceProvider, QQVoiceProvider
from .state import SessionVoiceState, get_voice_state_store


def _provider_for_mode(mode: str, bot: object | None = None):
    cfg = get_voice_service_plugin_config()
    if mode == "qq":
        return QQVoiceProvider(cfg, bot)
    if mode == "aliyun_cosyvoice":
        return AliyunCosyVoiceProvider(cfg)
    return AliyunVoiceProvider(cfg)


async def _get_state(session: SessionContext) -> SessionVoiceState:
    return await get_voice_state_store().get(session)


async def voice_list_voices(session: SessionContext, *, bot: object | None = None) -> list[VoiceItem]:
    state = await _get_state(session)
    provider = _provider_for_mode(state.synth_mode, bot)
    return await provider.list_voices(session, state)


async def voice_set_voice(
    session: SessionContext,
    voice_name: str,
    *,
    bot: object | None = None,
) -> SessionVoiceState:
    state = await _get_state(session)
    provider = _provider_for_mode(state.synth_mode, bot)
    voices = await provider.list_voices(session, state)
    selected = next(
        (
            item
            for item in voices
            if item.name == voice_name
            or item.display_name == voice_name
            or (item.voice_id and item.voice_id == voice_name)
        ),
        None,
    )
    if selected is None:
        raise VoiceServiceError(f"未找到音色: {voice_name}")

    store = get_voice_state_store()
    if selected.provider == "qq":
        return await store.set_qq_voice(session, selected.voice_id or selected.name)
    if selected.provider == "aliyun_cosyvoice":
        return await store.set_cosyvoice_voice(
            session,
            voice_name=selected.name,
            voice_id=selected.voice_id or selected.name,
            voice_type=selected.voice_type,
            target_model=selected.target_model,
        )
    return await store.set_qwen_voice(
        session,
        voice_name=selected.name,
        voice_id=selected.voice_id or selected.name,
        voice_type=selected.voice_type,
        target_model=selected.target_model,
    )


async def voice_synthesize(session: SessionContext, text: str, *, bot: object | None = None) -> AudioResult:
    state = await _get_state(session)
    provider = _provider_for_mode(state.synth_mode, bot)
    return await provider.synthesize(session, state, text)


async def voice_recognize_from_context(
    session: SessionContext,
    reply_voice: ReplyVoiceMessage,
    *,
    bot: object | None = None,
) -> TranscriptResult:
    state = await _get_state(session)
    provider = _provider_for_mode(state.recognize_mode, bot)
    return await provider.recognize(session, state, reply_voice)


async def voice_clone_voice(
    session: SessionContext,
    reply_voice: ReplyVoiceMessage,
    preferred_name: str,
    *,
    target_model: str | None = None,
) -> CloneResult:
    state = await _get_state(session)
    if state.synth_mode == "qq":
        raise VoiceServiceError("语音克隆音色仅支持阿里云合成模式")

    provider = _provider_for_mode(state.synth_mode)
    result = await provider.clone_voice(session, state, reply_voice, preferred_name, target_model=target_model)
    store = get_voice_state_store()
    if state.synth_mode == "aliyun_cosyvoice":
        await store.set_cosyvoice_voice(
            session,
            voice_name=result.voice_name,
            voice_id=result.voice_id,
            voice_type="custom",
            target_model=result.target_model,
        )
    else:
        await store.set_qwen_voice(
            session,
            voice_name=result.voice_name,
            voice_id=result.voice_id,
            voice_type="custom",
            target_model=result.target_model,
        )
    return result
