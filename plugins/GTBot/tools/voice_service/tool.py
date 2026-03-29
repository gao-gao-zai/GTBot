from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool
from nonebot.adapters.onebot.v11.message import Message, MessageSegment

from plugins.GTBot.GroupChatContext import GroupChatContext
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context

from .audio_utils import (
    VoiceServiceError,
    _extract_reply_message_id_from_event,
    cleanup_expired_cache,
    file_uri,
    resolve_message_voice,
    resolve_reply_voice,
)
from .config import get_voice_service_plugin_config
from .models import SessionContext, VoiceItem
from .providers import AliyunCosyVoiceProvider, AliyunVoiceProvider, QQVoiceProvider
from .state import SessionVoiceState, build_session_context, get_voice_state_store

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


ToolMode = Literal["qq", "aliyun_qwen", "aliyun_cosyvoice"]
RecognizeToolMode = Literal["qq", "aliyun_qwen"]
_VOICE_SERVICE_CACHE_CLEANED_KEY = "voice_service_cache_cleaned"
_VOICE_SERVICE_SESSION_KEY = "voice_service_prefetched_session_key"
_VOICE_SERVICE_STATE_KEY = "voice_service_prefetched_state"
_VOICE_SERVICE_REPLY_VOICE_KEY = "voice_service_prefetched_reply_voice"


def _build_session_from_runtime(runtime: ToolRuntime[GroupChatContext]) -> SessionContext:
    ctx = runtime.context
    group_id = getattr(ctx, "group_id", None)
    user_id = getattr(ctx, "user_id", None)
    if user_id is None:
        raise ValueError("runtime.context.user_id 缺失，无法使用语音工具")
    return build_session_context(user_id=int(user_id), group_id=int(group_id) if group_id is not None else None)


def _provider_for_mode(mode: str, runtime: ToolRuntime[GroupChatContext]):
    cfg = get_voice_service_plugin_config()
    if mode == "qq":
        return QQVoiceProvider(cfg, getattr(runtime.context, "bot", None))
    if mode == "aliyun_cosyvoice":
        return AliyunCosyVoiceProvider(cfg)
    return AliyunVoiceProvider(cfg)


def _get_plugin_extra() -> dict[str, Any] | None:
    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return None
    return plugin_ctx.extra


def _get_prefetched_state(session: SessionContext) -> SessionVoiceState | None:
    extra = _get_plugin_extra()
    if extra is None:
        return None
    if extra.get(_VOICE_SERVICE_SESSION_KEY) != session.session_key:
        return None
    prefetched_state = extra.get(_VOICE_SERVICE_STATE_KEY)
    if not isinstance(prefetched_state, SessionVoiceState):
        return None
    return prefetched_state.model_copy(deep=True)


def _get_prefetched_reply_voice(message_id: int) -> Any | None:
    extra = _get_plugin_extra()
    if extra is None:
        return None
    reply_voice = extra.get(_VOICE_SERVICE_REPLY_VOICE_KEY)
    if reply_voice is None:
        return None
    if int(getattr(reply_voice, "reply_message_id", -1)) != int(message_id):
        return None
    return reply_voice


async def _cleanup_expired_cache_once() -> None:
    extra = _get_plugin_extra()
    if extra is not None and bool(extra.get(_VOICE_SERVICE_CACHE_CLEANED_KEY)):
        return
    cfg = get_voice_service_plugin_config()
    await cleanup_expired_cache(cfg)
    if extra is not None:
        extra[_VOICE_SERVICE_CACHE_CLEANED_KEY] = True


async def prewarm_voice_service_context(plugin_ctx: Any) -> None:
    runtime_context = getattr(plugin_ctx, "runtime_context", None)
    if runtime_context is None:
        return

    user_id = getattr(runtime_context, "user_id", None)
    if user_id is None:
        return

    cfg = get_voice_service_plugin_config()
    try:
        await cleanup_expired_cache(cfg)
        plugin_ctx.extra[_VOICE_SERVICE_CACHE_CLEANED_KEY] = True
    except Exception:
        logger.debug("voice_service prewarm: cleanup cache failed", exc_info=True)

    try:
        group_id = getattr(runtime_context, "group_id", None)
        session = build_session_context(
            user_id=int(user_id),
            group_id=int(group_id) if group_id is not None else None,
        )
        state = await get_voice_state_store().get(session)
        plugin_ctx.extra[_VOICE_SERVICE_SESSION_KEY] = session.session_key
        plugin_ctx.extra[_VOICE_SERVICE_STATE_KEY] = state
    except Exception:
        logger.debug("voice_service prewarm: preload session state failed", exc_info=True)

    bot = getattr(runtime_context, "bot", None)
    event = getattr(runtime_context, "event", None)
    reply_message_id = _extract_reply_message_id_from_event(event)
    if bot is None or event is None or not isinstance(reply_message_id, int):
        return

    try:
        plugin_ctx.extra[_VOICE_SERVICE_REPLY_VOICE_KEY] = await resolve_reply_voice(bot, event, cfg)
    except Exception:
        logger.debug("voice_service prewarm: preload reply voice failed", exc_info=True)


def _apply_selected_voice(state: SessionVoiceState, selected: VoiceItem) -> SessionVoiceState:
    updated = state.model_copy(deep=True)
    if selected.provider == "qq":
        updated.qq.current_voice = selected.voice_id or selected.name
        return updated
    if selected.provider == "aliyun_cosyvoice":
        updated.cosyvoice.current_voice_name = selected.name
        updated.cosyvoice.current_voice_id = selected.voice_id or selected.name
        updated.cosyvoice.current_voice_type = selected.voice_type
        updated.cosyvoice.current_target_model = selected.target_model
        return updated
    updated.qwen.current_voice_name = selected.name
    updated.qwen.current_voice_id = selected.voice_id or selected.name
    updated.qwen.current_voice_type = selected.voice_type
    updated.qwen.current_target_model = selected.target_model
    return updated


async def _resolve_tool_state(
    runtime: ToolRuntime[GroupChatContext],
    *,
    synth_mode: ToolMode | None = None,
    recognize_mode: RecognizeToolMode | None = None,
    voice_name: str | None = None,
) -> tuple[SessionContext, SessionVoiceState]:
    session = _build_session_from_runtime(runtime)
    state = _get_prefetched_state(session)
    if state is None:
        store = get_voice_state_store()
        state = await store.get(session)
    updated = state.model_copy(deep=True)
    if synth_mode is not None:
        updated.synth_mode = synth_mode
    if recognize_mode is not None:
        updated.recognize_mode = recognize_mode

    if voice_name:
        provider = _provider_for_mode(updated.synth_mode, runtime)
        voices = await provider.list_voices(session, updated)
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
        updated = _apply_selected_voice(updated, selected)
    return session, updated


@tool("voice_synthesize")
async def voice_synthesize_tool(
    text: str,
    runtime: ToolRuntime[GroupChatContext],
    synth_mode: ToolMode | None = None,
    voice_name: str | None = None,
    style_text: str | None = None,
    rate: float | None = None,
    pitch: float | None = None,
    volume: int | None = None,
) -> str:
    """合成并发送语音消息。

    Args:
        text: 要合成的文本。
        synth_mode: 可选，覆盖当前合成模式，可取 `qq`、`aliyun_qwen`、`aliyun_cosyvoice`。
        voice_name: 可选，临时指定本次合成使用的音色名或音色 ID，不会改写会话默认音色。
        style_text: 可选，使用自然语言描述说话语气、情绪或风格，仅 `aliyun_cosyvoice` 支持。
        rate: 可选，语速，仅 `aliyun_cosyvoice` 支持。
        pitch: 可选，语调，仅 `aliyun_cosyvoice` 支持。
        volume: 可选，音量，仅 `aliyun_cosyvoice` 支持。

    Returns:
        本次语音发送结果摘要。
    """
    content = str(text or "").strip()
    if not content:
        raise ValueError("text 不能为空")

    await _cleanup_expired_cache_once()
    session, state = await _resolve_tool_state(runtime, synth_mode=synth_mode, voice_name=voice_name)

    if state.synth_mode != "aliyun_cosyvoice" and any(
        value is not None for value in (style_text, rate, pitch, volume)
    ):
        raise ValueError("style_text、rate、pitch、volume 仅阿里云 CosyVoice 模式支持")

    provider = _provider_for_mode(state.synth_mode, runtime)
    result = await provider.synthesize(
        session,
        state,
        content,
        style_text=style_text,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )

    bot = getattr(runtime.context, "bot", None)
    event = getattr(runtime.context, "event", None)
    if bot is None or event is None:
        raise ValueError("runtime.context 缺少 bot/event，无法发送语音")

    if result.delivery == "qq_group_ai":
        return f"已用 {_label_mode(state.synth_mode)} 模式发送语音，音色={result.voice_name}"

    if not result.audio_path:
        raise VoiceServiceError("语音合成完成，但未返回音频文件")

    await bot.send(
        event=event,
        message=Message(MessageSegment.record(file=file_uri(Path(result.audio_path)))),
    )
    return f"已用 {_label_mode(state.synth_mode)} 模式发送语音，音色={result.voice_name}"


@tool("voice_recognize")
async def voice_recognize_tool(
    runtime: ToolRuntime[GroupChatContext],
    message_id: int | None = None,
    recognize_mode: RecognizeToolMode | None = None,
) -> str:
    """识别语音或音频文件消息为文本。

    Args:
        message_id: 可选，要识别的目标消息 ID；不填时默认识别当前回复消息。
        recognize_mode: 可选，覆盖当前识别模式，可取 `qq` 或 `aliyun_qwen`。

    Returns:
        识别得到的文本。
    """
    cfg = get_voice_service_plugin_config()
    await cleanup_expired_cache(cfg)
    session, state = await _resolve_tool_state(runtime, recognize_mode=recognize_mode)

    bot = getattr(runtime.context, "bot", None)
    event = getattr(runtime.context, "event", None)
    if bot is None:
        raise ValueError("runtime.context 缺少 bot，无法识别语音")

    if message_id is None:
        if event is None:
            raise ValueError("runtime.context 缺少 event，未提供 message_id 时无法定位目标语音")
        prefetched_reply_message_id = _extract_reply_message_id_from_event(event)
        reply_voice = (
            _get_prefetched_reply_voice(prefetched_reply_message_id)
            if isinstance(prefetched_reply_message_id, int)
            else None
        )
        if reply_voice is None:
            reply_voice = await resolve_reply_voice(bot, event, cfg)
    else:
        reply_voice = _get_prefetched_reply_voice(int(message_id))
        if reply_voice is None:
            reply_voice = await resolve_message_voice(bot, int(message_id), cfg)

    provider = _provider_for_mode(state.recognize_mode, runtime)
    result = await provider.recognize(session, state, reply_voice)
    return result.text


def _label_mode(mode: str) -> str:
    if mode == "qq":
        return "QQ API"
    if mode == "aliyun_cosyvoice":
        return "阿里云 CosyVoice"
    return "阿里云 Qwen"
