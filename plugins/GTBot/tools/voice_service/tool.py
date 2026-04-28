from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from langchain.tools import ToolRuntime, tool
from nonebot.adapters.onebot.v11.message import Message, MessageSegment

from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.services.file_registry import register_local_file
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

VoiceProvider: TypeAlias = QQVoiceProvider | AliyunVoiceProvider | AliyunCosyVoiceProvider

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
_VOICE_SERVICE_AUDIO_SOURCE_TYPE_BY_MODE: dict[str, str] = {
    "aliyun_qwen": "voice_service_aliyun_qwen",
    "aliyun_cosyvoice": "voice_service_aliyun_cosyvoice",
}


def _build_session_from_runtime(runtime: ToolRuntime[GroupChatContext]) -> SessionContext:
    """从工具运行时构建 voice_service 会话上下文。

    语音服务的状态存储和权限判断都依赖统一的 `SessionContext`。这里仅从当前
    runtime 提取最少的用户/群信息，不在本层做额外的业务推断，避免后续状态键
    与命令侧或预热侧的逻辑不一致。

    Args:
        runtime: 当前工具运行时，要求 `runtime.context` 至少包含 `user_id`。

    Returns:
        可直接用于 voice_service 状态查询和提供方调用的会话上下文。

    Raises:
        ValueError: 当 `runtime.context.user_id` 缺失时抛出。
    """

    ctx = runtime.context
    group_id = getattr(ctx, "group_id", None)
    user_id = getattr(ctx, "user_id", None)
    if user_id is None:
        raise ValueError("runtime.context.user_id 缺失，无法使用语音工具")
    return build_session_context(user_id=int(user_id), group_id=int(group_id) if group_id is not None else None)


def _provider_for_mode(mode: str, runtime: ToolRuntime[GroupChatContext]) -> VoiceProvider:
    """根据当前模式选择具体语音提供方。

    Args:
        mode: 当前会话或本次调用指定的语音模式。
        runtime: 当前工具运行时，用于为 QQ 模式注入 bot。

    Returns:
        对应模式下可直接执行语音能力的提供方实例。
    """

    cfg = get_voice_service_plugin_config()
    if mode == "qq":
        return QQVoiceProvider(cfg, getattr(runtime.context, "bot", None))
    if mode == "aliyun_cosyvoice":
        return AliyunCosyVoiceProvider(cfg)
    return AliyunVoiceProvider(cfg)


def _get_plugin_extra() -> dict[str, Any] | None:
    """返回当前插件上下文的 `extra` 字典。

    Returns:
        当前请求共享的 `extra` 容器；若当前不在插件上下文内则返回 `None`。
    """

    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return None
    return plugin_ctx.extra


def _get_prefetched_state(session: SessionContext) -> SessionVoiceState | None:
    """读取预热阶段缓存的会话语音状态。

    只有当缓存命中的会话键与当前请求完全一致时才会复用，避免跨会话串用语音
    模式或音色状态。返回副本而不是原对象，防止工具调用过程中的临时覆盖污染
    预热结果。

    Args:
        session: 当前工具请求对应的会话上下文。

    Returns:
        命中时返回深拷贝后的会话语音状态，否则返回 `None`。
    """

    extra = _get_plugin_extra()
    if extra is None:
        return None
    if extra.get(_VOICE_SERVICE_SESSION_KEY) != session.session_key:
        return None
    prefetched_state = extra.get(_VOICE_SERVICE_STATE_KEY)
    if not isinstance(prefetched_state, SessionVoiceState):
        return None
    return cast(SessionVoiceState, prefetched_state.model_copy(deep=True))


def _get_prefetched_reply_voice(message_id: int) -> Any | None:
    """读取预热阶段缓存的回复语音样本。

    Args:
        message_id: 目标回复消息 ID。

    Returns:
        如果当前请求已经预取了同一条回复语音，则返回缓存对象；否则返回 `None`。
    """

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
    """在单次请求内至多执行一次缓存清理。

    Returns:
        无返回值。缓存清理完成后会在当前插件上下文中写入幂等标记。
    """

    extra = _get_plugin_extra()
    if extra is not None and bool(extra.get(_VOICE_SERVICE_CACHE_CLEANED_KEY)):
        return
    cfg = get_voice_service_plugin_config()
    await cleanup_expired_cache(cfg)
    if extra is not None:
        extra[_VOICE_SERVICE_CACHE_CLEANED_KEY] = True


def _register_audio_result_file(runtime: ToolRuntime[GroupChatContext], result: Any) -> str | None:
    """将本地语音合成结果注册为 GT 文件引用。

    该辅助函数只处理已经明确落地到本地磁盘的 `onebot_record` 结果，避免把 QQ
    平台原生语音消息这类不存在物理文件的结果错误纳入 GT 文件系统。注册采用
    best-effort 策略，异常由调用方记录日志后吞掉，以保持原有发送链路可用。

    Args:
        runtime: 当前工具运行时，用于提取用户、群和会话上下文。
        result: 语音合成结果对象，需要提供 `delivery`、`audio_path`、
            `provider`、`voice_name` 和 `mime_type` 等字段。

    Returns:
        注册成功时返回稳定的 `gfid:`；当前结果没有本地音频文件时返回 `None`。
    """

    if getattr(result, "delivery", None) != "onebot_record":
        return None

    audio_path = getattr(result, "audio_path", None)
    if not isinstance(audio_path, str) or not audio_path.strip():
        return None

    cfg = get_voice_service_plugin_config()
    context = runtime.context
    raw_session_id = getattr(context, "session_id", None)
    session_id = str(raw_session_id).strip() if raw_session_id is not None else ""
    return register_local_file(
        audio_path,
        kind="voice_audio",
        source_type=_VOICE_SERVICE_AUDIO_SOURCE_TYPE_BY_MODE.get(
            str(getattr(result, "provider", "")).strip(),
            "voice_service_audio",
        ),
        session_id=session_id or None,
        group_id=getattr(context, "group_id", None),
        user_id=getattr(context, "user_id", None),
        mime_type=getattr(result, "mime_type", None),
        original_name=Path(audio_path).name,
        extra={
            "voice_name": getattr(result, "voice_name", None),
            "provider": getattr(result, "provider", None),
        },
        expires_at=time.time() + float(cfg.storage.temp_ttl_sec),
    )


async def prewarm_voice_service_context(plugin_ctx: Any) -> None:
    """预热语音服务上下文所需的状态与回复语音缓存。

    该处理器会在进入 Agent 前主动完成三件事：清理过期缓存、读取当前会话的
    语音状态、以及在存在回复消息时预取被回复的语音样本。这样做可以减少真正
    调用工具时的重复 IO，并让后续工具优先复用同一请求内已准备好的上下文。

    Args:
        plugin_ctx: 当前插件上下文，要求可选提供 `runtime_context` 和 `extra`。
    """

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
    """将一次临时音色选择覆盖到会话状态副本上。

    Args:
        state: 当前会话语音状态。
        selected: 已经解析出的目标音色项。

    Returns:
        覆盖本次调用所需音色后的状态副本。
    """

    updated = state.model_copy(deep=True)
    if selected.provider == "qq":
        updated.qq.current_voice = selected.voice_id or selected.name
        return cast(SessionVoiceState, updated)
    if selected.provider == "aliyun_cosyvoice":
        updated.cosyvoice.current_voice_name = selected.name
        updated.cosyvoice.current_voice_id = selected.voice_id or selected.name
        updated.cosyvoice.current_voice_type = selected.voice_type
        updated.cosyvoice.current_target_model = selected.target_model
        return cast(SessionVoiceState, updated)
    updated.qwen.current_voice_name = selected.name
    updated.qwen.current_voice_id = selected.voice_id or selected.name
    updated.qwen.current_voice_type = selected.voice_type
    updated.qwen.current_target_model = selected.target_model
    return cast(SessionVoiceState, updated)


async def _resolve_tool_state(
    runtime: ToolRuntime[GroupChatContext],
    *,
    synth_mode: ToolMode | None = None,
    recognize_mode: RecognizeToolMode | None = None,
    voice_name: str | None = None,
) -> tuple[SessionContext, SessionVoiceState]:
    """解析当前工具调用实际应使用的会话状态。

    该函数会优先复用预热缓存，其次回落到状态存储。若调用方临时指定了模式或
    音色，则仅在本次调用的状态副本上覆盖，不会写回全局会话配置。

    Args:
        runtime: 当前工具运行时。
        synth_mode: 可选的本次语音合成模式覆盖值。
        recognize_mode: 可选的本次语音识别模式覆盖值。
        voice_name: 可选的本次临时音色名称或音色 ID。

    Returns:
        `(session, state)` 二元组，供后续提供方调用直接使用。

    Raises:
        VoiceServiceError: 当显式指定的音色在当前模式下不存在时抛出。
    """

    session = _build_session_from_runtime(runtime)
    state = _get_prefetched_state(session)
    if state is None:
        store = get_voice_state_store()
        state = await store.get(session)
    updated = cast(SessionVoiceState, state.model_copy(deep=True))
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

    该工具会根据当前会话的语音模式选择提供方，完成文本转语音后立刻向当前
    对话发送语音消息。对于会落地为本地音频文件的 `onebot_record` 结果，还会
    额外尝试把音频注册进 GT 文件系统，并把稳定 `gfid:` 一并返回给 Agent，
    以便后续工具继续复用该音频，而不必依赖裸路径。

    Args:
        text: 需要合成的文本内容，不能为空白字符串。
        runtime: 当前工具运行时，用于读取会话上下文并回发语音。
        synth_mode: 可选的本次语音合成模式覆盖值。
        voice_name: 可选的本次临时音色，不会改写会话默认音色。
        style_text: CosyVoice 模式下的自然语言风格提示。
        rate: CosyVoice 模式下的语速调节。
        pitch: CosyVoice 模式下的语调调节。
        volume: CosyVoice 模式下的音量调节。

    Returns:
        一段发送结果摘要；当本地音频已成功注册为 GT 文件时，会附带对应的
        `gfid:`。

    Raises:
        ValueError: 当文本为空、运行时缺少 bot/event，或为不支持的模式传入
            CosyVoice 专属参数时抛出。
        VoiceServiceError: 当语音合成成功但未返回可发送的本地音频文件时抛出。
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

    try:
        result.audio_file_ref = _register_audio_result_file(runtime, result)
    except Exception:
        logger.warning("voice_service synthesize: register GT file ref failed", exc_info=True)

    await bot.send(
        event=event,
        message=Message(MessageSegment.record(file=file_uri(Path(result.audio_path)))),
    )
    summary = f"已用 {_label_mode(state.synth_mode)} 模式发送语音，音色={result.voice_name}"
    if result.audio_file_ref:
        return f"{summary}，GT文件={result.audio_file_ref}"
    return summary


@tool("voice_recognize")
async def voice_recognize_tool(
    runtime: ToolRuntime[GroupChatContext],
    message_id: int | None = None,
    recognize_mode: RecognizeToolMode | None = None,
) -> str:
    """识别语音或音频消息为文本。

    Args:
        runtime: 当前工具运行时，用于读取 bot/event 等上下文。
        message_id: 可选的目标消息 ID；未提供时默认识别当前回复的语音消息。
        recognize_mode: 可选的本次识别模式覆盖值。

    Returns:
        识别得到的文本内容。

    Raises:
        ValueError: 当运行时缺少 bot，或在未提供 `message_id` 时缺少 event 时抛出。
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
    """返回用于用户提示的模式名称。

    Args:
        mode: 内部语音模式标识。

    Returns:
        可直接展示给用户的模式名称。
    """

    if mode == "qq":
        return "QQ API"
    if mode == "aliyun_cosyvoice":
        return "阿里云 CosyVoice"
    return "阿里云 Qwen"
