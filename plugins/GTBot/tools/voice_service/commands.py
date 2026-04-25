from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from nonebot import on_command, on_message
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.params import CommandArg
from nonebot.rule import Rule

from local_plugins.nonebot_plugin_gt_help import HelpArgumentSpec, HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import PermissionRole, get_permission_manager
from .audio_utils import VoiceServiceError, cleanup_expired_cache, file_uri, resolve_reply_voice
from .config import get_voice_service_plugin_config
from .models import ReplyVoiceMessage, SessionContext, VoiceItem
from .providers import AliyunCosyVoiceProvider, AliyunVoiceProvider, QQVoiceProvider
from .state import SessionVoiceState, build_session_context, get_voice_state_store


CLONE_FLOW_TIMEOUT_SEC = 30


@dataclass
class PendingCloneSession:
    session: SessionContext
    initiator_user_id: int
    reply_voice: ReplyVoiceMessage
    alias: str | None
    target_model: str | None
    step: Literal["alias", "target_model"]
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at

    def refresh(self) -> None:
        self.expires_at = datetime.now() + timedelta(seconds=CLONE_FLOW_TIMEOUT_SEC)


_pending_clone_sessions: dict[str, PendingCloneSession] = {}


def _build_session(event: MessageEvent) -> SessionContext:
    group_id = getattr(event, "group_id", None)
    return build_session_context(user_id=int(event.user_id), group_id=group_id)


async def _require_permission(rule: str, user_id: int) -> None:
    if rule == "all":
        return
    permission_manager = get_permission_manager()
    if await permission_manager.has_role(user_id, PermissionRole.ADMIN):
        return
    raise VoiceServiceError("你没有执行这条语音命令的权限")


def _normalize_synth_mode(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in {"qq", "qqapi", "qq api"}:
        return "qq"
    if text in {"aliyun", "aliyunqwen", "aliyun qwen", "阿里云", "阿里云qwen", "阿里云 qwen"}:
        return "aliyun_qwen"
    if text in {"cosyvoice", "aliyuncosyvoice", "aliyun cosyvoice", "阿里云cosyvoice", "阿里云 cosyvoice"}:
        return "aliyun_cosyvoice"
    return None


def _normalize_recognize_mode(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in {"qq", "qqapi", "qq api"}:
        return "qq"
    if text in {"aliyun", "aliyunqwen", "aliyun qwen", "阿里云", "阿里云qwen", "阿里云 qwen"}:
        return "aliyun_qwen"
    return None


def _mode_label(mode: str) -> str:
    if mode == "qq":
        return "QQ API"
    if mode == "aliyun_cosyvoice":
        return "阿里云 CosyVoice"
    return "阿里云 Qwen"


def _extract_text_arg(args: Message) -> str:
    return args.extract_plain_text().strip()


async def _get_state(event: MessageEvent) -> tuple[SessionContext, SessionVoiceState]:
    session = _build_session(event)
    store = get_voice_state_store()
    state = await store.get(session)
    return session, state


def _provider_for_mode(mode: str, bot: Bot | None = None):
    cfg = get_voice_service_plugin_config()
    if mode == "qq":
        return QQVoiceProvider(cfg, bot)
    if mode == "aliyun_cosyvoice":
        return AliyunCosyVoiceProvider(cfg)
    return AliyunVoiceProvider(cfg)


def _normalize_clone_cancel(text: str) -> bool:
    return text.strip() == "$"


def _cleanup_pending_clone_sessions() -> None:
    expired = [key for key, item in _pending_clone_sessions.items() if item.is_expired()]
    for key in expired:
        _pending_clone_sessions.pop(key, None)


def _store_pending_clone(item: PendingCloneSession) -> None:
    _cleanup_pending_clone_sessions()
    item.refresh()
    _pending_clone_sessions[item.session.session_key] = item


def _pop_pending_clone(session: SessionContext) -> PendingCloneSession | None:
    return _pending_clone_sessions.pop(session.session_key, None)


def _get_pending_clone(session: SessionContext) -> PendingCloneSession | None:
    return _pending_clone_sessions.get(session.session_key)


async def _has_pending_clone(_bot: Bot, event: MessageEvent) -> bool:
    session = _build_session(event)
    pending = _get_pending_clone(session)
    return pending is not None and pending.initiator_user_id == int(event.user_id)


def _clone_alias_prompt(mode: str) -> str:
    if mode == "aliyun_cosyvoice":
        return "请发送 CosyVoice 克隆音色名称，30 秒内有效，发送 $ 取消。"
    return "请发送 Qwen 克隆音色别名，30 秒内有效，发送 $ 取消。"


def _clone_target_model_prompt(mode: str, default_model: str) -> str:
    prefix = "CosyVoice" if mode == "aliyun_cosyvoice" else "Qwen"
    return (
        f"请发送 {prefix} 目标合成模型，30 秒内有效，发送 $ 取消。\n"
        f"发送 默认 可使用当前配置值: {default_model}"
    )


def _normalize_clone_target_model(raw: str, default_model: str) -> str:
    text = raw.strip()
    if not text or text.lower() in {"default", "默认"}:
        return default_model
    return text


def _current_engine_state(state: SessionVoiceState):
    return state.cosyvoice if state.synth_mode == "aliyun_cosyvoice" else state.qwen


def _format_voice_list(state: SessionVoiceState, voices: list[VoiceItem]) -> str:
    mode = state.synth_mode
    lines: list[str] = [
        f"合成模式: {_mode_label(state.synth_mode)}",
        f"识别模式: {_mode_label(state.recognize_mode)}",
    ]
    if mode == "qq":
        current = state.qq.current_voice or "(未设置)"
        lines.append(f"当前合成音色: {current}")
        if not voices:
            lines.append("可用音色: (空)")
            return "\n".join(lines)
        lines.append("可用音色:")
        for item in voices:
            prefix = "* " if state.qq.current_voice == (item.voice_id or item.name) else "- "
            lines.append(f"{prefix}{item.name}")
        return "\n".join(lines)

    engine_state = _current_engine_state(state)
    current = engine_state.current_voice_name or "(未设置)"
    lines.append(f"当前合成音色: {current}")
    builtin = [item for item in voices if item.voice_type == "builtin"]
    custom = [item for item in voices if item.voice_type == "custom"]
    lines.append("[系统音色]")
    if builtin:
        for item in builtin:
            prefix = "* " if engine_state.current_voice_id == (item.voice_id or item.name) else "- "
            extra_parts: list[str] = []
            if item.target_model:
                extra_parts.append(item.target_model)
            suffix = f" ({' | '.join(extra_parts)})" if extra_parts else ""
            lines.append(f"{prefix}{item.name}{suffix}")
    else:
        lines.append("- (空)")

    lines.append("[自定义音色]")
    if custom:
        for item in custom:
            prefix = "* " if engine_state.current_voice_id == (item.voice_id or item.name) else "- "
            custom_extra_parts: list[str] = []
            if item.voice_id and item.voice_id != item.name:
                custom_extra_parts.append(item.voice_id)
            if item.target_model:
                custom_extra_parts.append(item.target_model)
            suffix = f" ({' | '.join(custom_extra_parts)})" if custom_extra_parts else ""
            lines.append(f"{prefix}{item.name}{suffix}")
    else:
        lines.append("- (空)")
    return "\n".join(lines)


VoiceModeCommand = on_command("语音模式", priority=4, block=True)
VoiceSynthesizeModeCommand = on_command("语音合成模式", priority=4, block=True)
VoiceRecognizeModeCommand = on_command("语音识别模式", priority=4, block=True)
VoiceListCommand = on_command("语音音色列表", priority=4, block=True)
VoiceSetCommand = on_command("语音设置音色", priority=4, block=True)
VoiceSynthesizeCommand = on_command("语音合成", priority=4, block=True)
VoiceRecognizeCommand = on_command("语音识别", priority=4, block=True)
VoiceCloneCommand = on_command("语音克隆音色", priority=4, block=True)
VoiceCloneFlowCommand = on_message(rule=Rule(_has_pending_clone), priority=4, block=True)


def _help_role_from_rule(rule: str) -> PermissionRole:
    """将语音插件配置中的权限规则映射为帮助系统权限。

    Args:
        rule: 语音插件配置里的权限规则，当前支持 `all` 与 `admin`。

    Returns:
        PermissionRole: 帮助系统使用的最低权限等级。
    """
    return PermissionRole.USER if rule == "all" else PermissionRole.ADMIN


def _register_voice_help_items() -> None:
    """注册语音服务插件的全部命令帮助信息。"""
    cfg = get_voice_service_plugin_config()

    register_help(
        HelpCommandSpec(
            name="语音模式",
            category="语音服务",
            summary="同时切换语音合成和识别模式。",
            description="将当前会话的语音合成模式和识别模式一起切换为 QQ、阿里云 Qwen 或阿里云 CosyVoice。",
            arguments=(
                HelpArgumentSpec(
                    name="<模式>",
                    description="目标语音模式。",
                    value_hint="QQ / 阿里云Qwen / 阿里云CosyVoice",
                    example="阿里云CosyVoice",
                ),
            ),
            examples=(
                "/语音模式 QQ",
                "/语音模式 阿里云Qwen",
                "/语音模式 阿里云CosyVoice",
            ),
            required_role=_help_role_from_rule(cfg.permissions.manage_mode),
            audience="群聊和私聊",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音合成模式",
            category="语音服务",
            summary="只切换当前会话的语音合成模式。",
            description="切换当前会话的语音合成引擎，不影响语音识别模式。",
            arguments=(
                HelpArgumentSpec(
                    name="<模式>",
                    description="目标合成模式。",
                    value_hint="QQ / 阿里云Qwen / 阿里云CosyVoice",
                    example="QQ",
                ),
            ),
            examples=(
                "/语音合成模式 QQ",
                "/语音合成模式 阿里云CosyVoice",
            ),
            required_role=_help_role_from_rule(cfg.permissions.manage_mode),
            audience="群聊和私聊",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音识别模式",
            category="语音服务",
            summary="只切换当前会话的语音识别模式。",
            description="切换当前会话的语音识别引擎，目前支持 QQ 与阿里云 Qwen。",
            arguments=(
                HelpArgumentSpec(
                    name="<模式>",
                    description="目标识别模式。",
                    value_hint="QQ / 阿里云Qwen",
                    example="阿里云Qwen",
                ),
            ),
            examples=(
                "/语音识别模式 QQ",
                "/语音识别模式 阿里云Qwen",
            ),
            required_role=_help_role_from_rule(cfg.permissions.manage_mode),
            audience="群聊和私聊",
            sort_key=30,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音音色列表",
            category="语音服务",
            summary="查看当前模式下可用的语音音色列表。",
            description="列出当前会话可用的系统音色、自定义音色，以及当前正在使用的音色。",
            examples=(
                "/语音音色列表",
            ),
            required_role=_help_role_from_rule(cfg.permissions.set_voice),
            audience="群聊和私聊",
            sort_key=40,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音设置音色",
            category="语音服务",
            summary="设置当前会话的语音合成音色。",
            description="根据音色名、显示名或音色 ID 选择当前合成音色。",
            arguments=(
                HelpArgumentSpec(
                    name="<音色名>",
                    description="要切换到的目标音色名称、显示名或 ID。",
                    value_hint="音色名称",
                    example="xiaoyun",
                ),
            ),
            examples=(
                "/语音设置音色 xiaoyun",
            ),
            required_role=_help_role_from_rule(cfg.permissions.set_voice),
            audience="群聊和私聊",
            sort_key=50,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音合成",
            category="语音服务",
            summary="将文本转换为语音并发送。",
            description="使用当前会话配置的合成模式和音色，将输入文本转换成语音消息。",
            arguments=(
                HelpArgumentSpec(
                    name="<文本>",
                    description="要转换成语音的文本内容。",
                    value_hint="任意文本",
                    example="今天的会议十点开始",
                ),
            ),
            examples=(
                "/语音合成 今天的会议十点开始",
            ),
            required_role=_help_role_from_rule(cfg.permissions.synthesize),
            audience="群聊和私聊",
            sort_key=60,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音识别",
            category="语音服务",
            summary="识别回复语音中的文字内容。",
            description="回复一条语音消息后执行该命令，系统会识别并返回语音中的文本。",
            examples=(
                "/语音识别",
            ),
            required_role=_help_role_from_rule(cfg.permissions.recognize),
            audience="群聊和私聊，需回复语音消息",
            sort_key=70,
        )
    )
    register_help(
        HelpCommandSpec(
            name="语音克隆音色",
            category="语音服务",
            summary="通过回复语音样本创建自定义音色。",
            description="回复一条语音消息后执行该命令，系统会进入多步交互流程，收集音色名称和目标模型并克隆音色。",
            arguments=(
                HelpArgumentSpec(
                    name="[音色别名]",
                    description="可选的自定义音色名称；留空时系统会继续追问。",
                    required=False,
                    value_hint="自定义名称",
                    example="会议播报",
                ),
            ),
            examples=(
                "/语音克隆音色",
                "/语音克隆音色 会议播报",
            ),
            required_role=_help_role_from_rule(cfg.permissions.clone_voice),
            audience="群聊和私聊，需回复语音消息",
            sort_key=80,
        )
    )


_register_voice_help_items()


@VoiceModeCommand.handle()
async def _handle_voice_mode(event: MessageEvent, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.manage_mode, int(event.user_id))
        target_mode = _normalize_synth_mode(_extract_text_arg(args))
        if target_mode is None:
            await VoiceModeCommand.finish("用法: /语音模式 QQ 或 /语音模式 阿里云Qwen 或 /语音模式 阿里云CosyVoice")
            return

        session, _state = await _get_state(event)
        store = get_voice_state_store()
        if target_mode == "aliyun_cosyvoice":
            synth_state = await store.set_synth_mode(session, "aliyun_cosyvoice")
            state = await store.set_recognize_mode(session, "aliyun_qwen")
            await VoiceModeCommand.finish(
                "\n".join(
                    [
                        "已将合成模式切换到 阿里云 CosyVoice",
                        "已将识别模式切换到 阿里云 Qwen",
                        f"当前合成模式: {_mode_label(synth_state.synth_mode)}",
                        f"当前识别模式: {_mode_label(state.recognize_mode)}",
                    ]
                )
            )

        state = await store.set_mode(session, target_mode)  # type: ignore[arg-type]
        await VoiceModeCommand.finish(
            "\n".join(
                [
                    f"已将合成模式和识别模式都切换到 {_mode_label(target_mode)}",
                    f"当前合成模式: {_mode_label(state.synth_mode)}",
                    f"当前识别模式: {_mode_label(state.recognize_mode)}",
                ]
            )
        )
    except VoiceServiceError as exc:
        await VoiceModeCommand.finish(str(exc))


@VoiceSynthesizeModeCommand.handle()
async def _handle_voice_synthesize_mode(event: MessageEvent, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.manage_mode, int(event.user_id))
        target_mode = _normalize_synth_mode(_extract_text_arg(args))
        if target_mode is None:
            await VoiceSynthesizeModeCommand.finish(
                "用法: /语音合成模式 QQ 或 /语音合成模式 阿里云Qwen 或 /语音合成模式 阿里云CosyVoice"
            )
            return

        session, _state = await _get_state(event)
        state = await get_voice_state_store().set_synth_mode(session, target_mode)  # type: ignore[arg-type]
        await VoiceSynthesizeModeCommand.finish(f"已切换合成模式到 {_mode_label(state.synth_mode)}")
    except VoiceServiceError as exc:
        await VoiceSynthesizeModeCommand.finish(str(exc))


@VoiceRecognizeModeCommand.handle()
async def _handle_voice_recognize_mode(event: MessageEvent, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.manage_mode, int(event.user_id))
        target_mode = _normalize_recognize_mode(_extract_text_arg(args))
        if target_mode is None:
            await VoiceRecognizeModeCommand.finish("用法: /语音识别模式 QQ 或 /语音识别模式 阿里云Qwen")
            return

        session, _state = await _get_state(event)
        state = await get_voice_state_store().set_recognize_mode(session, target_mode)  # type: ignore[arg-type]
        await VoiceRecognizeModeCommand.finish(f"已切换识别模式到 {_mode_label(state.recognize_mode)}")
    except VoiceServiceError as exc:
        await VoiceRecognizeModeCommand.finish(str(exc))


@VoiceListCommand.handle()
async def _handle_voice_list(event: MessageEvent, bot: Bot) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.set_voice, int(event.user_id))
        await cleanup_expired_cache(cfg)
        session, state = await _get_state(event)
        provider = _provider_for_mode(state.synth_mode, bot)
        voices = await provider.list_voices(session, state)
        await VoiceListCommand.finish(_format_voice_list(state, voices))
    except VoiceServiceError as exc:
        await VoiceListCommand.finish(str(exc))


@VoiceSetCommand.handle()
async def _handle_voice_set(event: MessageEvent, bot: Bot, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.set_voice, int(event.user_id))
        voice_name = _extract_text_arg(args)
        if not voice_name:
            await VoiceSetCommand.finish("用法: /语音设置音色 <音色名>")

        session, state = await _get_state(event)
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
            await VoiceSetCommand.finish(f"未找到音色: {voice_name}")
            return

        store = get_voice_state_store()
        if selected.provider == "qq":
            await store.set_qq_voice(session, selected.voice_id or selected.name)
        elif selected.provider == "aliyun_cosyvoice":
            await store.set_cosyvoice_voice(
                session,
                voice_name=selected.name,
                voice_id=selected.voice_id or selected.name,
                voice_type=selected.voice_type,
                target_model=selected.target_model,
            )
        else:
            await store.set_qwen_voice(
                session,
                voice_name=selected.name,
                voice_id=selected.voice_id or selected.name,
                voice_type=selected.voice_type,
                target_model=selected.target_model,
            )
        await VoiceSetCommand.finish(f"已设置当前合成音色: {selected.name}")
    except VoiceServiceError as exc:
        await VoiceSetCommand.finish(str(exc))


@VoiceSynthesizeCommand.handle()
async def _handle_voice_synthesize(event: MessageEvent, bot: Bot, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.synthesize, int(event.user_id))
        await cleanup_expired_cache(cfg)

        text = _extract_text_arg(args)
        if not text:
            await VoiceSynthesizeCommand.finish("用法: /语音合成 <文本>")

        session, state = await _get_state(event)
        provider = _provider_for_mode(state.synth_mode, bot)
        result = await provider.synthesize(session, state, text)

        if result.delivery == "qq_group_ai":
            await VoiceSynthesizeCommand.finish()

        if not result.audio_path:
            await VoiceSynthesizeCommand.finish("语音合成完成，但未返回音频文件")

        await bot.send(
            event=event,
            message=Message(MessageSegment.record(file=file_uri(Path(result.audio_path)))),
        )
        await VoiceSynthesizeCommand.finish()
    except VoiceServiceError as exc:
        await VoiceSynthesizeCommand.finish(str(exc))


@VoiceRecognizeCommand.handle()
async def _handle_voice_recognize(event: MessageEvent, bot: Bot) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.recognize, int(event.user_id))
        await cleanup_expired_cache(cfg)
        session, state = await _get_state(event)
        reply_voice = await resolve_reply_voice(bot, event, cfg)
        provider = _provider_for_mode(state.recognize_mode, bot)
        result = await provider.recognize(session, state, reply_voice)
        await VoiceRecognizeCommand.finish(result.text)
    except VoiceServiceError as exc:
        await VoiceRecognizeCommand.finish(str(exc))


@VoiceCloneCommand.handle()
async def _handle_voice_clone(event: MessageEvent, bot: Bot, args: Message = CommandArg()) -> None:
    cfg = get_voice_service_plugin_config()
    try:
        await _require_permission(cfg.permissions.clone_voice, int(event.user_id))
        await cleanup_expired_cache(cfg)

        session, state = await _get_state(event)
        if state.synth_mode == "qq":
            await VoiceCloneCommand.finish("语音克隆音色仅支持阿里云合成模式")

        reply_voice = await resolve_reply_voice(bot, event, cfg)
        alias = _extract_text_arg(args) or None
        pending = PendingCloneSession(
            session=session,
            initiator_user_id=int(event.user_id),
            reply_voice=reply_voice,
            alias=alias,
            target_model=None,
            step="target_model" if alias else "alias",
            expires_at=datetime.now(),
        )
        _store_pending_clone(pending)

        if alias:
            default_model = (
                cfg.aliyun.cosyvoice.tts_custom_model
                if state.synth_mode == "aliyun_cosyvoice"
                else cfg.aliyun.qwen.tts_custom_model
            )
            await VoiceCloneCommand.finish(_clone_target_model_prompt(state.synth_mode, default_model))
        await VoiceCloneCommand.finish(_clone_alias_prompt(state.synth_mode))
    except VoiceServiceError as exc:
        await VoiceCloneCommand.finish(str(exc))


@VoiceCloneFlowCommand.handle()
async def _handle_voice_clone_flow(event: MessageEvent) -> None:
    cfg = get_voice_service_plugin_config()
    session = _build_session(event)
    pending = _get_pending_clone(session)
    if pending is None:
        return

    if pending.is_expired():
        _pop_pending_clone(session)
        await VoiceCloneFlowCommand.finish("语音克隆已超时取消。")

    text = event.get_plaintext().strip()
    if _normalize_clone_cancel(text):
        _pop_pending_clone(session)
        await VoiceCloneFlowCommand.finish("已取消本次语音克隆。")

    try:
        state = await get_voice_state_store().get(session)
        if pending.step == "alias":
            if not text:
                pending.refresh()
                await VoiceCloneFlowCommand.finish(_clone_alias_prompt(state.synth_mode))
            pending.alias = text
            pending.step = "target_model"
            _store_pending_clone(pending)
            default_model = (
                cfg.aliyun.cosyvoice.tts_custom_model
                if state.synth_mode == "aliyun_cosyvoice"
                else cfg.aliyun.qwen.tts_custom_model
            )
            await VoiceCloneFlowCommand.finish(_clone_target_model_prompt(state.synth_mode, default_model))

        alias = (pending.alias or "").strip()
        if not alias:
            pending.step = "alias"
            _store_pending_clone(pending)
            await VoiceCloneFlowCommand.finish(_clone_alias_prompt(state.synth_mode))

        default_model = (
            cfg.aliyun.cosyvoice.tts_custom_model
            if state.synth_mode == "aliyun_cosyvoice"
            else cfg.aliyun.qwen.tts_custom_model
        )
        target_model = _normalize_clone_target_model(text, default_model)
        provider = _provider_for_mode(state.synth_mode)
        result = await provider.clone_voice(
            session,
            state,
            pending.reply_voice,
            alias,
            target_model=target_model,
        )

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
        _pop_pending_clone(session)
        await VoiceCloneFlowCommand.finish(f"克隆成功，当前 {_mode_label(state.synth_mode)} 合成音色已切换为: {result.voice_name}")
    except VoiceServiceError as exc:
        pending.refresh()
        _store_pending_clone(pending)
        await VoiceCloneFlowCommand.finish(str(exc))
