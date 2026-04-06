import asyncio
import json
from time import time
from collections.abc import Sequence
from urllib.parse import urlparse
from langchain_core.tools.base import BaseTool
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List, Callable, Any, Union, Literal, TypeAlias
from typing import cast
from uuid import uuid4

from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from pathlib import Path
from asyncio import Semaphore, Queue, Lock, TimeoutError as AsyncTimeoutError, wait_for
from dataclasses import dataclass, field
from asyncio import sleep, create_task, Event

from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph

from ...DBmodel import GroupMessage
from ..message import GroupMessageManager, get_message_manager
from ...ConfigManager import total_config, ProcessedConfiguration
from ...llm_provider import build_chat_model
from ..shared import fun as Fun
from .. import cache as CacheManager
# 暂时禁用 SQLAlchemy 画像系统，改用 LongMemory/Qdrant 版本
# from .UserProfileManager import ProfileManager, get_profile_manager
from .context import GroupChatContext
from plugins.GTBot.services.plugin_system.facade import build_plugin_bundle, build_plugin_context
from plugins.GTBot.services.plugin_system.runtime import plugin_context_scope, set_response_status
from plugins.GTBot.services.plugin_system.types import (
    PluginBundle,
    PluginContext,
    PreAgentMessageAppenderBinding,
    PreAgentMessageInjectorBinding,
    PreAgentProcessorBinding,
    ResponseStatus,
)
from ...constants import (
    DEFAULT_BOT_NAME_PLACEHOLDER,
    NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES,
    SEND_MESSAGE_BLOCK_PATTERN,
    SEND_OUTPUT_BLOCK_PATTERN,
    SUPPORTED_CQ_CODES,
    NOTE_TAG_PATTERN,
    THINKING_TAG_PATTERN,
)
from .group_queue import GroupMessageQueueManager, MessageTask, group_message_queue_manager
from .private_queue import PrivateMessageTask, private_message_queue_manager
from .queue_payload import QueueMessageContent as PreparedQueueMessageContent, prepare_queue_messages
from .continuation import get_continuation_manager
from ..access import ChatAccessScope, get_chat_access_manager
from .internal_tools import (
    delete_message_tool,
    emoji_reaction_tool,
    poke_user_tool,
    send_group_message_tool,
    send_private_message_tool,
    send_like_tool,
)
from ..message.segments import deserialize_message_segments, serialize_message_segments

GroupChatContext.model_rebuild()

config = total_config.processed_configuration.current_config_group

ChatType: TypeAlias = Literal["group", "private"]
ChatSource: TypeAlias = Literal["passive", "proactive"]
ChatTriggerMode: TypeAlias = Literal["group_at", "private", "group_keyword", "group_auto", "group_continuation", "unknown"]


@dataclass(slots=True)
class ChatSession:
    session_id: str
    chat_type: ChatType
    group_id: int | None
    peer_user_id: int


@dataclass(slots=True)
class ChatTurn:
    session: ChatSession
    sender_user_id: int
    sender_name: str = ""
    anchor_message_id: int | None = None
    input_text: str = ""
    trigger_mode: ChatTriggerMode = "unknown"
    trigger_meta: dict[str, Any] = field(default_factory=dict)
    source: ChatSource = "passive"
    event: MessageEvent | None = None
    message: Message | None = None


QueuedChatMessage = PreparedQueueMessageContent


class ChatTransport:
    """聊天发送器基类。"""

    def __init__(
        self,
        *,
        bot: Bot,
        message_manager: GroupMessageManager,
        cache: CacheManager.UserCacheManager,
        turn: ChatTurn,
    ) -> None:
        """初始化发送器运行所需的依赖。"""
        self.bot = bot
        self.message_manager = message_manager
        self.cache = cache
        self.turn = turn

    @property
    def session(self) -> ChatSession:
        """返回当前轮次绑定的会话信息。"""
        return self.turn.session

    async def send_messages(self, messages: Sequence[QueuedChatMessage], interval: float = 0.2) -> None:
        """发送多条消息。"""
        raise NotImplementedError

    async def send_feedback(self, text: str, *, at_sender: bool = False) -> None:
        """发送反馈文本。"""
        raise NotImplementedError

    async def send_timeout(self, timeout_sec: float) -> None:
        """发送超时提示。"""
        await self.send_feedback(
            f"请求处理超时（{timeout_sec}秒），请稍后重试",
            at_sender=True,
        )

    async def send_error(self, error_detail: str) -> None:
        """发送错误提示。"""
        await self.send_feedback(
            f"处理请求时发生错误，请联系管理员或检查API状态。详情: {error_detail}",
            at_sender=True,
        )

    async def handle_processing_emoji(self) -> None:
        """在进入处理阶段时追加状态反馈。默认不执行任何操作。"""
        return

    async def handle_completion_emoji(self) -> None:
        """在处理完成后追加状态反馈。默认不执行任何操作。"""
        return

    async def handle_rejection_emoji(self) -> None:
        """在请求被拒绝时追加状态反馈。默认不执行任何操作。"""
        return

    async def handle_timeout_emoji(self) -> None:
        """在处理超时时追加状态反馈。默认不执行任何操作。"""
        return

    async def _record_outgoing_message(
        self,
        message_id: int,
        content: str,
        message: Message | None = None,
    ) -> None:
        """将机器人发出的消息补记入统一消息表。

        Args:
            message_id: 平台返回的消息 ID。
            content: 用于兼容旧逻辑的文本内容。
            message: 实际发送的 OneBot 消息对象，用于原样保存消息段结构。
        """
        try:
            await self.message_manager.add_chat_message(
                message_id=int(message_id),
                sender_user_id=int(self.bot.self_id),
                sender_name="",
                content=str(content),
                serialized_segments=serialize_message_segments(message),
                group_id=self.session.group_id,
                session_id=self.session.session_id,
                peer_user_id=self.session.peer_user_id,
            )
        except Exception as exc:
            logger.warning(f"记录机器人出站消息失败: {exc}")


class GroupChatTransport(ChatTransport):
    """群聊会话使用的发送器。"""

    async def send_messages(self, messages: Sequence[QueuedChatMessage], interval: float = 0.2) -> None:
        """通过群聊消息队列顺序发送多条消息。"""
        if self.session.group_id is None:
            return
        await _enqueue_group_messages(
            bot=self.bot,
            group_id=self.session.group_id,
            message_manager=self.message_manager,
            cache=self.cache,
            messages=messages,
            interval=interval,
        )

    async def send_feedback(self, text: str, *, at_sender: bool = False) -> None:
        """直接向当前群会话发送反馈消息。"""
        if self.session.group_id is None:
            return
        processed_message: Message = await Fun.text_to_message(
            text,
            whitelist=SUPPORTED_CQ_CODES,
        )
        output = Message()
        if at_sender and self.turn.sender_user_id > 0:
            output += MessageSegment.at(self.turn.sender_user_id)
            output += MessageSegment.text(" ")
        output += processed_message
        result = await self.bot.send_group_msg(group_id=self.session.group_id, message=output)
        await self._record_outgoing_message(int(result["message_id"]), str(output), output)

    async def handle_processing_emoji(self) -> None:
        """为当前群消息添加“处理中”表情贴。"""
        if self.turn.anchor_message_id is None or self.session.group_id is None:
            return
        await handle_processing_emoji(self.bot, self.turn.anchor_message_id, self.session.group_id)

    async def handle_completion_emoji(self) -> None:
        """为当前群消息添加“完成”表情贴。"""
        if self.turn.anchor_message_id is None or self.session.group_id is None:
            return
        await handle_completion_emoji(self.bot, self.turn.anchor_message_id, self.session.group_id)

    async def handle_rejection_emoji(self) -> None:
        """为当前群消息添加“拒绝”表情贴。"""
        if self.turn.anchor_message_id is None or self.session.group_id is None:
            return
        await handle_request_rejection(self.bot, self.turn.anchor_message_id, self.session.group_id)

    async def handle_timeout_emoji(self) -> None:
        """为当前群消息添加“超时”表情贴。"""
        if self.turn.anchor_message_id is None or self.session.group_id is None:
            return
        await handle_timeout_emoji(self.bot, self.turn.anchor_message_id, self.session.group_id)


class PrivateChatTransport(ChatTransport):
    """私聊会话使用的发送器。"""

    async def send_messages(self, messages: Sequence[QueuedChatMessage], interval: float = 0.2) -> None:
        """通过私聊消息队列发送多条消息。"""
        if not messages:
            return
        prepared_messages = await prepare_queue_messages(
            messages,
            scope=f"session {self.session.session_id}",
        )
        task = PrivateMessageTask(
            messages=prepared_messages,
            user_id=self.session.peer_user_id,
            interval=interval,
            session_id=self.session.session_id,
        )
        await private_message_queue_manager.enqueue(
            task,
            bot=self.bot,
            message_manager=self.message_manager,
            cache=self.cache,
        )

    async def handle_processing_emoji(self) -> None:
        """私聊场景暂不支持消息表情贴，因此直接跳过。"""
        return

    async def handle_completion_emoji(self) -> None:
        """私聊场景暂不支持消息表情贴，因此直接跳过。"""
        return

    async def handle_rejection_emoji(self) -> None:
        """私聊场景暂不支持消息表情贴，因此直接跳过。"""
        return

    async def handle_timeout_emoji(self) -> None:
        """私聊场景暂不支持消息表情贴，因此直接跳过。"""
        return

    async def send_feedback(self, text: str, *, at_sender: bool = False) -> None:
        """通过私聊队列发送一条反馈消息。"""
        prepared_messages = await prepare_queue_messages(
            [text],
            scope=f"session {self.session.session_id}",
        )
        task = PrivateMessageTask(
            messages=prepared_messages,
            user_id=self.session.peer_user_id,
            interval=0.0,
            session_id=self.session.session_id,
        )
        await private_message_queue_manager.enqueue(
            task,
            bot=self.bot,
            message_manager=self.message_manager,
            cache=self.cache,
        )

def _build_group_session(group_id: int) -> ChatSession:
    return ChatSession(
        session_id=f"group:{int(group_id)}",
        chat_type="group",
        group_id=int(group_id),
        peer_user_id=int(group_id),
    )


def _build_private_session(user_id: int) -> ChatSession:
    return ChatSession(
        session_id=f"private:{int(user_id)}",
        chat_type="private",
        group_id=None,
        peer_user_id=int(user_id),
    )


def _resolve_chat_access_target(session: ChatSession) -> tuple[ChatAccessScope, int] | None:
    """根据会话信息解析准入控制所需的范围和目标 ID。

    Args:
        session: 当前聊天会话描述。

    Returns:
        tuple[ChatAccessScope, int] | None:
            返回 `(scope, target_id)`，其中 `target_id` 为群号或私聊用户号。
            当会话信息不足以判断准入目标时返回 `None`。
    """
    if session.chat_type == "group":
        if session.group_id is None:
            return None
        return ChatAccessScope.GROUP, int(session.group_id)

    if session.peer_user_id <= 0:
        return None
    return ChatAccessScope.PRIVATE, int(session.peer_user_id)


def _resolve_proactive_trigger_mode(session: ChatSession) -> ChatTriggerMode:
    """为主动发起的会话推断默认触发模式。"""

    if session.chat_type == "group":
        return "group_auto"
    if session.chat_type == "private":
        return "private"
    return "unknown"


def _should_open_continuation_window(trigger_mode: ChatTriggerMode) -> bool:
    continuation_cfg = config.chat_model.continuation
    if not bool(getattr(continuation_cfg, "enabled", False)):
        return False
    if not str(getattr(continuation_cfg, "analyzer_model_id", "") or "").strip():
        return False

    scope = str(getattr(continuation_cfg, "scope", "all") or "all").strip()
    if scope == "all":
        return trigger_mode in {"group_at", "group_keyword", "group_auto", "group_continuation"}
    if scope == "explicit_only":
        return trigger_mode == "group_at"
    if scope == "exclude_auto":
        return trigger_mode in {"group_at", "group_keyword", "group_continuation"}
    return False


async def _find_latest_bot_message_id_for_session(
    *,
    session_id: str,
    self_user_id: int,
    msg_mg: GroupMessageManager,
) -> int | None:
    try:
        recent_messages = await msg_mg.get_recent_messages(limit=20, session_id=session_id)
    except Exception:
        logger.debug(
            "failed to lookup latest bot message for continuation window: session=%s",
            session_id,
            exc_info=True,
        )
        return None

    for item in recent_messages:
        if int(getattr(item, "user_id", 0) or 0) == int(self_user_id):
            return int(getattr(item, "message_id", 0) or 0) or None
    return None


def _resolve_continuation_window_start_time(
    *,
    turn: ChatTurn,
    relevant_messages: list[GroupMessage],
    self_user_id: int,
) -> float | None:
    inherited_start_time = turn.trigger_meta.get("continuation_history_started_at")
    if inherited_start_time is not None:
        try:
            return float(inherited_start_time)
        except Exception:
            pass

    for item in reversed(relevant_messages):
        message_user_id = int(getattr(item, "user_id", 0) or 0)
        if message_user_id == int(self_user_id):
            continue
        message_time = getattr(item, "send_time", None)
        if message_time is None:
            continue
        try:
            return float(message_time)
        except Exception:
            continue

    if turn.event is not None:
        event_time = getattr(turn.event, "time", None)
        if event_time is not None:
            try:
                return float(event_time)
            except Exception:
                return None
    return None


# ============================================================================
# 消息发送队列管理器（生产者-消费者模型）
# ============================================================================



# 创建model缓存
agent_cache_info: dict = {
    "model": None,
    "provider_type": None,
    "model_id": None,
    "base_url": None,
    "api_key": None,
    "extra_body": None,
    "streaming": None,
}


def _parse_bool_runtime_parameter(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _parse_streaming_settings(raw_parameters: dict[str, Any] | None) -> tuple[dict[str, Any], bool, int, float, bool]:
    """解析并剥离流式相关参数。

    约定从配置的 `chat_model.parameters` 中读取以下键：
    - `streaming` / `stream`: 是否启用流式（默认 False）
    - `stream_chunk_chars`: 每次向群里发送的最小字符数（默认 160）
    - `stream_flush_interval_sec`: 即使不足 chunk 也会刷新的最大间隔（秒，默认 0.8）

    这些键不会被传给 OpenAI API（避免未知参数）。其余参数会作为 `model_kwargs` 透传给模型。

    Args:
        raw_parameters: 原始参数字典（可能为 None）。

    Returns:
        tuple[dict[str, Any], bool, int, float, bool]:
            (
                clean_model_kwargs,
                streaming_enabled,
                stream_chunk_chars,
                stream_flush_interval_sec,
                process_tool_call_deltas,
            )
    """
    parameters: dict[str, Any] = dict(raw_parameters or {})

    streaming_enabled = bool(parameters.pop("streaming", False) or parameters.pop("stream", False))

    stream_chunk_chars_raw = parameters.pop("stream_chunk_chars", 160)
    try:
        stream_chunk_chars = max(20, int(stream_chunk_chars_raw))
    except Exception:
        stream_chunk_chars = 160

    stream_flush_interval_raw = parameters.pop("stream_flush_interval_sec", 0.8)
    try:
        stream_flush_interval_sec = max(0.1, float(stream_flush_interval_raw))
    except Exception:
        stream_flush_interval_sec = 0.8

    process_tool_call_deltas = _parse_bool_runtime_parameter(
        parameters.pop("process_tool_call_deltas", True),
        default=True,
    )

    return (
        parameters,
        streaming_enabled,
        stream_chunk_chars,
        stream_flush_interval_sec,
        process_tool_call_deltas,
    )


def _is_dashscope_openai_compatible_base_url(base_url: str | None) -> bool:
    """判断当前地址是否为阿里百炼 OpenAI 兼容接口。"""

    raw_value = str(base_url or "").strip()
    if not raw_value:
        return False

    try:
        parsed = urlparse(raw_value)
    except Exception:
        return False

    hostname = (parsed.hostname or "").strip().lower()
    path = (parsed.path or "").strip().lower().rstrip("/")

    if hostname not in {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }:
        return False

    return "compatible-mode" in path


def _should_force_non_streaming_tool_agent(
    *,
    provider_type: str | None,
    base_url: str | None,
    streaming_enabled: bool,
    process_tool_call_deltas: bool,
) -> bool:
    """阿里百炼相关接入在工具调用流式增量上不稳定时，退回非流式执行。"""

    if not streaming_enabled:
        return False
    if not process_tool_call_deltas:
        return False

    normalized_provider_type = str(provider_type or "").strip().lower()
    if normalized_provider_type == "dashscope":
        return True
    if normalized_provider_type != "openai_compatible":
        return False
    return _is_dashscope_openai_compatible_base_url(base_url)


async def _enqueue_group_messages(
    *,
    bot: Bot,
    group_id: int,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    messages: Sequence[QueuedChatMessage],
    interval: float,
) -> None:
    """预处理群聊消息后再交给发送队列。

    Args:
        bot: 当前 OneBot 机器人实例。
        group_id: 目标群号。
        message_manager: 消息管理器。
        cache: 用户缓存管理器。
        messages: 待入队的消息列表，允许文本、消息段或完整消息对象。
        interval: 多条消息之间的发送间隔秒数。
    """
    if not messages:
        return

    prepared_messages = await prepare_queue_messages(
        messages,
        scope=f"群组 {group_id}",
    )
    task = MessageTask(messages=prepared_messages, group_id=group_id, interval=interval)
    await group_message_queue_manager.enqueue(
        task,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
    )


class DirectAssistantOutputMiddleware(AgentMiddleware[AgentState, GroupChatContext]):
    """Route direct `<msg>...</msg>` assistant output through the active transport."""

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        try:
            context = getattr(runtime, "context", None)
            if context is None:
                return None

            transport: ChatTransport | None = getattr(context, "transport", None)
            streaming_enabled: bool = bool(getattr(context, "streaming_enabled", False))
            if transport is None:
                return None

            messages = state.get("messages", [])
            if not messages:
                return None
            _debug_log_last_ai_message(
                label="direct_output.before",
                messages=messages,
                session_id=str(getattr(context, "session_id", "")),
            )

            last_ai: AIMessage | None = None
            for item in reversed(messages):
                if isinstance(item, AIMessage):
                    last_ai = item
                    break
            if last_ai is None:
                return None

            content = getattr(last_ai, "content", "")
            if not isinstance(content, str):
                return None

            text = THINKING_TAG_PATTERN.sub("", content).strip()
            if not text:
                return None

            _, remaining = extract_note_tags(text)
            remaining = (remaining or "").strip()
            if not remaining:
                return None

            if streaming_enabled:
                return None

            async def _send() -> None:
                try:
                    messages_to_send = [msg for msg in await parse_send_output_blocks(remaining) if msg]
                    if not messages_to_send:
                        return
                    await transport.send_messages(messages_to_send, interval=0.2)
                    logger.info(
                        "queued direct AI output messages: "
                        f"count={len(messages_to_send)} "
                        f"session={getattr(context, 'session_id', '')}"
                    )
                except Exception:
                    logger.error("AI direct output dispatch failed", exc_info=True)

            create_task(_send())
            return None
        except Exception:
            logger.error("DirectAssistantOutputMiddleware.after_model failed", exc_info=True)
            return None


def _normalize_recovered_tool_call(
    *,
    tool_call_id: Any,
    tool_name: Any,
    tool_args: Any,
) -> dict[str, Any] | None:
    """将原始工具调用数据规范化为 LangChain `AIMessage.tool_calls` 期望的结构。"""

    normalized_name = str(tool_name or "").strip()
    if not normalized_name:
        return None
    if not isinstance(tool_args, dict):
        return None
    return {
        "id": str(tool_call_id or ""),
        "name": normalized_name,
        "args": tool_args,
        "type": "tool_call",
    }


def _try_parse_tool_call_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """尽量把百炼返回的工具参数字符串修复为 JSON object。"""

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        return None

    candidate = raw_arguments.strip()
    if not candidate:
        return None

    candidates: list[str] = [candidate]

    if candidate.startswith("{"):
        trailing_comma_trimmed = candidate.rstrip()
        if trailing_comma_trimmed.endswith(","):
            trailing_comma_trimmed = trailing_comma_trimmed[:-1].rstrip()
            if trailing_comma_trimmed:
                candidates.append(trailing_comma_trimmed)

        balance = candidate.count("{") - candidate.count("}")
        if balance > 0:
            candidates.append(candidate + ("}" * balance))
            if trailing_comma_trimmed != candidate:
                candidates.append(trailing_comma_trimmed + ("}" * balance))

    seen: set[str] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        try:
            parsed = json.loads(item)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _extract_tool_call_candidates_from_message(message: Any) -> list[dict[str, Any]]:
    """从 AIMessage 的原始字段中提取候选工具调用，兼容 DashScope/非标准返回。"""

    candidates: list[dict[str, Any]] = []

    additional_kwargs = getattr(message, "additional_kwargs", None)
    raw_tool_calls = additional_kwargs.get("tool_calls") if isinstance(additional_kwargs, dict) else None
    if isinstance(raw_tool_calls, list):
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            function_payload = item.get("function")
            if not isinstance(function_payload, dict):
                continue
            candidates.append(
                {
                    "id": item.get("id"),
                    "name": function_payload.get("name"),
                    "arguments": function_payload.get("arguments"),
                }
            )

    raw_content = getattr(message, "content", None)
    if isinstance(raw_content, list):
        for item in raw_content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type not in {"tool_call", "function_call", "server_tool_call"}:
                continue

            function_payload = item.get("function")
            if isinstance(function_payload, dict):
                candidates.append(
                    {
                        "id": item.get("id"),
                        "name": function_payload.get("name"),
                        "arguments": function_payload.get("arguments"),
                    }
                )
                continue

            candidates.append(
                {
                    "id": item.get("id") or item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or item.get("args"),
                }
            )

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if isinstance(invalid_tool_calls, list):
        for item in invalid_tool_calls:
            if isinstance(item, dict):
                candidates.append(
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "arguments": item.get("args") or item.get("arguments"),
                    }
                )
                continue

            candidates.append(
                {
                    "id": getattr(item, "id", None),
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "args", None) or getattr(item, "arguments", None),
                }
            )

    return candidates


def _recover_tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    """从原始返回中恢复 `tool_calls`，避免 `create_agent` 因解析失败提前结束。"""

    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in _extract_tool_call_candidates_from_message(message):
        normalized = _normalize_recovered_tool_call(
            tool_call_id=item.get("id"),
            tool_name=item.get("name"),
            tool_args=_try_parse_tool_call_arguments(item.get("arguments")),
        )
        if normalized is not None:
            dedupe_key = (
                str(normalized.get("id", "")),
                str(normalized.get("name", "")),
                json.dumps(normalized.get("args", {}), sort_keys=True, ensure_ascii=True),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            recovered.append(normalized)
    return recovered


def _replace_message_tool_calls(message: Any, tool_calls: list[dict[str, Any]]) -> Any:
    """生成携带修复后 `tool_calls` 的新消息对象。"""

    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(
                update={
                    "tool_calls": list(tool_calls),
                    "invalid_tool_calls": [],
                }
            )
        except Exception:
            pass

    try:
        setattr(message, "tool_calls", list(tool_calls))
        setattr(message, "invalid_tool_calls", [])
    except Exception:
        return message
    return message


def _truncate_log_value(value: Any, *, limit: int = 240) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _normalize_log_payload(value: Any, *, fallback_limit: int = 600) -> Any:
    """将日志对象尽量转换为可 JSON 序列化的结构。"""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return _truncate_log_value(value, limit=fallback_limit)


def _format_json_log_text(payload: Any, *, limit: int = 4000) -> str:
    """将日志载荷格式化为带缩进的 JSON 文本。"""

    normalized_payload = _normalize_log_payload(payload, fallback_limit=min(limit, 1200))
    try:
        text = json.dumps(normalized_payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = _truncate_log_value(normalized_payload, limit=limit)
    return _truncate_log_value(text, limit=limit)


def _serialize_message_content_for_logging(
    content: Any,
    *,
    content_limit: int | None,
    fallback_limit: int = 4000,
) -> Any:
    """按需序列化消息 content，并允许调用方控制是否截断。"""

    if isinstance(content, str):
        if content_limit is None:
            return content
        return _truncate_log_value(content, limit=content_limit)

    if isinstance(content, (int, float, bool)) or content is None:
        return content

    normalized = _normalize_log_payload(content, fallback_limit=fallback_limit)
    if content_limit is None or not isinstance(normalized, str):
        return normalized
    return _truncate_log_value(normalized, limit=content_limit)


def _summarize_ai_message(message: AIMessage) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", [])
    invalid_tool_calls = getattr(message, "invalid_tool_calls", [])
    additional_kwargs = getattr(message, "additional_kwargs", {})
    response_metadata = getattr(message, "response_metadata", {})
    finish_reason = response_metadata.get("finish_reason") if isinstance(response_metadata, dict) else None
    return {
        "role": "ai",
        "content": _truncate_log_value(getattr(message, "content", ""), limit=1200),
        "content_len": len(str(getattr(message, "content", ""))),
        "tool_calls": _normalize_log_payload(tool_calls, fallback_limit=1200),
        "invalid_tool_calls": _normalize_log_payload(invalid_tool_calls, fallback_limit=1200),
        "additional_kwargs": _normalize_log_payload(additional_kwargs, fallback_limit=1200),
        "response_metadata": _normalize_log_payload(response_metadata, fallback_limit=1200),
        "finish_reason": finish_reason,
    }


def _debug_log_last_ai_message(*, label: str, messages: list[Any], session_id: str = "") -> None:
    for item in reversed(messages):
        if isinstance(item, AIMessage):
            logger.debug(
                "agent ai diagnostic\n%s",
                _format_json_log_text(
                    {
                        "label": label,
                        "session_id": session_id,
                        "message": _summarize_ai_message(item),
                    },
                    limit=8000,
                ),
            )
            return
    logger.debug(
        "agent ai diagnostic\n%s",
        _format_json_log_text(
            {
                "label": label,
                "session_id": session_id,
                "message": None,
            },
            limit=1200,
        ),
    )


def _summarize_tool_for_logging(tool_obj: Any) -> dict[str, str]:
    name = str(getattr(tool_obj, "name", "") or getattr(tool_obj, "__name__", type(tool_obj).__name__))
    description = str(getattr(tool_obj, "description", "") or "").strip()
    return {
        "name": name,
        "description": _truncate_log_value(description, limit=160),
    }


def _callable_name_for_logging(target: Any) -> str:
    module_name = str(getattr(target, "__module__", "") or "")
    qualname = str(getattr(target, "__qualname__", "") or getattr(target, "__name__", "") or type(target).__name__)
    if module_name:
        return f"{module_name}.{qualname}"
    return qualname


def _message_role_for_logging(message: Any) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, ToolMessage):
        return "tool"
    return str(getattr(message, "type", type(message).__name__)).lower()


def _summarize_message_for_logging(message: Any, *, content_limit: int | None = 320) -> dict[str, Any]:
    content = getattr(message, "content", "")
    summary: dict[str, Any] = {
        "role": _message_role_for_logging(message),
        "content_len": len(str(content)),
        "content": _serialize_message_content_for_logging(
            content,
            content_limit=content_limit,
            fallback_limit=8000,
        ),
    }

    if isinstance(message, AIMessage):
        response_metadata = getattr(message, "response_metadata", {})
        finish_reason = response_metadata.get("finish_reason") if isinstance(response_metadata, dict) else None
        summary["tool_calls"] = len(getattr(message, "tool_calls", []) or [])
        summary["invalid_tool_calls"] = len(getattr(message, "invalid_tool_calls", []) or [])
        summary["finish_reason"] = finish_reason
    elif isinstance(message, ToolMessage):
        summary["name"] = getattr(message, "name", "")
        summary["tool_call_id"] = getattr(message, "tool_call_id", "")

    return summary


def _debug_log_message_sequence(
    *,
    label: str,
    messages: list[Any],
    session_id: str = "",
    limit_messages: int = 8,
) -> None:
    if not messages:
        logger.debug(
            "agent input diagnostic\n%s",
            _format_json_log_text(
                {
                    "label": label,
                    "session_id": session_id,
                    "message_count": 0,
                    "showing_last": 0,
                    "messages": [],
                },
                limit=1200,
            ),
        )
        return

    trimmed_messages = list(messages[-limit_messages:])
    payload = [_summarize_message_for_logging(message) for message in trimmed_messages]
    logger.debug(
        "agent input diagnostic\n%s",
        _format_json_log_text(
            {
                "label": label,
                "session_id": session_id,
                "message_count": len(messages),
                "showing_last": len(trimmed_messages),
                "messages": payload,
            },
            limit=8000,
        ),
    )


def _serialize_agent_message_for_logging(message: Any) -> dict[str, Any]:
    """将 LangChain 消息对象序列化为结构化日志字段。"""

    payload = _summarize_message_for_logging(message, content_limit=None)

    if isinstance(message, AIMessage):
        payload["tool_calls"] = _normalize_log_payload(
            getattr(message, "tool_calls", []),
            fallback_limit=1200,
        )
        payload["invalid_tool_calls"] = _normalize_log_payload(
            getattr(message, "invalid_tool_calls", []),
            fallback_limit=1200,
        )
        payload["additional_kwargs"] = _normalize_log_payload(
            getattr(message, "additional_kwargs", {}),
            fallback_limit=1200,
        )
        payload["response_metadata"] = _normalize_log_payload(
            getattr(message, "response_metadata", {}),
            fallback_limit=1200,
        )
    elif isinstance(message, ToolMessage):
        payload["additional_kwargs"] = _normalize_log_payload(
            getattr(message, "additional_kwargs", {}),
            fallback_limit=1200,
        )

    return payload


def _extract_non_message_fields_for_logging(payload: Any) -> dict[str, Any]:
    """提取顶层非 messages 字段，并转换为适合日志输出的结构。"""

    if not isinstance(payload, dict):
        return {"value": _normalize_log_payload(payload, fallback_limit=1200)}

    extracted: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "messages":
            continue
        extracted[str(key)] = _normalize_log_payload(value, fallback_limit=1200)
    return extracted


class ToolCallRecoveryMiddleware(AgentMiddleware[AgentState, GroupChatContext]):
    """在上游返回非标准工具调用结构时，尽量恢复为可执行的 `tool_calls`。"""

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        try:
            messages = state.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return None
            session_id = getattr(getattr(runtime, "context", None), "session_id", "")

            last_ai_index: int | None = None
            last_ai: AIMessage | None = None
            for idx in range(len(messages) - 1, -1, -1):
                item = messages[idx]
                if isinstance(item, AIMessage):
                    last_ai_index = idx
                    last_ai = item
                    break
            if last_ai_index is None or last_ai is None:
                return None

            _debug_log_last_ai_message(
                label="tool_recovery.before",
                messages=messages,
                session_id=session_id,
            )

            tool_calls = getattr(last_ai, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls:
                return None

            recovered_tool_calls = _recover_tool_calls_from_message(last_ai)
            if not recovered_tool_calls:
                response_metadata = getattr(last_ai, "response_metadata", None)
                finish_reason = response_metadata.get("finish_reason") if isinstance(response_metadata, dict) else None
                invalid_tool_calls = getattr(last_ai, "invalid_tool_calls", None)
                if finish_reason == "tool_calls" or (isinstance(invalid_tool_calls, list) and invalid_tool_calls):
                    logger.warning(
                        "tool-call recovery failed: "
                        f"finish_reason={finish_reason} "
                        f"invalid_count={len(invalid_tool_calls) if isinstance(invalid_tool_calls, list) else 0} "
                        f"session={session_id}"
                    )
                return None

            messages[last_ai_index] = _replace_message_tool_calls(last_ai, recovered_tool_calls)
            logger.warning(
                "recovered tool calls from non-standard model output: "
                f"count={len(recovered_tool_calls)} "
                f"names={[str(item.get('name', '')) for item in recovered_tool_calls]} "
                f"session={session_id}"
            )
            _debug_log_last_ai_message(
                label="tool_recovery.after",
                messages=messages,
                session_id=session_id,
            )
            return None
        except Exception:
            logger.error("ToolCallRecoveryMiddleware.after_model failed", exc_info=True)
            return None


# ============================================================================
# 响应锁管理器
# ============================================================================


class ResponseLockManager:
    """群聊响应事件锁管理器（非阻塞式拒绝）。

    用于控制同时进行的群聊响应事件数量，支持两个级别的限制：
    1. 单个群组级别：限制某个特定群组同时响应的事件数
    2. 全局级别：限制所有群组总共同时响应的事件数

    当锁满时会立即拒绝新请求，而不是等待。

    Example:
        >>> lock_mgr = ResponseLockManager(max_concurrent_responses_per_group=1, max_total_concurrent_responses=5)
        >>> if lock_mgr.try_acquire(group_id=123):
        ...     try:
        ...         # 处理聊群响应
        ...         pass
        ...     finally:
        ...         lock_mgr.release(group_id=123)
        ... else:
        ...     logger.info("锁满，拒绝本次请求")
    """

    def __init__(
        self,
        max_concurrent_responses_per_group: int = 1,
        max_total_concurrent_responses: int = 5,
    ) -> None:
        """初始化响应锁管理器。

        Args:
            max_concurrent_responses_per_group: 单个群组最多允许同时响应的事件数。
                0 表示不限制。默认为 1。
            max_total_concurrent_responses: 全局最多允许同时响应的总事件数。
                0 表示不限制。默认为 5。

        Note:
            如果两个参数都为 0，则不进行任何限制。
        """

        self.max_concurrent_responses_per_group = max_concurrent_responses_per_group
        self.max_total_concurrent_responses = max_total_concurrent_responses

        # 用于控制全局并发数
        self._global_semaphore: Semaphore | None = None
        if max_total_concurrent_responses > 0:
            self._global_semaphore = Semaphore(max_total_concurrent_responses)

        # 用于控制每个群组的并发数
        self._group_semaphores: dict[str, Semaphore] = {}

    def _get_group_semaphore(self, group_id: str) -> Semaphore | None:
        """获取指定群组的信号量。

        Args:
            group_id: 群组 ID。

        Returns:
            群组的信号量，如果不需要限制返回 None。
        """
        if self.max_concurrent_responses_per_group == 0:
            return None

        if group_id not in self._group_semaphores:
            self._group_semaphores[group_id] = Semaphore(self.max_concurrent_responses_per_group)

        return self._group_semaphores[group_id]

    def try_acquire(self, group_id: str) -> bool:
        """尝试获取响应锁（非阻塞）。

        立即尝试同时获取全局锁和群组锁。如果任一锁满，则立即返回 False。

        Args:
            group_id: 群组 ID。

        Returns:
            如果成功获得锁返回 True，否则返回 False。

        Note:
            - 获得锁后必须调用 release() 来释放
            - 如果返回 False，不需要调用 release()
        """
        # 尝试获取全局锁
        if self._global_semaphore is not None:
            if self._global_semaphore._value <= 0:
                return False
            self._global_semaphore._value -= 1

        # 尝试获取群组锁
        group_semaphore = self._get_group_semaphore(group_id)
        if group_semaphore is not None:
            if group_semaphore._value <= 0:
                # 回滚全局锁
                if self._global_semaphore is not None:
                    self._global_semaphore._value += 1
                return False
            group_semaphore._value -= 1

        return True

    def release(self, group_id: str) -> None:
        """释放响应锁。

        Args:
            group_id: 群组 ID。

        Note:
            只应该在 try_acquire() 返回 True 后调用。
        """
        # 释放群组锁
        group_semaphore = self._get_group_semaphore(group_id)
        if group_semaphore is not None:
            group_semaphore._value += 1

        # 释放全局锁
        if self._global_semaphore is not None:
            self._global_semaphore._value += 1


def parse_send_message_blocks(content: str) -> List[str]:
    """解析文本中的 `<msg>...</msg>` 消息块。

    约定：只有被完整 `<msg>...</msg>` 包裹的内容才允许发送到群里。
    该函数仅负责抽取消息块，不负责发送。

    Args:
        content: 可能包含一个或多个 `<msg>...</msg>` 块的原始文本。

    Returns:
        解析出的消息列表；未找到任何完整消息块时返回空列表。

    Example:
        >>> text = "<msg>你好！</msg>\n<msg>再见！</msg>"
        >>> parse_send_message_blocks(text)
        ['你好！', '再见！']
    """
    matches = SEND_MESSAGE_BLOCK_PATTERN.findall(content)
    # 过滤空消息并去除首尾空白
    return [msg.strip() for msg in matches if msg.strip()]


async def parse_send_output_blocks(content: str) -> List[str]:
    """解析文本中的 `<msg>` 与 `<meme>` 发送块。

    Args:
        content: 可能包含 `<msg>` 或 `<meme>` 块的原始文本。

    Returns:
        按出现顺序解析出的待发送消息文本或 CQ:image 字符串。
    """

    if not content:
        return []

    from plugins.GTBot.tools.meme.tool import resolve_meme_title_to_cq

    resolved: List[str] = []
    for match in SEND_OUTPUT_BLOCK_PATTERN.finditer(content):
        tag = str(match.group("tag") or "").strip().lower()
        block_content = str(match.group("content") or "").strip()
        if not block_content:
            continue

        if tag == "msg":
            resolved.append(block_content)
            continue

        if tag == "meme":
            cq = await resolve_meme_title_to_cq(block_content)
            if cq:
                resolved.append(cq)
            else:
                logger.warning("未找到可发送的表情包: %s", block_content)

    return resolved


def extract_note_tags(content: str) -> tuple[list[str], str]:
    """提取 <note>...</note> 标签并返回剩余文本。

    标签用途：把标签内文本作为“记事本的一条记录”写入会话记事本。
    该函数只负责解析与剔除标签，不负责落库/写入记事本。

    Args:
        content: 原始文本。

    Returns:
        tuple[list[str], str]: (note 列表, 剩余文本)。
            - note 列表会按出现顺序返回，且每条会做 strip。
            - 剩余文本会移除所有 note 标签并做 strip。
    """
    if not content:
        return [], ""

    notes = [n.strip() for n in NOTE_TAG_PATTERN.findall(content) if n and n.strip()]
    remaining = NOTE_TAG_PATTERN.sub("", content).strip()
    return notes, remaining


# ============================================================================
# 响应锁管理器
# ============================================================================


# 初始化全局响应锁管理器
response_lock_manager = ResponseLockManager(
    max_concurrent_responses_per_group=config.chat_model.max_concurrent_responses_per_group,
    max_total_concurrent_responses=config.chat_model.max_total_concurrent_responses
)
"""全局响应锁管理器，用于控制聊群响应的并发数。"""


# ============================================================================
# 插件系统
# ============================================================================


def convert_openai_to_langchain_messages(openai_messages: List[dict]) -> List:
    """将 OpenAI 格式的消息列表转换为 LangChain 格式的消息对象列表。
    
    OpenAI 格式示例:
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    LangChain 格式示例:
    [
        SystemMessage("You are a helpful assistant"),
        HumanMessage("Hello"),
        AIMessage("Hi there!")
    ]
    
    Args:
        openai_messages (List[dict]): OpenAI 格式的消息列表，每个消息包含 "role" 和 "content" 字段。
    
    Returns:
        List: LangChain 格式的消息对象列表（SystemMessage、HumanMessage、AIMessage）。
    
    Raises:
        ValueError: 当消息角色不是 "system"、"user" 或 "assistant" 时抛出异常。
    
    Example:
        >>> openai_msgs = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello"}
        ... ]
        >>> langchain_msgs = convert_openai_to_langchain_messages(openai_msgs)
        >>> len(langchain_msgs)
        3
    """
    langchain_messages: List = []
    
    for msg in openai_messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "system":
            langchain_messages.append(SystemMessage(content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content))
        else:
            raise ValueError(f"未支持的消息角色: {role}。支持的角色有: 'system', 'user', 'assistant'")
    
    return langchain_messages


def create_group_chat_agent(*, runtime_context: GroupChatContext, plugin_bundle: PluginBundle):
    """创建一个用于处理群聊消息的智能体。
    
    Returns:
        智能体实例，可用于处理群聊消息。
    
    Note:
        如果配置了 max_tool_calls_per_turn > 0，则会添加工具调用次数限制中间件。
        超过限制后，智能体将停止工具调用并返回错误信息。
    """

    if getattr(runtime_context, "chat_type", "group") == "private":
        tools = [
            send_private_message_tool,
            delete_message_tool,
            send_like_tool,
        ]
    else:
        tools = [
            send_group_message_tool,
            delete_message_tool,
            emoji_reaction_tool,
            poke_user_tool,
            send_like_tool,
        ]

    tools.extend(list(plugin_bundle.tools))
    logger.info(
        "agent tool inventory: "
        f"session={getattr(runtime_context, 'session_id', '')} "
        f"count={len(tools)} "
        f"tools={json.dumps([_summarize_tool_for_logging(tool_obj) for tool_obj in tools], ensure_ascii=False)}"
    )

    global agent_cache_info

    model: Any

    # ===== 1) 检查缓存是否命中 =====
    t1 = time()
    api_key = config.chat_model.api_key

    extra_body, configured_streaming_enabled, _, _, _ = _parse_streaming_settings(config.chat_model.parameters)
    streaming_enabled = bool(getattr(runtime_context, "streaming_enabled", configured_streaming_enabled))

    cache_hit = (
        agent_cache_info.get("model") is not None
        and agent_cache_info.get("provider_type") == config.chat_model.provider_type
        and agent_cache_info.get("model_id") == config.chat_model.model_id
        and agent_cache_info.get("base_url") == config.chat_model.base_url
        and agent_cache_info.get("api_key") == api_key
        and agent_cache_info.get("extra_body") == extra_body
        and agent_cache_info.get("streaming") == streaming_enabled
    )

    if cache_hit:
        model = cast(Any, agent_cache_info["model"])
        logger.debug("命中模型缓存")
    else:
        model = build_chat_model(
            provider_type=config.chat_model.provider_type,
            model_id=config.chat_model.model_id,
            base_url=config.chat_model.base_url,
            api_key=api_key,
            streaming=streaming_enabled,
            model_parameters=extra_body,
        )

        agent_cache_info.update({
            "model": model,
            "provider_type": config.chat_model.provider_type,
            "model_id": config.chat_model.model_id,
            "base_url": config.chat_model.base_url,
            "api_key": api_key,
            "extra_body": extra_body,
            "streaming": streaming_enabled,
        })
        logger.debug("模型缓存已更新")

    logger.info(f"模型创建耗时: {time() - t1:.2f}")

    if plugin_bundle.callbacks:
        with_config = getattr(model, "with_config", None)
        if callable(with_config):
            try:
                model = with_config(
                    {
                        "callbacks": list(plugin_bundle.callbacks),
                        "tags": list(plugin_bundle.tags),
                        "metadata": dict(plugin_bundle.metadata),
                    }
                )
            except Exception:
                logger.error("插件 callbacks 注入失败", exc_info=True)
        else:
            logger.warning("当前 model 不支持 with_config，已跳过插件 callbacks 注入")

    # ===== 2) 构建中间件 =====
    middleware: list[AgentMiddleware[Any, GroupChatContext]] = [
        AgentPerStepResponseLoggingMiddleware(),
        ToolCallRecoveryMiddleware(),
        DirectAssistantOutputMiddleware(),
    ]
    t1 = time()

    if config.chat_model.max_tool_calls_per_turn > 0:
        middleware.append(
            cast(
                AgentMiddleware[Any, GroupChatContext],
                ToolCallLimitMiddleware(
                    run_limit=config.chat_model.max_tool_calls_per_turn,
                    exit_behavior="continue",
                ),
            )
        )
        logger.debug(f"工具调用限制已启用: 单回合最多 {config.chat_model.max_tool_calls_per_turn} 次")

    logger.info(f"中间件创建耗时: {time() - t1:.2f}")

    if plugin_bundle.agent_middlewares:
        middleware.extend(cast(list[AgentMiddleware[Any, GroupChatContext]], list(plugin_bundle.agent_middlewares)))

    # ===== 3) 创建智能体 =====
    t1 = time()
    agent = create_agent(
        model=model,
        tools=tools,
        context_schema=GroupChatContext,
        middleware=middleware,
    )

    logger.info(f"实际智能体创建耗时: {time() - t1:.2f}")
    return agent


def format_agent_response_for_logging(response: dict) -> str:
    """将智能体响应格式化为带缩进的 JSON 文本。"""

    if not isinstance(response, dict):
        return _format_json_log_text({"value": response}, limit=12000)

    raw_messages = list(response.get("messages", []) or [])
    if raw_messages and isinstance(raw_messages[0], SystemMessage):
        raw_messages = raw_messages[1:]

    payload = {
        "messages": [_serialize_agent_message_for_logging(message) for message in raw_messages],
        "metadata": _extract_non_message_fields_for_logging(response),
    }
    return _format_json_log_text(payload, limit=50000)


def format_agent_response_metadata_for_logging(response: dict) -> str:
    """格式化智能体响应体中除 messages 外的顶层字段，便于排查链路问题。"""

    return _format_json_log_text(
        _extract_non_message_fields_for_logging(response),
        limit=4000,
    )


def format_agent_state_metadata_for_logging(state: Any) -> str:
    """格式化单次模型响应后的 state 顶层字段（排除 messages）。"""

    return _format_json_log_text(
        _extract_non_message_fields_for_logging(state),
        limit=4000,
    )


class AgentPerStepResponseLoggingMiddleware(AgentMiddleware[AgentState, GroupChatContext]):
    """记录每一次模型响应后的 state（排除 messages），用于排查多轮工具链。"""

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        try:
            context = getattr(runtime, "context", None)
            session_id = str(getattr(context, "session_id", ""))
            response_id = str(getattr(context, "response_id", ""))
            logger.info(
                "agent step response metadata (without messages)\n%s",
                _format_json_log_text(
                    {
                        "session_id": session_id,
                        "response_id": response_id,
                        "state": _extract_non_message_fields_for_logging(state),
                    },
                    limit=8000,
                ),
            )
            _debug_log_last_ai_message(
                label="per_step_response",
                messages=list(state.get("messages", []) or []),
                session_id=session_id,
            )
            return None
        except Exception:
            logger.error("AgentPerStepResponseLoggingMiddleware.after_model failed", exc_info=True)
            return None


async def process_assistant_direct_output(
    response: dict,
    bot: Bot,
    group_id: int,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    interval: float = 0.2
) -> None:
    """处理智能体响应中的直接输出文本（包含工具调用场景）。

    无论智能体是否调用了工具，本函数都会读取最后一条 `AIMessage.content`,
    并按以下规则处理：

    1) 支持 `<note>...</note>` 标签：仅剔除标签，不在核心路径持久化（交给 long_memory 插件）。
    2) 若剩余文本包含 ```send_message 代码块，则解析为多条消息发送。
    3) 否则将剩余文本作为单条消息发送。
    4) 若输出仅包含 `<note>` 且剔除后无可发送文本，则只记录不发送。

    Args:
        response (dict): 智能体响应字典，需包含 `messages`。
        bot (Bot): OneBot 机器人实例。
        group_id (int): 目标群组 ID。
        message_manager (GroupMessageManager): 消息管理器。
        cache (CacheManager.UserCacheManager): 缓存管理器。
        interval (float): 多条消息之间的发送间隔（秒）。默认 0.2。
    """
    from langchain_core.messages import AIMessage
    
    messages = response.get("messages", [])
    if not messages:
        return
    
    # 获取最后一条消息
    last_message = messages[-1]
    
    # 只处理 AIMessage 类型
    if not isinstance(last_message, AIMessage):
        return
    
    # 获取文本内容（可能为空；例如仅包含工具调用）
    content = last_message.content if hasattr(last_message, 'content') else ""
    if not content or not isinstance(content, str):
        return
    
    content = content.strip()
    if not content:
        return

    # 先剥离 <thinking>...</thinking>，避免其中出现字面量 <msg>/<note> 影响解析。
    content = THINKING_TAG_PATTERN.sub("", content).strip()
    if not content:
        return

    # 处理 <note>...</note>：仅剔除标签，不在核心路径持久化（交给 long_memory 插件）
    _, content = extract_note_tags(content)
    if not content:
        return
    
    # 尝试解析 send_message 代码块
    parsed_messages = await parse_send_output_blocks(content)
    if not parsed_messages:
        logger.error("AI 直接输出未包含任何完整 <msg>...</msg> 或 <meme>...</meme> 块，已跳过发送")
        return

    messages_to_send = parsed_messages
    logger.debug(f"解析到 {len(messages_to_send)} 条可发送消息块:\n{messages_to_send}")
    
    if parsed_messages:
        messages_to_send = parsed_messages
        logger.debug(f"解析到 {len(messages_to_send)} 条 send_message 代码块消息\n{messages_to_send}")
    else:
        logger.error("AI 直接输出未包含任何完整 <msg>...</msg> 块，已跳过发送")
        return

    await _enqueue_group_messages(
        bot=bot,
        group_id=group_id,
        message_manager=message_manager,
        cache=cache,
        messages=messages_to_send,
        interval=interval,
    )
    logger.info(f"已将 AI 直接输出的 {len(messages_to_send)} 条消息加入发送队列（群组 {group_id}）")


async def _invoke_agent_with_streaming_to_queue(
    *,
    agent: Any,
    chat_context: list[Any],
    runtime_context: GroupChatContext,
    invoke_config: RunnableConfig | None,
    stream_chunk_chars: int,
    stream_flush_interval_sec: float,
    process_tool_call_deltas: bool,
) -> dict:
    """以流式方式运行智能体，并把增量内容分段加入消息队列。

    Args:
        agent: `create_agent()` 返回的可运行对象。
        chat_context: LangChain message 列表（`HumanMessage`/`SystemMessage`/`AIMessage` 等）。
        runtime_context: 运行时上下文（注入到 ToolRuntime.context）。
        stream_chunk_chars: 发送分段最小字符数。
        stream_flush_interval_sec: 最长刷新间隔（秒）。

    Returns:
        dict: 智能体最终输出 state（通常包含 `messages`）。
    """

    buffer: str = ""
    final_output: dict | None = None

    # 流式标签剥离：
    # - 仅转发 <msg>...</msg> 内部文本（与非流式行为保持一致，避免把模型输出的合法文本吞掉）。
    # - <note>...</note> 仅用于记事本，不转发到群
    in_msg: bool = False
    in_note: bool = False
    in_thinking: bool = False
    tag_buf: str = ""  # 可能跨 chunk 的标签缓存（含 '<'..'>'）
    note_buf: str = ""
    thinking_emoji_sent: bool = False

    def _maybe_add_thinking_emoji_from_stream() -> None:
        """在流式输出检测到 `<thinking>` 起始标签时，尽快贴“思考中”表情。

        说明：
            - 不依赖 LangChain 的 token callbacks（在 astream_events 管线中可能不触发）。
            - 只在本次响应内触发一次。
            - 若插件上下文已标记 thinking_emoji_sent，则不重复贴。
        """

        nonlocal thinking_emoji_sent
        if thinking_emoji_sent:
            return

        # 尽量与 thinking 插件的“只贴一次”标记共用，避免重复。
        try:
            from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context

            ctx = get_current_plugin_context()
        except Exception:
            ctx = None

        if ctx is not None and ctx.extra.get("thinking_emoji_sent") is True:
            thinking_emoji_sent = True
            return

        thinking_emoji_sent = True
        if ctx is not None:
            ctx.extra["thinking_emoji_sent"] = True

        async def _send() -> None:
            try:
                if getattr(runtime_context, "chat_type", None) != "group":
                    return
                message_id = getattr(runtime_context, "message_id", None)
                if message_id is None:
                    return
                await Fun.set_msg_emoji_like(
                    bot=runtime_context.bot,
                    message_id=int(message_id),
                    emoji_id=314,
                )
            except Exception:
                return

        try:
            create_task(_send())
        except RuntimeError:
            return

    async def flush(force: bool) -> None:
        nonlocal buffer

        if not buffer:
            return

        if not force:
            return

        messages_to_send = [m for m in await parse_send_output_blocks(buffer) if m]
        buffer = ""
        if not messages_to_send:
            return

        transport = getattr(runtime_context, "transport", None)
        if transport is None:
            return
        await transport.send_messages(messages_to_send, interval=0.0)

    def _parse_xml_like_tag(tag: str) -> tuple[str | None, bool]:
        """解析形如 `<tag ...>` / `</tag>` 的标签。

        Args:
            tag: 包含尖括号的原始标签文本。

        Returns:
            tuple[str | None, bool]: (tag_name, is_end_tag)
                - tag_name: 小写标签名；无法解析时为 None。
                - is_end_tag: 是否为结束标签（例如 `</msg>`）。
        """
        stripped = tag.strip()
        if not (stripped.startswith("<") and stripped.endswith(">")):
            return None, False

        inner = stripped[1:-1].strip()
        if not inner:
            return None, False

        is_end_tag = inner.startswith("/")
        if is_end_tag:
            inner = inner[1:].strip()
        if not inner:
            return None, is_end_tag

        tag_name = inner.split()[0].lower()
        return tag_name, is_end_tag

    def _handle_tag(tag: str) -> tuple[bool, bool]:
        """处理 msg/note 标签。

        Args:
            tag: 原始标签文本（含 `<` `>`）。

        Returns:
            tuple[bool, bool]: (handled, flush_now)
                - handled: 是否为本解析器关心的标签（msg/note）。
                - flush_now: 是否建议立即 flush（仅在 `</msg>` 时为 True）。
        """
        nonlocal in_msg, in_note, in_thinking, note_buf, buffer

        tag_name, is_end_tag = _parse_xml_like_tag(tag)

        # thinking 块：内部内容全部忽略，不参与 msg/note 状态机
        # 仅在“当前不在 msg/note 内”时识别，避免嵌套导致状态异常。
        if tag_name == "thinking" and (not in_msg) and (not in_note):
            in_thinking = not is_end_tag
            if not is_end_tag:
                _maybe_add_thinking_emoji_from_stream()
            return True, False

        if in_thinking:
            return False, False

        if tag_name in {"msg", "meme"}:
            if is_end_tag:
                buffer += tag
                in_msg = False
                return True, True
            buffer += tag
            in_msg = True
            return True, False

        if tag_name == "note":
            if is_end_tag:
                in_note = False
                # note 的持久化统一交给 after_model 中间件处理，避免重复写入。
                note_buf = ""
                return True, False
            in_note = True
            note_buf = ""
            return True, False

        return False, False

    async def _ingest_stream_text(chunk_text: str) -> None:
        nonlocal buffer, tag_buf, in_note, in_msg, in_thinking, note_buf

        if not chunk_text:
            return

        idx = 0
        while idx < len(chunk_text):
            ch = chunk_text[idx]

            # 正在解析标签
            if tag_buf:
                tag_buf += ch
                if ch == ">":
                    handled, flush_now = _handle_tag(tag_buf)
                    if not handled:
                        # 未知标签：按普通文本保留（避免把模型输出的合法文本吞掉）。
                        if in_note:
                            note_buf += tag_buf
                        elif in_msg:
                            buffer += tag_buf
                    tag_buf = ""
                    if flush_now:
                        if not in_msg:
                            await flush(force=True)
                idx += 1
                continue

            # 标签开始（可能跨 chunk），先缓存，等拿到 '>' 再判断是否为 msg/note。
            if ch == "<":
                tag_buf = "<"
                idx += 1
                continue

            # note 内文本：只记录不输出
            if in_note:
                note_buf += ch
                idx += 1
                continue

            # thinking 内文本：完全忽略
            if in_thinking:
                idx += 1
                continue

            # 仅允许输出 <msg>...</msg> 内部文本
            if not in_msg:
                idx += 1
                continue

            buffer += ch
            idx += 1

        await flush(force=False)

    agent_input = cast(Any, {"messages": chat_context})

    def _extract_stream_text_from_chunk(chunk: Any) -> str:
        if chunk is None:
            return ""

        raw_content = getattr(chunk, "content", None)
        if isinstance(raw_content, str):
            return raw_content

        if not isinstance(raw_content, list):
            return ""

        text_parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"tool_call", "tool_call_chunk", "function_call", "server_tool_call"}:
                continue

            text_value = item.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
                continue

            if isinstance(text_value, dict):
                nested_text = text_value.get("value")
                if isinstance(nested_text, str):
                    text_parts.append(nested_text)
                    continue

            content_value = item.get("content")
            if isinstance(content_value, str):
                text_parts.append(content_value)

        return "".join(text_parts)

    def _chunk_contains_tool_call_delta(chunk: Any) -> bool:
        if chunk is None:
            return False

        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        if tool_call_chunks:
            return True

        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            return True

        additional_kwargs = getattr(chunk, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and additional_kwargs.get("tool_calls"):
            return True

        raw_content = getattr(chunk, "content", None)
        if not isinstance(raw_content, list):
            return False

        for item in raw_content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"tool_call", "tool_call_chunk", "function_call", "server_tool_call"}:
                return True

        return False

    async for event in agent.astream_events(
        input=agent_input,
        version="v2",
        context=runtime_context,
        config=invoke_config,
    ):
        event_type = event.get("event")
        data = event.get("data") or {}

        # 增量 token：不同版本/集成可能是 on_chat_model_stream 或 on_llm_stream
        if event_type in {"on_chat_model_stream", "on_llm_stream"}:
            chunk = data.get("chunk")
            if process_tool_call_deltas and _chunk_contains_tool_call_delta(chunk):
                logger.debug("skip streamed tool-call delta when relaying assistant text")
                chunk_content = _extract_stream_text_from_chunk(chunk)
            else:
                chunk_content = getattr(chunk, "content", None)

            if isinstance(chunk_content, str) and chunk_content:
                await _ingest_stream_text(chunk_content)
            elif process_tool_call_deltas and chunk_content:
                await _ingest_stream_text(chunk_content)
            continue

        # 链结束：尽量拿到最终 state
        if event_type == "on_chain_end":
            output = data.get("output")
            if isinstance(output, dict):
                final_output = output
                output_messages = output.get("messages", [])
                if isinstance(output_messages, list):
                    _debug_log_last_ai_message(
                        label="streaming.chain_end",
                        messages=output_messages,
                        session_id=str(getattr(runtime_context, "session_id", "")),
                    )

    await flush(force=True)

    if final_output is None:
        # 兜底：保持接口一致
        return {"messages": []}
    return final_output


def group_messages_by_role(
    messages: List[GroupMessage], 
    self_id: int
) -> List[tuple[str, List[GroupMessage]]]:
    """将消息按发送者角色分组，生成交替的 user/assistant 对话格式。
    
    遍历消息列表，根据消息发送者是否为机器人来判断角色类型，
    将连续相同角色的消息合并为一组，确保输出为交替的对话格式，
    同时保持消息的时间顺序。
    
    Args:
        messages (List[GroupMessage]): 消息对象列表，应按时间顺序排列。
        self_id (int): 机器人的用户 ID。
    
    Returns:
        List[tuple[str, List[GroupMessage]]]: 交替的对话分组列表。
            每个元素为 (角色, 消息对象列表)，角色为 "assistant" 或 "user"。
    
    Example:
        输入消息顺序: [用户A, 用户B, 机器人, 用户C, 机器人]
        输出结果:
        [
            ("user", [GroupMessage用户A, GroupMessage用户B]),
            ("assistant", [GroupMessage机器人]),
            ("user", [GroupMessage用户C]),
            ("assistant", [GroupMessage机器人])
        ]
    """
    if not messages:
        return []
    
    result: List[tuple[str, List[GroupMessage]]] = []
    current_role: str = ""
    current_messages: List[GroupMessage] = []
    
    for msg in messages:
        role = "assistant" if msg.user_id == self_id else "user"
        
        if role != current_role:
            # 角色切换，保存之前的分组
            if current_messages:
                result.append((current_role, current_messages))
            current_role = role
            current_messages = []
        
        current_messages.append(msg)
    
    # 保存最后一组
    if current_messages:
        result.append((current_role, current_messages))
    
    return result


def _replace_cq_code_for_chat_context(di: dict[str, str]) -> str | None:
    """清洗单个 CQ 码，保留构建聊天上下文所需的关键参数。

    Args:
        di: 由 `Fun.replace_cq_codes` 解析得到的 CQ 码字典。

    Returns:
        str | None: 替换后的 CQ 文本；返回 `None` 时保留原始 CQ 码。
    """

    if di["CQ"] == "mface":
        summary = di.get("summary", "")
        return "[CQ:mface" + (f",summary={summary}" if summary else "") + "]"
    if di["CQ"] == "record":
        file = di.get("file", "")
        file_size = di.get("file_size", "")
        return "[CQ:record" + (f",file={file}" if file else "") + (
            f",file_size={file_size}" if file_size else ""
        ) + "]"
    if di["CQ"] == "image":
        file = di.get("file", "")
        file_size = di.get("file_size", "")
        return "[CQ:image" + (f",file={file}" if file else "") + (
            f",file_size={file_size}" if file_size else ""
        ) + "]"

    text = Fun.generate_cq_code([di])[0]
    if len(text) > 100:
        return f"{text[:100]}...]"
    return text


def _copy_group_message_for_chat_context(message: GroupMessage) -> GroupMessage:
    """复制一条消息并清洗 CQ 码内容，避免原地修改原始消息。

    Args:
        message: 原始群消息对象。

    Returns:
        GroupMessage: 可安全用于聊天上下文构建的消息副本。
    """

    sanitized_content = Fun.replace_cq_codes(
        message.content,
        replace_func=_replace_cq_code_for_chat_context,
    )
    copied = getattr(message, "model_copy", None)
    if callable(copied):
        return cast(GroupMessage, copied(update={"content": sanitized_content}))
    return GroupMessage(**{**message.model_dump(), "content": sanitized_content})


async def _format_messages_for_chat_context(
    *,
    messages: List[GroupMessage],
    self_id: int,
    bot: Bot,
    cache: CacheManager.UserCacheManager,
) -> str:
    """将原始群消息格式化为 LLM 可读的聊天历史文本。

    Args:
        messages: 原始群消息列表。
        self_id: 机器人自身账号 ID。
        bot: OneBot 机器人实例。
        cache: 用户缓存管理器。

    Returns:
        str: 清洗 CQ 码后的聊天历史文本。若没有可用内容则返回空字符串。
    """

    sanitized_messages = [_copy_group_message_for_chat_context(message) for message in messages]
    return (
        await Fun.format_messages_to_text(
            sanitized_messages,
            template=config.message_format_placeholder,
            bot=bot,
            cache=cache,
            self_id=self_id,
        )
    ).strip()


def _build_chat_context_from_history_text(history_text: str) -> List[dict]:
    """基于已格式化的聊天历史文本构建 OpenAI 风格上下文。

    Args:
        history_text: 已格式化完成的聊天历史文本。

    Returns:
        List[dict]: 包含 system 与 user 消息的上下文列表。
    """

    context: List[dict] = [{"role": "system", "content": config.chat_model.prompt}]
    if history_text:
        context.append(
            {
                "role": "user",
                "content": "<messages>\n" + history_text + "\n</messages>",
            }
        )
    return context


async def create_group_chat_context(
    messages: List[GroupMessage], 
    self_id: int,
    bot: Bot,
    cache: CacheManager.UserCacheManager,
) -> List[dict]:
    """创建群聊消息上下文信息。
    
    将群消息列表转换为 LLM 可用的对话上下文格式，包含系统提示和按角色分组的消息。
    
    Args:
        messages (List[GroupMessage]): 消息对象列表。
        self_id (int): 机器人的用户 ID。
        bot (Bot): OneBot 机器人实例（用于缓存查询显示名）。
        cache (CacheManager.UserCacheManager): 用户缓存管理器。
        user_id (int): 当前交互用户的 ID。
        group_id (int): 当前群聊的 ID。
    
    Returns:
        List[dict]: LLM 对话上下文列表，每个元素包含 "role" 和 "content" 字段。
    """

    # 剔除部分CQ码参数
    def replace_func(di: dict[str, str]):
        if di["CQ"] == "mface": # 表情
            summary = di.get("summary", "")
            return "[CQ:mface" + (f",summary={summary}" if summary else "") + "]" 
        elif di["CQ"] == "record": # 语音
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:record" + (f",file={file}" if file else "") + \
                    (f",file_size={file_size}" if file_size else "") + "]"
        elif di["CQ"] == "image": # 图片
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:image" + (f",file={file}" if file else "") + \
                    (f",file_size={file_size}" if file_size else "") + "]"
        else:
            text = Fun.generate_cq_code([di])[0]
            if len(text) > 100:
                return f"{text[:100]}...]"

    for message in messages:
        message.content = Fun.replace_cq_codes(message.content, replace_func=replace_func)



    
    context: List[dict] = []

    # 添加系统提示
    context.append({
        "role": "system",
        "content": config.chat_model.prompt
    })

    # 画像注入已暂时禁用（SQLAlchemy 版本），后续改用 LongMemory/Qdrant 版本
    # 添加用户和助手消息：将整个历史信息合并为单个 HumanMessage（role="user"）
    history_text = (
        await Fun.format_messages_to_text(
            messages,
            template=config.message_format_placeholder,
            bot=bot,
            cache=cache,
            self_id=int(bot.self_id),
        )
    ).strip()
    if history_text:
        context.append({
            "role": "user",
            "content": "<messages>\n" + history_text + "\n</messages>",
        })
    
    return context


async def handle_request_rejection(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """处理被拒绝的请求，发送拒绝表情回应。
    
    当响应被拒绝（锁已满）时，如果配置了拒绝表情ID，则发送表情回应。
    
    Args:
        bot (Bot): OneBot 机器人实例。
        msg_id (int): 被拒绝的消息 ID。
        group_id (int): 群组 ID。
    
    Note:
        - 如果 rejection_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.rejection_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.rejection_emoji_id
        )
        logger.debug(f"已发送拒绝表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.rejection_emoji_id}）")
    except Exception as e:
        logger.error(f"发送拒绝表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_processing_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在开始处理请求时发送处理中表情贴。
    
    当开始处理用户请求时，如果配置了处理中表情ID，则发送表情贴。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 如果 processing_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.processing_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.processing_emoji_id
        )
        logger.debug(f"已发送处理中表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.processing_emoji_id}）")
    except Exception as e:
        logger.error(f"发送处理中表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_completion_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在完成请求处理后发送完成表情贴。
    
    当成功完成用户请求处理后，如果配置了完成表情ID，则发送表情贴。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 如果 completion_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.completion_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.completion_emoji_id
        )
        logger.debug(f"已发送完成表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.completion_emoji_id}）")
    except Exception as e:
        logger.error(f"发送完成表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_timeout_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在API请求超时时发送拒绝表情贴。
    
    当API请求超时时，如果配置了拒绝表情ID，则发送表情贴。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 复用 rejection_emoji_id 配置
        - 如果 rejection_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.rejection_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.rejection_emoji_id
        )
        logger.debug(f"已发送超时表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.rejection_emoji_id}）")
    except Exception as e:
        logger.error(f"发送超时表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")

def _build_transport(
    *,
    bot: Bot,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    turn: ChatTurn,
) -> ChatTransport:
    if turn.session.chat_type == "private":
        return PrivateChatTransport(
            bot=bot,
            message_manager=message_manager,
            cache=cache,
            turn=turn,
        )
    return GroupChatTransport(
        bot=bot,
        message_manager=message_manager,
        cache=cache,
        turn=turn,
    )


async def _load_turn_messages(
    *,
    turn: ChatTurn,
    message_manager: GroupMessageManager,
) -> list[GroupMessage]:
    max_messages = (
        config.chat_model.maximum_number_of_incoming_messages
        + NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES
    )

    if turn.anchor_message_id is not None:
        messages = await message_manager.get_nearby_messages(
            message_id=turn.anchor_message_id,
            session_id=turn.session.session_id,
            before=max_messages,
        )
        return messages[-config.chat_model.maximum_number_of_incoming_messages :]

    text = str(turn.input_text or "").strip()
    if not text:
        return []

    return [
        GroupMessage(
            message_id=int(turn.anchor_message_id or 0),
            group_id=int(turn.session.group_id or 0),
            user_id=int(turn.sender_user_id),
            user_name=str(turn.sender_name or ""),
            content=text,
            send_time=time(),
            is_withdrawn=False,
        )
    ]


async def _build_runtime_context(
    *,
    bot: Bot,
    turn: ChatTurn,
    transport: ChatTransport,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    streaming_enabled: bool,
    raw_messages: list[GroupMessage],
    response_id: str,
    response_status: ResponseStatus,
) -> GroupChatContext:
    return GroupChatContext(
        bot=bot,
        chat_type=turn.session.chat_type,
        session_id=turn.session.session_id,
        response_id=response_id,
        response_status=response_status,
        event_loop=asyncio.get_running_loop(),
        group_id=turn.session.group_id,
        user_id=turn.sender_user_id,
        message_id=turn.anchor_message_id,
        event=turn.event,
        message=turn.message,
        message_manager=message_manager,
        cache=cache,
        long_memory=None,
        streaming_enabled=streaming_enabled,
        trigger_mode=turn.trigger_mode,
        trigger_meta=dict(turn.trigger_meta),
        raw_messages=raw_messages,
        transport=transport,
    )


async def _execute_pre_agent_processor(
    *,
    plugin_ctx: PluginContext,
    binding: PreAgentProcessorBinding,
    processor_index: int,
) -> None:
    """执行单个 Agent 前置处理器并处理失败策略。

    Args:
        plugin_ctx: 当前请求的插件上下文。
        binding: 已解析好的前置处理器描述。
        processor_index: 当前处理器在本轮列表中的索引，用于日志定位。

    Raises:
        Exception: 当处理器被标记为必须等待且执行失败时，继续向上抛出。
    """

    try:
        await binding.processor(plugin_ctx)
    except Exception:
        logger.error(
            "pre-agent processor failed: response_id=%s index=%s wait_until_complete=%s",
            plugin_ctx.response_id,
            processor_index,
            binding.wait_until_complete,
            exc_info=True,
        )
        if binding.wait_until_complete:
            raise


async def _run_pre_agent_processors(
    *,
    plugin_ctx: PluginContext,
    processors: list[PreAgentProcessorBinding],
) -> None:
    """并行执行 Agent 启动前的插件处理器。

    Args:
        plugin_ctx: 当前请求的插件上下文。
        processors: 本轮请求启用的前置处理器列表。

    Raises:
        Exception: 任一“必须等待”的处理器失败时向上抛出，以阻断主链路。
    """

    if not processors:
        return

    set_response_status(plugin_ctx, "pre_agent_processing")

    required_tasks: list[asyncio.Task[None]] = []
    for index, binding in enumerate(processors):
        task = create_task(
            _execute_pre_agent_processor(
                plugin_ctx=plugin_ctx,
                binding=binding,
                processor_index=index,
            )
        )
        if binding.wait_until_complete:
            required_tasks.append(task)

    if required_tasks:
        set_response_status(plugin_ctx, "waiting_required_pre_agent_processors")
        await asyncio.gather(*required_tasks)


def _start_pre_agent_processors(
    *,
    plugin_ctx: PluginContext,
    processors: list[PreAgentProcessorBinding],
) -> list[asyncio.Task[None]]:
    """并行启动 Agent 前置处理器，并返回必须等待的任务列表。"""

    if not processors:
        return []

    set_response_status(plugin_ctx, "pre_agent_processing")

    required_tasks: list[asyncio.Task[None]] = []
    for index, binding in enumerate(processors):
        task = create_task(
            _execute_pre_agent_processor(
                plugin_ctx=plugin_ctx,
                binding=binding,
                processor_index=index,
            )
        )
        if binding.wait_until_complete:
            required_tasks.append(task)

    return required_tasks


async def _wait_pre_agent_processors(
    *,
    plugin_ctx: PluginContext,
    required_tasks: list[asyncio.Task[None]],
) -> None:
    """等待所有必须完成的 Agent 前置处理器。"""

    if required_tasks:
        set_response_status(plugin_ctx, "waiting_required_pre_agent_processors")
        await asyncio.gather(*required_tasks)


def _normalize_message_appender_result(result: object) -> list[BaseMessage]:
    """将消息追加器结果规范化为 LangChain message 列表。"""

    if result is None:
        return []
    if isinstance(result, BaseMessage):
        return [result]
    if isinstance(result, list) and all(isinstance(item, BaseMessage) for item in result):
        return list(result)
    raise TypeError("前置消息追加器必须返回 BaseMessage、list[BaseMessage] 或 None。")


def _prepend_messages_after_system_block(
    *,
    messages: list[BaseMessage],
    extra_messages: list[BaseMessage],
) -> list[BaseMessage]:
    """将消息插入到起始 SystemMessage 连续块之后。"""

    if not extra_messages:
        return list(messages)

    system_messages = [
        item
        for item in messages
        if isinstance(item, SystemMessage) or getattr(item, "type", None) == "system"
    ]
    if system_messages:
        other_messages = [item for item in messages if item not in system_messages]
        return list(system_messages) + list(extra_messages) + other_messages

    return list(extra_messages) + list(messages)


async def _apply_pre_agent_message_injectors(
    *,
    plugin_ctx: PluginContext,
    chat_context: list[BaseMessage],
    injectors: list[PreAgentMessageInjectorBinding],
) -> list[BaseMessage]:
    """按优先级串行执行前置消息注入器。"""

    current_messages = list(chat_context)
    for index, binding in enumerate(injectors):
        try:
            produced = await binding.injector(plugin_ctx, list(current_messages))
            if not isinstance(produced, list) or not all(isinstance(item, BaseMessage) for item in produced):
                raise TypeError("前置消息注入器必须返回 list[BaseMessage]。")
            current_messages = list(produced)
        except Exception:
            logger.error(
                "pre-agent message injector failed: response_id=%s index=%s",
                plugin_ctx.response_id,
                index,
                exc_info=True,
            )
    return current_messages


async def _apply_pre_agent_message_appenders(
    *,
    plugin_ctx: PluginContext,
    chat_context: list[BaseMessage],
    appenders: list[PreAgentMessageAppenderBinding],
) -> list[BaseMessage]:
    """按优先级串行执行前置消息追加器。"""

    current_messages = list(chat_context)
    for index, binding in enumerate(appenders):
        try:
            extra_messages = _normalize_message_appender_result(await binding.appender(plugin_ctx))
            if not extra_messages:
                continue

            if binding.position == "prepend":
                current_messages = _prepend_messages_after_system_block(
                    messages=current_messages,
                    extra_messages=extra_messages,
                )
            else:
                current_messages = list(current_messages) + list(extra_messages)
        except Exception:
            logger.error(
                "pre-agent message appender failed: response_id=%s index=%s position=%s",
                plugin_ctx.response_id,
                index,
                binding.position,
                exc_info=True,
            )
    return current_messages


async def _run_pre_agent_message_injection_stage(
    *,
    plugin_ctx: PluginContext,
    chat_context: list[BaseMessage],
    plugin_bundle: PluginBundle,
) -> list[BaseMessage]:
    """执行前置消息注入阶段，并返回最终送入 Agent 的消息列表。"""

    set_response_status(plugin_ctx, "injecting_messages")
    injected_messages = await _apply_pre_agent_message_injectors(
        plugin_ctx=plugin_ctx,
        chat_context=chat_context,
        injectors=list(plugin_bundle.pre_agent_message_injectors),
    )
    return await _apply_pre_agent_message_appenders(
        plugin_ctx=plugin_ctx,
        chat_context=injected_messages,
        appenders=list(plugin_bundle.pre_agent_message_appenders),
    )


async def run_chat_turn(
    *,
    turn: ChatTurn,
    transport: ChatTransport,
    bot: Bot,
    msg_mg: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
) -> None:
    """执行一轮聊天请求的完整主链路。

    Args:
        turn: 本轮输入的会话与消息描述。
        transport: 当前会话对应的发送器。
        bot: OneBot 机器人实例。
        msg_mg: 消息管理器。
        cache: 用户缓存管理器。
    """
    first_time = time()
    session_id = turn.session.session_id
    response_id = uuid4().hex
    response_status: ResponseStatus = "initialized"
    continuation_manager = get_continuation_manager()
    await continuation_manager.close_window(session_id, reason="main_chain_started")
    access_target = _resolve_chat_access_target(turn.session)
    if access_target is not None:
        access_scope, access_target_id = access_target
        access_manager = get_chat_access_manager()
        if not await access_manager.is_allowed(access_scope, access_target_id):
            logger.info(
                "chat access denied: session=%s scope=%s target_id=%s",
                session_id,
                access_scope.value,
                access_target_id,
            )
            return

    if not response_lock_manager.try_acquire(session_id):
        logger.warning("response lock is full, reject session=%s", session_id)
        await transport.handle_rejection_emoji()
        return

    plugin_ctx: PluginContext | None = None
    runtime_context: GroupChatContext | None = None
    history_text_task: asyncio.Task[str] | None = None
    plugin_bundle_task: asyncio.Task[PluginBundle] | None = None
    agent_task: asyncio.Task[Any] | None = None

    try:
        await transport.handle_processing_emoji()

        response_status = "collecting_prerequisites"
        relevant_messages = await _load_turn_messages(turn=turn, message_manager=msg_mg)
        logger.debug(
            "processing chat turn: session=%s message_id=%s context_count=%s",
            session_id,
            turn.anchor_message_id,
            len(relevant_messages),
        )

        (
            _,
            streaming_enabled,
            stream_chunk_chars,
            stream_flush_interval_sec,
            process_tool_call_deltas,
        ) = _parse_streaming_settings(
            config.chat_model.parameters
        )
        if _should_force_non_streaming_tool_agent(
            provider_type=config.chat_model.provider_type,
            base_url=config.chat_model.base_url,
            streaming_enabled=streaming_enabled,
            process_tool_call_deltas=process_tool_call_deltas,
        ):
            logger.warning(
                "dashscope openai-compatible streaming is disabled for tool-capable agent execution; "
                "falling back to non-streaming invoke to preserve tool-call chain"
            )
            streaming_enabled = False

        runtime_context = await _build_runtime_context(
            bot=bot,
            turn=turn,
            transport=transport,
            message_manager=msg_mg,
            cache=cache,
            streaming_enabled=streaming_enabled,
            raw_messages=relevant_messages,
            response_id=response_id,
            response_status=response_status,
        )

        plugin_ctx = build_plugin_context(
            raw_messages=relevant_messages,
            response_id=response_id,
            response_status=response_status,
            runtime_context=runtime_context,
            trigger_mode=turn.trigger_mode,
            trigger_meta=turn.trigger_meta,
        )
        history_text_task = asyncio.create_task(
            _format_messages_for_chat_context(
                messages=relevant_messages,
                self_id=int(bot.self_id),
                bot=bot,
                cache=cache,
            )
        )
        plugin_bundle_task = asyncio.create_task(asyncio.to_thread(build_plugin_bundle, plugin_ctx))

        plugin_bundle = await plugin_bundle_task
        agent_task = asyncio.create_task(
            asyncio.to_thread(
                create_group_chat_agent,
                runtime_context=runtime_context,
                plugin_bundle=plugin_bundle,
            )
        )

        history_text = await history_text_task
        if history_text:
            logger.debug("chat context messages: %s", history_text)

        chat_context = convert_openai_to_langchain_messages(
            _build_chat_context_from_history_text(history_text)
        )
        _debug_log_message_sequence(
            label="pre_injection",
            messages=cast(list[Any], chat_context),
            session_id=session_id,
        )
        chat_agent = await agent_task

        invoke_config: RunnableConfig = {
            "recursion_limit": max(1, int(config.chat_model.recursion_limit)),
        }
        if plugin_bundle.callbacks:
            invoke_config["callbacks"] = list(plugin_bundle.callbacks)
            invoke_config["tags"] = list(plugin_bundle.tags)
            invoke_config["metadata"] = dict(plugin_bundle.metadata)

        logger.debug(
            "plugin bundle diagnostic: "
            f"session={session_id} "
            f"tools={len(plugin_bundle.tools)} "
            f"middlewares={len(plugin_bundle.agent_middlewares)} "
            f"pre_processors={json.dumps([_callable_name_for_logging(item.processor) for item in plugin_bundle.pre_agent_processors], ensure_ascii=False)} "
            f"injectors={json.dumps([_callable_name_for_logging(item.injector) for item in plugin_bundle.pre_agent_message_injectors], ensure_ascii=False)} "
            f"appenders={json.dumps([_callable_name_for_logging(item.appender) for item in plugin_bundle.pre_agent_message_appenders], ensure_ascii=False)}"
        )

        timeout_sec = config.chat_model.api_timeout_sec
        with plugin_context_scope(plugin_ctx):
            required_processor_tasks = _start_pre_agent_processors(
                plugin_ctx=plugin_ctx,
                processors=list(plugin_bundle.pre_agent_processors),
            )
            await _wait_pre_agent_processors(
                plugin_ctx=plugin_ctx,
                required_tasks=required_processor_tasks,
            )
            chat_context = await _run_pre_agent_message_injection_stage(
                plugin_ctx=plugin_ctx,
                chat_context=cast(list[BaseMessage], chat_context),
                plugin_bundle=plugin_bundle,
            )
            _debug_log_message_sequence(
                label="post_injection",
                messages=cast(list[Any], chat_context),
                session_id=session_id,
            )
            set_response_status(plugin_ctx, "agent_running")
            agent_input = cast(Any, {"messages": chat_context})

            if streaming_enabled:
                api_coro = _invoke_agent_with_streaming_to_queue(
                    agent=chat_agent,
                    chat_context=chat_context,
                    runtime_context=runtime_context,
                    invoke_config=invoke_config,
                    stream_chunk_chars=stream_chunk_chars,
                    stream_flush_interval_sec=stream_flush_interval_sec,
                    process_tool_call_deltas=process_tool_call_deltas,
                )
            else:
                api_coro = chat_agent.ainvoke(
                    input=agent_input,
                    context=runtime_context,
                    config=invoke_config,
                )

            if timeout_sec > 0:
                try:
                    response = await wait_for(api_coro, timeout=timeout_sec)
                except AsyncTimeoutError:
                    set_response_status(plugin_ctx, "timed_out")
                    logger.error("API request timed out: session=%s timeout=%ss", session_id, timeout_sec)
                    await transport.handle_timeout_emoji()
                    await transport.send_timeout(timeout_sec)
                    return
            else:
                response = await api_coro

            set_response_status(plugin_ctx, "completed")

        response_metadata = format_agent_response_metadata_for_logging(response)
        logger.info("chat agent response metadata (without messages)\n%s", response_metadata)
        formatted_response = format_agent_response_for_logging(response)
        logger.info(f"chat agent response\n{formatted_response}")
        await transport.handle_completion_emoji()
        if turn.session.chat_type == "group" and turn.session.group_id is not None:
            if _should_open_continuation_window(turn.trigger_mode):
                latest_bot_message_id = await _find_latest_bot_message_id_for_session(
                    session_id=session_id,
                    self_user_id=int(bot.self_id),
                    msg_mg=msg_mg,
                )
                continuation_started_at = _resolve_continuation_window_start_time(
                    turn=turn,
                    relevant_messages=relevant_messages,
                    self_user_id=int(bot.self_id),
                )
                if latest_bot_message_id is not None:
                    await continuation_manager.open_window(
                        session_id=session_id,
                        group_id=int(turn.session.group_id),
                        source_trigger_mode=turn.trigger_mode,
                        last_response_id=response_id,
                        last_bot_message_id=latest_bot_message_id,
                        trigger_started_at=continuation_started_at,
                        bot=bot,
                        msg_mg=msg_mg,
                        cache=cache,
                    )
        logger.info(f"chat turn finished: session={session_id} total={time() - first_time:.2f}s")
    except AsyncTimeoutError:
        if plugin_ctx is not None:
            set_response_status(plugin_ctx, "timed_out")
        elif runtime_context is not None:
            runtime_context.response_status = "timed_out"
        logger.error("API request timed out: session=%s", session_id)
        await transport.handle_timeout_emoji()
    except Exception as e:
        if plugin_ctx is not None:
            set_response_status(plugin_ctx, "failed")
        elif runtime_context is not None:
            runtime_context.response_status = "failed"
        try:
            error_detail = repr(e)
        except Exception:
            error_detail = "unknown error"
        logger.error("chat turn failed: session=%s", session_id, exc_info=True)
        await transport.send_error(error_detail)
    finally:
        for pending_task in (history_text_task, plugin_bundle_task, agent_task):
            if pending_task is not None and not pending_task.done():
                pending_task.cancel()
        response_lock_manager.release(session_id)


async def run_proactive_chat_turn(
    *,
    session: ChatSession,
    input_text: str,
    bot: Bot,
    msg_mg: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    sender_user_id: int | None = None,
    sender_name: str = "",
) -> None:
    turn = ChatTurn(
        session=session,
        sender_user_id=int(sender_user_id or session.peer_user_id),
        sender_name=sender_name,
        anchor_message_id=None,
        input_text=str(input_text or ""),
        trigger_mode=_resolve_proactive_trigger_mode(session),
        trigger_meta={},
        source="proactive",
        event=None,
        message=None,
    )
    transport = _build_transport(
        bot=bot,
        message_manager=msg_mg,
        cache=cache,
        turn=turn,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)


async def run_group_auto_chat_turn(
    *,
    latest_message: GroupMessage,
    trigger_meta: dict[str, Any] | None,
    bot: Bot,
    msg_mg: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
) -> None:
    """基于最近一条群消息构造自动触发请求并进入统一聊天主链路。

    Args:
        latest_message: 触发时最近一条群消息记录。
        trigger_meta: 自动触发专属元数据，会透传给 runtime/plugin context。
        bot: OneBot 机器人实例。
        msg_mg: 消息管理器。
        cache: 用户缓存管理器。
    """

    group_id = int(getattr(latest_message, "group_id", 0) or 0)
    if group_id <= 0:
        raise ValueError("latest_message.group_id 无效，无法构造群聊自动触发")

    sender_user_id = int(getattr(latest_message, "user_id", 0) or 0)
    sender_name = str(getattr(latest_message, "user_name", "") or "")
    anchor_message_id = int(getattr(latest_message, "message_id", 0) or 0)
    input_text = str(getattr(latest_message, "content", "") or "")
    serialized_segments = getattr(latest_message, "serialized_segments", None)

    turn = ChatTurn(
        session=_build_group_session(group_id),
        sender_user_id=sender_user_id,
        sender_name=sender_name,
        anchor_message_id=anchor_message_id if anchor_message_id > 0 else None,
        input_text=input_text,
        trigger_mode="group_auto",
        trigger_meta=dict(trigger_meta or {}),
        source="proactive",
        event=None,
        message=deserialize_message_segments(serialized_segments),
    )
    transport = _build_transport(
        bot=bot,
        message_manager=msg_mg,
        cache=cache,
        turn=turn,
    )
    await run_chat_turn(turn=turn, transport=transport, bot=bot, msg_mg=msg_mg, cache=cache)
