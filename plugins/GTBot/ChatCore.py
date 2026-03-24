from time import time
from collections.abc import Sequence
from langchain_core.tools.base import BaseTool
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List, Callable, Any, Union, Literal
from typing import cast

from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from pathlib import Path
from asyncio import Semaphore, Queue, Lock, TimeoutError as AsyncTimeoutError, wait_for
from dataclasses import dataclass
from asyncio import sleep, create_task, Event

from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI # TODO: 未来支持更多的提供商
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph
from pydantic import SecretStr

from .DBmodel import GroupMessage
from .MassageManager import GroupMessageManager, get_message_manager
from .ConfigManager import total_config, ProcessedConfiguration
from . import Fun
from . import CacheManager
# 暂时禁用 SQLAlchemy 画像系统，改用 LongMemory/Qdrant 版本
# from .UserProfileManager import ProfileManager, get_profile_manager
from .GroupChatContext import GroupChatContext
from plugins.GTBot.services.plugin_system.facade import build_plugin_bundle, build_plugin_context
from plugins.GTBot.services.plugin_system.runtime import plugin_context_scope
from plugins.GTBot.services.plugin_system.types import PluginBundle
from .constants import (
    DEFAULT_BOT_NAME_PLACEHOLDER,
    NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES,
    SEND_MESSAGE_BLOCK_PATTERN,
    SEND_OUTPUT_BLOCK_PATTERN,
    SUPPORTED_CQ_CODES,
    NOTE_TAG_PATTERN,
    THINKING_TAG_PATTERN,
)
from .GroupMessageQueueManager import GroupMessageQueueManager, MessageTask, group_message_queue_manager
from .PrivateMessageQueueManager import PrivateMessageTask, private_message_queue_manager
from .QueueMessagePayload import QueueMessageContent as PreparedQueueMessageContent, prepare_queue_messages
from .ChatAccessManager import ChatAccessScope, get_chat_access_manager
from .Internal_tools import (
    delete_message_tool,
    emoji_reaction_tool,
    poke_user_tool,
    send_group_message_tool,
    send_private_message_tool,
    send_like_tool,
)

GroupChatContext.model_rebuild()

config = total_config.processed_configuration.current_config_group

ChatType = Literal["group", "private"]
ChatSource = Literal["passive", "proactive"]


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

    async def _record_outgoing_message(self, message_id: int, content: str) -> None:
        """将机器人发出的消息补记入统一消息表。"""
        try:
            await self.message_manager.add_chat_message(
                message_id=int(message_id),
                sender_user_id=int(self.bot.self_id),
                sender_name="",
                content=str(content),
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
        await self._record_outgoing_message(int(result["message_id"]), str(output))

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


# ============================================================================
# 消息发送队列管理器（生产者-消费者模型）
# ============================================================================



# 创建model缓存
agent_cache_info: dict = {
    "model": None,
    "model_id": None,
    "base_url": None,
    "api_key": None,
    "extra_body": None,
    "streaming": None,
}


def _parse_streaming_settings(raw_parameters: dict[str, Any] | None) -> tuple[dict[str, Any], bool, int, float]:
    """解析并剥离流式相关参数。

    约定从配置的 `chat_model.parameters` 中读取以下键：
    - `streaming` / `stream`: 是否启用流式（默认 False）
    - `stream_chunk_chars`: 每次向群里发送的最小字符数（默认 160）
    - `stream_flush_interval_sec`: 即使不足 chunk 也会刷新的最大间隔（秒，默认 0.8）

    这些键不会被传给 OpenAI API（避免未知参数）。其余参数会作为 `model_kwargs` 透传给模型。

    Args:
        raw_parameters: 原始参数字典（可能为 None）。

    Returns:
        tuple[dict[str, Any], bool, int, float]:
            (clean_model_kwargs, streaming_enabled, stream_chunk_chars, stream_flush_interval_sec)
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

    return parameters, streaming_enabled, stream_chunk_chars, stream_flush_interval_sec


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
                        "queued direct AI output messages: count=%s session=%s",
                        len(messages_to_send),
                        getattr(context, "session_id", ""),
                    )
                except Exception:
                    logger.error("AI direct output dispatch failed", exc_info=True)

            create_task(_send())
            return None
        except Exception:
            logger.error("DirectAssistantOutputMiddleware.after_model failed", exc_info=True)
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

    global agent_cache_info

    model: Any

    # ===== 1) 检查缓存是否命中 =====
    t1 = time()
    api_key = config.chat_model.api_key

    extra_body, streaming_enabled, _, _ = _parse_streaming_settings(config.chat_model.parameters)

    cache_hit = (
        agent_cache_info.get("model") is not None
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
        model = ChatOpenAI(
            model=config.chat_model.model_id,
            base_url=config.chat_model.base_url,
            api_key=SecretStr(api_key),
            streaming=streaming_enabled,
            extra_body=extra_body
        )

        agent_cache_info.update({
            "model": model,
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
    middleware: list[AgentMiddleware[Any, GroupChatContext]] = [DirectAssistantOutputMiddleware()]
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
    """将智能体响应格式化为人类可读的日志格式。
    
    解析智能体返回的字典，提取关键信息并格式化为清晰的日志输出。
    
    Args:
        response (dict): 智能体的原始响应字典，包含 messages 等键。
    
    Returns:
        str: 格式化后的人类可读字符串。
    """
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("智能体响应摘要")
    lines.append("=" * 50)
    
    # 提取消息列表
    msgs: list[AIMessage|HumanMessage|SystemMessage|ToolMessage] = response.get("messages", [])
    # 剔除第一段SYSTEM消息
    if msgs and isinstance(msgs[0], SystemMessage):
        del msgs[0]
    for i in msgs:
        lines.append(f"==== {i.type} ====")
        lines.append(cast(str, i.content))

    
    return "\n".join(lines)


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
    stream_chunk_chars: int,
    stream_flush_interval_sec: float,
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

    async for event in agent.astream_events(
        input={"messages": chat_context},
        version="v2",
        context=runtime_context,
    ):
        event_type = event.get("event")
        data = event.get("data") or {}

        # 增量 token：不同版本/集成可能是 on_chat_model_stream 或 on_llm_stream
        if event_type in {"on_chat_model_stream", "on_llm_stream"}:
            chunk = data.get("chunk")
            chunk_content = getattr(chunk, "content", None)
            if isinstance(chunk_content, str) and chunk_content:
                await _ingest_stream_text(chunk_content)
            continue

        # 链结束：尽量拿到最终 state
        if event_type == "on_chain_end":
            output = data.get("output")
            if isinstance(output, dict):
                final_output = output

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
            "content": "<messgaes>\n" + history_text + "\n</messgaes>",
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
) -> GroupChatContext:
    return GroupChatContext(
        bot=bot,
        chat_type=turn.session.chat_type,
        session_id=turn.session.session_id,
        group_id=turn.session.group_id,
        user_id=turn.sender_user_id,
        message_id=turn.anchor_message_id,
        event=turn.event,
        message=turn.message,
        message_manager=message_manager,
        cache=cache,
        long_memory=None,
        streaming_enabled=streaming_enabled,
        raw_messages=raw_messages,
        transport=transport,
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
            if turn.source != "proactive":
                await transport.send_feedback("当前会话未被允许使用 GTBot")
            return

    if not response_lock_manager.try_acquire(session_id):
        logger.warning("response lock is full, reject session=%s", session_id)
        await transport.handle_rejection_emoji()
        return

    try:
        await transport.handle_processing_emoji()

        relevant_messages = await _load_turn_messages(turn=turn, message_manager=msg_mg)
        logger.debug(
            "processing chat turn: session=%s message_id=%s context_count=%s",
            session_id,
            turn.anchor_message_id,
            len(relevant_messages),
        )
        if relevant_messages:
            logger.debug(
                "chat context messages: %s",
                await Fun.format_messages_to_text(
                    relevant_messages,
                    template=config.message_format_placeholder,
                    bot=bot,
                    cache=cache,
                    self_id=int(bot.self_id),
                ),
            )

        _, streaming_enabled, stream_chunk_chars, stream_flush_interval_sec = _parse_streaming_settings(
            config.chat_model.parameters
        )

        runtime_context = await _build_runtime_context(
            bot=bot,
            turn=turn,
            transport=transport,
            message_manager=msg_mg,
            cache=cache,
            streaming_enabled=streaming_enabled,
            raw_messages=relevant_messages,
        )

        chat_context = await create_group_chat_context(
            messages=relevant_messages,
            self_id=int(bot.self_id),
            bot=bot,
            cache=cache,
        )
        plugin_ctx = build_plugin_context(raw_messages=relevant_messages, runtime_context=runtime_context)
        plugin_bundle = build_plugin_bundle(plugin_ctx)
        chat_agent = create_group_chat_agent(runtime_context=runtime_context, plugin_bundle=plugin_bundle)
        chat_context = convert_openai_to_langchain_messages(chat_context)

        if streaming_enabled:
            api_coro = _invoke_agent_with_streaming_to_queue(
                agent=chat_agent,
                chat_context=chat_context,
                runtime_context=runtime_context,
                stream_chunk_chars=stream_chunk_chars,
                stream_flush_interval_sec=stream_flush_interval_sec,
            )
        else:
            api_coro = chat_agent.ainvoke(
                input={"messages": chat_context},
                context=runtime_context,
            )

        timeout_sec = config.chat_model.api_timeout_sec
        with plugin_context_scope(plugin_ctx):
            if timeout_sec > 0:
                try:
                    response = await wait_for(api_coro, timeout=timeout_sec)
                except AsyncTimeoutError:
                    logger.error("API request timed out: session=%s timeout=%ss", session_id, timeout_sec)
                    await transport.handle_timeout_emoji()
                    await transport.send_timeout(timeout_sec)
                    return
            else:
                response = await api_coro

        formatted_response = format_agent_response_for_logging(response)
        logger.info("chat agent response\n%s", formatted_response)
        await transport.handle_completion_emoji()
        logger.info("chat turn finished: session=%s total=%.2fs", session_id, time() - first_time)
    except AsyncTimeoutError:
        logger.error("API request timed out: session=%s", session_id)
        await transport.handle_timeout_emoji()
    except Exception as e:
        try:
            error_detail = repr(e)
        except Exception:
            error_detail = "unknown error"
        logger.error("chat turn failed: session=%s", session_id, exc_info=True)
        await transport.send_error(error_detail)
    finally:
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


