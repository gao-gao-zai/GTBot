from time import time
from langchain_core.tools.base import BaseTool
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List, Callable, Any, Union
from typing import cast

from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends, EventMessage
from nonebot import on, on_message, get_driver, on_notice
from nonebot.rule import to_me
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
    SUPPORTED_CQ_CODES,
    NOTE_TAG_PATTERN
)
from .GroupMessageQueueManager import GroupMessageQueueManager, MessageTask, group_message_queue_manager
from .Internal_tools import (
    delete_message_tool,
    emoji_reaction_tool,
    poke_user_tool,
    send_group_message_tool,
    send_like_tool,
)

GroupChatContext.model_rebuild()

config = total_config.processed_configuration.current_config_group


# ============================================================================
# 消息发送队列管理器（生产者-消费者模型）
# ============================================================================



# 创建model缓存
agent_cache_info: dict = {
    "model": None,
    "model_id": None,
    "base_url": None,
    "api_key": None,
    "model_kwargs": None,
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
    messages: list[str],
    interval: float,
) -> None:
    """将多条消息加入发送队列。

    Args:
        bot: OneBot 机器人实例。
        group_id: 目标群号。
        message_manager: 消息管理器。
        cache: 用户缓存管理器。
        messages: 待发送文本列表。
        interval: 多条消息之间的间隔（秒）。
    """
    if not messages:
        return

    task = MessageTask(messages=messages, group_id=group_id, interval=interval)
    await group_message_queue_manager.enqueue(
        task,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
    )


class DirectAssistantOutputMiddleware(AgentMiddleware[AgentState, GroupChatContext]):
    """处理智能体最终 AIMessage 的“直接输出”。

    目标：把原先 handler 里 `process_assistant_direct_output()` 的核心行为下沉到中间件，
    让业务层只负责发起调用（ainvoke/astream_events）和超时/表情等外围控制。

    当前实现：
    - 解析并写入 `<note>...</note>` 到记事本
    - 若非流式模式：把剩余文本作为一条消息入队发送

    Note:
        - 用户侧已表示 ```send_message 块逻辑已废弃，本中间件不再拆分该语法。
        - 中间件 hook 不是 async，这里使用 create_task 调度异步入队。
    """

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        try:
            context = getattr(runtime, "context", None)
            if context is None:
                return None

            bot: Bot | None = getattr(context, "bot", None)
            group_id: int | None = getattr(context, "group_id", None)
            message_manager: GroupMessageManager | None = getattr(context, "message_manager", None)
            cache: CacheManager.UserCacheManager | None = getattr(context, "cache", None)
            streaming_enabled: bool = bool(getattr(context, "streaming_enabled", False))

            if bot is None or group_id is None or message_manager is None or cache is None:
                return None

            last_ai: AIMessage | None = None
            messages = state.get("messages", [])
            if not messages:
                return None

            # after_model 场景：通常最后一条就是本次模型返回的 AIMessage
            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            if last_ai is None:
                return None

            content = getattr(last_ai, "content", "")
            if not isinstance(content, str):
                return None

            text = content.strip()
            if not text:
                return None

            _, remaining = extract_note_tags(text)
            remaining = (remaining or "").strip()
            if not remaining:
                return None

            # 仅发送被 <msg>...</msg> 完整包裹的内容。
            # 没有完整 msg 块则不发送（避免把工具回合/中间推理等内容刷到群里）。
            messages_to_send = [m for m in parse_send_message_blocks(remaining) if m]
            if not messages_to_send:
                return None

            if streaming_enabled:
                # 流式模式下，增量输出由 handler 负责分段入队，避免重复发送。
                return None

            async def _send() -> None:
                try:
                    await _enqueue_group_messages(
                        bot=bot,
                        group_id=group_id,
                        message_manager=message_manager,
                        cache=cache,
                        messages=messages_to_send,
                        interval=0.2,
                    )
                    logger.info(
                        f"已将 AI 直接输出的 {len(messages_to_send)} 条消息加入发送队列（群组 {group_id}）"
                    )
                except Exception:
                    logger.error("AI 直接输出入队失败", exc_info=True)

            create_task(_send())
            return None
        except Exception:
            logger.error("DirectAssistantOutputMiddleware.after_model 执行失败", exc_info=True)
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
        self._group_semaphores: dict[int, Semaphore] = {}

    def _get_group_semaphore(self, group_id: int) -> Semaphore | None:
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

    def try_acquire(self, group_id: int) -> bool:
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

    def release(self, group_id: int) -> None:
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

    tools: List[BaseTool] = [
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

    model_kwargs, streaming_enabled, _, _ = _parse_streaming_settings(config.chat_model.parameters)

    cache_hit = (
        agent_cache_info.get("model") is not None
        and agent_cache_info.get("model_id") == config.chat_model.model_id
        and agent_cache_info.get("base_url") == config.chat_model.base_url
        and agent_cache_info.get("api_key") == api_key
        and agent_cache_info.get("model_kwargs") == model_kwargs
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
            model_kwargs=model_kwargs,
        )

        agent_cache_info.update({
            "model": model,
            "model_id": config.chat_model.model_id,
            "base_url": config.chat_model.base_url,
            "api_key": api_key,
            "model_kwargs": model_kwargs,
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

    # 处理 <note>...</note>：仅剔除标签，不在核心路径持久化（交给 long_memory 插件）
    _, content = extract_note_tags(content)
    if not content:
        return
    
    # 尝试解析 send_message 代码块
    parsed_messages = parse_send_message_blocks(content)
    
    if parsed_messages:
        messages_to_send = parsed_messages
        logger.debug(f"解析到 {len(messages_to_send)} 条 send_message 代码块消息\n{messages_to_send}")
    else:
        logger.error("AI 直接输出未包含任何完整 <msg>...</msg> 块，已跳过发送")
        return

    task = MessageTask(messages=messages_to_send, group_id=group_id, interval=interval)
    await group_message_queue_manager.enqueue(
        task,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
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
    tag_buf: str = ""  # 可能跨 chunk 的标签缓存（含 '<'..'>'）
    note_buf: str = ""

    async def flush(force: bool) -> None:
        nonlocal buffer

        if not buffer:
            return

        if not force:
            return

        messages_to_send = [m for m in parse_send_message_blocks(buffer) if m]
        buffer = ""
        if not messages_to_send:
            return

        await _enqueue_group_messages(
            bot=runtime_context.bot,
            group_id=runtime_context.group_id,
            message_manager=runtime_context.message_manager,
            cache=runtime_context.cache,
            messages=messages_to_send,
            interval=0.0,
        )

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
        nonlocal in_msg, in_note, note_buf, buffer

        tag_name, is_end_tag = _parse_xml_like_tag(tag)
        if tag_name == "msg":
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
        nonlocal buffer, tag_buf, in_note, in_msg, note_buf

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
    user_id: int,
    group_id: int
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
            "content": "[历史对话]\n\n" + history_text,
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


GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)


@GroupChatProactiveRequest.handle()
async def handle_group_chat_request(
    event: GroupMessageEvent, 
    bot: Bot, 
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager)
):
    """处理群聊中的主动请求消息。
    
    该函数实现了响应事件的锁机制，支持两个级别的并发控制：
    - 群组级别：限制单个群组同时进行的响应事件数
    - 全局级别：限制所有群组总共同时进行的响应事件数
    
    当锁满时会直接拒绝本次请求。
    
    新增功能：
    - 接收请求时发送处理中表情贴（processing_emoji_id）
    - 完成请求时发送完成表情贴（completion_emoji_id）
    - API 请求超时处理（api_timeout_sec）
    """
    
    msg_id: int = event.message_id
    group_id: int = event.group_id

    first_time: float = time()

    get_lock_time: float = time()
    # 尝试获取响应锁（非阻塞）
    if not response_lock_manager.try_acquire(group_id):
        logger.warning(f"响应锁已满（群组 {group_id}，消息ID {msg_id}），拒绝本次请求")
        await handle_request_rejection(bot, msg_id, group_id)
        return
    logger.info(f"获取锁耗时: {time() - get_lock_time:.2f}s")
    
    try:
        logger.info(f"获得响应锁（群组 {group_id}，消息ID {msg_id}），开始处理请求")
        
        # 发送处理中表情贴
        await handle_processing_emoji(bot, msg_id, group_id)
        
        max_messages: int = config.chat_model.maximum_number_of_incoming_messages\
                            + NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES

        get_message_time = time()
        # 获取聊天记录信息
        messages: List[GroupMessage] = await msg_mg.get_nearby_messages(
            message_id=msg_id,
            group_id=group_id,
            before=max_messages
        )
        logger.info(f"获取聊天记录耗时: {time() - get_message_time:.2f}s")

        # 截取最后 N 条消息作为上下文
        relevant_messages: list[GroupMessage] = messages[-config.chat_model.maximum_number_of_incoming_messages:]
        logger.debug(
            f"处理群聊请求: 群号 {group_id}, 消息ID {msg_id}, 上下文消息数 {len(relevant_messages)}"
        )
        logger.debug(
            "上下文消息列表: %s",
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

        runtime_context = GroupChatContext(
            bot=bot,
            event=event,
            message=msg,
            group_id=group_id,
            user_id=event.user_id,
            message_id=msg_id,
            message_manager=msg_mg,
            cache=cache,
            long_memory=None,
            streaming_enabled=streaming_enabled,
            raw_messages=relevant_messages,
        )

        creat_agent_time = time()
        # 创建聊天上下文
        chat_context = await create_group_chat_context(
            messages=relevant_messages,
            self_id=int(bot.self_id),
            bot=bot,
            cache=cache,
            user_id=event.user_id,
            group_id=group_id
        )
        logger.info(f"创建聊天上下文耗时: {time() - creat_agent_time:.2f}s")
        t1 = time()
        # 创建聊天智能体
        plugin_ctx = build_plugin_context(raw_messages=relevant_messages, runtime_context=runtime_context)

        plugin_bundle = build_plugin_bundle(plugin_ctx)
        chat_agent = create_group_chat_agent(runtime_context=runtime_context, plugin_bundle=plugin_bundle)
        logger.info(f"t1: {time() - t1:.2f}s")
        t1 = time()
        # 生成适配 LangChain 格式的消息
        chat_context = convert_openai_to_langchain_messages(chat_context)
        logger.info(f"t2: {time() - t1:.2f}s")
        
        logger.info(f"创建agent耗时: {time()-creat_agent_time:.2f}s")
        logger.info(f"总耗时: {time() - first_time:.2f}s")

        # LongMemory 工具通过 ToolRuntime.context 获取 long_memory 与会话信息，无需额外注入。

        # 构建 API 调用的协程
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

        
        # 根据配置决定是否使用超时控制
        timeout_sec = config.chat_model.api_timeout_sec
        with plugin_context_scope(plugin_ctx):
            if timeout_sec > 0:
                try:
                    response = await wait_for(api_coro, timeout=timeout_sec)
                except AsyncTimeoutError:
                    logger.error(f"API 请求超时（群组 {group_id}，消息ID {msg_id}，超时时间 {timeout_sec}秒）")
                    await handle_timeout_emoji(bot, msg_id, group_id)
                    await GroupChatProactiveRequest.send(
                        f"请求处理超时（{timeout_sec}秒），请稍后重试",
                        at_sender=True
                    )
                    return
            else:
                # 不设置超时
                response = await api_coro

        # 格式化并输出人类可读的响应日志
        formatted_response = format_agent_response_for_logging(response)
        logger.info(f"群聊智能体响应:\n{formatted_response}")
        
        # AI 直接输出的文本内容已下沉到 DirectAssistantOutputMiddleware.after_agent
        
        # 发送完成表情贴
        await handle_completion_emoji(bot, msg_id, group_id)
        
        logger.info(f"响应处理完成（群组 {group_id}，消息ID {msg_id}），释放响应锁")
    
    except AsyncTimeoutError:
        # 超时错误已在上面处理，这里作为备用捕获
        logger.error(f"API 请求超时（群组 {group_id}，消息ID {msg_id}）")
        await handle_timeout_emoji(bot, msg_id, group_id)
    
    except Exception as e:
        try:
            error_detail = repr(e) 
        except:
            error_detail = "无法解析的具体错误类型"

        logger.error(f"处理群聊请求时发生错误（群组 {group_id}，消息ID {msg_id}）")
        logger.error(f"错误堆栈: ", exc_info=True)

        await GroupChatProactiveRequest.send(
            f"处理请求时发生错误，请联系管理员或检查API状态。详情: {error_detail}",
            at_sender=True
        )
    
    finally:
        response_lock_manager.release(group_id)