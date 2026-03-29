from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from langchain_core.messages import BaseMessage


ResponseStatus: TypeAlias = Literal[
    "initialized",
    "collecting_prerequisites",
    "pre_agent_processing",
    "injecting_messages",
    "waiting_required_pre_agent_processors",
    "agent_running",
    "completed",
    "failed",
    "timed_out",
]

MessageAppendPosition: TypeAlias = Literal["prepend", "append"]


@dataclass
class PluginContext:
    """单次请求的插件上下文。

    说明：
        `raw_messages` 主要用于插件读取分析，不要求与最终 LLM 输入强绑定。

    Attributes:
        raw_messages: 本次对话原始消息列表。
        response_id: 本次响应的唯一 ID。
        response_status: 当前响应流程状态。
        runtime_context: 宿主运行时上下文。
        trigger_mode: 当前触发模式。
        trigger_meta: 当前触发元数据。
        extra: 插件间共享的临时数据容器。
    """

    raw_messages: list[Any]
    response_id: str = ""
    response_status: ResponseStatus = "initialized"
    runtime_context: Any | None = None
    trigger_mode: str | None = None
    trigger_meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


EnabledPredicate: TypeAlias = Callable[[PluginContext], bool]
PreAgentProcessor: TypeAlias = Callable[[PluginContext], Awaitable[None]]
PreAgentMessageInjector: TypeAlias = Callable[
    [PluginContext, list[BaseMessage]],
    Awaitable[list[BaseMessage]],
]
PreAgentMessageAppenderResult: TypeAlias = BaseMessage | list[BaseMessage] | None
PreAgentMessageAppender: TypeAlias = Callable[
    [PluginContext],
    Awaitable[PreAgentMessageAppenderResult],
]


@dataclass(frozen=True)
class PreAgentProcessorBinding:
    """单次请求中可执行的 Agent 前置处理器描述。"""

    processor: PreAgentProcessor
    wait_until_complete: bool = False


@dataclass(frozen=True)
class PreAgentMessageInjectorBinding:
    """单次请求中可执行的前置消息注入器描述。"""

    injector: PreAgentMessageInjector


@dataclass(frozen=True)
class PreAgentMessageAppenderBinding:
    """单次请求中可执行的前置消息追加器描述。"""

    appender: PreAgentMessageAppender
    position: MessageAppendPosition = "append"


@dataclass(frozen=True)
class PluginBundle:
    """本次请求最终装配得到的插件产物集合。"""

    tools: list[Any] = field(default_factory=list)
    agent_middlewares: list[Any] = field(default_factory=list)
    callbacks: list[Any] = field(default_factory=list)
    pre_agent_processors: list[PreAgentProcessorBinding] = field(default_factory=list)
    pre_agent_message_injectors: list[PreAgentMessageInjectorBinding] = field(default_factory=list)
    pre_agent_message_appenders: list[PreAgentMessageAppenderBinding] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
