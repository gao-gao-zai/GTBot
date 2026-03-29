from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import (
    EnabledPredicate,
    MessageAppendPosition,
    PluginContext,
    PreAgentMessageAppender,
    PreAgentMessageInjector,
    PreAgentProcessor,
)


@dataclass(frozen=True)
class _RegisteredItem:
    priority: int
    enabled: EnabledPredicate | None


@dataclass(frozen=True)
class _RegisteredTool(_RegisteredItem):
    tool: Any


@dataclass(frozen=True)
class _RegisteredToolFactory(_RegisteredItem):
    factory: Callable[[PluginContext], list[Any]]


@dataclass(frozen=True)
class _RegisteredMiddleware(_RegisteredItem):
    middleware: Any


@dataclass(frozen=True)
class _RegisteredCallback(_RegisteredItem):
    callback: Any


@dataclass(frozen=True)
class _RegisteredPreAgentProcessor(_RegisteredItem):
    processor: PreAgentProcessor
    wait_until_complete: bool


@dataclass(frozen=True)
class _RegisteredPreAgentMessageInjector(_RegisteredItem):
    injector: PreAgentMessageInjector


@dataclass(frozen=True)
class _RegisteredPreAgentMessageAppender(_RegisteredItem):
    appender: PreAgentMessageAppender
    position: MessageAppendPosition


class PluginRegistry:
    """插件注册容器。

    插件通过 `register(registry)` 向宿主声明其提供的工具、中间件、回调、
    Agent 前置处理器与前置消息注入能力。
    """

    def __init__(self) -> None:
        self._tools: list[_RegisteredTool] = []
        self._tool_factories: list[_RegisteredToolFactory] = []
        self._middlewares: list[_RegisteredMiddleware] = []
        self._callbacks: list[_RegisteredCallback] = []
        self._pre_agent_processors: list[_RegisteredPreAgentProcessor] = []
        self._pre_agent_message_injectors: list[_RegisteredPreAgentMessageInjector] = []
        self._pre_agent_message_appenders: list[_RegisteredPreAgentMessageAppender] = []

    def add_tool(self, tool: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个固定 tool。"""

        self._tools.append(_RegisteredTool(priority=priority, enabled=enabled, tool=tool))

    def add_tool_factory(
        self,
        factory: Callable[[PluginContext], list[Any]],
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
    ) -> None:
        """注册一个动态 tool 工厂。"""

        self._tool_factories.append(
            _RegisteredToolFactory(priority=priority, enabled=enabled, factory=factory)
        )

    def add_agent_middleware(
        self,
        middleware: Any,
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
    ) -> None:
        """注册一个 Agent middleware。"""

        self._middlewares.append(
            _RegisteredMiddleware(priority=priority, enabled=enabled, middleware=middleware)
        )

    def add_callback(self, callback: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个 callback handler。"""

        self._callbacks.append(_RegisteredCallback(priority=priority, enabled=enabled, callback=callback))

    def add_pre_agent_processor(
        self,
        processor: PreAgentProcessor,
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
        wait_until_complete: bool = False,
    ) -> None:
        """注册一个 Agent 启动前的异步处理器。"""

        self._pre_agent_processors.append(
            _RegisteredPreAgentProcessor(
                priority=priority,
                enabled=enabled,
                processor=processor,
                wait_until_complete=wait_until_complete,
            )
        )

    def add_pre_agent_message_injector(
        self,
        injector: PreAgentMessageInjector,
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
    ) -> None:
        """注册一个前置消息注入器。

        该注入器会收到最终送入 LLM 的 LangChain message 列表，并返回新的列表。
        """

        self._pre_agent_message_injectors.append(
            _RegisteredPreAgentMessageInjector(
                priority=priority,
                enabled=enabled,
                injector=injector,
            )
        )

    def add_pre_agent_message_appender(
        self,
        appender: PreAgentMessageAppender,
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
        position: MessageAppendPosition = "append",
    ) -> None:
        """注册一个前置消息追加器。"""

        if position not in {"prepend", "append"}:
            raise ValueError(f"不支持的消息追加位置: {position}")

        self._pre_agent_message_appenders.append(
            _RegisteredPreAgentMessageAppender(
                priority=priority,
                enabled=enabled,
                appender=appender,
                position=position,
            )
        )

    def iter_tools(self) -> list[_RegisteredTool]:
        return list(self._tools)

    def iter_tool_factories(self) -> list[_RegisteredToolFactory]:
        return list(self._tool_factories)

    def iter_middlewares(self) -> list[_RegisteredMiddleware]:
        return list(self._middlewares)

    def iter_callbacks(self) -> list[_RegisteredCallback]:
        return list(self._callbacks)

    def iter_pre_agent_processors(self) -> list[_RegisteredPreAgentProcessor]:
        return list(self._pre_agent_processors)

    def iter_pre_agent_message_injectors(self) -> list[_RegisteredPreAgentMessageInjector]:
        return list(self._pre_agent_message_injectors)

    def iter_pre_agent_message_appenders(self) -> list[_RegisteredPreAgentMessageAppender]:
        return list(self._pre_agent_message_appenders)
