from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import EnabledPredicate, PluginContext, PreAgentProcessor


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


class PluginRegistry:
    """插件注册容器。

    插件通过 `register(registry)` 向宿主声明其提供的 tools、middlewares、
    callbacks 与 Agent 前置处理器。
    """

    def __init__(self) -> None:
        self._tools: list[_RegisteredTool] = []
        self._tool_factories: list[_RegisteredToolFactory] = []
        self._middlewares: list[_RegisteredMiddleware] = []
        self._callbacks: list[_RegisteredCallback] = []
        self._pre_agent_processors: list[_RegisteredPreAgentProcessor] = []

    def add_tool(self, tool: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个固定 tool。

        Args:
            tool: LangChain tool 实例。
            priority: 优先级；值越小越早参与合并。
            enabled: 是否启用的判断函数。
        """

        self._tools.append(_RegisteredTool(priority=priority, enabled=enabled, tool=tool))

    def add_tool_factory(
        self,
        factory: Callable[[PluginContext], list[Any]],
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
    ) -> None:
        """注册一个动态 tool 工厂。

        Args:
            factory: `(ctx) -> tools` 形式的工厂函数。
            priority: 优先级；值越小越早参与合并。
            enabled: 是否启用的判断函数。
        """

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
        """注册一个 Agent middleware。

        Args:
            middleware: `langchain.agents.middleware.AgentMiddleware` 实例。
            priority: 优先级；值越小越早参与合并。
            enabled: 是否启用的判断函数。
        """

        self._middlewares.append(
            _RegisteredMiddleware(priority=priority, enabled=enabled, middleware=middleware)
        )

    def add_callback(self, callback: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个 callback handler。

        Args:
            callback: `BaseCallbackHandler` 等回调对象。
            priority: 优先级；值越小越早参与合并。
            enabled: 是否启用的判断函数。
        """

        self._callbacks.append(_RegisteredCallback(priority=priority, enabled=enabled, callback=callback))

    def add_pre_agent_processor(
        self,
        processor: PreAgentProcessor,
        *,
        priority: int = 0,
        enabled: EnabledPredicate | None = None,
        wait_until_complete: bool = False,
    ) -> None:
        """注册一个 Agent 启动前的异步处理器。

        Args:
            processor: `(ctx) -> awaitable` 形式的异步处理器。
            priority: 优先级；值越小越早启动。
            enabled: 是否启用的判断函数。
            wait_until_complete: 是否必须等待处理器完成后再启动 Agent。
        """

        self._pre_agent_processors.append(
            _RegisteredPreAgentProcessor(
                priority=priority,
                enabled=enabled,
                processor=processor,
                wait_until_complete=wait_until_complete,
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
