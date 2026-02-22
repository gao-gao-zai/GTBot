from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import EnabledPredicate, PluginContext


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


class PluginRegistry:
    """插件注册容器。

    插件通过 `register(registry)` 向宿主声明其提供的 tools/middlewares/callbacks。
    """

    def __init__(self) -> None:
        self._tools: list[_RegisteredTool] = []
        self._tool_factories: list[_RegisteredToolFactory] = []
        self._middlewares: list[_RegisteredMiddleware] = []
        self._callbacks: list[_RegisteredCallback] = []

    def add_tool(self, tool: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个 tool。

        Args:
            tool: LangChain tool 实例。
            priority: 优先级（越小越先合并）。
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
        """注册一个动态工具工厂。

        Args:
            factory: (ctx) -> tools 列表。
            priority: 优先级（越小越先合并）。
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
            priority: 优先级（越小越先合并）。
            enabled: 是否启用的判断函数。
        """

        self._middlewares.append(
            _RegisteredMiddleware(priority=priority, enabled=enabled, middleware=middleware)
        )

    def add_callback(self, callback: Any, *, priority: int = 0, enabled: EnabledPredicate | None = None) -> None:
        """注册一个 callback handler。

        Args:
            callback: `BaseCallbackHandler` 等回调对象。
            priority: 优先级（越小越先合并）。
            enabled: 是否启用的判断函数。
        """

        self._callbacks.append(_RegisteredCallback(priority=priority, enabled=enabled, callback=callback))

    def iter_tools(self) -> list[_RegisteredTool]:
        return list(self._tools)

    def iter_tool_factories(self) -> list[_RegisteredToolFactory]:
        return list(self._tool_factories)

    def iter_middlewares(self) -> list[_RegisteredMiddleware]:
        return list(self._middlewares)

    def iter_callbacks(self) -> list[_RegisteredCallback]:
        return list(self._callbacks)
