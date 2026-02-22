from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias


@dataclass
class PluginContext:
    """单次调用的插件上下文。

    说明：
        `raw_messages` 主要用于插件读取分析，不要求与 LLM 上下文强绑定。

    Attributes:
        raw_messages: 本次对话原始消息列表（领域模型）。
        runtime_context: 运行期上下文（宿主传入，工具/中间件可通过 ToolRuntime.context 访问）。
        extra: 插件间共享的临时数据容器。
    """

    raw_messages: list[Any]
    runtime_context: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


EnabledPredicate: TypeAlias = Callable[[PluginContext], bool]


@dataclass(frozen=True)
class PluginBundle:
    """本次调用最终装配的插件产物集合。"""

    tools: list[Any] = field(default_factory=list)
    agent_middlewares: list[Any] = field(default_factory=list)
    callbacks: list[Any] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
