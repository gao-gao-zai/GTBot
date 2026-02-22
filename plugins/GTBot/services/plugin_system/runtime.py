from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .types import PluginContext


_current_plugin_context: ContextVar[PluginContext | None] = ContextVar(
    "gtbot_current_plugin_context",
    default=None,
)


def get_current_plugin_context() -> PluginContext | None:
    """获取当前协程上下文中的插件上下文。

    Returns:
        当前插件上下文；若未设置则为 None。
    """

    return _current_plugin_context.get()


@contextmanager
def plugin_context_scope(ctx: PluginContext) -> Iterator[None]:
    """在一个作用域内设置当前插件上下文。

    Args:
        ctx: 本次调用的插件上下文。

    Yields:
        None。
    """

    token = _current_plugin_context.set(ctx)
    try:
        yield
    finally:
        _current_plugin_context.reset(token)
