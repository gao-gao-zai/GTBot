from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .types import PluginContext, ResponseStatus


_current_plugin_context: ContextVar[PluginContext | None] = ContextVar(
    "gtbot_current_plugin_context",
    default=None,
)


def get_current_plugin_context() -> PluginContext | None:
    """获取当前协程上下文中的插件上下文。

    Returns:
        PluginContext | None: 当前插件上下文；若未设置则返回 `None`。
    """

    return _current_plugin_context.get()


@contextmanager
def plugin_context_scope(ctx: PluginContext) -> Iterator[None]:
    """在一个作用域内设置当前插件上下文。

    Args:
        ctx: 本次请求的插件上下文。

    Yields:
        None: 供调用方在该作用域内执行插件逻辑。
    """

    token = _current_plugin_context.set(ctx)
    try:
        yield
    finally:
        _current_plugin_context.reset(token)


def set_response_status(ctx: PluginContext, status: ResponseStatus) -> None:
    """同步更新插件上下文与运行时上下文中的响应状态。

    Args:
        ctx: 当前请求对应的插件上下文。
        status: 需要写入的响应流程状态。
    """

    ctx.response_status = status
    runtime_context = getattr(ctx, "runtime_context", None)
    if runtime_context is not None and hasattr(runtime_context, "response_status"):
        setattr(runtime_context, "response_status", status)
