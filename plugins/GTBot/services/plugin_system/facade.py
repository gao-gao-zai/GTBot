from __future__ import annotations

from typing import Any

from ...ConfigManager import total_config, Processed
from .manager import PluginManager
from .types import PluginBundle, PluginContext, ResponseStatus


_config: Processed.GeneralConfiguration = total_config.processed_configuration.config
_plugin_manager = PluginManager(plugin_dir=_config.plugin_dir)
_plugin_manager.load()


def get_plugin_manager() -> PluginManager:
    return _plugin_manager


def build_plugin_context(
    *,
    raw_messages: list[Any],
    response_id: str = "",
    response_status: ResponseStatus = "initialized",
    runtime_context=None,
    trigger_mode: str | None = None,
    trigger_meta: dict[str, Any] | None = None,
) -> PluginContext:
    """构建单次请求的插件上下文。

    Args:
        raw_messages: 本轮请求的原始消息列表。
        response_id: 响应唯一 ID。
        response_status: 当前响应流程状态。
        runtime_context: 宿主运行时上下文。
        trigger_mode: 触发模式。
        trigger_meta: 触发元数据。

    Returns:
        PluginContext: 可被插件共享与读写的上下文对象。
    """

    return PluginContext(
        raw_messages=raw_messages,
        response_id=response_id,
        response_status=response_status,
        runtime_context=runtime_context,
        trigger_mode=trigger_mode,
        trigger_meta=dict(trigger_meta or {}),
    )


def build_plugin_bundle(ctx: PluginContext) -> PluginBundle:
    return _plugin_manager.build(ctx)
