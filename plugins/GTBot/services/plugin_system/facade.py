from __future__ import annotations

from ...ConfigManager import total_config, Processed
from .manager import PluginManager
from .types import PluginBundle, PluginContext


_config: Processed.GeneralConfiguration = total_config.processed_configuration.config
_plugin_manager = PluginManager(plugin_dir=_config.plugin_dir)
_plugin_manager.load()


def get_plugin_manager() -> PluginManager:
    return _plugin_manager


def build_plugin_context(
    *,
    raw_messages: list,
    runtime_context=None,
    trigger_mode: str | None = None,
) -> PluginContext:
    return PluginContext(
        raw_messages=raw_messages,
        runtime_context=runtime_context,
        trigger_mode=trigger_mode,
    )


def build_plugin_bundle(ctx: PluginContext) -> PluginBundle:
    return _plugin_manager.build(ctx)
