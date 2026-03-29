from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from plugins.GTBot.services.plugin_system.registry import PluginRegistry as _PluginRegistry
    from plugins.GTBot.services.plugin_system.types import PluginContext as _PluginContext

    PluginRegistryT: TypeAlias = _PluginRegistry
    PluginContextT: TypeAlias = _PluginContext
else:
    PluginRegistryT: TypeAlias = Any
    PluginContextT: TypeAlias = Any

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot.services.plugin_system.registry import PluginRegistry
    from plugins.GTBot.services.plugin_system.types import PluginContext
    from plugins.GTBot.tools.voice_service import register

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    PluginRegistry = None  # type: ignore[assignment]
    PluginContext = None  # type: ignore[assignment]
    register = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _collect_enabled_tool_names(registry: PluginRegistryT, ctx: PluginContextT) -> list[str]:
    names: list[str] = []
    for item in registry.iter_tools():
        if item.enabled is not None and not bool(item.enabled(ctx)):
            continue
        names.append(str(getattr(item.tool, "name", "")))
    return names


def _require_test_runtime() -> tuple[Any, Any, Any]:
    """返回测试所需的运行时对象，并在依赖缺失时显式跳过。"""

    if _IMPORT_ERROR is not None or PluginRegistry is None or PluginContext is None or register is None:
        raise unittest.SkipTest(f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
    return PluginRegistry, PluginContext, register


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestVoiceServicePluginUnit(unittest.TestCase):
    def test_voice_tools_disabled_for_group_auto(self) -> None:
        plugin_registry_cls, plugin_context_cls, register_plugin = _require_test_runtime()
        registry = plugin_registry_cls()
        register_plugin(registry)

        group_auto_ctx = plugin_context_cls(raw_messages=[], trigger_mode="group_auto")
        group_at_ctx = plugin_context_cls(raw_messages=[], trigger_mode="group_at")

        self.assertEqual(_collect_enabled_tool_names(registry, group_auto_ctx), [])
        self.assertEqual(
            _collect_enabled_tool_names(registry, group_at_ctx),
            ["voice_recognize", "voice_synthesize"],
        )


if __name__ == "__main__":
    unittest.main()
