from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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


def _collect_enabled_pre_agent_processors(registry: PluginRegistryT, ctx: PluginContextT) -> list[Any]:
    processors: list[Any] = []
    for item in registry.iter_pre_agent_processors():
        if item.enabled is not None and not bool(item.enabled(ctx)):
            continue
        processors.append(item.processor)
    return processors


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
        self.assertEqual(len(_collect_enabled_pre_agent_processors(registry, group_auto_ctx)), 0)
        self.assertEqual(len(_collect_enabled_pre_agent_processors(registry, group_at_ctx)), 1)

    def test_prewarm_voice_service_context_populates_extra(self) -> None:
        _plugin_registry_cls, plugin_context_cls, _register_plugin = _require_test_runtime()
        from plugins.GTBot.tools.voice_service import tool as voice_tool_mod
        from plugins.GTBot.tools.voice_service.models import ReplyVoiceMessage
        from plugins.GTBot.tools.voice_service.state import SessionVoiceState

        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            runtime_context=SimpleNamespace(
                user_id=1001,
                group_id=2002,
                bot=object(),
                event=SimpleNamespace(reply=SimpleNamespace(message_id=3003)),
            ),
        )
        prefetched_state = SessionVoiceState()
        prefetched_reply_voice = ReplyVoiceMessage(reply_message_id=3003, normalized_wav_path="cached.wav")

        with (
            patch.object(voice_tool_mod, "cleanup_expired_cache", AsyncMock()) as cleanup_cache,
            patch.object(
                voice_tool_mod,
                "get_voice_service_plugin_config",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                voice_tool_mod,
                "get_voice_state_store",
                return_value=SimpleNamespace(get=AsyncMock(return_value=prefetched_state)),
            ),
            patch.object(
                voice_tool_mod,
                "resolve_reply_voice",
                AsyncMock(return_value=prefetched_reply_voice),
            ) as resolve_reply_voice,
        ):
            import asyncio

            asyncio.run(voice_tool_mod.prewarm_voice_service_context(plugin_ctx))

        self.assertTrue(plugin_ctx.extra["voice_service_cache_cleaned"])
        self.assertEqual(plugin_ctx.extra["voice_service_prefetched_session_key"], "group:2002")
        self.assertIs(plugin_ctx.extra["voice_service_prefetched_state"], prefetched_state)
        self.assertIs(plugin_ctx.extra["voice_service_prefetched_reply_voice"], prefetched_reply_voice)
        cleanup_cache.assert_awaited_once()
        resolve_reply_voice.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
