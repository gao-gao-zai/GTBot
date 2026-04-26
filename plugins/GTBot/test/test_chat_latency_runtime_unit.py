from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plugins.GTBot.services.chat.runtime as _chat_core
    from plugins.GTBot.services.chat.latency_monitor import get_chat_latency_monitor as _get_chat_latency_monitor
    from plugins.GTBot.services.plugin_system.types import PluginBundle as _PluginBundle

    chat_core: Any | None = _chat_core
    get_chat_latency_monitor: Any | None = _get_chat_latency_monitor
    plugin_bundle_cls: Any | None = _PluginBundle

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    chat_core = None
    get_chat_latency_monitor = None
    plugin_bundle_cls = None
    _IMPORT_ERROR = exc


class _FakeTransport:
    def __init__(self) -> None:
        self.timeout_calls: list[float] = []
        self.error_calls: list[str] = []
        self.processing_calls = 0
        self.completion_calls = 0
        self.timeout_emoji_calls = 0
        self.rejection_calls = 0

    async def handle_processing_emoji(self) -> None:
        self.processing_calls += 1

    async def handle_completion_emoji(self) -> None:
        self.completion_calls += 1

    async def handle_rejection_emoji(self) -> None:
        self.rejection_calls += 1

    async def handle_timeout_emoji(self) -> None:
        self.timeout_emoji_calls += 1

    async def send_timeout(self, timeout_sec: float) -> None:
        self.timeout_calls.append(float(timeout_sec))

    async def send_error(self, error_detail: str) -> None:
        self.error_calls.append(str(error_detail))


class _FakeResponseLockManager:
    def __init__(self, *, acquire_result: bool = True) -> None:
        self.acquire_result = acquire_result
        self.released_sessions: list[str] = []

    def try_acquire(self, _: str) -> bool:
        return self.acquire_result

    def release(self, session_id: str) -> None:
        self.released_sessions.append(session_id)


class _FakeContinuationManager:
    def __init__(self) -> None:
        self.close_calls: list[tuple[str, str]] = []
        self.open_calls: list[dict[str, object]] = []

    async def close_window(self, session_id: str, *, reason: str) -> None:
        self.close_calls.append((session_id, reason))

    async def open_window(self, **kwargs: object) -> None:
        self.open_calls.append(dict(kwargs))


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestChatLatencyRuntimeUnit(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        monitor_factory = get_chat_latency_monitor
        assert monitor_factory is not None
        monitor_factory().reset()

    def tearDown(self) -> None:
        monitor_factory = get_chat_latency_monitor
        assert monitor_factory is not None
        monitor_factory().reset()

    def _build_turn(self) -> Any:
        assert chat_core is not None
        return chat_core.ChatTurn(
            session=chat_core.ChatSession(
                session_id="group:123",
                chat_type="group",
                group_id=123,
                peer_user_id=123,
            ),
            sender_user_id=456,
            sender_name="Alice",
            anchor_message_id=789,
            input_text="hello",
            trigger_mode="group_at",
        )

    async def test_run_chat_turn_should_record_completed_latency_snapshot(self) -> None:
        chat_core_mod = chat_core
        plugin_bundle_cls_local = plugin_bundle_cls
        monitor_factory = get_chat_latency_monitor
        assert chat_core_mod is not None
        assert plugin_bundle_cls_local is not None
        assert monitor_factory is not None
        monitor = monitor_factory()
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_continuation_manager = _FakeContinuationManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")

        async def fake_build_runtime_context(**kwargs: Any) -> Any:
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        async def test_pre_agent_processor(_: Any) -> None:
            return None

        class _FakeAgent:
            async def ainvoke(self, *, input: Any, context: Any, config: Any) -> dict[str, Any]:
                _ = input
                _ = context
                _ = config
                return {"messages": []}

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(chat_core_mod, "get_continuation_manager", return_value=fake_continuation_manager),
            patch.object(chat_core_mod, "_load_turn_messages", AsyncMock(return_value=[SimpleNamespace(user_id=456, send_time=1234.5)])),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(return_value="user")),
            patch.object(
                chat_core_mod,
                "build_plugin_bundle",
                return_value=plugin_bundle_cls_local(
                    pre_agent_processors=cast(
                        Any,
                        [
                            SimpleNamespace(
                                processor=test_pre_agent_processor,
                                wait_until_complete=True,
                            )
                        ],
                    )
                ),
            ),
            patch.object(chat_core_mod, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core_mod, "_should_open_continuation_window", return_value=True),
            patch.object(chat_core_mod, "_find_latest_bot_message_id_for_session", AsyncMock(return_value=9527)),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core_mod.run_chat_turn(
                turn=self._build_turn(),
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        snapshot = monitor.snapshot()
        last_completed = snapshot["last_completed"]
        self.assertEqual(last_completed["outcome"], "completed")
        self.assertIn("processing_emoji", last_completed["stages_ms"])
        self.assertIn("load_turn_messages", last_completed["stages_ms"])
        self.assertIn("parse_streaming_settings", last_completed["stages_ms"])
        self.assertIn("build_runtime_context", last_completed["stages_ms"])
        self.assertIn("build_plugin_context", last_completed["stages_ms"])
        self.assertIn("build_history_text", last_completed["stages_ms"])
        self.assertIn("build_plugin_bundle", last_completed["stages_ms"])
        self.assertIn("create_agent", last_completed["stages_ms"])
        self.assertIn("pre_agent_processors", last_completed["stages_ms"])
        self.assertTrue(
            any(
                stage_name.startswith("pre_agent_processor:")
                and stage_name.endswith("test_pre_agent_processor")
                for stage_name in last_completed["stages_ms"]
            )
        )
        self.assertIn("inject_messages", last_completed["stages_ms"])
        self.assertIn("agent_invoke", last_completed["stages_ms"])
        self.assertIn("completion_emoji", last_completed["stages_ms"])
        self.assertIn("continuation_window", last_completed["stages_ms"])
        self.assertIn("total", last_completed["stages_ms"])
        self.assertEqual(snapshot["inflight_count"], 0)

    async def test_run_chat_turn_should_record_timeout_outcome(self) -> None:
        chat_core_mod = chat_core
        plugin_bundle_cls_local = plugin_bundle_cls
        monitor_factory = get_chat_latency_monitor
        assert chat_core_mod is not None
        assert plugin_bundle_cls_local is not None
        assert monitor_factory is not None
        monitor = monitor_factory()
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_continuation_manager = _FakeContinuationManager()

        async def fake_build_runtime_context(**kwargs: Any) -> Any:
            return SimpleNamespace(response_id=kwargs["response_id"], response_status=kwargs["response_status"])

        class _FakeAgent:
            async def ainvoke(self, *, input: Any, context: Any, config: Any) -> dict[str, Any]:
                _ = input
                _ = context
                _ = config
                assert chat_core_mod is not None
                raise chat_core_mod.AsyncTimeoutError()

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(chat_core_mod, "get_continuation_manager", return_value=fake_continuation_manager),
            patch.object(chat_core_mod, "_load_turn_messages", AsyncMock(return_value=[])),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(return_value="user")),
            patch.object(chat_core_mod, "build_plugin_bundle", return_value=plugin_bundle_cls_local()),
            patch.object(chat_core_mod, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core_mod.run_chat_turn(
                turn=self._build_turn(),
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(monitor.snapshot()["last_completed"]["outcome"], "timed_out")

    async def test_run_chat_turn_should_record_lock_rejected_outcome_without_inflight_leak(self) -> None:
        chat_core_mod = chat_core
        monitor_factory = get_chat_latency_monitor
        assert chat_core_mod is not None
        assert monitor_factory is not None
        monitor = monitor_factory()
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager(acquire_result=False)
        fake_continuation_manager = _FakeContinuationManager()

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(chat_core_mod, "get_continuation_manager", return_value=fake_continuation_manager),
        ):
            await chat_core_mod.run_chat_turn(
                turn=self._build_turn(),
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["inflight_count"], 0)
        self.assertEqual(transport.rejection_calls, 1)
        self.assertEqual(fake_lock_manager.released_sessions, [])
        last_completed = snapshot["last_completed"]
        if last_completed is not None:
            self.assertEqual(last_completed["outcome"], "lock_rejected")


if __name__ == "__main__":
    unittest.main()
