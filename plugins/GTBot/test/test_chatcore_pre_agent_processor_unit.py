from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plugins.GTBot.ChatCore as chat_core
    from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context
    from plugins.GTBot.services.plugin_system.types import PluginBundle, PreAgentProcessorBinding

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    chat_core = None  # type: ignore[assignment]
    get_current_plugin_context = None  # type: ignore[assignment]
    PluginBundle = None  # type: ignore[assignment]
    PreAgentProcessorBinding = None  # type: ignore[assignment]
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
    def __init__(self) -> None:
        self.released_sessions: list[str] = []

    def try_acquire(self, _: str) -> bool:
        return True

    def release(self, session_id: str) -> None:
        self.released_sessions.append(session_id)


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestChatCorePreAgentProcessorUnit(unittest.IsolatedAsyncioTestCase):
    async def test_run_pre_agent_processors_runs_in_parallel_and_waits_only_required(self) -> None:
        assert chat_core is not None
        assert PreAgentProcessorBinding is not None

        release_background = asyncio.Event()
        background_started = asyncio.Event()
        timeline: list[str] = []
        plugin_ctx = SimpleNamespace(response_id="resp_1", response_status="initialized", runtime_context=None, extra={})

        async def background_processor(ctx) -> None:  # noqa: ANN001
            timeline.append(f"background_start:{ctx.response_id}")
            background_started.set()
            await release_background.wait()
            timeline.append("background_done")

        async def required_processor(ctx) -> None:  # noqa: ANN001
            timeline.append(f"required_start:{ctx.response_status}")
            await asyncio.sleep(0)
            timeline.append("required_done")

        await chat_core._run_pre_agent_processors(
            plugin_ctx=plugin_ctx,
            processors=[
                PreAgentProcessorBinding(processor=background_processor, wait_until_complete=False),
                PreAgentProcessorBinding(processor=required_processor, wait_until_complete=True),
            ],
        )

        self.assertTrue(background_started.is_set())
        self.assertEqual(plugin_ctx.response_status, "waiting_required_pre_agent_processors")
        self.assertIn("required_done", timeline)
        self.assertNotIn("background_done", timeline)

        release_background.set()
        await asyncio.sleep(0)
        self.assertIn("background_done", timeline)

    async def test_run_pre_agent_processors_raises_when_required_processor_fails(self) -> None:
        assert chat_core is not None
        assert PreAgentProcessorBinding is not None

        plugin_ctx = SimpleNamespace(response_id="resp_fail", response_status="initialized", runtime_context=None, extra={})

        async def failing_processor(_: object) -> None:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await chat_core._run_pre_agent_processors(
                plugin_ctx=plugin_ctx,
                processors=[PreAgentProcessorBinding(processor=failing_processor, wait_until_complete=True)],
            )

        self.assertEqual(plugin_ctx.response_status, "waiting_required_pre_agent_processors")

    async def test_run_chat_turn_exposes_response_id_and_status_to_processors_and_agent(self) -> None:
        assert chat_core is not None
        assert get_current_plugin_context is not None
        assert PluginBundle is not None
        assert PreAgentProcessorBinding is not None

        observed: dict[str, object] = {}
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")
        pre_processor_ran = asyncio.Event()

        async def pre_processor(ctx) -> None:  # noqa: ANN001
            observed["processor_response_id"] = ctx.response_id
            observed["processor_status"] = ctx.response_status
            ctx.extra["from_pre_processor"] = "ready"
            pre_processor_ran.set()

        async def fake_build_runtime_context(**kwargs):  # noqa: ANN003
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        class _FakeAgent:
            async def ainvoke(self, *, input, context, config):  # noqa: ANN001, ANN003
                observed["agent_input"] = input
                observed["agent_response_id"] = context.response_id
                observed["agent_status"] = context.response_status
                observed["processor_completed_before_agent"] = pre_processor_ran.is_set()
                current_plugin_ctx = get_current_plugin_context()
                observed["plugin_extra_seen_by_agent"] = dict(getattr(current_plugin_ctx, "extra", {}))
                return {"messages": []}

        turn = chat_core.ChatTurn(
            session=chat_core.ChatSession(
                session_id="group_123",
                chat_type="group",
                group_id=123,
                peer_user_id=456,
            ),
            sender_user_id=456,
            sender_name="Alice",
            anchor_message_id=789,
            input_text="hello",
            trigger_mode="group_at",
        )

        fake_access_manager = SimpleNamespace(is_allowed=AsyncMock(return_value=True))

        with (
            patch.object(chat_core, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core, "response_lock_manager", fake_lock_manager),
            patch.object(chat_core, "_load_turn_messages", AsyncMock(return_value=[])),
            patch.object(chat_core, "_parse_streaming_settings", return_value=(None, False, 0, 0.0)),
            patch.object(chat_core, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core, "create_group_chat_context", AsyncMock(return_value=[])),
            patch.object(chat_core, "build_plugin_bundle", return_value=PluginBundle(
                pre_agent_processors=[
                    PreAgentProcessorBinding(processor=pre_processor, wait_until_complete=True),
                ]
            )),
            patch.object(chat_core, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core, "convert_openai_to_langchain_messages", side_effect=lambda messages: messages),
            patch.object(chat_core, "get_chat_access_manager", return_value=fake_access_manager),
            patch.object(chat_core.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core.run_chat_turn(
                turn=turn,
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(transport.processing_calls, 1)
        self.assertEqual(transport.completion_calls, 1)
        self.assertEqual(transport.error_calls, [])
        self.assertEqual(fake_lock_manager.released_sessions, ["group_123"])
        self.assertTrue(observed["processor_completed_before_agent"])
        self.assertEqual(observed["agent_status"], "agent_running")
        self.assertEqual(fake_runtime_context.response_status, "completed")
        self.assertEqual(observed["plugin_extra_seen_by_agent"], {"from_pre_processor": "ready"})
        self.assertIsInstance(observed["processor_response_id"], str)
        self.assertTrue(bool(observed["processor_response_id"]))
        self.assertEqual(observed["processor_response_id"], observed["agent_response_id"])
        self.assertEqual(observed["processor_response_id"], fake_runtime_context.response_id)


if __name__ == "__main__":
    unittest.main()
