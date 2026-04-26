from __future__ import annotations

import asyncio
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
    from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context as _get_current_plugin_context
    from plugins.GTBot.services.plugin_system.types import (
        PluginBundle as _PluginBundle,
        PluginContext as _PluginContext,
        PreAgentMessageAppenderBinding as _PreAgentMessageAppenderBinding,
        PreAgentMessageInjectorBinding as _PreAgentMessageInjectorBinding,
        PreAgentProcessorBinding as _PreAgentProcessorBinding,
    )
    from langchain_core.messages import BaseMessage as _BaseMessage, HumanMessage as _HumanMessage, SystemMessage as _SystemMessage

    chat_core: Any | None = _chat_core
    get_current_plugin_context: Any | None = _get_current_plugin_context
    plugin_bundle_cls: Any | None = _PluginBundle
    plugin_context_cls: Any | None = _PluginContext
    pre_agent_message_appender_binding_cls: Any | None = _PreAgentMessageAppenderBinding
    pre_agent_message_injector_binding_cls: Any | None = _PreAgentMessageInjectorBinding
    pre_agent_processor_binding_cls: Any | None = _PreAgentProcessorBinding
    base_message_cls: Any | None = _BaseMessage
    human_message_cls: Any | None = _HumanMessage
    system_message_cls: Any | None = _SystemMessage

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    chat_core = None
    get_current_plugin_context = None
    plugin_bundle_cls = None
    plugin_context_cls = None
    pre_agent_message_appender_binding_cls = None
    pre_agent_message_injector_binding_cls = None
    pre_agent_processor_binding_cls = None
    base_message_cls = None
    human_message_cls = None
    system_message_cls = None
    _IMPORT_ERROR = exc


class _FakeTransport:
    def __init__(self) -> None:
        self.timeout_calls: list[float] = []
        self.error_calls: list[str] = []
        self.sent_messages: list[list[Any]] = []
        self.processing_calls = 0
        self.completion_calls = 0
        self.timeout_emoji_calls = 0
        self.rejection_calls = 0
        self.silent_emoji_calls = 0

    async def handle_processing_emoji(self) -> None:
        self.processing_calls += 1

    async def handle_completion_emoji(self) -> None:
        self.completion_calls += 1

    async def handle_rejection_emoji(self) -> None:
        self.rejection_calls += 1

    async def handle_timeout_emoji(self) -> None:
        self.timeout_emoji_calls += 1

    async def handle_silent_emoji(self) -> None:
        self.silent_emoji_calls += 1

    async def send_timeout(self, timeout_sec: float) -> None:
        self.timeout_calls.append(float(timeout_sec))

    async def send_error(self, error_detail: str) -> None:
        self.error_calls.append(str(error_detail))

    async def send_messages(self, messages: Any, interval: float = 0.2) -> None:
        self.sent_messages.append(list(messages))


class _FakeResponseLockManager:
    def __init__(self) -> None:
        self.released_sessions: list[str] = []

    def try_acquire(self, _: str) -> bool:
        return True

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


def _require_test_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """返回测试所需的运行时对象，并在依赖缺失时显式跳过。"""

    if (
        _IMPORT_ERROR is not None
        or chat_core is None
        or get_current_plugin_context is None
        or plugin_bundle_cls is None
        or plugin_context_cls is None
        or pre_agent_message_appender_binding_cls is None
        or pre_agent_message_injector_binding_cls is None
        or pre_agent_processor_binding_cls is None
        or base_message_cls is None
        or human_message_cls is None
        or system_message_cls is None
    ):
        raise unittest.SkipTest(f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")

    return (
        chat_core,
        get_current_plugin_context,
        plugin_bundle_cls,
        plugin_context_cls,
        pre_agent_message_appender_binding_cls,
        pre_agent_message_injector_binding_cls,
        pre_agent_processor_binding_cls,
        base_message_cls,
        human_message_cls,
        system_message_cls,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestChatCorePreAgentProcessorUnit(unittest.IsolatedAsyncioTestCase):
    async def test_message_injection_stage_serializes_injectors_and_honors_prepend(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context,
            plugin_bundle_cls,
            plugin_context_cls,
            pre_agent_message_appender_binding_cls,
            pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            base_message_cls,
            human_message_cls,
            system_message_cls,
        ) = _require_test_runtime()

        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            response_id="resp_inject",
            response_status="initialized",
            runtime_context=None,
        )
        observed_messages: list[list[str]] = []

        async def injector_one(ctx: Any, messages: list[Any]) -> list[Any]:
            observed_messages.append([type(item).__name__ for item in messages])
            return list(messages) + [human_message_cls(content="inject_one")]

        async def injector_two(ctx: Any, messages: list[Any]) -> list[Any]:
            observed_messages.append([getattr(item, "content", "") for item in messages])
            return [human_message_cls(content="inject_two")] + list(messages)

        async def prepend_appender(ctx: Any) -> object:
            return human_message_cls(content="prepended")

        async def append_appender(ctx: Any) -> object:
            return cast(list[Any], [human_message_cls(content="appended")])

        initial_messages = [
            system_message_cls(content="sys"),
            human_message_cls(content="user"),
        ]

        result = await chat_core_mod._run_pre_agent_message_injection_stage(
            plugin_ctx=plugin_ctx,
            chat_context=initial_messages,
            plugin_bundle=plugin_bundle_cls(
                pre_agent_message_injectors=[
                    pre_agent_message_injector_binding_cls(injector=injector_one),
                    pre_agent_message_injector_binding_cls(injector=injector_two),
                ],
                pre_agent_message_appenders=[
                    pre_agent_message_appender_binding_cls(appender=prepend_appender, position="prepend"),
                    pre_agent_message_appender_binding_cls(appender=append_appender, position="append"),
                ],
            ),
        )

        self.assertEqual(plugin_ctx.response_status, "injecting_messages")
        self.assertEqual(observed_messages[0], ["SystemMessage", "HumanMessage"])
        self.assertIn("inject_one", observed_messages[1])
        self.assertEqual([getattr(item, "content", "") for item in result], ["sys", "prepended", "inject_two", "user", "inject_one", "appended"])

    async def test_message_injection_stage_logs_and_continues_on_injector_error(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context,
            plugin_bundle_cls,
            plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            response_id="resp_log",
            response_status="initialized",
            runtime_context=None,
        )

        async def bad_injector(ctx: Any, messages: list[Any]) -> list[Any]:
            raise RuntimeError("bad injector")

        async def good_injector(ctx: Any, messages: list[Any]) -> list[Any]:
            return list(messages) + [human_message_cls(content="ok")]

        result = await chat_core_mod._run_pre_agent_message_injection_stage(
            plugin_ctx=plugin_ctx,
            chat_context=[human_message_cls(content="user")],
            plugin_bundle=plugin_bundle_cls(
                pre_agent_message_injectors=[
                    pre_agent_message_injector_binding_cls(injector=bad_injector),
                    pre_agent_message_injector_binding_cls(injector=good_injector),
                ],
            ),
        )

        self.assertEqual([getattr(item, "content", "") for item in result], ["user", "ok"])

    async def test_message_injection_stage_does_not_mutate_raw_messages(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context,
            plugin_bundle_cls,
            plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        raw_message = SimpleNamespace(content="[CQ:image,file=demo.png]")
        plugin_ctx = plugin_context_cls(
            raw_messages=[raw_message],
            response_id="resp_raw",
            response_status="initialized",
            runtime_context=SimpleNamespace(raw_messages=[raw_message]),
        )

        async def rewrite_injector(ctx: Any, messages: list[Any]) -> list[Any]:
            return [human_message_cls(content="[CQ:image,file=demo.png,title=cat,file_size=1]")]

        result = await chat_core_mod._run_pre_agent_message_injection_stage(
            plugin_ctx=plugin_ctx,
            chat_context=[human_message_cls(content="[CQ:image,file=demo.png]")],
            plugin_bundle=plugin_bundle_cls(
                pre_agent_message_injectors=[
                    pre_agent_message_injector_binding_cls(injector=rewrite_injector),
                ],
            ),
        )

        self.assertEqual(
            [getattr(item, "content", "") for item in result],
            ["[CQ:image,file=demo.png,title=cat,file_size=1]"],
        )
        self.assertEqual(raw_message.content, "[CQ:image,file=demo.png]")

    async def test_run_pre_agent_processors_runs_in_parallel_and_waits_only_required(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context,
            _plugin_bundle_cls,
            plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        release_background = asyncio.Event()
        background_started = asyncio.Event()
        timeline: list[str] = []
        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            response_id="resp_1",
            response_status="initialized",
            runtime_context=None,
        )

        async def background_processor(ctx) -> None:  # noqa: ANN001
            timeline.append(f"background_start:{ctx.response_id}")
            background_started.set()
            await release_background.wait()
            timeline.append("background_done")

        async def required_processor(ctx) -> None:  # noqa: ANN001
            timeline.append(f"required_start:{ctx.response_status}")
            await asyncio.sleep(0)
            timeline.append("required_done")

        await chat_core_mod._run_pre_agent_processors(
            plugin_ctx=plugin_ctx,
            processors=[
                pre_agent_processor_binding_cls(processor=background_processor, wait_until_complete=False),
                pre_agent_processor_binding_cls(processor=required_processor, wait_until_complete=True),
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
        (
            chat_core_mod,
            _get_current_plugin_context,
            _plugin_bundle_cls,
            plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            response_id="resp_fail",
            response_status="initialized",
            runtime_context=None,
        )

        async def failing_processor(_: object) -> None:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await chat_core_mod._run_pre_agent_processors(
                plugin_ctx=plugin_ctx,
                processors=[pre_agent_processor_binding_cls(processor=failing_processor, wait_until_complete=True)],
            )

        self.assertEqual(plugin_ctx.response_status, "waiting_required_pre_agent_processors")

    async def test_run_chat_turn_exposes_response_id_and_status_to_processors_and_agent(self) -> None:
        (
            chat_core_mod,
            get_current_plugin_context_fn,
            plugin_bundle_cls,
            _plugin_context_cls,
            pre_agent_message_appender_binding_cls,
            pre_agent_message_injector_binding_cls,
            pre_agent_processor_binding_cls,
            _base_message_cls,
            human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        observed: dict[str, object] = {}
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")
        pre_processor_ran = asyncio.Event()
        fake_callback = object()

        async def pre_processor(ctx) -> None:  # noqa: ANN001
            observed["processor_response_id"] = ctx.response_id
            observed["processor_status"] = ctx.response_status
            ctx.extra["from_pre_processor"] = "ready"
            pre_processor_ran.set()

        async def inject_messages(ctx: Any, messages: list[Any]) -> list[Any]:
            observed["inject_status"] = ctx.response_status
            observed["inject_input_types"] = [type(item).__name__ for item in messages]
            return list(messages) + [human_message_cls(content="inject_tail")]

        async def append_messages(ctx: Any) -> object:
            observed["append_status"] = ctx.response_status
            return human_message_cls(content="prepend_hint")

        async def fake_build_runtime_context(**kwargs):  # noqa: ANN003
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        class _FakeAgent:
            async def ainvoke(self, *, input, context, config):  # noqa: ANN001, ANN003
                observed["agent_input"] = input
                observed["agent_response_id"] = context.response_id
                observed["agent_status"] = context.response_status
                observed["agent_callbacks"] = list(config.get("callbacks", []))
                observed["processor_completed_before_agent"] = pre_processor_ran.is_set()
                current_plugin_ctx = get_current_plugin_context_fn()
                observed["plugin_extra_seen_by_agent"] = dict(getattr(current_plugin_ctx, "extra", {}))
                return {"messages": []}

        turn = chat_core_mod.ChatTurn(
            session=chat_core_mod.ChatSession(
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
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(
                chat_core_mod,
                "_load_turn_messages",
                AsyncMock(
                    return_value=[
                        SimpleNamespace(
                            user_id=456,
                            send_time=1234.5,
                        )
                    ]
                ),
            ),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(return_value="user")),
            patch.object(chat_core_mod, "build_plugin_bundle", return_value=plugin_bundle_cls(
                pre_agent_processors=[
                    pre_agent_processor_binding_cls(processor=pre_processor, wait_until_complete=True),
                ],
                pre_agent_message_injectors=[
                    pre_agent_message_injector_binding_cls(injector=inject_messages),
                ],
                pre_agent_message_appenders=[
                    pre_agent_message_appender_binding_cls(appender=append_messages, position="prepend"),
                ],
                callbacks=[fake_callback],
            )),
            patch.object(chat_core_mod, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core_mod, "get_chat_access_manager", return_value=fake_access_manager),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core_mod.run_chat_turn(
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
        self.assertEqual(observed["inject_status"], "injecting_messages")
        self.assertEqual(observed["append_status"], "injecting_messages")
        self.assertEqual(observed["inject_input_types"], ["SystemMessage", "HumanMessage"])
        self.assertEqual(observed["agent_status"], "agent_running")
        self.assertEqual(observed["agent_callbacks"], [fake_callback])
        self.assertEqual(fake_runtime_context.response_status, "completed")
        self.assertEqual(observed["plugin_extra_seen_by_agent"], {"from_pre_processor": "ready"})
        self.assertIsInstance(observed["processor_response_id"], str)
        self.assertTrue(bool(observed["processor_response_id"]))
        self.assertEqual(observed["processor_response_id"], observed["agent_response_id"])
        self.assertEqual(observed["processor_response_id"], fake_runtime_context.response_id)
        agent_input = cast(dict[str, Any], observed["agent_input"])
        expected_messages = [
            chat_core_mod.config.chat_model.prompt,
            "prepend_hint",
            "<messages>\nuser\n</messages>",
            "inject_tail",
        ]
        self.assertEqual(
            [getattr(item, "content", "") for item in agent_input["messages"]],
            expected_messages,
        )

    async def test_run_chat_turn_waits_required_processors_before_message_injection(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            pre_agent_message_injector_binding_cls,
            pre_agent_processor_binding_cls,
            _base_message_cls,
            human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        observed: dict[str, object] = {}
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")

        async def required_processor(ctx: Any) -> None:
            await asyncio.sleep(0)
            ctx.extra["long_memory_related_memories"] = "ready"
            observed["processor_status_when_done"] = ctx.response_status

        async def inject_messages(ctx: Any, messages: list[Any]) -> list[Any]:
            observed["inject_seen_related"] = ctx.extra.get("long_memory_related_memories")
            observed["inject_status"] = ctx.response_status
            return list(messages) + [human_message_cls(content=str(observed["inject_seen_related"]))]

        async def fake_build_runtime_context(**kwargs: Any) -> Any:  # noqa: ANN003
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        class _FakeAgent:
            async def ainvoke(self, *, input: Any, context: Any, config: Any) -> dict[str, Any]:  # noqa: ANN003
                observed["agent_input"] = input
                observed["agent_status"] = context.response_status
                _ = config
                return {"messages": []}

        turn = chat_core_mod.ChatTurn(
            session=chat_core_mod.ChatSession(
                session_id="group_789",
                chat_type="group",
                group_id=789,
                peer_user_id=456,
            ),
            sender_user_id=456,
            sender_name="Alice",
            anchor_message_id=790,
            input_text="hello",
            trigger_mode="group_at",
        )

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(
                chat_core_mod,
                "_load_turn_messages",
                AsyncMock(
                    return_value=[
                        SimpleNamespace(
                            user_id=456,
                            send_time=1234.5,
                        )
                    ]
                ),
            ),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(return_value="user")),
            patch.object(
                chat_core_mod,
                "build_plugin_bundle",
                return_value=plugin_bundle_cls(
                    pre_agent_processors=[
                        pre_agent_processor_binding_cls(processor=required_processor, wait_until_complete=True),
                    ],
                    pre_agent_message_injectors=[
                        pre_agent_message_injector_binding_cls(injector=inject_messages),
                    ],
                ),
            ),
            patch.object(chat_core_mod, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core_mod.run_chat_turn(
                turn=turn,
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(observed["processor_status_when_done"], "waiting_required_pre_agent_processors")
        self.assertEqual(observed["inject_status"], "injecting_messages")
        self.assertEqual(observed["inject_seen_related"], "ready")
        self.assertEqual(observed["agent_status"], "agent_running")
        agent_input = cast(dict[str, Any], observed["agent_input"])
        self.assertEqual(
            [getattr(item, "content", "") for item in agent_input["messages"]],
            [
                chat_core_mod.config.chat_model.prompt,
                "<messages>\nuser\n</messages>",
                "ready",
            ],
        )

    async def test_run_chat_turn_starts_agent_build_before_history_format_finishes(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        import time as time_module

        timeline: list[str] = []
        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")

        async def fake_format_messages_for_chat_context(**kwargs: Any) -> str:  # noqa: ANN003
            timeline.append("history_start")
            await asyncio.sleep(0.05)
            timeline.append("history_done")
            return "user"

        async def fake_build_runtime_context(**kwargs: Any) -> Any:  # noqa: ANN003
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        def fake_build_plugin_bundle(_: Any) -> Any:
            timeline.append("bundle_start")
            time_module.sleep(0.01)
            timeline.append("bundle_done")
            return plugin_bundle_cls()

        class _FakeAgent:
            async def ainvoke(self, *, input: Any, context: Any, config: Any) -> dict[str, Any]:  # noqa: ANN003
                timeline.append("agent_invoke")
                return {"messages": []}

        def fake_create_group_chat_agent(*, runtime_context: Any, plugin_bundle: Any) -> Any:
            _ = runtime_context
            _ = plugin_bundle
            timeline.append("agent_build_start")
            return _FakeAgent()

        turn = chat_core_mod.ChatTurn(
            session=chat_core_mod.ChatSession(
                session_id="group_456",
                chat_type="group",
                group_id=456,
                peer_user_id=789,
            ),
            sender_user_id=789,
            sender_name="Bob",
            anchor_message_id=123,
            input_text="hello",
            trigger_mode="group_at",
        )

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(chat_core_mod, "_load_turn_messages", AsyncMock(return_value=[])),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(side_effect=fake_format_messages_for_chat_context)),
            patch.object(chat_core_mod, "build_plugin_bundle", side_effect=fake_build_plugin_bundle),
            patch.object(chat_core_mod, "create_group_chat_agent", side_effect=fake_create_group_chat_agent),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
        ):
            await chat_core_mod.run_chat_turn(
                turn=turn,
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        self.assertLess(timeline.index("bundle_done"), timeline.index("history_done"))
        self.assertLess(timeline.index("history_done"), timeline.index("agent_invoke"))

    async def test_run_chat_turn_closes_old_continuation_window_and_opens_new_one_on_success(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        transport = _FakeTransport()
        fake_lock_manager = _FakeResponseLockManager()
        fake_runtime_context = SimpleNamespace(response_id="", response_status="")
        fake_continuation_manager = _FakeContinuationManager()

        async def fake_build_runtime_context(**kwargs: Any) -> Any:  # noqa: ANN003
            fake_runtime_context.response_id = kwargs["response_id"]
            fake_runtime_context.response_status = kwargs["response_status"]
            return fake_runtime_context

        class _FakeAgent:
            async def ainvoke(self, *, input: Any, context: Any, config: Any) -> dict[str, Any]:  # noqa: ANN003
                _ = input
                _ = context
                _ = config
                return {"messages": []}

        turn = chat_core_mod.ChatTurn(
            session=chat_core_mod.ChatSession(
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

        with (
            patch.object(chat_core_mod, "_resolve_chat_access_target", return_value=None),
            patch.object(chat_core_mod, "response_lock_manager", fake_lock_manager),
            patch.object(
                chat_core_mod,
                "_load_turn_messages",
                AsyncMock(
                    return_value=[
                        SimpleNamespace(
                            user_id=456,
                            send_time=1234.5,
                        )
                    ]
                ),
            ),
            patch.object(chat_core_mod, "_parse_streaming_settings", return_value=(None, False, 0, 0.0, True)),
            patch.object(chat_core_mod, "_build_runtime_context", AsyncMock(side_effect=fake_build_runtime_context)),
            patch.object(chat_core_mod, "_format_messages_for_chat_context", AsyncMock(return_value="user")),
            patch.object(chat_core_mod, "build_plugin_bundle", return_value=plugin_bundle_cls()),
            patch.object(chat_core_mod, "create_group_chat_agent", return_value=_FakeAgent()),
            patch.object(chat_core_mod.config.chat_model, "api_timeout_sec", 0),
            patch.object(chat_core_mod, "get_continuation_manager", return_value=fake_continuation_manager),
            patch.object(chat_core_mod, "_find_latest_bot_message_id_for_session", AsyncMock(return_value=9527)),
            patch.object(chat_core_mod, "_should_open_continuation_window", return_value=True),
        ):
            await chat_core_mod.run_chat_turn(
                turn=turn,
                transport=transport,  # type: ignore[arg-type]
                bot=SimpleNamespace(self_id="114514"),  # type: ignore[arg-type]
                msg_mg=object(),  # type: ignore[arg-type]
                cache=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(fake_continuation_manager.close_calls, [("group:123", "main_chain_started")])
        self.assertEqual(len(fake_continuation_manager.open_calls), 1)
        self.assertEqual(fake_continuation_manager.open_calls[0]["session_id"], "group:123")
        self.assertEqual(fake_continuation_manager.open_calls[0]["group_id"], 123)
        self.assertEqual(fake_continuation_manager.open_calls[0]["source_trigger_mode"], "group_at")
        self.assertEqual(fake_continuation_manager.open_calls[0]["last_bot_message_id"], 9527)
        self.assertEqual(fake_continuation_manager.open_calls[0]["trigger_started_at"], 1234.5)

    def test_dashscope_openai_compatible_detection_and_fallback_scope(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        self.assertTrue(
            chat_core_mod._is_dashscope_openai_compatible_base_url(
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        )
        self.assertFalse(
            chat_core_mod._is_dashscope_openai_compatible_base_url(
                "https://dashscope.aliyuncs.com/api/v1"
            )
        )
        self.assertFalse(
            chat_core_mod._is_dashscope_openai_compatible_base_url(
                "https://api.deepseek.com/v1"
            )
        )

        self.assertTrue(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="openai_compatible",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming_enabled=True,
                process_tool_call_deltas=True,
            )
        )
        self.assertFalse(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="openai_compatible",
                base_url="https://api.deepseek.com/v1",
                streaming_enabled=True,
                process_tool_call_deltas=True,
            )
        )
        self.assertTrue(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="dashscope",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming_enabled=True,
                process_tool_call_deltas=True,
            )
        )
        self.assertTrue(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="dashscope",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                streaming_enabled=True,
                process_tool_call_deltas=True,
            )
        )
        self.assertFalse(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="openai_compatible",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming_enabled=False,
                process_tool_call_deltas=True,
            )
        )
        self.assertFalse(
            chat_core_mod._should_force_non_streaming_tool_agent(
                provider_type="openai_compatible",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming_enabled=True,
                process_tool_call_deltas=False,
            )
        )

    def test_resolve_continuation_window_start_time_prefers_inherited_history_start(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        turn = chat_core_mod.ChatTurn(
            session=chat_core_mod.ChatSession(
                session_id="group:123",
                chat_type="group",
                group_id=123,
                peer_user_id=123,
            ),
            sender_user_id=456,
            trigger_mode="group_continuation",
            trigger_meta={"continuation_history_started_at": 111.5},
        )

        resolved = chat_core_mod._resolve_continuation_window_start_time(
            turn=turn,
            relevant_messages=[
                SimpleNamespace(user_id=456, send_time=999.0),
            ],
            self_user_id=114514,
        )

        self.assertEqual(resolved, 111.5)

    def test_create_group_chat_agent_uses_runtime_streaming_flag(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        captured: dict[str, Any] = {}
        runtime_context = SimpleNamespace(chat_type="group", streaming_enabled=False)

        def fake_build_chat_model(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        with (
            patch.dict(
                chat_core_mod.agent_cache_info,
                {
                    "model": None,
                    "provider_type": None,
                    "model_id": None,
                    "base_url": None,
                    "api_key": None,
                    "extra_body": None,
                    "streaming": None,
                },
                clear=True,
            ),
            patch.object(chat_core_mod.config.chat_model, "provider_type", "openai_compatible"),
            patch.object(chat_core_mod.config.chat_model, "model_id", "test-model"),
            patch.object(chat_core_mod.config.chat_model, "base_url", "https://example.test/v1"),
            patch.object(chat_core_mod.config.chat_model, "api_key", "secret"),
            patch.object(chat_core_mod.config.chat_model, "parameters", {"streaming": True, "temperature": 0.3}),
            patch.object(chat_core_mod, "build_chat_model", side_effect=fake_build_chat_model),
            patch.object(chat_core_mod, "create_agent", return_value="fake-agent"),
        ):
            agent = chat_core_mod.create_group_chat_agent(
                runtime_context=runtime_context,
                plugin_bundle=plugin_bundle_cls(),
            )

        self.assertEqual(agent, "fake-agent")
        self.assertFalse(captured["streaming"])
        self.assertEqual(captured["model_parameters"], {"temperature": 0.3})

    def test_parse_streaming_settings_extracts_process_tool_call_deltas_flag(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        (
            clean_model_kwargs,
            streaming_enabled,
            stream_chunk_chars,
            stream_flush_interval_sec,
            process_tool_call_deltas,
        ) = chat_core_mod._parse_streaming_settings(
            {
                "stream": True,
                "stream_chunk_chars": 88,
                "stream_flush_interval_sec": 1.2,
                "process_tool_call_deltas": "false",
                "temperature": 0.5,
            }
        )

        self.assertEqual(clean_model_kwargs, {"temperature": 0.5})
        self.assertTrue(streaming_enabled)
        self.assertEqual(stream_chunk_chars, 88)
        self.assertEqual(stream_flush_interval_sec, 1.2)
        self.assertFalse(process_tool_call_deltas)

    def test_extract_text_from_responses_api_content_blocks(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        content = [
            {"type": "reasoning", "reasoning": "hidden"},
            {"type": "text", "text": "<msg>你好</msg>"},
            {"type": "tool_call", "name": "send_group_message", "args": {"text": "skip"}},
            {"type": "message", "content": [{"type": "output_text", "text": "<msg>嵌套</msg>"}]},
            {"type": "text", "text": {"value": "<meme>猫</meme>"}},
        ]

        self.assertEqual(
            chat_core_mod._extract_text_from_message_content(content),
            "<msg>你好</msg><msg>嵌套</msg><meme>猫</meme>",
        )

    async def test_streaming_parser_recovers_from_unclosed_thinking_before_msg(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        class FakeAgent:
            async def astream_events(self, **_: Any) -> Any:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=[
                                {
                                    "type": "text",
                                    "text": "<thinking>commentary without closing tag",
                                }
                            ]
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=[
                                {
                                    "type": "text",
                                    "text": "的。</msg><msg>搜到了喵</msg>",
                                }
                            ]
                        )
                    },
                }
                yield {"event": "on_chain_end", "data": {"output": {"messages": []}}}

        transport = _FakeTransport()
        runtime_context = SimpleNamespace(
            transport=transport,
            chat_type="group",
            message_id=123,
            bot=object(),
            session_id="group:123",
        )

        await chat_core_mod._invoke_agent_with_streaming_to_queue(
            agent=FakeAgent(),
            chat_context=[],
            runtime_context=runtime_context,
            response_id="test-response-id",
            invoke_config=None,
            stream_chunk_chars=20,
            stream_flush_interval_sec=0.1,
            process_tool_call_deltas=True,
        )

        self.assertEqual(transport.sent_messages, [["搜到了喵"]])

    async def test_streaming_xml_parser_dispatches_completed_msg_blocks(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        class FakeAgent:
            async def astream_events(self, **_: Any) -> Any:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=[
                                {
                                    "type": "text",
                                    "text": "<thinking>先想一下</thinking><msg>第一条</msg>",
                                }
                            ]
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=[
                                {
                                    "type": "text",
                                    "text": "<msg priority=\"high\">第二条</msg><silent />",
                                }
                            ]
                        )
                    },
                }
                yield {"event": "on_chain_end", "data": {"output": {"messages": []}}}

        transport = _FakeTransport()
        runtime_context = SimpleNamespace(
            transport=transport,
            chat_type="group",
            message_id=123,
            bot=object(),
            session_id="group:123",
        )

        await chat_core_mod._invoke_agent_with_streaming_to_queue(
            agent=FakeAgent(),
            chat_context=[],
            runtime_context=runtime_context,
            response_id="stream-xml-response-id",
            invoke_config=None,
            stream_chunk_chars=20,
            stream_flush_interval_sec=0.1,
            process_tool_call_deltas=True,
        )

        self.assertEqual(transport.sent_messages, [["第一条"], ["第二条"]])

    def test_parse_send_message_blocks_supports_self_closing_silent(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        messages = chat_core_mod.parse_send_message_blocks(
            '<msg priority="high"> 你好 </msg><silent /><msg>世界</msg>'
        )

        self.assertEqual(messages, ["你好", "世界"])

    async def test_parse_send_output_blocks_uses_xml_parser_for_msg_and_meme(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        with patch(
            "plugins.GTBot.tools.meme.tool.resolve_meme_title_to_cq",
            new=AsyncMock(return_value="[CQ:image,file=cat.png]"),
        ) as mocked_resolver:
            messages = await chat_core_mod.parse_send_output_blocks(
                '<msg>你好</msg><silent /><meme title="cat">猫猫震惊</meme>'
            )

        self.assertEqual(messages, ["你好", "[CQ:image,file=cat.png]"])
        mocked_resolver.assert_awaited_once_with("猫猫震惊")

    def test_extract_note_tags_preserves_silent_when_using_xml_parser(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        notes, remaining = chat_core_mod.extract_note_tags(
            '<silent /><note>这轮不发言</note><msg>不会被发送</msg>'
        )

        self.assertEqual(notes, ["这轮不发言"])
        self.assertEqual(remaining, "<silent /><msg>不会被发送</msg>")

    async def test_process_assistant_direct_output_silent_skips_error_log(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        response = {
            "messages": [
                chat_core_mod.AIMessage(content="<silent /><note>当前不需要我发言</note>")
            ]
        }

        with (
            patch.object(chat_core_mod, "logger") as mocked_logger,
            patch.object(chat_core_mod, "_enqueue_group_messages", new=AsyncMock()) as mocked_enqueue,
        ):
            await chat_core_mod.process_assistant_direct_output(
                response=response,
                bot=object(),
                group_id=123,
                message_manager=object(),
                cache=object(),
            )

        mocked_enqueue.assert_not_called()
        mocked_logger.error.assert_not_called()

    async def test_direct_assistant_output_middleware_silent_triggers_transport_emoji(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        middleware = chat_core_mod.DirectAssistantOutputMiddleware()
        transport = _FakeTransport()
        state = {
            "messages": [
                chat_core_mod.AIMessage(content="<thinking>观察一下</thinking><silent /><note>这轮不插话</note>")
            ]
        }
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                transport=transport,
                streaming_enabled=False,
                session_id="group:123",
            )
        )

        result = middleware.after_model(state, runtime)
        self.assertIsNone(result)

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(transport.silent_emoji_calls, 1)
        self.assertEqual(transport.sent_messages, [])

    async def test_streaming_xml_parser_silent_triggers_transport_emoji(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        class FakeAgent:
            async def astream_events(self, **_: Any) -> Any:
                yield {
                    "event": "on_chat_model_stream",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=[
                                {
                                    "type": "text",
                                    "text": "<thinking>先判断</thinking><silent />",
                                }
                            ]
                        )
                    },
                }
                yield {"event": "on_chain_end", "data": {"output": {"messages": []}}}

        transport = _FakeTransport()
        runtime_context = SimpleNamespace(
            transport=transport,
            chat_type="group",
            message_id=123,
            bot=object(),
            session_id="group:123",
        )

        await chat_core_mod._invoke_agent_with_streaming_to_queue(
            agent=FakeAgent(),
            chat_context=[],
            runtime_context=runtime_context,
            response_id="stream-silent-response-id",
            invoke_config=None,
            stream_chunk_chars=20,
            stream_flush_interval_sec=0.1,
            process_tool_call_deltas=True,
        )

        self.assertEqual(transport.silent_emoji_calls, 1)
        self.assertEqual(transport.sent_messages, [])

    def test_recover_tool_calls_from_additional_kwargs_with_missing_closing_brace(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        message = chat_core_mod.AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Hangzhou"',
                        },
                    }
                ]
            },
            response_metadata={"finish_reason": "tool_calls"},
        )

        recovered = chat_core_mod._recover_tool_calls_from_message(message)

        self.assertEqual(
            recovered,
            [
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "args": {"location": "Hangzhou"},
                    "type": "tool_call",
                }
            ],
        )

    def test_tool_call_recovery_middleware_promotes_invalid_tool_calls(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        middleware = chat_core_mod.ToolCallRecoveryMiddleware()
        ai_message = chat_core_mod.AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {
                    "id": "call_2",
                    "name": "lookup_user",
                    "args": '{"user_id": 42',
                    "error": "missing closing brace",
                }
            ],
            response_metadata={"finish_reason": "tool_calls"},
        )
        state = {"messages": [ai_message]}
        runtime = SimpleNamespace(context=SimpleNamespace(session_id="session_test"))

        result = middleware.after_model(state, runtime)

        self.assertIsNone(result)
        repaired = state["messages"][-1]
        self.assertEqual(
            repaired.tool_calls,
            [
                {
                    "id": "call_2",
                    "name": "lookup_user",
                    "args": {"user_id": 42},
                    "type": "tool_call",
                }
            ],
        )
        self.assertEqual(repaired.invalid_tool_calls, [])

    def test_format_agent_raw_responses_for_logging_extracts_raw_response(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        ai_message = chat_core_mod.AIMessage(
            content="hello",
            additional_kwargs={
                "raw_response": {
                    "provider_type": "openai_compatible",
                    "api_style": "chat_completions",
                    "status_code": 200,
                    "headers": {"x-request-id": "req_123"},
                    "body_json": {"id": "chatcmpl_1"},
                    "body_text": None,
                    "request_id": "req_123",
                }
            },
            response_metadata={"finish_reason": "stop", "raw_response_available": True},
        )

        with patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", True):
            formatted = chat_core_mod.format_agent_raw_responses_for_logging({"messages": [ai_message]})

        self.assertIn("req_123", formatted)
        self.assertIn("chatcmpl_1", formatted)
        self.assertIn("openai_compatible", formatted)

    def test_format_agent_raw_responses_for_logging_truncates_long_fields(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        long_value = "x" * 350
        ai_message = chat_core_mod.AIMessage(
            content="hello",
            additional_kwargs={
                "raw_response": {
                    "provider_type": "openai_compatible",
                    "api_style": "chat_completions",
                    "status_code": 200,
                    "headers": {"x-long-header": long_value},
                    "body_json": {"long_field": long_value},
                    "body_text": long_value,
                    "request_id": "req_long",
                }
            },
            response_metadata={"finish_reason": "stop", "raw_response_available": True},
        )

        with patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", True):
            formatted = chat_core_mod.format_agent_raw_responses_for_logging({"messages": [ai_message]})

        self.assertIn("truncated", formatted)
        self.assertIn("req_long", formatted)
        self.assertNotIn(long_value, formatted)

    def test_agent_per_step_response_logging_middleware_logs_raw_response(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        middleware = chat_core_mod.AgentPerStepResponseLoggingMiddleware()
        ai_message = chat_core_mod.AIMessage(
            content="hello",
            additional_kwargs={
                "raw_response": {
                    "provider_type": "openai_compatible",
                    "api_style": "chat_completions",
                    "status_code": 200,
                    "headers": {"x-request-id": "req_step"},
                    "body_json": {"id": "chatcmpl_step"},
                    "body_text": None,
                    "request_id": "req_step",
                }
            },
            response_metadata={"finish_reason": "stop", "raw_response_available": True},
        )
        state = {"messages": [ai_message], "foo": "bar"}
        runtime = SimpleNamespace(context=SimpleNamespace(session_id="session_test", response_id="resp_test"))

        with (
            patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", True),
            patch.object(chat_core_mod.logger, "info") as logger_info_mock,
        ):
            result = middleware.after_model(state, runtime)

        self.assertIsNone(result)
        logged_text = "\n".join(str(call.args[0]) for call in logger_info_mock.call_args_list if call.args)
        self.assertIn("agent raw response", logged_text)
        self.assertIn("agent raw response diagnostic", logged_text)

    def test_log_model_raw_response_capability_logs_support_flag(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        model = SimpleNamespace(_gtbot_raw_response_available=True)
        with (
            patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", True),
            patch.object(chat_core_mod.logger, "info") as logger_info_mock,
        ):
            chat_core_mod._log_model_raw_response_capability(
                model=model,
                provider_type="openai_compatible",
                model_id="demo-model",
                base_url="https://example.test/v1",
                streaming=True,
                session_id="session_test",
            )

        logged_text = "\n".join(str(call.args[0]) for call in logger_info_mock.call_args_list if call.args)
        self.assertIn("chat model raw-response capability", logged_text)

    def test_log_last_ai_raw_response_diagnostic_logs_payload_status(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        ai_message = chat_core_mod.AIMessage(
            content="hello",
            additional_kwargs={"raw_response": {"request_id": "req_diag"}},
            response_metadata={"raw_response_available": True, "finish_reason": "stop"},
        )
        with (
            patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", True),
            patch.object(chat_core_mod.logger, "info") as logger_info_mock,
        ):
            chat_core_mod._log_last_ai_raw_response_diagnostic(
                label="final_response",
                messages=[ai_message],
                session_id="session_test",
                response_id="resp_test",
            )

        logged_text = "\n".join(str(call.args[0]) for call in logger_info_mock.call_args_list if call.args)
        self.assertIn("agent raw response diagnostic", logged_text)

    def test_format_agent_raw_responses_for_logging_returns_empty_when_disabled(self) -> None:
        (
            chat_core_mod,
            _get_current_plugin_context_fn,
            _plugin_bundle_cls,
            _plugin_context_cls,
            _pre_agent_message_appender_binding_cls,
            _pre_agent_message_injector_binding_cls,
            _pre_agent_processor_binding_cls,
            _base_message_cls,
            _human_message_cls,
            _system_message_cls,
        ) = _require_test_runtime()

        ai_message = chat_core_mod.AIMessage(
            content="hello",
            additional_kwargs={"raw_response": {"request_id": "req_disabled"}},
            response_metadata={"raw_response_available": True},
        )

        with patch.object(chat_core_mod, "ENABLE_RAW_RESPONSE_LOGGING", False):
            formatted = chat_core_mod.format_agent_raw_responses_for_logging({"messages": [ai_message]})

        self.assertEqual(formatted, "")


if __name__ == "__main__":
    unittest.main()
