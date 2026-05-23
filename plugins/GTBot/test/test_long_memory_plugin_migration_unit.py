from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_package(name: str, path: Path) -> ModuleType:
    """确保测试所需的包对象存在于 `sys.modules` 中。

    Args:
        name: 需要准备的包名。
        path: 包目录绝对路径。

    Returns:
        ModuleType: 对应的包模块对象。
    """

    pkg = sys.modules.get(name)
    if isinstance(pkg, ModuleType):
        return pkg

    pkg = ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    pkg.__file__ = str(path / "__init__.py")
    pkg.__package__ = name
    sys.modules[name] = pkg
    return pkg


def _load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """按文件路径加载模块。

    Args:
        module_name: 目标模块名。
        file_path: 模块文件路径。

    Returns:
        ModuleType: 已执行完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_name} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_long_memory_test_runtime() -> tuple[type, type, type, ModuleType]:
    """安装长期记忆迁移测试所需的最小宿主与依赖桩。

    Returns:
        tuple[type, type, type, ModuleType]:
            依次为 `PluginContext`、`plugin_context_scope`、`PluginRegistry` 和
            已加载的 `plugins.GTBot.tools.long_memory` 模块。
    """

    _ensure_package("plugins", ROOT / "plugins")
    gtbot_pkg = _ensure_package("plugins.GTBot", ROOT / "plugins" / "GTBot")
    _ensure_package("plugins.GTBot.services", ROOT / "plugins" / "GTBot" / "services")
    _ensure_package(
        "plugins.GTBot.services.plugin_system",
        ROOT / "plugins" / "GTBot" / "services" / "plugin_system",
    )
    _ensure_package("plugins.GTBot.tools", ROOT / "plugins" / "GTBot" / "tools")

    setattr(gtbot_pkg, "Fun", SimpleNamespace())

    types_mod = _load_module_from_path(
        "plugins.GTBot.services.plugin_system.types",
        ROOT / "plugins" / "GTBot" / "services" / "plugin_system" / "types.py",
    )
    runtime_mod = _load_module_from_path(
        "plugins.GTBot.services.plugin_system.runtime",
        ROOT / "plugins" / "GTBot" / "services" / "plugin_system" / "runtime.py",
    )
    registry_mod = _load_module_from_path(
        "plugins.GTBot.services.plugin_system.registry",
        ROOT / "plugins" / "GTBot" / "services" / "plugin_system" / "registry.py",
    )

    qdrant_mod = ModuleType("qdrant_client")
    setattr(qdrant_mod, "AsyncQdrantClient", object)
    sys.modules["qdrant_client"] = qdrant_mod
    qdrant_models_mod = ModuleType("qdrant_client.models")
    for name in ("FieldCondition", "Filter", "MatchValue", "PointIdsList"):
        setattr(qdrant_models_mod, name, object)
    sys.modules["qdrant_client.models"] = qdrant_models_mod

    nonebot_mod = ModuleType("nonebot")

    class _FakeCommand:
        def handle(self):  # noqa: ANN201
            def _decorator(func):  # noqa: ANN001
                return func

            return _decorator

    setattr(
        nonebot_mod,
        "logger",
        SimpleNamespace(debug=Mock(), info=Mock(), warning=Mock(), error=Mock()),
    )
    setattr(nonebot_mod, "on_command", lambda *args, **kwargs: _FakeCommand())
    sys.modules["nonebot"] = nonebot_mod

    _ensure_package("nonebot.adapters", ROOT)
    _ensure_package("nonebot.adapters.onebot", ROOT)
    onebot_v11_mod = ModuleType("nonebot.adapters.onebot.v11")
    setattr(onebot_v11_mod, "Bot", object)
    sys.modules["nonebot.adapters.onebot.v11"] = onebot_v11_mod

    event_mod = ModuleType("nonebot.adapters.onebot.v11.event")
    setattr(event_mod, "GroupMessageEvent", object)
    setattr(event_mod, "MessageEvent", object)
    sys.modules["nonebot.adapters.onebot.v11.event"] = event_mod

    message_mod = ModuleType("nonebot.adapters.onebot.v11.message")
    setattr(message_mod, "Message", str)
    sys.modules["nonebot.adapters.onebot.v11.message"] = message_mod

    exception_mod = ModuleType("nonebot.adapters.onebot.v11.exception")
    setattr(exception_mod, "ActionFailed", RuntimeError)
    sys.modules["nonebot.adapters.onebot.v11.exception"] = exception_mod

    bot_mod = ModuleType("nonebot.adapters.onebot.v11.bot")
    setattr(bot_mod, "Bot", object)
    sys.modules["nonebot.adapters.onebot.v11.bot"] = bot_mod

    params_mod = ModuleType("nonebot.params")
    setattr(params_mod, "CommandArg", lambda: None)
    setattr(params_mod, "Depends", lambda fn: fn)
    sys.modules["nonebot.params"] = params_mod

    massage_manager_mod = ModuleType("plugins.GTBot.services.message")
    setattr(massage_manager_mod, "GroupMessageManager", object)
    setattr(massage_manager_mod, "get_message_manager", lambda: None)
    sys.modules["plugins.GTBot.services.message"] = massage_manager_mod

    llm_provider_mod = ModuleType("plugins.GTBot.llm_provider")
    setattr(llm_provider_mod, "build_chat_model", lambda *args, **kwargs: None)
    sys.modules["plugins.GTBot.llm_provider"] = llm_provider_mod

    gtbot_model_mod = ModuleType("plugins.GTBot.model")
    setattr(gtbot_model_mod, "Message", object)
    sys.modules["plugins.GTBot.model"] = gtbot_model_mod

    permission_manager_mod = ModuleType("plugins.GTBot.services.permission")
    setattr(permission_manager_mod, "PermissionError", RuntimeError)
    setattr(permission_manager_mod, "PermissionRole", SimpleNamespace(ADMIN="admin"))
    setattr(permission_manager_mod, "get_permission_manager", lambda: SimpleNamespace(require_role=AsyncMock()))
    sys.modules["plugins.GTBot.services.permission"] = permission_manager_mod

    async def _stub_search_event_log_info(*args: Any, **kwargs: Any) -> str:
        return ""

    async def _stub_search_group_profile_info(*args: Any, **kwargs: Any) -> str:
        return ""

    async def _stub_search_public_knowledge(*args: Any, **kwargs: Any) -> str:
        return ""

    async def _stub_search_user_profile_info(*args: Any, **kwargs: Any) -> str:
        return ""

    async def _stub_memory_tool(*args: Any, **kwargs: Any) -> str:
        return ""

    tool_mod = ModuleType("plugins.GTBot.tools.long_memory.tool")
    setattr(tool_mod, "_impl_search_event_log_info", _stub_search_event_log_info)
    setattr(tool_mod, "_impl_search_group_profile_info", _stub_search_group_profile_info)
    setattr(tool_mod, "_impl_search_public_knowledge", _stub_search_public_knowledge)
    setattr(tool_mod, "_impl_search_user_profile_info", _stub_search_user_profile_info)
    setattr(tool_mod, "PUBLIC_KNOWLEDGE_GROUP", "public_knowledge")
    for name in (
        "add_event_log_info",
        "add_group_profile_info",
        "add_public_knowledge",
        "add_user_profile_info",
        "delete_event_log_info",
        "delete_group_profile_info",
        "delete_public_knowledge",
        "delete_user_profile_info",
        "get_event_log_info",
        "get_group_profile_info",
        "get_public_knowledge",
        "get_user_profile_info",
        "search_event_log_info",
        "search_group_profile_info",
        "search_public_knowledge",
        "search_user_profile_info",
        "update_event_log_info",
        "update_group_profile_info",
        "update_public_knowledge",
        "update_user_profile_info",
    ):
        setattr(tool_mod, name, _stub_memory_tool)
    setattr(tool_mod, "normalize_session_id", lambda value: str(value or "").strip())
    sys.modules["plugins.GTBot.tools.long_memory.tool"] = tool_mod

    mapping_manager_mod = ModuleType("plugins.GTBot.tools.long_memory.MappingManager")
    setattr(mapping_manager_mod, "mapping_manager", SimpleNamespace(get_short_id=lambda **kwargs: "sid"))
    sys.modules["plugins.GTBot.tools.long_memory.MappingManager"] = mapping_manager_mod

    for name in (
        "notepad",
        "VectorGenerator",
        "UserProfile",
        "GroupProfileQdrant",
        "EventLogManager",
        "PublicKnowledge",
    ):
        sys.modules[f"plugins.GTBot.tools.long_memory.{name}"] = ModuleType(
            f"plugins.GTBot.tools.long_memory.{name}"
        )

    config_mod = ModuleType("plugins.GTBot.tools.long_memory.config")

    def _model_dump_empty() -> dict[str, Any]:
        return {}

    setattr(
        config_mod,
        "get_long_memory_plugin_config",
        lambda: SimpleNamespace(
            auto_init=False,
            recall=SimpleNamespace(model_dump=_model_dump_empty),
            ingest=SimpleNamespace(model_dump=_model_dump_empty),
            post_llm_ingest=SimpleNamespace(recent_n=20, delay_seconds=0.0),
        ),
    )
    sys.modules["plugins.GTBot.tools.long_memory.config"] = config_mod

    sys.modules["plugins.GTBot.tools.long_memory.memory_editor"] = ModuleType(
        "plugins.GTBot.tools.long_memory.memory_editor"
    )

    sys.modules.pop("plugins.GTBot.tools.long_memory", None)
    long_memory_mod = importlib.import_module("plugins.GTBot.tools.long_memory")
    return (
        getattr(types_mod, "PluginContext"),
        getattr(runtime_mod, "plugin_context_scope"),
        getattr(registry_mod, "PluginRegistry"),
        long_memory_mod,
    )


class TestLongMemoryPluginMigrationUnit(unittest.TestCase):
    def test_long_memory_register_uses_processor_injector_and_callback(self) -> None:
        _plugin_context_cls, _plugin_context_scope, registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        registry = registry_cls()
        long_memory_mod.register(registry)

        self.assertEqual(len(registry.iter_tools()), 3)
        self.assertEqual(len(registry.iter_pre_agent_processors()), 1)
        self.assertEqual(len(registry.iter_pre_agent_message_injectors()), 1)
        self.assertEqual(len(registry.iter_callbacks()), 1)
        self.assertEqual(len(registry.iter_middlewares()), 0)
        self.assertTrue(registry.iter_pre_agent_processors()[0].wait_until_complete)

    def test_prepare_long_memory_recall_uses_cached_refresh_strategy(self) -> None:
        plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        class _RecallConfig:
            def __init__(self, **_: Any) -> None:
                pass

        runtime_context = SimpleNamespace(
            session_id="group_1",
            raw_messages=[SimpleNamespace(message_id=1, user_id=2, send_time=1.0)],
            group_id=1,
            user_id=2,
        )
        plugin_ctx = plugin_context_cls(raw_messages=list(runtime_context.raw_messages), runtime_context=runtime_context)
        recall_manager = SimpleNamespace(
            add_message=AsyncMock(),
            get_current_related_memories=AsyncMock(return_value="related_memories"),
        )

        with (
            patch.object(long_memory_mod, "get_long_memory_recall_manager", return_value=recall_manager),
            patch.object(
                long_memory_mod,
                "import_module",
                return_value=SimpleNamespace(LongMemoryRecallConfig=_RecallConfig),
            ),
        ):
            asyncio.run(long_memory_mod.prepare_long_memory_recall(plugin_ctx))

        self.assertEqual(plugin_ctx.extra["long_memory_related_memories"], "related_memories")
        self.assertTrue(plugin_ctx.extra["_long_memory_recall_prepared"])
        recall_manager.add_message.assert_awaited_once()
        recall_manager.get_current_related_memories.assert_awaited_once()
        self.assertFalse(recall_manager.get_current_related_memories.await_args.kwargs["force_refresh"])

    def test_prepare_long_memory_recall_falls_back_to_old_cache(self) -> None:
        plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        class _RecallConfig:
            def __init__(self, **_: Any) -> None:
                pass

        cached_related = SimpleNamespace(name="cached")
        runtime_context = SimpleNamespace(
            session_id="group_1",
            raw_messages=[SimpleNamespace(message_id=1, user_id=2, send_time=1.0)],
            group_id=1,
            user_id=2,
        )
        plugin_ctx = plugin_context_cls(raw_messages=list(runtime_context.raw_messages), runtime_context=runtime_context)
        recall_manager = SimpleNamespace(
            add_message=AsyncMock(),
            get_current_related_memories=AsyncMock(side_effect=RuntimeError("boom")),
            _sessions={"group_1": SimpleNamespace(related=cached_related)},
        )

        with (
            patch.object(long_memory_mod, "get_long_memory_recall_manager", return_value=recall_manager),
            patch.object(
                long_memory_mod,
                "import_module",
                return_value=SimpleNamespace(LongMemoryRecallConfig=_RecallConfig),
            ),
        ):
            asyncio.run(long_memory_mod.prepare_long_memory_recall(plugin_ctx))

        self.assertIs(plugin_ctx.extra["long_memory_related_memories"], cached_related)
        self.assertTrue(plugin_ctx.extra["_long_memory_recall_prepared"])

    def test_inject_long_memory_context_prepends_recall_then_notepad(self) -> None:
        plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        long_memory_mod.long_memory_manager = SimpleNamespace(  # type: ignore[attr-defined]
            notepad_manager=SimpleNamespace(
                has_session=lambda session_id: session_id == "group_1",
                get_notes=lambda session_id: "memo note" if session_id == "group_1" else "",
            )
        )
        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            runtime_context=SimpleNamespace(session_id="group_1", group_id=1, user_id=2),
            extra={"long_memory_related_memories": "placeholder"},
        )
        messages = [
            long_memory_mod.SystemMessage("system"),
            long_memory_mod.HumanMessage("meme prompt\n\n<messages>history</messages>"),
        ]

        with patch.object(
            long_memory_mod,
            "_format_related_long_memories",
            return_value="<long_term_memory_retrieval_hit>\nrecall\n</long_term_memory_retrieval_hit>",
        ):
            updated_messages = asyncio.run(long_memory_mod.inject_long_memory_context(plugin_ctx, messages))

        self.assertEqual(len(updated_messages), 2)
        self.assertEqual(
            getattr(updated_messages[1], "content", ""),
            "<long_term_memory_retrieval_hit>\nrecall\n</long_term_memory_retrieval_hit>\n\n"
            "<note>\nmemo note\n</note>\n\n"
            "meme prompt\n\n<messages>history</messages>",
        )
        self.assertTrue(plugin_ctx.extra["_long_memory_recall_injected"])
        self.assertTrue(plugin_ctx.extra["_long_memory_notepad_injected"])

    def test_post_llm_ingest_callback_uses_run_dedup_and_scheduler(self) -> None:
        plugin_context_cls, plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        callback = long_memory_mod.LongMemoryPostLLMIngestCallback()
        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            runtime_context=SimpleNamespace(session_id="group_1", group_id=1, user_id=2),
        )

        async def _run() -> Any:
            with (
                plugin_context_scope(plugin_ctx),
                patch.object(long_memory_mod, "_schedule_post_llm_ingest_task") as schedule_task,
            ):
                callback.on_chain_start({}, {"messages": ["start"]}, run_id="run_1")
                callback.on_chain_end({}, run_id="run_1")
                callback.on_chain_start({}, {"messages": ["start"]}, run_id="run_2")
                callback.on_chain_error(RuntimeError("boom"), run_id="run_2")
                return schedule_task

        schedule_task = asyncio.run(_run())
        schedule_task.assert_called_once()
        self.assertEqual(schedule_task.call_args.kwargs["session_id"], "group_1")
        self.assertIs(schedule_task.call_args.kwargs["runtime_context"], plugin_ctx.runtime_context)
        self.assertIsNotNone(schedule_task.call_args.kwargs["event_loop"])
        self.assertEqual(callback._run_to_session, {})
        self.assertEqual(callback._run_to_runtime_context, {})
        self.assertEqual(callback._run_to_event_loop, {})

    def test_schedule_post_llm_ingest_task_cancels_old_task_and_cleans_up(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        async def _run() -> None:
            runtime_one = SimpleNamespace(name="one")
            runtime_two = SimpleNamespace(name="two")
            first_started = asyncio.Event()
            second_started = asyncio.Event()
            allow_second_finish = asyncio.Event()
            first_cancelled = {"value": False}

            async def fake_post_llm_ingest_recent_messages(*, session_id: str, runtime_context: Any) -> None:
                if runtime_context is runtime_one:
                    first_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        first_cancelled["value"] = True
                        raise

                if runtime_context is runtime_two:
                    second_started.set()
                    await allow_second_finish.wait()

            with patch.object(
                long_memory_mod,
                "_post_llm_ingest_recent_messages",
                side_effect=fake_post_llm_ingest_recent_messages,
            ):
                long_memory_mod._schedule_post_llm_ingest_task(session_id="group_1", runtime_context=runtime_one)
                await asyncio.wait_for(first_started.wait(), timeout=1.0)

                long_memory_mod._schedule_post_llm_ingest_task(session_id="group_1", runtime_context=runtime_two)
                await asyncio.wait_for(second_started.wait(), timeout=1.0)
                await asyncio.sleep(0)
                self.assertTrue(first_cancelled["value"])

                allow_second_finish.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            self.assertNotIn("group_1", long_memory_mod._post_llm_ingest_tasks)

        asyncio.run(_run())

    def test_schedule_post_llm_ingest_task_uses_event_loop_fallback(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        fake_task = Mock()
        setattr(fake_task, "done", lambda: False)
        fake_loop = Mock()

        def fake_create_task(coro: Any) -> Any:
            coro.close()
            return fake_task

        def fake_call_soon_threadsafe(callback: Any) -> None:
            callback()

        fake_loop.call_soon_threadsafe.side_effect = fake_call_soon_threadsafe
        fake_loop.is_closed.return_value = False

        with (
            patch.object(long_memory_mod.asyncio, "get_running_loop", side_effect=RuntimeError("no running event loop")),
            patch.object(long_memory_mod.asyncio, "create_task", side_effect=fake_create_task),
        ):
            long_memory_mod._schedule_post_llm_ingest_task(
                session_id="group_1",
                runtime_context=SimpleNamespace(name="ctx"),
                event_loop=fake_loop,
            )

        fake_loop.call_soon_threadsafe.assert_called_once()
        self.assertIs(long_memory_mod._post_llm_ingest_tasks["group_1"], fake_task)
        long_memory_mod._post_llm_ingest_tasks.clear()

    def test_build_recall_long_term_memory_payload_uses_current_scope_filters(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        event_search = AsyncMock(
            return_value="short_id=e1 similarity=0.900000 details=生日礼物安排"
        )
        user_search = AsyncMock(
            return_value="user_id=123 short_id=u1 similarity=0.800000 text=喜欢手账"
        )
        group_search = AsyncMock(
            return_value="short_id=g1 similarity=0.7000 category=氛围 text=最近在筹备生日"
        )
        public_search = AsyncMock(
            return_value="short_id=p1 title=偏好 similarity=0.600000 content=喜欢香薰"
        )

        with (
            patch.object(long_memory_mod, "_impl_search_event_log_info", event_search),
            patch.object(long_memory_mod, "_impl_search_user_profile_info", user_search),
            patch.object(long_memory_mod, "_impl_search_group_profile_info", group_search),
            patch.object(long_memory_mod, "_impl_search_public_knowledge", public_search),
        ):
            payload = asyncio.run(
                long_memory_mod._build_recall_long_term_memory_payload(
                    long_memory=SimpleNamespace(),
                    query="阿梓 生日",
                    scope="current",
                    layer="auto",
                    limit=3,
                    session_id="group_42",
                    group_id=42,
                )
            )

        self.assertEqual(payload["query"], "阿梓 生日")
        self.assertIsNone(payload["results"]["event_log"]["error"])
        self.assertEqual(payload["results"]["event_log"]["items"][0]["short_id"], "e1")
        self.assertEqual(payload["results"]["user_profile"]["items"][0]["user_id"], 123)
        self.assertEqual(payload["results"]["group_profile"]["items"][0]["group_id"], 42)
        self.assertEqual(payload["results"]["public_knowledge"]["items"][0]["title"], "偏好")
        event_await_args = event_search.await_args
        group_await_args = group_search.await_args
        self.assertIsNotNone(event_await_args)
        self.assertIsNotNone(group_await_args)
        assert event_await_args is not None
        assert group_await_args is not None
        self.assertEqual(event_await_args.kwargs["session_id"], "group_42")
        self.assertEqual(group_await_args.kwargs["group_id"], 42)

    def test_build_recall_long_term_memory_payload_handles_layer_errors(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        with patch.object(
            long_memory_mod,
            "_impl_search_public_knowledge",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            payload = asyncio.run(
                long_memory_mod._build_recall_long_term_memory_payload(
                    long_memory=SimpleNamespace(),
                    query="偏好",
                    scope="current",
                    layer="public_knowledge",
                    limit=2,
                    session_id="group_42",
                    group_id=42,
                )
            )

        self.assertEqual(payload["results"]["public_knowledge"]["items"], [])
        self.assertIn("public_knowledge 检索失败", payload["results"]["public_knowledge"]["error"])

    def test_build_recall_long_term_memory_payload_allows_global_event_log_search(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        event_search = AsyncMock(
            return_value="short_id=e2 similarity=0.550000 details=跨会话事件"
        )

        with patch.object(long_memory_mod, "_impl_search_event_log_info", event_search):
            payload = asyncio.run(
                long_memory_mod._build_recall_long_term_memory_payload(
                    long_memory=SimpleNamespace(),
                    query="跨群活动",
                    scope="global",
                    layer="event_log",
                    limit=2,
                    session_id="group_42",
                    group_id=42,
                )
            )

        event_await_args = event_search.await_args
        self.assertIsNotNone(event_await_args)
        assert event_await_args is not None
        self.assertIsNone(event_await_args.kwargs["session_id"])
        self.assertEqual(payload["results"]["event_log"]["items"][0]["short_id"], "e2")
        self.assertIsNone(payload["results"]["event_log"]["error"])

    def test_recall_long_term_memory_returns_stable_json(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                long_memory=SimpleNamespace(),
                session_id="group_7",
                group_id=7,
                user_id=1001,
            )
        )
        fake_payload = {
            "query": "生日",
            "scope": "current",
            "layer": "auto",
            "results": {
                "event_log": {"items": [{"short_id": "e1", "details": "安排", "similarity": 0.9}], "error": None},
                "user_profile": {"items": [], "error": None},
                "group_profile": {"items": [], "error": None},
                "public_knowledge": {"items": [], "error": None},
            },
        }

        with patch.object(
            long_memory_mod,
            "_build_recall_long_term_memory_payload",
            AsyncMock(return_value=fake_payload),
        ):
            raw = asyncio.run(
                long_memory_mod.recall_long_term_memory.coroutine(  # type: ignore[attr-defined]
                    query="生日",
                    runtime=runtime,
                    scope="current",
                    layer="auto",
                    limit=5,
                )
            )

        parsed = json.loads(raw)
        self.assertEqual(parsed["query"], "生日")
        self.assertEqual(parsed["results"]["event_log"]["items"][0]["short_id"], "e1")

    def test_remember_for_long_memory_stores_pending_request(self) -> None:
        plugin_context_cls, plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                session_id="group_7",
                group_id=7,
                user_id=1001,
            )
        )
        plugin_ctx = plugin_context_cls(raw_messages=[], extra={})

        with plugin_context_scope(plugin_ctx):
            result = asyncio.run(
                long_memory_mod.remember_for_long_memory.coroutine(  # type: ignore[attr-defined]
                    request="请记住阿梓喜欢香薰。",
                    runtime=runtime,
                )
            )

        self.assertEqual(result, "已记录本轮记忆请求，稍后会交给长期记忆整理器处理。")
        self.assertEqual(
            long_memory_mod._pending_ingest_memory_requests_by_session["group_7"],
            ["请记住阿梓喜欢香薰。"],
        )
        self.assertEqual(plugin_ctx.extra["long_memory_memory_requests"], ["请记住阿梓喜欢香薰。"])

    def test_remember_for_long_memory_caps_pending_requests_per_session(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        session_id = "group_7"
        max_items = long_memory_mod._LONG_MEMORY_MEMORY_REQUEST_MAX_ITEMS_PER_SESSION
        long_memory_mod._pending_ingest_memory_requests_by_session[session_id] = [
            f"旧请求{i}" for i in range(max_items)
        ]

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                session_id=session_id,
                group_id=7,
                user_id=1001,
            )
        )
        asyncio.run(
            long_memory_mod.remember_for_long_memory.coroutine(  # type: ignore[attr-defined]
                request="最新请求",
                runtime=runtime,
            )
        )

        stored = long_memory_mod._pending_ingest_memory_requests_by_session[session_id]
        self.assertEqual(len(stored), max_items)
        self.assertEqual(stored[-1], "最新请求")
        self.assertNotIn("旧请求0", stored)

    def test_post_llm_ingest_recent_messages_appends_memory_request_message(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        class _IngestConfig:
            def __init__(self, **_: Any) -> None:
                pass

        ingest_manager = SimpleNamespace(add_message=AsyncMock())
        long_memory_mod._pending_ingest_memory_requests_by_session["group_7"] = ["请记住阿梓喜欢香薰。"]
        runtime_context = SimpleNamespace(
            group_id=7,
            user_id=1001,
            message_manager=SimpleNamespace(
                get_recent_messages=AsyncMock(
                    return_value=[
                        SimpleNamespace(
                            db_id=11,
                            message_id=1,
                            user_id=1001,
                            user_name="高崽",
                            send_time=1.0,
                            content="普通聊天",
                        )
                    ]
                )
            ),
        )

        with (
            patch.object(long_memory_mod, "long_memory_manager", SimpleNamespace(), create=True),
            patch.object(
                long_memory_mod,
                "import_module",
                return_value=SimpleNamespace(LongMemoryIngestConfig=_IngestConfig),
            ),
            patch.object(long_memory_mod, "get_long_memory_ingest_manager", return_value=ingest_manager),
        ):
            asyncio.run(
                long_memory_mod._post_llm_ingest_recent_messages(
                    session_id="group_7",
                    runtime_context=runtime_context,
                )
            )

        self.assertEqual(ingest_manager.add_message.await_count, 2)
        synthetic_message = ingest_manager.add_message.await_args_list[-1].kwargs["message"]
        self.assertIn("[LONG_MEMORY_REQUEST]", synthetic_message.content)
        self.assertIn("请记住阿梓喜欢香薰。", synthetic_message.content)
        self.assertNotIn("group_7", long_memory_mod._pending_ingest_memory_requests_by_session)

    def test_post_llm_ingest_recent_messages_keeps_memory_request_when_ingest_unavailable(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        class _IngestConfig:
            def __init__(self, **_: Any) -> None:
                pass

        long_memory_mod._pending_ingest_memory_requests_by_session["group_7"] = ["请记住阿梓喜欢香薰。"]
        runtime_context = SimpleNamespace(
            group_id=7,
            user_id=1001,
            message_manager=SimpleNamespace(
                get_recent_messages=AsyncMock(
                    return_value=[
                        SimpleNamespace(
                            db_id=11,
                            message_id=1,
                            user_id=1001,
                            user_name="高崽",
                            send_time=1.0,
                            content="普通聊天",
                        )
                    ]
                )
            ),
        )

        with (
            patch.object(long_memory_mod, "long_memory_manager", SimpleNamespace(), create=True),
            patch.object(
                long_memory_mod,
                "import_module",
                return_value=SimpleNamespace(LongMemoryIngestConfig=_IngestConfig),
            ),
            patch.object(long_memory_mod, "get_long_memory_ingest_manager", return_value=None),
        ):
            asyncio.run(
                long_memory_mod._post_llm_ingest_recent_messages(
                    session_id="group_7",
                    runtime_context=runtime_context,
                )
            )

        self.assertEqual(
            long_memory_mod._pending_ingest_memory_requests_by_session["group_7"],
            ["请记住阿梓喜欢香薰。"],
        )

    def test_default_ingest_prompt_mentions_long_memory_request(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, _long_memory_mod = _install_long_memory_test_runtime()

        sys.modules.pop("plugins.GTBot.tools.long_memory.IngestManager", None)
        ingest_manager_mod = importlib.import_module("plugins.GTBot.tools.long_memory.IngestManager")

        prompt = ingest_manager_mod._default_ingest_prompt()

        self.assertIn("[LONG_MEMORY_REQUEST]", prompt)
        self.assertIn("主对话 agent 主动提交的记忆意图", prompt)

    def test_recall_long_term_memory_falls_back_to_module_manager(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                session_id="group_7",
                group_id=7,
                user_id=1001,
            )
        )
        fake_long_memory = SimpleNamespace(name="module-manager")
        fake_payload = {
            "query": "生日",
            "scope": "current",
            "layer": "auto",
            "results": {
                "event_log": {"items": [], "error": None},
                "user_profile": {"items": [], "error": None},
                "group_profile": {"items": [], "error": None},
                "public_knowledge": {"items": [], "error": None},
            },
        }

        with (
            patch.object(long_memory_mod, "long_memory_manager", fake_long_memory, create=True),
            patch.object(
                long_memory_mod,
                "_build_recall_long_term_memory_payload",
                AsyncMock(return_value=fake_payload),
            ) as build_payload,
        ):
            raw = asyncio.run(
                long_memory_mod.recall_long_term_memory.coroutine(  # type: ignore[attr-defined]
                    query="生日",
                    runtime=runtime,
                    scope="current",
                    layer="auto",
                    limit=5,
                )
            )

        build_await_args = build_payload.await_args
        self.assertIsNotNone(build_await_args)
        assert build_await_args is not None
        self.assertIs(build_await_args.kwargs["long_memory"], fake_long_memory)
        parsed = json.loads(raw)
        self.assertEqual(parsed["scope"], "current")

    def test_recall_long_term_memory_returns_stable_error_when_manager_unavailable(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                session_id="group_7",
                group_id=7,
                user_id=1001,
            )
        )

        with patch.object(long_memory_mod, "long_memory_manager", None, create=True):
            raw = asyncio.run(
                long_memory_mod.recall_long_term_memory.coroutine(  # type: ignore[attr-defined]
                    query="生日",
                    runtime=runtime,
                    scope="current",
                    layer="auto",
                    limit=5,
                )
            )

        parsed = json.loads(raw)
        self.assertEqual(parsed["query"], "生日")
        self.assertEqual(parsed["results"]["event_log"]["items"], [])
        self.assertEqual(parsed["results"]["event_log"]["error"], "long_memory_unavailable")
        self.assertEqual(parsed["results"]["public_knowledge"]["error"], "long_memory_unavailable")

    def test_render_recall_long_term_memory_text_uses_unified_sections(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        text = long_memory_mod._render_recall_long_term_memory_text(
            {
                "query": "生日",
                "scope": "current",
                "layer": "auto",
                "results": {
                    "event_log": {"items": [{"short_id": "e1", "details": "安排", "similarity": 0.9}], "error": None},
                    "user_profile": {"items": [{"user_id": 1, "short_id": "u1", "text": "喜欢蛋糕", "similarity": 0.8}], "error": None},
                    "group_profile": {"items": [], "error": None},
                    "public_knowledge": {"items": [], "error": "未检索到公共知识。"},
                },
            }
        )

        self.assertIn("[长期记忆检索结果]", text)
        self.assertIn("### 事件日志", text)
        self.assertIn("short_id=e1", text)
        self.assertIn("### 用户画像", text)
        self.assertIn("user_id=1", text)
        self.assertIn("### 群画像", text)
        self.assertIn("无", text)
        self.assertIn("### 公共知识", text)
        self.assertIn("error=未检索到公共知识。", text)

    def test_handle_search_long_memory_uses_unified_payload_renderer(self) -> None:
        _plugin_context_cls, _plugin_context_scope, _registry_cls, long_memory_mod = _install_long_memory_test_runtime()

        class _Args:
            def extract_plain_text(self) -> str:
                return "生日"

        payload = {
            "query": "生日",
            "scope": "current",
            "layer": "auto",
            "results": {
                "event_log": {"items": [{"short_id": "e1", "details": "安排", "similarity": 0.9}], "error": None},
                "user_profile": {"items": [], "error": None},
                "group_profile": {"items": [], "error": None},
                "public_knowledge": {"items": [], "error": None},
            },
        }

        send_mock = AsyncMock()
        finish_mock = AsyncMock()
        with (
            patch.object(long_memory_mod, "_ensure_long_memory_admin", AsyncMock()),
            patch.object(long_memory_mod, "_build_recall_long_term_memory_payload", AsyncMock(return_value=payload)),
            patch.object(long_memory_mod.SearchLongMemory, "send", send_mock, create=True),
            patch.object(long_memory_mod.SearchLongMemory, "finish", finish_mock, create=True),
        ):
            long_memory_mod.long_memory_manager = SimpleNamespace()  # type: ignore[attr-defined]
            asyncio.run(
                long_memory_mod.handle_search_long_memory(
                    bot=SimpleNamespace(),
                    event=SimpleNamespace(user_id=1001, group_id=42),
                    args=_Args(),
                )
        )

        finish_mock.assert_not_awaited()
        send_mock.assert_awaited()
        sent_text = "\n".join(str(call.args[0]) for call in send_mock.await_args_list)
        self.assertIn("[长期记忆检索结果]", sent_text)
        self.assertIn("### 事件日志", sent_text)


if __name__ == "__main__":
    unittest.main()
