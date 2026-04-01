from __future__ import annotations

import asyncio
import importlib
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

    massage_manager_mod = ModuleType("plugins.GTBot.MassageManager")
    setattr(massage_manager_mod, "GroupMessageManager", object)
    setattr(massage_manager_mod, "get_message_manager", lambda: None)
    sys.modules["plugins.GTBot.MassageManager"] = massage_manager_mod

    permission_manager_mod = ModuleType("plugins.GTBot.PermissionManager")
    setattr(permission_manager_mod, "PermissionError", RuntimeError)
    setattr(permission_manager_mod, "PermissionRole", SimpleNamespace(ADMIN="admin"))
    setattr(permission_manager_mod, "get_permission_manager", lambda: SimpleNamespace(require_role=AsyncMock()))
    sys.modules["plugins.GTBot.PermissionManager"] = permission_manager_mod

    tool_mod = ModuleType("plugins.GTBot.tools.long_memory.tool")
    setattr(tool_mod, "_impl_search_event_log_info", object())
    setattr(tool_mod, "_impl_search_group_profile_info", object())
    setattr(tool_mod, "_impl_search_public_knowledge", object())
    setattr(tool_mod, "_impl_search_user_profile_info", object())
    setattr(tool_mod, "normalize_session_id", lambda value: str(value or "").strip())
    sys.modules["plugins.GTBot.tools.long_memory.tool"] = tool_mod

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

        self.assertEqual(len(registry.iter_tools()), 1)
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

        long_memory_mod.long_memory_manager = SimpleNamespace(
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


if __name__ == "__main__":
    unittest.main()
