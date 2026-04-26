from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Awaitable, Callable, Protocol, cast
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _CallbackProtocol(Protocol):
    """描述当前测试会直接调用的 callback 接口。"""

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> Any:
        """处理链启动事件。"""

    def on_chat_model_end(self, response: Any, **kwargs: Any) -> Any:
        """处理聊天模型结束事件。"""

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """处理 LLM 结束事件。"""

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """处理工具启动事件。"""

    def on_llm_new_token(self, token: Any, **kwargs: Any) -> Any:
        """处理流式 token 事件。"""


def _ensure_package(name: str, path: Path) -> ModuleType:
    """确保测试所需的包对象存在于 `sys.modules` 中。"""

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
    """从指定路径加载测试模块。"""

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_name} -> {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_gtbot_test_stubs(*, include_nonebot: bool = False) -> tuple[type, type, ModuleType]:
    """安装加载插件模块所需的最小宿主桩对象。

    Args:
        include_nonebot: 是否同时注入带假 logger 的 `nonebot` 模块。

    Returns:
        tuple[type, type, ModuleType]: 依次为 `PluginContext`、`plugin_context_scope`
            所在模块中的上下文管理器宿主和 registry 模块。
    """

    _ensure_package("plugins", ROOT / "plugins")
    gtbot_pkg = _ensure_package("plugins.GTBot", ROOT / "plugins" / "GTBot")
    _ensure_package("plugins.GTBot.services", ROOT / "plugins" / "GTBot" / "services")
    _ensure_package(
        "plugins.GTBot.services.shared",
        ROOT / "plugins" / "GTBot" / "services" / "shared",
    )
    _ensure_package(
        "plugins.GTBot.services.plugin_system",
        ROOT / "plugins" / "GTBot" / "services" / "plugin_system",
    )

    test_data_dir = ROOT / ".tmp_plugin_migration_test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    setattr(
        gtbot_pkg,
        "Fun",
        SimpleNamespace(
        generate_cq_string=lambda cq_type, data: f"[CQ:{cq_type}," + ",".join(
            f"{key}={value}" for key, value in data.items()
        ) + "]",
        parse_single_cq=lambda _: {"CQ": "image", "file": "demo.png"},
        ),
    )
    fun_mod = ModuleType("plugins.GTBot.services.shared.fun")
    setattr(
        fun_mod,
        "generate_cq_string",
        lambda cq_type, data: f"[CQ:{cq_type}," + ",".join(
            f"{key}={value}" for key, value in data.items()
        ) + "]",
    )
    setattr(fun_mod, "parse_single_cq", lambda _: {"CQ": "image", "file": "demo.png"})
    sys.modules["plugins.GTBot.services.shared.fun"] = fun_mod

    config_manager_mod = ModuleType("plugins.GTBot.ConfigManager")
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: test_data_dir))
    sys.modules["plugins.GTBot.ConfigManager"] = config_manager_mod

    group_chat_context_mod = ModuleType("plugins.GTBot.services.chat.context")

    class GroupChatContext:  # noqa: D401
        """测试用 GroupChatContext 桩对象。"""

        pass

    setattr(group_chat_context_mod, "GroupChatContext", GroupChatContext)
    sys.modules["plugins.GTBot.services.chat.context"] = group_chat_context_mod

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

    if include_nonebot:
        nonebot_mod = ModuleType("nonebot")
        setattr(
            nonebot_mod,
            "logger",
            SimpleNamespace(
                debug=Mock(),
                info=Mock(),
                warning=Mock(),
                error=Mock(),
            ),
        )
        sys.modules["nonebot"] = nonebot_mod

    return getattr(types_mod, "PluginContext"), getattr(runtime_mod, "plugin_context_scope"), registry_mod


class _FakeRegistry:
    """记录插件注册行为的轻量测试注册表。"""

    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.agent_middlewares: list[Any] = []
        self.callbacks: list[_CallbackProtocol] = []
        self.pre_agent_processors: list[Callable[[Any], Awaitable[None]]] = []
        self.pre_agent_message_injectors: list[Any] = []
        self.pre_agent_message_appenders: list[Any] = []

    def add_tool(self, tool: object, **_: object) -> None:
        self.tools.append(tool)

    def add_agent_middleware(self, middleware: object, **_: object) -> None:
        self.agent_middlewares.append(middleware)

    def add_callback(self, callback: _CallbackProtocol, **_: object) -> None:
        self.callbacks.append(callback)

    def add_pre_agent_processor(self, processor: Callable[[Any], Awaitable[None]], **_: object) -> None:
        self.pre_agent_processors.append(processor)

    def add_pre_agent_message_injector(self, injector: object, **_: object) -> None:
        self.pre_agent_message_injectors.append(injector)

    def add_pre_agent_message_appender(self, appender: object, **_: object) -> None:
        self.pre_agent_message_appenders.append(appender)


class TestNonLongMemoryPluginMigrationUnit(unittest.TestCase):
    def test_meme_register_uses_message_injector_instead_of_middleware(self) -> None:
        meme_init_mod = _load_module_from_path(
            f"_gtbot_test_meme_init_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "meme" / "__init__.py",
        )
        registry = _FakeRegistry()
        fake_tool_mod = SimpleNamespace(
            save_meme=object(),
            inject_meme_context_into_messages=object(),
        )

        with patch.object(
            meme_init_mod.importlib,
            "import_module",
            side_effect=[
                SimpleNamespace(get_meme_plugin_config=lambda: object()),
                fake_tool_mod,
            ],
        ):
            meme_init_mod.register(registry)

        self.assertEqual(len(registry.tools), 1)
        self.assertEqual(len(registry.pre_agent_message_injectors), 1)
        self.assertEqual(registry.agent_middlewares, [])

    def test_inject_meme_context_into_messages_prepends_into_main_human_message(self) -> None:
        _install_gtbot_test_stubs(include_nonebot=True)
        try:
            meme_tool_mod = _load_module_from_path(
                f"_gtbot_test_meme_tool_{id(self)}",
                ROOT / "plugins" / "GTBot" / "tools" / "meme" / "tool.py",
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"缺少运行依赖，已跳过: {exc}") from exc

        with patch.object(
            meme_tool_mod,
            "build_meme_context_prompt",
            AsyncMock(return_value="meme prompt"),
        ):
            messages = asyncio.run(
                meme_tool_mod.inject_meme_context_into_messages(
                    SimpleNamespace(),
                    [meme_tool_mod.HumanMessage(content="<messages>history</messages>")],
                )
            )

        self.assertEqual(
            [getattr(message, "content", "") for message in messages],
            ["meme prompt\n\n<messages>history</messages>"],
        )

    def test_vlm_register_uses_message_injector_instead_of_middleware(self) -> None:
        vlm_init_mod = _load_module_from_path(
            f"_gtbot_test_vlm_init_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "vlm_image" / "__init__.py",
        )
        registry = _FakeRegistry()
        fake_tool_mod = SimpleNamespace(
            vlm_describe_image=object(),
            prewarm_vlm_image_cq_titles=object(),
            inject_vlm_image_cq_titles=object(),
        )

        with patch.object(vlm_init_mod.importlib, "import_module", return_value=fake_tool_mod):
            vlm_init_mod.register(registry)

        self.assertEqual(len(registry.tools), 1)
        self.assertEqual(len(registry.pre_agent_processors), 1)
        self.assertEqual(len(registry.pre_agent_message_injectors), 1)
        self.assertEqual(registry.agent_middlewares, [])

    def test_vlm_injector_only_updates_final_messages(self) -> None:
        _install_gtbot_test_stubs(include_nonebot=True)
        try:
            vlm_tool_mod = _load_module_from_path(
                f"_gtbot_test_vlm_tool_{id(self)}",
                ROOT / "plugins" / "GTBot" / "tools" / "vlm_image" / "tool.py",
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"缺少运行依赖，已跳过: {exc}") from exc
        raw_message = SimpleNamespace(content="[CQ:image,file=demo.png]")
        plugin_ctx = SimpleNamespace(
            runtime_context=SimpleNamespace(bot=object(), raw_messages=[raw_message]),
        )
        original_messages = [vlm_tool_mod.HumanMessage(content="[CQ:image,file=demo.png]")]

        with (
            patch.object(
                vlm_tool_mod.importlib,
                "import_module",
                return_value=SimpleNamespace(
                    get_vlm_image_plugin_config=lambda: SimpleNamespace(max_image_size_bytes=1),
                ),
            ),
            patch.object(
                vlm_tool_mod,
                "_inject_titles_into_text",
                AsyncMock(return_value="[CQ:image,file=demo.png,title=cat,file_size=1]"),
            ),
        ):
            updated_messages = asyncio.run(
                vlm_tool_mod.inject_vlm_image_cq_titles(plugin_ctx, original_messages)
            )

        self.assertEqual(
            [getattr(message, "content", "") for message in updated_messages],
            ["[CQ:image,file=demo.png,title=cat,file_size=1]"],
        )
        self.assertEqual(raw_message.content, "[CQ:image,file=demo.png]")

    def test_vlm_prewarm_reuses_prefetched_payload_and_hash(self) -> None:
        plugin_context_cls, _plugin_context_scope, _registry_mod = _install_gtbot_test_stubs(include_nonebot=True)
        try:
            vlm_tool_mod = _load_module_from_path(
                f"_gtbot_test_vlm_prewarm_{id(self)}",
                ROOT / "plugins" / "GTBot" / "tools" / "vlm_image" / "tool.py",
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"缺少运行依赖，已跳过: {exc}") from exc

        plugin_ctx = plugin_context_cls(
            raw_messages=[{"raw_message": "[CQ:image,file=demo.png]"}],
            runtime_context=SimpleNamespace(bot=object()),
        )
        original_messages = [vlm_tool_mod.HumanMessage(content="[CQ:image,file=demo.png]")]
        cached_record = vlm_tool_mod.CachedImageRecord(
            image_hash="abc123",
            title="cat",
            image_size_bytes=12,
        )

        fake_sha = Mock()
        fake_sha.hexdigest.return_value = "abc123"
        with (
            patch.object(
                vlm_tool_mod.importlib,
                "import_module",
                return_value=SimpleNamespace(
                    get_vlm_image_plugin_config=lambda: SimpleNamespace(max_image_size_bytes=128),
                ),
            ),
            patch.object(
                vlm_tool_mod,
                "_call_onebot_get_image",
                AsyncMock(return_value={"file": "demo.png", "file_size": 12, "url": "https://example.com/demo.png"}),
            ) as call_onebot_get_image,
            patch.object(
                vlm_tool_mod,
                "_find_cached_records_by_size",
                AsyncMock(return_value=[cached_record]),
            ) as find_cached_records_by_size,
            patch.object(
                vlm_tool_mod,
                "_resolve_image_bytes_from_onebot_data",
                AsyncMock(return_value=(b"demo-image", None)),
            ) as resolve_image_bytes,
            patch.object(vlm_tool_mod.hashlib, "sha256", return_value=fake_sha),
        ):
            asyncio.run(vlm_tool_mod.prewarm_vlm_image_cq_titles(plugin_ctx))

        self.assertEqual(
            plugin_ctx.extra["vlm_image_prefetched_payload_cache"]["demo.png"]["file_size"],
            12,
        )
        self.assertEqual(plugin_ctx.extra["vlm_image_prefetched_hash_cache"]["demo.png"], "abc123")
        call_onebot_get_image.assert_awaited_once()
        resolve_image_bytes.assert_awaited_once()
        find_cached_records_by_size.assert_awaited_once()

        with (
            patch.object(
                vlm_tool_mod.importlib,
                "import_module",
                return_value=SimpleNamespace(
                    get_vlm_image_plugin_config=lambda: SimpleNamespace(max_image_size_bytes=128),
                ),
            ),
            patch.object(
                vlm_tool_mod,
                "_call_onebot_get_image",
                AsyncMock(side_effect=AssertionError("injector should reuse prefetched payload")),
            ) as call_onebot_get_image_after_prewarm,
            patch.object(
                vlm_tool_mod,
                "_find_cached_records_by_size",
                AsyncMock(return_value=[cached_record]),
            ),
        ):
            updated_messages = asyncio.run(
                vlm_tool_mod.inject_vlm_image_cq_titles(plugin_ctx, original_messages)
            )

        self.assertEqual(
            [getattr(message, "content", "") for message in updated_messages],
            ["[CQ:image,file=demo.png,title=cat,file_size=12]"],
        )
        call_onebot_get_image_after_prewarm.assert_not_awaited()

    def test_demo_plugin_registers_pre_agent_processor_without_middleware(self) -> None:
        plugin_context_cls, _plugin_context_scope, _registry_mod = _install_gtbot_test_stubs()
        demo_mod = _load_module_from_path(
            f"_gtbot_test_demo_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "demo_plugin.py",
        )
        registry = _FakeRegistry()

        demo_mod.register(registry)
        self.assertEqual(len(registry.pre_agent_processors), 1)
        self.assertEqual(len(registry.callbacks), 1)
        self.assertEqual(registry.agent_middlewares, [])

        plugin_ctx = plugin_context_cls(raw_messages=[1, 2, 3])
        result = registry.pre_agent_processors[0](plugin_ctx)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        self.assertEqual(plugin_ctx.extra["demo_raw_messages_count"], 3)

    def test_debug_llm_memory_callback_logs_once_without_middleware(self) -> None:
        plugin_context_cls, plugin_context_scope, _registry_mod = _install_gtbot_test_stubs(
            include_nonebot=True
        )
        debug_mod = _load_module_from_path(
            f"_gtbot_test_debug_llm_memory_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "debug_llm_memory.py",
        )
        registry = _FakeRegistry()

        debug_mod.register(registry)
        self.assertEqual(len(registry.callbacks), 1)
        self.assertEqual(registry.agent_middlewares, [])

        callback = cast(_CallbackProtocol, registry.callbacks[0])
        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            runtime_context=SimpleNamespace(group_id=123, user_id=456),
        )

        with plugin_context_scope(plugin_ctx):
            callback.on_chain_start({}, {"messages": ["start"]}, run_id="run_1")
            callback.on_chat_model_end(
                SimpleNamespace(
                    generations=[[SimpleNamespace(message=SimpleNamespace(type="ai", content="final response"))]]
                ),
                run_id="run_1",
            )
            callback.on_llm_end(object(), run_id="run_1")

        self.assertTrue(plugin_ctx.extra["debug_llm_memory_start_logged"])
        self.assertTrue(plugin_ctx.extra["debug_llm_memory_final_logged"])
        self.assertEqual(debug_mod.logger.debug.call_count, 2)
        self.assertIn("start", str(debug_mod.logger.debug.call_args_list[0]))
        self.assertIn("final response", str(debug_mod.logger.debug.call_args_list[1]))

    def test_thinking_registers_callback_without_middleware_and_supports_fallbacks(self) -> None:
        plugin_context_cls, plugin_context_scope, _registry_mod = _install_gtbot_test_stubs()
        thinking_mod = _load_module_from_path(
            f"_gtbot_test_thinking_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "thinking.py",
        )
        registry = _FakeRegistry()

        thinking_mod.register(registry)
        self.assertEqual(len(registry.callbacks), 1)
        self.assertEqual(registry.agent_middlewares, [])

        callback = cast(_CallbackProtocol, registry.callbacks[0])

        with patch.object(thinking_mod, "_maybe_add_thinking_emoji") as maybe_add_thinking_emoji:
            callback.on_tool_start({"name": "thinking"}, "")
            self.assertEqual(maybe_add_thinking_emoji.call_count, 1)

            maybe_add_thinking_emoji.reset_mock()
            with plugin_context_scope(plugin_context_cls(raw_messages=[])):
                callback.on_llm_new_token("<")
                callback.on_llm_new_token("thinking")
            self.assertEqual(maybe_add_thinking_emoji.call_count, 1)

            maybe_add_thinking_emoji.reset_mock()
            with plugin_context_scope(plugin_context_cls(raw_messages=[])):
                callback.on_llm_new_token(
                    [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<thinking>structured</thinking>",
                                }
                            ],
                        }
                    ]
                )
            self.assertEqual(maybe_add_thinking_emoji.call_count, 1)

            maybe_add_thinking_emoji.reset_mock()
            callback.on_chat_model_end(
                SimpleNamespace(generations=[[SimpleNamespace(text="<thinking>done</thinking>")]])
            )
            self.assertEqual(maybe_add_thinking_emoji.call_count, 1)

            maybe_add_thinking_emoji.reset_mock()
            callback.on_chat_model_end(
                SimpleNamespace(
                    generations=[
                        [
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=[
                                        {
                                            "type": "text",
                                            "text": "<thinking>structured</thinking>",
                                        }
                                    ]
                                )
                            )
                        ]
                    ]
                )
            )
            self.assertEqual(maybe_add_thinking_emoji.call_count, 1)

    def test_thinking_emoji_uses_event_loop_fallback_without_leaking_coroutine(self) -> None:
        plugin_context_cls, plugin_context_scope, _registry_mod = _install_gtbot_test_stubs()
        thinking_mod = _load_module_from_path(
            f"_gtbot_test_thinking_fallback_{id(self)}",
            ROOT / "plugins" / "GTBot" / "tools" / "thinking.py",
        )

        plugin_ctx = plugin_context_cls(
            raw_messages=[],
            runtime_context=SimpleNamespace(
                bot=object(),
                chat_type="group",
                message_id=123,
                event_loop=None,
            ),
        )
        fake_task = Mock()
        fake_loop = Mock()
        fake_loop.is_closed.return_value = False

        def fake_create_task(coro: Any) -> Any:
            coro.close()
            return fake_task

        def fake_call_soon_threadsafe(callback: Any) -> None:
            callback()

        fake_loop.call_soon_threadsafe.side_effect = fake_call_soon_threadsafe
        plugin_ctx.runtime_context.event_loop = fake_loop

        with (
            plugin_context_scope(plugin_ctx),
            patch.object(thinking_mod.asyncio, "get_running_loop", side_effect=RuntimeError("no running event loop")),
            patch.object(thinking_mod.asyncio, "create_task", side_effect=fake_create_task) as create_task,
        ):
            thinking_mod._maybe_add_thinking_emoji()

        fake_loop.call_soon_threadsafe.assert_called_once()
        create_task.assert_called_once()
        self.assertTrue(plugin_ctx.extra["thinking_emoji_sent"])


if __name__ == "__main__":
    unittest.main()
