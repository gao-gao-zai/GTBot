from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Awaitable, Callable, ClassVar, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。

    Args:
        module_qualname: 目标模块完整限定名。
        file_path: 模块文件绝对路径。

    Returns:
        已执行完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_markdown_image_import_stubs() -> dict[str, Any]:
    """为 Markdown 图片插件测试安装最小依赖桩。

    Returns:
        保存已注册 GT 文件映射的内存字典，供断言使用。
    """

    if "langchain.tools" not in sys.modules:
        langchain_mod = sys.modules.setdefault("langchain", ModuleType("langchain"))
        tools_mod = ModuleType("langchain.tools")

        class ToolRuntime:
            """测试用 `ToolRuntime`。"""

            def __init__(self, context=None) -> None:
                self.context = context

            def __class_getitem__(cls, _item):
                return cls

        def tool(_name: str):
            def decorator(func):
                return func

            return decorator

        setattr(tools_mod, "ToolRuntime", ToolRuntime)
        setattr(tools_mod, "tool", tool)
        sys.modules["langchain.tools"] = tools_mod
        setattr(langchain_mod, "tools", tools_mod)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    data_root = Path(tempfile.gettempdir()) / "markdown_image_test_data"
    data_root.mkdir(parents=True, exist_ok=True)
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    plugin_config_mod = sys.modules.setdefault(
        "plugins.GTBot.tools.markdown_image.config",
        ModuleType("plugins.GTBot.tools.markdown_image.config"),
    )
    setattr(
        plugin_config_mod,
        "get_markdown_image_plugin_config",
        lambda: SimpleNamespace(
            render=SimpleNamespace(
                auto_width=True,
                width=None,
                min_width=560,
                max_width=1200,
                padding=32,
                scale=2.0,
                theme="default",
                code_theme="default",
                custom_css="",
            )
        ),
    )

    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    context_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context")
    )
    setattr(services_mod, "chat", chat_mod)
    setattr(chat_mod, "context", context_mod)

    class GroupChatContext:
        """测试用聊天上下文。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)

    nonebot_mod = sys.modules.setdefault("nonebot", ModuleType("nonebot"))
    adapters_mod = sys.modules.setdefault("nonebot.adapters", ModuleType("nonebot.adapters"))
    onebot_mod = sys.modules.setdefault("nonebot.adapters.onebot", ModuleType("nonebot.adapters.onebot"))
    v11_mod = sys.modules.setdefault("nonebot.adapters.onebot.v11", ModuleType("nonebot.adapters.onebot.v11"))
    onebot_message_mod = sys.modules.setdefault(
        "nonebot.adapters.onebot.v11.message",
        ModuleType("nonebot.adapters.onebot.v11.message"),
    )
    setattr(nonebot_mod, "adapters", adapters_mod)
    setattr(adapters_mod, "onebot", onebot_mod)
    setattr(onebot_mod, "v11", v11_mod)
    setattr(v11_mod, "message", onebot_message_mod)

    class Message:
        """测试用消息对象。"""

        def __init__(self, segment: object) -> None:
            self.segment = segment

    class MessageSegment:
        """测试用消息片段工厂。"""

        @staticmethod
        def image(*, file: str) -> str:
            return f"[CQ:image,file={file}]"

    setattr(onebot_message_mod, "Message", Message)
    setattr(onebot_message_mod, "MessageSegment", MessageSegment)

    registry: dict[str, Any] = {}
    file_registry_mod = sys.modules.setdefault(
        "plugins.GTBot.services.file_registry", ModuleType("plugins.GTBot.services.file_registry")
    )

    def register_local_file(path: str | Path, **kwargs) -> str:
        file_id = f"gfid:{uuid4().hex[:12]}"
        registry[file_id] = SimpleNamespace(local_path=Path(path).resolve(), kwargs=kwargs)
        return file_id

    setattr(file_registry_mod, "register_local_file", register_local_file)
    setattr(file_registry_mod, "_registry_store", registry)
    return registry


def _load_markdown_image_package(plugin_dir: str) -> str:
    """加载 Markdown 图片插件测试包而不触发宿主顶层导入链。

    Args:
        plugin_dir: 插件目录绝对路径。

    Returns:
        当前测试专用的包名。
    """

    _install_markdown_image_import_stubs()
    package_name = f"_markdown_image_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    config_path = str(Path(plugin_dir) / "config.py")
    if Path(config_path).exists():
        _load_module_from_path(f"{package_name}.config", config_path)
    _load_module_from_path(f"{package_name}.renderer", str(Path(plugin_dir) / "renderer.py"))
    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    return package_name


def _get_async_tool_callable(tool_obj: object) -> Callable[..., Awaitable[Any]]:
    """返回测试可直接 `await` 的异步工具实现。

    Args:
        tool_obj: `@tool` 装饰后的工具对象或原始函数。

    Returns:
        可直接在测试中等待执行结果的协程函数。
    """

    return cast(Callable[..., Awaitable[Any]], getattr(tool_obj, "coroutine", tool_obj))


class TestMarkdownImageTool(unittest.IsolatedAsyncioTestCase):
    """验证 Markdown 图片工具的核心行为。"""

    pkg: ClassVar[str]
    tool_mod: ClassVar[ModuleType]
    renderer_mod: ClassVar[ModuleType]
    registry: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        _install_markdown_image_import_stubs()
        cls.pkg = _load_markdown_image_package(plugin_dir)
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])
        cls.renderer_mod = __import__(f"{cls.pkg}.renderer", fromlist=["dummy"])
        file_registry_mod = sys.modules["plugins.GTBot.services.file_registry"]
        cls.registry = cast(dict[str, Any], getattr(file_registry_mod, "_registry_store"))

    async def test_send_markdown_image_should_register_file_and_send_image(self) -> None:
        """发送 Markdown 图片时应注册 GT 文件并回发图片消息。"""

        output_file = Path(tempfile.gettempdir()) / "markdown_image_render_test.png"
        output_file.write_bytes(b"png")
        render_result = self.renderer_mod.MarkdownRenderResult(
            image_path=output_file,
            width=960,
            height=540,
        )
        bot = SimpleNamespace(send=AsyncMock())
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                bot=bot,
                event=object(),
                session_id="group:123",
                group_id=123,
                user_id=456,
            )
        )

        with patch.object(
            self.tool_mod,
            "render_markdown_to_image",
            AsyncMock(return_value=render_result),
        ) as render_mock:
            result = await _get_async_tool_callable(self.tool_mod.send_markdown_image)(
                "## 标题\n\n- 列表",
                runtime=runtime,
            )

        self.assertIn("Markdown 图片已发送", result)
        self.assertIn("GT文件=gfid:", result)
        bot.send.assert_awaited_once()
        render_mock.assert_awaited_once()
        await_args = render_mock.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertTrue(await_args.kwargs["auto_width"])
        self.assertEqual(await_args.kwargs["theme"], "default")
        self.assertEqual(await_args.kwargs["code_theme"], "default")
        self.assertEqual(len(self.registry), 1)
        handle = next(iter(self.registry.values()))
        self.assertEqual(handle.kwargs["kind"], "markdown_image")
        self.assertEqual(handle.kwargs["extra"]["render_width"], 960)
        self.assertEqual(handle.kwargs["extra"]["render_height"], 540)

    async def test_send_markdown_image_should_use_transport_when_event_missing(self) -> None:
        """自动触发缺少 `event` 时应优先走 transport 发送图片。"""

        output_file = Path(tempfile.gettempdir()) / "markdown_image_render_test_transport.png"
        output_file.write_bytes(b"png")
        render_result = self.renderer_mod.MarkdownRenderResult(
            image_path=output_file,
            width=800,
            height=600,
        )
        transport = SimpleNamespace(send_messages=AsyncMock())
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                bot=SimpleNamespace(send=AsyncMock()),
                event=None,
                transport=transport,
                session_id="group:auto",
                group_id=123,
                user_id=456,
            )
        )

        with patch.object(
            self.tool_mod,
            "render_markdown_to_image",
            AsyncMock(return_value=render_result),
        ):
            await _get_async_tool_callable(self.tool_mod.send_markdown_image)(
                "$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$",
                runtime=runtime,
            )

        transport.send_messages.assert_awaited_once()
        runtime.context.bot.send.assert_not_awaited()

    async def test_send_markdown_image_should_forward_theme_and_custom_css(self) -> None:
        """工具层应把插件配置中的样式参数透传给渲染器。"""

        output_file = Path(tempfile.gettempdir()) / "markdown_image_render_test_theme.png"
        output_file.write_bytes(b"png")
        render_result = self.renderer_mod.MarkdownRenderResult(
            image_path=output_file,
            width=720,
            height=400,
        )
        bot = SimpleNamespace(send=AsyncMock())
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                bot=bot,
                event=object(),
                session_id="group:123",
                group_id=123,
                user_id=456,
            )
        )

        with patch.object(
            self.tool_mod,
            "render_markdown_to_image",
            AsyncMock(return_value=render_result),
        ) as render_mock, patch.object(
            self.tool_mod,
            "get_markdown_image_plugin_config",
            return_value=SimpleNamespace(
                render=SimpleNamespace(
                    auto_width=True,
                    width=None,
                    min_width=500,
                    max_width=900,
                    padding=24,
                    scale=2.5,
                    theme="glass",
                    code_theme="light",
                    custom_css="h1 { color: red; }",
                )
            ),
        ) as config_mock:
            await _get_async_tool_callable(self.tool_mod.send_markdown_image)(
                "```python\nprint('hi')\n```",
                runtime=runtime,
            )

        config_mock.assert_called_once()
        await_args = render_mock.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.kwargs["min_width"], 500)
        self.assertEqual(await_args.kwargs["max_width"], 900)
        self.assertEqual(await_args.kwargs["padding"], 24)
        self.assertEqual(await_args.kwargs["scale"], 2.5)
        self.assertEqual(await_args.kwargs["theme"], "glass")
        self.assertEqual(await_args.kwargs["code_theme"], "light")
        self.assertEqual(await_args.kwargs["custom_css"], "h1 { color: red; }")

    async def test_send_markdown_image_should_raise_when_send_context_missing(self) -> None:
        """既无 transport 又无完整 `bot/event` 时应抛出异常。"""

        output_file = Path(tempfile.gettempdir()) / "markdown_image_render_test_missing_context.png"
        output_file.write_bytes(b"png")
        render_result = self.renderer_mod.MarkdownRenderResult(
            image_path=output_file,
            width=640,
            height=360,
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                bot=None,
                event=None,
                transport=None,
            )
        )

        with patch.object(
            self.tool_mod,
            "render_markdown_to_image",
            AsyncMock(return_value=render_result),
        ):
            with self.assertRaises(ValueError):
                await _get_async_tool_callable(self.tool_mod.send_markdown_image)(
                    "hello",
                    runtime=runtime,
                )


if __name__ == "__main__":
    unittest.main()
