from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


class _FakeMatcher:
    """提供最小行为的命令 matcher 桩对象。"""

    def __init__(self) -> None:
        self.finish = AsyncMock()
        self._handler = None

    def handle(self):
        def decorator(func):
            self._handler = func
            return func

        return decorator


def _install_openai_draw_import_stubs() -> None:
    """为 openai_draw 测试安装最小依赖桩。"""

    if "langchain.tools" not in sys.modules:
        langchain_mod = sys.modules.setdefault("langchain", ModuleType("langchain"))
        tools_mod = ModuleType("langchain.tools")

        class ToolRuntime:
            """测试用 ToolRuntime。"""

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

    nonebot_mod = sys.modules.setdefault("nonebot", ModuleType("nonebot"))
    setattr(nonebot_mod, "logger", SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None))
    setattr(nonebot_mod, "get_driver", lambda: (_ for _ in ()).throw(RuntimeError("not initialized")))
    setattr(nonebot_mod, "on_command", lambda *a, **k: _FakeMatcher())

    adapters_mod = sys.modules.setdefault("nonebot.adapters", ModuleType("nonebot.adapters"))
    onebot_mod = sys.modules.setdefault("nonebot.adapters.onebot", ModuleType("nonebot.adapters.onebot"))
    v11_mod = sys.modules.setdefault("nonebot.adapters.onebot.v11", ModuleType("nonebot.adapters.onebot.v11"))
    v11_event_mod = sys.modules.setdefault(
        "nonebot.adapters.onebot.v11.event", ModuleType("nonebot.adapters.onebot.v11.event")
    )
    v11_message_mod = sys.modules.setdefault(
        "nonebot.adapters.onebot.v11.message", ModuleType("nonebot.adapters.onebot.v11.message")
    )
    setattr(adapters_mod, "onebot", onebot_mod)
    setattr(onebot_mod, "v11", v11_mod)

    class Bot:
        """测试用 Bot 桩。"""

    class MessageEvent:
        """测试用消息事件桩。"""

    class Message:
        """测试用消息对象桩。"""

        def __init__(self, text: str = "") -> None:
            self._text = text

        def extract_plain_text(self) -> str:
            return self._text

    setattr(v11_mod, "Bot", Bot)
    setattr(v11_event_mod, "MessageEvent", MessageEvent)
    setattr(v11_message_mod, "Message", Message)

    params_mod = sys.modules.setdefault("nonebot.params", ModuleType("nonebot.params"))
    setattr(params_mod, "CommandArg", lambda: None)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    data_root = Path(tempfile.gettempdir()) / "openai_draw_test_data"
    data_root.mkdir(parents=True, exist_ok=True)
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    model_mod = sys.modules.setdefault("plugins.GTBot.model", ModuleType("plugins.GTBot.model"))

    class MessageTask:
        """测试用群消息任务。"""

        def __init__(self, messages, group_id, interval) -> None:
            self.messages = messages
            self.group_id = group_id
            self.interval = interval

    setattr(model_mod, "MessageTask", MessageTask)

    cache_mod = sys.modules.setdefault("plugins.GTBot.services.cache", ModuleType("plugins.GTBot.services.cache"))
    setattr(cache_mod, "get_user_cache_manager", AsyncMock(return_value=SimpleNamespace(name="cache")))

    message_mod = sys.modules.setdefault("plugins.GTBot.services.message", ModuleType("plugins.GTBot.services.message"))
    setattr(message_mod, "get_message_manager", AsyncMock(return_value=SimpleNamespace(name="message_manager")))

    help_mod = sys.modules.setdefault("plugins.GTBot.services.help", ModuleType("plugins.GTBot.services.help"))

    class HelpArgumentSpec:
        """测试用帮助参数结构。"""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class HelpCommandSpec:
        """测试用帮助命令结构。"""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    setattr(help_mod, "HelpArgumentSpec", HelpArgumentSpec)
    setattr(help_mod, "HelpCommandSpec", HelpCommandSpec)
    setattr(help_mod, "register_help", lambda spec: spec)

    context_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context")
    )

    class GroupChatContext:
        """测试用聊天上下文。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)

    group_queue_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.group_queue", ModuleType("plugins.GTBot.services.chat.group_queue")
    )
    private_queue_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.private_queue", ModuleType("plugins.GTBot.services.chat.private_queue")
    )
    queue_payload_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.queue_payload", ModuleType("plugins.GTBot.services.chat.queue_payload")
    )

    class PrivateMessageTask:
        """测试用私聊消息任务。"""

        def __init__(self, messages, user_id, interval, session_id) -> None:
            self.messages = messages
            self.user_id = user_id
            self.interval = interval
            self.session_id = session_id

    setattr(group_queue_mod, "group_message_queue_manager", SimpleNamespace(enqueue=AsyncMock()))
    setattr(private_queue_mod, "private_message_queue_manager", SimpleNamespace(enqueue=AsyncMock()))
    setattr(private_queue_mod, "PrivateMessageTask", PrivateMessageTask)
    setattr(queue_payload_mod, "prepare_queue_messages", AsyncMock(side_effect=lambda messages, scope: messages))

    vlm_tool_mod = sys.modules.setdefault("plugins.GTBot.tools.vlm_image.tool", ModuleType("plugins.GTBot.tools.vlm_image.tool"))
    setattr(vlm_tool_mod, "_call_onebot_get_image", AsyncMock(return_value={"file": "source.png"}))
    setattr(
        vlm_tool_mod,
        "_resolve_image_bytes_from_onebot_data",
        AsyncMock(side_effect=[(b"source-bytes", Path("source.png")), (b"mask-bytes", Path("mask.png"))]),
    )

    permission_mod = sys.modules.setdefault(
        "local_plugins.nonebot_plugin_gt_permission", ModuleType("local_plugins.nonebot_plugin_gt_permission")
    )

    class PermissionRole(str, Enum):
        """测试用权限枚举。"""

        USER = "user"
        ADMIN = "admin"

    class PermissionError(Exception):
        """测试用权限异常。"""

    setattr(permission_mod, "PermissionRole", PermissionRole)
    setattr(permission_mod, "PermissionError", PermissionError)
    setattr(permission_mod, "require_admin", AsyncMock())


def _load_openai_draw_package(plugin_dir: str) -> str:
    """加载 openai_draw 测试包而不触发宿主顶层导入链。

    Args:
        plugin_dir: 插件目录路径。

    Returns:
        动态构造出的测试包名。
    """

    _install_openai_draw_import_stubs()
    package_name = f"_openai_draw_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.config", str(Path(plugin_dir) / "config.py"))
    _load_module_from_path(f"{package_name}.client", str(Path(plugin_dir) / "client.py"))
    _load_module_from_path(f"{package_name}.manager", str(Path(plugin_dir) / "manager.py"))
    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    _load_module_from_path(f"{package_name}.commands", str(Path(plugin_dir) / "commands.py"))
    return package_name


class TestOpenAIDrawConfig(unittest.TestCase):
    """验证配置加载、回退与默认文件生成行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_openai_draw_package(plugin_dir)
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    def test_should_create_default_config_when_missing(self) -> None:
        """配置文件不存在时应自动写入默认配置。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_config_example_path", return_value=example_path
            ):
                cfg = self.config_mod.reload_openai_draw_plugin_config()
                self.assertTrue(config_path.exists())
                self.assertTrue(example_path.exists())
                self.assertEqual(cfg.model, "gpt-image-1")
                self.assertEqual(cfg.default_size, "1024x1024")
                self.assertIn("1920x1080", cfg.allowed_sizes)
                self.assertIn("1080x1920", cfg.allowed_sizes)

    def test_invalid_config_should_fallback_to_defaults(self) -> None:
        """非法配置应回退默认值并重写配置文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            config_path.write_text("[]", encoding="utf-8")
            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_config_example_path", return_value=example_path
            ):
                cfg = self.config_mod.reload_openai_draw_plugin_config()
                self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)
                self.assertEqual(parsed["model"], "gpt-image-1")


class TestOpenAIDrawTool(unittest.IsolatedAsyncioTestCase):
    """验证 Agent tool 的参数校验与提交行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_openai_draw_package(plugin_dir)
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    async def test_should_raise_when_prompt_empty(self) -> None:
        """空提示词应被拒绝。"""

        runtime = SimpleNamespace(context=SimpleNamespace())
        with self.assertRaises(ValueError):
            await self.tool_mod.openai_draw_image("", runtime)

    async def test_should_raise_when_runtime_missing(self) -> None:
        """缺少运行时上下文时应抛出异常。"""

        with self.assertRaises(ValueError):
            await self.tool_mod.openai_draw_image("test", SimpleNamespace(context=None))

    async def test_should_raise_when_size_invalid(self) -> None:
        """非法尺寸应被拒绝。"""

        ctx = SimpleNamespace(
            chat_type="group",
            session_id="group:123",
            group_id=123,
            user_id=456,
            bot=object(),
            message_manager=object(),
            cache=object(),
        )
        runtime = SimpleNamespace(context=ctx)
        with patch.object(
            self.tool_mod,
            "get_openai_draw_plugin_config",
            return_value=self.config_mod.OpenAIDrawPluginConfig(),
        ):
            with self.assertRaises(ValueError):
                await self.tool_mod.openai_draw_image("test", runtime, size="2048x2048")

    async def test_edit_should_raise_when_image_missing(self) -> None:
        """编辑图工具在缺少显式原图参数时应抛出异常。"""

        ctx = SimpleNamespace(
            chat_type="group",
            session_id="group:123",
            group_id=123,
            user_id=456,
            bot=object(),
            message_manager=object(),
            cache=object(),
        )
        runtime = SimpleNamespace(context=ctx)
        with self.assertRaises(ValueError):
            await self.tool_mod.openai_edit_image("test", runtime, [])

    async def test_resolve_input_image_should_support_url(self) -> None:
        """显式图片参数为 URL 时应通过下载解析图片内容。"""
        with patch.object(
            self.tool_mod,
            "OpenAIInputImage",
            self.tool_mod.OpenAIInputImage,
        ), patch.dict(
            sys.modules,
            {
                "plugins.GTBot.tools.vlm_image.tool": SimpleNamespace(
                    _call_onebot_get_image=AsyncMock(side_effect=AssertionError("should not call get_image")),
                    _resolve_image_bytes_from_onebot_data=AsyncMock(
                        return_value=(b"source-bytes", Path("source.png"))
                    ),
                )
            },
            clear=False,
        ):
            result = await self.tool_mod._resolve_input_image(
                bot=object(),
                image="https://example.com/source.png",
                max_size_bytes=1024 * 1024,
                parameter_name="image",
            )
        self.assertEqual(result.image_bytes, b"source-bytes")

    async def test_resolve_input_images_should_enforce_max_count(self) -> None:
        """原图列表超过上限时应直接拒绝。"""

        with self.assertRaises(ValueError):
            await self.tool_mod._resolve_input_images(
                bot=object(),
                images=["a", "b"],
                max_size_bytes=1024,
                max_count=1,
            )

    async def test_edit_should_use_explicit_images_and_mask(self) -> None:
        """编辑图工具应使用显式传入的多张原图和遮罩图提交任务。"""

        ctx = SimpleNamespace(
            chat_type="group",
            session_id="group:123",
            group_id=123,
            user_id=456,
            bot=object(),
            message_manager=object(),
            cache=object(),
        )
        runtime = SimpleNamespace(context=ctx)
        manager = SimpleNamespace(
            submit=AsyncMock(return_value=SimpleNamespace(job_id="job-edit")),
            snapshot=AsyncMock(return_value={"queued_count": 0, "running_count": 1}),
        )
        with patch.object(
            self.tool_mod,
            "_resolve_input_images",
            AsyncMock(
                return_value=(
                    self.tool_mod.OpenAIInputImage(file_name="source.png", image_bytes=b"source"),
                    self.tool_mod.OpenAIInputImage(file_name="style.png", image_bytes=b"style"),
                )
            ),
        ) as resolve_images_mock, patch.object(
            self.tool_mod,
            "_resolve_input_image",
            AsyncMock(return_value=self.tool_mod.OpenAIInputImage(file_name="mask.png", image_bytes=b"mask")),
        ) as resolve_image_mock, patch.object(
            self.tool_mod,
            "get_openai_draw_queue_manager",
            return_value=manager,
        ):
            result = await self.tool_mod.openai_edit_image(
                "test",
                runtime,
                ["source-ref", "style-ref"],
                mask="mask-ref",
            )
        self.assertIn("job=job-edit", result)
        self.assertIn("不会阻塞当前智能体", result)
        self.assertEqual(resolve_images_mock.await_count, 1)
        self.assertEqual(resolve_image_mock.await_count, 1)
        submitted_spec = manager.submit.await_args.args[0]
        self.assertEqual(submitted_spec.input_images[0].file_name, "source.png")
        self.assertEqual(submitted_spec.input_images[1].file_name, "style.png")
        self.assertEqual(submitted_spec.mask_image.file_name, "mask.png")


class TestOpenAIDrawManager(unittest.IsolatedAsyncioTestCase):
    """验证队列、保存图片和通知行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_openai_draw_package(plugin_dir)
        cls.manager_mod = __import__(f"{cls.pkg}.manager", fromlist=["dummy"])
        cls.client_mod = __import__(f"{cls.pkg}.client", fromlist=["dummy"])
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    async def test_should_reject_when_queue_full(self) -> None:
        """队列满时应立即拒绝新任务。"""

        cfg = self.config_mod.OpenAIDrawPluginConfig(max_queue_size=1)
        with patch.object(self.manager_mod, "get_openai_draw_plugin_config", return_value=cfg):
            manager = self.manager_mod.OpenAIDrawQueueManager()
            manager._workers_started = True
            spec = self.manager_mod.OpenAIDrawJobSpec(
                chat_type="group",
                session_id="group:1",
                prompt="p1",
                size="1024x1024",
                quality="auto",
                background="auto",
                output_format="png",
                group_id=1,
                requester_user_id=2,
                target_user_id=2,
                bot=object(),
                message_manager=object(),
                cache=object(),
            )
            await manager.submit(spec)
            with self.assertRaises(RuntimeError):
                await manager.submit(spec)

    async def test_should_save_image_when_response_contains_b64(self) -> None:
        """当接口返回 `b64_json` 时应能正确落盘。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self.config_mod.OpenAIDrawPluginConfig(download_dir=temp_dir)
            manager = self.manager_mod.OpenAIDrawQueueManager()
            state = self.manager_mod.OpenAIDrawJobState(
                job_id="job1",
                spec=self.manager_mod.OpenAIDrawJobSpec(
                    chat_type="group",
                    session_id="group:1",
                    prompt="test",
                    size="1024x1024",
                    quality="auto",
                    background="auto",
                    output_format="png",
                    group_id=1,
                    requester_user_id=2,
                    target_user_id=2,
                    bot=object(),
                    message_manager=object(),
                    cache=object(),
                ),
                created_at=1.0,
            )
            response = self.client_mod.OpenAIDrawResponse(
                created=1,
                data=[self.client_mod.OpenAIImageResult(b64_json="aGVsbG8=", url=None, revised_prompt=None)],
                raw_payload={},
            )
            with patch.object(self.manager_mod, "get_openai_draw_plugin_config", return_value=cfg), patch.object(
                self.manager_mod, "OpenAIDrawClient"
            ) as client_cls:
                client_cls.return_value.generate_image = AsyncMock(return_value=response)
                await manager._execute_openai_job(state)
                self.assertIsNotNone(state.result_image_path)
                self.assertTrue(Path(state.result_image_path).exists())
                self.assertEqual(Path(state.result_image_path).read_bytes(), b"hello")

    async def test_should_save_image_when_response_contains_url(self) -> None:
        """当接口返回 `url` 时应能下载并保存图片。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self.config_mod.OpenAIDrawPluginConfig(download_dir=temp_dir)
            manager = self.manager_mod.OpenAIDrawQueueManager()
            state = self.manager_mod.OpenAIDrawJobState(
                job_id="job2",
                spec=self.manager_mod.OpenAIDrawJobSpec(
                    chat_type="group",
                    session_id="group:1",
                    prompt="test",
                    size="1024x1024",
                    quality="auto",
                    background="auto",
                    output_format="png",
                    group_id=1,
                    requester_user_id=2,
                    target_user_id=2,
                    bot=object(),
                    message_manager=object(),
                    cache=object(),
                ),
                created_at=1.0,
            )
            response = self.client_mod.OpenAIDrawResponse(
                created=1,
                data=[
                    self.client_mod.OpenAIImageResult(
                        b64_json=None,
                        url="https://example.com/result.png",
                        revised_prompt=None,
                    )
                ],
                raw_payload={},
            )
            with patch.object(self.manager_mod, "get_openai_draw_plugin_config", return_value=cfg), patch.object(
                self.manager_mod, "OpenAIDrawClient"
            ) as client_cls, patch.object(
                manager, "_download_image_bytes", AsyncMock(return_value=b"png-bytes")
            ):
                client_cls.return_value.generate_image = AsyncMock(return_value=response)
                await manager._execute_openai_job(state)
                self.assertTrue(Path(state.result_image_path).exists())
                self.assertEqual(Path(state.result_image_path).read_bytes(), b"png-bytes")

    async def test_should_call_edit_api_when_mode_is_edit(self) -> None:
        """编辑图任务应调用编辑图接口而不是文生图接口。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self.config_mod.OpenAIDrawPluginConfig(download_dir=temp_dir)
            manager = self.manager_mod.OpenAIDrawQueueManager()
            state = self.manager_mod.OpenAIDrawJobState(
                job_id="job_edit",
                spec=self.manager_mod.OpenAIDrawJobSpec(
                    chat_type="group",
                    session_id="group:1",
                    prompt="edit test",
                    size="1024x1024",
                    quality="auto",
                    background="auto",
                    input_fidelity="high",
                    output_format="png",
                    mode="edit",
                    input_images=(
                        self.manager_mod.OpenAIInputImage(file_name="source.png", image_bytes=b"source"),
                    ),
                    mask_image=self.manager_mod.OpenAIInputImage(file_name="mask.png", image_bytes=b"mask"),
                    group_id=1,
                    requester_user_id=2,
                    target_user_id=2,
                    bot=object(),
                    message_manager=object(),
                    cache=object(),
                ),
                created_at=1.0,
            )
            response = self.client_mod.OpenAIDrawResponse(
                created=1,
                data=[self.client_mod.OpenAIImageResult(b64_json="aGVsbG8=", url=None, revised_prompt=None)],
                raw_payload={},
            )
            with patch.object(self.manager_mod, "get_openai_draw_plugin_config", return_value=cfg), patch.object(
                self.manager_mod, "OpenAIDrawClient"
            ) as client_cls:
                client_cls.return_value.edit_image = AsyncMock(return_value=response)
                client_cls.return_value.generate_image = AsyncMock()
                await manager._execute_openai_job(state)
                client_cls.return_value.edit_image.assert_awaited()
                client_cls.return_value.generate_image.assert_not_called()
                self.assertEqual(Path(state.result_image_path).read_bytes(), b"hello")

    async def test_should_enqueue_group_notification_on_success(self) -> None:
        """成功任务应构造群聊通知并入队。"""

        manager = self.manager_mod.OpenAIDrawQueueManager()
        state = self.manager_mod.OpenAIDrawJobState(
            job_id="job3",
            spec=self.manager_mod.OpenAIDrawJobSpec(
                chat_type="group",
                session_id="group:1",
                prompt="test",
                size="1024x1024",
                quality="auto",
                background="auto",
                output_format="png",
                group_id=123,
                requester_user_id=456,
                target_user_id=456,
                bot=object(),
                message_manager=object(),
                cache=object(),
            ),
            created_at=1.0,
            status="succeeded",
            result_image_path="C:/tmp/result.png",
        )
        enqueue_mock = self.manager_mod.group_message_queue_manager.enqueue
        enqueue_mock.reset_mock()
        await manager._notify_group(state)
        enqueue_mock.assert_awaited()
        task = enqueue_mock.await_args.args[0]
        self.assertIn("[绘图完成]", task.messages[0])
        self.assertIn("[CQ:image,file=C:/tmp/result.png]", task.messages[1])

    async def test_should_reject_private_target_user_override(self) -> None:
        """私聊场景不应允许给第三方发图。"""

        tool_mod = __import__(f"{self.pkg}.tool", fromlist=["dummy"])
        ctx = SimpleNamespace(
            chat_type="private",
            session_id="private:456",
            group_id=None,
            user_id=456,
            bot=object(),
            message_manager=object(),
            cache=object(),
        )
        runtime = SimpleNamespace(context=ctx)
        with self.assertRaises(ValueError):
            await tool_mod.openai_draw_image("test", runtime, target_user_id=789)


class TestOpenAIDrawCommands(unittest.IsolatedAsyncioTestCase):
    """验证命令层的任务摘要输出。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_openai_draw_package(plugin_dir)
        cls.commands_mod = __import__(f"{cls.pkg}.commands", fromlist=["dummy"])

    async def test_draw_tasks_command_should_render_snapshot(self) -> None:
        """`/绘图任务` 应输出运行中和排队任务摘要。"""

        event = SimpleNamespace(user_id=123)
        snapshot = {
            "running_count": 1,
            "queued_count": 1,
            "queue_max": 10,
            "running": [
                {
                    "job_id": "job_running",
                    "status": "running",
                    "size": "1024x1024",
                    "quality": "auto",
                    "target_user_id": 123,
                    "prompt_preview": "running prompt",
                }
            ],
            "queued": [
                {
                    "job_id": "job_queued",
                    "status": "queued",
                    "size": "1536x1024",
                    "quality": "high",
                    "target_user_id": 123,
                    "prompt_preview": "queued prompt",
                }
            ],
        }
        finish_mock = AsyncMock()
        with patch.object(self.commands_mod, "get_openai_draw_queue_manager", return_value=SimpleNamespace(snapshot=AsyncMock(return_value=snapshot))), patch.object(
            self.commands_mod.DrawTasksCommand, "finish", finish_mock
        ):
            await self.commands_mod.handle_draw_tasks_command(event)
        self.assertIsNotNone(finish_mock.await_args)
        rendered = finish_mock.await_args.args[0]
        self.assertIn("running=1 queued=1/10", rendered)
        self.assertIn("job_running", rendered)
        self.assertIn("job_queued", rendered)


class TestOpenAIDrawClient(unittest.IsolatedAsyncioTestCase):
    """验证客户端的网络异常提示与重试行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_openai_draw_package(plugin_dir)
        cls.client_mod = __import__(f"{cls.pkg}.client", fromlist=["dummy"])
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    async def test_request_error_should_include_exception_type(self) -> None:
        """请求建立失败时，错误文本应包含异常类型和 URL。"""

        client = self.client_mod.OpenAIDrawClient(
            self.config_mod.OpenAIDrawPluginConfig(api_key="test-key")
        )
        with patch.object(
            client,
            "_post_once",
            AsyncMock(side_effect=self.client_mod.OpenAIDrawClientError("请求绘图网关失败: ConnectError: boom url=https://example.com/v1/images/generations")),
        ):
            with self.assertRaises(self.client_mod.OpenAIDrawClientError) as ctx:
                await client.generate_image(
                    prompt="test",
                    size="1024x1024",
                    quality="auto",
                    background="auto",
                    output_format="png",
                )
        self.assertIn("ConnectError", str(ctx.exception))
        self.assertIn("url=", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
