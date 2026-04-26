from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, patch


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。"""

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_vlm_image_import_stubs() -> dict[str, Path]:
    """为 `vlm_image.tool` 安装最小导入桩。"""

    langchain_mod = sys.modules.setdefault("langchain", ModuleType("langchain"))
    if "langchain.tools" not in sys.modules:
        tools_mod = ModuleType("langchain.tools")

        class ToolRuntime:
            """提供最小行为的测试版 `ToolRuntime`。"""

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

    if "langchain_core.messages" not in sys.modules:
        messages_mod = ModuleType("langchain_core.messages")

        class BaseMessage:
            """提供与被测模块兼容的最小消息基类。"""

            def __init__(self, content=None) -> None:
                self.content = content

            def model_copy(self, update=None):
                payload = {"content": self.content}
                if isinstance(update, dict):
                    payload.update(update)
                return self.__class__(content=payload.get("content"))

        class HumanMessage(BaseMessage):
            """提供测试用的人类消息类型。"""

        setattr(messages_mod, "BaseMessage", BaseMessage)
        setattr(messages_mod, "HumanMessage", HumanMessage)
        sys.modules["langchain_core.messages"] = messages_mod

    sys.modules.setdefault("aiosqlite", ModuleType("aiosqlite"))
    sys.modules.setdefault("httpx", ModuleType("httpx"))
    sys.modules.setdefault("requests", ModuleType("requests"))

    nonebot_mod = sys.modules.setdefault("nonebot", ModuleType("nonebot"))
    setattr(
        nonebot_mod,
        "logger",
        SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    )

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    data_root = Path(tempfile.gettempdir()) / "vlm_image_test_data"
    data_root.mkdir(parents=True, exist_ok=True)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    shared_mod = sys.modules.setdefault("plugins.GTBot.services.shared", ModuleType("plugins.GTBot.services.shared"))
    fun_mod = sys.modules.setdefault("plugins.GTBot.services.shared.fun", ModuleType("plugins.GTBot.services.shared.fun"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    context_mod = sys.modules.setdefault("plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context"))
    file_registry_mod = sys.modules.setdefault(
        "plugins.GTBot.services.file_registry", ModuleType("plugins.GTBot.services.file_registry")
    )
    setattr(shared_mod, "fun", fun_mod)
    setattr(services_mod, "shared", shared_mod)
    setattr(services_mod, "chat", chat_mod)
    setattr(services_mod, "file_registry", file_registry_mod)
    setattr(chat_mod, "context", context_mod)

    class GroupChatContext:
        """提供测试用聊天上下文类型。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)
    setattr(fun_mod, "parse_single_cq", lambda _text: {})
    setattr(fun_mod, "generate_cq_string", lambda _type, data: str(data))

    sample_image = data_root / "sample.png"
    sample_image.write_bytes(b"sample-image")

    def resolve_file(file_id: str):
        if file_id != "gtfile:test-image":
            raise FileNotFoundError(file_id)
        return SimpleNamespace(local_path=sample_image, mime_type="image/png", extra={})

    setattr(file_registry_mod, "resolve_file", resolve_file)

    tools_pkg = sys.modules.setdefault("plugins.GTBot.tools", ModuleType("plugins.GTBot.tools"))
    vlm_image_pkg = sys.modules.setdefault("plugins.GTBot.tools.vlm_image", ModuleType("plugins.GTBot.tools.vlm_image"))
    setattr(tools_pkg, "vlm_image", vlm_image_pkg)

    config_mod = sys.modules.setdefault("plugins.GTBot.tools.vlm_image.config", ModuleType("plugins.GTBot.tools.vlm_image.config"))
    setattr(
        config_mod,
        "get_vlm_image_plugin_config",
        lambda: SimpleNamespace(
            title_recommended_chars=10,
            title_max_chars=20,
            description_recommended_chars=None,
            description_max_chars=300,
            max_image_size_bytes=5 * 1024 * 1024,
        ),
    )
    return {"sample_image": sample_image}


class VLMImageQuestionUnitTest(unittest.TestCase):
    tool_mod: ClassVar[ModuleType]

    """覆盖识图插件协议层的非 IO 行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        _install_vlm_image_import_stubs()
        root = Path(__file__).resolve().parents[5]
        cls.tool_mod = _load_module_from_path(
            "plugins.GTBot.tools.vlm_image.tool",
            str(root / "plugins" / "GTBot" / "tools" / "vlm_image" / "tool.py"),
        )

    def test_build_prompt_should_switch_to_plain_answer_mode_when_question_present(self) -> None:
        prompt = self.tool_mod._build_vlm_prompt("第二个人在做什么？")

        self.assertIn("请先看图，再直接回答用户的补充问题。", prompt)
        self.assertIn("如果图中信息不足以确定答案，请直接明确说明无法从图中确认，不要编造。", prompt)
        self.assertIn("用户问题：第二个人在做什么？", prompt)
        self.assertNotIn("<description>", prompt)
        self.assertNotIn("<title>", prompt)

    def test_parse_should_extract_title_and_description_from_normal_xml(self) -> None:
        parsed = self.tool_mod._parse_vlm_xml_result(
            "<description>两个人在看手机。</description>"
            "<title>看手机</title>",
        )

        self.assertEqual(parsed.title, "看手机")
        self.assertEqual(parsed.description, "两个人在看手机。")
        self.assertIsNone(parsed.answer)

    def test_parse_should_reject_extra_text_in_normal_xml(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "包含额外文本"):
            self.tool_mod._parse_vlm_xml_result(
                "<description>两个人在看手机。</description><title>看手机</title>补充一句",
            )

    def test_format_should_append_answer_line_when_present(self) -> None:
        result = self.tool_mod.ImageAnalysisResult(
            title="看手机",
            description="两个人在看手机。",
            answer="第二个人正低头看手机。",
        )

        self.assertEqual(
            self.tool_mod._format_analysis_result(result),
            "标题：看手机\n描述：两个人在看手机。\n回答：第二个人正低头看手机。",
        )


class VLMImageFileIdUnitTest(unittest.IsolatedAsyncioTestCase):
    tool_mod: ClassVar[ModuleType]

    """覆盖 `file_id` 输入协议。"""

    @classmethod
    def setUpClass(cls) -> None:
        _install_vlm_image_import_stubs()
        root = Path(__file__).resolve().parents[5]
        cls.tool_mod = _load_module_from_path(
            "plugins.GTBot.tools.vlm_image.tool",
            str(root / "plugins" / "GTBot" / "tools" / "vlm_image" / "tool.py"),
        )

    async def test_vlm_describe_image_should_read_from_file_id_without_get_image(self) -> None:
        runtime = SimpleNamespace(context=SimpleNamespace(bot=object()))
        with patch.object(
            self.tool_mod,
            "_call_onebot_get_image",
            AsyncMock(side_effect=AssertionError("should not call get_image")),
        ), patch.object(
            self.tool_mod,
            "_get_cached_result",
            AsyncMock(return_value=None),
        ), patch.object(
            self.tool_mod,
            "_upsert_cached_result",
            AsyncMock(),
        ), patch.object(
            self.tool_mod,
            "_call_vlm_api",
            AsyncMock(return_value="<description>测试描述</description><title>测试标题</title>"),
        ):
            result = await self.tool_mod.vlm_describe_image("gtfile:test-image", runtime)
        self.assertIn("标题：测试标题", result)
        self.assertIn("描述：测试描述", result)


if __name__ == "__main__":
    unittest.main()
