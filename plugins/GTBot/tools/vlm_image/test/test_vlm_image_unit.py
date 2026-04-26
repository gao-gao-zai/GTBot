from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。

    这里不走正常包导入，是为了在单测里只加载 `vlm_image.tool` 本身，
    避免被项目其他运行时依赖阻塞。每次加载前调用方应确保相关桩模块
    已提前注入 `sys.modules`。

    Args:
        module_qualname: 目标模块的完整限定名。
        file_path: 模块文件的绝对路径。

    Returns:
        已执行完成的模块对象，供测试直接调用内部函数。

    Raises:
        RuntimeError: 当无法创建模块 spec 或执行器时抛出。
    """
    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_vlm_image_import_stubs() -> None:
    """为 `vlm_image.tool` 安装最小导入桩。

    当前测试只覆盖提示词、XML 解析和结果格式化，因此只提供模块导入时
    必需的最小对象，不模拟任何真实网络或机器人行为。
    """
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
            """返回透传原函数的装饰器，避免影响被测函数签名。"""

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

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    data_root = Path(tempfile.gettempdir()) / "vlm_image_test_data"
    data_root.mkdir(parents=True, exist_ok=True)
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    shared_mod = sys.modules.setdefault("plugins.GTBot.services.shared", ModuleType("plugins.GTBot.services.shared"))
    fun_mod = sys.modules.setdefault("plugins.GTBot.services.shared.fun", ModuleType("plugins.GTBot.services.shared.fun"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    context_mod = sys.modules.setdefault("plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context"))
    setattr(shared_mod, "fun", fun_mod)
    setattr(services_mod, "shared", shared_mod)
    setattr(services_mod, "chat", chat_mod)
    setattr(chat_mod, "context", context_mod)

    class GroupChatContext:
        """提供测试用聊天上下文类型。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)
    setattr(fun_mod, "parse_single_cq", lambda _text: {})
    setattr(fun_mod, "generate_cq_string", lambda _type, data: str(data))

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


class VLMImageQuestionUnitTest(unittest.TestCase):
    tool_mod: ClassVar[ModuleType]

    """覆盖识图插件自定义提问分支的协议层行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载被测模块，供所有测试复用。"""
        _install_vlm_image_import_stubs()
        root = Path(__file__).resolve().parents[5]
        cls.tool_mod = _load_module_from_path(
            "plugins.GTBot.tools.vlm_image.tool",
            str(root / "plugins" / "GTBot" / "tools" / "vlm_image" / "tool.py"),
        )

    def test_build_prompt_should_switch_to_plain_answer_mode_when_question_present(self) -> None:
        """有自定义问题时，提示词应直接要求模型回答问题本身。"""
        prompt = self.tool_mod._build_vlm_prompt("第二个人在做什么？")

        self.assertIn("请先看图，再直接回答用户的补充问题。", prompt)
        self.assertIn("如果图中信息不足以确定答案，请直接明确说明无法从图中确认，不要编造。", prompt)
        self.assertIn("用户问题：第二个人在做什么？", prompt)
        self.assertNotIn("<description>", prompt)
        self.assertNotIn("<title>", prompt)

    def test_parse_should_extract_title_and_description_from_normal_xml(self) -> None:
        """普通识图结果仍应按历史 XML 协议解析标题和描述。"""
        parsed = self.tool_mod._parse_vlm_xml_result(
            "<description>两个人在看手机。</description>"
            "<title>看手机</title>",
        )

        self.assertEqual(parsed.title, "看手机")
        self.assertEqual(parsed.description, "两个人在看手机。")
        self.assertIsNone(parsed.answer)

    def test_parse_should_reject_extra_text_in_normal_xml(self) -> None:
        """普通识图结果若带额外文本，仍应明确报协议错误。"""
        with self.assertRaisesRegex(RuntimeError, "包含额外文本"):
            self.tool_mod._parse_vlm_xml_result(
                "<description>两个人在看手机。</description><title>看手机</title>补充一句",
            )

    def test_format_should_append_answer_line_when_present(self) -> None:
        """格式化结果时仍应兼容显式 answer 字段。"""
        result = self.tool_mod.ImageAnalysisResult(
            title="看手机",
            description="两个人在看手机。",
            answer="第二个人正低头看手机。",
        )

        self.assertEqual(
            self.tool_mod._format_analysis_result(result),
            "标题：看手机\n描述：两个人在看手机。\n回答：第二个人正低头看手机。",
        )


if __name__ == "__main__":
    unittest.main()
