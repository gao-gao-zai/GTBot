from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch
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


def _install_import_stubs() -> None:
    """为表达式求解器插件测试安装最小依赖桩。"""

    if "langchain.tools" not in sys.modules:
        langchain_mod = sys.modules.setdefault("langchain", ModuleType("langchain"))
        tools_mod = ModuleType("langchain.tools")

        class ToolRuntime:
            """测试桩版 `ToolRuntime`。"""

            def __init__(self, context=None) -> None:
                self.context = context

            def __class_getitem__(cls, _item):
                return cls

        def tool(name: str):
            """构造最小 `@tool` 装饰器桩。

            Args:
                name: 工具名。

            Returns:
                一个会给函数附加基础元数据的装饰器。
            """

            def decorator(func):
                func.name = name
                func.description = func.__doc__ or ""
                return func

            return decorator

        setattr(tools_mod, "ToolRuntime", ToolRuntime)
        setattr(tools_mod, "tool", tool)
        sys.modules["langchain.tools"] = tools_mod
        setattr(langchain_mod, "tools", tools_mod)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    context_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.context",
        ModuleType("plugins.GTBot.services.chat.context"),
    )
    setattr(plugins_mod, "GTBot", gtbot_mod)
    setattr(gtbot_mod, "services", services_mod)
    setattr(services_mod, "chat", chat_mod)
    setattr(chat_mod, "context", context_mod)

    class GroupChatContext:
        """测试桩版 `GroupChatContext`。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)


def _load_python_expression_solver_package(plugin_dir: str) -> str:
    """加载表达式求解器插件测试包而不经过宿主顶层导入链。

    Args:
        plugin_dir: 插件目录绝对路径。

    Returns:
        当前测试专用的动态包名。
    """

    _install_import_stubs()
    package_name = f"_python_expression_solver_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.config", str(Path(plugin_dir) / "config.py"))
    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    _load_module_from_path(f"{package_name}", str(Path(plugin_dir) / "__init__.py"))
    return package_name


class TestPythonExpressionSolverConfig(unittest.TestCase):
    """验证插件配置加载、回退与默认文件生成行为。"""

    pkg: ClassVar[str]
    config_mod: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_python_expression_solver_package(plugin_dir)
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])

    def test_invalid_config_should_fallback_to_defaults(self) -> None:
        """非法配置应回退到默认值并重写配置文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            config_path.write_text("[]", encoding="utf-8")

            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_example_path", return_value=example_path
            ):
                cfg = self.config_mod.reload_python_expression_solver_plugin_config()
                self.assertTrue(cfg.enabled)
                self.assertEqual(cfg.max_user_result_length_cap, 100)
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)

    def test_config_should_read_custom_user_cap(self) -> None:
        """配置文件中的用户上限应能被正确读取。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            example_path = root / "config.json.example"
            config_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "max_user_result_length_cap": 256,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(self.config_mod, "_config_path", return_value=config_path), patch.object(
                self.config_mod, "_example_path", return_value=example_path
            ):
                cfg = self.config_mod.reload_python_expression_solver_plugin_config()
                self.assertEqual(cfg.max_user_result_length_cap, 256)


class _FakeRegistry:
    """提供最小行为的注册器测试桩。"""

    def __init__(self) -> None:
        """初始化一个仅记录已注册工具的测试注册器。"""

        self.tools: list[Any] = []

    def add_tool(self, tool: Any) -> None:
        """记录一次工具注册。

        Args:
            tool: 待注册的工具对象。
        """

        self.tools.append(tool)


class TestPythonExpressionSolverTool(unittest.TestCase):
    """验证表达式求值、安全限制与工具注册描述。"""

    pkg: ClassVar[str]
    tool_mod: ClassVar[ModuleType]
    init_mod: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_python_expression_solver_package(plugin_dir)
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])
        cls.init_mod = __import__(cls.pkg)

    def test_solver_should_evaluate_math_expression(self) -> None:
        """应能求值白名单数学表达式。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("math.sqrt(16) + sum([1, 2, 3])")
        self.assertEqual(result, "10.0")

    def test_solver_should_evaluate_mpmath_expression(self) -> None:
        """应能求值白名单 `mpmath` 表达式。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("mpmath.nstr(mpmath.pi, 6)")
        self.assertEqual(result, "3.14159")

    def test_solver_should_support_conditional_expression(self) -> None:
        """应支持表达式级三元条件分支。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("'pos' if 3 > 2 else 'neg'")
        self.assertEqual(result, "pos")

    def test_solver_should_support_boolean_and_comparison_expression(self) -> None:
        """应支持条件表达式所需的布尔与比较运算。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("1 if (1 < 2 and not False) else 0")
        self.assertEqual(result, "1")

    def test_solver_should_support_chained_comparison(self) -> None:
        """应支持 Python 原生链式比较语法。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("1 if 1 < 2 <= 2 else 0")
        self.assertEqual(result, "1")

    def test_solver_should_support_subscript_access(self) -> None:
        """应支持列表与字典的下标访问。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl('{"a": [1, 2, 3]}["a"][1]')
        self.assertEqual(result, "2")

    def test_solver_should_support_slice_access(self) -> None:
        """应支持字符串或序列切片。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl('"abcdef"[1:5:2]')
        self.assertEqual(result, "bd")

    def test_solver_should_support_membership_and_identity_comparison(self) -> None:
        """应支持成员测试与身份比较。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl('1 if ("a" in {"a": 1} and None is None) else 0')
        self.assertEqual(result, "1")

    def test_solver_should_support_bitwise_operations(self) -> None:
        """应支持常见位运算表达式。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("(5 & 3) | (1 << 3)")
        self.assertEqual(result, "9")

    def test_solver_should_return_syntax_error_message(self) -> None:
        """语法错误应返回可展示的错误文本。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("1 +")
        self.assertIn("表达式语法错误", result)

    def test_solver_should_reject_unsafe_name(self) -> None:
        """应把未开放名称访问转换成错误文本返回。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("__import__('os').system('dir')")
        self.assertIn("只允许访问白名单模块的单层公开属性", result)

    def test_solver_should_reject_non_module_attribute_chain(self) -> None:
        """应把非法属性访问转换成错误文本返回。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl("'abc'.upper()")
        self.assertIn("只允许访问白名单模块的单层公开属性", result)

    def test_solver_should_return_error_when_result_too_long(self) -> None:
        """结果长度超过本次上限时应返回错误文本而不是截断。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=100),
        ):
            result = self.tool_mod.solve_python_expression_impl(
                "'a' * 8",
                max_output_length=5,
            )
        self.assertIn("结果超出本次允许的最大返回长度", result)
        self.assertIn("result_length=8", result)
        self.assertIn("limit=5", result)

    def test_solver_should_return_error_when_limit_above_user_cap(self) -> None:
        """本次上限超过配置硬上限时应返回错误文本。"""

        with patch.object(
            self.tool_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(enabled=True, max_user_result_length_cap=20),
        ):
            result = self.tool_mod.solve_python_expression_impl("1 + 1", max_output_length=21)
        self.assertIn("max_output_length 超过当前用户配置允许的最大上限", result)

    def test_register_should_publish_dynamic_description(self) -> None:
        """注册阶段应把当前配置上限写入工具描述。"""

        fake_registry = _FakeRegistry()
        with patch.object(
            self.init_mod,
            "get_python_expression_solver_plugin_config",
            return_value=SimpleNamespace(max_user_result_length_cap=321),
        ):
            self.init_mod.register(fake_registry)

        self.assertEqual(len(fake_registry.tools), 1)
        description = getattr(fake_registry.tools[0], "description", "")
        self.assertIn("321", description)
        self.assertIn("默认 50", description)
        self.assertIn("最高上限", description)
        self.assertIn("mpmath", description)
        self.assertIn("math`->`math", description)
        self.assertIn("decimal`->`decimal", description)
        self.assertIn("下标", description)
        self.assertIn("位运算", description)

    def test_tool_signature_should_not_require_runtime_context(self) -> None:
        """工具公开签名不应再声明运行时上下文参数。"""

        params = signature(self.tool_mod.solve_python_expression).parameters
        self.assertEqual(tuple(params), ("expression", "max_output_length"))


if __name__ == "__main__":
    unittest.main()
