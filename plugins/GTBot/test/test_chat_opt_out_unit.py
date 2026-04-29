from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]


def _ensure_package(name: str, path: Path) -> ModuleType:
    """确保测试加载链路上的包对象存在于 `sys.modules`。

    Args:
        name: 包名。
        path: 包对应的目录路径。

    Returns:
        已注册到 `sys.modules` 的包模块对象。
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
    """从指定路径加载测试目标模块。

    Args:
        module_name: 注册到 `sys.modules` 的模块名。
        file_path: 模块文件路径。

    Returns:
        已执行完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_name} -> {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestChatOptOutUnit(unittest.TestCase):
    """覆盖群关键词免触发规则的核心匹配行为。"""

    mod: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls) -> None:
        """安装最小宿主桩并加载免触发模块。

        该模块依赖 `plugins.GTBot.ConfigManager` 和 `plugins.GTBot.Logger`。
        单测使用轻量桩对象代替真实宿主，避免引入 NoneBot 初始化和完整配置加载。
        """

        _ensure_package("plugins", ROOT / "plugins")
        _ensure_package("plugins.GTBot", ROOT / "plugins" / "GTBot")
        _ensure_package("plugins.GTBot.services", ROOT / "plugins" / "GTBot" / "services")
        _ensure_package(
            "plugins.GTBot.services.trigger",
            ROOT / "plugins" / "GTBot" / "services" / "trigger",
        )

        logger_module = ModuleType("plugins.GTBot.Logger")
        setattr(logger_module, "logger", SimpleNamespace(warning=lambda *args, **kwargs: None))
        sys.modules["plugins.GTBot.Logger"] = logger_module

        chat_opt_out_cfg = SimpleNamespace(
            enabled=True,
            rules=[
                SimpleNamespace(id="suffix_rule", enabled=True, type="suffix", value="#别回"),
                SimpleNamespace(
                    id="expr_rule",
                    enabled=True,
                    type="expr",
                    value="contains_any(normalized_text, ['不聊天']) and not mentioned_bot",
                ),
            ],
        )
        config_manager_module = ModuleType("plugins.GTBot.ConfigManager")
        setattr(
            config_manager_module,
            "total_config",
            SimpleNamespace(
                processed_configuration=SimpleNamespace(
                    current_config_group=SimpleNamespace(
                        chat_model=SimpleNamespace(chat_opt_out=chat_opt_out_cfg)
                    )
                )
            ),
        )
        sys.modules["plugins.GTBot.ConfigManager"] = config_manager_module

        cls.mod = _load_module_from_path(
            "plugins.GTBot.services.trigger.opt_out",
            ROOT / "plugins" / "GTBot" / "services" / "trigger" / "opt_out.py",
        )

    def test_suffix_rule_matches_normalized_text(self) -> None:
        """后缀规则应在轻量归一化后生效。"""

        manager = self.mod.ChatOptOutManager()
        context = self.mod.ChatOptOutContext(
            text="今天只是随口提一下猫娘   #别回 ",
            normalized_text=manager.normalize_text("今天只是随口提一下猫娘   #别回 "),
            group_id=123,
            user_id=456,
            to_me=False,
            mentioned_bot=False,
            trigger_keyword="猫娘",
        )

        self.assertEqual(manager.match_rule(context), "suffix_rule")

    def test_expression_rule_matches_safe_expression(self) -> None:
        """表达式规则应能使用白名单函数和上下文字段。"""

        manager = self.mod.ChatOptOutManager()
        context = self.mod.ChatOptOutContext(
            text="这个词只是讨论，不聊天",
            normalized_text=manager.normalize_text("这个词只是讨论，不聊天"),
            group_id=123,
            user_id=456,
            to_me=False,
            mentioned_bot=False,
            trigger_keyword="聊天",
        )

        self.assertEqual(manager.match_rule(context), "expr_rule")

    def test_unsafe_expression_is_rejected(self) -> None:
        """未授权表达式节点应被拒绝执行。"""

        evaluator = self.mod.SafeExpressionEvaluator()
        context = self.mod.ChatOptOutContext(
            text="测试",
            normalized_text="测试",
            group_id=1,
            user_id=2,
            to_me=False,
            mentioned_bot=False,
            trigger_keyword=None,
        )

        with self.assertRaises(ValueError):
            evaluator.evaluate("__import__('os').system('echo test')", context)


if __name__ == "__main__":
    unittest.main()
