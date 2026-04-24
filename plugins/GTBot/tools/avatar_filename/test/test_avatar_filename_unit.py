from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按文件路径加载模块并注册到 `sys.modules`。

    Args:
        module_qualname: 目标模块的完整限定名。
        file_path: 模块文件绝对路径。

    Returns:
        已加载完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


def _install_avatar_filename_import_stubs() -> None:
    """为头像文件名插件测试安装最小依赖桩。"""

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

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    setattr(plugins_mod, "GTBot", gtbot_mod)

    config_manager_mod = sys.modules.setdefault("plugins.GTBot.ConfigManager", ModuleType("plugins.GTBot.ConfigManager"))
    data_root = Path(tempfile.gettempdir()) / "avatar_filename_test_data"
    data_root.mkdir(parents=True, exist_ok=True)
    setattr(config_manager_mod, "total_config", SimpleNamespace(get_data_dir_path=lambda: data_root))

    context_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context")
    )

    class GroupChatContext:
        """测试用聊天上下文。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)


def _load_avatar_filename_package(plugin_dir: str) -> str:
    """加载头像文件名插件测试包而不触发宿主顶层导入链。

    Args:
        plugin_dir: 插件目录路径。

    Returns:
        动态构造出的测试包名。
    """

    _install_avatar_filename_import_stubs()
    package_name = f"_avatar_filename_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    return package_name


class TestAvatarFilenameTool(unittest.IsolatedAsyncioTestCase):
    """验证头像文件名工具的核心行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_avatar_filename_package(plugin_dir)
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])

    async def test_get_user_avatar_filename_should_save_file(self) -> None:
        """获取用户头像时应保存本地缓存文件并返回相对路径。"""

        fake_cache = SimpleNamespace(
            get_stranger_info=AsyncMock(return_value=SimpleNamespace(nickname="测试用户"))
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                user_id=123456,
                bot=object(),
                cache=fake_cache,
            )
        )
        with patch.object(
            self.tool_mod,
            "_download_avatar_bytes",
            AsyncMock(return_value=(b"avatar-bytes", "image/jpeg")),
        ), patch.object(self.tool_mod, "Path", self.tool_mod.Path):
            result = await self.tool_mod.get_user_avatar_filename(runtime)
        self.assertTrue(result.startswith("data\\avatar_filename\\") or result.startswith("data/avatar_filename/"))
        self.assertIn("user_avatar_123456_", result)

    async def test_get_group_avatar_filename_should_save_file(self) -> None:
        """获取群头像时应保存本地缓存文件并返回相对路径。"""

        fake_cache = SimpleNamespace(
            get_group_info=AsyncMock(return_value=SimpleNamespace(group_name="测试群"))
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                group_id=654321,
                bot=object(),
                cache=fake_cache,
            )
        )
        with patch.object(
            self.tool_mod,
            "_download_avatar_bytes",
            AsyncMock(return_value=(b"group-avatar", "image/png")),
        ), patch.object(self.tool_mod, "Path", self.tool_mod.Path):
            result = await self.tool_mod.get_group_avatar_filename(runtime)
        self.assertTrue(result.startswith("data\\avatar_filename\\") or result.startswith("data/avatar_filename/"))
        self.assertIn("group_avatar_654321_", result)

    async def test_get_group_avatar_filename_should_raise_when_group_missing(self) -> None:
        """群聊上下文缺少群号时应抛出异常。"""

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                group_id=0,
                bot=object(),
                cache=SimpleNamespace(get_group_info=AsyncMock()),
            )
        )
        with self.assertRaises(ValueError):
            await self.tool_mod.get_group_avatar_filename(runtime)


if __name__ == "__main__":
    unittest.main()
