from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Awaitable, Callable, cast
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
        """模拟 NoneBot `handle()` 装饰器注册回调。"""

        def decorator(func):
            self._handler = func
            return func

        return decorator


def _install_friend_management_import_stubs() -> None:
    """为好友管理插件测试安装最小依赖桩。

    当前测试既要加载工具模块，也要加载命令模块，因此需要同时提供 LangChain、
    NoneBot、帮助系统和共享工具层的最小替身，确保导入过程不依赖宿主完整初始化。
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

    nonebot_mod = sys.modules.setdefault("nonebot", ModuleType("nonebot"))
    setattr(
        nonebot_mod,
        "logger",
        SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None),
    )
    setattr(nonebot_mod, "get_driver", lambda: (_ for _ in ()).throw(RuntimeError("not initialized")))
    setattr(nonebot_mod, "on_command", lambda *a, **k: _FakeMatcher())

    adapters_mod = sys.modules.setdefault("nonebot.adapters", ModuleType("nonebot.adapters"))
    onebot_mod = sys.modules.setdefault("nonebot.adapters.onebot", ModuleType("nonebot.adapters.onebot"))
    v11_mod = sys.modules.setdefault("nonebot.adapters.onebot.v11", ModuleType("nonebot.adapters.onebot.v11"))
    v11_event_mod = sys.modules.setdefault(
        "nonebot.adapters.onebot.v11.event", ModuleType("nonebot.adapters.onebot.v11.event")
    )
    setattr(adapters_mod, "onebot", onebot_mod)
    setattr(onebot_mod, "v11", v11_mod)

    class Bot:
        """测试用 Bot 桩。"""

    class MessageEvent:
        """测试用消息事件桩。"""

    setattr(v11_mod, "Bot", Bot)
    setattr(v11_event_mod, "MessageEvent", MessageEvent)

    permission_mod = sys.modules.setdefault(
        "local_plugins.nonebot_plugin_gt_permission",
        ModuleType("local_plugins.nonebot_plugin_gt_permission"),
    )
    setattr(permission_mod, "PermissionRole", SimpleNamespace(USER="user", ADMIN="admin"))
    setattr(permission_mod, "has_role", AsyncMock(return_value=False))

    help_mod = sys.modules.setdefault("plugins.GTBot.services.help", ModuleType("plugins.GTBot.services.help"))

    class HelpCommandSpec:
        """测试用帮助命令结构。"""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    setattr(help_mod, "HelpCommandSpec", HelpCommandSpec)
    setattr(help_mod, "register_help", lambda spec: spec)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    shared_mod = sys.modules.setdefault("plugins.GTBot.services.shared", ModuleType("plugins.GTBot.services.shared"))
    fun_mod = sys.modules.setdefault("plugins.GTBot.services.shared.fun", ModuleType("plugins.GTBot.services.shared.fun"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    context_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.context", ModuleType("plugins.GTBot.services.chat.context")
    )

    setattr(plugins_mod, "GTBot", gtbot_mod)
    setattr(gtbot_mod, "services", services_mod)
    setattr(services_mod, "shared", shared_mod)
    setattr(shared_mod, "fun", fun_mod)
    setattr(services_mod, "chat", chat_mod)
    setattr(chat_mod, "context", context_mod)

    class GroupChatContext:
        """测试用聊天上下文。"""

    setattr(context_mod, "GroupChatContext", GroupChatContext)
    setattr(fun_mod, "send_like", AsyncMock(return_value={"status": "ok"}))


def _get_async_tool_callable(tool_obj: object) -> Callable[..., Awaitable[Any]]:
    """返回测试可直接 `await` 的异步工具实现。

    Args:
        tool_obj: `@tool` 装饰后的工具对象或原始函数。

    Returns:
        可直接在测试中等待执行结果的协程函数。
    """

    return cast(Callable[..., Awaitable[Any]], getattr(tool_obj, "coroutine", tool_obj))


def _load_friend_management_package(plugin_dir: str) -> str:
    """加载好友管理插件测试包而不触发宿主顶层导入链。

    Args:
        plugin_dir: 插件目录绝对路径。

    Returns:
        当前测试专用的包名。
    """

    _install_friend_management_import_stubs()
    package_name = f"_friend_management_unittestpkg_{uuid4().hex}"
    pkg = ModuleType(package_name)
    pkg.__path__ = [plugin_dir]  # type: ignore[attr-defined]
    pkg.__file__ = str(Path(plugin_dir) / "__init__.py")
    pkg.__package__ = package_name
    sys.modules[package_name] = pkg

    _load_module_from_path(f"{package_name}.config", str(Path(plugin_dir) / "config.py"))
    _load_module_from_path(f"{package_name}.like_gate", str(Path(plugin_dir) / "like_gate.py"))
    _load_module_from_path(f"{package_name}.usage_limits", str(Path(plugin_dir) / "usage_limits.py"))
    _load_module_from_path(f"{package_name}.tool", str(Path(plugin_dir) / "tool.py"))
    _load_module_from_path(f"{package_name}.commands", str(Path(plugin_dir) / "commands.py"))
    _load_module_from_path(f"{package_name}.__init__", str(Path(plugin_dir) / "__init__.py"))
    return package_name


class TestFriendManagementUnit(unittest.IsolatedAsyncioTestCase):
    """验证好友管理插件的点赞工具与命令行为。"""

    pkg: str
    config_mod: ModuleType
    like_gate_mod: ModuleType
    usage_mod: ModuleType
    tool_mod: ModuleType
    commands_mod: ModuleType
    init_mod: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        """加载一次被测模块，供后续测试复用。"""

        plugin_dir = str(Path(__file__).resolve().parents[1])
        cls.pkg = _load_friend_management_package(plugin_dir)
        cls.config_mod = __import__(f"{cls.pkg}.config", fromlist=["dummy"])
        cls.like_gate_mod = __import__(f"{cls.pkg}.like_gate", fromlist=["dummy"])
        cls.usage_mod = __import__(f"{cls.pkg}.usage_limits", fromlist=["dummy"])
        cls.tool_mod = __import__(f"{cls.pkg}.tool", fromlist=["dummy"])
        cls.commands_mod = __import__(f"{cls.pkg}.commands", fromlist=["dummy"])
        cls.init_mod = __import__(f"{cls.pkg}.__init__", fromlist=["dummy"])

    async def test_send_like_tool_should_call_fun_send_like_with_times_ten(self) -> None:
        """点赞工具应固定使用 `times=10` 调用底层接口。"""

        runtime = SimpleNamespace(context=SimpleNamespace(bot=object(), user_id=1001, group_id=2002))
        fake_cfg = SimpleNamespace(timeout_sec=3.0, max_likes_per_user_per_day=10)

        with patch.object(self.tool_mod, "get_friend_management_plugin_config", return_value=fake_cfg), patch.object(
            self.tool_mod,
            "calculate_like_send_times",
            return_value=10,
        ), patch.object(
            self.tool_mod,
            "get_friend_management_like_limit_manager",
            return_value=SimpleNamespace(record_like=lambda **kwargs: None),
        ), patch.object(self.tool_mod.Fun, "send_like", new=AsyncMock(return_value={"status": "ok"})) as send_like_mock:
            result = await _get_async_tool_callable(self.tool_mod.send_like_tool)(123456, runtime=runtime)

        self.assertEqual(result, "sent like to user 123456 successfully")
        send_like_mock.assert_awaited_once_with(runtime.context.bot, 123456, times=10)

    async def test_handle_like_command_should_like_event_sender(self) -> None:
        """手动点赞命令应直接给命令发起人点满赞。"""

        bot = object()
        event = SimpleNamespace(user_id=654321)
        matcher = self.commands_mod.LikeCommand
        matcher.finish.reset_mock()

        fake_cfg = SimpleNamespace(max_likes_per_user_per_day=10)
        with patch.object(self.commands_mod, "get_friend_management_plugin_config", return_value=fake_cfg), patch.object(
            self.commands_mod,
            "calculate_like_send_times",
            return_value=10,
        ), patch.object(
            self.commands_mod,
            "get_friend_management_like_limit_manager",
            return_value=SimpleNamespace(record_like=lambda **kwargs: None),
        ), patch.object(self.commands_mod.Fun, "send_like", new=AsyncMock(return_value={"status": "ok"})) as send_like_mock:
            await self.commands_mod.handle_like_command(bot=bot, event=event)

        send_like_mock.assert_awaited_once_with(bot, 654321, times=10)
        matcher.finish.assert_awaited_once_with("已给你点满赞 (654321)")

    async def test_register_should_gate_like_tool_with_config(self) -> None:
        """插件注册时应为点赞工具附带配置开关判定。"""

        captured: list[tuple[object, Any]] = []

        class FakeRegistry:
            """最小化插件注册器桩。"""

            def add_tool(self, tool_obj: object, enabled: Any = None) -> None:
                """记录注册到插件系统的工具与条件。

                Args:
                    tool_obj: 被注册的工具对象。
                    enabled: 可选启用判定函数。
                """

                captured.append((tool_obj, enabled))

        fake_cfg = SimpleNamespace(expose_like_tool_to_agent=True)
        with patch.object(self.init_mod, "get_friend_management_plugin_config", return_value=fake_cfg):
            self.init_mod.register(FakeRegistry())

        self.assertEqual(len(captured), 2)
        self.assertIs(captured[0][0], self.init_mod.delete_friend_tool)
        self.assertIsNone(captured[0][1])
        self.assertIs(captured[1][0], self.init_mod.send_like_tool)
        enabled = captured[1][1]
        self.assertTrue(callable(enabled))
        with patch.object(self.init_mod, "get_friend_management_plugin_config", return_value=fake_cfg):
            self.assertTrue(enabled(SimpleNamespace()))

    def test_like_limit_manager_should_reset_by_shanghai_natural_day(self) -> None:
        """点赞限额应按北京时间自然日重置。"""

        state_path = Path(tempfile.gettempdir()) / f"friend_management_like_usage_{uuid4().hex}.json"
        manager = self.usage_mod.FriendManagementLikeLimitManager(state_path=state_path)
        cfg = SimpleNamespace(max_likes_per_user_per_day=10)

        tz = self.usage_mod._SHANGHAI_TZ
        same_day_morning = datetime(2026, 1, 1, 9, 0, tzinfo=tz).timestamp()
        same_day_evening = datetime(2026, 1, 1, 21, 0, tzinfo=tz).timestamp()
        next_day_morning = datetime(2026, 1, 2, 9, 0, tzinfo=tz).timestamp()

        try:
            self.assertEqual(manager.get_remaining_likes(cfg=cfg, user_id=1001, now_ts=same_day_morning), 10)
            manager.record_like(cfg=cfg, user_id=1001, count=7, now_ts=same_day_morning)
            self.assertEqual(manager.get_remaining_likes(cfg=cfg, user_id=1001, now_ts=same_day_evening), 3)
            self.assertEqual(manager.get_remaining_likes(cfg=cfg, user_id=1001, now_ts=next_day_morning), 10)
        finally:
            state_path.unlink(missing_ok=True)

    async def test_handle_like_command_should_send_partial_when_quota_remaining_is_lower_than_ten(self) -> None:
        """当当天剩余额度不足 10 时，命令应只发送剩余额度允许的数量。"""

        bot = object()
        event = SimpleNamespace(user_id=654321)
        matcher = self.commands_mod.LikeCommand
        matcher.finish.reset_mock()

        fake_cfg = SimpleNamespace(max_likes_per_user_per_day=6)
        with patch.object(self.commands_mod, "get_friend_management_plugin_config", return_value=fake_cfg), patch.object(
            self.commands_mod,
            "calculate_like_send_times",
            return_value=3,
        ), patch.object(
            self.commands_mod,
            "get_friend_management_like_limit_manager",
            return_value=SimpleNamespace(record_like=lambda **kwargs: None),
        ), patch.object(self.commands_mod.Fun, "send_like", new=AsyncMock(return_value={"status": "ok"})) as send_like_mock:
            await self.commands_mod.handle_like_command(bot=bot, event=event)

        send_like_mock.assert_awaited_once_with(bot, 654321, times=3)
        matcher.finish.assert_awaited_once_with("已给你点赞 3 次，今天额度已用尽 (654321)")

    async def test_handle_zanwo_command_should_reject_when_today_like_count_is_below_threshold(self) -> None:
        """`赞我` 在未满足当天点赞门槛时应直接拒绝。"""

        bot = object()
        event = SimpleNamespace(user_id=24680)
        matcher = self.commands_mod.ZanWoCommand
        matcher.finish.reset_mock()

        fake_cfg = SimpleNamespace(
            require_likes_before_zanwo=5,
            max_likes_per_user_per_day=10,
        )
        with patch.object(self.commands_mod, "get_friend_management_plugin_config", return_value=fake_cfg), patch.object(
            self.commands_mod,
            "should_bypass_zanwo_gate",
            new=AsyncMock(return_value=False),
        ), patch.object(
            self.commands_mod,
            "get_today_like_count_for_bot_from_user",
            new=AsyncMock(return_value=2),
        ):
            await self.commands_mod.handle_zanwo_command(bot=bot, event=event)

        matcher.finish.assert_awaited_once_with("你今天给机器人点的赞还不够，需要至少 5 个，当前只有 2 个。")

    async def test_handle_zanwo_command_should_bypass_gate_for_exempt_user(self) -> None:
        """`赞我` 在命中豁免时应跳过门槛查询并继续点赞。"""

        bot = object()
        event = SimpleNamespace(user_id=13579)
        matcher = self.commands_mod.ZanWoCommand
        matcher.finish.reset_mock()

        fake_cfg = SimpleNamespace(
            require_likes_before_zanwo=5,
            max_likes_per_user_per_day=10,
        )
        with patch.object(self.commands_mod, "get_friend_management_plugin_config", return_value=fake_cfg), patch.object(
            self.commands_mod,
            "should_bypass_zanwo_gate",
            new=AsyncMock(return_value=True),
        ), patch.object(
            self.commands_mod,
            "get_today_like_count_for_bot_from_user",
            new=AsyncMock(return_value=0),
        ) as like_count_mock, patch.object(
            self.commands_mod,
            "calculate_like_send_times",
            return_value=10,
        ), patch.object(
            self.commands_mod,
            "get_friend_management_like_limit_manager",
            return_value=SimpleNamespace(record_like=lambda **kwargs: None),
        ), patch.object(self.commands_mod.Fun, "send_like", new=AsyncMock(return_value={"status": "ok"})) as send_like_mock:
            await self.commands_mod.handle_zanwo_command(bot=bot, event=event)

        like_count_mock.assert_not_awaited()
        send_like_mock.assert_awaited_once_with(bot, 13579, times=10)
        matcher.finish.assert_awaited_once_with("已给你点满赞 (13579)")

    async def test_should_bypass_zanwo_gate_should_allow_admin_when_enabled(self) -> None:
        """当配置开启管理员豁免时，管理员应直接跳过 `赞我` 门槛。"""

        fake_cfg = SimpleNamespace(
            is_zanwo_gate_exempt=lambda user_id: False,
            exempt_admin_for_zanwo_gate=True,
        )
        with patch.object(self.like_gate_mod, "has_role", new=AsyncMock(return_value=True)) as has_role_mock:
            result = await self.like_gate_mod.should_bypass_zanwo_gate(cfg=fake_cfg, user_id=10086)

        self.assertTrue(result)
        has_role_mock.assert_awaited_once_with(10086, self.like_gate_mod.PermissionRole.ADMIN)


if __name__ == "__main__":
    unittest.main()
