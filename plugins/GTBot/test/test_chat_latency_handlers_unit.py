from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, patch


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按路径加载测试目标模块。"""

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


class _FakeMatcher:
    """提供最小行为的命令 matcher 测试桩对象。"""

    def __init__(self) -> None:
        self.finish = AsyncMock()
        self._handler = None

    def handle(self):
        def decorator(func):
            self._handler = func
            return func

        return decorator


class _FakeMessageSegment:
    """模拟 OneBot 消息段工厂。"""

    @staticmethod
    def text(content: str) -> str:
        return content

    @staticmethod
    def image(*, file: str) -> str:
        return f"[CQ:image,file={file}]"


class _FakeMessage(list[str]):
    """模拟可追加的 OneBot 消息对象。"""

    def __init__(self, initial: str | None = None) -> None:
        super().__init__()
        if initial is not None:
            self.append(initial)

    def __str__(self) -> str:
        return "".join(str(item) for item in self)


def _install_import_stubs() -> None:
    """为命令模块测试安装最小导入桩。"""

    nonebot_mod = sys.modules.setdefault("nonebot", ModuleType("nonebot"))
    setattr(nonebot_mod, "on_command", lambda *a, **k: _FakeMatcher())

    adapters_mod = sys.modules.setdefault("nonebot.adapters", ModuleType("nonebot.adapters"))
    onebot_mod = sys.modules.setdefault("nonebot.adapters.onebot", ModuleType("nonebot.adapters.onebot"))
    v11_mod = sys.modules.setdefault("nonebot.adapters.onebot.v11", ModuleType("nonebot.adapters.onebot.v11"))
    event_mod = sys.modules.setdefault("nonebot.adapters.onebot.v11.event", ModuleType("nonebot.adapters.onebot.v11.event"))
    setattr(adapters_mod, "onebot", onebot_mod)
    setattr(onebot_mod, "v11", v11_mod)
    setattr(v11_mod, "Message", _FakeMessage)
    setattr(v11_mod, "MessageSegment", _FakeMessageSegment)

    class MessageEvent:
        pass

    setattr(event_mod, "MessageEvent", MessageEvent)

    help_mod = sys.modules.setdefault("local_plugins.nonebot_plugin_gt_help", ModuleType("local_plugins.nonebot_plugin_gt_help"))

    class HelpCommandSpec:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    setattr(help_mod, "HelpCommandSpec", HelpCommandSpec)
    setattr(help_mod, "register_help", lambda spec: spec)

    permission_mod = sys.modules.setdefault(
        "local_plugins.nonebot_plugin_gt_permission",
        ModuleType("local_plugins.nonebot_plugin_gt_permission"),
    )

    class PermissionRole:
        ADMIN = "admin"

    class PermissionError(Exception):
        pass

    permission_manager = SimpleNamespace(require_role=AsyncMock())
    setattr(permission_mod, "PermissionRole", PermissionRole)
    setattr(permission_mod, "PermissionError", PermissionError)
    setattr(permission_mod, "get_permission_manager", lambda: permission_manager)

    plugins_mod = sys.modules.setdefault("plugins", ModuleType("plugins"))
    gtbot_mod = sys.modules.setdefault("plugins.GTBot", ModuleType("plugins.GTBot"))
    services_mod = sys.modules.setdefault("plugins.GTBot.services", ModuleType("plugins.GTBot.services"))
    chat_mod = sys.modules.setdefault("plugins.GTBot.services.chat", ModuleType("plugins.GTBot.services.chat"))
    setattr(plugins_mod, "GTBot", gtbot_mod)
    setattr(gtbot_mod, "services", services_mod)
    setattr(services_mod, "chat", chat_mod)

    latency_chart_mod = sys.modules.setdefault(
        "plugins.GTBot.services.chat.latency_chart",
        ModuleType("plugins.GTBot.services.chat.latency_chart"),
    )
    setattr(chat_mod, "latency_chart", latency_chart_mod)

    class GanttRenderResult:
        def __init__(self, status: str, image_path: Path | None = None, detail: str = "") -> None:
            self.status = status
            self.image_path = image_path
            self.detail = detail

    setattr(latency_chart_mod, "GanttRenderResult", GanttRenderResult)
    setattr(
        latency_chart_mod,
        "prepare_last_completed_latency_gantt",
        lambda *a, **k: GanttRenderResult("no_last_completed"),
    )


class TestChatLatencyHandlersUnit(unittest.IsolatedAsyncioTestCase):
    handlers_mod: ClassVar[ModuleType]

    """验证聊天延迟命令和独立甘特图命令的渲染与权限处理。"""

    @classmethod
    def setUpClass(cls) -> None:
        _install_import_stubs()
        root = Path(__file__).resolve().parents[1]
        latency_monitor_path = root / "services" / "chat" / "latency_monitor.py"
        handlers_path = root / "ChatLatencyHandlers.py"
        _load_module_from_path("plugins.GTBot.services.chat.latency_monitor", str(latency_monitor_path))
        cls.handlers_mod = _load_module_from_path("plugins.GTBot.ChatLatencyHandlers", str(handlers_path))

    def test_render_latency_summary_should_include_inflight_and_averages(self) -> None:
        snapshot = {
            "inflight_count": 1,
            "inflight": [
                {
                    "response_id": "resp_1",
                    "session_id": "group:1",
                    "trigger_mode": "group_at",
                    "chat_type": "group",
                    "current_stage": "agent_invoke",
                    "elapsed_ms": 123.45,
                }
            ],
            "recent": {
                "sample_count": 2,
                "average_total_ms": 456.78,
                "average_stages_ms": {"agent_invoke": 300.0, "load_turn_messages": 20.0},
                "outcome_counts": {"completed": 1, "failed": 1},
            },
            "last_completed": {
                "response_id": "resp_last",
                "outcome": "completed",
                "trigger_mode": "group_at",
                "chat_type": "group",
                "total_ms": 500.0,
                "stages_ms": {"agent_invoke": 400.0},
                "stage_spans": [{"name": "agent_invoke", "start_ms": 60.0, "end_ms": 460.0, "duration_ms": 400.0}],
            },
        }

        rendered = self.handlers_mod._render_latency_summary(snapshot, gantt_status="success")
        self.assertIn("当前运行中请求: 1", rendered)
        self.assertIn("触发=群聊@触发", rendered)
        self.assertIn("当前阶段=模型/智能体执行", rendered)
        self.assertIn("最近100次平均总耗时: 456.78ms", rendered)
        self.assertIn("结果=成功", rendered)
        self.assertIn("甘特图已作为图片附带发送", rendered)

    def test_build_gantt_reply_should_return_image_message_when_chart_exists(self) -> None:
        result = self.handlers_mod.GanttRenderResult(status="success", image_path=Path("C:/tmp/chat_latency.png"))
        rendered = self.handlers_mod._build_gantt_reply(result)
        self.assertIsNotNone(rendered)
        self.assertEqual(str(rendered), "[CQ:image,file=C:/tmp/chat_latency.png]")

    def test_render_gantt_error_message_should_distinguish_render_failure(self) -> None:
        result = self.handlers_mod.GanttRenderResult(status="write_failed", detail="kaleido boom")
        self.assertIn("导出失败", self.handlers_mod._render_gantt_error_message(result))
        self.assertIn("kaleido boom", self.handlers_mod._render_gantt_error_message(result))

    async def test_handle_chat_latency_gantt_command_should_return_tip_when_no_chart(self) -> None:
        event = SimpleNamespace(user_id=123)
        finish_mock = AsyncMock()
        with (
            patch.object(self.handlers_mod, "_ensure_admin", AsyncMock()),
            patch.object(
                self.handlers_mod,
                "_prepare_gantt",
                return_value=self.handlers_mod.GanttRenderResult("no_last_completed"),
            ),
            patch.object(self.handlers_mod.ChatLatencyGanttCommand, "finish", finish_mock),
        ):
            await self.handlers_mod.handle_chat_latency_gantt_command(event)

        await_args = finish_mock.await_args
        assert await_args is not None
        self.assertIn("暂无可用的聊天甘特图", await_args.args[0])


if __name__ == "__main__":
    unittest.main()
