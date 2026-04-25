from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


def _load_module_from_path(module_qualname: str, file_path: str) -> ModuleType:
    """按路径加载测试目标模块。

    Args:
        module_qualname: 需要注册到 `sys.modules` 的限定模块名。
        file_path: 目标模块文件绝对路径。

    Returns:
        ModuleType: 已执行完成的模块对象。
    """

    spec = importlib.util.spec_from_file_location(module_qualname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec: {module_qualname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_qualname] = module
    spec.loader.exec_module(module)
    return module


class TestChatLatencyMonitorUnit(unittest.TestCase):
    """验证聊天延迟监控器的样本累计与窗口统计行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        module = _load_module_from_path(
            "plugins.GTBot.services.chat.latency_monitor",
            str(root / "services" / "chat" / "latency_monitor.py"),
        )
        cls.monitor_cls = module.ChatLatencyMonitor

    def test_monitor_should_record_stage_and_total_for_single_request(self) -> None:
        """单请求记录多个阶段后应能读到阶段耗时与总耗时。"""

        monitor = self.monitor_cls(recent_window_size=5)
        monitor.start_request("resp_1", "group:1", "group_at", "group")
        monitor.record_stage_duration("resp_1", "load_turn_messages", 0.012)
        monitor.record_stage_duration("resp_1", "agent_invoke", 0.034)
        completed = monitor.finish_request("resp_1", "completed")

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed["outcome"], "completed")
        self.assertIn("load_turn_messages", completed["stages_ms"])
        self.assertIn("agent_invoke", completed["stages_ms"])
        self.assertIn("total", completed["stages_ms"])
        self.assertEqual(len(completed["stage_spans"]), 2)
        self.assertGreaterEqual(float(completed["total_ms"]), 0.0)

    def test_monitor_should_keep_recent_window_and_drop_old_samples(self) -> None:
        """最近窗口超过上限后应淘汰最旧样本，但累计统计仍保留。"""

        monitor = self.monitor_cls(recent_window_size=2)
        for index in range(3):
            response_id = f"resp_{index}"
            monitor.start_request(response_id, f"group:{index}", "group_at", "group")
            monitor.record_stage_duration(response_id, "agent_invoke", 0.001 * (index + 1))
            monitor.finish_request(response_id, "completed")

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["recent"]["sample_count"], 2)
        self.assertEqual(snapshot["lifetime"]["sample_count"], 3)
        self.assertEqual(snapshot["last_completed"]["response_id"], "resp_2")

    def test_monitor_should_include_failed_and_timeout_outcomes(self) -> None:
        """失败与超时请求也应进入结果分布统计。"""

        monitor = self.monitor_cls(recent_window_size=10)
        for outcome in ("failed", "timed_out"):
            monitor.start_request(f"resp_{outcome}", "group:1", "group_at", "group")
            monitor.record_stage_duration(f"resp_{outcome}", "agent_invoke", 0.01)
            monitor.finish_request(f"resp_{outcome}", outcome)

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["recent"]["outcome_counts"]["failed"], 1)
        self.assertEqual(snapshot["recent"]["outcome_counts"]["timed_out"], 1)


if __name__ == "__main__":
    unittest.main()
