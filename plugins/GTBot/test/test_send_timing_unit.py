from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nonebot.adapters.onebot.v11.message import Message, MessageSegment

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot.services.chat.pending_message import PendingQueuedMessageHandle
    from plugins.GTBot.services.chat.send_timing import (
        build_placeholder_queued_message_item,
        build_queued_message_items,
        calculate_message_delay_seconds,
        resolve_placeholder_timeout_seconds,
    )

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    PendingQueuedMessageHandle = None  # type: ignore[assignment]
    build_placeholder_queued_message_item = None  # type: ignore[assignment]
    build_queued_message_items = None  # type: ignore[assignment]
    calculate_message_delay_seconds = None  # type: ignore[assignment]
    resolve_placeholder_timeout_seconds = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestSendTimingUnit(unittest.TestCase):
    """验证 Agent 队列发送节奏计算的核心规则。"""

    def test_calculate_delay_for_plain_text_message(self) -> None:
        """纯文本消息应按基础时间加字数增量计算。"""

        assert calculate_message_delay_seconds is not None
        send_timing = SimpleNamespace(
            base_interval_seconds=0.5,
            per_char_seconds=0.1,
            jitter_seconds=0.0,
            max_interval_seconds=5.0,
            non_text_equivalent_chars={},
        )

        delay = calculate_message_delay_seconds(
            Message("你好呀"),
            send_timing=send_timing,
        )

        self.assertAlmostEqual(delay, 0.8)

    def test_calculate_delay_counts_non_text_segments_by_type(self) -> None:
        """非文本消息段应按配置的等效字符数参与计算。"""

        assert calculate_message_delay_seconds is not None
        send_timing = SimpleNamespace(
            base_interval_seconds=0.2,
            per_char_seconds=0.05,
            jitter_seconds=0.0,
            max_interval_seconds=5.0,
            non_text_equivalent_chars={"image": 20, "at": 3},
        )
        message = Message()
        message += MessageSegment.text("hi")
        message += MessageSegment.at(123456)
        message += MessageSegment.image("cat.png")

        delay = calculate_message_delay_seconds(
            message,
            send_timing=send_timing,
        )

        self.assertAlmostEqual(delay, 0.2 + (2 + 3 + 20) * 0.05)

    def test_calculate_delay_clamps_with_jitter_and_maximum(self) -> None:
        """随机抖动后的结果应同时遵守最小和最大边界。"""

        assert calculate_message_delay_seconds is not None
        send_timing = SimpleNamespace(
            base_interval_seconds=0.4,
            per_char_seconds=0.0,
            jitter_seconds=0.2,
            max_interval_seconds=0.5,
            non_text_equivalent_chars={},
        )

        with patch("plugins.GTBot.services.chat.send_timing.uniform", return_value=-0.2):
            low_delay = calculate_message_delay_seconds(
                Message("x"),
                send_timing=send_timing,
            )
        self.assertAlmostEqual(low_delay, 0.4)

        with patch("plugins.GTBot.services.chat.send_timing.uniform", return_value=0.2):
            high_delay = calculate_message_delay_seconds(
                Message("x"),
                send_timing=send_timing,
            )
        self.assertAlmostEqual(high_delay, 0.5)

    def test_build_queued_message_items_uses_override_when_provided(self) -> None:
        """显式传入固定延迟时，不应再读取自动节奏计算。"""

        assert build_queued_message_items is not None
        queued_items = build_queued_message_items(
            [Message("a"), Message("b")],
            interval_override=0.75,
        )

        self.assertEqual(len(queued_items), 2)
        self.assertAlmostEqual(queued_items[0].delay_seconds, 0.75)
        self.assertAlmostEqual(queued_items[1].delay_seconds, 0.75)
        self.assertFalse(queued_items[0].force_wait)
        self.assertGreater(queued_items[0].enqueued_at, 0.0)

    def test_build_queued_message_items_can_enable_force_wait(self) -> None:
        """显式开启强制等待时，应把标记写入所有队列项。"""

        assert build_queued_message_items is not None
        queued_items = build_queued_message_items(
            [Message("a")],
            interval_override=0.5,
            force_wait=True,
        )

        self.assertEqual(len(queued_items), 1)
        self.assertAlmostEqual(queued_items[0].delay_seconds, 0.5)
        self.assertTrue(queued_items[0].force_wait)

    def test_resolve_placeholder_timeout_seconds_uses_default_when_missing(self) -> None:
        """占位超时未显式传入时，应回退到当前配置中的默认值。"""

        assert resolve_placeholder_timeout_seconds is not None
        with patch(
            "plugins.GTBot.services.chat.send_timing.get_current_send_timing_config",
            return_value=SimpleNamespace(placeholder_timeout_seconds=88.0),
        ):
            resolved = resolve_placeholder_timeout_seconds(None)

        self.assertEqual(resolved, 88.0)

    def test_build_placeholder_queued_message_item_uses_override_and_force_wait(self) -> None:
        """占位队列项应写入句柄、超时覆盖值与显式等待参数。"""

        assert PendingQueuedMessageHandle is not None
        assert build_placeholder_queued_message_item is not None

        item = build_placeholder_queued_message_item(
            PendingQueuedMessageHandle(scope="群组 1"),
            timeout_override=12.5,
            interval_override=0.4,
            force_wait=True,
        )

        self.assertTrue(item.is_placeholder())
        self.assertEqual(item.placeholder_timeout_sec, 12.5)
        self.assertEqual(item.delay_seconds, 0.4)
        self.assertTrue(item.force_wait)


if __name__ == "__main__":
    unittest.main()
