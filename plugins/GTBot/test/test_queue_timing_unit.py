from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot.model import QueuedMessageItem
    from plugins.GTBot.services.chat.group_queue import GroupMessageQueueManager

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc

    if TYPE_CHECKING:
        from plugins.GTBot.model import QueuedMessageItem
        from plugins.GTBot.services.chat.group_queue import GroupMessageQueueManager
    else:
        QueuedMessageItem = cast(Any, None)
        GroupMessageQueueManager = cast(Any, None)


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestQueueTimingUnit(unittest.IsolatedAsyncioTestCase):
    """验证队列根据上一条发送时间自动结算剩余等待时间。"""

    async def test_non_force_wait_only_waits_remaining_gap(self) -> None:
        """非强制等待模式下，应只补足距离上一条发送还差的时间。"""

        assert QueuedMessageItem is not None
        assert GroupMessageQueueManager is not None

        manager = GroupMessageQueueManager()
        item = QueuedMessageItem(
            message="hello",
            delay_seconds=1.0,
            force_wait=False,
            enqueued_at=100.0,
        )
        with patch("plugins.GTBot.services.chat.group_queue.time", return_value=100.4), patch(
            "plugins.GTBot.services.chat.group_queue.sleep",
            AsyncMock(),
        ) as sleep_mock:
            await manager._wait_until_sendable(item, last_sent_at=100.0)

        sleep_mock.assert_awaited_once()
        await_args = sleep_mock.await_args
        assert await_args is not None
        waited = await_args.args[0]
        self.assertAlmostEqual(waited, 0.6)

    async def test_force_wait_uses_enqueue_time_only(self) -> None:
        """强制等待模式下，应从入队时刻起等待固定时长。"""

        assert QueuedMessageItem is not None
        assert GroupMessageQueueManager is not None

        manager = GroupMessageQueueManager()
        item = QueuedMessageItem(
            message="hello",
            delay_seconds=1.0,
            force_wait=True,
            enqueued_at=100.0,
        )
        with patch("plugins.GTBot.services.chat.group_queue.time", return_value=100.2), patch(
            "plugins.GTBot.services.chat.group_queue.sleep",
            AsyncMock(),
        ) as sleep_mock:
            await manager._wait_until_sendable(item, last_sent_at=150.0)

        sleep_mock.assert_awaited_once()
        await_args = sleep_mock.await_args
        assert await_args is not None
        waited = await_args.args[0]
        self.assertAlmostEqual(waited, 0.8)


if __name__ == "__main__":
    unittest.main()
