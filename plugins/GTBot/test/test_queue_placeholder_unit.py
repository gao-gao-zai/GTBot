from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

from nonebot.adapters.onebot.v11.message import Message

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot.model import MessageTask, QueuedMessageItem
    from plugins.GTBot.services.chat.group_queue import GroupMessageQueueManager, _QueuedMessageTask
    from plugins.GTBot.services.chat.pending_message import PendingQueuedMessageHandle
    from plugins.GTBot.services.chat.private_queue import (
        PrivateMessageQueueManager,
        PrivateMessageTask,
        _QueuedPrivateMessageTask,
    )

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc

    if TYPE_CHECKING:
        from plugins.GTBot.model import MessageTask, QueuedMessageItem
        from plugins.GTBot.services.chat.group_queue import GroupMessageQueueManager, _QueuedMessageTask
        from plugins.GTBot.services.chat.pending_message import PendingQueuedMessageHandle
        from plugins.GTBot.services.chat.private_queue import (
            PrivateMessageQueueManager,
            PrivateMessageTask,
            _QueuedPrivateMessageTask,
        )
    else:
        MessageTask = cast(Any, None)
        QueuedMessageItem = cast(Any, None)
        GroupMessageQueueManager = cast(Any, None)
        _QueuedMessageTask = cast(Any, None)
        PendingQueuedMessageHandle = cast(Any, None)
        PrivateMessageQueueManager = cast(Any, None)
        PrivateMessageTask = cast(Any, None)
        _QueuedPrivateMessageTask = cast(Any, None)


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestQueuePlaceholderUnit(unittest.IsolatedAsyncioTestCase):
    """验证群聊与私聊消息队列对占位条目的处理规则。"""

    def _build_group_runtime(self) -> tuple[Any, Any, Any]:
        """构造群聊队列处理所需的最小运行时依赖桩。"""

        bot = SimpleNamespace(self_id="999", send_group_msg=AsyncMock(return_value={"message_id": 1001}))
        message_manager = SimpleNamespace(add_message=AsyncMock())
        cache = SimpleNamespace(get_user_name=AsyncMock(return_value="bot"))
        return bot, message_manager, cache

    def _build_private_runtime(self) -> tuple[Any, Any, Any]:
        """构造私聊队列处理所需的最小运行时依赖桩。"""

        bot = SimpleNamespace(self_id="999", send_private_msg=AsyncMock(return_value={"message_id": 2001}))
        message_manager = SimpleNamespace(add_chat_message=AsyncMock())
        cache = SimpleNamespace(get_user_name=AsyncMock(return_value="bot"))
        return bot, message_manager, cache

    async def test_group_placeholder_should_send_fulfilled_content_in_order(self) -> None:
        """群聊占位在及时 fulfill 后，应按原顺序发送其内容和后续消息。"""

        assert GroupMessageQueueManager is not None
        assert MessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedMessageTask is not None

        manager = GroupMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="群组 123")
        self.assertTrue(handle.fulfill(Message("placeholder")))
        task = MessageTask(
            group_id=123,
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=10.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_group_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()), patch(
            "plugins.GTBot.services.chat.group_queue.time",
            side_effect=[11.0, 12.0],
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=5.0,
            )

        self.assertEqual(updated_last_sent_at, 12.0)
        self.assertEqual(bot.send_group_msg.await_count, 2)
        self.assertEqual(str(bot.send_group_msg.await_args_list[0].kwargs["message"]), "placeholder")
        self.assertEqual(str(bot.send_group_msg.await_args_list[1].kwargs["message"]), "after")
        message_manager.add_message.assert_awaited()

    async def test_group_placeholder_timeout_should_skip_without_refreshing_last_sent_at(self) -> None:
        """群聊占位超时后，应静默跳过并让后续条目继续沿用旧的发送基准时间。"""

        assert GroupMessageQueueManager is not None
        assert MessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedMessageTask is not None

        manager = GroupMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="群组 123")
        task = MessageTask(
            group_id=123,
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=0.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_group_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()) as wait_mock, patch(
            "plugins.GTBot.services.chat.group_queue.time",
            return_value=12.0,
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=5.0,
            )

        self.assertEqual(handle.status, "timed_out")
        self.assertEqual(updated_last_sent_at, 12.0)
        self.assertEqual(wait_mock.await_args_list[0].kwargs["last_sent_at"], 5.0)
        self.assertEqual(wait_mock.await_args_list[1].kwargs["last_sent_at"], 5.0)
        bot.send_group_msg.assert_awaited_once()
        self.assertEqual(str(bot.send_group_msg.await_args.kwargs["message"]), "after")
        message_manager.add_message.assert_awaited_once()

    async def test_group_placeholder_cancel_should_skip_without_refreshing_last_sent_at(self) -> None:
        """群聊占位取消后，应静默跳过并保留旧的发送基准时间。"""

        assert GroupMessageQueueManager is not None
        assert MessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedMessageTask is not None

        manager = GroupMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="群组 123")
        self.assertTrue(handle.cancel())
        task = MessageTask(
            group_id=123,
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=10.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_group_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()) as wait_mock, patch(
            "plugins.GTBot.services.chat.group_queue.time",
            return_value=12.0,
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=5.0,
            )

        self.assertEqual(updated_last_sent_at, 12.0)
        self.assertEqual(wait_mock.await_args_list[0].kwargs["last_sent_at"], 5.0)
        self.assertEqual(wait_mock.await_args_list[1].kwargs["last_sent_at"], 5.0)
        bot.send_group_msg.assert_awaited_once()
        self.assertEqual(str(bot.send_group_msg.await_args.kwargs["message"]), "after")

    async def test_private_placeholder_should_send_fulfilled_content_in_order(self) -> None:
        """私聊占位在及时 fulfill 后，应按原顺序发送其内容和后续消息。"""

        assert PrivateMessageQueueManager is not None
        assert PrivateMessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedPrivateMessageTask is not None

        manager = PrivateMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="session private:123")
        self.assertTrue(handle.fulfill(Message("placeholder")))
        task = PrivateMessageTask(
            user_id=123,
            session_id="private:123",
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=10.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_private_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()), patch(
            "plugins.GTBot.services.chat.private_queue.time",
            side_effect=[21.0, 22.0],
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedPrivateMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=8.0,
            )

        self.assertEqual(updated_last_sent_at, 22.0)
        self.assertEqual(bot.send_private_msg.await_count, 2)
        self.assertEqual(str(bot.send_private_msg.await_args_list[0].kwargs["message"]), "placeholder")
        self.assertEqual(str(bot.send_private_msg.await_args_list[1].kwargs["message"]), "after")
        message_manager.add_chat_message.assert_awaited()

    async def test_private_placeholder_timeout_should_skip_without_refreshing_last_sent_at(self) -> None:
        """私聊占位超时后，应静默跳过并保留旧的发送基准时间。"""

        assert PrivateMessageQueueManager is not None
        assert PrivateMessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedPrivateMessageTask is not None

        manager = PrivateMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="session private:123")
        task = PrivateMessageTask(
            user_id=123,
            session_id="private:123",
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=0.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_private_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()) as wait_mock, patch(
            "plugins.GTBot.services.chat.private_queue.time",
            return_value=22.0,
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedPrivateMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=8.0,
            )

        self.assertEqual(handle.status, "timed_out")
        self.assertEqual(updated_last_sent_at, 22.0)
        self.assertEqual(wait_mock.await_args_list[0].kwargs["last_sent_at"], 8.0)
        self.assertEqual(wait_mock.await_args_list[1].kwargs["last_sent_at"], 8.0)
        bot.send_private_msg.assert_awaited_once()
        self.assertEqual(str(bot.send_private_msg.await_args.kwargs["message"]), "after")

    async def test_private_placeholder_cancel_should_skip_without_refreshing_last_sent_at(self) -> None:
        """私聊占位取消后，应静默跳过并保留旧的发送基准时间。"""

        assert PrivateMessageQueueManager is not None
        assert PrivateMessageTask is not None
        assert QueuedMessageItem is not None
        assert PendingQueuedMessageHandle is not None
        assert _QueuedPrivateMessageTask is not None

        manager = PrivateMessageQueueManager()
        handle = PendingQueuedMessageHandle(scope="session private:123")
        self.assertTrue(handle.cancel())
        task = PrivateMessageTask(
            user_id=123,
            session_id="private:123",
            messages=[
                QueuedMessageItem(
                    placeholder_handle=handle,
                    placeholder_timeout_sec=10.0,
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
                QueuedMessageItem(
                    message=Message("after"),
                    delay_seconds=0.0,
                    force_wait=False,
                    enqueued_at=1.0,
                ),
            ],
        )
        bot, message_manager, cache = self._build_private_runtime()

        with patch.object(manager, "_wait_until_sendable", AsyncMock()) as wait_mock, patch(
            "plugins.GTBot.services.chat.private_queue.time",
            return_value=22.0,
        ):
            updated_last_sent_at = await manager._process_task(
                _QueuedPrivateMessageTask(
                    task=task,
                    bot=bot,
                    message_manager=message_manager,
                    cache=cache,
                ),
                last_sent_at=8.0,
            )

        self.assertEqual(updated_last_sent_at, 22.0)
        self.assertEqual(wait_mock.await_args_list[0].kwargs["last_sent_at"], 8.0)
        self.assertEqual(wait_mock.await_args_list[1].kwargs["last_sent_at"], 8.0)
        bot.send_private_msg.assert_awaited_once()
        self.assertEqual(str(bot.send_private_msg.await_args.kwargs["message"]), "after")


if __name__ == "__main__":
    unittest.main()
