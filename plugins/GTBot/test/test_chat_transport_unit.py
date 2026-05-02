from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plugins.GTBot.services.chat.runtime as chat_runtime

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    chat_runtime = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestChatTransportUnit(unittest.IsolatedAsyncioTestCase):
    """验证聊天发送器对 Agent 与非 Agent 消息的边界处理。"""

    async def test_private_send_feedback_uses_direct_send_path(self) -> None:
        """私聊反馈消息应改为直发，不再进入私聊消息队列。"""

        assert chat_runtime is not None
        turn = chat_runtime.ChatTurn(
            session=chat_runtime.ChatSession(
                session_id="private:123",
                chat_type="private",
                group_id=None,
                peer_user_id=123,
            ),
            sender_user_id=123,
        )
        transport = chat_runtime.PrivateChatTransport(
            bot=cast(Any, object()),
            message_manager=cast(Any, object()),
            cache=cast(Any, object()),
            turn=turn,
        )

        with patch.object(chat_runtime, "send_private_messages_direct", AsyncMock()) as direct_send_mock, patch.object(
            chat_runtime.private_message_queue_manager,
            "enqueue",
            AsyncMock(),
        ) as enqueue_mock:
            await transport.send_feedback("系统提示")

        direct_send_mock.assert_awaited_once()
        enqueue_mock.assert_not_awaited()

    async def test_private_reserve_message_slot_should_enqueue_placeholder_with_default_timeout(self) -> None:
        """私聊占位接口应把占位条目压入私聊队列，并回退到默认超时。"""

        assert chat_runtime is not None
        turn = chat_runtime.ChatTurn(
            session=chat_runtime.ChatSession(
                session_id="private:123",
                chat_type="private",
                group_id=None,
                peer_user_id=123,
            ),
            sender_user_id=123,
        )
        transport = chat_runtime.PrivateChatTransport(
            bot=cast(Any, object()),
            message_manager=cast(Any, object()),
            cache=cast(Any, object()),
            turn=turn,
        )

        with patch(
            "plugins.GTBot.services.chat.send_timing.get_current_send_timing_config",
            return_value=SimpleNamespace(placeholder_timeout_seconds=66.0),
        ), patch.object(
            chat_runtime.private_message_queue_manager,
            "enqueue",
            AsyncMock(),
        ) as enqueue_mock:
            handle = await transport.reserve_message_slot()

        self.assertEqual(handle.status, "pending")
        enqueue_mock.assert_awaited_once()
        await_args = enqueue_mock.await_args
        assert await_args is not None
        task = await_args.args[0]
        self.assertEqual(len(task.messages), 1)
        self.assertTrue(task.messages[0].is_placeholder())
        self.assertEqual(task.messages[0].placeholder_timeout_sec, 66.0)

    async def test_group_reserve_message_slot_should_enqueue_placeholder_with_override(self) -> None:
        """群聊占位接口应透传显式超时、等待时长和强制等待标记。"""

        assert chat_runtime is not None
        turn = chat_runtime.ChatTurn(
            session=chat_runtime.ChatSession(
                session_id="group:456",
                chat_type="group",
                group_id=456,
                peer_user_id=456,
            ),
            sender_user_id=123,
        )
        transport = chat_runtime.GroupChatTransport(
            bot=cast(Any, object()),
            message_manager=cast(Any, object()),
            cache=cast(Any, object()),
            turn=turn,
        )

        with patch.object(
            chat_runtime.group_message_queue_manager,
            "enqueue",
            AsyncMock(),
        ) as enqueue_mock:
            handle = await transport.reserve_message_slot(
                timeout_sec=12.5,
                interval=0.4,
                force_wait=True,
            )

        self.assertEqual(handle.status, "pending")
        enqueue_mock.assert_awaited_once()
        await_args = enqueue_mock.await_args
        assert await_args is not None
        task = await_args.args[0]
        self.assertEqual(task.group_id, 456)
        self.assertEqual(len(task.messages), 1)
        self.assertTrue(task.messages[0].is_placeholder())
        self.assertEqual(task.messages[0].placeholder_timeout_sec, 12.5)
        self.assertEqual(task.messages[0].delay_seconds, 0.4)
        self.assertTrue(task.messages[0].force_wait)


if __name__ == "__main__":
    unittest.main()
