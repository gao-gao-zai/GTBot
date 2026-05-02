from __future__ import annotations

import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
