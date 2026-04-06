from __future__ import annotations

import json
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
    import plugins.GTBot.services.chat.continuation as continuation_mod

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    continuation_mod = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"runtime dependencies unavailable: {_IMPORT_ERROR}")
class TestContinuationUnit(unittest.IsolatedAsyncioTestCase):
    """验证群聊续聊窗口的核心状态流转与提示词载荷构造。"""

    def _patch_continuation_cfg(self, **overrides: object) -> Any:
        """临时覆盖续聊配置，便于单测稳定控制阈值与模型参数。

        Args:
            **overrides: 需要覆盖到假配置对象上的字段。

        Returns:
            `patch.object(...)` 生成的 patcher，供 `with` 语句使用。
        """

        assert continuation_mod is not None
        fake_cfg = SimpleNamespace(
            enabled=True,
            window_seconds=30.0,
            debounce_seconds=10.0,
            scope="all",
            analyzer_provider="judge",
            analyzer_model_id="judge-mini",
            analyzer_provider_type="openai_compatible",
            analyzer_base_url="https://judge.example/v1",
            analyzer_api_key="secret",
            analyzer_parameters={},
            max_pending_messages=4,
            max_accumulated_messages=2,
            pre_history_messages=2,
            max_analyzer_context_messages=40,
        )
        for key, value in overrides.items():
            setattr(fake_cfg, key, value)
        return patch.object(continuation_mod.config.chat_model, "continuation", fake_cfg)

    def _fake_bot(self, self_id: str = "114514") -> Any:
        """构造仅包含 `self_id` 的机器人替身对象。"""

        return cast(Any, SimpleNamespace(self_id=self_id))

    def _fake_msg_manager(self, recent_messages: list[Any] | None = None) -> Any:
        """构造最小化消息管理器替身。

        Args:
            recent_messages: 需要由 `get_recent_messages` 返回的消息列表。

        Returns:
            带有异步 `get_recent_messages` 方法的替身对象。
        """

        return cast(
            Any,
            SimpleNamespace(
                get_recent_messages=AsyncMock(return_value=list(recent_messages or [])),
            ),
        )

    def _fake_record(
        self,
        *,
        message_id: int,
        user_id: int,
        send_time: float,
        content: str = "",
        user_name: str = "",
        serialized_segments: str | None = None,
    ) -> Any:
        """构造满足续聊流程访问字段的消息记录替身。"""

        return cast(
            Any,
            SimpleNamespace(
                db_id=message_id,
                group_id=123,
                message_id=message_id,
                user_id=user_id,
                user_name=user_name,
                send_time=send_time,
                content=content,
                serialized_segments=serialized_segments,
                is_withdrawn=False,
            ),
        )

    async def test_register_incoming_message_forces_final_analysis_at_accumulated_limit(self) -> None:
        """累计消息达到上限时，应立即执行最终分析并关闭窗口。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        bot = self._fake_bot()
        msg_mg = self._fake_msg_manager()

        with self._patch_continuation_cfg():
            await manager.open_window(
                session_id="group:123",
                group_id=123,
                source_trigger_mode="group_at",
                last_response_id="resp_1",
                last_bot_message_id=999,
            )
            self.assertTrue(await manager.has_active_window("group:123"))

            with patch.object(manager, "_run_analysis_and_maybe_reply", new=AsyncMock()) as analysis_mock:
                await manager.register_incoming_message(
                    session_id="group:123",
                    group_id=123,
                    message_id=1,
                    bot=bot,
                    msg_mg=msg_mg,
                    cache=object(),
                )
                self.assertTrue(await manager.has_active_window("group:123"))

                await manager.register_incoming_message(
                    session_id="group:123",
                    group_id=123,
                    message_id=2,
                    bot=bot,
                    msg_mg=msg_mg,
                    cache=object(),
                )

            self.assertFalse(await manager.has_active_window("group:123"))
            analysis_mock.assert_awaited_once()
            self.assertIsNotNone(analysis_mock.await_args)
            if analysis_mock.await_args is not None:
                await_kwargs = analysis_mock.await_args.kwargs
                self.assertTrue(await_kwargs["final_analysis"])

    async def test_negative_analysis_keeps_window_and_positive_analysis_closes_it(self) -> None:
        """判定为不回复时保留窗口，判定为回复时应立即关窗并续接主链路。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        bot = self._fake_bot()
        msg_mg = self._fake_msg_manager()
        state = continuation_mod.ContinuationWindowState(
            session_id="group:456",
            group_id=456,
            opened_at=0.0,
            expires_at=9999999999.0,
            source_trigger_mode="group_keyword",
            last_response_id="resp_2",
            last_bot_message_id=123,
            pending_message_ids=[1001],
            accumulated_message_ids=[1001],
        )

        with self._patch_continuation_cfg():
            manager._windows[state.session_id] = state
            with (
                patch.object(
                    manager,
                    "_analyze_window",
                    new=AsyncMock(
                        return_value=continuation_mod.ContinuationAnalysisResult(
                            should_reply=False,
                            confidence=0.2,
                            reason="not_about_bot",
                            focus_message_ids=[],
                            raw_output='{"should_reply": false}',
                        )
                    ),
                ),
                patch.object(manager, "_continue_with_main_chain", new=AsyncMock()) as continue_mock,
            ):
                await manager._run_analysis_and_maybe_reply(
                    state=state,
                    bot=bot,
                    msg_mg=msg_mg,
                    cache=object(),
                    final_analysis=False,
                    pending_message_ids=[1001],
                )

            self.assertTrue(await manager.has_active_window("group:456"))
            continue_mock.assert_not_awaited()

            manager._windows[state.session_id] = state
            with (
                patch.object(
                    manager,
                    "_analyze_window",
                    new=AsyncMock(
                        return_value=continuation_mod.ContinuationAnalysisResult(
                            should_reply=True,
                            confidence=0.9,
                            reason="follow_up_to_bot",
                            focus_message_ids=[1001],
                            raw_output='{"should_reply": true}',
                        )
                    ),
                ),
                patch.object(manager, "_continue_with_main_chain", new=AsyncMock()) as continue_mock,
            ):
                await manager._run_analysis_and_maybe_reply(
                    state=state,
                    bot=bot,
                    msg_mg=msg_mg,
                    cache=object(),
                    final_analysis=False,
                    pending_message_ids=[1001],
                )

            self.assertFalse(await manager.has_active_window("group:456"))
            continue_mock.assert_awaited_once()

    async def test_open_window_keeps_timeout_from_response_end_but_records_history_start(self) -> None:
        """开窗超时应从响应结束算起，但历史起点需要单独记住。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()

        with self._patch_continuation_cfg(), patch.object(continuation_mod, "time", return_value=200.0):
            await manager.open_window(
                session_id="group:789",
                group_id=789,
                source_trigger_mode="group_at",
                last_response_id="resp_3",
                last_bot_message_id=888,
                trigger_started_at=150.0,
            )

        state = manager._windows["group:789"]
        self.assertEqual(state.opened_at, 200.0)
        self.assertEqual(state.expires_at, 230.0)
        self.assertEqual(state.history_started_at, 150.0)

    async def test_open_window_preloads_messages_received_during_response(self) -> None:
        """响应期间到达的用户消息应在开窗后直接并入待分析缓冲区。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        bot = self._fake_bot()
        msg_mg = self._fake_msg_manager(
            [
                self._fake_record(message_id=10, user_id=20001, send_time=119.0),
                self._fake_record(message_id=11, user_id=20001, send_time=120.0),
                self._fake_record(message_id=12, user_id=114514, send_time=121.0),
                self._fake_record(message_id=13, user_id=20002, send_time=122.0),
            ]
        )

        with self._patch_continuation_cfg(max_accumulated_messages=4), patch.object(
            continuation_mod, "time", return_value=200.0
        ):
            await manager.open_window(
                session_id="group:900",
                group_id=900,
                source_trigger_mode="group_at",
                last_response_id="resp_9",
                last_bot_message_id=777,
                trigger_started_at=120.0,
                bot=bot,
                msg_mg=msg_mg,
                cache=object(),
            )

        state = manager._windows["group:900"]
        self.assertEqual(state.buffered_message_ids, [13])
        self.assertEqual(state.accumulated_message_ids, [])
        self.assertEqual(state.pending_message_ids, [13])
        self.assertIsNotNone(state.debounce_task)
        await manager.close_window("group:900", reason="test_cleanup")

    async def test_open_window_preloaded_messages_do_not_consume_accumulated_limit(self) -> None:
        """响应期间预载入的消息不应占用累计上限计数。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        bot = self._fake_bot()
        msg_mg = self._fake_msg_manager(
            [
                self._fake_record(message_id=21, user_id=20001, send_time=121.0),
                self._fake_record(message_id=22, user_id=20002, send_time=122.0),
            ]
        )

        with self._patch_continuation_cfg(max_accumulated_messages=2), patch.object(
            continuation_mod, "time", return_value=200.0
        ):
            await manager.open_window(
                session_id="group:901",
                group_id=901,
                source_trigger_mode="group_at",
                last_response_id="resp_10",
                last_bot_message_id=778,
                trigger_started_at=120.0,
                bot=bot,
                msg_mg=msg_mg,
                cache=object(),
            )

        self.assertIn("group:901", manager._windows)
        state = manager._windows["group:901"]
        self.assertEqual(state.buffered_message_ids, [21, 22])
        self.assertEqual(state.accumulated_message_ids, [])
        await manager.close_window("group:901", reason="test_cleanup")

    def test_build_prompt_payload_includes_last_llm_reply_and_excludes_buffered_from_limit_count(self) -> None:
        """提示词载荷中需要保留机器人回复，并正确标记是否计入累计上限。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        state = continuation_mod.ContinuationWindowState(
            session_id="group:902",
            group_id=902,
            opened_at=100.0,
            expires_at=130.0,
            source_trigger_mode="group_at",
            last_response_id="resp_11",
            last_bot_message_id=500,
            buffered_message_ids=[301],
            accumulated_message_ids=[302],
        )
        bot_message = self._fake_record(
            message_id=500,
            user_id=114514,
            user_name="GTBot",
            send_time=99.0,
            content="上一条机器人回复",
        )
        conversation_messages: list[Any] = [
            self._fake_record(
                message_id=500,
                user_id=114514,
                user_name="GTBot",
                send_time=99.0,
                content="上一条机器人回复",
            ),
            self._fake_record(
                message_id=301,
                user_id=20001,
                user_name="Alice",
                send_time=101.0,
                content="补充一句",
            ),
            self._fake_record(
                message_id=302,
                user_id=20002,
                user_name="Bob",
                send_time=102.0,
                content="再追问一句",
            ),
        ]
        analysis_messages: list[Any] = [conversation_messages[-1]]

        with patch.object(continuation_mod, "time", return_value=105.0):
            payload = manager._build_prompt_payload(
                state=state,
                last_bot_message=bot_message,
                conversation_messages=conversation_messages,
                analysis_messages=analysis_messages,
                focus_messages=analysis_messages,
                final_analysis=False,
                formatted_history="[GTBot]: 上一条机器人回复\n[Alice]: 补充一句\n[Bob]: 再追问一句",
                bot_self_id=114514,
            )

        self.assertEqual(payload["session"]["accumulated_count"], 1)
        self.assertEqual(payload["session"]["buffered_count"], 1)
        self.assertEqual(payload["conversation_messages"][0]["role"], "assistant")
        self.assertEqual(payload["conversation_messages"][0]["message_id"], 500)
        self.assertFalse(payload["conversation_messages"][1]["counts_towards_limit"])
        self.assertTrue(payload["conversation_messages"][2]["counts_towards_limit"])
        self.assertIn("上一条机器人回复", payload["formatted_chat_history"])
        self.assertEqual(payload["session"]["focus_count"], 1)
        self.assertEqual(payload["focus_messages"][0]["message_id"], 302)

    async def test_analyze_window_formats_full_conversation_history_for_analyzer(self) -> None:
        """分析器输入应包含累计对话轨迹和清洗后的格式化聊天记录。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        state = continuation_mod.ContinuationWindowState(
            session_id="group:903",
            group_id=903,
            opened_at=100.0,
            expires_at=130.0,
            source_trigger_mode="group_continuation",
            last_response_id="resp_12",
            last_bot_message_id=500,
            history_started_at=98.0,
            buffered_message_ids=[301],
            accumulated_message_ids=[302],
        )
        msg_mg = self._fake_msg_manager(
            [
                self._fake_record(
                    message_id=302,
                    user_id=20002,
                    user_name="Bob",
                    send_time=102.0,
                    content="继续追问",
                ),
                self._fake_record(
                    message_id=301,
                    user_id=20001,
                    user_name="Alice",
                    send_time=101.0,
                    content="[CQ:image,file=a.jpg]",
                ),
                self._fake_record(
                    message_id=500,
                    user_id=114514,
                    user_name="GTBot",
                    send_time=99.0,
                    content="上一条机器人回复",
                ),
            ]
        )
        captured_payload: dict[str, Any] = {}
        bot = self._fake_bot()

        def fake_build_chat_model(**kwargs: object) -> SimpleNamespace:
            _ = kwargs

            async def ainvoke(messages: list[Any]) -> SimpleNamespace:
                last_message = cast(Any, messages[-1])
                captured_payload.update(json.loads(last_message.content))
                return SimpleNamespace(content='{"should_reply": false, "reason": "not_needed"}')

            return SimpleNamespace(ainvoke=ainvoke)

        with (
            self._patch_continuation_cfg(max_accumulated_messages=4, max_pending_messages=4),
            patch.object(
                manager,
                "_format_conversation_for_analyzer",
                new=AsyncMock(return_value="[GTBot]: 上一条机器人回复\n[Alice]: [CQ:image,file=a.jpg]\n[Bob]: 继续追问"),
            ) as format_mock,
            patch.object(continuation_mod, "build_chat_model", side_effect=fake_build_chat_model),
        ):
            result = await manager._analyze_window(
                state=state,
                analysis_message_ids=[301, 302],
                focus_message_ids=[302],
                bot=bot,
                msg_mg=msg_mg,
                cache=object(),
            )

        self.assertIsNotNone(result)
        format_mock.assert_awaited_once()
        self.assertEqual(
            captured_payload["formatted_chat_history"],
            "[GTBot]: 上一条机器人回复\n[Alice]: [CQ:image,file=a.jpg]\n[Bob]: 继续追问",
        )
        conversation_payload = cast(list[dict[str, Any]], captured_payload["conversation_messages"])
        analysis_payload = cast(list[dict[str, Any]], captured_payload["analysis_messages"])
        focus_payload = cast(list[dict[str, Any]], captured_payload["focus_messages"])
        self.assertEqual(len(conversation_payload), 3)
        self.assertEqual(conversation_payload[0]["role"], "assistant")
        self.assertEqual(analysis_payload[0]["message_id"], 301)
        self.assertEqual(analysis_payload[1]["message_id"], 302)
        self.assertEqual(focus_payload[0]["message_id"], 302)

    def test_select_history_messages_for_analyzer_includes_pre_history_messages(self) -> None:
        """分析上下文应向前补齐配置指定数量的响应前历史。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        state = continuation_mod.ContinuationWindowState(
            session_id="group:904",
            group_id=904,
            opened_at=100.0,
            expires_at=130.0,
            source_trigger_mode="group_continuation",
            last_response_id="resp_13",
            history_started_at=50.0,
        )
        recent_messages: list[Any] = [
            self._fake_record(message_id=1, user_id=20001, user_name="A", send_time=10.0, content="old1"),
            self._fake_record(message_id=2, user_id=20002, user_name="B", send_time=20.0, content="old2"),
            self._fake_record(message_id=3, user_id=20003, user_name="C", send_time=30.0, content="old3"),
            self._fake_record(message_id=4, user_id=20004, user_name="D", send_time=40.0, content="old4"),
            self._fake_record(message_id=5, user_id=114514, user_name="GTBot", send_time=55.0, content="bot reply"),
            self._fake_record(message_id=6, user_id=20005, user_name="E", send_time=56.0, content="new1"),
            self._fake_record(message_id=7, user_id=20006, user_name="F", send_time=57.0, content="new2"),
        ]

        with self._patch_continuation_cfg(pre_history_messages=2):
            history = manager._select_history_messages_for_analyzer(
                recent_messages=recent_messages,
                state=state,
                bot_self_id=114514,
            )

        self.assertEqual([item.message_id for item in history], [3, 4, 5, 6, 7])

    def test_select_history_messages_for_analyzer_truncates_to_max_context_messages(self) -> None:
        """分析上下文超过上限时，应只保留最后一段窗口相关消息。"""

        assert continuation_mod is not None
        manager = continuation_mod.ContinuationManager()
        state = continuation_mod.ContinuationWindowState(
            session_id="group:905",
            group_id=905,
            opened_at=100.0,
            expires_at=130.0,
            source_trigger_mode="group_continuation",
            last_response_id="resp_14",
            history_started_at=50.0,
        )
        recent_messages: list[Any] = [
            self._fake_record(
                message_id=index,
                user_id=20000 + index,
                user_name=f"U{index}",
                send_time=float(index),
                content=f"m{index}",
            )
            for index in range(1, 9)
        ]

        with self._patch_continuation_cfg(pre_history_messages=2, max_analyzer_context_messages=4):
            history = manager._select_history_messages_for_analyzer(
                recent_messages=recent_messages,
                state=state,
                bot_self_id=114514,
            )

        self.assertEqual([item.message_id for item in history], [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
