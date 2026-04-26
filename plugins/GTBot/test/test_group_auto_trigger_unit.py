from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plugins.GTBot.services.trigger.auto import GroupAutoTriggerManager as _GroupAutoTriggerManager
    from plugins.GTBot.services.trigger.scan import run_group_auto_trigger_scan as _run_group_auto_trigger_scan
    from plugins.GTBot.services.message.segments import serialize_message_segments as _serialize_message_segments
    from plugins.GTBot.model import GroupMessageRecord as _GroupMessageRecord

    group_auto_trigger_manager_cls: Any | None = _GroupAutoTriggerManager
    run_group_auto_trigger_scan_fn: Any | None = _run_group_auto_trigger_scan
    serialize_message_segments_fn: Any | None = _serialize_message_segments
    group_message_record_cls: Any | None = _GroupMessageRecord

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    group_auto_trigger_manager_cls = None
    run_group_auto_trigger_scan_fn = None
    serialize_message_segments_fn = None
    group_message_record_cls = None
    _IMPORT_ERROR = exc


class _FakeMessageManager:
    """提供最近群消息的轻量测试替身。"""

    def __init__(self, messages_by_group: dict[int, list[Any]]) -> None:
        self._messages_by_group = messages_by_group

    async def get_recent_messages(self, limit: int = 10, group_id: int | None = None, **_: object) -> list[Any]:
        """按群号返回最近消息列表。

        Args:
            limit: 需要返回的最大消息数。
            group_id: 目标群号。
            **_: 忽略的其余查询参数。

        Returns:
            截断后的最近消息列表。
        """

        messages = list(self._messages_by_group.get(int(group_id or 0), []))
        return messages[: int(limit)]


def _require_test_runtime() -> tuple[Any, Any, Any, Any]:
    """返回当前测试所需的运行时对象。

    Returns:
        依次返回自动触发管理器类、扫描函数、消息序列化函数、消息记录类。

    Raises:
        unittest.SkipTest: 当前环境缺少运行依赖时抛出。
    """

    if (
        _IMPORT_ERROR is not None
        or group_auto_trigger_manager_cls is None
        or run_group_auto_trigger_scan_fn is None
        or serialize_message_segments_fn is None
        or group_message_record_cls is None
    ):
        raise unittest.SkipTest(f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")

    return (
        group_auto_trigger_manager_cls,
        run_group_auto_trigger_scan_fn,
        serialize_message_segments_fn,
        group_message_record_cls,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"运行环境缺少依赖，已跳过: {_IMPORT_ERROR}")
class TestGroupAutoTriggerUnit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        (
            group_auto_trigger_manager_cls,
            run_group_auto_trigger_scan_fn,
            serialize_message_segments_fn,
            group_message_record_cls,
        ) = _require_test_runtime()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.manager = group_auto_trigger_manager_cls(
            session_maker=self.session_maker,
            engine_obj=self.engine,
        )
        self.run_group_auto_trigger_scan = run_group_auto_trigger_scan_fn
        self.serialize_message_segments = serialize_message_segments_fn
        self.group_message_record_cls = group_message_record_cls

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def test_manager_entry_settings_and_state(self) -> None:
        created = await self.manager.add_entry(group_id=123456, operator_user_id=1)
        self.assertTrue(created)
        self.assertTrue(await self.manager.is_group_enabled(123456))
        self.assertFalse(await self.manager.is_group_enabled(654321))

        await self.manager.set_probability(group_id=None, probability="80%", operator_user_id=1)
        await self.manager.set_cooldown_seconds(group_id=None, cooldown_seconds=900, operator_user_id=1)
        await self.manager.set_probability(group_id=123456, probability=25, operator_user_id=1)
        await self.manager.set_cooldown_seconds(group_id=123456, cooldown_seconds=120, operator_user_id=1)

        self.assertEqual(await self.manager.get_effective_probability(123456), 25.0)
        self.assertEqual(await self.manager.get_effective_probability(999999), 80.0)
        self.assertEqual(await self.manager.get_effective_cooldown_seconds(123456), 120.0)
        self.assertEqual(await self.manager.get_effective_cooldown_seconds(999999), 900.0)

        await self.manager.mark_triggered(123456, triggered_at=1000.0)
        self.assertEqual(await self.manager.get_last_triggered_at(123456), 1000.0)
        self.assertFalse(await self.manager.is_cooldown_ready(123456, now_ts=1050.0, cooldown_seconds=120.0))
        self.assertTrue(await self.manager.is_cooldown_ready(123456, now_ts=1125.0, cooldown_seconds=120.0))

    async def test_scan_triggers_and_marks_state(self) -> None:
        await self.manager.add_entry(group_id=10001, operator_user_id=1)
        await self.manager.set_probability(group_id=10001, probability=100, operator_user_id=1)
        await self.manager.set_cooldown_seconds(group_id=10001, cooldown_seconds=60, operator_user_id=1)

        serialized_segments = self.serialize_message_segments([])
        messages = [
            self.group_message_record_cls(
                db_id=1,
                message_id=10,
                group_id=10001,
                user_id=20,
                user_name="Alice",
                content="hello",
                serialized_segments=serialized_segments,
                send_time=100.0,
                is_withdrawn=False,
            ),
            self.group_message_record_cls(
                db_id=2,
                message_id=11,
                group_id=10001,
                user_id=21,
                user_name="Bob",
                content="world",
                serialized_segments='[{"type":"text","data":{"text":"world"}}]',
                send_time=120.0,
                is_withdrawn=False,
            ),
        ]
        message_manager = _FakeMessageManager({10001: list(reversed(messages))})
        calls: list[dict[str, object]] = []

        async def _runner(**kwargs: object) -> None:
            calls.append(kwargs)

        triggered = await self.run_group_auto_trigger_scan(
            bot=object(),  # type: ignore[arg-type]
            manager=self.manager,
            message_manager=message_manager,
            cache=object(),
            runner=_runner,
            now_ts=2000.0,
        )

        self.assertEqual(triggered, [10001])
        self.assertEqual(len(calls), 1)
        first_call = cast(dict[str, Any], calls[0])
        latest_message = first_call["latest_message"]
        trigger_meta = cast(dict[str, Any], first_call["trigger_meta"])
        self.assertEqual(getattr(latest_message, "message_id", None), 11)
        self.assertEqual(trigger_meta["reason"], "scheduled")
        self.assertEqual(trigger_meta["latest_message_id"], 11)
        self.assertEqual(trigger_meta["recent_message_count"], 2)
        self.assertTrue(trigger_meta["has_serialized_message"])
        self.assertEqual(await self.manager.get_last_triggered_at(10001), 2000.0)

    async def test_scan_respects_probability_and_cooldown(self) -> None:
        await self.manager.add_entry(group_id=20002, operator_user_id=1)
        await self.manager.set_probability(group_id=20002, probability=50, operator_user_id=1)
        await self.manager.set_cooldown_seconds(group_id=20002, cooldown_seconds=300, operator_user_id=1)
        await self.manager.mark_triggered(20002, triggered_at=1000.0)

        message_manager = _FakeMessageManager(
            {
                20002: [
                    self.group_message_record_cls(
                        db_id=1,
                        message_id=20,
                        group_id=20002,
                        user_id=99,
                        user_name="Carol",
                        content="hello again",
                        serialized_segments=None,
                        send_time=100.0,
                        is_withdrawn=False,
                    )
                ]
            }
        )

        async def _runner(**_: object) -> None:
            raise AssertionError("cooldown/probability 未拦截，runner 不应被调用")

        triggered = await self.run_group_auto_trigger_scan(
            bot=object(),  # type: ignore[arg-type]
            manager=self.manager,
            message_manager=message_manager,
            cache=object(),
            runner=_runner,
            now_ts=1100.0,
        )
        self.assertEqual(triggered, [])

        with patch("plugins.GTBot.group_auto_trigger_tasks.random.random", return_value=0.99):
            triggered = await self.run_group_auto_trigger_scan(
                bot=object(),  # type: ignore[arg-type]
                manager=self.manager,
                message_manager=message_manager,
                cache=object(),
                runner=_runner,
                now_ts=1500.0,
            )
        self.assertEqual(triggered, [])


if __name__ == "__main__":
    unittest.main()
