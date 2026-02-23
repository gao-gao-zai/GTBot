from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import cast


# 允许从仓库根目录直接运行本脚本（无需安装为 site-packages）。
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 禁用 LongMemory 导入即初始化（避免测试时连接外部 Qdrant/Embedding）。
os.environ.setdefault("GTBOT_LONGMEMORY_AUTOINIT", "0")

from plugins.GTBot.model import GroupMessage
from plugins.GTBot.tools.long_memory.IngestManager import LongMemoryIngestConfig, LongMemoryIngestManager
from plugins.GTBot.tools.long_memory import LongMemoryContainer


class _FakePublicKnowledge:
    async def search_public_knowledge(self, query: str, *, n_results: int = 5, min_similarity: float | None = None):
        return []


class _FakeEventLogManager:
    async def search_events(
        self,
        query: str,
        *,
        n_results: int = 5,
        session_id: str | None = None,
        relevant_members_any: list[int] | None = None,
        min_similarity: float | None = None,
        order_by: str = "similarity",
        order: str = "desc",
        touch_last_called: bool = True,
    ):
        return []


class _FakeUserProfileManager:
    async def search_user_profiles(
        self,
        query: str,
        *,
        n_results: int = 10,
        order_by: str = "similarity",
        order: str = "desc",
    ):
        return []


class _FakeGroupProfileManager:
    async def get_group_profiles(
        self,
        group_id: int,
        *,
        limit: int = 20,
        sort_by: str = "last_updated",
        sort_order: str = "desc",
        touch_read_time: bool = True,
    ):
        return {"id": group_id, "description": []}


class _FakeLongMemory:
    def __init__(self) -> None:
        self.public_knowledge = _FakePublicKnowledge()
        self.event_log_manager = _FakeEventLogManager()
        self.user_profile_manager = _FakeUserProfileManager()
        self.group_profile_manager = _FakeGroupProfileManager()


async def _fake_runner(runtime_ctx, messages, extra_context) -> str:
    return f"ok messages={len(messages)}"


async def main() -> None:
    cfg = LongMemoryIngestConfig(
        processed_capacity=3,
        pending_capacity=4,
        flush_pending_threshold=3,
        idle_flush_seconds=0.2,
        flush_max_messages=2,
        max_concurrent_flushes=1,
        model_id="",
        base_url="",
        api_key="",
    )

    mgr = LongMemoryIngestManager(
        config=cfg,
        long_memory=cast(LongMemoryContainer, _FakeLongMemory()),
        ingest_runner=_fake_runner,
    )

    sid = "group_123"

    # 1) pending 溢出：只保留 B 条
    for i in range(6):
        await mgr.add_message(
            session_id=sid,
            message=GroupMessage(
                message_id=100 + i,
                group_id=123,
                user_id=1,
                user_name="u",
                content=f"m{i}",
                send_time=time.time(),
                is_withdrawn=False,
            ),
            group_id=123,
            user_id=1,
        )

    # 等待 idle flush
    await asyncio.sleep(0.25)

    # 再强制 flush 一次，确保不抛异常
    await mgr.flush_session(session_id=sid, group_id=123, user_id=1, reason="manual")

    print("ingest manager queue test done")


if __name__ == "__main__":
    asyncio.run(main())
