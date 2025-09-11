# 在文件顶部的 imports 区，加入：
import asyncio
import time
import uuid
from pathlib import Path
from collections import defaultdict
import sys

dir_path = Path(__file__).parent

sys.path.append(str(dir_path))

from config_manager import config_group_data as gcm

# 在全局变量区替换原来的 chat_lock 机制，加入 SessionManager 实现
class SessionManager:
    """
    简单的会话隔离与并发限制管理器。
    每次请求都会生成一个唯一的 session_id（uuid4）。
    管理器会控制：
      - 全局同时活跃会话数
      - 单群组同时活跃会话数
      - 单私聊同时活跃会话数（保留接口，群/私聊混合的判定需要调用者提供 is_private）
    配置优先从 gcm.session_control 中读取（如果存在），否则使用默认值。
    """
    def __init__(self):
        # 从 gcm 获取配置（如果不存在则用默认值）
        sc = getattr(gcm, "session_control", None)
        if sc:
            self.max_global = getattr(sc, "max_global_sessions", 10)
            self.max_per_group = getattr(sc, "max_group_sessions", 3)
            self.max_per_private = getattr(sc, "max_private_sessions", 3)
            # 可选：按 group_id 细粒度覆盖，期望是 dict {group_id: limit}
            self.per_group_overrides = getattr(sc, "per_group_overrides", {}) or {}
        else:
            self.max_global = 10
            self.max_per_group = 3
            self.max_per_private = 3
            self.per_group_overrides = {}

        # 内部状态
        self._lock = asyncio.Lock()
        self.active_sessions: dict[str, dict] = {}   # session_id -> info
        self.count_per_group: dict[int, int] = defaultdict(int)
        self.count_per_private: dict[int, int] = defaultdict(int)

    async def acquire(self, group_id: int | None, user_id: int | None, is_private: bool = False) -> tuple[str | None, str | None]:
        """
        尝试创建一个会话。如果成功返回 (session_id, None)。
        如果因限制拒绝，返回 (None, reason_str)。
        """
        async with self._lock:
            # 全局限制
            if len(self.active_sessions) >= self.max_global:
                return None, "global_limit"

            # 群限制（group_id 可能为 None）
            if group_id is not None:
                limit = self.per_group_overrides.get(group_id, self.max_per_group)
                if self.count_per_group.get(group_id, 0) >= limit:
                    return None, "group_limit"

            # 私聊限制（如果适用）
            if is_private and user_id is not None:
                if self.count_per_private.get(user_id, 0) >= self.max_per_private:
                    return None, "private_limit"

            # 通过所有检查，分配 session
            sid = str(uuid.uuid4())
            info = {
                "session_id": sid,
                "group_id": group_id,
                "user_id": user_id,
                "is_private": is_private,
                "start_time": time.time(),
            }
            self.active_sessions[sid] = info
            if group_id is not None:
                self.count_per_group[group_id] += 1
            if is_private and user_id is not None:
                self.count_per_private[user_id] += 1

            return sid, None

    async def release(self, session_id: str):
        """释放会话（如果存在）"""
        async with self._lock:
            info = self.active_sessions.pop(session_id, None)
            if not info:
                return
            gid = info.get("group_id")
            uid = info.get("user_id")
            is_private = info.get("is_private", False)
            if gid is not None:
                self.count_per_group[gid] = max(0, self.count_per_group.get(gid, 0) - 1)
                if self.count_per_group[gid] == 0:
                    del self.count_per_group[gid]
            if is_private and uid is not None:
                self.count_per_private[uid] = max(0, self.count_per_private.get(uid, 0) - 1)
                if self.count_per_private[uid] == 0:
                    del self.count_per_private[uid]

    async def stats(self) -> dict:
        """返回当前统计信息，便于调试"""
        async with self._lock:
            return {
                "active_total": len(self.active_sessions),
                "per_group": dict(self.count_per_group),
                "per_private": dict(self.count_per_private),
            }