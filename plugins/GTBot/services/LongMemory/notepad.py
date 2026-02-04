"""用于 LLM 的中短期记忆记事本。

该模块提供单个记事本 `Notepad`，以及按会话 ID 管理多个记事本的
`SessionNotepadManager`。
"""

from __future__ import annotations

import datetime
from collections import OrderedDict
from dataclasses import dataclass
from time import time





@dataclass
class Arecord:
    """带时间戳的一条记录。

    Attributes:
        timestamp: 记录的 Unix 时间戳（秒）。
        content: 记录内容。
    """

    timestamp: float
    content: str
    




class Notepad:
    """记事本类，用于管理和存储带时间戳的记录。

    该类维护一个具有最大长度限制的记录列表。当记录数量超过限制时，会自动移除最早的记录。

    Attributes:
        notes: 存储 `Arecord` 对象的列表。
        max_length: 记事本允许存储的最大记录条数。
    """

    def __init__(self, max_length: int = 15):
        """初始化记事本。

        Args:
            max_length: 最大记录条数。默认为 15。
        """
        if max_length <= 0:
            raise ValueError("max_length 必须为正整数")
        self.notes: list[Arecord] = []
        self.max_length = max_length

    def add_note(self, note: Arecord | str) -> None:
        """向记事本添加一条新记录。

        如果传入的是字符串，会自动将其封装为带有当前时间戳的 `Arecord` 对象。
        如果添加记录后总数超过 `max_length`，将删除最早的记录。

        Args:
            note: 记录内容，可以是 `Arecord` 对象或纯字符串。
        """
        if isinstance(note, str):
            note = Arecord(timestamp=time(), content=note)
        
        self.notes.append(note)

        if len(self.notes) >  self.max_length:
            del self.notes[:-self.max_length]

    def get_notes(self) -> str:
        """获取所有记录的格式化字符串。

        将 `notes` 中的所有记录转换为 "[HH:MM:SS]: 内容" 的格式，并以换行符拼接。

        Returns:
            格式化后的所有记录内容。
        """
        notes: list[str] = []
        for note in self.notes:
            time_str = datetime.datetime.fromtimestamp(note.timestamp).strftime('%H:%M:%S')
            notes.append(f"[{time_str}]: {note.content}")
        return "\n".join(notes)

    def clear_notes(self) -> None:
        """清空记事本中的所有记录。"""
        self.notes = []


@dataclass
class _SessionEntry:
    """内部会话条目。

    Attributes:
        notepad: 会话对应的记事本实例。
        last_used: 最近一次访问时间（Unix 时间戳，秒）。
    """

    notepad: Notepad
    last_used: float


class SessionNotepadManager:
    """会话记事本管理器，用于管理不同会话的记事本。

    不同会话通过唯一的 `session_id` 区分。默认情况下每个会话拥有一个独立 `Notepad`。
    可选启用 `max_sessions` 以限制会话数量，超限时按 LRU（最近最少使用）策略淘汰。

    Attributes:
        notepad_max_length: 新创建会话记事本的最大记录条数。
        max_sessions: 允许同时存在的最大会话数；为 `None` 表示不限制。
        session_timeout_seconds: 会话闲置超时时间（秒）；为 `None` 表示不超时。
    """

    def __init__(
        self,
        notepad_max_length: int = 15,
        max_sessions: int | None = None,
        session_timeout_seconds: float | None = None,
    ) -> None:
        """初始化会话记事本管理器。

        Args:
            notepad_max_length: 每个会话记事本的最大记录条数。
            max_sessions: 最大会话数限制；为 `None` 表示不限制。
            session_timeout_seconds: 会话闲置超时时间（秒）；为 `None` 表示不超时。
        """
        if notepad_max_length <= 0:
            raise ValueError("notepad_max_length 必须为正整数")
        if max_sessions is not None and max_sessions <= 0:
            raise ValueError("max_sessions 必须为正整数或 None")
        if session_timeout_seconds is not None and session_timeout_seconds <= 0:
            raise ValueError("session_timeout_seconds 必须为正数或 None")

        self.notepad_max_length = notepad_max_length
        self.max_sessions = max_sessions
        self.session_timeout_seconds = session_timeout_seconds
        self._notepads: OrderedDict[str, _SessionEntry] = OrderedDict()

    def has_session(self, session_id: str) -> bool:
        """判断会话是否已存在。

        Args:
            session_id: 会话唯一 ID。

        Returns:
            会话是否存在。
        """
        self._evict_expired(time())
        return session_id in self._notepads

    def get_notepad(self, session_id: str) -> Notepad:
        """获取指定会话的记事本，不存在则创建。

        Args:
            session_id: 会话唯一 ID。

        Returns:
            该会话对应的 `Notepad` 实例。
        """
        now = time()
        self._evict_expired(now)

        entry = self._notepads.get(session_id)
        if entry is None:
            entry = _SessionEntry(
                notepad=Notepad(max_length=self.notepad_max_length),
                last_used=now,
            )
            self._notepads[session_id] = entry
        else:
            entry.last_used = now
            self._notepads.move_to_end(session_id)

        self._evict_if_needed()
        return entry.notepad

    def add_note(self, session_id: str, note: Arecord | str) -> None:
        """向指定会话追加一条记录。

        Args:
            session_id: 会话唯一 ID。
            note: 记录内容，可以是 `Arecord` 对象或纯字符串。
        """
        self.get_notepad(session_id).add_note(note)

    def get_notes(self, session_id: str) -> str:
        """获取指定会话的所有记录（格式化）。

        Args:
            session_id: 会话唯一 ID。

        Returns:
            格式化后的所有记录内容。
        """
        return self.get_notepad(session_id).get_notes()

    def clear_session(self, session_id: str) -> None:
        """清空指定会话的记录（保留会话）。

        Args:
            session_id: 会话唯一 ID。
        """
        self.get_notepad(session_id).clear_notes()

    def remove_session(self, session_id: str) -> bool:
        """移除指定会话及其记事本。

        Args:
            session_id: 会话唯一 ID。

        Returns:
            是否成功移除（会话存在则为 True）。
        """
        self._evict_expired(time())
        if session_id not in self._notepads:
            return False
        del self._notepads[session_id]
        return True

    def clear_all(self) -> None:
        """清空所有会话及其记事本。"""
        self._notepads.clear()

    def session_ids(self) -> list[str]:
        """获取当前已管理的会话 ID 列表。

        Returns:
            会话 ID 列表。
        """
        self._evict_expired(time())
        return list(self._notepads.keys())

    def sweep_expired(self) -> int:
        """手动清理已超时的会话。

        Returns:
            本次清理移除的会话数量。
        """
        before = len(self._notepads)
        self._evict_expired(time())
        return before - len(self._notepads)

    def _evict_if_needed(self) -> None:
        """当会话数量超过上限时按 LRU 淘汰。"""
        if self.max_sessions is None:
            return
        while len(self._notepads) > self.max_sessions:
            self._notepads.popitem(last=False)

    def _evict_expired(self, now: float) -> None:
        """清理超过闲置超时的会话。

        该方法假设 `self._notepads` 的顺序与最近使用时间一致（访问会 move_to_end）。
        因此只需要从最旧的会话开始检查并弹出。

        Args:
            now: 当前时间戳（秒）。
        """
        if self.session_timeout_seconds is None:
            return

        expire_before = now - self.session_timeout_seconds
        while self._notepads:
            oldest_session_id, oldest_entry = next(iter(self._notepads.items()))
            if oldest_entry.last_used >= expire_before:
                return
            del self._notepads[oldest_session_id]