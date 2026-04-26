from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from plugins.GTBot.ConfigManager import total_config

from .models import ManagedFileRecord


class ManagedFileStore:
    """负责文件注册表 SQLite 读写的底层存储对象。

    该类只处理数据库初始化、记录读写与删除，不承担业务层的路径校验或文件生成。
    所有写操作都通过同一把进程内锁串行化，避免多协程或多线程并发注册时出现
    表结构初始化竞态或局部写入覆盖。
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """初始化文件注册表存储。

        Args:
            db_path: 自定义 SQLite 路径。未传入时默认使用 GTBot 数据目录下的
                `file_registry.sqlite3`。
        """

        data_dir = total_config.get_data_dir_path()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path) if db_path is not None else data_dir / "file_registry.sqlite3"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> Path:
        """返回当前存储对象使用的数据库路径。"""

        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """创建新的 SQLite 连接。

        Returns:
            已配置 `Row` 工厂的数据库连接对象。
        """

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化文件注册表数据库结构。

        该方法是幂等的，可在多次构造或测试重建过程中重复调用，不会破坏已有数据。
        """

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS managed_files (
                        file_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        local_path TEXT NOT NULL,
                        original_name TEXT,
                        mime_type TEXT,
                        size_bytes INTEGER NOT NULL,
                        sha256 TEXT NOT NULL,
                        session_id TEXT,
                        group_id INTEGER,
                        user_id INTEGER,
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        extra_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_managed_files_sha256 ON managed_files(sha256)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_managed_files_kind_created ON managed_files(kind, created_at)"
                )
                conn.commit()

    def insert_record(self, record: ManagedFileRecord) -> None:
        """插入一条新的文件记录。

        Args:
            record: 待持久化的文件记录。
        """

        payload = (
            record.file_id,
            record.kind,
            record.source_type,
            record.local_path.resolve().as_posix(),
            record.original_name,
            record.mime_type,
            int(record.size_bytes),
            record.sha256,
            record.session_id,
            record.group_id,
            record.user_id,
            float(record.created_at),
            record.expires_at,
            json.dumps(record.extra, ensure_ascii=False, sort_keys=True),
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO managed_files(
                        file_id, kind, source_type, local_path, original_name, mime_type,
                        size_bytes, sha256, session_id, group_id, user_id, created_at,
                        expires_at, extra_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                conn.commit()

    def get_record(self, file_id: str) -> ManagedFileRecord | None:
        """按 `file_id` 读取文件记录。

        Args:
            file_id: 稳定文件句柄。

        Returns:
            成功命中时返回对应记录；否则返回 `None`。
        """

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT file_id, kind, source_type, local_path, original_name, mime_type,
                       size_bytes, sha256, session_id, group_id, user_id, created_at,
                       expires_at, extra_json
                FROM managed_files
                WHERE file_id = ?
                """,
                (str(file_id),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete_record(self, file_id: str) -> ManagedFileRecord | None:
        """删除一条文件记录并返回删除前的内容。

        Args:
            file_id: 目标文件句柄。

        Returns:
            若记录存在则返回删除前的记录；否则返回 `None`。
        """

        existing = self.get_record(file_id)
        if existing is None:
            return None
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM managed_files WHERE file_id = ?", (str(file_id),))
                conn.commit()
        return existing

    def _row_to_record(self, row: sqlite3.Row) -> ManagedFileRecord:
        """将 SQLite 行对象转换为文件记录。

        Args:
            row: 原始数据库行。

        Returns:
            规范化后的文件记录对象。
        """

        extra_raw = row["extra_json"]
        extra: dict[str, Any]
        if isinstance(extra_raw, str) and extra_raw.strip():
            loaded = json.loads(extra_raw)
            extra = loaded if isinstance(loaded, dict) else {}
        else:
            extra = {}
        return ManagedFileRecord(
            file_id=str(row["file_id"]),
            kind=str(row["kind"]),
            source_type=str(row["source_type"]),
            local_path=Path(str(row["local_path"])).resolve(),
            original_name=str(row["original_name"]) if row["original_name"] is not None else None,
            mime_type=str(row["mime_type"]) if row["mime_type"] is not None else None,
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
            group_id=int(row["group_id"]) if row["group_id"] is not None else None,
            user_id=int(row["user_id"]) if row["user_id"] is not None else None,
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]) if row["expires_at"] is not None else None,
            extra=extra,
        )

