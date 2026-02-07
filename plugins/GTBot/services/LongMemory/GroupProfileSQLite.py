from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import aiosqlite

from .model import GroupProfileDescriptionWithId, GroupProfileWithDescriptionIds


@dataclass(frozen=True)
class _Row:
    """内部行结构。

    Attributes:
        doc_id: 自增主键。
        group_id: 群号。
        description: 描述文本。
        category: 可选分类。
        creation_time: 创建时间戳。
        last_updated: 最近更新时间戳。
        last_read: 最近读取时间戳。
    """

    doc_id: int
    group_id: int
    description: str
    category: str | None
    creation_time: float
    last_updated: float
    last_read: float


class SQLiteGroupProfileStore:
    """群画像 SQLite 存储（无相似度检索）。

    说明：
        - 使用内嵌 SQLite 文件存储群画像条目。
        - 不做向量化，不依赖 Qdrant。
        - 每条画像一行记录，`doc_id` 为自增主键（对外以字符串返回）。

    表结构（默认表名 group_profiles）：
        - doc_id INTEGER PRIMARY KEY AUTOINCREMENT
        - group_id INTEGER NOT NULL
        - description TEXT NOT NULL
        - category TEXT
        - creation_time REAL NOT NULL
        - last_updated REAL NOT NULL
        - last_read REAL NOT NULL

    Attributes:
        db_path: SQLite 文件路径。
        table_name: 表名，默认 "group_profiles"。
    """

    def __init__(self, *, db_path: str | Path, table_name: str = "group_profiles") -> None:
        """初始化存储。

        Args:
            db_path: SQLite 数据库文件路径。
            table_name: 表名。

        Raises:
            ValueError: table_name 为空。
        """

        table = str(table_name).strip()
        if not table:
            raise ValueError("table_name 不能为空")

        self.db_path: Path = Path(db_path)
        self.table_name: str = table
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @classmethod
    def default_db_path(cls) -> Path:
        """获取默认数据库路径。

        Returns:
            Path: 默认路径（plugins/GTBot/data/group_profiles.db）。
        """

        # LongMemory -> services -> GTBot -> plugins
        gtbot_dir = Path(__file__).resolve().parents[2]
        return gtbot_dir / "data" / "group_profiles.db"

    async def connect(self) -> None:
        """建立数据库连接并启用 WAL。"""

        if self._conn is not None:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.commit()

    async def close(self) -> None:
        """关闭数据库连接。"""

        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    async def __aenter__(self) -> "SQLiteGroupProfileStore":
        await self.connect()
        await self.create_tables()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def create_tables(self) -> None:
        """创建表与索引（若不存在）。"""

        await self.connect()
        assert self._conn is not None

        table = self.table_name
        await self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                creation_time REAL NOT NULL,
                last_updated REAL NOT NULL,
                last_read REAL NOT NULL
            );
            """
        )
        await self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_group_id ON {table}(group_id);"
        )
        await self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_last_updated ON {table}(last_updated);"
        )
        await self._conn.commit()

    async def add_group_profile(
        self,
        group_id: int,
        profile_texts: str | Sequence[str],
        *,
        category: str | None = None,
    ) -> list[str]:
        """新增群画像条目。

        Args:
            group_id: 群号。
            profile_texts: 单条或多条文本。
            category: 可选分类。

        Returns:
            list[str]: 新增记录的 doc_id 列表（字符串）。
        """

        await self.create_tables()
        assert self._conn is not None

        if isinstance(profile_texts, str):
            texts = [profile_texts]
        else:
            texts = [str(x) for x in profile_texts]

        texts = [t.strip() for t in texts if str(t).strip()]
        if not texts:
            return []

        now_ts = float(time.time())
        cat = None if category is None else str(category).strip() or None

        doc_ids: list[str] = []
        async with self._write_lock:
            for text in texts:
                cur = await self._conn.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (group_id, description, category, creation_time, last_updated, last_read)
                    VALUES
                        (?, ?, ?, ?, ?, ?);
                    """,
                    (int(group_id), text, cat, now_ts, now_ts, now_ts),
                )
                doc_ids.append(str(cur.lastrowid))
            await self._conn.commit()

        return doc_ids

    async def get_group_profiles(
        self,
        group_id: int,
        *,
        limit: int = 50,
        sort_by: Literal["creation_time", "last_updated", "last_read"] = "last_updated",
        sort_order: Literal["asc", "desc"] = "desc",
        touch_read_time: bool = True,
    ) -> GroupProfileWithDescriptionIds:
        """获取某个群的画像条目。

        Args:
            group_id: 群号。
            limit: 返回条数上限。
            sort_by: 排序字段。
            sort_order: 排序顺序。
            touch_read_time: 是否把返回的记录 last_read 更新为当前时间。

        Returns:
            GroupProfileWithDescriptionIds: 群画像（带 doc_id）。
        """

        await self.create_tables()
        assert self._conn is not None

        if limit <= 0:
            return GroupProfileWithDescriptionIds(id=int(group_id), description=[])

        if sort_by not in ("creation_time", "last_updated", "last_read"):
            raise ValueError(f"不支持的 sort_by: {sort_by}")

        order = "ASC" if sort_order == "asc" else "DESC"
        async with self._conn.execute(
            f"""
            SELECT doc_id, group_id, description, category, creation_time, last_updated, last_read
            FROM {self.table_name}
            WHERE group_id = ?
            ORDER BY {sort_by} {order}
            LIMIT ?;
            """,
            (int(group_id), int(limit)),
        ) as cursor:
            rows = await cursor.fetchall()

        items = [
            GroupProfileDescriptionWithId(doc_id=str(r["doc_id"]), text=str(r["description"]))
            for r in rows
        ]

        if touch_read_time and items:
            await self.touch_read_time_by_doc_id([x.doc_id for x in items])

        return GroupProfileWithDescriptionIds(id=int(group_id), description=items)

    async def update_by_doc_id(
        self,
        doc_id: str,
        *,
        description: str | None = None,
        category: str | None = None,
        last_updated: float | None = None,
    ) -> bool:
        """按 doc_id 更新一条记录。

        Args:
            doc_id: 记录 ID（自增主键，字符串）。
            description: 新的描述文本；不传则不更新。
            category: 新的分类；传空字符串会清空分类；不传则不更新。
            last_updated: 显式指定更新时间；不传则使用当前时间。

        Returns:
            bool: 是否更新成功（记录不存在则 False）。
        """

        await self.create_tables()
        assert self._conn is not None

        key = str(doc_id).strip()
        if not key.isdigit():
            raise ValueError("doc_id 必须为数字字符串")

        fields: list[str] = []
        params: list[Any] = []

        if description is not None:
            text = str(description).strip()
            if not text:
                raise ValueError("description 不能为空")
            fields.append("description = ?")
            params.append(text)

        if category is not None:
            cat = str(category).strip()
            if cat:
                fields.append("category = ?")
                params.append(cat)
            else:
                fields.append("category = NULL")

        ts = float(time.time()) if last_updated is None else float(last_updated)
        fields.append("last_updated = ?")
        params.append(ts)

        if not fields:
            return False

        params.append(int(key))

        async with self._write_lock:
            cur = await self._conn.execute(
                f"UPDATE {self.table_name} SET {', '.join(fields)} WHERE doc_id = ?;",
                tuple(params),
            )
            await self._conn.commit()

        return int(cur.rowcount) > 0

    async def touch_read_time_by_doc_id(
        self,
        doc_id: str | Sequence[str],
        *,
        timestamp: float | None = None,
    ) -> int:
        """批量更新 last_read。"""

        await self.create_tables()
        assert self._conn is not None

        ids = [str(doc_id)] if isinstance(doc_id, str) else [str(x) for x in doc_id]
        keys = [x.strip() for x in ids if str(x).strip()]
        keys = [x for x in keys if x.isdigit()]
        if not keys:
            return 0

        ts = float(time.time()) if timestamp is None else float(timestamp)

        placeholders = ",".join("?" for _ in keys)
        async with self._write_lock:
            cur = await self._conn.execute(
                f"UPDATE {self.table_name} SET last_read = ? WHERE doc_id IN ({placeholders});",
                (ts, *[int(x) for x in keys]),
            )
            await self._conn.commit()
        return int(cur.rowcount)

    async def delete_by_doc_id(self, doc_id: str) -> bool:
        """按 doc_id 删除一条记录。"""

        await self.create_tables()
        assert self._conn is not None

        key = str(doc_id).strip()
        if not key.isdigit():
            raise ValueError("doc_id 必须为数字字符串")

        async with self._write_lock:
            cur = await self._conn.execute(
                f"DELETE FROM {self.table_name} WHERE doc_id = ?;",
                (int(key),),
            )
            await self._conn.commit()

        return int(cur.rowcount) > 0

    async def delete_all_by_group_id(self, group_id: int) -> int:
        """删除某个群的全部画像条目。"""

        await self.create_tables()
        assert self._conn is not None

        async with self._write_lock:
            cur = await self._conn.execute(
                f"DELETE FROM {self.table_name} WHERE group_id = ?;",
                (int(group_id),),
            )
            await self._conn.commit()

        return int(cur.rowcount)
