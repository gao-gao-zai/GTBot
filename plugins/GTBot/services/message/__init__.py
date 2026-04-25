import asyncio
import time
from typing import Any, Optional, Union

from sqlalchemy import and_, asc, delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from ...DBmodel import Base, ChatMessages, GroupMessages
from ...model import GroupMessage, GroupMessageRecord


CHAT_MESSAGES_SERIALIZED_SEGMENTS_COLUMN = "serialized_segments"
CHAT_MESSAGES_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id_id "
    "ON chat_messages (session_id, id)",
    "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_sender_id "
    "ON chat_messages (session_id, sender_user_id, id)",
)


def _group_session_id(group_id: int) -> str:
    return f"group:{int(group_id)}"


def _private_session_id(user_id: int) -> str:
    return f"private:{int(user_id)}"


class GroupMessageManager:
    """Message manager backed by the unified chat_messages table."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.session_maker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._ensure_chat_messages_schema()
        await self._ensure_chat_messages_indexes()
        await self._migrate_legacy_group_messages()

    async def _ensure_chat_messages_schema(self) -> None:
        """确保统一消息表包含原样消息段列。

        该方法只负责轻量 schema 自检与补列，不会回填历史数据。当前实现
        仅针对 SQLite 数据库，使用 `PRAGMA table_info` 检查列是否存在。
        """
        async with self.engine.begin() as conn:
            rows = (await conn.exec_driver_sql("PRAGMA table_info(chat_messages)")).fetchall()
            column_names = {str(row[1]) for row in rows if len(row) > 1}
            if CHAT_MESSAGES_SERIALIZED_SEGMENTS_COLUMN in column_names:
                return

            await conn.exec_driver_sql(
                "ALTER TABLE chat_messages "
                f"ADD COLUMN {CHAT_MESSAGES_SERIALIZED_SEGMENTS_COLUMN} TEXT"
            )

    async def _ensure_chat_messages_indexes(self) -> None:
        """确保统一消息表包含关键复合索引。

        这些索引用于覆盖按会话拉取最近消息、按会话+发送者过滤，以及按会话
        获取附近消息的核心查询路径。该方法使用幂等 SQL，适用于现有数据库
        的启动补齐。
        """
        async with self.engine.begin() as conn:
            for ddl in CHAT_MESSAGES_INDEX_DDL:
                await conn.exec_driver_sql(ddl)

    async def _migrate_legacy_group_messages(self) -> None:
        async with self.session_maker() as session:
            chat_count = await session.scalar(select(func.count()).select_from(ChatMessages))
            if int(chat_count or 0) > 0:
                return

            legacy_rows = (
                await session.execute(select(GroupMessages).order_by(asc(GroupMessages.id)))
            ).scalars().all()
            if not legacy_rows:
                return

            session.add_all(
                [
                    ChatMessages(
                        message_id=row.message_id,
                        session_id=_group_session_id(row.group_id),
                        chat_type="group",
                        group_id=row.group_id,
                        peer_user_id=row.group_id,
                        sender_user_id=row.user_id,
                        sender_name=row.user_name,
                        content=row.content,
                        send_time=row.send_time,
                        is_withdrawn=row.is_withdrawn,
                    )
                    for row in legacy_rows
                ]
            )
            await session.commit()

    @staticmethod
    def _derive_chat_fields(
        *,
        group_id: int | None,
        session_id: str | None,
        peer_user_id: int | None,
        sender_user_id: int,
    ) -> tuple[str, str, int | None, int]:
        gid = int(group_id) if isinstance(group_id, int) and group_id > 0 else None
        if session_id:
            sid = str(session_id).strip()
        elif gid is not None:
            sid = _group_session_id(gid)
        else:
            sid = _private_session_id(int(peer_user_id or sender_user_id))

        chat_type = "group" if gid is not None or sid.startswith("group:") else "private"
        if chat_type == "group":
            gid = gid if gid is not None else int(peer_user_id or 0)
            peer = int(peer_user_id or gid or 0)
        else:
            gid = None
            peer = int(peer_user_id or sender_user_id)
        return sid, chat_type, gid, peer

    @staticmethod
    def _row_to_record(row: ChatMessages) -> GroupMessageRecord:
        return GroupMessageRecord(
            db_id=int(row.id),
            message_id=int(row.message_id),
            group_id=int(row.group_id or 0),
            user_id=int(row.sender_user_id),
            user_name=str(row.sender_name or ""),
            content=str(row.content or ""),
            serialized_segments=row.serialized_segments,
            send_time=float(row.send_time or 0.0),
            is_withdrawn=bool(row.is_withdrawn),
        )

    async def add_chat_message(
        self,
        *,
        message_id: int,
        sender_user_id: int,
        sender_name: str = "",
        content: str = "",
        serialized_segments: str | None = None,
        send_time: float | None = None,
        is_withdrawn: bool = False,
        group_id: int | None = None,
        session_id: str | None = None,
        peer_user_id: int | None = None,
    ) -> GroupMessageRecord:
        sid, chat_type, gid, peer = self._derive_chat_fields(
            group_id=group_id,
            session_id=session_id,
            peer_user_id=peer_user_id,
            sender_user_id=sender_user_id,
        )
        orm_msg = ChatMessages(
            message_id=int(message_id),
            session_id=sid,
            chat_type=chat_type,
            group_id=gid,
            peer_user_id=int(peer),
            sender_user_id=int(sender_user_id),
            sender_name=str(sender_name or ""),
            content=str(content or ""),
            serialized_segments=serialized_segments,
            send_time=float(send_time if send_time is not None else time.time()),
            is_withdrawn=bool(is_withdrawn),
        )

        async with self.session_maker() as session:
            async with session.begin():
                session.add(orm_msg)
                await session.flush()
            await session.refresh(orm_msg)
            return self._row_to_record(orm_msg)

    async def add_message(
        self,
        msg: Optional[Union[GroupMessage, GroupMessageRecord]] = None,
        **kwargs: Any,
    ) -> GroupMessageRecord:
        if msg is not None:
            data = msg.model_dump()
            data.pop("db_id", None)
            return await self.add_chat_message(
                message_id=int(data["message_id"]),
                group_id=int(data.get("group_id", 0) or 0) or None,
                sender_user_id=int(data["user_id"]),
                sender_name=str(data.get("user_name", "")),
                content=str(data.get("content", "")),
                serialized_segments=data.get("serialized_segments"),
                send_time=float(data.get("send_time", time.time()) or time.time()),
                is_withdrawn=bool(data.get("is_withdrawn", False)),
                peer_user_id=int(data.get("group_id", 0) or data["user_id"]),
            )

        required = {"message_id", "sender_user_id"}
        if not required.issubset(kwargs):
            missing = required - set(kwargs)
            raise ValueError(f"missing required fields: {sorted(missing)}")
        return await self.add_chat_message(**kwargs)

    async def get_message_by_msg_id(self, message_id: int) -> GroupMessageRecord:
        async with self.session_maker() as session:
            row = (
                await session.execute(
                    select(ChatMessages).where(ChatMessages.message_id == int(message_id))
                )
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"message {message_id} not found")
            return self._row_to_record(row)

    async def get_message_by_db_id(self, db_id: int) -> GroupMessageRecord:
        async with self.session_maker() as session:
            row = (
                await session.execute(select(ChatMessages).where(ChatMessages.id == int(db_id)))
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"db row {db_id} not found")
            return self._row_to_record(row)

    async def get_recent_messages(
        self,
        limit: int = 10,
        group_id: Optional[int] = None,
        user_id: Optional[int] = None,
        asc_order: bool = False,
        session_id: str | None = None,
    ) -> list[GroupMessageRecord]:
        async with self.session_maker() as session:
            stmt = select(ChatMessages)
            conditions = []

            if session_id:
                conditions.append(ChatMessages.session_id == str(session_id))
            elif group_id is not None:
                conditions.append(ChatMessages.session_id == _group_session_id(int(group_id)))
            elif user_id is not None:
                conditions.append(ChatMessages.session_id == _private_session_id(int(user_id)))

            if group_id is not None and (session_id or user_id is not None):
                conditions.append(ChatMessages.group_id == int(group_id))
            if user_id is not None and group_id is not None:
                conditions.append(ChatMessages.sender_user_id == int(user_id))

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(asc(ChatMessages.id) if asc_order else desc(ChatMessages.id)).limit(int(limit))
            rows = (await session.execute(stmt)).scalars().all()
            return [self._row_to_record(row) for row in rows]

    async def get_nearby_messages(
        self,
        target_msg: Optional[GroupMessageRecord] = None,
        db_id: Optional[int] = None,
        message_id: Optional[int] = None,
        group_id: Optional[int] = None,
        user_id: Optional[int] = None,
        before: int = 10,
        after: int = 0,
        include_target: bool = True,
        session_id: str | None = None,
    ) -> list[GroupMessage]:
        async with self.session_maker() as session:
            target_row: ChatMessages | None = None
            if target_msg is not None:
                target_row = (
                    await session.execute(select(ChatMessages).where(ChatMessages.id == int(target_msg.db_id)))
                ).scalar_one_or_none()
            elif db_id is not None:
                target_row = (
                    await session.execute(select(ChatMessages).where(ChatMessages.id == int(db_id)))
                ).scalar_one_or_none()
            elif message_id is not None:
                target_row = (
                    await session.execute(
                        select(ChatMessages).where(ChatMessages.message_id == int(message_id))
                    )
                ).scalar_one_or_none()

            if target_row is None:
                raise ValueError("target message not found")

            sid = (
                str(session_id).strip()
                if session_id
                else _group_session_id(int(group_id))
                if group_id is not None
                else target_row.session_id
            )

            conditions = [ChatMessages.session_id == sid]
            if user_id is not None and group_id is not None:
                conditions.append(ChatMessages.sender_user_id == int(user_id))

            prev_rows: list[ChatMessages] = []
            if before > 0:
                prev_rows = list(
                    reversed(
                        (
                            await session.execute(
                                select(ChatMessages)
                                .where(and_(ChatMessages.id < target_row.id, *conditions))
                                .order_by(desc(ChatMessages.id))
                                .limit(int(before))
                            )
                        ).scalars().all()
                    )
                )

            next_rows: list[ChatMessages] = []
            if after > 0:
                next_rows = list(
                    (
                        await session.execute(
                            select(ChatMessages)
                            .where(and_(ChatMessages.id > target_row.id, *conditions))
                            .order_by(asc(ChatMessages.id))
                            .limit(int(after))
                        )
                    ).scalars().all()
                )

            rows: list[ChatMessages] = prev_rows + ([target_row] if include_target else []) + list(next_rows)
            return [self._row_to_record(row).to_domain() for row in rows]

    async def update_message(
        self,
        identify_by_msg_id: Optional[int] = None,
        identify_by_db_id: Optional[int] = None,
        **update_fields: Any,
    ) -> None:
        if not update_fields:
            return

        normalized: dict[str, Any] = {}
        field_map = {
            "content": "content",
            "serialized_segments": "serialized_segments",
            "is_withdrawn": "is_withdrawn",
            "sender_name": "sender_name",
            "send_time": "send_time",
        }
        for key, value in update_fields.items():
            mapped = field_map.get(key)
            if mapped is not None:
                normalized[mapped] = value

        if not normalized:
            return

        stmt = update(ChatMessages)
        if identify_by_db_id is not None:
            stmt = stmt.where(ChatMessages.id == int(identify_by_db_id))
        elif identify_by_msg_id is not None:
            stmt = stmt.where(ChatMessages.message_id == int(identify_by_msg_id))
        else:
            raise ValueError("message locator is required")

        async with self.session_maker() as session:
            async with session.begin():
                result = await session.execute(stmt.values(**normalized))
                if result.rowcount == 0:  # type: ignore[attr-defined]
                    raise ValueError("message not found")

    async def delete_message(
        self,
        message_id: Optional[int] = None,
        db_id: Optional[int] = None,
    ) -> None:
        stmt = delete(ChatMessages)
        if db_id is not None:
            stmt = stmt.where(ChatMessages.id == int(db_id))
        elif message_id is not None:
            stmt = stmt.where(ChatMessages.message_id == int(message_id))
        else:
            raise ValueError("message locator is required")

        async with self.session_maker() as session:
            async with session.begin():
                result = await session.execute(stmt)
                if result.rowcount == 0:  # type: ignore[attr-defined]
                    raise ValueError("message not found")

    async def mark_message_withdrawn(self, message_id: int) -> None:
        await self.update_message(identify_by_msg_id=message_id, is_withdrawn=True)


NONEBOT_ENV: bool = False
_driver = None
try:
    from nonebot import get_driver

    NONEBOT_ENV = True
    try:
        _driver = get_driver()
    except Exception:  # noqa: BLE001
        _driver = None
except ImportError:
    NONEBOT_ENV = False

if NONEBOT_ENV:
    from ...ConfigManager import total_config
    from ...constants import DEFAULT_DB_FILENAME

    _init_lock = asyncio.Lock()
    _manager_ready = asyncio.Event()

    if _driver is not None:
        @_driver.on_startup
        async def _init_message_manager() -> None:
            global message_manager
            async with _init_lock:
                data_dir = total_config.processed_configuration.config.data_dir_path
                db_path = data_dir / DEFAULT_DB_FILENAME
                async_db_url = f"sqlite+aiosqlite:///{db_path}"
                engine = create_async_engine(async_db_url, echo=False)
                message_manager = GroupMessageManager(engine)
                await message_manager.create_tables()
                _manager_ready.set()

    async def get_message_manager() -> GroupMessageManager:
        await _manager_ready.wait()
        return message_manager
