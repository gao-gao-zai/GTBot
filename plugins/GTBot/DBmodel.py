import time
import asyncio
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from sqlalchemy import (
    select,
    String,
    INTEGER,
    BOOLEAN,
    FLOAT,
    Text,
    UniqueConstraint,
    Index,
    update,
    delete,
    desc,
    asc,
    and_,
    or_,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from pydantic import BaseModel, Field

from .constants import DEFAULT_DB_FALLBACK_PATH, DEFAULT_DB_FILENAME


# --- 配置初始化和数据库路径配置 ---
try:
    from .ConfigManager import total_config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from ConfigManager import total_config

# 初始化配置
try:
    DATA_DIR = total_config.processed_configuration.config.data_dir_path
    DB_PATH = DATA_DIR / DEFAULT_DB_FILENAME
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"
except Exception as e:
    # 如果配置加载失败，使用默认路径
    print(f"警告：配置加载失败，使用默认数据库路径: {e}")
    DB_PATH = DEFAULT_DB_FALLBACK_PATH
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


from .model import (
    GroupAllInfo, 
    GroupInfo, 
    GroupMemberInfo, 
    StrangerInfo, 
    UserProfile, 
    GroupProfile,
    GroupMessage,
    GroupMessageRecord
)

# --- 基础模型 ---
class Base(DeclarativeBase):
    pass

class CacheBaseMixin:
    """缓存模型通用字段"""

    last_update_time: Mapped[float] = mapped_column(
        FLOAT, default=0.0, index=True, nullable=False
    )
    last_access_time: Mapped[float] = mapped_column(
        FLOAT, default=0.0, index=True, nullable=False
    )


class GroupMessages(Base):
    """群聊消息模型 (ORM)"""
    __tablename__ = "group_messages"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    """数据库自增ID"""
    
    message_id: Mapped[int] = mapped_column(INTEGER, index=True, unique=True) 
    """消息ID (平台侧ID)"""
    
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, default=0)
    """群组ID"""
    
    user_id: Mapped[int] = mapped_column(INTEGER, index=True)
    """用户ID"""
    
    user_name: Mapped[str] = mapped_column(String, default="")
    """用户昵称"""
    
    content: Mapped[str] = mapped_column(String, default="")
    """消息内容"""
    
    send_time: Mapped[float] = mapped_column(FLOAT, index=True, default=0.0)
    """发送时间戳"""
    
    is_withdrawn: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    """是否已被撤回"""

    def to_pydantic(self) -> "GroupMessageRecord":
        """转换为 Pydantic 持久化记录模型。"""
        return GroupMessageRecord(
            db_id=self.id,
            message_id=self.message_id,
            group_id=self.group_id,
            user_id=self.user_id,
            user_name=self.user_name,
            content=self.content,
            send_time=self.send_time,
            is_withdrawn=self.is_withdrawn
        )


class ChatMessages(Base):
    """Unified chat message model for group and private sessions."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_id_id", "session_id", "id"),
        Index("ix_chat_messages_session_sender_id", "session_id", "sender_user_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(INTEGER, index=True, unique=True)
    session_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    chat_type: Mapped[str] = mapped_column(String, index=True, default="group")
    group_id: Mapped[int | None] = mapped_column(INTEGER, index=True, nullable=True, default=None)
    peer_user_id: Mapped[int] = mapped_column(INTEGER, index=True, default=0)
    sender_user_id: Mapped[int] = mapped_column(INTEGER, index=True, default=0)
    sender_name: Mapped[str] = mapped_column(String, default="")
    content: Mapped[str] = mapped_column(String, default="")
    serialized_segments: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    send_time: Mapped[float] = mapped_column(FLOAT, index=True, default=0.0)
    is_withdrawn: Mapped[bool] = mapped_column(BOOLEAN, default=False)





# ============================================================================
# 缓存数据模型
# ============================================================================




class CachedGroupInfo(CacheBaseMixin, Base):
    """群信息缓存"""

    __tablename__ = "cached_group_info"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "GroupInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        
        # 解析 groupAll 字段
        group_all_raw = raw.get("groupAll")
        group_all = GroupAllInfo.from_raw(group_all_raw) if group_all_raw else None
        
        return GroupInfo(
            group_id=raw.get("group_id", self.group_id),
            group_name=raw.get("group_name", ""),
            group_memo=raw.get("group_memo", ""),
            group_create_time=raw.get("group_create_time", 0),
            member_count=raw.get("member_count", 0),
            max_member_count=raw.get("max_member_count", 0),
            remark_name=raw.get("remark_name", ""),
            group_all=group_all,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )


class CachedGroupMemberInfo(CacheBaseMixin, Base):
    """群成员信息缓存"""

    __tablename__ = "cached_group_member_info"
    __table_args__ = (
        UniqueConstraint("group_id", "user_id", name="uq_group_member"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "GroupMemberInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        known_fields = {
            "group_id", "user_id", "nickname", "card", "sex", "age", "area",
            "level", "qq_level", "join_time", "last_sent_time", "title_expire_time",
            "unfriendly", "card_changeable", "is_robot", "shut_up_timestamp",
            "role", "title"
        }
        extra = {k: v for k, v in raw.items() if k not in known_fields}
        return GroupMemberInfo(
            group_id=raw.get("group_id", self.group_id),
            user_id=raw.get("user_id", self.user_id),
            nickname=raw.get("nickname", ""),
            card=raw.get("card", ""),
            sex=raw.get("sex", "unknown"),
            age=raw.get("age", 0),
            area=raw.get("area", ""),
            level=str(raw.get("level", "")),
            qq_level=raw.get("qq_level", 0),
            join_time=raw.get("join_time", 0),
            last_sent_time=raw.get("last_sent_time", 0),
            title_expire_time=raw.get("title_expire_time", 0),
            unfriendly=raw.get("unfriendly", False),
            card_changeable=raw.get("card_changeable", False),
            is_robot=raw.get("is_robot", False),
            shut_up_timestamp=raw.get("shut_up_timestamp", 0),
            role=raw.get("role", "member"),
            title=raw.get("title", ""),
            extra=extra,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )


class CachedStrangerInfo(CacheBaseMixin, Base):
    """陌生人信息缓存"""

    __tablename__ = "cached_stranger_info"

    user_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="{}")

    def to_pydantic(self) -> "StrangerInfo":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        known_fields = {
            "user_id", "nickname", "sex", "age", "qid", "level", "login_days",
            "reg_time", "long_nick", "city", "country", "birthday_year",
            "birthday_month", "birthday_day"
        }
        extra = {k: v for k, v in raw.items() if k not in known_fields}
        return StrangerInfo(
            user_id=raw.get("user_id", self.user_id),
            nickname=raw.get("nickname", ""),
            sex=raw.get("sex", "unknown"),
            age=raw.get("age", 0),
            qid=raw.get("qid", ""),
            level=raw.get("level", 0),
            login_days=raw.get("login_days", 0),
            reg_time=raw.get("reg_time", 0),
            long_nick=raw.get("long_nick", ""),
            city=raw.get("city", ""),
            country=raw.get("country", ""),
            birthday_year=raw.get("birthday_year", 0),
            birthday_month=raw.get("birthday_month", 0),
            birthday_day=raw.get("birthday_day", 0),
            extra=extra,
            last_update_time=self.last_update_time,
            last_access_time=self.last_access_time
        )

# ============================================================================
# 用户画像模型
# ============================================================================

class UserProfileModel(Base):
    """用户画像模型 (ORM)"""

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="[]")

    def to_pydantic(self) -> "UserProfile":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        return UserProfile(
            user_id=self.user_id,
            description=raw
        )

class GroupProfileModel(Base):
    """群画像模型 (ORM)"""

    __tablename__ = "group_profiles"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    data: Mapped[str] = mapped_column(Text, default="[]")

    def to_pydantic(self) -> "GroupProfile":
        """转换为Pydantic模型"""
        import json
        raw = json.loads(self.data)
        return GroupProfile(
            group_id=self.group_id,
            description=raw
        )


def _set_sqlite_pragma(dbapi_conn, connection_record):
    """设置 SQLite 连接参数，启用 WAL 模式提升写入性能。
    
    Args:
        dbapi_conn: 底层数据库连接。
        connection_record: 连接记录（SQLAlchemy 内部使用）。
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=10000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


engine = create_async_engine(ASYNC_DB_URL, echo=False)

# 注册事件监听器，在每个连接建立时设置 WAL 模式
from sqlalchemy import event
event.listen(engine.sync_engine, "connect", _set_sqlite_pragma)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def init_all_tables() -> None:
    """初始化 ORM 表结构。

    在 NoneBot 启动或首次使用缓存前调用，以确保 `data.db` 中存在
    `group_messages` 及各类缓存表。

    Raises:
        SQLAlchemyError: 当底层数据库连接或建表失败时抛出。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
