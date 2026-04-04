from __future__ import annotations

import asyncio
import time
from enum import Enum

from sqlalchemy import FLOAT, INTEGER, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column

from ...DBmodel import Base, async_session_maker, engine
from ...Logger import logger


class ChatAccessScope(str, Enum):
    """聊天准入控制的会话范围枚举。"""

    GROUP = "group"
    PRIVATE = "private"


class ChatAccessMode(str, Enum):
    """聊天准入控制的模式枚举。"""

    OFF = "off"
    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"


class ChatAccessListType(str, Enum):
    """聊天准入控制的名单类型枚举。"""

    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"


class ChatAccessModeModel(Base):
    """会话准入模式表。

    Attributes:
        scope: 控制范围，`group` 或 `private`。
        mode: 当前范围的准入模式。
        updated_at: 最近一次修改时间戳。
        updated_by: 最近一次修改该模式的操作人。
    """

    __tablename__ = "chat_access_modes"

    scope: Mapped[str] = mapped_column(String, primary_key=True)
    mode: Mapped[str] = mapped_column(String, nullable=False, default=ChatAccessMode.OFF.value)
    updated_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    updated_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class ChatAccessEntryModel(Base):
    """会话准入名单项表。

    Attributes:
        id: 自增主键。
        scope: 控制范围，`group` 或 `private`。
        target_id: 群号或私聊用户号。
        list_type: 名单类型，`blacklist` 或 `whitelist`。
        created_at: 创建时间戳。
        created_by: 创建该名单项的操作人。
    """

    __tablename__ = "chat_access_entries"
    __table_args__ = (
        UniqueConstraint("scope", "target_id", "list_type", name="uq_chat_access_entry"),
    )

    id: Mapped[int] = mapped_column(INTEGER, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String, index=True, nullable=False)
    target_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    list_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    created_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class ChatAccessManager:
    """统一管理群聊和私聊准入策略。

    该管理器负责两类信息：
    1. 每个会话范围的准入模式（关闭、黑名单、白名单）。
    2. 对应范围下的黑白名单目标 ID。

    所有配置都持久化在 SQLite 中，并通过宿主已有的 SQLAlchemy 异步会话访问。
    """

    def __init__(self) -> None:
        """初始化会话准入管理器。"""
        self._session_maker = async_session_maker
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_tables(self) -> None:
        """确保会话准入相关数据表已经创建。"""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with engine.begin() as conn:
                await conn.run_sync(ChatAccessModeModel.metadata.create_all)
            self._initialized = True

    def _normalize_scope(self, scope: ChatAccessScope | str) -> ChatAccessScope:
        """标准化会话范围枚举值。

        Args:
            scope: 原始范围值。

        Returns:
            ChatAccessScope: 归一化后的范围枚举。
        """
        if isinstance(scope, ChatAccessScope):
            return scope
        return ChatAccessScope(str(scope).strip().lower())

    def _normalize_mode(self, mode: ChatAccessMode | str) -> ChatAccessMode:
        """标准化准入模式枚举值。

        Args:
            mode: 原始模式值。

        Returns:
            ChatAccessMode: 归一化后的模式枚举。
        """
        if isinstance(mode, ChatAccessMode):
            return mode
        return ChatAccessMode(str(mode).strip().lower())

    def _normalize_list_type(self, list_type: ChatAccessListType | str) -> ChatAccessListType:
        """标准化名单类型枚举值。

        Args:
            list_type: 原始名单类型值。

        Returns:
            ChatAccessListType: 归一化后的名单类型枚举。
        """
        if isinstance(list_type, ChatAccessListType):
            return list_type
        return ChatAccessListType(str(list_type).strip().lower())

    async def get_mode(self, scope: ChatAccessScope | str) -> ChatAccessMode:
        """获取某个会话范围当前的准入模式。

        Args:
            scope: 目标范围，支持群聊或私聊。

        Returns:
            ChatAccessMode: 当前范围的准入模式；若尚未配置则返回 `OFF`。
        """
        await self._ensure_tables()
        normalized_scope = self._normalize_scope(scope)
        async with self._session_maker() as session:
            row = await session.get(ChatAccessModeModel, normalized_scope.value)
            if row is None:
                return ChatAccessMode.OFF
            return self._normalize_mode(row.mode)

    async def set_mode(
        self,
        scope: ChatAccessScope | str,
        mode: ChatAccessMode | str,
        operator_user_id: int,
    ) -> ChatAccessMode:
        """设置某个会话范围的准入模式。

        Args:
            scope: 目标范围，支持群聊或私聊。
            mode: 目标模式，支持关闭、黑名单、白名单。
            operator_user_id: 操作该配置的管理员 QQ 号。

        Returns:
            ChatAccessMode: 设置后的模式。
        """
        await self._ensure_tables()
        normalized_scope = self._normalize_scope(scope)
        normalized_mode = self._normalize_mode(mode)
        now = time.time()

        async with self._session_maker() as session:
            row = await session.get(ChatAccessModeModel, normalized_scope.value)
            if row is None:
                row = ChatAccessModeModel(
                    scope=normalized_scope.value,
                    mode=normalized_mode.value,
                    updated_at=now,
                    updated_by=int(operator_user_id),
                )
                session.add(row)
            else:
                row.mode = normalized_mode.value
                row.updated_at = now
                row.updated_by = int(operator_user_id)
            await session.commit()

        logger.info(
            "用户 %s 已将 %s 会话准入模式设置为 %s",
            int(operator_user_id),
            normalized_scope.value,
            normalized_mode.value,
        )
        return normalized_mode

    async def list_entries(
        self,
        scope: ChatAccessScope | str,
        list_type: ChatAccessListType | str,
    ) -> list[int]:
        """列出某个范围下指定名单中的所有目标 ID。

        Args:
            scope: 目标范围，支持群聊或私聊。
            list_type: 名单类型，支持黑名单或白名单。

        Returns:
            list[int]: 按升序返回的目标 ID 列表。
        """
        await self._ensure_tables()
        normalized_scope = self._normalize_scope(scope)
        normalized_list_type = self._normalize_list_type(list_type)

        async with self._session_maker() as session:
            result = await session.execute(
                select(ChatAccessEntryModel.target_id)
                .where(
                    ChatAccessEntryModel.scope == normalized_scope.value,
                    ChatAccessEntryModel.list_type == normalized_list_type.value,
                )
                .order_by(ChatAccessEntryModel.target_id.asc())
            )
            return [int(target_id) for target_id in result.scalars().all()]

    async def add_entry(
        self,
        scope: ChatAccessScope | str,
        list_type: ChatAccessListType | str,
        target_id: int,
        operator_user_id: int,
    ) -> bool:
        """向指定范围的黑白名单中添加目标 ID。

        Args:
            scope: 目标范围，支持群聊或私聊。
            list_type: 名单类型，支持黑名单或白名单。
            target_id: 要加入名单的群号或用户号。
            operator_user_id: 执行操作的管理员 QQ 号。

        Returns:
            bool: 新增成功返回 `True`；若目标已存在则返回 `False`。
        """
        await self._ensure_tables()
        normalized_scope = self._normalize_scope(scope)
        normalized_list_type = self._normalize_list_type(list_type)
        normalized_target_id = int(target_id)

        async with self._session_maker() as session:
            result = await session.execute(
                select(ChatAccessEntryModel)
                .where(
                    ChatAccessEntryModel.scope == normalized_scope.value,
                    ChatAccessEntryModel.list_type == normalized_list_type.value,
                    ChatAccessEntryModel.target_id == normalized_target_id,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return False

            session.add(
                ChatAccessEntryModel(
                    scope=normalized_scope.value,
                    list_type=normalized_list_type.value,
                    target_id=normalized_target_id,
                    created_at=time.time(),
                    created_by=int(operator_user_id),
                )
            )
            await session.commit()

        logger.info(
            "用户 %s 已将 %s 加入 %s 的 %s",
            int(operator_user_id),
            normalized_target_id,
            normalized_scope.value,
            normalized_list_type.value,
        )
        return True

    async def remove_entry(
        self,
        scope: ChatAccessScope | str,
        list_type: ChatAccessListType | str,
        target_id: int,
        operator_user_id: int,
    ) -> bool:
        """从指定范围的黑白名单中移除目标 ID。

        Args:
            scope: 目标范围，支持群聊或私聊。
            list_type: 名单类型，支持黑名单或白名单。
            target_id: 要移除的群号或用户号。
            operator_user_id: 执行操作的管理员 QQ 号。

        Returns:
            bool: 移除成功返回 `True`；若目标不存在则返回 `False`。
        """
        await self._ensure_tables()
        normalized_scope = self._normalize_scope(scope)
        normalized_list_type = self._normalize_list_type(list_type)
        normalized_target_id = int(target_id)

        async with self._session_maker() as session:
            result = await session.execute(
                select(ChatAccessEntryModel)
                .where(
                    ChatAccessEntryModel.scope == normalized_scope.value,
                    ChatAccessEntryModel.list_type == normalized_list_type.value,
                    ChatAccessEntryModel.target_id == normalized_target_id,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return False

            await session.delete(existing)
            await session.commit()

        logger.info(
            "用户 %s 已将 %s 从 %s 的 %s 移除",
            int(operator_user_id),
            normalized_target_id,
            normalized_scope.value,
            normalized_list_type.value,
        )
        return True

    async def is_allowed(self, scope: ChatAccessScope | str, target_id: int) -> bool:
        """判断某个会话目标是否允许进入聊天主链路。

        Args:
            scope: 目标范围，支持群聊或私聊。
            target_id: 群号或私聊用户号。

        Returns:
            bool: `True` 表示允许，`False` 表示拒绝。
        """
        normalized_scope = self._normalize_scope(scope)
        normalized_target_id = int(target_id)
        mode = await self.get_mode(normalized_scope)

        if mode == ChatAccessMode.OFF:
            return True

        if mode == ChatAccessMode.BLACKLIST:
            blacklist = await self.list_entries(normalized_scope, ChatAccessListType.BLACKLIST)
            return normalized_target_id not in set(blacklist)

        whitelist = await self.list_entries(normalized_scope, ChatAccessListType.WHITELIST)
        return normalized_target_id in set(whitelist)

    async def describe_scope(self, scope: ChatAccessScope | str) -> str:
        """生成某个会话范围当前准入配置的文本描述。

        Args:
            scope: 目标范围，支持群聊或私聊。

        Returns:
            str: 包含模式、黑名单和白名单的多行描述文本。
        """
        normalized_scope = self._normalize_scope(scope)
        mode = await self.get_mode(normalized_scope)
        whitelist = await self.list_entries(normalized_scope, ChatAccessListType.WHITELIST)
        blacklist = await self.list_entries(normalized_scope, ChatAccessListType.BLACKLIST)

        mode_labels = {
            ChatAccessMode.OFF: "关闭",
            ChatAccessMode.BLACKLIST: "黑名单模式",
            ChatAccessMode.WHITELIST: "白名单模式",
        }
        scope_labels = {
            ChatAccessScope.GROUP: "群聊",
            ChatAccessScope.PRIVATE: "私聊",
        }
        lines = [
            f"{scope_labels[normalized_scope]}权限模式: {mode_labels[mode]}",
            f"白名单: {', '.join(str(x) for x in whitelist) if whitelist else '(空)'}",
            f"黑名单: {', '.join(str(x) for x in blacklist) if blacklist else '(空)'}",
        ]
        return "\n".join(lines)


_chat_access_manager = ChatAccessManager()


def get_chat_access_manager() -> ChatAccessManager:
    """返回共享的会话准入管理器实例。"""
    return _chat_access_manager
