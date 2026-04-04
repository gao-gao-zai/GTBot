from __future__ import annotations

import asyncio
import time
from enum import Enum

from sqlalchemy import FLOAT, INTEGER, select
from sqlalchemy.orm import Mapped, mapped_column

from ...ConfigManager import total_config
from ...DBmodel import Base, async_session_maker, engine
from ...Logger import logger


class PermissionRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    OWNER = "owner"


_ROLE_LEVELS: dict[PermissionRole, int] = {
    PermissionRole.USER: 0,
    PermissionRole.ADMIN: 1,
    PermissionRole.OWNER: 2,
}


class PermissionError(Exception):
    """权限不足时抛出的异常。"""


class AdminUserModel(Base):
    """管理员用户表。"""

    __tablename__ = "admin_users"

    user_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    created_at: Mapped[float] = mapped_column(FLOAT, default=0.0, nullable=False)
    created_by: Mapped[int] = mapped_column(INTEGER, default=0, nullable=False)


class PermissionManager:
    """统一的权限管理器。"""

    def __init__(self) -> None:
        self._session_maker = async_session_maker
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def owner_user_ids(self) -> set[int]:
        owner_ids = total_config.processed_configuration.config.owner_user_ids
        return {int(user_id) for user_id in owner_ids}

    async def _ensure_tables(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with engine.begin() as conn:
                await conn.run_sync(AdminUserModel.metadata.create_all)
            self._initialized = True

    def is_owner(self, user_id: int) -> bool:
        return int(user_id) in self.owner_user_ids

    async def is_admin(self, user_id: int) -> bool:
        if self.is_owner(user_id):
            return True
        await self._ensure_tables()
        async with self._session_maker() as session:
            admin = await session.get(AdminUserModel, int(user_id))
            return admin is not None

    async def get_role(self, user_id: int) -> PermissionRole:
        if self.is_owner(user_id):
            return PermissionRole.OWNER
        if await self.is_admin(user_id):
            return PermissionRole.ADMIN
        return PermissionRole.USER

    async def has_role(self, user_id: int, required_role: PermissionRole | str) -> bool:
        role = await self.get_role(user_id)
        required = self._normalize_role(required_role)
        return _ROLE_LEVELS[role] >= _ROLE_LEVELS[required]

    async def require_role(self, user_id: int, required_role: PermissionRole | str) -> PermissionRole:
        required = self._normalize_role(required_role)
        role = await self.get_role(user_id)
        if _ROLE_LEVELS[role] < _ROLE_LEVELS[required]:
            raise PermissionError(self._permission_denied_message(required))
        return role

    async def list_admin_ids(self) -> list[int]:
        await self._ensure_tables()
        async with self._session_maker() as session:
            result = await session.execute(select(AdminUserModel.user_id).order_by(AdminUserModel.user_id.asc()))
            return [int(user_id) for user_id in result.scalars().all()]

    async def add_admin(self, target_user_id: int, operator_user_id: int) -> bool:
        await self._ensure_tables()
        target_user_id = int(target_user_id)
        operator_user_id = int(operator_user_id)

        if self.is_owner(target_user_id):
            raise ValueError("该用户已经是所有者，无需再设置为管理员。")

        async with self._session_maker() as session:
            existing = await session.get(AdminUserModel, target_user_id)
            if existing is not None:
                return False
            session.add(
                AdminUserModel(
                    user_id=target_user_id,
                    created_at=time.time(),
                    created_by=operator_user_id,
                )
            )
            await session.commit()
        logger.info(f"用户 {operator_user_id} 已将 {target_user_id} 提拔为管理员")
        return True

    async def remove_admin(self, target_user_id: int, operator_user_id: int) -> bool:
        await self._ensure_tables()
        target_user_id = int(target_user_id)
        operator_user_id = int(operator_user_id)

        if self.is_owner(target_user_id):
            raise ValueError("所有者由配置文件管理，不能通过管理员命令降级。")

        async with self._session_maker() as session:
            existing = await session.get(AdminUserModel, target_user_id)
            if existing is None:
                return False
            await session.delete(existing)
            await session.commit()
        logger.info(f"用户 {operator_user_id} 已将管理员 {target_user_id} 降级")
        return True

    async def describe_user_role(self, user_id: int) -> str:
        role = await self.get_role(int(user_id))
        labels = {
            PermissionRole.USER: "用户",
            PermissionRole.ADMIN: "管理员",
            PermissionRole.OWNER: "所有者",
        }
        return labels[role]

    def _normalize_role(self, role: PermissionRole | str) -> PermissionRole:
        if isinstance(role, PermissionRole):
            return role
        return PermissionRole(str(role).lower())

    def _permission_denied_message(self, required_role: PermissionRole) -> str:
        if required_role == PermissionRole.OWNER:
            return "只有所有者可以执行这个操作"
        if required_role == PermissionRole.ADMIN:
            return "只有管理员或所有者可以执行这个操作"
        return "你没有执行这个操作的权限"


_permission_manager = PermissionManager()


def get_permission_manager() -> PermissionManager:
    return _permission_manager
