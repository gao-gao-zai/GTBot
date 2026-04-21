from __future__ import annotations

import asyncio
import time
from enum import Enum
from pathlib import Path

import aiosqlite
from nonebot import logger

from .config import get_permission_config, resolve_database_path


class PermissionRole(str, Enum):
    """定义插件统一使用的三层权限模型。

    `USER` 表示普通用户，`ADMIN` 表示运行期可增删的管理员，
    `OWNER` 表示来自静态配置、拥有最高权限且不能被命令降级的用户。
    """

    USER = "user"
    ADMIN = "admin"
    OWNER = "owner"


_ROLE_LEVELS: dict[PermissionRole, int] = {
    PermissionRole.USER: 0,
    PermissionRole.ADMIN: 1,
    PermissionRole.OWNER: 2,
}


class PermissionError(Exception):
    """表示用户不满足当前操作所需的权限要求。"""


class PermissionManager:
    """管理 owner/admin/user 权限判定与管理员持久化。

    该类使用插件自己的 sqlite 文件维护管理员列表，不依赖 GTBot 的数据库基座。
    所有者列表始终来自插件配置，因此不会被命令动态修改；管理员列表则可在运行时增删。
    """

    def __init__(self, database_path: Path | None = None) -> None:
        """初始化权限管理器。

        Args:
            database_path: 可选的 sqlite 数据库路径。未传入时使用插件配置解析出的默认路径。
        """

        self._database_path = Path(database_path) if database_path is not None else resolve_database_path()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def owner_user_ids(self) -> set[int]:
        """返回配置中声明的所有者集合。

        Returns:
            当前配置下的所有者 QQ 号集合。
        """

        return {int(user_id) for user_id in get_permission_config().owner_user_ids}

    async def _ensure_tables(self) -> None:
        """确保管理员表已存在。

        该方法带有惰性初始化和锁保护，避免并发命令在首次访问时重复建表。
        表结构只包含管理员 QQ 号以及审计字段，不承载 owner 数据。
        """

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self._database_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS admin_users (
                        user_id INTEGER PRIMARY KEY,
                        created_at REAL NOT NULL DEFAULT 0,
                        created_by INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                await db.commit()
            self._initialized = True

    def is_owner(self, user_id: int) -> bool:
        """判断用户是否为配置中的所有者。

        Args:
            user_id: 需要判断的 QQ 号。

        Returns:
            当用户位于配置的 owner 列表中时返回 `True`。
        """

        return int(user_id) in self.owner_user_ids

    async def is_admin(self, user_id: int) -> bool:
        """判断用户是否具备管理员身份。

        owner 会被视作天然管理员，因此该方法对 owner 也返回 `True`。

        Args:
            user_id: 需要判断的 QQ 号。

        Returns:
            当用户为 owner 或存在于管理员表中时返回 `True`。
        """

        if self.is_owner(user_id):
            return True
        await self._ensure_tables()
        async with aiosqlite.connect(self._database_path) as db:
            async with db.execute(
                "SELECT 1 FROM admin_users WHERE user_id = ? LIMIT 1",
                (int(user_id),),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def get_role(self, user_id: int) -> PermissionRole:
        """解析用户当前的有效权限等级。

        Args:
            user_id: 需要解析的 QQ 号。

        Returns:
            用户当前的权限等级。
        """

        if self.is_owner(user_id):
            return PermissionRole.OWNER
        if await self.is_admin(user_id):
            return PermissionRole.ADMIN
        return PermissionRole.USER

    async def has_role(self, user_id: int, required_role: PermissionRole | str) -> bool:
        """判断用户是否至少具备目标权限等级。

        Args:
            user_id: 需要校验的 QQ 号。
            required_role: 目标权限等级，支持枚举或字符串。

        Returns:
            当用户权限层级不低于目标权限时返回 `True`。
        """

        role = await self.get_role(int(user_id))
        required = self._normalize_role(required_role)
        return _ROLE_LEVELS[role] >= _ROLE_LEVELS[required]

    async def require_role(self, user_id: int, required_role: PermissionRole | str) -> PermissionRole:
        """强制要求用户至少具备目标权限等级。

        Args:
            user_id: 需要校验的 QQ 号。
            required_role: 目标权限等级，支持枚举或字符串。

        Returns:
            用户当前实际拥有的权限等级。

        Raises:
            PermissionError: 当用户权限不足时抛出。
        """

        required = self._normalize_role(required_role)
        role = await self.get_role(int(user_id))
        if _ROLE_LEVELS[role] < _ROLE_LEVELS[required]:
            raise PermissionError(self._permission_denied_message(required))
        return role

    async def list_admin_ids(self) -> list[int]:
        """返回当前动态管理员列表。

        owner 由静态配置管理，因此此处仅返回 sqlite 中保存的管理员。

        Returns:
            按 QQ 号升序排列的管理员列表。
        """

        await self._ensure_tables()
        async with aiosqlite.connect(self._database_path) as db:
            async with db.execute("SELECT user_id FROM admin_users ORDER BY user_id ASC") as cursor:
                rows = await cursor.fetchall()
        return [int(row[0]) for row in rows]

    async def add_admin(self, target_user_id: int, operator_user_id: int) -> bool:
        """将目标用户加入管理员列表。

        owner 不允许重复设置为管理员，因为其权限已由配置固定提供。

        Args:
            target_user_id: 目标用户 QQ 号。
            operator_user_id: 执行操作的用户 QQ 号。

        Returns:
            当本次实际新增管理员时返回 `True`；若目标已是管理员则返回 `False`。

        Raises:
            ValueError: 当目标用户已经是 owner 时抛出。
        """

        await self._ensure_tables()
        target_user_id = int(target_user_id)
        operator_user_id = int(operator_user_id)
        if self.is_owner(target_user_id):
            raise ValueError("该用户已经是所有者，无需再设置为管理员。")

        async with aiosqlite.connect(self._database_path) as db:
            async with db.execute(
                "SELECT 1 FROM admin_users WHERE user_id = ? LIMIT 1",
                (target_user_id,),
            ) as cursor:
                if await cursor.fetchone() is not None:
                    return False
            await db.execute(
                "INSERT INTO admin_users (user_id, created_at, created_by) VALUES (?, ?, ?)",
                (target_user_id, time.time(), operator_user_id),
            )
            await db.commit()
        logger.info(f"用户 {operator_user_id} 已将 {target_user_id} 提拔为管理员")
        return True

    async def remove_admin(self, target_user_id: int, operator_user_id: int) -> bool:
        """将目标用户从管理员列表中移除。

        owner 不能通过命令降级，防止运行时状态覆盖静态配置。

        Args:
            target_user_id: 目标用户 QQ 号。
            operator_user_id: 执行操作的用户 QQ 号。

        Returns:
            当本次实际移除了管理员时返回 `True`；若目标原本就不是管理员则返回 `False`。

        Raises:
            ValueError: 当目标用户是 owner 时抛出。
        """

        await self._ensure_tables()
        target_user_id = int(target_user_id)
        operator_user_id = int(operator_user_id)
        if self.is_owner(target_user_id):
            raise ValueError("所有者由配置文件管理，不能通过管理员命令降级。")

        async with aiosqlite.connect(self._database_path) as db:
            cursor = await db.execute("DELETE FROM admin_users WHERE user_id = ?", (target_user_id,))
            await db.commit()
            removed = cursor.rowcount > 0
        if removed:
            logger.info(f"用户 {operator_user_id} 已将管理员 {target_user_id} 降级")
        return removed

    async def describe_user_role(self, user_id: int) -> str:
        """将用户权限格式化为中文标签。

        Args:
            user_id: 需要描述的 QQ 号。

        Returns:
            中文权限名称，便于直接发送给用户。
        """

        labels = {
            PermissionRole.USER: "普通用户",
            PermissionRole.ADMIN: "管理员",
            PermissionRole.OWNER: "所有者",
        }
        return labels[await self.get_role(int(user_id))]

    def _normalize_role(self, role: PermissionRole | str) -> PermissionRole:
        """将输入统一转换为权限枚举。

        Args:
            role: 枚举或字符串形式的权限等级。

        Returns:
            规范化后的权限枚举对象。

        Raises:
            ValueError: 当传入的字符串无法映射到已知权限等级时抛出。
        """

        if isinstance(role, PermissionRole):
            return role
        return PermissionRole(str(role).strip().lower())

    def _permission_denied_message(self, required_role: PermissionRole) -> str:
        """生成统一的权限不足提示文案。

        Args:
            required_role: 当前操作要求的最低权限。

        Returns:
            面向终端用户的中文提示文案。
        """

        if required_role == PermissionRole.OWNER:
            return "只有所有者可以执行这个操作。"
        if required_role == PermissionRole.ADMIN:
            return "只有管理员或所有者可以执行这个操作。"
        return "你没有执行这个操作的权限。"


_permission_manager = PermissionManager()


def get_permission_manager() -> PermissionManager:
    """返回全局共享的权限管理器实例。

    Returns:
        当前进程内复用的权限管理器对象。
    """

    return _permission_manager
