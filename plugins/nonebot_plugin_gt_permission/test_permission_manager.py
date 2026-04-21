from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_plugins.nonebot_plugin_gt_permission import (
    GTPermissionConfig,
    PermissionError,
    PermissionManager,
    PermissionRole,
)


class TestPermissionManager(unittest.IsolatedAsyncioTestCase):
    """覆盖权限插件核心角色与持久化行为。"""

    def setUp(self) -> None:
        """为每个测试创建独立的 sqlite 路径。"""

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.database_path = Path(self._tmpdir.name) / "admin_users.db"

    def _build_manager(self, owner_user_ids: list[int] | None = None) -> PermissionManager:
        """构造带测试配置补丁的权限管理器。

        Args:
            owner_user_ids: 当前测试场景使用的 owner 列表。

        Returns:
            指向临时 sqlite 文件的权限管理器实例。
        """

        config = GTPermissionConfig(
            owner_user_ids=owner_user_ids or [],
            database_path=str(self.database_path),
            enable_commands=True,
        )
        patcher = patch("plugins.nonebot_plugin_gt_permission.manager.get_permission_config", return_value=config)
        patcher.start()
        self.addCleanup(patcher.stop)
        return PermissionManager(database_path=self.database_path)

    async def test_owner_admin_user_roles_are_resolved_correctly(self) -> None:
        """owner、admin 与普通用户应被正确区分。"""

        manager = self._build_manager(owner_user_ids=[10001])
        await manager.add_admin(10002, 10001)

        self.assertEqual(await manager.get_role(10001), PermissionRole.OWNER)
        self.assertEqual(await manager.get_role(10002), PermissionRole.ADMIN)
        self.assertEqual(await manager.get_role(10003), PermissionRole.USER)

    async def test_has_role_and_require_role_follow_role_hierarchy(self) -> None:
        """权限层级判断应遵循 USER < ADMIN < OWNER。"""

        manager = self._build_manager(owner_user_ids=[10001])
        await manager.add_admin(10002, 10001)

        self.assertTrue(await manager.has_role(10001, PermissionRole.ADMIN))
        self.assertTrue(await manager.has_role(10002, PermissionRole.USER))
        self.assertFalse(await manager.has_role(10003, PermissionRole.ADMIN))
        with self.assertRaises(PermissionError):
            await manager.require_role(10003, PermissionRole.ADMIN)

    async def test_add_and_remove_admin_are_idempotent(self) -> None:
        """管理员增删应正确处理重复调用。"""

        manager = self._build_manager(owner_user_ids=[10001])

        self.assertTrue(await manager.add_admin(10002, 10001))
        self.assertFalse(await manager.add_admin(10002, 10001))
        self.assertEqual(await manager.list_admin_ids(), [10002])
        self.assertTrue(await manager.remove_admin(10002, 10001))
        self.assertFalse(await manager.remove_admin(10002, 10001))

    async def test_owner_cannot_be_promoted_or_demoted(self) -> None:
        """owner 不能通过管理员命令被提拔或降级。"""

        manager = self._build_manager(owner_user_ids=[10001])

        with self.assertRaises(ValueError):
            await manager.add_admin(10001, 10001)
        with self.assertRaises(ValueError):
            await manager.remove_admin(10001, 10001)

    async def test_table_initialization_is_idempotent(self) -> None:
        """重复初始化表结构不应报错，也不应破坏已有数据。"""

        manager = self._build_manager(owner_user_ids=[10001])
        await manager._ensure_tables()
        await manager._ensure_tables()
        await manager.add_admin(10002, 10001)
        await manager._ensure_tables()

        self.assertEqual(await manager.list_admin_ids(), [10002])
