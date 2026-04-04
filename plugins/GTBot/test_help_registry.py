from __future__ import annotations

import unittest
from unittest.mock import patch

from plugins.GTBot.services.help import (
    HelpArgumentSpec,
    HelpCommandSpec,
    HelpRegistry,
    render_help_categories,
    render_help_category,
    render_help_detail,
)
from plugins.GTBot.services.permission import PermissionRole


class _FakePermissionManager:
    """用于帮助系统测试的权限管理桩对象。"""

    def __init__(self, allowed_roles: set[PermissionRole]) -> None:
        """初始化权限白名单。

        Args:
            allowed_roles: 允许通过的权限等级集合。
        """
        self._allowed_roles = allowed_roles

    async def has_role(self, user_id: int, required_role: PermissionRole | str) -> bool:
        """模拟权限判断。

        Args:
            user_id: 当前用户 ID，测试中不参与判断。
            required_role: 命令要求的最低权限。

        Returns:
            bool: 是否允许访问该命令。
        """
        _ = user_id
        if isinstance(required_role, str):
            required_role = PermissionRole(required_role)
        return required_role in self._allowed_roles


class TestHelpRegistry(unittest.IsolatedAsyncioTestCase):
    """覆盖帮助注册中心的核心行为。"""

    def _build_spec(
        self,
        *,
        name: str,
        aliases: tuple[str, ...] = (),
        required_role: PermissionRole = PermissionRole.USER,
    ) -> HelpCommandSpec:
        """构造最小帮助项定义，减少测试样板代码。

        Args:
            name: 主命令名。
            aliases: 可选别名列表。
            required_role: 命令所需最低权限。

        Returns:
            HelpCommandSpec: 供测试使用的帮助项对象。
        """
        return HelpCommandSpec(
            name=name,
            aliases=aliases,
            category="测试分组",
            summary=f"{name} 简介",
            description=f"{name} 说明",
            required_role=required_role,
        )

    def test_register_rejects_duplicate_alias(self) -> None:
        """重复别名应被拒绝，避免查询歧义。"""
        registry = HelpRegistry()
        registry.register(self._build_spec(name="帮助", aliases=("menu",)))

        with self.assertRaises(ValueError):
            registry.register(self._build_spec(name="菜单", aliases=("menu",)))

    def test_find_any_spec_supports_alias(self) -> None:
        """命令查询应同时支持主命令名和别名。"""
        registry = HelpRegistry()
        spec = self._build_spec(name="帮助", aliases=("menu", "help"))
        registry.register(spec)

        self.assertIs(registry.find_any_spec("/帮助"), spec)
        self.assertIs(registry.find_any_spec("menu"), spec)
        self.assertIs(registry.find_any_spec("/HELP"), spec)

    async def test_get_visible_specs_filters_by_permission(self) -> None:
        """帮助菜单应仅返回当前用户有权限查看的命令。"""
        registry = HelpRegistry()
        user_spec = self._build_spec(name="查看权限", required_role=PermissionRole.USER)
        admin_spec = self._build_spec(name="查看管理员列表", required_role=PermissionRole.ADMIN)
        registry.register(user_spec)
        registry.register(admin_spec)

        with patch(
            "plugins.GTBot.services.help.get_permission_manager",
            return_value=_FakePermissionManager({PermissionRole.USER}),
        ):
            visible_specs = await registry.get_visible_specs(123)

        self.assertEqual([spec.name for spec in visible_specs], ["查看权限"])

    async def test_find_visible_spec_hides_inaccessible_command(self) -> None:
        """详情查询遇到无权限命令时应表现为未命中。"""
        registry = HelpRegistry()
        registry.register(self._build_spec(name="提拔管理员", required_role=PermissionRole.OWNER))

        with patch(
            "plugins.GTBot.services.help.get_permission_manager",
            return_value=_FakePermissionManager(set()),
        ):
            spec = await registry.find_visible_spec("提拔管理员", 123)

        self.assertIsNone(spec)

    async def test_suggest_visible_commands_returns_close_matches(self) -> None:
        """模糊建议应只基于当前用户可见命令生成。"""
        registry = HelpRegistry()
        registry.register(self._build_spec(name="查看权限"))
        registry.register(self._build_spec(name="查看管理员列表", required_role=PermissionRole.ADMIN))

        with patch(
            "plugins.GTBot.services.help.get_permission_manager",
            return_value=_FakePermissionManager({PermissionRole.USER}),
        ):
            suggestions = await registry.suggest_visible_commands("查看权", 123)

        self.assertEqual(suggestions, ["查看权限"])

    async def test_find_visible_category_returns_category_specs(self) -> None:
        """集合查询应返回对应集合下的可见命令列表。"""
        registry = HelpRegistry()
        registry.register(self._build_spec(name="查看权限"))
        registry.register(
            HelpCommandSpec(
                name="查看会话权限",
                category="会话权限",
                summary="查看会话配置。",
                description="查看会话配置。",
            )
        )

        with patch(
            "plugins.GTBot.services.help.get_permission_manager",
            return_value=_FakePermissionManager({PermissionRole.USER}),
        ):
            category = await registry.find_visible_category("会话权限", 123)

        self.assertIsNotNone(category)
        assert category is not None
        self.assertEqual(category[0], "会话权限")
        self.assertEqual([spec.name for spec in category[1]], ["查看会话权限"])

    def test_render_help_detail_contains_arguments_and_examples(self) -> None:
        """命令详情渲染应包含参数和示例信息。"""
        spec = HelpCommandSpec(
            name="设置会话权限模式",
            aliases=("setmode",),
            category="会话权限",
            summary="设置会话权限模式。",
            description="用于切换群聊或私聊的准入模式。",
            arguments=(
                HelpArgumentSpec(
                    name="<群聊|私聊>",
                    description="目标会话范围。",
                    value_hint="群聊 或 私聊",
                    example="群聊",
                ),
            ),
            examples=("/设置会话权限模式 群聊 白名单",),
            required_role=PermissionRole.ADMIN,
            audience="管理员命令",
        )

        rendered = "\n".join(render_help_detail(spec))

        self.assertIn("命令: /设置会话权限模式", rendered)
        self.assertIn("参数:", rendered)
        self.assertIn("示例:", rendered)
        self.assertIn("/setmode", rendered)

    def test_render_help_categories_lists_category_names(self) -> None:
        """默认帮助首页应优先展示命令集合。"""
        rendered_chunks = render_help_categories(
            [
                ("权限管理", [self._build_spec(name="查看权限")]),
                (
                    "会话权限",
                    [
                        HelpCommandSpec(
                            name="查看会话权限",
                            category="会话权限",
                            summary="查看会话配置。",
                            description="查看会话配置。",
                            sort_key=10,
                        )
                    ],
                ),
            ]
        )

        rendered = "\n".join(rendered_chunks)
        self.assertIn("GTBot 命令集合", rendered)
        self.assertIn("- 权限管理 (1 条)", rendered)
        self.assertIn("- 会话权限 (1 条)", rendered)

    def test_render_help_category_shows_signatures_without_examples(self) -> None:
        """集合详情页应展示命令签名，但不展开示例。"""
        rendered_chunks = render_help_category(
            "会话权限",
            [
                HelpCommandSpec(
                    name="设置会话权限模式",
                    category="会话权限",
                    summary="设置会话权限模式。",
                    description="切换准入模式。",
                    arguments=(
                        HelpArgumentSpec(
                            name="<群聊|私聊>",
                            description="目标会话范围。",
                        ),
                        HelpArgumentSpec(
                            name="<关闭|黑名单|白名单>",
                            description="目标模式。",
                        ),
                    ),
                    examples=("/设置会话权限模式 群聊 白名单",),
                )
            ],
        )

        rendered = "\n".join(rendered_chunks)
        self.assertIn("命令集合: 会话权限", rendered)
        self.assertIn("/设置会话权限模式 <群聊|私聊> <关闭|黑名单|白名单>", rendered)
        self.assertNotIn("示例:", rendered)


if __name__ == "__main__":
    unittest.main()
