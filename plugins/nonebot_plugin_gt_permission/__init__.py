from __future__ import annotations

from nonebot.plugin import PluginMetadata

from .commands import register_permission_commands
from .config import GTPermissionConfig, get_permission_config
from .manager import PermissionError, PermissionManager, PermissionRole, get_permission_manager

__plugin_meta__ = PluginMetadata(
    name="GT Permission",
    description="提供 owner/admin/user 三级权限、管理员持久化与权限管理命令。",
    usage=(
        "配置 owner_user_ids 后加载插件，即可使用以下命令：\n"
        "/查看权限 [用户QQ号]\n"
        "/查看管理员列表\n"
        "/提拔管理员 <用户QQ号>\n"
        "/降级管理员 <用户QQ号>"
    ),
    type="application",
    config=GTPermissionConfig,
    supported_adapters={"nonebot.adapters.onebot.v11"},
)


try:
    register_permission_commands()
except ValueError:
    pass


async def get_role(user_id: int) -> PermissionRole:
    """返回用户当前的有效权限等级。

    该函数会复用全局共享的权限管理器，先判断配置中的 owner，
    再判断 sqlite 中持久化的管理员列表，最后回落为普通用户。

    Args:
        user_id: 需要查询权限的 QQ 号。

    Returns:
        用户当前对应的权限等级枚举值。
    """

    return await get_permission_manager().get_role(int(user_id))


async def has_role(user_id: int, required_role: PermissionRole | str) -> bool:
    """检查用户是否至少具备指定权限等级。

    Args:
        user_id: 需要校验的 QQ 号。
        required_role: 目标权限等级，支持枚举或字符串。

    Returns:
        当用户权限不低于 `required_role` 时返回 `True`。
    """

    return await get_permission_manager().has_role(int(user_id), required_role)


async def require_role(user_id: int, required_role: PermissionRole | str) -> PermissionRole:
    """强制要求用户至少具备指定权限等级。

    Args:
        user_id: 需要校验的 QQ 号。
        required_role: 目标权限等级，支持枚举或字符串。

    Returns:
        用户当前实际拥有的权限等级。

    Raises:
        PermissionError: 当用户权限不足以满足要求时抛出。
    """

    return await get_permission_manager().require_role(int(user_id), required_role)


async def require_admin(user_id: int) -> PermissionRole:
    """强制要求用户具备管理员或所有者权限。

    Args:
        user_id: 需要校验的 QQ 号。

    Returns:
        用户当前实际拥有的权限等级。

    Raises:
        PermissionError: 当用户不是管理员或所有者时抛出。
    """

    return await require_role(int(user_id), PermissionRole.ADMIN)


async def require_owner(user_id: int) -> PermissionRole:
    """强制要求用户具备所有者权限。

    Args:
        user_id: 需要校验的 QQ 号。

    Returns:
        用户当前实际拥有的权限等级。

    Raises:
        PermissionError: 当用户不是所有者时抛出。
    """

    return await require_role(int(user_id), PermissionRole.OWNER)


__all__ = [
    "GTPermissionConfig",
    "PermissionError",
    "PermissionManager",
    "PermissionRole",
    "get_permission_config",
    "get_permission_manager",
    "get_role",
    "has_role",
    "require_admin",
    "require_owner",
    "require_role",
]
