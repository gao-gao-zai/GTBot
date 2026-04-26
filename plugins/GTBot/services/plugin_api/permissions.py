from __future__ import annotations

from typing import cast

from local_plugins.nonebot_plugin_gt_permission import (
    PermissionError,
    PermissionManager,
    PermissionRole,
    get_permission_manager as _get_permission_manager,
)


def get_permission_manager() -> PermissionManager:
    """Return the shared GTBot permission manager for plugin use."""

    return _get_permission_manager()


async def get_role(user_id: int) -> PermissionRole:
    """Resolve the effective role of a user."""

    return await get_permission_manager().get_role(int(user_id))


async def has_role(user_id: int, required_role: PermissionRole | str) -> bool:
    """Check whether a user has at least the required role."""

    return cast(bool, await get_permission_manager().has_role(int(user_id), required_role))


async def require_role(user_id: int, required_role: PermissionRole | str) -> PermissionRole:
    """Require a user to have at least the required role."""

    return await get_permission_manager().require_role(int(user_id), required_role)


async def require_admin(user_id: int) -> PermissionRole:
    """Require ADMIN or OWNER permission."""

    return await require_role(int(user_id), PermissionRole.ADMIN)


async def require_owner(user_id: int) -> PermissionRole:
    """Require OWNER permission."""

    return await require_role(int(user_id), PermissionRole.OWNER)


__all__ = [
    "PermissionError",
    "PermissionManager",
    "PermissionRole",
    "get_permission_manager",
    "get_role",
    "has_role",
    "require_admin",
    "require_owner",
    "require_role",
]
