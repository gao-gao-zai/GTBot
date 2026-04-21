from __future__ import annotations

from local_plugins.nonebot_plugin_gt_permission import (
    PermissionError,
    PermissionManager,
    PermissionRole,
    get_permission_manager,
    get_role,
    has_role,
    require_admin,
    require_owner,
    require_role,
)

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
