from __future__ import annotations

from typing import Any

from .permissions import (
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


def get_cost_service():
    """延迟返回 GTBot 统一消费账本服务实例。

    这里使用函数内导入来避免 `services.cost` 与 `services.plugin_api` 在包初始化阶段
    形成循环依赖。对调用方来说，使用方式与直接 re-export 保持一致。

    Returns:
        GTBot 统一消费账本服务实例。
    """

    from .costs import get_cost_service as _get_cost_service

    return _get_cost_service()


async def record_cost(**kwargs: Any) -> bool:
    """延迟调用插件显式写账接口，避免包初始化阶段的循环导入。

    Args:
        **kwargs: 透传给 `services.plugin_api.costs.record_cost()` 的命名参数。

    Returns:
        是否成功写入新账单。
    """

    from .costs import record_cost as _record_cost

    return await _record_cost(**kwargs)


async def record_cost_for_current_request(**kwargs: Any) -> bool:
    """延迟调用当前请求写账接口，避免包初始化阶段的循环导入。

    Args:
        **kwargs: 透传给 `services.plugin_api.costs.record_cost_for_current_request()` 的命名参数。

    Returns:
        是否成功写入新账单。
    """

    from .costs import record_cost_for_current_request as _record_cost_for_current_request

    return await _record_cost_for_current_request(**kwargs)

__all__ = [
    "PermissionError",
    "PermissionManager",
    "PermissionRole",
    "get_cost_service",
    "record_cost",
    "record_cost_for_current_request",
    "get_permission_manager",
    "get_role",
    "has_role",
    "require_admin",
    "require_owner",
    "require_role",
]
