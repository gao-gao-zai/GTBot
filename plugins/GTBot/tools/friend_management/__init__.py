from __future__ import annotations

from .config import get_friend_management_plugin_config
from .tool import delete_friend_tool, send_like_tool


def register(registry) -> None:  # noqa: ANN001
    """向 GTBot 插件系统注册好友管理相关工具。

    当前插件同时提供删好友工具与点赞工具。删好友能力继续由 `enabled`
    总开关控制；点赞工具则由 `expose_like_tool_to_agent` 单独控制是否暴露给
    Agent，避免把低风险能力和高风险能力绑死在同一个开关上。

    Args:
        registry: GTBot 插件注册器，用于接收当前插件要暴露的工具。
    """

    get_friend_management_plugin_config()
    registry.add_tool(delete_friend_tool)
    registry.add_tool(send_like_tool, enabled=_like_tool_enabled)


def _like_tool_enabled(_ctx) -> bool:  # noqa: ANN001
    """按当前配置判断是否应向 Agent 暴露点赞工具。

    Args:
        _ctx: GTBot 插件上下文。当前判定只依赖静态配置，因此不读取上下文字段。

    Returns:
        当配置允许向 Agent 暴露点赞工具时返回 `True`。
    """

    return bool(get_friend_management_plugin_config().expose_like_tool_to_agent)


try:
    from nonebot import get_driver  # type: ignore

    get_driver()
except Exception:  # noqa: BLE001
    get_driver = None


if get_driver is not None:
    from . import commands  # noqa: F401
