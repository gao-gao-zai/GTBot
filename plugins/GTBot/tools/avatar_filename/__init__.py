from __future__ import annotations


def register(registry) -> None:  # noqa: ANN001
    """注册头像文件名插件提供的 Agent 工具。

    该插件只暴露给 GTBot Agent 使用，不额外注册聊天命令。
    工具会在本地缓存用户头像或群头像，并返回缓存文件名与路径信息。

    Args:
        registry: GTBot 插件注册器，用于接收本插件暴露的工具定义。
    """

    from .tool import get_group_avatar_filename, get_user_avatar_filename

    registry.add_tool(get_user_avatar_filename)
    registry.add_tool(get_group_avatar_filename)
