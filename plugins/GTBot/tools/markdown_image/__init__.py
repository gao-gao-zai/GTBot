from __future__ import annotations


def register(registry) -> None:  # noqa: ANN001
    """注册 Markdown 图片工具。

    该插件为 GTBot Agent 暴露一个把 Markdown 渲染为 PNG 并直接发送到当前
    会话的工具。插件本身不注册额外命令，也不依赖宿主之外的发送编排逻辑。

    Args:
        registry: GTBot 插件注册器，用于接收本插件暴露的工具定义。
    """

    from .config import get_markdown_image_plugin_config
    from .tool import send_markdown_image

    get_markdown_image_plugin_config()
    registry.add_tool(send_markdown_image)
