from __future__ import annotations

from .config import get_openai_draw_plugin_config


def register(registry) -> None:  # noqa: ANN001
    """注册 OpenAI 绘图插件提供的 Agent tool。

    Args:
        registry: GTBot 插件注册器，用于接收本插件暴露的工具定义。
    """

    get_openai_draw_plugin_config()

    from .tool import openai_draw_image, openai_edit_image

    registry.add_tool(openai_draw_image)
    registry.add_tool(openai_edit_image)


try:
    from nonebot import get_driver  # type: ignore

    get_driver()
except Exception:  # noqa: BLE001
    get_driver = None


if get_driver is not None:
    from . import commands  # noqa: F401
