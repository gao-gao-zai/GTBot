from __future__ import annotations


def register(registry) -> None:  # noqa: ANN001
    """注册模型直读图片工具。

    该插件向 Agent 暴露一个多模态工具结果：当当前模型和 LangChain 适配层支持
    工具 artifact 中的图片内容时，模型可以直接读取图片，而不是只能看到文字说明。

    Args:
        registry: GTBot 插件注册器，用于接收本插件暴露的工具定义。
    """

    from .tool import get_image_for_model

    registry.add_tool(get_image_for_model)
