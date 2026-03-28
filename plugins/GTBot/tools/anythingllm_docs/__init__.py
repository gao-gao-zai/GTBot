from __future__ import annotations

from .config import get_anythingllm_docs_plugin_config


def register(registry) -> None:  # noqa: ANN001
    """注册 AnythingLLM 文档插件提供的 Agent 工具。

    Args:
        registry: GTBot 插件注册器，会在启动时接收本插件的工具定义。
    """

    get_anythingllm_docs_plugin_config()

    from .tool import list_anythingllm_documents_tool, query_anythingllm_documents_tool

    registry.add_tool(list_anythingllm_documents_tool)
    registry.add_tool(query_anythingllm_documents_tool)


try:
    from nonebot import get_driver  # type: ignore
except Exception:  # noqa: BLE001
    get_driver = None


if get_driver is not None:
    from . import commands  # noqa: F401
