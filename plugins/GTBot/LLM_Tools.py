from typing import List

from langchain_core.tools.base import BaseTool

from plugins.GTBot.services.plugin_system.facade import build_plugin_bundle, build_plugin_context

# --- 提供给外部获取工具的接口 ---
def get_current_tools() -> List[BaseTool]:
    bundle = build_plugin_bundle(build_plugin_context(raw_messages=[], runtime_context=None))
    return bundle.tools


__all__ = [
    "build_plugin_bundle",
    "build_plugin_context",
    "get_current_tools",
]
