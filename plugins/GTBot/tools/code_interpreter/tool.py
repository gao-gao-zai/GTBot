from __future__ import annotations

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.GroupChatContext import GroupChatContext


@tool("code_interpreter")
def code_interpreter(code: str, runtime: ToolRuntime[GroupChatContext]) -> str:
    """Code interpreter 工具（模板占位）。

    Args:
        code: 代码文本（当前不执行，仅用于占位展示）。
        runtime: 运行期对象（可用于读取上下文）。

    Returns:
        工具返回文本。
    """

    _ = getattr(runtime, "context", None)
    preview = (code or "").strip().replace("\n", " ")[:80]
    return (
        "Code interpreter 插件模板已加载（占位实现）。"
        "当前版本不执行代码；请后续补充沙箱/权限策略后再开启执行。"
        f" input_preview={preview}"
    )
