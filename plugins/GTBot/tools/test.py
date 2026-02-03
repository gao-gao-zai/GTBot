from langchain.tools import BaseTool, tool, ToolRuntime

from ._api import GroupChatContext




@tool
def test_tool(runtime: ToolRuntime[GroupChatContext], param: str) -> str:
    """这是一个测试工具，返回传入的参数。

    Args:
        param (str): 传入的参数字符串。
    Returns:
        str: 返回传入的参数和当前群号。
    """
    return f"你传入的参数是: {param}, 当前群号: {runtime.context.group_id}"