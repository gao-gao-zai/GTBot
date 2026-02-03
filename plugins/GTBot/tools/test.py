"""用于测试外部工具连通性的示例工具。"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

from langchain.tools import ToolRuntime, tool

try:
    from ._GroupChatContext import GroupChatContext
except Exception:
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from _GroupChatContext import GroupChatContext


@tool
async def test(runtime: ToolRuntime[GroupChatContext]) -> str:
    """测试工具是否可被加载并正常运行。

    Args:
        runtime: LangChain 运行时对象，包含上下文与调用信息。

    Returns:
        一段包含当前用户与时间的文本。
    """
    user_id = runtime.context.event.get_user_id()
    now_text = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    return f"测试成功！当前用户: {user_id}\n当前时间: {now_text}"