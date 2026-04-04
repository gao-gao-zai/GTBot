from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.callbacks import BaseCallbackHandler

from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context


@tool("demo_last_raw_message")
def demo_last_raw_message(runtime: ToolRuntime[GroupChatContext]) -> str:
    """
    演示工具：返回当前上下文中 raw_messages 的最后一条消息的相关信息。
    这个工具可以用来测试和展示插件系统中如何访问和使用 raw_messages
    """
    ctx = runtime.context
    raw_messages = getattr(ctx, "raw_messages", [])
    if not raw_messages:
        return "raw_messages 为空"

    last = raw_messages[-1]
    user_name = getattr(last, "user_name", "")
    group_id = getattr(last, "group_id", None)
    content = getattr(last, "content", "")
    content_preview = str(content)[:80]
    return f"raw_messages={len(raw_messages)} last.user_name={user_name} last.group_id={group_id} last.content={content_preview}"


async def demo_capture_raw_messages_count(ctx: Any) -> None:
    """在 Agent 启动前记录当前请求的原始消息数量。

    Args:
        ctx: 当前请求的插件上下文。
    """

    ctx.extra["demo_raw_messages_count"] = len(getattr(ctx, "raw_messages", []))


class DemoRawMessagesCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        ctx = get_current_plugin_context()
        if ctx is None:
            return
        ctx.extra["demo_callback_seen_raw_messages"] = len(ctx.raw_messages)


def register(registry) -> None:  # noqa: ANN001
    registry.add_tool(demo_last_raw_message)
    registry.add_pre_agent_processor(demo_capture_raw_messages_count)
    registry.add_callback(DemoRawMessagesCallback())
