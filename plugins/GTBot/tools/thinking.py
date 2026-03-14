from __future__ import annotations

import asyncio
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.callbacks import BaseCallbackHandler
from langchain.tools import tool

from plugins.GTBot.services.plugin_system.registry import PluginRegistry
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context


@tool("thinking")
def thinking(text: str) -> str:
    """内部思考工具。任何时候，当你需要思考、规划、自省、分析问题或澄清疑问时，都可以随时调用此工具。
    将你的推演过程、情绪感受、对当前对话状态的判断或者下一步的动作计划写在参数中。

    Args:
        text: 你当下的思考内容，可以是一段内心独白、疑惑、假设或步骤规划。

    Returns:
        空字符串。
    """

    _ = text
    return ""


THINKING_EMOJI_ID: Final[int] = 314


def _maybe_add_thinking_emoji() -> None:
    """如果尚未触发，则对用户原消息添加“thinking”表情贴。

    该函数是同步入口：内部通过 `asyncio.create_task` 调度异步发送。
    """

    ctx = get_current_plugin_context()
    if ctx is None:
        return

    if ctx.extra.get("thinking_emoji_sent") is True:
        return

    runtime = getattr(ctx, "runtime_context", None)
    bot = getattr(runtime, "bot", None)
    message_id = getattr(runtime, "message_id", None)
    if bot is None or message_id is None:
        return

    ctx.extra["thinking_emoji_sent"] = True

    async def _send() -> None:
        try:
            from plugins.GTBot import Fun

            await Fun.set_msg_emoji_like(bot=bot, message_id=int(message_id), emoji_id=THINKING_EMOJI_ID)
        except Exception:
            # 表情失败不影响主流程
            return

    try:
        asyncio.create_task(_send())
    except RuntimeError:
        # 没有运行中的事件循环（例如某些同步测试场景）直接跳过
        return


class ThinkingEmojiMiddleware(AgentMiddleware[AgentState, Any]):
    """当检测到正在调用 `thinking` 工具时，立刻加表情贴。

    说明：
        这里使用 `awrap_tool_call`，可以在工具调用开始（无需等工具执行完成）时触发。
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        tool_name = getattr(getattr(request, "tool_call", None), "name", None) or getattr(
            getattr(request, "tool", None),
            "name",
            None,
        )
        if tool_name == "thinking":
            _maybe_add_thinking_emoji()

        return await handler(request)

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """非流式兜底：若最终输出包含 `<thinking`，也触发表情。"""

        try:
            messages = state.get("messages", [])
            if not messages:
                return None

            last = messages[-1]
            content = getattr(last, "content", "")
            if isinstance(content, str) and "<thinking" in content:
                _maybe_add_thinking_emoji()
        except Exception:
            return None

        return None


class ThinkingStreamCallback(BaseCallbackHandler):
    """在流式 token 中检测 `<thinking` 前缀并触发表情贴。"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:  # noqa: ANN401
        ctx = get_current_plugin_context()
        if ctx is None:
            return None

        if ctx.extra.get("thinking_emoji_sent") is True:
            return None

        # 维护一个小尾巴，支持跨 token 拼接匹配（例如 '<' + 'thinking'）。
        tail = str(ctx.extra.get("thinking_stream_tail", ""))
        tail = (tail + (token or ""))[-64:]
        ctx.extra["thinking_stream_tail"] = tail

        if "<thinking" in tail:
            _maybe_add_thinking_emoji()
        return None


def register(registry: PluginRegistry) -> None:
    """注册 thinking 工具到 GTBot 插件系统。

    Args:
        registry: GTBot 插件注册表。
    """

    registry.add_tool(thinking)
    registry.add_agent_middleware(ThinkingEmojiMiddleware())
    registry.add_callback(ThinkingStreamCallback())
