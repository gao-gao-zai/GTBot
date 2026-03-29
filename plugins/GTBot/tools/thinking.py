from __future__ import annotations

import asyncio
from typing import Any, Final

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
    chat_type = getattr(runtime, "chat_type", None)
    message_id = getattr(runtime, "message_id", None)
    if bot is None or message_id is None or chat_type != "group":
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


def _message_contains_thinking_tag(message: Any) -> bool:
    """判断单条消息或生成结果中是否包含 `<thinking` 标记。"""

    content = getattr(message, "content", None)
    if isinstance(content, str) and "<thinking" in content:
        return True

    text = getattr(message, "text", None)
    if isinstance(text, str) and "<thinking" in text:
        return True

    return False


def _response_contains_thinking_tag(response: Any) -> bool:
    """从 LLM 返回对象中尽力提取文本并查找 `<thinking` 标记。"""

    if _message_contains_thinking_tag(response):
        return True

    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        for batch in generations:
            one_batch = batch if isinstance(batch, list) else [batch]
            for generation in one_batch:
                if _message_contains_thinking_tag(generation):
                    return True

                message = getattr(generation, "message", None)
                if _message_contains_thinking_tag(message):
                    return True

    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if _message_contains_thinking_tag(message):
                    return True

    return False


class ThinkingStreamCallback(BaseCallbackHandler):
    """统一处理 thinking 工具与模型输出触发的表情反馈。"""

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:  # noqa: ANN401
        tool_name = str(serialized.get("name") or serialized.get("id") or "").strip()
        if tool_name == "thinking":
            _maybe_add_thinking_emoji()
        return None

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

    def on_chat_model_end(self, response: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if _response_contains_thinking_tag(response):
            _maybe_add_thinking_emoji()
        return None

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if _response_contains_thinking_tag(response):
            _maybe_add_thinking_emoji()
        return None


def register(registry: PluginRegistry) -> None:
    """注册 thinking 工具到 GTBot 插件系统。

    Args:
        registry: GTBot 插件注册表。
    """

    registry.add_tool(thinking)
    registry.add_callback(ThinkingStreamCallback())
