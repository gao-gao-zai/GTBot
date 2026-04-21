from __future__ import annotations

import asyncio
from typing import Any, Final

from langchain_core.callbacks import BaseCallbackHandler
from langchain.tools import tool

from plugins.GTBot.services.plugin_system.registry import PluginRegistry
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context


@tool("thinking")
def thinking(text: str) -> str:
    """内部思考工具。"""

    _ = text
    return ""


THINKING_EMOJI_ID: Final[int] = 314


def _maybe_add_thinking_emoji() -> None:
    """如果尚未触发，则对用户原消息添加 thinking 表情贴。"""

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
            from plugins.GTBot.services.shared import fun as Fun

            await Fun.set_msg_emoji_like(
                bot=bot,
                message_id=int(message_id),
                emoji_id=THINKING_EMOJI_ID,
            )
        except Exception:
            return

    def _start_task() -> None:
        asyncio.create_task(_send())

    event_loop = getattr(runtime, "event_loop", None)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if event_loop is None or bool(getattr(event_loop, "is_closed", lambda: True)()):
            return
        event_loop.call_soon_threadsafe(_start_task)
        return

    _start_task()


def _message_contains_thinking_tag(message: Any) -> bool:
    """判断单条消息或生成结果中是否包含 `<thinking` 标记。"""

    content = _extract_text_from_stream_value(getattr(message, "content", None))
    if "<thinking" in content:
        return True

    text = getattr(message, "text", None)
    if isinstance(text, str) and "<thinking" in text:
        return True

    return False


def _extract_text_from_stream_value(value: Any) -> str:
    """从 LangChain 流式 token 或结构化 content block 中提取可搜索文本。

    Responses API 可能把单个 token 或 `AIMessage.content` 表示为由字典组成的列表，
    其中真实文本通常位于 `text` 或 `content` 字段。该函数只拼接文本块并跳过工具调用块，
    避免把工具参数误判为可见输出。

    Args:
        value: LangChain 回调传入的 token、消息内容或 content block 列表。

    Returns:
        可用于检测 `<thinking>` 标记的文本；无法识别时返回空字符串。
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        item_type = str(value.get("type", "")).strip().lower()
        if item_type in {"tool_call", "tool_call_chunk", "function_call", "server_tool_call"}:
            return ""
        if item_type in {"reasoning", "thinking"}:
            reasoning = value.get("reasoning") or value.get("thinking")
            return reasoning if isinstance(reasoning, str) else ""

        text_value = value.get("text")
        if isinstance(text_value, str):
            return text_value
        if isinstance(text_value, dict):
            nested_text = text_value.get("value")
            if isinstance(nested_text, str):
                return nested_text

        content_value = value.get("content")
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            return _extract_text_from_stream_value(content_value)
        return ""

    if isinstance(value, list):
        return "".join(_extract_text_from_stream_value(item) for item in value)

    return ""


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

        token_text = _extract_text_from_stream_value(token)
        if not token_text:
            return None

        tail = str(ctx.extra.get("thinking_stream_tail", ""))
        tail = (tail + token_text)[-64:]
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
    """注册 thinking 工具到 GTBot 插件系统。"""

    registry.add_tool(thinking)
    registry.add_callback(ThinkingStreamCallback())
