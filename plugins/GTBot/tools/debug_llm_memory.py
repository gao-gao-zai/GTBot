from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from nonebot import logger

from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context


def _is_system_message(msg: Any) -> bool:
    msg_type = getattr(msg, "type", None)
    if isinstance(msg_type, str) and msg_type.lower() == "system":
        return True
    return msg.__class__.__name__.lower() == "systemmessage"


def _format_one_message(msg: Any) -> str:
    role = getattr(msg, "type", None)
    if not isinstance(role, str) or not role:
        role = msg.__class__.__name__

    content = getattr(msg, "content", None)
    if isinstance(content, str):
        text = content
    else:
        text = repr(content)

    return f"{role}: {text}".strip()


def _format_memory(messages: Any) -> str:
    if not messages:
        return "<empty>"

    batch: list[list[Any]]
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        batch = messages
    elif isinstance(messages, list):
        batch = [messages]
    else:
        return repr(messages)

    lines: list[str] = []
    for bi, one in enumerate(batch):
        msgs = list(one)
        if msgs and _is_system_message(msgs[0]):
            msgs = msgs[1:]

        if len(batch) > 1:
            lines.append(f"[batch {bi}]")

        for i, m in enumerate(msgs):
            lines.append(f"{i + 1}. {_format_one_message(m)}")

    text = "\n".join(lines)
    if len(text) > 20000:
        return text[:20000] + "\n...(truncated)"
    return text


def _get_group_user_from_plugin_ctx() -> tuple[Any, Any]:
    plugin_ctx = get_current_plugin_context()
    runtime_ctx = getattr(plugin_ctx, "runtime_context", None) if plugin_ctx is not None else None
    group_id = getattr(runtime_ctx, "group_id", None) if runtime_ctx is not None else None
    user_id = getattr(runtime_ctx, "user_id", None) if runtime_ctx is not None else None
    return group_id, user_id


def _log_memory(*, title: str, messages: Any) -> None:
    group_id, user_id = _get_group_user_from_plugin_ctx()
    logger.debug(f"{title} (group_id={group_id} user_id={user_id}):\n{_format_memory(messages)}")


def _mark_final_logged() -> None:
    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return
    plugin_ctx.extra["debug_llm_memory_final_logged"] = True


def _mark_start_logged() -> None:
    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return
    plugin_ctx.extra["debug_llm_memory_start_logged"] = True


def _start_already_logged() -> bool:
    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return False
    return bool(plugin_ctx.extra.get("debug_llm_memory_start_logged"))


def _final_already_logged() -> bool:
    plugin_ctx = get_current_plugin_context()
    if plugin_ctx is None:
        return False
    return bool(plugin_ctx.extra.get("debug_llm_memory_final_logged"))


def _extract_messages_from_outputs(outputs: Any) -> Any:
    if not isinstance(outputs, dict):
        return None

    if outputs.get("messages") is not None:
        return outputs.get("messages")

    inner = outputs.get("output")
    if isinstance(inner, dict) and inner.get("messages") is not None:
        return inner.get("messages")

    return None


class DebugLLMMemoryMiddleware(AgentMiddleware[AgentState, Any]):
    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        if _start_already_logged():
            return None

        messages = state.get("messages")
        if messages is None:
            return None

        _log_memory(title="LLM memory[start]", messages=messages)
        _mark_start_logged()
        return None

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        if _final_already_logged():
            return None

        messages = state.get("messages") or []
        last_ai: AIMessage | None = None
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                last_ai = m
                break
        if last_ai is None:
            return None

        tool_calls = getattr(last_ai, "tool_calls", None)
        if tool_calls:
            return None

        _log_memory(title="LLM memory", messages=messages)
        _mark_final_logged()
        return None


class DebugLLMMemoryCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self._messages_by_run: dict[str, Any] = {}

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        if key in self._messages_by_run:
            return

        messages = inputs.get("messages")
        if messages is not None:
            self._messages_by_run[key] = messages
            if not _start_already_logged():
                _log_memory(title="LLM memory[start]", messages=messages)
                _mark_start_logged()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        messages = _extract_messages_from_outputs(outputs)
        if messages is not None:
            _log_memory(title="LLM memory", messages=messages)
            _mark_final_logged()

        self._messages_by_run.pop(key, None)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._messages_by_run.pop(key, None)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._messages_by_run[key] = messages

        if not _start_already_logged():
            _log_memory(title="LLM memory[start]", messages=messages)
            _mark_start_logged()

    def on_chat_model_end(self, response: Any, **kwargs: Any) -> None:
        if _final_already_logged():
            return
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        messages = self._messages_by_run.pop(key, None)
        if messages is None:
            return

        _log_memory(title="LLM memory", messages=messages)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._messages_by_run.pop(key, None)

    def on_chat_model_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        self._messages_by_run.pop(key, None)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if _final_already_logged():
            return
        run_id = kwargs.get("run_id")
        key = str(run_id) if run_id is not None else "__default__"
        messages = self._messages_by_run.pop(key, None)
        if messages is None:
            return

        _log_memory(title="LLM memory", messages=messages)


def register(registry) -> None:  # noqa: ANN001
    logger.info("debug_llm_memory 插件已加载")
    registry.add_agent_middleware(DebugLLMMemoryMiddleware())
    registry.add_callback(DebugLLMMemoryCallback())
