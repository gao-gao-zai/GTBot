from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GTBOT_LONGMEMORY_AUTOINIT", "0")

from langchain_core.messages import AIMessage, HumanMessage

from plugins.GTBot.tools.long_memory.config import LongMemoryPluginConfig
from plugins.GTBot.tools.long_memory.memory_editor import (
    MemoryEditorSearchArgs,
    MemoryEditorHistoryStore,
    _format_memory_editor_exception,
    _format_memory_editor_tool_exception,
    _is_global_search_target,
    _run_memory_editor_tool,
    parse_memory_target,
    resolve_memory_editor_llm_settings,
)


def test_parse_memory_target_accepts_valid_targets() -> None:
    group_target = parse_memory_target(layer="event_log", target="group_123")
    assert group_target.session_id == "group_123"
    assert group_target.group_id == 123

    private_target = parse_memory_target(layer="event_log", target="private_456")
    assert private_target.session_id == "private_456"
    assert private_target.user_id == 456

    user_target = parse_memory_target(layer="user_profile", target="user:789")
    assert user_target.user_id == 789

    global_target = parse_memory_target(layer="public_knowledge", target="global")
    assert global_target.raw_target == "global"


def test_search_args_allow_omitted_target_and_global_search_aliases() -> None:
    payload = MemoryEditorSearchArgs.model_validate({"layer": "event_log", "query": "店员"})
    assert payload.target is None
    assert _is_global_search_target(payload.target) is True
    assert _is_global_search_target("all") is True
    assert _is_global_search_target("全局") is True
    assert _is_global_search_target("group_123") is False


def test_parse_memory_target_rejects_layer_target_mismatch() -> None:
    try:
        parse_memory_target(layer="group_profile", target="private_123")
    except ValueError as exc:
        assert "不允许" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_history_store_isolation_and_fifo_trim() -> None:
    store = MemoryEditorHistoryStore()

    store.append_turn(admin_user_id=10001, user_text="u1", assistant_text="a1", history_max_messages=3)
    store.append_turn(admin_user_id=10001, user_text="u2", assistant_text="a2", history_max_messages=3)
    store.append_turn(admin_user_id=10002, user_text="u3", assistant_text="a3", history_max_messages=4)

    history_one = store.get_history(10001)
    assert len(history_one) == 3
    assert isinstance(history_one[0], AIMessage)
    assert isinstance(history_one[1], HumanMessage)
    assert isinstance(history_one[2], AIMessage)

    history_two = store.get_history(10002)
    assert len(history_two) == 2
    assert store.clear(10001) is True
    assert store.get_history(10001) == []


def test_resolve_memory_editor_llm_settings_falls_back_to_ingest() -> None:
    plugin_config = LongMemoryPluginConfig()
    plugin_config.ingest.provider_type = "openai_compatible"
    plugin_config.ingest.model_id = "ingest-model"
    plugin_config.ingest.base_url = "https://example.invalid/v1"
    plugin_config.ingest.api_key = "ingest-key"
    plugin_config.ingest.model_parameters = {"temperature": 0.2}

    provider_type, model_id, base_url, api_key, model_parameters = resolve_memory_editor_llm_settings(
        plugin_config=plugin_config
    )

    assert provider_type == "openai_compatible"
    assert model_id == "ingest-model"
    assert base_url == "https://example.invalid/v1"
    assert api_key == "ingest-key"
    assert model_parameters["temperature"] == 0.2


def test_resolve_memory_editor_llm_settings_prefers_editor_override() -> None:
    plugin_config = LongMemoryPluginConfig()
    plugin_config.ingest.provider_type = "openai_compatible"
    plugin_config.ingest.model_id = "ingest-model"
    plugin_config.ingest.base_url = "https://example.invalid/v1"
    plugin_config.ingest.api_key = "ingest-key"
    plugin_config.ingest.model_parameters = {"temperature": 0.2, "top_p": 0.9}

    plugin_config.memory_editor.llm.model_id = "editor-model"
    plugin_config.memory_editor.llm.model_parameters = {"temperature": 0.6}

    provider_type, model_id, base_url, api_key, model_parameters = resolve_memory_editor_llm_settings(
        plugin_config=plugin_config
    )

    assert provider_type == "openai_compatible"
    assert model_id == "editor-model"
    assert base_url == "https://example.invalid/v1"
    assert api_key == "ingest-key"
    assert model_parameters["temperature"] == 0.6
    assert model_parameters["top_p"] == 0.9


def test_format_memory_editor_exception_for_connection_reset() -> None:
    exc = ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。")
    message = _format_memory_editor_exception(exc)
    assert "远程服务中断" in message
    assert "memory_editor.llm" in message


def test_format_memory_editor_exception_for_auth_error() -> None:
    exc = RuntimeError("401 Unauthorized")
    message = _format_memory_editor_exception(exc)
    assert "鉴权失败" in message


def test_format_memory_editor_tool_exception_for_value_error() -> None:
    message = _format_memory_editor_tool_exception(ValueError("target 不能为空。"))
    assert message == "工具调用失败：target 不能为空。"


async def _raise_runtime_error() -> str:
    raise RuntimeError("boom")


async def test_run_memory_editor_tool_returns_error_text_instead_of_raising() -> None:
    result = await _run_memory_editor_tool(
        tool_name="memory_search",
        operation="检索记忆",
        context={"layer": "event_log", "target": "group_123"},
        runner=_raise_runtime_error(),
    )
    assert result == "工具调用异常：boom"
