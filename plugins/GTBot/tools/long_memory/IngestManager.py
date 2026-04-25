from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import re
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Literal, TypeAlias

from nonebot import logger
from pydantic import BaseModel, ConfigDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from pydantic import SecretStr

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

from plugins.GTBot.model import Message
from . import LongMemoryContainer
from .MappingManager import mapping_manager
from .tool import (
    PUBLIC_KNOWLEDGE_GROUP,
    add_event_log_info,
    add_group_profile_info,
    add_public_knowledge,
    add_user_profile_info,
    delete_event_log_info,
    delete_group_profile_info,
    delete_public_knowledge,
    delete_user_profile_info,
    get_event_log_info,
    get_group_profile_info,
    get_public_knowledge,
    get_user_profile_info,
    search_event_log_info,
    search_group_profile_info,
    search_public_knowledge,
    search_user_profile_info,
    update_event_log_info,
    update_group_profile_info,
    update_public_knowledge,
    update_user_profile_info,
)

from plugins.GTBot.tools.long_memory.config import get_long_memory_plugin_config


def _clean_cq_codes_like_chat(text: str) -> str:
    """清洗文本中的 CQ 码（对齐 Chat.py 的简化规则）。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """

    def _format_cq(di: dict[str, str]) -> str:
        cq_type = di.get("CQ", "")
        if cq_type == "mface":
            summary = di.get("summary", "")
            return "[CQ:mface" + (f",summary={summary}" if summary else "") + "]"

        if cq_type == "record":
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:record" + (f",file={file}" if file else "") + (
                f",file_size={file_size}" if file_size else ""
            ) + "]"

        if cq_type == "image":
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:image" + (f",file={file}" if file else "") + (
                f",file_size={file_size}" if file_size else ""
            ) + "]"

        parts: list[str] = [f"CQ:{cq_type}"] if cq_type else ["CQ:"]
        for key, value in di.items():
            if key == "CQ":
                continue
            escaped_value = str(value).replace("]", "\\]")
            parts.append(f"{key}={escaped_value}")

        out = f"[{','.join(parts)}]"
        if len(out) > 100:
            return out[:97] + "...]"
        return out

    pattern = r"(\[CQ:[^\]]+\])"

    def _replace_match(match: re.Match[str]) -> str:
        full_match = match.group(0)
        cq_str = full_match[4:-1]
        parts = cq_str.split(",", 1)
        cq_dict: dict[str, str] = {"CQ": parts[0].strip()}

        if len(parts) > 1:
            for segment in re.split(r",\s*(?=[^=]+=)", parts[1]):
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                cq_dict[key.strip()] = value.strip().replace("\\]", "]")

        return _format_cq(cq_dict)

    return re.sub(pattern, _replace_match, str(text or ""))


def _payload_get_ts(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _resolve_activity_ts(payload: dict[str, Any]) -> float:
    return max(
        _payload_get_ts(payload, "creation_time"),
        _payload_get_ts(payload, "last_updated"),
        _payload_get_ts(payload, "last_read"),
        _payload_get_ts(payload, "last_called_time"),
    )


async def _cleanup_qdrant_entries_by_type(
    *,
    client: Any,
    collection_name: str,
    payload_type_value: str,
    cutoff_ts: float,
    page_size: int = 256,
    delete_batch_size: int = 256,
) -> int:
    if page_size <= 0 or delete_batch_size <= 0:
        raise ValueError("page_size/delete_batch_size 必须为正整数")

    flt = Filter(
        must=[
            FieldCondition(key="type", match=MatchValue(value=str(payload_type_value))),
        ],
    )

    deleted = 0
    offset: Any = None
    delete_buf: list[Any] = []

    while True:
        batch, next_offset = await client.scroll(
            collection_name=str(collection_name),
            scroll_filter=flt,
            limit=min(int(page_size), 10_000),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in batch:
            payload = p.payload or {}
            if str(payload.get("type", "")) != str(payload_type_value):
                continue
            activity_ts = _resolve_activity_ts(payload)
            if float(activity_ts) <= float(cutoff_ts):
                delete_buf.append(p.id)

            if len(delete_buf) >= int(delete_batch_size):
                await client.delete(
                    collection_name=str(collection_name),
                    points_selector=PointIdsList(points=list(delete_buf)),
                )
                deleted += len(delete_buf)
                delete_buf.clear()

        if not batch or next_offset is None:
            break
        offset = next_offset

    if delete_buf:
        await client.delete(
            collection_name=str(collection_name),
            points_selector=PointIdsList(points=list(delete_buf)),
        )
        deleted += len(delete_buf)
        delete_buf.clear()

    return int(deleted)


def _clean_messages_for_ingest(messages: list[Message]) -> list[Message]:
    """为入库 LLM 构造清洗后的消息列表（仅用于本次调用）。

    Args:
        messages: 原始消息列表。

    Returns:
        清洗后的消息列表。
    """

    out: list[Message] = []
    for m in messages:
        raw = str(getattr(m, "content", "") or "")
        cleaned = _clean_cq_codes_like_chat(raw)
        if cleaned == raw:
            out.append(m)
            continue
        try:
            out.append(m.model_copy(update={"content": cleaned}))
        except Exception:
            out.append(m)
    return out


def _extract_similarity_score(obj: Any) -> float | None:
    """尝试从命中项中提取相似度分数。

    说明：
        LongMemory 的 Qdrant 检索命中通常包含 `similarity` 字段（或 `score`）。
        本函数用于在入库预取阶段做“阀值过滤”。

    Args:
        obj: 命中对象。

    Returns:
        float | None: 相似度分数；若无法提取则返回 None。
    """

    if obj is None:
        return None

    for name in ("similarity", "score"):
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)):
            return float(value)

    if isinstance(obj, dict):
        for key in ("similarity", "score"):
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def _take_hits_above_threshold(
    hits: Any,
    *,
    similarity_threshold: float,
    max_items: int,
) -> list[Any]:
    """按“相似度阀值 + 最大容量”筛选命中列表。

    规则：
        - 仅保留 `similarity >= similarity_threshold` 的条目。
        - 最多返回 `max_items` 条（命中通常已按相似度从高到低排序）。

    Args:
        hits: 命中项列表。
        similarity_threshold: 相似度阀值。
        max_items: 最大返回条数。

    Returns:
        list[Any]: 过滤后的命中列表。
    """

    if max_items <= 0:
        return []

    items = hits if isinstance(hits, list) else []

    out: list[Any] = []
    for item in items:
        score = _extract_similarity_score(item)
        if score is None:
            continue
        if score < float(similarity_threshold):
            continue
        out.append(item)
        if len(out) >= int(max_items):
            break

    return out


def _to_jsonable(obj: Any) -> Any:
    """将对象转换为可 JSON 序列化的结构。

    Args:
        obj: 任意对象。

    Returns:
        Any: 可被 json.dumps 处理的对象。
    """

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _sanitize_public_knowledge_hits_for_llm(hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = str(getattr(hit, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        short_id = str(
            mapping_manager.get_short_id(
                layer="public_knowledge",
                group=PUBLIC_KNOWLEDGE_GROUP,
                long_id=doc_id,
            )
        )
        out.append(
            {
                "short_id": short_id,
                "title": str(getattr(hit, "title", "") or ""),
                "content": str(getattr(hit, "content", "") or ""),
                "similarity": _extract_similarity_score(hit),
                "distance": getattr(hit, "distance", None),
            }
        )
    return out


def _sanitize_event_log_hits_for_llm(*, session_id: str, hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = str(getattr(hit, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        short_id = str(
            mapping_manager.get_short_id(
                layer="event_log",
                group=str(session_id),
                long_id=doc_id,
            )
        )
        out.append(
            {
                "short_id": short_id,
                "event_name": str(getattr(hit, "event_name", "") or ""),
                "session_id": str(getattr(hit, "session_id", "") or ""),
                "relevant_members": _to_jsonable(getattr(hit, "relevant_members", [])),
                "details": str(getattr(hit, "details", "") or ""),
                "similarity": _extract_similarity_score(hit),
                "distance": getattr(hit, "distance", None),
            }
        )
    return out


def _sanitize_user_profile_hits_for_llm(hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = str(getattr(hit, "doc_id", "") or "").strip()
        user_id = int(getattr(hit, "user_id", 0) or 0)
        if not doc_id or user_id <= 0:
            continue
        short_id = str(
            mapping_manager.get_short_id(
                layer="user_profile",
                group=str(user_id),
                long_id=doc_id,
            )
        )
        out.append(
            {
                "user_id": user_id,
                "short_id": short_id,
                "text": str(getattr(hit, "text", "") or ""),
                "similarity": _extract_similarity_score(hit),
                "distance": getattr(hit, "distance", None),
            }
        )
    return out


def _sanitize_group_profile_hits_for_llm(*, group_id: int, hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = str(getattr(hit, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        short_id = str(
            mapping_manager.get_short_id(
                layer="group_profile",
                group=str(group_id),
                long_id=doc_id,
            )
        )
        out.append(
            {
                "group_id": int(getattr(hit, "group_id", group_id) or group_id),
                "short_id": short_id,
                "category": getattr(hit, "category", None),
                "text": str(getattr(hit, "text", "") or ""),
                "similarity": _extract_similarity_score(hit),
                "distance": getattr(hit, "distance", None),
            }
        )
    return out


def _sanitize_group_profile_for_llm(*, group_id: int, profile: Any) -> dict[str, Any] | None:
    if profile is None:
        return None

    desc = getattr(profile, "description", None)
    if not isinstance(desc, list):
        return None

    items: list[dict[str, Any]] = []
    for item in desc:
        doc_id = str(getattr(item, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        short_id = str(
            mapping_manager.get_short_id(
                layer="group_profile",
                group=str(group_id),
                long_id=doc_id,
            )
        )
        items.append(
            {
                "short_id": short_id,
                "text": str(getattr(item, "text", "") or ""),
            }
        )

    return {
        "id": int(getattr(profile, "id", group_id) or group_id),
        "description": items,
    }


def _format_ingest_input(
    *,
    runtime_ctx: "LongMemoryIngestRuntimeContext",
    messages: list[Message],
    prefetch: dict[str, Any],
    max_message_chars: int = 800,
) -> str:
    """将入库输入格式化为更易读的文本。

    说明：
        目标是让 LLM 更容易理解输入内容：
        - 消息列表（按序号）
        - 预取上下文（仍保留 JSON 块，便于精确引用）

    Args:
        runtime_ctx: ToolRuntime 上下文。
        messages: 本次要整理的消息列表。
        prefetch: 预取上下文。
        max_message_chars: 单条消息 content 最大展示字符数。

    Returns:
        str: 格式化后的文本。
    """

    def _clip(text: str) -> str:
        s = str(text or "")
        if len(s) <= int(max_message_chars):
            return s
        return s[: int(max_message_chars)] + "…(truncated)"

    lines: list[str] = []
    lines.append("# 长期记忆入库输入")
    lines.append("")
    lines.append("## 最近聊天消息（按时间顺序）")
    if not messages:
        lines.append("(empty)")
    else:
        for idx, m in enumerate(messages, start=1):
            mid = getattr(m, "message_id", None)
            uid = getattr(m, "user_id", None)
            uname = getattr(m, "user_name", None)
            ts = getattr(m, "send_time", None)
            content = _clip(str(getattr(m, "content", "") or ""))
            lines.append(f"{idx}. message_id={mid} user_id={uid} user_name={uname} send_time={ts}")
            lines.append(f"   content: {content}")

    lines.append("")
    lines.append("## 预取上下文（JSON）")
    lines.append(
        "以下是与本次消息可能相关的长期记忆命中项（已做阀值/容量过滤）。\n"
        "请优先依据这些信息做去重与更新（update），必要时再新增（add）。"
    )
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            _to_jsonable(prefetch),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    return "\n".join(lines)


def _clip_text(text: str, max_chars: int, *, suffix: str = "...(truncated)") -> str:
    """按字符数截断文本。"""

    s = str(text or "")
    if max_chars <= 0:
        return ""
    if len(s) <= int(max_chars):
        return s
    if max_chars <= len(suffix):
        return s[: int(max_chars)]
    return s[: int(max_chars) - len(suffix)] + suffix


def _json_chars(obj: Any) -> int:
    """估算对象序列化后的字符数。"""

    return len(json.dumps(_to_jsonable(obj), ensure_ascii=False, sort_keys=True))


def _sort_prefetch_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按相似度稳定排序预取结果。"""

    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for idx, item in enumerate(items):
        score = _extract_similarity_score(item)
        ranked.append((idx, float(score) if score is not None else -1.0, item))
    ranked.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    return [item for _, _, item in ranked]


def _project_prefetch_item(item: dict[str, Any], *, allowed_fields: list[str]) -> dict[str, Any]:
    """投影出单层预取结果允许暴露给 LLM 的字段。"""

    out: dict[str, Any] = {}
    for field_name in allowed_fields:
        if field_name not in item:
            continue
        value = item.get(field_name)
        if value is None:
            continue
        out[field_name] = _to_jsonable(value)
    return out


def _fit_item_to_budget(
    item: dict[str, Any],
    *,
    text_fields: list[str],
    max_chars: int,
) -> dict[str, Any] | None:
    """将单条记录压缩到字符预算内。"""

    if max_chars <= 0:
        return None

    candidate = dict(item)
    if _json_chars(candidate) <= int(max_chars):
        return candidate

    mutable_fields = [name for name in text_fields if isinstance(candidate.get(name), str) and candidate.get(name)]
    if not mutable_fields:
        return None

    floor_chars = 16
    for _ in range(6):
        current_chars = _json_chars(candidate)
        if current_chars <= int(max_chars):
            return candidate

        overflow = current_chars - int(max_chars)
        changed = False
        active_fields = [name for name in mutable_fields if isinstance(candidate.get(name), str) and candidate.get(name)]
        if not active_fields:
            break

        for name in active_fields:
            current = str(candidate.get(name) or "")
            if len(current) <= floor_chars:
                continue
            shrink_by = max(8, overflow // max(len(active_fields), 1))
            target = max(floor_chars, len(current) - shrink_by)
            clipped = _clip_text(current, target)
            if clipped != current:
                candidate[name] = clipped
                changed = True
            if _json_chars(candidate) <= int(max_chars):
                return candidate

        if not changed:
            break

    return candidate if _json_chars(candidate) <= int(max_chars) else None


def _budget_prefetch_items(
    items: list[dict[str, Any]],
    *,
    allowed_fields: list[str],
    text_fields: list[str],
    max_items: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """按条数和字符预算压缩单层预取命中。"""

    if max_items <= 0 or max_chars <= 0 or not items:
        return []

    selected: list[dict[str, Any]] = []
    remaining = int(max_chars)
    for raw_item in _sort_prefetch_items(items):
        if len(selected) >= int(max_items) or remaining <= 0:
            break

        projected = _project_prefetch_item(raw_item, allowed_fields=allowed_fields)
        if not projected:
            continue

        slots_left = max(1, int(max_items) - len(selected))
        per_item_budget = max(64, remaining // slots_left)
        candidate = _fit_item_to_budget(
            projected,
            text_fields=text_fields,
            max_chars=min(remaining, per_item_budget),
        )
        if candidate is None:
            candidate = _fit_item_to_budget(projected, text_fields=text_fields, max_chars=remaining)
        if candidate is None:
            continue

        candidate_chars = _json_chars(candidate)
        if candidate_chars > remaining:
            continue

        selected.append(candidate)
        remaining -= candidate_chars

    return selected


def _budget_group_profile(
    profile: dict[str, Any] | None,
    *,
    max_items: int,
    max_chars: int,
) -> dict[str, Any] | None:
    """按预算压缩稳定群画像。"""

    if not profile or max_items <= 0 or max_chars <= 0:
        return None

    description = profile.get("description")
    if not isinstance(description, list) or not description:
        return None

    result: dict[str, Any] = {"id": profile.get("id"), "description": []}
    remaining = int(max_chars) - _json_chars(result)
    if remaining <= 0:
        return None

    for item in description[: int(max_items)]:
        if remaining <= 0 or not isinstance(item, dict):
            break
        projected = _project_prefetch_item(item, allowed_fields=["short_id", "text"])
        if not projected:
            continue
        candidate = _fit_item_to_budget(projected, text_fields=["text"], max_chars=remaining)
        if candidate is None:
            continue
        candidate_chars = _json_chars(candidate)
        if candidate_chars > remaining:
            continue
        result["description"].append(candidate)
        remaining -= candidate_chars

    return result if result["description"] else None


def _build_budgeted_prefetch_for_llm(
    *,
    session_id: str,
    group_id: int | None,
    user_id: int | None,
    query_text: str,
    prefetch: dict[str, Any],
    config: "LongMemoryIngestConfig",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构建送入 LLM 的预算化预取上下文。"""

    group_profile_raw = prefetch.get("group_profile")
    group_profile_desc = group_profile_raw.get("description", []) if isinstance(group_profile_raw, dict) else []
    out: dict[str, Any] = {
        "session_id": session_id,
        "group_id": group_id,
        "user_id": user_id,
        "query": _clip_text(query_text, int(config.query_max_chars), suffix="...(clipped)"),
        "event_log_hits": [],
        "group_profile": None,
        "user_profile_hits": [],
        "group_profile_hits": [],
        "public_knowledge_hits": [],
    }

    stats: dict[str, Any] = {
        "candidate_counts": {
            "event_log_hits": len(prefetch.get("event_log_hits", []) or []),
            "group_profile": len(group_profile_desc),
            "user_profile_hits": len(prefetch.get("user_profile_hits", []) or []),
            "group_profile_hits": len(prefetch.get("group_profile_hits", []) or []),
            "public_knowledge_hits": len(prefetch.get("public_knowledge_hits", []) or []),
        },
        "selected_counts": {},
        "selected_chars": {},
        "prefetch_total_chars": 0,
        "global_budget_hit": False,
        "query_chars": len(out["query"]),
    }

    remaining = int(config.prefetch_total_chars)

    event_hits = _budget_prefetch_items(
        list(prefetch.get("event_log_hits", []) or []),
        allowed_fields=["short_id", "event_name", "details", "similarity"],
        text_fields=["event_name", "details"],
        max_items=int(config.event_log_llm_max_items),
        max_chars=min(int(config.event_log_llm_max_chars), max(0, remaining)),
    )
    out["event_log_hits"] = event_hits
    event_chars = _json_chars(event_hits) if event_hits else 0
    remaining -= event_chars
    stats["selected_counts"]["event_log_hits"] = len(event_hits)
    stats["selected_chars"]["event_log_hits"] = event_chars

    group_profile = _budget_group_profile(
        group_profile_raw if isinstance(group_profile_raw, dict) else None,
        max_items=int(config.group_profile_stable_llm_max_items),
        max_chars=min(int(config.group_profile_stable_llm_max_chars), max(0, remaining)),
    )
    out["group_profile"] = group_profile
    group_profile_chars = _json_chars(group_profile) if group_profile else 0
    remaining -= group_profile_chars
    stats["selected_counts"]["group_profile"] = len((group_profile or {}).get("description", [])) if isinstance(group_profile, dict) else 0
    stats["selected_chars"]["group_profile"] = group_profile_chars

    for key, allowed_fields, text_fields, max_items, max_chars in [
        (
            "user_profile_hits",
            ["user_id", "short_id", "text", "similarity"],
            ["text"],
            int(config.user_profile_llm_max_items),
            int(config.user_profile_llm_max_chars),
        ),
        (
            "group_profile_hits",
            ["group_id", "short_id", "category", "text", "similarity"],
            ["category", "text"],
            int(config.group_profile_hits_llm_max_items),
            int(config.group_profile_hits_llm_max_chars),
        ),
        (
            "public_knowledge_hits",
            ["short_id", "title", "content", "similarity"],
            ["title", "content"],
            int(config.public_knowledge_llm_max_items),
            int(config.public_knowledge_llm_max_chars),
        ),
    ]:
        selected_items = _budget_prefetch_items(
            list(prefetch.get(key, []) or []),
            allowed_fields=allowed_fields,
            text_fields=text_fields,
            max_items=max_items,
            max_chars=min(max_chars, max(0, remaining)),
        )
        out[key] = selected_items
        selected_chars = _json_chars(selected_items) if selected_items else 0
        remaining -= selected_chars
        stats["selected_counts"][key] = len(selected_items)
        stats["selected_chars"][key] = selected_chars

    stats["prefetch_total_chars"] = sum(int(v) for v in stats["selected_chars"].values())
    stats["global_budget_hit"] = remaining <= 0
    return out, stats


def _build_budgeted_message_view(
    messages: list[Message],
    *,
    max_messages: int,
    max_chars_per_item: int,
    total_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构建送入 LLM 的预算化最近消息视图。"""

    recent_messages = list(messages)[-int(max_messages):]
    selected_reversed: list[dict[str, Any]] = []
    total_content_chars = 0
    clipped_messages = 0

    for message in reversed(recent_messages):
        remaining = int(total_chars) - total_content_chars
        if remaining <= 0:
            break

        raw_content = str(getattr(message, "content", "") or "")
        content = _clip_text(raw_content, min(int(max_chars_per_item), remaining), suffix="...(clipped)")
        if content != raw_content:
            clipped_messages += 1

        selected_reversed.append(
            {
                "message_id": getattr(message, "message_id", None),
                "user_id": getattr(message, "user_id", None),
                "user_name": getattr(message, "user_name", None),
                "send_time": getattr(message, "send_time", None),
                "content": content,
            }
        )
        total_content_chars += len(content)

    selected = list(reversed(selected_reversed))
    stats = {
        "candidate_count": len(recent_messages),
        "selected_count": len(selected),
        "content_chars": total_content_chars,
        "clipped_messages": clipped_messages,
    }
    return selected, stats


def _build_ingest_input_text(
    *,
    runtime_ctx: "LongMemoryIngestRuntimeContext",
    messages: list[dict[str, Any]],
    prefetch: dict[str, Any],
) -> str:
    """将预算化后的 ingest 输入格式化为文本。"""

    lines: list[str] = []
    lines.append("# LongMemory Ingest Input")
    lines.append("")
    lines.append(
        f"session_id={runtime_ctx.session_id} group_id={runtime_ctx.group_id} user_id={runtime_ctx.user_id}"
    )
    lines.append("")
    lines.append("## Recent Messages")
    if not messages:
        lines.append("(empty)")
    else:
        for idx, message in enumerate(messages, start=1):
            lines.append(
                f"{idx}. message_id={message.get('message_id')} user_id={message.get('user_id')} "
                f"user_name={message.get('user_name')} send_time={message.get('send_time')}"
            )
            lines.append(f"   content: {message.get('content', '')}")

    lines.append("")
    lines.append("## Prefetched Memory Context (JSON)")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            _to_jsonable(prefetch),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    return "\n".join(lines)


def _collect_tool_usage_stats(messages: list[Any]) -> dict[str, Any]:
    """统计一次 ingest 运行中的工具调用情况。"""

    counter: Counter[str] = Counter()
    total = 0
    repeated_query_calls = 0
    prev_query_name: str | None = None

    for message in messages:
        msg_type = str(getattr(message, "type", "") or "").strip().lower()
        role = str(getattr(message, "role", "") or "").strip().lower()
        name = str(getattr(message, "name", "") or "").strip()
        tool_call_id = getattr(message, "tool_call_id", None)
        is_tool_message = bool(name) and (msg_type == "tool" or role == "tool" or tool_call_id is not None)
        if not is_tool_message:
            continue

        total += 1
        counter[name] += 1
        is_query_tool = name.startswith("get_") or name.startswith("search_")
        if is_query_tool and prev_query_name is not None:
            repeated_query_calls += 1
        prev_query_name = name if is_query_tool else None

    return {
        "total_calls": total,
        "tool_counts": dict(sorted(counter.items())),
        "repeated_query_calls": repeated_query_calls,
        "has_repeated_query_calls": repeated_query_calls > 0,
    }


@dataclass(frozen=True, slots=True)
class LongMemoryIngestConfig:
    """长期记忆入库管理器配置。

    说明：
        A/B/C/D/E 来自你的需求描述。

        - `processed_capacity` (A)：已处理消息最多保留条数。
        - `pending_capacity` (B)：未处理消息最多保留条数（超出会丢弃并告警）。
        - `flush_pending_threshold` (C)：当未处理消息数达到该阈值时立即触发一次整理。
        - `idle_flush_seconds` (D)：当 D 秒内无新消息加入时触发一次整理。
        - `flush_max_messages` (E)：每次整理最多交给入库 LLM 的消息条数（取最近的 E 条）。

    Attributes:
        processed_capacity: 已处理队列容量 A。
        pending_capacity: 未处理队列容量 B。
        flush_pending_threshold: 阈值触发 C。
        idle_flush_seconds: 空闲触发 D（秒）。
        flush_max_messages: 每次最多处理 E 条。
        max_concurrent_flushes: 全局并发入库数限制。
        model_id: 入库 LLM 的 model id。
        base_url: 入库 LLM 的 base_url。
        api_key: 入库 LLM 的 api_key。
        model_parameters: OpenAI 兼容参数透传（会剔除 streaming）。
        public_knowledge_similarity_threshold: 公共知识相似度阀值（仅保留 similarity>=阀值的命中）。
        public_knowledge_max_items: 公共知识命中最大返回条数（容量上限）。
        event_log_similarity_threshold: 事件日志相似度阀值（仅保留 similarity>=阀值的命中）。
        event_log_max_items: 事件日志命中最大返回条数（容量上限）。
        user_profile_similarity_threshold: 用户画像相似度阀值（仅保留 similarity>=阀值的命中）。
        user_profile_max_items: 用户画像命中最大返回条数（容量上限）。
        group_profile_min_items_threshold: 群画像条目数阀值（SQLite 无相似度分数，使用“条目数”作为阀值）。
        group_profile_max_items: 群画像最大返回条数（SQLite 按时间排序，容量上限）。
    """

    processed_capacity: int = 200
    pending_capacity: int = 200
    flush_pending_threshold: int = 20
    idle_flush_seconds: float = 60.0
    flush_max_messages: int = 20

    max_concurrent_flushes: int = 2

    query_max_chars: int = 300
    message_max_chars_per_item: int = 200
    message_total_chars: int = 1600
    prefetch_total_chars: int = 1200

    provider_type: Literal["openai_compatible", "openai_responses", "anthropic", "gemini", "dashscope"] = "openai_compatible"
    model_id: str = ""
    base_url: str = ""
    api_key: str = ""
    model_parameters: dict[str, Any] = field(default_factory=dict)

    public_knowledge_similarity_threshold: float = 0.0
    public_knowledge_max_items: int = 5
    public_knowledge_llm_max_items: int = 2
    public_knowledge_llm_max_chars: int = 180

    event_log_similarity_threshold: float = 0.0
    event_log_max_items: int = 5
    event_log_llm_max_items: int = 2
    event_log_llm_max_chars: int = 260

    user_profile_similarity_threshold: float = 0.0
    user_profile_max_items: int = 10
    user_profile_llm_max_items: int = 4
    user_profile_llm_max_chars: int = 260

    group_profile_min_items_threshold: int = 0
    group_profile_max_items: int = 20
    group_profile_hits_llm_max_items: int = 4
    group_profile_hits_llm_max_chars: int = 260
    group_profile_stable_llm_max_items: int = 4
    group_profile_stable_llm_max_chars: int = 260

    def validate(self) -> "LongMemoryIngestConfig":
        """校验并归一化配置。

        Returns:
            LongMemoryIngestConfig: 校验后的配置（同一实例）。

        Raises:
            ValueError: 当配置不满足基本约束时。
        """

        if self.processed_capacity <= 0:
            raise ValueError("processed_capacity(A) 必须 > 0")
        if self.pending_capacity <= 0:
            raise ValueError("pending_capacity(B) 必须 > 0")
        if self.flush_pending_threshold <= 0:
            raise ValueError("flush_pending_threshold(C) 必须 > 0")
        if self.flush_max_messages <= 0:
            raise ValueError("flush_max_messages(E) 必须 > 0")
        if self.idle_flush_seconds <= 0:
            raise ValueError("idle_flush_seconds(D) 必须 > 0")
        if self.flush_max_messages > self.pending_capacity:
            raise ValueError("E 不能大于 B（flush_max_messages > pending_capacity）")
        if self.flush_pending_threshold > self.pending_capacity:
            raise ValueError("C 不能大于 B（flush_pending_threshold > pending_capacity）")
        if self.max_concurrent_flushes <= 0:
            raise ValueError("max_concurrent_flushes 必须 > 0")

        if not (0.0 <= float(self.public_knowledge_similarity_threshold) <= 1.0):
            raise ValueError("public_knowledge_similarity_threshold 必须在 [0, 1] 范围内")
        if self.public_knowledge_max_items <= 0:
            raise ValueError("public_knowledge_max_items 必须 > 0")

        if not (0.0 <= float(self.event_log_similarity_threshold) <= 1.0):
            raise ValueError("event_log_similarity_threshold 必须在 [0, 1] 范围内")
        if self.event_log_max_items <= 0:
            raise ValueError("event_log_max_items 必须 > 0")

        if not (0.0 <= float(self.user_profile_similarity_threshold) <= 1.0):
            raise ValueError("user_profile_similarity_threshold 必须在 [0, 1] 范围内")
        if self.user_profile_max_items <= 0:
            raise ValueError("user_profile_max_items 必须 > 0")

        if self.group_profile_max_items <= 0:
            raise ValueError("group_profile_max_items 必须 > 0")
        if self.group_profile_min_items_threshold < 0:
            raise ValueError("group_profile_min_items_threshold 必须 >= 0")
        if self.group_profile_min_items_threshold > self.group_profile_max_items:
            raise ValueError("group_profile_min_items_threshold 不能大于 group_profile_max_items")
        return self


def _validate_long_memory_ingest_config(self: LongMemoryIngestConfig) -> LongMemoryIngestConfig:
    """校验并归一化 ingest 配置。"""

    if self.processed_capacity <= 0:
        raise ValueError("processed_capacity(A) must be > 0")
    if self.pending_capacity <= 0:
        raise ValueError("pending_capacity(B) must be > 0")
    if self.flush_pending_threshold <= 0:
        raise ValueError("flush_pending_threshold(C) must be > 0")
    if self.flush_max_messages <= 0:
        raise ValueError("flush_max_messages(E) must be > 0")
    if self.idle_flush_seconds <= 0:
        raise ValueError("idle_flush_seconds(D) must be > 0")
    if self.flush_max_messages > self.pending_capacity:
        raise ValueError("flush_max_messages cannot be greater than pending_capacity")
    if self.flush_pending_threshold > self.pending_capacity:
        raise ValueError("flush_pending_threshold cannot be greater than pending_capacity")
    if self.max_concurrent_flushes <= 0:
        raise ValueError("max_concurrent_flushes must be > 0")
    if self.query_max_chars <= 0:
        raise ValueError("query_max_chars must be > 0")
    if self.message_max_chars_per_item <= 0:
        raise ValueError("message_max_chars_per_item must be > 0")
    if self.message_total_chars <= 0:
        raise ValueError("message_total_chars must be > 0")
    if self.prefetch_total_chars < 0:
        raise ValueError("prefetch_total_chars must be >= 0")

    if not (0.0 <= float(self.public_knowledge_similarity_threshold) <= 1.0):
        raise ValueError("public_knowledge_similarity_threshold must be in [0, 1]")
    if self.public_knowledge_max_items <= 0:
        raise ValueError("public_knowledge_max_items must be > 0")
    if self.public_knowledge_llm_max_items < 0:
        raise ValueError("public_knowledge_llm_max_items must be >= 0")
    if self.public_knowledge_llm_max_chars < 0:
        raise ValueError("public_knowledge_llm_max_chars must be >= 0")

    if not (0.0 <= float(self.event_log_similarity_threshold) <= 1.0):
        raise ValueError("event_log_similarity_threshold must be in [0, 1]")
    if self.event_log_max_items <= 0:
        raise ValueError("event_log_max_items must be > 0")
    if self.event_log_llm_max_items < 0:
        raise ValueError("event_log_llm_max_items must be >= 0")
    if self.event_log_llm_max_chars < 0:
        raise ValueError("event_log_llm_max_chars must be >= 0")

    if not (0.0 <= float(self.user_profile_similarity_threshold) <= 1.0):
        raise ValueError("user_profile_similarity_threshold must be in [0, 1]")
    if self.user_profile_max_items <= 0:
        raise ValueError("user_profile_max_items must be > 0")
    if self.user_profile_llm_max_items < 0:
        raise ValueError("user_profile_llm_max_items must be >= 0")
    if self.user_profile_llm_max_chars < 0:
        raise ValueError("user_profile_llm_max_chars must be >= 0")

    if self.group_profile_max_items <= 0:
        raise ValueError("group_profile_max_items must be > 0")
    if self.group_profile_min_items_threshold < 0:
        raise ValueError("group_profile_min_items_threshold must be >= 0")
    if self.group_profile_min_items_threshold > self.group_profile_max_items:
        raise ValueError("group_profile_min_items_threshold cannot exceed group_profile_max_items")
    if self.group_profile_hits_llm_max_items < 0:
        raise ValueError("group_profile_hits_llm_max_items must be >= 0")
    if self.group_profile_hits_llm_max_chars < 0:
        raise ValueError("group_profile_hits_llm_max_chars must be >= 0")
    if self.group_profile_stable_llm_max_items < 0:
        raise ValueError("group_profile_stable_llm_max_items must be >= 0")
    if self.group_profile_stable_llm_max_chars < 0:
        raise ValueError("group_profile_stable_llm_max_chars must be >= 0")
    return self


LongMemoryIngestConfig.validate = _validate_long_memory_ingest_config  # type: ignore[method-assign]


class LongMemoryIngestRuntimeContext(BaseModel):
    """入库 LLM 的最小 ToolRuntime 上下文。

    说明：
        - LongMemory 工具只依赖 `runtime.context.long_memory/session_id/group_id/user_id`。
        - 为避免依赖完整的 `GroupChatContext`，这里定义最小字段集合。

    Attributes:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（建议直接提供，格式：group_<群号>/private_<QQ号>）。
        group_id: 群号（可选）。
        user_id: 用户 ID（可选）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    long_memory: Any
    session_id: str
    group_id: int | None = None
    user_id: int | None = None


@dataclass(slots=True)
class _SessionState:
    """单个会话的入库状态。"""

    processed: Deque[Message]
    pending: Deque[Message]
    seq: int = 0
    last_msg_at: float = 0.0
    last_flush_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle_task: asyncio.Task[None] | None = None


def _validate_ingest_provider_type(provider_type: str) -> IngestProviderType:
    normalized = str(provider_type or "openai_compatible").strip() or "openai_compatible"
    if normalized not in {"openai_compatible", "openai_responses", "anthropic", "gemini", "dashscope"}:
        raise ValueError(
            "provider_type must be one of openai_compatible/openai_responses/anthropic/gemini/dashscope"
        )
    return normalized  # type: ignore[return-value]


def _legacy_default_ingest_prompt() -> str:
    """默认入库提示词。

    Returns:
        str: system prompt。
    """

    return """
# 角色
你是一个客观且具备“自我纠错能力”的记忆记录员。你的任务是如实记录聊天中发生的所有事实与变化，尽量做到“应记尽记，简洁明了”，但必须放到正确的记忆类型里。**当发现旧记忆是错误或玩笑时，你必须能够修改和覆盖它们。**

# 工具调用策略（减少不必要的循环）
你的目标是减少不必要的来回与重复调用。
这**不要求你“一次性把所有工作做完”**，但要求你避免反复交替执行“写入→再检索→再写入”的低效循环。

原则：
1. **先规划再执行**：先根据“最近聊天消息 + 预取上下文”做一次整体判断，列出本次需要更新的记忆点（事件日志/用户画像/群画像/公共知识）。
2. **预取优先**：输入已包含预取命中（hits）与群画像等信息时，优先直接基于这些命中做去重与更新（update/upsert），**不要重复调用 search/get 工具**。
3. **先补齐再落地**：如果确实需要 search/get 来补齐信息，优先把必要的检索集中在前面完成；当信息足够后，再按类别一次性调用多个写入/更新/删除工具完成落地。
4. **谨慎检索**：只有当预取上下文明显不足、且确实需要更多历史信息才能决定更新/新增时，才额外调用 search/get；并尽量做到每类最多一次，避免重复检索。

# 输入格式
你将收到以下格式的输入：

## 1. 最近聊天消息（按时间顺序）
每条消息包含：
- message_id: 消息 ID
- user_id: 发送者 QQ 号
- user_name: 发送者昵称（群聊时有）
- send_time: 发送时间（Unix 时间戳）
- content: 消息内容

## 2. 预取上下文（JSON 格式）
包含与本次消息可能相关的记忆命中：
- event_log_hits: 事件日志命中（同 session 的历史事件）
- user_profile_hits: 用户画像命中（相关用户的个人信息）
- group_profile: 群画像（当前群组特征）
- group_profile_hits: 群画像语义命中（同群的相关条目）
- public_knowledge_hits: 公共知识命中（通用知识/事实）

# 记录原则
1. **客观记录**：只记录事实，不做价值判断。
2. **应记尽记**：尽量让每条输入消息都“在记忆里有落点”（事件日志/用户画像/群画像/公共知识至少其一）。
3. **放对地方**：不同类型的内容必须写入对应的记忆工具；不要把“群内即时对话/诉求”误写为公共知识。
4. **优先更新与覆写（核心原则）**：**写入前必须检查预取上下文**。
   - 如果已有相似记录，必须优先选择更新（update/upsert）而不是新增（add）。
   - **【重要】如果聊天内容明确指出了预取上下文中的旧记忆是错误的、过期的、或是之前的玩笑，你必须用正确的信息彻底覆盖/修改旧信息。绝不能仅仅在错误信息后追加，必须输出修正后的最终事实。**
5. **不编造**：只根据聊天内容记录，不添加未提及的信息。
6. **排除通用常识**：**严禁将 AI 预训练知识中已存在的通用事实写入公共知识**（如：MC 合成配方、科学定义、公开菜谱、通用代码教程）。
   - 此类内容若被讨论，仅记录为【事件日志】。
   - 除非聊天中产生了**特定变体、群内共识、独特见解或上下文相关的结论**，否则不允许写入公共知识。
7. **简洁明了**：记录内容需**精炼**。去除冗余对话，保留核心事实；使用结构化要点，避免长篇大论。

# 信息提取与多维分流（非常重要）
聊天消息往往包含多维度的信息。你**必须独立评估**每一条消息是否包含以下四类信息。
**不要“命中一个就停止”，同一段对话往往需要同时调用多个工具（例如：既记录发生了什么事件，又把提炼出的规则写进群画像）。**
同时，你需要将这些多维度的信息尽量在**同一轮工具调用**中完成落地，以减少不必要的往返。

## A. 事件日志（event_log）—— 必须记录“动作与过程”
凡是“发生了什么、谁说了什么、进行了什么讨论/教学/纠错”，记录动作本身。
*注意：即使你将具体规则提取到了画像中，事件日志也需要记录“某人宣布/教学了某规则”这一过程。*

## B. 用户画像（user_profile）—— 独立提取“个人特质”
在事件的表象下，提取“某个用户稳定的属性/偏好/习惯/长期计划”。
*判断要点：即使过了很久，这个属性依然有效。*
*操作指引：若预取上下文中有相似画像，请合并内容后 update。若发现旧画像是错误的，直接传入修正后的内容覆盖。*

## C. 群画像（group_profile）—— 独立提取“群环境与规则”
在事件的表象下，提取“整个群的长期特征/群规/群文化”。也包括“群内工具/机器人/插件说明”。
*操作指引：若预取上下文中有相似群画像，请合并内容后更新。若群规或配置有变，直接覆盖旧记录。*

## D. 公共知识（public_knowledge）——最严格
**前置过滤（一票否决）：**
- **通用常识过滤**：如果内容是 AI 已知的通用事实/教程/定义（如游戏攻略、科学定义），**禁止写入**。
- 包含群内专有实体（机器人名/内部系统名/群内代号/特定群配置）。
- 依赖群权限或群配置才能成立的流程/命令说明。
- 无法保证“换一个群/换一个人也成立”。
- 记录的是“某人想要/需要/提出的需求/计划/请求”本身。

只有满足以下【全部】条件，才允许写 public_knowledge：
1) **独特性**：不是 AI 预训练知识中随处可见的内容，而是聊天中产生的**独特结论、特定配置、 niche 知识或经过讨论确认的特殊规则**。
2) 去标识化：不得包含任何 QQ 号、昵称、群号、群内指代（“群友/我们群”也尽量不要）
3) 可复用：换一个群/换一个人也成立（通用事实、常识、教程、方法论、明确规则）
4) 非诉求：不得记录“某人想要/需要/提出的需求/计划/请求”本身

*操作指引：调用此工具时，若发现预取上下文中有相似知识，请务必使用 mode: "upsert" 进行更新。如果是纠错，必须用正确的 content 替换错误的 content。*

# 多维调用正反例（仔细学习如何同时使用多个工具）

## 综合正例 1（包含群画像与事件的双重提取）
输入："用户A(123)说：我教你一下我们群里Bella Bot的用法，你@它发'签到'就能领积分。"
正确做法：
- 工具1【事件日志】：记录 "用户A(123) 教学了本群 Bella Bot 的使用方法。"
- 工具2【群画像】：新增或更新群画像，内容为 "Bella Bot 交互说明：@Bot发送'签到'可领积分。"

## 综合正例 2（纠错与覆写旧记忆 - 极其重要）
预取上下文：群画像中有一条 "群主(999)规定：本群禁止发涩图，违者踢出。"
输入："群主(999)说：昨天说禁止发涩图是开玩笑的，大家随便发，别太过分就行。"
正确做法：
- 工具1【事件日志】：记录 "群主(999) 澄清昨天禁止发涩图的规定是玩笑，现允许适度发送。"
- 工具2【群画像】：**必须更新并覆盖旧记录**，将内容修改为 "群规：允许适度发送涩图（群主澄清之前的禁令是玩笑）。" （绝不能保留“禁止发涩图”的旧结论）。

## 反例 1（禁止写入 public_knowledge - 通用常识）
输入："群友 A 问 MC 里铁镐怎么合成，群友 B 回答 3 个铁锭 2 根木棍"
正确做法：
- 只调用【事件日志】：记录"A 询问合成配方，B 进行了回答"
- **不写公共知识**：因为这是游戏通用常识，AI 已知。

# 工具使用说明

## add_event_log_info - 记录事件日志
记录聊天中发生的具体事件、对话、活动。
参数：
- details: 事件详情（必填）
- event_name: 事件名称（可选）
- relevant_members: 相关成员列表 [user_id, ...]
- time_slots: 时间段 [{"start_time": 时间戳，"end_time": 时间戳}]

## add_user_profile_info - 记录用户画像
记录用户的个人特征、偏好、习惯等。**如果是纠正旧画像，请在 info 中直接传入正确的完整内容以实现覆盖。**
参数：
- user_id: 用户 QQ 号
- info: 信息内容（字符串或字符串列表）

## add_group_profile_info - 记录群画像
记录群组整体特征、群规等。**如果是纠正旧群规/旧结论，请在 info 中直接传入修正后的完整内容以实现覆盖。**
参数：
- group_id: 群号
- info: 信息内容（字符串或字符串列表）
- category: 分类（可选）

## add_public_knowledge - 记录公共知识
记录对所有人有用的通用知识、事实。
参数：
- title: 标题
- content: 内容（**若是纠错，此处必须是修正后的正确内容，不要保留错误信息**）
- mode: "add" 或 "upsert"（默认 upsert，**纠错时必须使用 upsert**）
- threshold: 相似度阈值（默认 0.85）

# 输出要求
在执行完所有工具调用后，用简短的一句话总结本次记录的内容。
"""


class LongMemoryIngestManager:
    """长期记忆入库管理器。

    该管理器按 session_id 隔离维护两段队列：
        - processed: 已经整理过的消息（最多 A 条）。
        - pending: 尚未整理的消息（最多 B 条，溢出会丢弃并告警）。

    当触发条件满足时（C 阈值或 D 空闲），会把最近最多 E 条消息交给入库 LLM，
    并附带预取的公共知识/事件/画像上下文，驱动 LLM 使用 LongMemory 工具完成整理。

    Attributes:
        config: 入库配置。
        long_memory: LongMemory 服务容器。
    """

    def __init__(
        self,
        *,
        config: LongMemoryIngestConfig,
        long_memory: LongMemoryContainer,
        ingest_prompt: str | None = None,
        ingest_runner: Callable[[LongMemoryIngestRuntimeContext, list[Message], dict[str, Any]], Awaitable[str]]
        | None = None,
    ) -> None:
        """初始化入库管理器。

        Args:
            config: 入库配置。
            long_memory: LongMemory 服务容器。
            ingest_prompt: 可选自定义 system prompt。
            ingest_runner: 可选自定义执行器（用于测试/替换 LLM）。
        """

        self.config = dataclasses.replace(
            config.validate(),
            provider_type=_validate_ingest_provider_type(config.provider_type),
        )
        self.long_memory: LongMemoryContainer = long_memory
        self._sessions: dict[str, _SessionState] = {}
        self._global_sem = asyncio.Semaphore(int(self.config.max_concurrent_flushes))
        self._ingest_prompt: str = ingest_prompt or _default_ingest_prompt()
        self._ingest_runner = ingest_runner or self._default_ingest_runner

        self._cleanup_lock = asyncio.Lock()
        self._cleanup_last_run_at: float = 0.0

    async def _maybe_cleanup_expired_entries(self) -> None:
        now_ts = float(time.time())
        try:
            cleanup_cfg = getattr(get_long_memory_plugin_config(), "cleanup", None)
        except Exception:
            cleanup_cfg = None

        min_interval_seconds = float(getattr(cleanup_cfg, "min_interval_seconds", 24.0 * 3600.0) or 0.0)
        if (now_ts - float(self._cleanup_last_run_at)) < float(min_interval_seconds):
            return

        async with self._cleanup_lock:
            now_ts2 = float(time.time())
            if (now_ts2 - float(self._cleanup_last_run_at)) < float(min_interval_seconds):
                return
            self._cleanup_last_run_at = now_ts2

        try:
            client = getattr(self.long_memory.public_knowledge, "client", None)
            collection_name = getattr(self.long_memory.public_knowledge, "collection_name", "")
            if client is None or not str(collection_name).strip():
                return

            def _layer(layer_name: str) -> Any:
                return getattr(cleanup_cfg, layer_name, None)

            total = 0
            for t in ["user_profile", "group_profile", "event_log", "public_knowledge"]:
                layer = _layer(t)
                enable = bool(getattr(layer, "enable", True))
                if not enable:
                    continue
                expire_days = int(getattr(layer, "expire_days", 14) or 14)
                if expire_days < 1:
                    continue
                cutoff_ts = now_ts - float(expire_days) * 24.0 * 3600.0
                total += await _cleanup_qdrant_entries_by_type(
                    client=client,
                    collection_name=str(collection_name),
                    payload_type_value=str(t),
                    cutoff_ts=float(cutoff_ts),
                )

            if total > 0:
                logger.info(f"LongMemory 清理过期条目完成: deleted={total}")
        except Exception as exc:
            logger.warning(f"LongMemory 清理过期条目异常: {type(exc).__name__}: {exc!s}")

    def _get_or_create_session(self, session_id: str) -> _SessionState:
        """获取或创建会话状态。"""

        sid = str(session_id).strip()
        if not sid:
            sid = "test_session"

        state = self._sessions.get(sid)
        if state is not None:
            return state

        state = _SessionState(
            processed=deque(maxlen=int(self.config.processed_capacity)),
            pending=deque(maxlen=int(self.config.pending_capacity)),
        )
        self._sessions[sid] = state
        return state

    async def add_message(
        self,
        *,
        session_id: str,
        message: Message,
        group_id: int | None = None,
        user_id: int | None = None,
    ) -> None:
        """向指定会话追加一条消息，并触发必要的入库。

        Args:
            session_id: 会话 ID。
            message: 消息对象（Pydantic Message）。
            group_id: 可选群号。
            user_id: 可选用户 ID。
        """

        sid = str(session_id).strip() or "test_session"
        state = self._get_or_create_session(sid)

        async with state.lock:
            now = float(time.time())
            state.last_msg_at = now

            # pending deque 本身有 maxlen，但我们要在丢弃时告警；因此手动控制。
            if len(state.pending) >= int(self.config.pending_capacity):
                dropped = state.pending.popleft()
                logger.warning(
                    f"LongMemory pending 队列溢出：已丢弃最旧未处理消息（session={sid} "
                    f"message_id={getattr(dropped, 'message_id', None)} "
                    f"user_id={getattr(dropped, 'user_id', None)}）"
                )

            state.pending.append(message)

            # 重置 idle flush 任务
            if state.idle_task is not None and not state.idle_task.done():
                state.idle_task.cancel()
            state.idle_task = asyncio.create_task(
                self._idle_flush_after(
                    session_id=sid,
                    group_id=group_id,
                    user_id=user_id,
                    delay_seconds=float(self.config.idle_flush_seconds),
                )
            )

            if len(state.pending) >= int(self.config.flush_pending_threshold):
                # 阈值触发：不在锁内执行重活
                asyncio.create_task(
                    self.flush_session(session_id=sid, group_id=group_id, user_id=user_id, reason="threshold")
                )

    async def _idle_flush_after(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        delay_seconds: float,
    ) -> None:
        """空闲触发：在 D 秒无新消息后 flush。"""

        try:
            await asyncio.sleep(float(delay_seconds))
            await self.flush_session(session_id=session_id, group_id=group_id, user_id=user_id, reason="idle")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                f"LongMemory idle flush 任务异常（session={session_id}）: {type(exc).__name__}: {exc!r}"
            )
            logger.error(traceback.format_exc())

    async def flush_session(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        reason: str,
    ) -> None:
        """对某个会话执行一次入库整理。"""

        sid = str(session_id).strip() or "test_session"
        state = self._get_or_create_session(sid)

        # 单 session 互斥：避免 threshold/idle 重复触发
        async with state.lock:
            if not state.pending:
                return

            # 取最近 E 条（保持原顺序）
            e = int(self.config.flush_max_messages)
            batch = list(state.pending)[-e:]

            # 从 pending 中移除这段尾部
            for _ in range(min(len(state.pending), len(batch))):
                state.pending.pop()

        # 全局并发限制
        async with self._global_sem:
            try:
                cleaned_batch = _clean_messages_for_ingest(batch)
                extra = await self._prefetch_context(
                    sid,
                    group_id=group_id,
                    user_id=user_id,
                    messages=cleaned_batch,
                )
                runtime_ctx = LongMemoryIngestRuntimeContext(
                    long_memory=self.long_memory,
                    session_id=sid,
                    group_id=group_id,
                    user_id=user_id,
                )
                summary = await self._ingest_runner(runtime_ctx, cleaned_batch, extra)
                logger.info(
                    f"LongMemory 入库完成（session={sid} reason={reason}）: {str(summary)[:500]}"
                )

                try:
                    await self._maybe_cleanup_expired_entries()
                except Exception:
                    pass

                async with state.lock:
                    for m in batch:
                        state.processed.append(m)
                    state.last_flush_at = float(time.time())
            except Exception as exc:
                logger.error(
                    f"LongMemory 入库失败（session={sid} reason={reason} batch={len(batch)}）: "
                    f"{type(exc).__name__}: {exc!r}"
                )
                logger.error(traceback.format_exc())
                # 失败：把 batch 重新塞回 pending 的尾部，尽量不丢
                async with state.lock:
                    for m in batch:
                        if len(state.pending) >= int(self.config.pending_capacity):
                            dropped = state.pending.popleft()
                            logger.warning(
                                f"LongMemory pending 回滚时溢出：丢弃最旧未处理消息（session={sid} "
                                f"message_id={getattr(dropped, 'message_id', None)}）"
                            )
                        state.pending.append(m)

    async def _prefetch_context(
        self,
        session_id: str,
        *,
        group_id: int | None,
        user_id: int | None,
        messages: list[Message],
    ) -> dict[str, Any]:
        """预取与本次消息可能相关的长期记忆上下文。

        Args:
            session_id: 会话 ID。
            group_id: 群号。
            user_id: 用户 ID。
            messages: 本次要整理的消息列表。

        Returns:
            dict[str, Any]: 预取上下文，供入库 LLM 参考。
        """

        query_text = "\n".join([str(getattr(m, "content", "") or "") for m in messages]).strip()
        query_text = query_text[:800]

        out: dict[str, Any] = {
            "session_id": session_id,
            "group_id": group_id,
            "user_id": user_id,
            "query": query_text,
        }

        # 公共知识
        try:
            hits = await self.long_memory.public_knowledge.search_public_knowledge(
                query_text,
                n_results=int(self.config.public_knowledge_max_items),
                min_similarity=float(self.config.public_knowledge_similarity_threshold),
            )
            filtered = _take_hits_above_threshold(
                hits,
                similarity_threshold=float(self.config.public_knowledge_similarity_threshold),
                max_items=int(self.config.public_knowledge_max_items),
            )
            out["public_knowledge_hits"] = _sanitize_public_knowledge_hits_for_llm(filtered)
        except Exception as exc:
            out["public_knowledge_hits"] = []
            out["public_knowledge_error"] = str(exc)

        # 事件日志（限定 session）
        try:
            hits2 = await self.long_memory.event_log_manager.search_events(
                query_text,
                n_results=int(self.config.event_log_max_items),
                session_id=session_id,
                min_similarity=float(self.config.event_log_similarity_threshold),
                order_by="similarity",
                order="desc",
            )
            filtered2 = _take_hits_above_threshold(
                hits2,
                similarity_threshold=float(self.config.event_log_similarity_threshold),
                max_items=int(self.config.event_log_max_items),
            )
            out["event_log_hits"] = _sanitize_event_log_hits_for_llm(session_id=session_id, hits=filtered2)
        except Exception as exc:
            out["event_log_hits"] = []
            out["event_log_error"] = str(exc)

        # 用户画像（全库）
        try:
            hits3 = await self.long_memory.user_profile_manager.search_user_profiles(
                query_text,
                n_results=int(self.config.user_profile_max_items),
                order_by="similarity",
                order="desc",
            )
            filtered3 = _take_hits_above_threshold(
                hits3,
                similarity_threshold=float(self.config.user_profile_similarity_threshold),
                max_items=int(self.config.user_profile_max_items),
            )
            out["user_profile_hits"] = _sanitize_user_profile_hits_for_llm(filtered3)
        except Exception as exc:
            out["user_profile_hits"] = []
            out["user_profile_error"] = str(exc)

        # 群画像（仅群聊）
        if group_id and int(group_id) > 0:
            try:
                # 语义命中（用于提示 LLM 做去重与 update）
                hits4 = await self.long_memory.group_profile_manager.search_group_profiles(
                    query_text,
                    group_id=int(group_id),
                    n_results=int(self.config.group_profile_max_items),
                    min_similarity=None,
                    order_by="similarity",
                    order="desc",
                    touch_last_called=True,
                )
                out["group_profile_hits"] = _sanitize_group_profile_hits_for_llm(
                    group_id=int(group_id),
                    hits=hits4,
                )

                profile = await self.long_memory.group_profile_manager.get_group_profiles(
                    int(group_id),
                    limit=int(self.config.group_profile_max_items),
                    sort_by="last_updated",
                    sort_order="desc",
                )
                desc = getattr(profile, "description", None)
                if isinstance(desc, list) and len(desc) < int(self.config.group_profile_min_items_threshold):
                    out["group_profile"] = None
                else:
                    out["group_profile"] = _sanitize_group_profile_for_llm(
                        group_id=int(group_id),
                        profile=profile,
                    )
            except Exception as exc:
                out["group_profile_hits"] = []
                out["group_profile"] = None
                out["group_profile_error"] = str(exc)
        else:
            out["group_profile_hits"] = []
            out["group_profile"] = None

        return out

    def _resolve_llm_settings(self) -> tuple[str, str, str, str, dict[str, Any]]:
        """解析入库 LLM 的连接参数。

        Returns:
            tuple[str, str, str, str, dict[str, Any]]: (provider_type, model_id, base_url, api_key, model_kwargs)
        """

        provider_type = _validate_ingest_provider_type(self.config.provider_type)
        model_id = (self.config.model_id or "").strip()
        base_url = (self.config.base_url or "").strip()
        api_key = (self.config.api_key or "").strip()

        model_kwargs: dict[str, Any] = dict(self.config.model_parameters or {})

        # 移除 streaming 相关参数（避免透传未知字段）
        for k in ["stream", "streaming", "stream_chunk_chars", "stream_flush_interval_sec"]:
            model_kwargs.pop(k, None)

        return provider_type, model_id, base_url, api_key, model_kwargs

    def _build_ingest_model(
        self,
        *,
        provider_type: str,
        model_id: str,
        base_url: str,
        api_key: str,
        model_kwargs: dict[str, Any],
    ) -> Any:
        if provider_type == "openai_compatible":
            return ChatOpenAI(
                model=model_id,
                base_url=base_url,
                api_key=SecretStr(api_key or ""),
                streaming=False,
                model_kwargs=model_kwargs,
            )

        if provider_type == "openai_responses":
            responses_kwargs = dict(model_kwargs)
            responses_kwargs["use_responses_api"] = True
            return ChatOpenAI(
                model=model_id,
                base_url=base_url,
                api_key=SecretStr(api_key or ""),
                streaming=False,
                model_kwargs=responses_kwargs,
            )

        if provider_type == "anthropic":
            try:
                anthropic_mod = importlib.import_module("langchain_anthropic")
            except ImportError as exc:
                raise RuntimeError(
                    "provider_type=anthropic requires installing langchain-anthropic"
                ) from exc

            chat_cls = getattr(anthropic_mod, "ChatAnthropic", None)
            if chat_cls is None:
                raise RuntimeError("langchain_anthropic.ChatAnthropic is unavailable")
            return chat_cls(
                model=model_id,
                base_url=base_url,
                api_key=SecretStr(api_key or ""),
                streaming=False,
                model_kwargs=model_kwargs,
            )

        if provider_type == "gemini":
            module_candidates = [
                ("langchain_google_genai", "ChatGoogleGenerativeAI"),
                ("langchain_google_vertexai", "ChatVertexAI"),
            ]
            last_error: BaseException | None = None
            for module_name, class_name in module_candidates:
                try:
                    provider_mod = importlib.import_module(module_name)
                except ImportError as exc:
                    last_error = exc
                    continue

                chat_cls = getattr(provider_mod, class_name, None)
                if chat_cls is None:
                    continue
                return chat_cls(
                    model=model_id,
                    google_api_key=SecretStr(api_key or ""),
                    streaming=False,
                    model_kwargs=model_kwargs,
                )

            raise RuntimeError(
                "provider_type=gemini requires installing langchain-google-genai or langchain-google-vertexai"
            ) from last_error

        if provider_type == "dashscope":
            try:
                tongyi_mod = importlib.import_module("langchain_community.chat_models.tongyi")
            except ImportError as exc:
                raise RuntimeError(
                    "provider_type=dashscope requires langchain-community and dashscope"
                ) from exc

            chat_cls = getattr(tongyi_mod, "ChatTongyi", None)
            if chat_cls is None:
                raise RuntimeError("langchain_community.chat_models.tongyi.ChatTongyi is unavailable")

            dashscope_kwargs = dict(model_kwargs)
            if base_url:
                dashscope_kwargs.setdefault("base_url", base_url)
            return chat_cls(
                model=model_id,
                api_key=api_key or None,
                streaming=False,
                model_kwargs=dashscope_kwargs,
            )

        raise RuntimeError(f"unsupported ingest provider_type: {provider_type}")

    def _build_long_memory_tools(self) -> list[BaseTool]:
        """构建入库可用的 LongMemory 工具列表。

        Returns:
            list[BaseTool]: 工具列表。
        """

        return [
            add_user_profile_info,
            delete_user_profile_info,
            update_user_profile_info,
            get_user_profile_info,
            search_user_profile_info,
            add_group_profile_info,
            delete_group_profile_info,
            update_group_profile_info,
            get_group_profile_info,
            search_group_profile_info,
            add_event_log_info,
            get_event_log_info,
            search_event_log_info,
            update_event_log_info,
            delete_event_log_info,
            add_public_knowledge,
            get_public_knowledge,
            search_public_knowledge,
            update_public_knowledge,
            delete_public_knowledge,
        ]

    async def _default_ingest_runner(
        self,
        runtime_ctx: LongMemoryIngestRuntimeContext,
        messages: list[Message],
        extra_context: dict[str, Any],
    ) -> str:
        """默认入库执行器：调用 LLM + LongMemory 工具完成整理。

        Args:
            runtime_ctx: ToolRuntime 上下文。
            messages: 本次消息列表。
            extra_context: 预取上下文。

        Returns:
            str: LLM 的摘要输出。
        """

        provider_type, model_id, base_url, api_key, model_kwargs = self._resolve_llm_settings()
        if not model_id:
            raise ValueError("缺少入库 LLM 配置：model_id/base_url 不能为空")

        if provider_type in {"openai_compatible", "openai_responses", "anthropic"} and not base_url:
            raise ValueError(f"provider_type={provider_type} requires non-empty base_url")

        model = self._build_ingest_model(
            provider_type=provider_type,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_kwargs=model_kwargs,
        )

        tools = self._build_long_memory_tools()
        agent = create_agent(
            model=model,
            tools=tools,
            context_schema=LongMemoryIngestRuntimeContext,
            middleware=[],
        )

        chat_context = [
            SystemMessage(content=self._ingest_prompt),
            HumanMessage(
                content=(
                    "请根据以下输入整理长期记忆，并使用工具进行写入/更新/删除。\n\n"
                    + _format_ingest_input(
                        runtime_ctx=runtime_ctx,
                        messages=messages,
                        prefetch=extra_context,
                    )
                )
            ),
        ]

        result = await agent.ainvoke(
            input={"messages": chat_context},
            context=runtime_ctx,
        )

        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if not msgs:
            return "(no messages)"

        last = msgs[-1]
        content = getattr(last, "content", "")
        return str(content or "").strip() or "(empty)"


def _default_ingest_prompt() -> str:
    """返回精简后的 ingest system prompt。"""

    return """
# 角色
你是长期记忆整理器。你的职责是把最近聊天中值得长期保留的信息写入正确的记忆层，并优先修正旧记忆。

# 本轮目标
先理解最近消息和预取记忆，再用尽量少的工具回合完成本轮写入。

# 先规划后调用工具
- 在第一次工具调用前，先在内部完成本轮写入计划：哪些信息应写入、应更新、应忽略。
- 先判断四类记忆：event_log、user_profile、group_profile、public_knowledge。
- 优先依赖输入里的预取上下文做决策，不要把工具调用当成再次阅读输入的手段。
- 不要命中一类记忆就停止判断；同一段对话可能需要同时写入多类记忆。

# 工具使用原则
- 优先 update_*，再 add_*。
- 如果预取结果已经给出可更新的 short_id，优先直接 update_*，不要重复 get_* 或 search_*。
- get_* / search_* 只在信息明显不足时才作为补充确认工具。
- 能合并写入的内容尽量合并，避免把同一主题拆成多次零碎调用。
- 没有必要时不要调用 delete_*。
- 使用工具时只传 short_id，不要猜测、拼接或改写 ID。
- 同一轮尽量在少量工具调用内完成全部必要写入，不要边查边试错。

# 记录边界
- 只记录有稳定价值的事实、偏好、规则、关系、长期计划或事件。
- 信息不足时宁可不写，也不要编造。
- 不要把通用常识、预训练常识写入 public_knowledge。
- public_knowledge 只收录脱离当前群和当前成员后仍然成立、且不是通用常识的独特知识。

# 四类记忆职责
- event_log：记录本轮真实发生的过程、讨论、澄清、教学、约定、行为和变化。
- user_profile：记录个人稳定特征、偏好、习惯、长期计划、立场、能力或持续关系。
- group_profile：记录群体层面的规则、文化、共识、常驻工具用法、长期环境信息。
- public_knowledge：只记录去身份化后仍成立、可复用、且不是通用常识的独特知识或规则。

# 多层落盘规则
- 如果一段对话既包含“发生了什么”，又暴露了“某人的稳定特征”，通常要同时写 event_log 和 user_profile。
- 如果一段对话既包含事件，又形成了群规则、群共识或群工具说明，通常要同时写 event_log 和 group_profile。
- 不要把本应写入 user_profile 或 group_profile 的内容错误塞进 public_knowledge。
- 不要因为已经写了一层记忆，就漏掉同一段对话中的另一层有效记忆。

# 更新优先级
- 发现旧记忆与当前聊天冲突时，必须覆盖旧结论，不要把新旧矛盾内容同时保留。
- 对 user_profile/group_profile/public_knowledge，若已有相近条目，优先合并更新。
- 对 event_log，记录本轮真实发生的过程、讨论、澄清、教学、约定和变化。
- 如果当前聊天明确说明旧记忆是错误的、过期的、玩笑式的或已被撤销，必须用修正后的最终事实覆盖旧记忆。

# public_knowledge 严格过滤
- 以下内容通常不要写入 public_knowledge：
- 通用常识、模型预训练常识。
- 依赖当前群配置、当前群权限、当前群插件、当前群成员身份才成立的信息。
- 某个人的诉求、计划、要求、偏好、临时需求。
- 群内一次性对话内容、短期上下文、临时安排。
- 只有当内容去身份化后仍成立、可复用、并且不是通用常识时，才允许写入 public_knowledge。

# 输出纪律
- 完成后只用一句简短中文总结本轮写入结果。
""".strip()


async def _long_memory_ingest_prefetch_context(
    self: LongMemoryIngestManager,
    session_id: str,
    *,
    group_id: int | None,
    user_id: int | None,
    messages: list[Message],
) -> dict[str, Any]:
    """构建预算化的 ingest 预取上下文。"""

    query_text = "\n".join(str(getattr(m, "content", "") or "") for m in messages).strip()
    query_text = _clip_text(query_text, int(self.config.query_max_chars), suffix="...(clipped)")

    raw_prefetch: dict[str, Any] = {
        "event_log_hits": [],
        "group_profile": None,
        "user_profile_hits": [],
        "group_profile_hits": [],
        "public_knowledge_hits": [],
    }

    try:
        hits = await self.long_memory.public_knowledge.search_public_knowledge(
            query_text,
            n_results=int(self.config.public_knowledge_max_items),
            min_similarity=float(self.config.public_knowledge_similarity_threshold),
        )
        filtered = _take_hits_above_threshold(
            hits,
            similarity_threshold=float(self.config.public_knowledge_similarity_threshold),
            max_items=int(self.config.public_knowledge_max_items),
        )
        raw_prefetch["public_knowledge_hits"] = _sanitize_public_knowledge_hits_for_llm(filtered)
    except Exception as exc:
        logger.warning(f"LongMemory public_knowledge prefetch failed: {type(exc).__name__}: {exc!s}")

    try:
        event_hits = await self.long_memory.event_log_manager.search_events(
            query_text,
            n_results=int(self.config.event_log_max_items),
            session_id=session_id,
            min_similarity=float(self.config.event_log_similarity_threshold),
            order_by="similarity",
            order="desc",
        )
        filtered = _take_hits_above_threshold(
            event_hits,
            similarity_threshold=float(self.config.event_log_similarity_threshold),
            max_items=int(self.config.event_log_max_items),
        )
        raw_prefetch["event_log_hits"] = _sanitize_event_log_hits_for_llm(session_id=session_id, hits=filtered)
    except Exception as exc:
        logger.warning(f"LongMemory event_log prefetch failed: {type(exc).__name__}: {exc!s}")

    try:
        user_hits = await self.long_memory.user_profile_manager.search_user_profiles(
            query_text,
            n_results=int(self.config.user_profile_max_items),
            order_by="similarity",
            order="desc",
        )
        filtered = _take_hits_above_threshold(
            user_hits,
            similarity_threshold=float(self.config.user_profile_similarity_threshold),
            max_items=int(self.config.user_profile_max_items),
        )
        raw_prefetch["user_profile_hits"] = _sanitize_user_profile_hits_for_llm(filtered)
    except Exception as exc:
        logger.warning(f"LongMemory user_profile prefetch failed: {type(exc).__name__}: {exc!s}")

    if group_id and int(group_id) > 0:
        try:
            hits = await self.long_memory.group_profile_manager.search_group_profiles(
                query_text,
                group_id=int(group_id),
                n_results=int(self.config.group_profile_max_items),
                min_similarity=None,
                order_by="similarity",
                order="desc",
                touch_last_called=True,
            )
            raw_prefetch["group_profile_hits"] = _sanitize_group_profile_hits_for_llm(
                group_id=int(group_id),
                hits=hits,
            )

            profile = await self.long_memory.group_profile_manager.get_group_profiles(
                int(group_id),
                limit=int(self.config.group_profile_max_items),
                sort_by="last_updated",
                sort_order="desc",
            )
            desc = getattr(profile, "description", None)
            if not (isinstance(desc, list) and len(desc) < int(self.config.group_profile_min_items_threshold)):
                raw_prefetch["group_profile"] = _sanitize_group_profile_for_llm(
                    group_id=int(group_id),
                    profile=profile,
                )
        except Exception as exc:
            logger.warning(f"LongMemory group_profile prefetch failed: {type(exc).__name__}: {exc!s}")

    budgeted_prefetch, budget_stats = _build_budgeted_prefetch_for_llm(
        session_id=session_id,
        group_id=group_id,
        user_id=user_id,
        query_text=query_text,
        prefetch=raw_prefetch,
        config=self.config,
    )
    budgeted_prefetch["__budget_stats__"] = budget_stats
    return budgeted_prefetch


async def _long_memory_default_ingest_runner(
    self: LongMemoryIngestManager,
    runtime_ctx: LongMemoryIngestRuntimeContext,
    messages: list[Message],
    extra_context: dict[str, Any],
) -> str:
    """执行预算化 ingest，并记录预算与工具调用统计。"""

    provider_type, model_id, base_url, api_key, model_kwargs = self._resolve_llm_settings()
    if not model_id:
        raise ValueError("missing ingest model_id")

    if provider_type in {"openai_compatible", "openai_responses", "anthropic"} and not base_url:
        raise ValueError(f"provider_type={provider_type} requires non-empty base_url")

    model = self._build_ingest_model(
        provider_type=provider_type,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        model_kwargs=model_kwargs,
    )

    tools = self._build_long_memory_tools()
    agent = create_agent(
        model=model,
        tools=tools,
        context_schema=LongMemoryIngestRuntimeContext,
        middleware=[],
    )

    prefetch_for_llm = dict(extra_context or {})
    prefetch_stats = prefetch_for_llm.pop("__budget_stats__", {})
    message_view, message_stats = _build_budgeted_message_view(
        messages,
        max_messages=int(self.config.flush_max_messages),
        max_chars_per_item=int(self.config.message_max_chars_per_item),
        total_chars=int(self.config.message_total_chars),
    )
    input_text = _build_ingest_input_text(
        runtime_ctx=runtime_ctx,
        messages=message_view,
        prefetch=prefetch_for_llm,
    )

    logger.info(
        "LongMemory ingest budget: "
        f"session={runtime_ctx.session_id} "
        f"messages={message_stats.get('selected_count', 0)}/{message_stats.get('candidate_count', 0)} "
        f"message_chars={message_stats.get('content_chars', 0)} "
        f"prefetch_candidates={prefetch_stats.get('candidate_counts', {})} "
        f"prefetch_selected={prefetch_stats.get('selected_counts', {})} "
        f"prefetch_chars={prefetch_stats.get('selected_chars', {})} "
        f"prefetch_total_chars={prefetch_stats.get('prefetch_total_chars', 0)} "
        f"global_budget_hit={prefetch_stats.get('global_budget_hit', False)}"
    )

    chat_context = [
        SystemMessage(content=self._ingest_prompt),
        HumanMessage(
            content=(
                "请基于以下输入整理长期记忆，并尽量用最少的工具回合完成必要写入。\n\n"
                + input_text
            )
        ),
    ]

    result = await agent.ainvoke(
        input={"messages": chat_context},
        context=runtime_ctx,
    )

    msgs = result.get("messages", []) if isinstance(result, dict) else []
    tool_stats = _collect_tool_usage_stats(msgs)
    logger.info(
        "LongMemory ingest tool usage: "
        f"session={runtime_ctx.session_id} "
        f"total_calls={tool_stats.get('total_calls', 0)} "
        f"tool_counts={tool_stats.get('tool_counts', {})} "
        f"repeated_query_calls={tool_stats.get('repeated_query_calls', 0)}"
    )

    if not msgs:
        return "(no messages)"

    last = msgs[-1]
    content = getattr(last, "content", "")
    return str(content or "").strip() or "(empty)"


setattr(LongMemoryIngestManager, "_prefetch_context", _long_memory_ingest_prefetch_context)
setattr(LongMemoryIngestManager, "_default_ingest_runner", _long_memory_default_ingest_runner)
IngestProviderType: TypeAlias = Literal[
    "openai_compatible",
    "openai_responses",
    "anthropic",
    "gemini",
    "dashscope",
]
