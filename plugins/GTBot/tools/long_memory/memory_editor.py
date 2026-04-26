from __future__ import annotations

import importlib
import json
import socket
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from nonebot import logger, on_command
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg
from pydantic import BaseModel, ConfigDict, Field

from local_plugins.nonebot_plugin_gt_help import HelpArgumentSpec, HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, require_admin

from plugins.GTBot.llm_provider import build_chat_model
from .MappingManager import mapping_manager
from .config import LongMemoryPluginConfig, get_long_memory_plugin_config
from .tool import (
    _impl_add_event_log_info,
    _impl_add_group_profile_info,
    _impl_add_public_knowledge,
    _impl_add_user_profile_info,
    _impl_delete_event_log_info,
    _impl_delete_group_profile_info,
    _impl_delete_public_knowledge,
    _impl_delete_user_profile_info,
    _impl_search_event_log_info,
    _impl_search_group_profile_info,
    _impl_search_public_knowledge,
    _impl_search_user_profile_info,
    _impl_update_event_log_info,
    _impl_update_group_profile_info,
    _impl_update_public_knowledge,
    _impl_update_user_profile_info,
    normalize_session_id,
)


MemoryLayer = Literal["event_log", "group_profile", "user_profile", "public_knowledge"]


class MemoryEditorSearchArgs(BaseModel):
    """管理员记忆检索命令参数。"""

    layer: MemoryLayer
    target: str | None = None
    query: str
    limit: int = Field(default=5, ge=1, le=100)
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryEditorGetArgs(BaseModel):
    """管理员记忆查看命令参数。"""

    layer: MemoryLayer
    target: str
    short_id: str | list[str]


class MemoryEditorAddArgs(BaseModel):
    """管理员记忆新建命令参数。"""

    layer: MemoryLayer
    target: str
    payload: dict[str, Any]


class MemoryEditorUpdateArgs(BaseModel):
    """管理员记忆修改命令参数。"""

    layer: MemoryLayer
    target: str
    short_id: str
    payload: dict[str, Any]


class MemoryEditorDeleteArgs(BaseModel):
    """管理员记忆删除命令参数。"""

    layer: MemoryLayer
    target: str
    short_id: str | list[str]


@dataclass(frozen=True, slots=True)
class ResolvedMemoryTarget:
    """统一 target 语法解析结果。"""

    raw_target: str
    layer: MemoryLayer
    session_id: str | None = None
    group_id: int | None = None
    user_id: int | None = None


class MemoryEditorRuntimeContext(BaseModel):
    """记忆编辑 LLM 的最小 ToolRuntime 上下文。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    long_memory: Any


class MemoryEditorHistoryStore:
    """管理员记忆编辑历史的进程内存存储。"""

    def __init__(self) -> None:
        """初始化历史存储。"""

        self._histories: dict[int, deque[BaseMessage]] = {}

    def get_history(self, admin_user_id: int) -> list[BaseMessage]:
        """返回指定管理员的历史消息快照。"""

        history = self._histories.get(int(admin_user_id))
        if history is None:
            return []
        return list(history)

    def append_turn(
        self,
        *,
        admin_user_id: int,
        user_text: str,
        assistant_text: str,
        history_max_messages: int,
    ) -> None:
        """向历史中追加一轮问答，并按上限裁剪。"""

        history = self._histories.setdefault(int(admin_user_id), deque())
        history.append(HumanMessage(content=str(user_text)))
        history.append(AIMessage(content=str(assistant_text)))
        while len(history) > int(history_max_messages):
            history.popleft()

    def clear(self, admin_user_id: int) -> bool:
        """清空指定管理员的历史消息。"""

        return self._histories.pop(int(admin_user_id), None) is not None


_memory_editor_history_store = MemoryEditorHistoryStore()


def _get_long_memory_manager() -> Any | None:
    """运行时获取 LongMemory 管理器，兼容 auto_init=false。"""

    package_module = importlib.import_module("plugins.GTBot.tools.long_memory")
    return getattr(package_module, "long_memory_manager", None)


def _memory_editor_prompt() -> str:
    """返回管理员记忆编辑专用 prompt。"""

    return """
# 角色
你是管理员记忆编辑助手，只处理 long memory，不负责普通聊天。

# 系统简介
- 这是一个长期记忆系统，当前只暴露 4 类记忆层：
- `event_log`：事件日志，记录某个群会话或私聊会话里发生过的具体事件、对话事实、阶段性状态
- `group_profile`：群画像，记录某个群长期稳定的规则、偏好、风格、禁忌、常见设定
- `user_profile`：用户画像，记录某个用户长期稳定的兴趣、偏好、身份信息、说话习惯
- `public_knowledge`：公共知识，记录与单个群/用户无关、可全局复用的知识条目
- 你的职责是把管理员的自然语言意图翻译为对这 4 类记忆的检索、查看、新建、修改或删除操作

# 本轮目标
先理解管理员意图，再用最少必要工具完成检索、查看、新建、修改或删除。

# target 命名规则
- 对查看、新建、修改、删除这类非搜索操作，target 必须明确且符合以下规则：
- `event_log` 使用 `group_<群号>` 或 `private_<QQ号>`
- `group_profile` 使用 `group_<群号>`
- `user_profile` 使用 `user:<QQ号>`
- `public_knowledge` 使用 `global`
- 对检索操作，允许省略 target，或使用 `all`、`*`、`全局`、`global_search` 表示全局搜索
- 即使在检索场景，如果管理员给了显式 target，也必须检查它是否与 layer 匹配

# ID 规则
- 所有真正可操作的记录都依赖工具返回的 `short_id`
- `short_id` 是 long_id 的短别名，由系统按 `(layer, group/target)` 作用域生成
- `short_id` 默认是小写字母和数字组成的短串，通常长度较短；它不是全局唯一，只在对应 layer 与 target 范围内有意义
- 同一个 `short_id` 可能在不同 layer 或不同 target 下重复出现，所以修改、查看、删除时必须同时带对 layer 与 target
- 管理员有时会口头用 `E03`、`G12`、`U05`、`P09` 这类写法表达“事件/群画像/用户画像/公共知识”的层级线索；这可以辅助理解意图，但不是底层硬规则
- 正式执行时，必须以工具实际返回的 `short_id` 为准，不得自己编造、补零、猜前缀或做格式转换

# 常见 payload 结构
- 新建 `event_log` 时，`payload.details` 必填；`event_name`、`relevant_members`、`time_slots` 可选
- 新建 `group_profile` 时，`payload.text` 必填；`category` 可选
- 新建 `user_profile` 时，`payload.text` 必填
- 新建 `public_knowledge` 时，`payload.content` 必填；`title` 可选
- 修改 `event_log` 时，至少提供一个可更新字段：`details`、`event_name`、`relevant_members`、`time_slots`
- 修改 `group_profile` 时必须提供 `text`；`category` 可选
- 修改 `user_profile` 时必须提供 `text`
- 修改 `public_knowledge` 时至少提供 `title` 或 `content`

# 工具纪律
- 只使用 `memory_search`、`memory_get`、`memory_add`、`memory_update`、`memory_delete`
- short_id 只能使用工具返回结果里的值，不得编造
- layer 与 target 不匹配时，先要求管理员补充或改正，不要猜
- 删除和修改前先确认命中项；批量删除时要明确列出 short_id；不确定时先检索或查看
- 无法确定时宁可少做，不要误改
- 如果管理员的描述里只给了模糊线索，没有足够信息唯一定位条目，先追问或先做检索，不要直接修改/删除

# 记录边界
- 只围绕管理员指定的 long memory 操作
- 不要假装看到了群聊上下文、召回记忆、记事本或持久化会话
- 不要把“可能是长期信息”和“管理员明确要求你操作哪条记忆”混为一谈；这里只处理显式的管理操作

# 输出纪律
- 面向管理员简洁回答
- 说明你做了什么、命中了什么、用了什么 layer/target、哪些 short_id 被改动
""".strip()


def _split_long_text(text: str, *, max_total_chars: int, max_chunk_chars: int) -> list[str]:
    """按字符预算拆分长文本。"""

    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []
    if len(normalized_text) > int(max_total_chars):
        normalized_text = normalized_text[: int(max_total_chars) - 20].rstrip() + "\n...(已截断)"

    chunks: list[str] = []
    remaining_text = normalized_text
    while remaining_text:
        current_cut = min(len(remaining_text), int(max_chunk_chars))
        newline_index = remaining_text.rfind("\n", 0, current_cut)
        if newline_index <= 0:
            newline_index = current_cut
        chunks.append(remaining_text[:newline_index].rstrip())
        remaining_text = remaining_text[newline_index:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


async def _send_as_forward(
    *,
    bot: Bot,
    event: MessageEvent,
    chunks: list[str],
    name: str,
) -> None:
    """将结果作为合并转发发送到群聊或私聊。"""

    if not chunks:
        return

    try:
        bot_uin = int(str(getattr(bot, "self_id", "") or "").strip() or 0)
    except Exception:
        bot_uin = 0

    nodes = [
        {
            "type": "node",
            "data": {
                "uin": bot_uin,
                "name": name,
                "content": [{"type": "text", "data": {"text": chunk}}],
            },
        }
        for chunk in chunks
    ]

    if isinstance(event, GroupMessageEvent):
        await bot.call_api("send_group_forward_msg", group_id=int(event.group_id), messages=nodes)
        return

    await bot.call_api("send_private_forward_msg", user_id=int(event.user_id), messages=nodes)


async def _send_success_result(
    *,
    bot: Bot,
    event: MessageEvent,
    command_name: str,
    body: str,
) -> None:
    """发送成功结果，优先使用合并转发，失败时降级为分段发送。"""

    cfg = get_long_memory_plugin_config().memory_editor
    result_text = f"{command_name}\n\n{str(body).strip() or '(empty)'}"
    chunks = _split_long_text(
        result_text,
        max_total_chars=int(cfg.forward_max_total_chars),
        max_chunk_chars=int(cfg.forward_max_chunk_chars),
    )
    if not chunks:
        chunks = [result_text]

    try:
        await _send_as_forward(bot=bot, event=event, chunks=chunks, name="GTBot")
        return
    except ActionFailed as exc:
        logger.warning(f"记忆编辑合并转发失败，降级为分段发送: {exc!s}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"记忆编辑合并转发异常，降级为分段发送: {type(exc).__name__}: {exc!s}")

    for chunk in chunks:
        await bot.send(event=event, message=chunk)


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    """将输入安全转换为正整数。"""

    try:
        number = int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} 必须是正整数。") from exc

    if number <= 0:
        raise ValueError(f"{field_name} 必须是正整数。")
    return number


def _coerce_optional_float(value: Any, *, field_name: str) -> float | None:
    """将输入安全转换为可选浮点数。"""

    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} 必须是数字。") from exc


def _coerce_member_list(value: Any, *, field_name: str) -> list[int] | None:
    """将输入转换为成员 ID 列表。"""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是整数列表。")

    members: list[int] = []
    for item in value:
        members.append(_coerce_positive_int(item, field_name=field_name))
    return members


def _ensure_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    """确保输入是 JSON 对象。"""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 JSON 对象。")
    return dict(value)


def _normalize_short_ids(short_id: str | list[str]) -> list[str]:
    """将 short_id 参数规范为非空列表。"""

    raw_values = [short_id] if isinstance(short_id, str) else list(short_id)
    normalized = [str(item).strip() for item in raw_values if str(item).strip()]
    if not normalized:
        raise ValueError("short_id 不能为空。")
    return normalized


def _resolve_short_ids(
    *,
    layer: MemoryLayer,
    group: str,
    short_id: str | list[str],
) -> tuple[list[str], list[str]]:
    """将 short_id 批量解析为 long_id。"""

    normalized_short_ids = _normalize_short_ids(short_id)
    mapped = mapping_manager.get_long_id(layer=layer, group=group, short_id=normalized_short_ids)
    mapped_list = mapped if isinstance(mapped, list) else [mapped]
    if len(mapped_list) < len(normalized_short_ids):
        mapped_list = mapped_list + [None] * (len(normalized_short_ids) - len(mapped_list))
    elif len(mapped_list) > len(normalized_short_ids):
        mapped_list = mapped_list[: len(normalized_short_ids)]

    resolved_long_ids: list[str] = []
    missing_short_ids: list[str] = []
    for raw_short_id, raw_long_id in zip(normalized_short_ids, mapped_list, strict=False):
        if raw_long_id is None:
            missing_short_ids.append(raw_short_id)
            continue
        resolved_long_ids.append(str(raw_long_id))
    return resolved_long_ids, missing_short_ids


def parse_memory_target(*, layer: MemoryLayer, target: str) -> ResolvedMemoryTarget:
    """解析并校验统一 target 语法。"""

    normalized_target = str(target or "").strip()
    if not normalized_target:
        raise ValueError("target 不能为空。")

    if normalized_target == "global":
        if layer != "public_knowledge":
            raise ValueError("只有 public_knowledge 可以使用 target=global。")
        return ResolvedMemoryTarget(raw_target=normalized_target, layer=layer)

    if normalized_target.startswith("group_"):
        group_id = _coerce_positive_int(normalized_target.split("_", 1)[1], field_name="group_id")
        if layer not in {"event_log", "group_profile"}:
            raise ValueError(f"{layer} 不允许使用 group_* target。")
        return ResolvedMemoryTarget(
            raw_target=normalized_target,
            layer=layer,
            session_id=normalize_session_id(normalized_target),
            group_id=group_id,
        )

    if normalized_target.startswith("private_"):
        user_id = _coerce_positive_int(normalized_target.split("_", 1)[1], field_name="user_id")
        if layer != "event_log":
            raise ValueError(f"{layer} 不允许使用 private_* target。")
        return ResolvedMemoryTarget(
            raw_target=normalized_target,
            layer=layer,
            session_id=normalize_session_id(normalized_target),
            user_id=user_id,
        )

    if normalized_target.startswith("user:"):
        user_id = _coerce_positive_int(normalized_target.split(":", 1)[1], field_name="user_id")
        if layer != "user_profile":
            raise ValueError(f"{layer} 不允许使用 user:* target。")
        return ResolvedMemoryTarget(raw_target=normalized_target, layer=layer, user_id=user_id)

    raise ValueError("target 必须是 group_<id>、private_<id>、user:<id> 或 global。")


def _normalize_search_target_text(target: str | None) -> str:
    """规范化搜索场景下的 target 文本。

    Args:
        target: 原始 target 参数，允许为 `None`、空字符串或显式 target。

    Returns:
        str: 去除首尾空白后的 target；未提供时返回空字符串。
    """

    return str(target or "").strip()


def _is_global_search_target(target: str | None) -> bool:
    """判断搜索是否应按全局范围执行。

    Args:
        target: 原始搜索 target。

    Returns:
        bool: 当 target 为空、`all`、`*`、`全局` 或 `global_search` 时返回 `True`。
    """

    normalized_target = _normalize_search_target_text(target).lower()
    return normalized_target in {"", "*", "all", "全局", "global_search"}


def _parse_return_fields(*, raw_value: Any, allowed: set[str], default: str) -> list[str]:
    """解析搜索工具的字段列表配置。

    Args:
        raw_value: 调用方传入的字段字符串。
        allowed: 允许的字段集合。
        default: 默认字段列表，使用逗号分隔。

    Returns:
        list[str]: 经过校验后的字段列表，顺序与输入一致。

    Raises:
        ValueError: 字段列表中包含非法字段。
    """

    text = str(raw_value or "").strip().lower()
    fields = [item.strip() for item in (text or default).split(",") if item.strip()]
    invalid_fields = [field for field in fields if field not in allowed]
    if invalid_fields:
        raise ValueError(f"return_content 包含无效字段={invalid_fields}，有效字段={sorted(allowed)}。")
    return fields


async def _search_event_logs_globally(
    *,
    long_memory: Any,
    query: str,
    limit: int,
    extra_payload: dict[str, Any],
) -> str:
    """执行跨会话的事件日志全局搜索。

    Args:
        long_memory: LongMemory 容器实例。
        query: 检索文本。
        limit: 返回条数上限。
        extra_payload: 额外检索参数。

    Returns:
        str: 格式化后的检索结果文本。
    """

    allowed_fields = {
        "short_id",
        "event_name",
        "session_id",
        "relevant_members",
        "similarity",
        "distance",
        "details",
    }
    return_fields = _parse_return_fields(
        raw_value=extra_payload.get("return_content"),
        allowed=allowed_fields,
        default="session_id,short_id,event_name,similarity,distance,details",
    )

    hits = await long_memory.event_log_manager.search_events(
        str(query),
        n_results=int(limit),
        session_id=None,
        relevant_members_any=_coerce_member_list(
            extra_payload.get("relevant_members_any"),
            field_name="relevant_members_any",
        ),
        min_similarity=_coerce_optional_float(
            extra_payload.get("min_similarity"),
            field_name="min_similarity",
        ),
        order_by="similarity",
        order="desc",
        touch_last_called=True,
    )
    if not hits:
        return f"未检索到事件日志：query={query}。"

    lines: list[str] = []
    for hit in hits:
        session_id = str(getattr(hit, "session_id", "") or "").strip()
        doc_id = str(getattr(hit, "doc_id", "") or "").strip()
        short_id = (
            mapping_manager.get_short_id(layer="event_log", group=session_id, long_id=doc_id)
            if session_id and doc_id
            else "<unknown>"
        )
        parts: list[str] = []
        for field in return_fields:
            if field == "short_id":
                parts.append(f"short_id={short_id}")
            elif field == "event_name":
                parts.append(f"event_name={getattr(hit, 'event_name', '')}")
            elif field == "session_id":
                parts.append(f"session_id={session_id}")
            elif field == "relevant_members":
                parts.append(f"relevant_members={getattr(hit, 'relevant_members', [])}")
            elif field == "similarity":
                parts.append(f"similarity={float(getattr(hit, 'similarity', 0.0) or 0.0):.6f}")
            elif field == "distance":
                parts.append(f"distance={float(getattr(hit, 'distance', 0.0) or 0.0):.6f}")
            elif field == "details":
                text = str(getattr(hit, "details", "") or "").replace("\n", " ").strip()
                parts.append(f"details={text}")
        if parts:
            lines.append(" ".join(parts))

    return "\n".join(lines) if lines else f"未检索到有效事件日志命中：query={query}。"


async def _search_group_profiles_globally(
    *,
    long_memory: Any,
    query: str,
    limit: int,
    extra_payload: dict[str, Any],
) -> str:
    """执行跨群的群画像全局搜索。

    Args:
        long_memory: LongMemory 容器实例。
        query: 检索文本。
        limit: 返回条数上限。
        extra_payload: 额外检索参数。

    Returns:
        str: 格式化后的检索结果文本。
    """

    manager = getattr(long_memory, "group_profile_manager", None)
    client = getattr(manager, "client", None)
    vector_generator = getattr(manager, "vector_generator", None)
    collection_name = getattr(manager, "collection_name", "")
    if manager is None or client is None or vector_generator is None or not collection_name:
        raise ValueError("long_memory.group_profile_manager 不可用。")

    allowed_fields = {"group_id", "short_id", "similarity", "distance", "category", "text"}
    return_fields = _parse_return_fields(
        raw_value=extra_payload.get("return_content"),
        allowed=allowed_fields,
        default="group_id,short_id,similarity,distance,category,text",
    )
    query_vector = await vector_generator.embed_query(str(query))
    hits_response = await client.query_points(
        collection_name=collection_name,
        query=[float(value) for value in query_vector.tolist()],
        limit=int(limit),
        query_filter=manager._build_type_filter(),
        with_payload=True,
        with_vectors=False,
    )

    similarity_threshold = float(extra_payload.get("similarity_threshold", 0.0) or 0.0)
    lines: list[str] = []
    touched_doc_ids: list[str] = []
    for point in hits_response.points:
        payload = dict(point.payload or {})
        if str(payload.get("type", "")) != getattr(manager, "payload_type_value", "group_profile"):
            continue

        similarity = float(point.score or 0.0)
        if similarity < similarity_threshold:
            continue

        doc_id = str(point.id)
        group_id = int(payload.get("group_id", 0) or 0)
        if group_id <= 0:
            continue
        short_id = mapping_manager.get_short_id(layer="group_profile", group=str(group_id), long_id=doc_id)
        touched_doc_ids.append(doc_id)

        parts: list[str] = []
        for field in return_fields:
            if field == "group_id":
                parts.append(f"group_id={group_id}")
            elif field == "short_id":
                parts.append(f"short_id={short_id}")
            elif field == "similarity":
                parts.append(f"similarity={similarity:.4f}")
            elif field == "distance":
                parts.append(f"distance={float(1.0 - similarity):.4f}")
            elif field == "category":
                parts.append(f"category={str(payload.get('category', '') or '').strip()}")
            elif field == "text":
                parts.append(f"text={str(payload.get('description', '') or '').replace('\n', ' ').strip()}")
        if parts:
            lines.append(" ".join(parts))

    if touched_doc_ids:
        try:
            await manager.touch_called_time_by_doc_id(touched_doc_ids)
        except Exception:  # noqa: BLE001
            pass

    return "\n".join(lines) if lines else f"未检索到群画像：query={query}。"


async def _get_user_profile_by_short_id(
    *,
    short_id: str | list[str],
    target: ResolvedMemoryTarget,
) -> str:
    """按短 ID 精确读取用户画像。"""

    resolved_long_ids, missing_short_ids = _resolve_short_ids(
        layer="user_profile",
        group=str(target.user_id),
        short_id=short_id,
    )
    if missing_short_ids:
        raise ValueError(f"short_id 未找到映射: {missing_short_ids}")

    long_memory = _get_long_memory_manager()
    manager = getattr(long_memory, "user_profile_manager", None)
    client = getattr(manager, "client", None)
    if manager is None or client is None:
        raise ValueError("long_memory.user_profile_manager 不可用。")

    qdrant_ids, doc_keys = manager._normalize_point_ids(resolved_long_ids)
    points = await client.retrieve(
        collection_name=manager.collection_name,
        ids=qdrant_ids,
        with_payload=True,
        with_vectors=False,
    )
    point_map = {str(point.id): point for point in points}

    lines: list[str] = []
    for raw_short_id, doc_key in zip(_normalize_short_ids(short_id), doc_keys, strict=False):
        point = point_map.get(str(doc_key))
        if point is None:
            continue
        payload = dict(point.payload or {})
        if str(payload.get("type", "")) != getattr(manager, "payload_type_value", "user_profile"):
            continue
        if int(payload.get("id", 0) or 0) != int(target.user_id or 0):
            continue
        text = str(payload.get("description", "") or "").replace("\n", " ").strip()
        lines.append(f"- short_id={raw_short_id} user_id={target.user_id} text={text}")

    if not lines:
        return "未找到用户画像命中。"

    try:
        await manager.touch_read_time_by_doc_id(resolved_long_ids)
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


async def _get_group_profile_by_short_id(
    *,
    short_id: str | list[str],
    target: ResolvedMemoryTarget,
) -> str:
    """按短 ID 精确读取群画像。"""

    resolved_long_ids, missing_short_ids = _resolve_short_ids(
        layer="group_profile",
        group=str(target.group_id),
        short_id=short_id,
    )
    if missing_short_ids:
        raise ValueError(f"short_id 未找到映射: {missing_short_ids}")

    long_memory = _get_long_memory_manager()
    manager = getattr(long_memory, "group_profile_manager", None)
    client = getattr(manager, "client", None)
    if manager is None or client is None:
        raise ValueError("long_memory.group_profile_manager 不可用。")

    existing = await manager.get_existing_doc_ids(int(target.group_id or 0), resolved_long_ids)
    missing_doc_ids = [doc_id for doc_id in resolved_long_ids if doc_id not in existing]
    if missing_doc_ids:
        raise ValueError(f"目标群画像已不存在: {missing_doc_ids}")

    qdrant_ids, doc_keys = manager._normalize_point_ids(resolved_long_ids)
    points = await client.retrieve(
        collection_name=manager.collection_name,
        ids=qdrant_ids,
        with_payload=True,
        with_vectors=False,
    )
    point_map = {str(point.id): point for point in points}

    lines: list[str] = []
    for raw_short_id, doc_key in zip(_normalize_short_ids(short_id), doc_keys, strict=False):
        point = point_map.get(str(doc_key))
        if point is None:
            continue
        payload = dict(point.payload or {})
        if str(payload.get("type", "")) != getattr(manager, "payload_type_value", "group_profile"):
            continue
        if int(payload.get("group_id", 0) or 0) != int(target.group_id or 0):
            continue
        text = str(payload.get("description", "") or "").replace("\n", " ").strip()
        category = str(payload.get("category", "") or "").strip()
        lines.append(
            f"- short_id={raw_short_id} group_id={target.group_id} "
            f"category={category} text={text}"
        )

    if not lines:
        return "未找到群画像命中。"

    try:
        await manager.touch_read_time_by_doc_id(resolved_long_ids)
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


async def memory_search_impl(
    *,
    layer: MemoryLayer,
    target: str | None = None,
    query: str,
    limit: int = 5,
    extra: dict[str, Any] | None = None,
) -> str:
    """执行统一记忆检索。"""

    long_memory = _get_long_memory_manager()
    if long_memory is None:
        raise ValueError("LongMemory 未初始化。")

    extra_payload = dict(extra or {})
    normalized_target = _normalize_search_target_text(target)

    if layer == "event_log":
        if _is_global_search_target(normalized_target):
            return await _search_event_logs_globally(
                long_memory=long_memory,
                query=str(query),
                limit=int(limit),
                extra_payload=extra_payload,
            )
        resolved_target = parse_memory_target(layer=layer, target=normalized_target)
        return await _impl_search_event_log_info(
            long_memory,
            session_id=str(resolved_target.session_id),
            query=str(query),
            relevant_members_any=_coerce_member_list(
                extra_payload.get("relevant_members_any"),
                field_name="relevant_members_any",
            ),
            limit=int(limit),
            min_similarity=_coerce_optional_float(
                extra_payload.get("min_similarity"),
                field_name="min_similarity",
            ),
            return_content=str(
                extra_payload.get(
                    "return_content",
                    "short_id,event_name,similarity,distance,details",
                )
            ),
        )

    if layer == "group_profile":
        if _is_global_search_target(normalized_target):
            return await _search_group_profiles_globally(
                long_memory=long_memory,
                query=str(query),
                limit=int(limit),
                extra_payload=extra_payload,
            )
        resolved_target = parse_memory_target(layer=layer, target=normalized_target)
        return await _impl_search_group_profile_info(
            long_memory,
            group_id=int(resolved_target.group_id or 0),
            query=str(query),
            limit=int(limit),
            similarity_threshold=float(extra_payload.get("similarity_threshold", 0.0) or 0.0),
            return_content=str(
                extra_payload.get("return_content", "short_id,similarity,distance,category,text")
            ),
        )

    if layer == "user_profile":
        if normalized_target and not _is_global_search_target(normalized_target):
            parse_memory_target(layer=layer, target=normalized_target)
        return await _impl_search_user_profile_info(
            long_memory,
            query=str(query),
            max_users=int(extra_payload.get("max_users", 5) or 5),
            limit=int(limit),
            mode=cast(Any, str(extra_payload.get("mode", "direct") or "direct")),
            return_content=str(extra_payload.get("return_content", "user_id,short_id,text")),
            similarity_threshold=float(extra_payload.get("similarity_threshold", 0.0) or 0.0),
        )

    if normalized_target and not _is_global_search_target(normalized_target):
        parse_memory_target(layer=layer, target=normalized_target)
    return await _impl_search_public_knowledge(
        long_memory,
        query=str(query),
        limit=int(limit),
        min_similarity=_coerce_optional_float(
            extra_payload.get("min_similarity"),
            field_name="min_similarity",
        ),
        return_content=str(extra_payload.get("return_content", "short_id,title,similarity,content")),
    )


async def memory_get_impl(
    *,
    layer: MemoryLayer,
    target: str,
    short_id: str | list[str],
) -> str:
    """执行统一记忆查看。"""

    long_memory = _get_long_memory_manager()
    if long_memory is None:
        raise ValueError("LongMemory 未初始化。")

    resolved_target = parse_memory_target(layer=layer, target=target)
    if layer == "event_log":
        from .tool import _impl_get_event_log_info

        return await _impl_get_event_log_info(
            long_memory,
            session_id=str(resolved_target.session_id),
            short_id=short_id,
        )

    if layer == "group_profile":
        return await _get_group_profile_by_short_id(short_id=short_id, target=resolved_target)

    if layer == "user_profile":
        return await _get_user_profile_by_short_id(short_id=short_id, target=resolved_target)

    from .tool import _impl_get_public_knowledge

    return await _impl_get_public_knowledge(long_memory, short_id=short_id)


def _validate_add_payload(*, layer: MemoryLayer, payload: dict[str, Any]) -> dict[str, Any]:
    """校验新建命令的 payload。"""

    normalized_payload = _ensure_dict(payload, field_name="payload")
    if layer == "event_log":
        details = str(normalized_payload.get("details", "") or "").strip()
        if not details:
            raise ValueError("event_log.payload.details 不能为空。")
        return {
            "details": details,
            "event_name": str(normalized_payload.get("event_name", "") or ""),
            "relevant_members": _coerce_member_list(
                normalized_payload.get("relevant_members"),
                field_name="relevant_members",
            ),
            "time_slots": normalized_payload.get("time_slots"),
        }

    if layer == "user_profile":
        text = str(normalized_payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("user_profile.payload.text 不能为空。")
        return {"text": text}

    if layer == "group_profile":
        text = str(normalized_payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("group_profile.payload.text 不能为空。")
        return {
            "text": text,
            "category": str(normalized_payload.get("category", "") or "").strip() or None,
        }

    content = str(normalized_payload.get("content", "") or "").strip()
    if not content:
        raise ValueError("public_knowledge.payload.content 不能为空。")
    return {
        "title": str(normalized_payload.get("title", "") or ""),
        "content": content,
    }


def _validate_update_payload(*, layer: MemoryLayer, payload: dict[str, Any]) -> dict[str, Any]:
    """校验修改命令的 payload。"""

    normalized_payload = _ensure_dict(payload, field_name="payload")
    if layer == "event_log":
        allowed_payload = {
            "details": normalized_payload.get("details"),
            "event_name": normalized_payload.get("event_name"),
            "relevant_members": _coerce_member_list(
                normalized_payload.get("relevant_members"),
                field_name="relevant_members",
            ),
            "time_slots": normalized_payload.get("time_slots"),
        }
        if all(value is None for value in allowed_payload.values()):
            raise ValueError("event_log.payload 至少需要一个可更新字段。")
        return allowed_payload

    if layer == "user_profile":
        if "text" not in normalized_payload:
            raise ValueError("user_profile.payload.text 不能为空。")
        return {"text": str(normalized_payload.get("text", ""))}

    if layer == "group_profile":
        if "text" not in normalized_payload:
            raise ValueError("group_profile.payload.text 不能为空。")
        return {
            "text": str(normalized_payload.get("text", "")),
            "category": str(normalized_payload.get("category", "")) if "category" in normalized_payload else None,
        }

    if "title" not in normalized_payload and "content" not in normalized_payload:
        raise ValueError("public_knowledge.payload 至少需要 title 或 content。")
    return {
        "title": str(normalized_payload.get("title", "")) if "title" in normalized_payload else None,
        "content": str(normalized_payload.get("content", "")) if "content" in normalized_payload else None,
    }


async def memory_add_impl(
    *,
    layer: MemoryLayer,
    target: str,
    payload: dict[str, Any],
) -> str:
    """执行统一记忆新建。"""

    long_memory = _get_long_memory_manager()
    if long_memory is None:
        raise ValueError("LongMemory 未初始化。")

    resolved_target = parse_memory_target(layer=layer, target=target)
    validated_payload = _validate_add_payload(layer=layer, payload=payload)

    if layer == "event_log":
        return await _impl_add_event_log_info(
            long_memory,
            session_id=str(resolved_target.session_id),
            details=str(validated_payload["details"]),
            event_name=str(validated_payload.get("event_name", "") or ""),
            relevant_members=cast(list[int] | None, validated_payload.get("relevant_members")),
            time_slots=cast(list[dict[str, float]] | None, validated_payload.get("time_slots")),
        )

    if layer == "group_profile":
        return await _impl_add_group_profile_info(
            long_memory,
            group_id=int(resolved_target.group_id or 0),
            info=str(validated_payload["text"]),
            category=cast(str | None, validated_payload.get("category")),
        )

    if layer == "user_profile":
        return await _impl_add_user_profile_info(
            long_memory,
            user_id=int(resolved_target.user_id or 0),
            info=str(validated_payload["text"]),
        )

    return await _impl_add_public_knowledge(
        long_memory,
        title=str(validated_payload.get("title", "") or ""),
        content=str(validated_payload["content"]),
        mode="add",
    )


async def memory_update_impl(
    *,
    layer: MemoryLayer,
    target: str,
    short_id: str,
    payload: dict[str, Any],
) -> str:
    """执行统一记忆修改。"""

    long_memory = _get_long_memory_manager()
    if long_memory is None:
        raise ValueError("LongMemory 未初始化。")

    resolved_target = parse_memory_target(layer=layer, target=target)
    validated_payload = _validate_update_payload(layer=layer, payload=payload)

    if layer == "event_log":
        return await _impl_update_event_log_info(
            long_memory,
            session_id=str(resolved_target.session_id),
            short_id=str(short_id),
            details=cast(str | None, validated_payload.get("details")),
            event_name=cast(str | None, validated_payload.get("event_name")),
            relevant_members=cast(list[int] | None, validated_payload.get("relevant_members")),
            time_slots=cast(list[dict[str, float]] | None, validated_payload.get("time_slots")),
        )

    if layer == "group_profile":
        return await _impl_update_group_profile_info(
            long_memory,
            group_id=int(resolved_target.group_id or 0),
            short_id=str(short_id),
            new_info=str(validated_payload["text"]),
            category=cast(str | None, validated_payload.get("category")),
        )

    if layer == "user_profile":
        return await _impl_update_user_profile_info(
            long_memory,
            user_id=int(resolved_target.user_id or 0),
            short_id=str(short_id),
            new_info=str(validated_payload["text"]),
        )

    return await _impl_update_public_knowledge(
        long_memory,
        short_id=str(short_id),
        title=cast(str | None, validated_payload.get("title")),
        content=cast(str | None, validated_payload.get("content")),
    )


async def memory_delete_impl(
    *,
    layer: MemoryLayer,
    target: str,
    short_id: str | list[str],
) -> str:
    """执行统一记忆删除。"""

    long_memory = _get_long_memory_manager()
    if long_memory is None:
        raise ValueError("LongMemory 未初始化。")

    resolved_target = parse_memory_target(layer=layer, target=target)
    if layer == "event_log":
        return await _impl_delete_event_log_info(
            long_memory,
            session_id=str(resolved_target.session_id),
            short_id=short_id,
        )

    if layer == "group_profile":
        return await _impl_delete_group_profile_info(
            long_memory,
            group_id=int(resolved_target.group_id or 0),
            short_id=short_id,
        )

    if layer == "user_profile":
        return await _impl_delete_user_profile_info(
            long_memory,
            user_id=int(resolved_target.user_id or 0),
            short_id=short_id,
        )

    return await _impl_delete_public_knowledge(long_memory, short_id=short_id)


def _format_memory_editor_tool_exception(exc: BaseException) -> str:
    """将工具调用异常转换为可返回给 Agent 的错误文本。

    Args:
        exc: 工具执行过程中捕获到的异常对象。

    Returns:
        str: 适合直接作为工具返回值交给 Agent 的错误说明。
    """

    message = str(exc).strip() or type(exc).__name__
    if isinstance(exc, ValueError):
        return f"工具调用失败：{message}"
    return f"工具调用异常：{message}"


async def _run_memory_editor_tool(
    *,
    tool_name: str,
    operation: str,
    context: dict[str, Any],
    runner: Awaitable[str],
) -> str:
    """统一执行记忆编辑工具，并将异常转为普通返回文本。

    Args:
        tool_name: 工具名称，用于日志定位。
        operation: 用户可理解的操作描述。
        context: 关键入参上下文，用于异常日志排查。
        runner: 实际执行工具逻辑的 awaitable 对象。

    Returns:
        str: 工具成功结果，或格式化后的错误文本。
    """

    try:
        return await runner
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception(
            f"记忆编辑工具执行失败: {tool_name}",
            exc,
            operation=operation,
            **context,
        )
        return _format_memory_editor_tool_exception(exc)


@tool("memory_search")
async def memory_search(
    layer: MemoryLayer,
    query: str,
    runtime: ToolRuntime[MemoryEditorRuntimeContext],
    target: str = "all",
    limit: int = 5,
    extra: dict[str, Any] | None = None,
) -> str:
    """统一记忆检索工具。"""

    _ = runtime
    return await _run_memory_editor_tool(
        tool_name="memory_search",
        operation="检索记忆",
        context={"layer": layer, "target": target, "query": query, "limit": limit},
        runner=memory_search_impl(layer=layer, target=target, query=query, limit=limit, extra=extra),
    )


@tool("memory_get")
async def memory_get(
    layer: MemoryLayer,
    target: str,
    short_id: str | list[str],
    runtime: ToolRuntime[MemoryEditorRuntimeContext],
) -> str:
    """统一记忆查看工具。"""

    _ = runtime
    return await _run_memory_editor_tool(
        tool_name="memory_get",
        operation="查看记忆",
        context={"layer": layer, "target": target, "short_id": short_id},
        runner=memory_get_impl(layer=layer, target=target, short_id=short_id),
    )


@tool("memory_add")
async def memory_add(
    layer: MemoryLayer,
    target: str,
    payload: dict[str, Any],
    runtime: ToolRuntime[MemoryEditorRuntimeContext],
) -> str:
    """统一记忆新建工具。"""

    _ = runtime
    return await _run_memory_editor_tool(
        tool_name="memory_add",
        operation="新建记忆",
        context={"layer": layer, "target": target},
        runner=memory_add_impl(layer=layer, target=target, payload=payload),
    )


@tool("memory_update")
async def memory_update(
    layer: MemoryLayer,
    target: str,
    short_id: str,
    payload: dict[str, Any],
    runtime: ToolRuntime[MemoryEditorRuntimeContext],
) -> str:
    """统一记忆修改工具。"""

    _ = runtime
    return await _run_memory_editor_tool(
        tool_name="memory_update",
        operation="修改记忆",
        context={"layer": layer, "target": target, "short_id": short_id},
        runner=memory_update_impl(layer=layer, target=target, short_id=short_id, payload=payload),
    )


@tool("memory_delete")
async def memory_delete(
    layer: MemoryLayer,
    target: str,
    short_id: str | list[str],
    runtime: ToolRuntime[MemoryEditorRuntimeContext],
) -> str:
    """统一记忆删除工具。"""

    _ = runtime
    return await _run_memory_editor_tool(
        tool_name="memory_delete",
        operation="删除记忆",
        context={"layer": layer, "target": target, "short_id": short_id},
        runner=memory_delete_impl(layer=layer, target=target, short_id=short_id),
    )


def _build_memory_editor_tools() -> list[Any]:
    """返回记忆编辑 LLM 可用工具列表。"""

    return [
        memory_search,
        memory_get,
        memory_add,
        memory_update,
        memory_delete,
    ]


def resolve_memory_editor_llm_settings(
    *,
    plugin_config: LongMemoryPluginConfig,
) -> tuple[str, str, str, str, dict[str, Any]]:
    """解析记忆编辑 LLM 配置，并对 ingest 配置做回退。"""

    editor_cfg = plugin_config.memory_editor
    ingest_cfg = plugin_config.ingest
    editor_llm_cfg = editor_cfg.llm

    provider_type = str(editor_llm_cfg.provider_type or "").strip() or str(ingest_cfg.provider_type or "").strip()
    model_id = str(editor_llm_cfg.model_id or "").strip() or str(ingest_cfg.model_id or "").strip()
    base_url = str(editor_llm_cfg.base_url or "").strip() or str(ingest_cfg.base_url or "").strip()
    api_key = str(editor_llm_cfg.api_key or "").strip() or str(ingest_cfg.api_key or "").strip()

    model_parameters = dict(ingest_cfg.model_parameters or {})
    editor_parameters = dict(editor_llm_cfg.model_parameters or {})
    if editor_parameters:
        model_parameters.update(editor_parameters)

    ingest_module = importlib.import_module("plugins.GTBot.tools.long_memory.IngestManager")
    validate_provider_type = getattr(ingest_module, "_validate_ingest_provider_type")
    return (
        str(validate_provider_type(provider_type)),
        model_id,
        base_url,
        api_key,
        model_parameters,
    )


def _build_memory_editor_model(
    *,
    provider_type: str,
    model_id: str,
    base_url: str,
    api_key: str,
    model_parameters: dict[str, Any],
) -> Any:
    """构建记忆编辑链路使用的聊天模型实例。

    记忆编辑默认使用非流式调用；当 provider 为 `openai_responses` 时，
    额外补齐 `responses/v1` 输出版本，以兼容当前编辑链路的 JSON 解析方式。

    Args:
        provider_type: 归一化后的提供商类型。
        model_id: 上游模型 ID。
        base_url: 提供商基础地址。
        api_key: 提供商 API 密钥。
        model_parameters: 归一化后的模型参数。

    Returns:
        可直接交给 LangChain agent 使用的模型对象。
    """

    adapter_parameters = dict(model_parameters)
    if provider_type == "openai_responses":
        adapter_parameters.setdefault("output_version", "responses/v1")
    return build_chat_model(
        provider_type=provider_type,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        streaming=False,
        model_parameters=adapter_parameters,
    )


class AdminMemoryEditorManager:
    """管理员记忆编辑 LLM 的执行器。"""

    def __init__(self, *, history_store: MemoryEditorHistoryStore) -> None:
        """初始化管理员记忆编辑管理器。"""

        self._history_store = history_store

    def clear_history(self, admin_user_id: int) -> bool:
        """清空指定管理员的历史上下文。"""

        return self._history_store.clear(int(admin_user_id))

    async def run_turn(self, *, admin_user_id: int, user_text: str) -> str:
        """执行一轮管理员记忆编辑对话。"""

        long_memory = _get_long_memory_manager()
        if long_memory is None:
            raise ValueError("LongMemory 未初始化。")

        plugin_config = get_long_memory_plugin_config()
        editor_cfg = plugin_config.memory_editor
        if not bool(editor_cfg.enable):
            raise ValueError("记忆编辑系统当前已禁用。")

        provider_type, model_id, base_url, api_key, model_parameters = resolve_memory_editor_llm_settings(
            plugin_config=plugin_config
        )
        if not model_id:
            raise ValueError("记忆编辑 LLM 缺少 model_id 配置。")
        if provider_type in {"openai_compatible", "openai_responses", "anthropic"} and not base_url:
            raise ValueError(f"provider_type={provider_type} requires non-empty base_url")

        try:
            model = _build_memory_editor_model(
                provider_type=provider_type,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                model_parameters=model_parameters,
            )
        except Exception as exc:  # noqa: BLE001
            _log_memory_editor_exception(
                "记忆编辑模型初始化失败",
                exc,
                provider_type=provider_type,
                model_id=model_id,
            )
            raise RuntimeError(_format_memory_editor_exception(exc)) from exc

        middleware: list[Any] = []
        if int(editor_cfg.max_tool_calls_per_turn) > 0:
            middleware.append(
                ToolCallLimitMiddleware(
                    run_limit=int(editor_cfg.max_tool_calls_per_turn),
                    exit_behavior="continue",
                )
            )

        agent = create_agent(
            model=model,
            tools=_build_memory_editor_tools(),
            context_schema=MemoryEditorRuntimeContext,
            middleware=middleware,
        )

        history_messages = self._history_store.get_history(int(admin_user_id))
        input_messages: list[BaseMessage] = [
            SystemMessage(content=_memory_editor_prompt()),
            *history_messages,
            HumanMessage(content=str(user_text)),
        ]

        try:
            result = await agent.ainvoke(
                input=cast(Any, {"messages": input_messages}),
                context=MemoryEditorRuntimeContext(long_memory=long_memory),
            )
        except Exception as exc:  # noqa: BLE001
            _log_memory_editor_exception(
                "记忆编辑模型调用失败",
                exc,
                admin_user_id=admin_user_id,
                provider_type=provider_type,
                model_id=model_id,
            )
            raise RuntimeError(_format_memory_editor_exception(exc)) from exc

        output_messages = result.get("messages", []) if isinstance(result, dict) else []
        final_text = ""
        for message in reversed(output_messages):
            if isinstance(message, AIMessage):
                final_text = str(message.content or "").strip()
                break
        if not final_text:
            final_text = "(empty)"

        self._history_store.append_turn(
            admin_user_id=int(admin_user_id),
            user_text=str(user_text),
            assistant_text=final_text,
            history_max_messages=int(editor_cfg.history_max_messages),
        )
        return final_text


_admin_memory_editor_manager = AdminMemoryEditorManager(history_store=_memory_editor_history_store)


def _parse_command_json(raw_text: str) -> dict[str, Any]:
    """解析管理员命令中的 JSON 文本。"""

    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("缺少 JSON 参数。")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("命令参数必须是 JSON 对象。")
    return parsed


async def _ensure_editor_admin(event: MessageEvent) -> None:
    """确保当前用户具备管理员权限。"""

    await require_admin(int(event.user_id))


def _command_success_title(*, name: str, layer: str, target: str) -> str:
    """构造结构化命令的展示标题。"""

    return f"{name}\nlayer={layer}\ntarget={target}"


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    """展开异常链，便于分类底层错误。

    Args:
        exc: 原始异常对象。

    Returns:
        list[BaseException]: 按从外到内顺序展开的异常链列表。
    """

    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(current)
        next_exc = current.__cause__ or current.__context__
        if next_exc is current:
            break
        current = next_exc
    return chain


def _format_memory_editor_exception(exc: BaseException) -> str:
    """将记忆编辑链路异常转换为面向管理员的友好提示。

    Args:
        exc: 捕获到的原始异常。

    Returns:
        str: 适合直接发送到聊天界面的简短错误说明。
    """

    chain = _iter_exception_chain(exc)
    messages = [str(item).strip() for item in chain if str(item).strip()]
    combined_message = " | ".join(messages)

    if any(isinstance(item, (ConnectionResetError, ConnectionAbortedError, socket.error)) for item in chain):
        return "记忆编辑失败：连接到记忆编辑模型时被远程服务中断，请稍后重试；如果持续出现，请检查 memory_editor.llm 的 base_url、api_key 和上游服务状态。"

    lowered = combined_message.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "记忆编辑失败：记忆编辑模型请求超时，请稍后重试；如果经常超时，可以检查上游模型服务是否稳定。"

    if "401" in lowered or "403" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        return "记忆编辑失败：记忆编辑模型鉴权失败，请检查 memory_editor.llm 的 api_key、base_url 和供应商配置是否正确。"

    if "404" in lowered or "not found" in lowered:
        return "记忆编辑失败：记忆编辑模型接口或模型名称不可用，请检查 memory_editor.llm 的 base_url 和 model_id。"

    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return "记忆编辑失败：记忆编辑模型触发了限流或额度限制，请稍后重试，或检查当前模型服务额度。"

    if isinstance(exc, ImportError) or "requires installing" in lowered or "is unavailable" in lowered:
        return f"记忆编辑失败：当前环境缺少记忆编辑模型依赖或 provider 不可用。原始信息：{str(exc).strip() or type(exc).__name__}"

    if isinstance(exc, ValueError):
        return f"记忆编辑失败：{str(exc).strip() or '参数或配置不合法。'}"

    return f"记忆编辑失败：记忆编辑模型调用异常。原始信息：{str(exc).strip() or type(exc).__name__}"


def _log_memory_editor_exception(message: str, exc: BaseException, **context: Any) -> None:
    """记录带完整堆栈的记忆编辑异常日志。

    Args:
        message: 日志前缀说明。
        exc: 捕获到的异常对象。
        **context: 便于排查的附加上下文字段。
    """

    context_parts = [f"{key}={value}" for key, value in context.items()]
    context_text = " ".join(context_parts)
    if context_text:
        logger.exception("%s %s", message, context_text, exc_info=exc)
        return
    logger.exception("%s", message, exc_info=exc)


def _register_memory_editor_help_items() -> None:
    """注册管理员记忆编辑系统的帮助信息。"""

    register_help(
        HelpCommandSpec(
            name="记忆检索",
            category="长期记忆管理",
            summary="按层和 target 检索长期记忆，成功结果以合并转发返回。",
            description=(
                "仅管理员可用。参数必须是 JSON 对象，用于指定 `layer`、`query` 以及可选的 `target`、"
                "`limit`、`extra`。支持的 layer 为 `event_log`、`group_profile`、`user_profile`、"
                "`public_knowledge`。其中 `target` 可省略或写成 `all` 表示全局搜索；也可以显式使用 "
                "`group_<id>`、`private_<id>`、`user:<id>`、`global` 指定范围。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<json>",
                    description="包含 layer、target、query 等字段的 JSON 对象。",
                    value_hint='JSON，如 {"layer":"event_log","query":"涩图"}',
                    example='{"layer":"event_log","query":"涩图","limit":5}',
                ),
            ),
            examples=(
                '/记忆检索 {"layer":"event_log","query":"店员","limit":5}',
                '/记忆检索 {"layer":"event_log","target":"group_123","query":"涩图","limit":5}',
                '/记忆检索 {"layer":"public_knowledge","target":"global","query":"Rust"}',
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=510,
        )
    )
    register_help(
        HelpCommandSpec(
            name="记忆查看",
            category="长期记忆管理",
            summary="按短 ID 精确查看指定记忆条目。",
            description=(
                "仅管理员可用。参数必须是 JSON 对象，至少包含 `layer`、`target`、`short_id`。"
                "`short_id` 可以是单个字符串，也可以是字符串数组；所有短 ID 都会通过内部映射解析，"
                "不会暴露底层 doc_id。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<json>",
                    description="包含 layer、target、short_id 的 JSON 对象。",
                    value_hint='JSON，如 {"layer":"group_profile","target":"group_123","short_id":["G12","G18"]}',
                    example='{"layer":"group_profile","target":"group_123","short_id":["G12","G18"]}',
                ),
            ),
            examples=(
                '/记忆查看 {"layer":"group_profile","target":"group_123","short_id":["G12","G18"]}',
                '/记忆查看 {"layer":"public_knowledge","target":"global","short_id":"P09"}',
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=511,
        )
    )
    register_help(
        HelpCommandSpec(
            name="记忆新建",
            category="长期记忆管理",
            summary="向指定层新增一条长期记忆。",
            description=(
                "仅管理员可用。参数必须是 JSON 对象，至少包含 `layer`、`target`、`payload`。"
                "不同 layer 的 payload 要求不同：`event_log.details`、`user_profile.text`、"
                "`group_profile.text`、`public_knowledge.content` 为必填字段。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<json>",
                    description="包含 layer、target、payload 的 JSON 对象。",
                    value_hint='JSON，如 {"layer":"user_profile","target":"user:123456","payload":{"text":"喜欢 Rust"}}',
                    example='{"layer":"user_profile","target":"user:123456","payload":{"text":"喜欢 Rust"}}',
                ),
            ),
            examples=(
                '/记忆新建 {"layer":"user_profile","target":"user:123456","payload":{"text":"喜欢 Rust"}}',
                '/记忆新建 {"layer":"public_knowledge","target":"global","payload":{"title":"Rust","content":"一门系统编程语言"}}',
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=512,
        )
    )
    register_help(
        HelpCommandSpec(
            name="记忆修改",
            category="长期记忆管理",
            summary="按短 ID 修改现有长期记忆。",
            description=(
                "仅管理员可用。参数必须是 JSON 对象，包含 `layer`、`target`、`short_id`、`payload`。"
                "修改前请先通过 `/记忆检索` 或 `/记忆查看` 确认命中项，避免误改。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<json>",
                    description="包含 layer、target、short_id、payload 的 JSON 对象。",
                    value_hint='JSON，如 {"layer":"public_knowledge","target":"global","short_id":"P09","payload":{"title":"...","content":"..."}}',
                    example='{"layer":"public_knowledge","target":"global","short_id":"P09","payload":{"title":"Rust","content":"更新后的内容"}}',
                ),
            ),
            examples=(
                '/记忆修改 {"layer":"public_knowledge","target":"global","short_id":"P09","payload":{"title":"Rust","content":"更新后的内容"}}',
                '/记忆修改 {"layer":"group_profile","target":"group_123","short_id":"G12","payload":{"category":"规则","text":"禁止刷屏"}}',
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=513,
        )
    )
    register_help(
        HelpCommandSpec(
            name="记忆删除",
            category="长期记忆管理",
            summary="按短 ID 删除一条或多条长期记忆。",
            description=(
                "仅管理员可用。参数必须是 JSON 对象，包含 `layer`、`target`、`short_id`。"
                "`short_id` 可以是单个字符串，也可以是字符串数组；建议先检索或查看，再执行删除。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<json>",
                    description="包含 layer、target、short_id 的 JSON 对象。",
                    value_hint='JSON，如 {"layer":"event_log","target":"private_123456","short_id":["E03","E07"]}',
                    example='{"layer":"event_log","target":"private_123456","short_id":["E03","E07"]}',
                ),
            ),
            examples=(
                '/记忆删除 {"layer":"event_log","target":"private_123456","short_id":["E03","E07"]}',
                '/记忆删除 {"layer":"user_profile","target":"user:123456","short_id":"U05"}',
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=514,
        )
    )
    register_help(
        HelpCommandSpec(
            name="记忆编辑",
            category="长期记忆管理",
            summary="与管理员专用记忆编辑 LLM 单轮对话，按需调用记忆工具。",
            description=(
                "仅管理员可用。该命令只会把“你的问题、编辑 LLM 自己的历史回复、系统提示词、工具调用结果”"
                "送入模型，不会带入群聊上下文、recall、notepad 或持久化会话。历史按管理员全局隔离，"
                "可用 `/清空记忆编辑上下文` 清除。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<自然语言问题>",
                    description="直接描述你要检索、修改、新建或删除哪类记忆；检索时可说全局搜索或指定 target。",
                    value_hint="自然语言",
                    example="帮我全局检索和店员有关的记忆",
                ),
            ),
            examples=(
                "/记忆编辑 帮我全局检索和店员有关的记忆",
                "/记忆编辑 帮我检索 group_123 里和涩图有关的 event_log",
                "/记忆编辑 把 global 里的 P09 标题改成 Rust，并更新正文",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=515,
        )
    )
    register_help(
        HelpCommandSpec(
            name="清空记忆编辑上下文",
            category="长期记忆管理",
            summary="清空当前管理员自己的记忆编辑 LLM 历史上下文。",
            description=(
                "仅管理员可用。该命令只清空当前管理员自己的编辑历史，不影响其他管理员，"
                "也不会删除任何长期记忆数据。"
            ),
            examples=(
                "/清空记忆编辑上下文",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=516,
        )
    )


MemorySearchCommand = on_command("记忆检索", priority=-5, block=True)
MemoryGetCommand = on_command("记忆查看", priority=-5, block=True)
MemoryAddCommand = on_command("记忆新建", priority=-5, block=True)
MemoryUpdateCommand = on_command("记忆修改", priority=-5, block=True)
MemoryDeleteCommand = on_command("记忆删除", priority=-5, block=True)
MemoryEditCommand = on_command("记忆编辑", priority=-5, block=True)
ClearMemoryEditorContextCommand = on_command("清空记忆编辑上下文", priority=-5, block=True)


_register_memory_editor_help_items()


@MemorySearchCommand.handle()
async def handle_memory_search(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆检索命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemorySearchCommand.finish(str(exc))

    try:
        payload = MemoryEditorSearchArgs.model_validate(_parse_command_json(args.extract_plain_text()))
        result_text = await memory_search_impl(
            layer=payload.layer,
            target=payload.target,
            query=payload.query,
            limit=int(payload.limit),
            extra=payload.extra,
        )
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆检索失败", exc, user_id=int(event.user_id))
        await MemorySearchCommand.finish(f"记忆检索失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name=_command_success_title(
            name="记忆检索结果",
            layer=payload.layer,
            target=payload.target or "all",
        ),
        body=result_text,
    )


@MemoryGetCommand.handle()
async def handle_memory_get(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆查看命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemoryGetCommand.finish(str(exc))

    try:
        payload = MemoryEditorGetArgs.model_validate(_parse_command_json(args.extract_plain_text()))
        result_text = await memory_get_impl(layer=payload.layer, target=payload.target, short_id=payload.short_id)
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆查看失败", exc, user_id=int(event.user_id))
        await MemoryGetCommand.finish(f"记忆查看失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name=_command_success_title(name="记忆查看结果", layer=payload.layer, target=payload.target),
        body=result_text,
    )


@MemoryAddCommand.handle()
async def handle_memory_add(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆新建命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemoryAddCommand.finish(str(exc))

    try:
        payload = MemoryEditorAddArgs.model_validate(_parse_command_json(args.extract_plain_text()))
        result_text = await memory_add_impl(layer=payload.layer, target=payload.target, payload=payload.payload)
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆新建失败", exc, user_id=int(event.user_id))
        await MemoryAddCommand.finish(f"记忆新建失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name=_command_success_title(name="记忆新建结果", layer=payload.layer, target=payload.target),
        body=result_text,
    )


@MemoryUpdateCommand.handle()
async def handle_memory_update(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆修改命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemoryUpdateCommand.finish(str(exc))

    try:
        payload = MemoryEditorUpdateArgs.model_validate(_parse_command_json(args.extract_plain_text()))
        result_text = await memory_update_impl(
            layer=payload.layer,
            target=payload.target,
            short_id=payload.short_id,
            payload=payload.payload,
        )
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆修改失败", exc, user_id=int(event.user_id))
        await MemoryUpdateCommand.finish(f"记忆修改失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name=_command_success_title(name="记忆修改结果", layer=payload.layer, target=payload.target),
        body=result_text,
    )


@MemoryDeleteCommand.handle()
async def handle_memory_delete(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆删除命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemoryDeleteCommand.finish(str(exc))

    try:
        payload = MemoryEditorDeleteArgs.model_validate(_parse_command_json(args.extract_plain_text()))
        result_text = await memory_delete_impl(layer=payload.layer, target=payload.target, short_id=payload.short_id)
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆删除失败", exc, user_id=int(event.user_id))
        await MemoryDeleteCommand.finish(f"记忆删除失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name=_command_success_title(name="记忆删除结果", layer=payload.layer, target=payload.target),
        body=result_text,
    )


@MemoryEditCommand.handle()
async def handle_memory_edit(
    bot: Bot,
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理管理员记忆编辑 LLM 命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await MemoryEditCommand.finish(str(exc))

    question = args.extract_plain_text().strip()
    if not question:
        await MemoryEditCommand.finish("用法: /记忆编辑 <自然语言问题>")

    try:
        result_text = await _admin_memory_editor_manager.run_turn(
            admin_user_id=int(event.user_id),
            user_text=question,
        )
    except Exception as exc:  # noqa: BLE001
        _log_memory_editor_exception("管理员记忆编辑失败", exc, user_id=int(event.user_id))
        await MemoryEditCommand.finish(f"记忆编辑失败: {exc!s}")

    await _send_success_result(
        bot=bot,
        event=event,
        command_name="记忆编辑结果",
        body=result_text,
    )


@ClearMemoryEditorContextCommand.handle()
async def handle_clear_memory_editor_context(event: MessageEvent) -> None:
    """处理清空记忆编辑上下文命令。"""

    try:
        await _ensure_editor_admin(event)
    except PermissionError as exc:
        await ClearMemoryEditorContextCommand.finish(str(exc))

    cleared = _admin_memory_editor_manager.clear_history(int(event.user_id))
    if cleared:
        await ClearMemoryEditorContextCommand.finish("已清空当前管理员的记忆编辑上下文。")
    await ClearMemoryEditorContextCommand.finish("当前管理员没有可清空的记忆编辑上下文。")


__all__ = [
    "AdminMemoryEditorManager",
    "MemoryEditorHistoryStore",
    "MemoryEditorRuntimeContext",
    "ResolvedMemoryTarget",
    "memory_add_impl",
    "memory_delete_impl",
    "memory_get_impl",
    "memory_search_impl",
    "memory_update_impl",
    "parse_memory_target",
    "resolve_memory_editor_llm_settings",
]
