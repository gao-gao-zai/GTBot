import time
from collections import defaultdict
from typing import Any, Literal, Protocol, runtime_checkable

from langchain.tools import ToolRuntime, tool



from plugins.GTBot.services.LongMemory import LongMemoryContainer
from .MappingManager import mapping_manager

from .model import EventLog, TimeSlot


MAX_USER_PROFILE_NUMBER = 15
MAX_GROUP_PROFILE_NUMBER = 20


# ========= 事件日志部分（Qdrant，相似度检索） =========


TEST_SESSION_ID: str = "test_session"


@runtime_checkable
class LongMemoryToolContext(Protocol):
    """LongMemory 工具运行上下文（最小依赖集合）。

    说明：
        LangChain 会在调用工具时注入 `runtime`，其中 `runtime.context` 由宿主传入。
        为了避免工具依赖完整的 GroupChatContext/具体聊天框架对象，这里仅约定本工具
        实际会读取到的字段。

    Attributes:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（可选，推荐直接提供）。
        group_id: 群号（可选，CLI/测试环境可用来推断会话）。
        user_id: 用户 ID（可选，私聊/CLI 场景可用来推断会话）。
    """

    long_memory: LongMemoryContainer
    group_id: int | None
    user_id: int | None


def _infer_session_id(context: LongMemoryToolContext) -> str:
    """从上下文推断会话 ID。

    Args:
        context: 工具上下文。

    Returns:
        会话 ID（例如 group_123 / private_456）；无法推断则回退到 TEST_SESSION_ID。
    """

    session_id = getattr(context, "session_id", None)
    if session_id:
        return normalize_session_id(session_id)

    gid = int(getattr(context, "group_id", 0) or 0)
    if gid > 0:
        return normalize_session_id(f"group_{gid}")

    uid = int(getattr(context, "user_id", 0) or 0)
    if uid > 0:
        return normalize_session_id(f"private_{uid}")

    return TEST_SESSION_ID


def normalize_session_id(session_id: str | None) -> str:
    """规范化会话 ID。

    说明：
        旧版工具通过运行时事件对象推断会话 ID。
        为了去除 ToolRuntime 依赖，现改为显式传入 session_id。

        - 当 session_id 为空/空白时，回落到固定测试会话 ID。

    Args:
        session_id: 会话 ID，例如："group_<群号>"、"private_<QQ号>"。

    Returns:
        规范化后的会话 ID。
    """

    sid = (session_id or "").strip()
    return sid or TEST_SESSION_ID


def _parse_time_slots(time_slots: list[dict[str, float]] | None) -> list[TimeSlot]:
    """将 JSON 形式的 time_slots 转为 TimeSlot 列表。

    Args:
        time_slots: 时间段列表，每项包含 start_time/end_time（Unix 时间戳，秒）。

    Returns:
        list[TimeSlot]: 解析后的时间段列表。
    """

    if not time_slots:
        return []

    slots: list[TimeSlot] = []
    for item in time_slots:
        if not isinstance(item, dict):
            continue
        try:
            start_time = float(item.get("start_time", 0.0))
            end_time = float(item.get("end_time", 0.0))
        except Exception:
            continue
        if start_time <= 0.0 or end_time <= 0.0 or end_time < start_time:
            continue
        slots.append(TimeSlot(start_time=start_time, end_time=end_time))
    return slots


async def _impl_add_event_log_info(
    long_memory: LongMemoryContainer,
    session_id: str,
    details: str,
    event_name: str = "",
    relevant_members: list[int] | None = None,
    time_slots: list[dict[str, float]] | None = None,
) -> str:
    """新增事件日志，并返回 short_id。

    说明：
        - 事件日志存储在 Qdrant，底层返回真实 `doc_id`（point id）。
        - 本工具会将 `doc_id` 映射为短 ID（short_id）返回给 LLM，避免泄露数据库 ID。

    Args:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（例如 group_123 / private_456）。
        details: 事件详情（必填，不能为空）。
        event_name: 事件名（可重复，仅作辅助检索/展示）。
        relevant_members: 相关成员 ID（QQ 号）列表；为 None 时默认空列表。
        time_slots: 时间段列表（可选），每项包含 start_time/end_time。

    Returns:
        str: 操作结果描述（包含新增条目的 `short_id`）。
    """

    sid = normalize_session_id(session_id)
    text = str(details).strip()
    if not text:
        return "未写入：details 为空。"

    members = [int(x) for x in (relevant_members or []) if int(x) > 0]
    slots = _parse_time_slots(time_slots)

    doc_id = await long_memory.event_log_manager.add_event(
        EventLog(time_slots=slots, details=text, relevant_members=members, session_id=str(sid)),
        event_name=str(event_name),
    )

    short_id = mapping_manager.get_short_id(layer="event_log", group=str(sid), long_id=str(doc_id))
    return f"已新增事件日志。session_id={sid} short_id={short_id}"


async def _impl_get_event_log_info(
    long_memory: LongMemoryContainer,
    session_id: str,
    short_id: str | list[str],
) -> str:
    """按 short_id 获取事件日志内容。

    注意：
        - 为避免“部分成功、部分失败”导致状态不一致，只要传入 short_id 中任意一个
          无法解析，本工具将拒绝并不执行任何读取。

    Args:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（例如 group_123 / private_456）。
        short_id: 事件条目 short_id（单条或列表）。

    Returns:
        str: 事件日志条目列表（每条包含 short_id、event_name、relevant_members、time_slots、details）。
    """

    sid = normalize_session_id(session_id)

    input_ids: list[str] = [short_id] if isinstance(short_id, str) else [str(x) for x in short_id]
    input_ids = [str(x).strip() for x in input_ids if str(x).strip()]
    if not input_ids:
        return "未获取：short_id 为空。"

    mapped = mapping_manager.get_long_id(layer="event_log", group=str(sid), short_id=input_ids)
    mapped_list: list[str | None] = mapped if isinstance(mapped, list) else [mapped]
    if len(mapped_list) < len(input_ids):
        mapped_list = mapped_list + [None] * (len(input_ids) - len(mapped_list))
    elif len(mapped_list) > len(input_ids):
        mapped_list = mapped_list[: len(input_ids)]

    resolved: list[str] = []
    missing: list[str] = []
    for raw_id, doc_id in zip(input_ids, mapped_list, strict=False):
        if doc_id is None:
            missing.append(raw_id)
            continue
        resolved.append(str(doc_id))

    if missing:
        return (
            f"已拒绝获取：short_id 中存在未找到的条目={missing}。"
            f"为避免部分成功，未执行任何读取操作。"
        )

    try:
        items = await long_memory.event_log_manager.get_by_doc_id(resolved)
    except Exception as exc:  # noqa: BLE001
        return f"未获取：读取事件日志失败：{exc}"

    try:
        await long_memory.event_log_manager.touch_called_time_by_doc_id(resolved)
    except Exception:  # noqa: BLE001
        pass

    items_list = items if isinstance(items, list) else [items]
    lines: list[str] = []
    for raw_sid, item in zip(input_ids, items_list, strict=False):
        members = ",".join(str(int(x)) for x in (item.relevant_members or []))
        slot_text = ";".join(f"{float(s.start_time)}~{float(s.end_time)}" for s in (item.time_slots or []))
        details = str(item.details).replace("\n", " ").strip()
        lines.append(
            f"short_id={raw_sid} event_name={item.event_name} session_id={item.session_id} "
            f"relevant_members=[{members}] time_slots=[{slot_text}] details={details}"
        )

    return "\n".join(lines) if lines else "未找到事件日志。"



async def _impl_search_event_log_info(
    long_memory: LongMemoryContainer,
    session_id: str,
    query: str,
    relevant_members_any: list[int] | None = None,
    limit: int = 5,
    min_similarity: float | None = None,
    return_content: str = "short_id,event_name,similarity,distance,details",
) -> str:
    """按相似度检索事件日志（返回 short_id）。

    Args:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（例如 group_123 / private_456）。
        query: 查询文本。
        relevant_members_any: 相关成员过滤（任意命中）。
        limit: 返回条目数量上限。
        min_similarity: 最小相似度阈值（低于该值的命中会被丢弃）。
        return_content: 返回字段列表（逗号分隔）。可选字段：short_id、event_name、session_id、relevant_members、similarity、distance、details。

    Returns:
        str: 命中结果文本。
    """

    q = str(query).strip()
    if not q:
        return "未检索：query 为空。"

    if int(limit) <= 0:
        return f"未检索：limit 必须为正数，当前={limit}。"

    sid = normalize_session_id(session_id)
    members_any = [int(x) for x in (relevant_members_any or []) if int(x) > 0] or None

    fields_raw = str(return_content).strip().lower()
    return_fields = {x.strip() for x in fields_raw.split(",") if x.strip()} if fields_raw else {
        "short_id",
        "event_name",
        "similarity",
        "distance",
        "details",
    }
    valid_fields = {
        "short_id",
        "event_name",
        "session_id",
        "relevant_members",
        "similarity",
        "distance",
        "details",
    }
    invalid_fields = return_fields - valid_fields
    if invalid_fields:
        return f"未检索：return_content 包含无效字段={invalid_fields}，有效字段={valid_fields}。"

    try:
        hits = await long_memory.event_log_manager.search_events(
            q,
            n_results=int(limit),
            session_id=str(sid),
            relevant_members_any=members_any,
            min_similarity=min_similarity,
            order_by="similarity",
            order="desc",
            touch_last_called=True,
        )
    except Exception as exc:  # noqa: BLE001
        return f"未检索：检索失败：{exc}"

    if not hits:
        return f"未检索到事件日志：query={q}。"

    lines: list[str] = []
    for h in hits:
        doc_id = str(getattr(h, "doc_id", ""))
        short_id = mapping_manager.get_short_id(layer="event_log", group=str(sid), long_id=doc_id)

        parts: list[str] = []
        if "short_id" in return_fields:
            parts.append(f"short_id={short_id}")
        if "event_name" in return_fields:
            parts.append(f"event_name={getattr(h, 'event_name', '')}")
        if "session_id" in return_fields:
            parts.append(f"session_id={getattr(h, 'session_id', '')}")
        if "relevant_members" in return_fields:
            members = getattr(h, "relevant_members", [])
            parts.append(f"relevant_members={members}")
        if "similarity" in return_fields:
            parts.append(f"similarity={float(getattr(h, 'similarity', 0.0) or 0.0):.6f}")
        if "distance" in return_fields:
            parts.append(f"distance={float(getattr(h, 'distance', 0.0) or 0.0):.6f}")
        if "details" in return_fields:
            text = str(getattr(h, "details", "")).replace("\n", " ").strip()
            parts.append(f"details={text}")
        lines.append(" ".join(parts))

    return "\n".join(lines)


async def _impl_update_event_log_info(
    long_memory: LongMemoryContainer,
    session_id: str,
    short_id: str,
    details: str | None = None,
    event_name: str | None = None,
    relevant_members: list[int] | None = None,
    time_slots: list[dict[str, float]] | None = None,
) -> str:
    """更新事件日志（仅接收 short_id）。

    Args:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（例如 group_123 / private_456）。
        short_id: 要更新的事件条目 short_id。
        details: 新的详情内容（可选）。
        event_name: 新的事件名（可选）。
        relevant_members: 新的相关成员列表（可选）。
        time_slots: 新的时间段列表（可选）。

    Returns:
        str: 操作结果描述。
    """

    sid = normalize_session_id(session_id)

    mapped = mapping_manager.get_long_id(layer="event_log", group=str(sid), short_id=str(short_id).strip())
    if mapped is None:
        return f"未更新：未找到对应的事件条目，short_id={short_id}。"

    update_kwargs: dict[str, Any] = {"last_updated": time.time()}
    if details is not None:
        update_kwargs["details"] = str(details)
    if event_name is not None:
        update_kwargs["event_name"] = str(event_name)
    if relevant_members is not None:
        update_kwargs["relevant_members"] = [int(x) for x in relevant_members if int(x) > 0]
    if time_slots is not None:
        update_kwargs["time_slots"] = _parse_time_slots(time_slots)

    if len(update_kwargs) == 1:
        return "未更新：未提供任何可更新字段。"

    try:
        updated = await long_memory.event_log_manager.update_by_doc_id(str(mapped), **update_kwargs)
    except Exception as exc:  # noqa: BLE001
        return f"未更新：更新失败：{exc}"

    if int(updated) <= 0:
        return f"未更新：事件条目 short_id={short_id} 不存在或内容无变化。"
    return f"已更新事件条目，session_id={sid} short_id={short_id}。"


async def _impl_delete_event_log_info(
    long_memory: LongMemoryContainer,
    session_id: str,
    short_id: str | list[str],
) -> str:
    """删除事件日志（仅接收 short_id）。

    注意：
        - 为了避免“部分成功、部分失败”造成状态不一致，只要传入的 short_id 中
          任意一个不存在/无法解析，本工具将直接拒绝并不执行任何删除。

    Args:
        long_memory: LongMemory 服务容器。
        session_id: 会话 ID（例如 group_123 / private_456）。
        short_id: 要删除的事件条目 short_id（单条或列表）。

    Returns:
        str: 操作结果描述。
    """

    sid = normalize_session_id(session_id)

    input_ids: list[str] = [short_id] if isinstance(short_id, str) else [str(x) for x in short_id]
    input_ids = [str(x).strip() for x in input_ids if str(x).strip()]
    if not input_ids:
        return "未删除：short_id 为空。"

    mapped = mapping_manager.get_long_id(layer="event_log", group=str(sid), short_id=input_ids)
    mapped_list: list[str | None] = mapped if isinstance(mapped, list) else [mapped]
    if len(mapped_list) < len(input_ids):
        mapped_list = mapped_list + [None] * (len(input_ids) - len(mapped_list))
    elif len(mapped_list) > len(input_ids):
        mapped_list = mapped_list[: len(input_ids)]

    resolved: list[str] = []
    missing: list[str] = []
    for raw_id, doc_id in zip(input_ids, mapped_list, strict=False):
        if doc_id is None:
            missing.append(raw_id)
            continue
        resolved.append(str(doc_id))

    if missing:
        return (
            f"已拒绝删除：short_id 中存在未找到的条目={missing}。"
            f"为避免部分删除，未执行任何删除操作。"
        )

    try:
        deleted = await long_memory.event_log_manager.delete_by_doc_id(resolved)
    except Exception as exc:  # noqa: BLE001
        return f"未删除：删除失败：{exc}"
    return f"已删除事件日志条目，session_id={sid} short_id={input_ids}，删除数={deleted}。"


# ========= 用户画像部分 =========

async def _impl_add_user_profile_info(
    long_memory: LongMemoryContainer,
    user_id: int,
    info: list[str] | str,
) -> str:
    """向用户画像中添加信息，并返回 short_id。

    说明：
        - 本工具会把真实数据库 doc_id（Qdrant point id）映射为短 ID（short_id）。
        - 返回给 LLM 的只会是 short_id；数据库 doc_id 对 LLM 隐藏。

    Args:
        long_memory: LongMemory 服务容器。
        user_id: 用户 ID（QQ 号）。
        info: 要添加的信息或信息列表。

    Returns:
        str: 操作结果描述（包含新增画像条目的 `short_id` 列表）。
    """
    lenght: int = await long_memory.user_profile_manager.count_user_profile_descriptions(user_id)
    if lenght + (len(info) if isinstance(info, list) else 1) > MAX_USER_PROFILE_NUMBER:
        return f"未写入：用户 {user_id} 的画像已达上限（{MAX_USER_PROFILE_NUMBER} 条）。"
    
    doc_ids: list[str] = await long_memory.user_profile_manager.add_user_profile(
        user_id=user_id,
        profile_texts=info
    )
    if not doc_ids:
        return f"未写入：用户 {user_id} 的画像内容为空。"

    short_ids = mapping_manager.get_short_id(
        layer="user_profile",
        group=str(user_id),
        long_id=doc_ids,
    )

    return f"已向用户 {user_id} 的画像中添加信息。short_id={short_ids}"

async def _impl_delete_user_profile_info(
    long_memory: LongMemoryContainer,
    user_id: int,
    short_id: str | list[str],
) -> str:
    """删除用户画像中的指定信息（仅接收 short_id）。

    注意：
        - 为了避免“部分成功、部分失败”造成状态不一致，只要传入的 short_id 中
          任意一个不存在/无法解析，本工具将直接拒绝并不执行任何删除。

    Args:
        long_memory: LongMemory 服务容器。
        user_id: 用户 ID（QQ 号）。
        short_id: 要删除的画像条目 short_id（单条或列表）。

            - short_id 由 `add_user_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。
    Returns:
        str: 操作结果描述。
    """
    input_ids: list[str] = [short_id] if isinstance(short_id, str) else [str(x) for x in short_id]
    input_ids = [str(x).strip() for x in input_ids if str(x).strip()]
    if not input_ids:
        return "未删除：short_id 为空。"

    mapped = mapping_manager.get_long_id(
        layer="user_profile",
        group=str(user_id),
        short_id=input_ids,
    )

    # mapped 在 list 输入下应为 list[Optional[str]]，这里做健壮处理。
    mapped_list: list[str | None]
    if isinstance(mapped, list):
        mapped_list = mapped
    else:
        mapped_list = [mapped]

    # 健壮性：若映射长度与输入不一致，按缺失处理。
    if len(mapped_list) < len(input_ids):
        mapped_list = mapped_list + [None] * (len(input_ids) - len(mapped_list))
    elif len(mapped_list) > len(input_ids):
        mapped_list = mapped_list[: len(input_ids)]

    resolved_long_ids: list[str] = []
    missing_short_ids: list[str] = []

    for raw_id, long_id in zip(input_ids, mapped_list, strict=False):
        if long_id is None:
            missing_short_ids.append(raw_id)
            continue
        resolved_long_ids.append(long_id)

    # 只要有任意 short_id 无法解析，则拒绝并不执行任何删除。
    if missing_short_ids:
        return (
            f"已拒绝删除：short_id 中存在未找到的条目={missing_short_ids}。"
            f"为避免部分删除，未执行任何删除操作。"
        )

    if not resolved_long_ids:
        return f"未删除：未找到对应的画像条目，short_id={input_ids}。"

    deleted = await long_memory.user_profile_manager.delete_by_doc_id(resolved_long_ids)
    return f"已删除用户 {user_id} 的画像条目，short_id={input_ids}，删除数={deleted}。"

async def _impl_update_user_profile_info(
    long_memory: LongMemoryContainer,
    user_id: int,
    short_id: str,
    new_info: str,
) -> str:
    """更新用户画像中的指定信息（仅接收 short_id）。

    Args:
        long_memory: LongMemory 服务容器。
        user_id: 用户 ID（QQ 号）。
        short_id: 要更新的画像条目 short_id。

            - short_id 由 `add_user_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。
        new_info: 新的信息内容。
    Returns:
        str: 操作结果描述。
    """
    mapped = mapping_manager.get_long_id(
        layer="user_profile",
        group=str(user_id),
        short_id=short_id,
    )

    if mapped is None:
        return f"未更新：未找到对应的画像条目，short_id={short_id}。"

    updated = await long_memory.user_profile_manager.update_by_doc_id(
        doc_id=mapped,
        descriptions=new_info,
        last_updated=time.time(),
        
    )
    if not updated:
        return f"未更新：用户 {user_id} 的画像条目 short_id={short_id} 内容无变化。"

    return f"已更新用户 {user_id} 的画像条目，short_id={short_id}。"

async def _impl_get_user_profile_info(
    long_memory: LongMemoryContainer,
    user_id: int | list[int],
) -> str:
    """获取用户画像（返回 short_id，不泄露 doc_id）。

    Args:
        long_memory: LongMemory 服务容器。
        user_id: 用户 ID（QQ 号）或用户 ID 列表。

    Returns:
        str: 用户画像信息（每条包含 short_id 与文本内容）。
    """

    profiles = await long_memory.user_profile_manager.get_user_profiles(user_id=user_id, limit=MAX_USER_PROFILE_NUMBER)

    profiles_list = profiles if isinstance(profiles, list) else [profiles]

    lines: list[str] = []
    for p in profiles_list:
        doc_ids = [x.doc_id for x in p.description]
        short_ids = mapping_manager.get_short_id(
            layer="user_profile",
            group=str(p.id),
            long_id=doc_ids,
            
        )
        short_ids_list = short_ids if isinstance(short_ids, list) else [short_ids]

        lines.append(f"用户 {p.id} 画像条目数={len(p.description)}")
        for sid, item in zip(short_ids_list, p.description, strict=False):
            text = str(item.text).replace("\n", " ").strip()
            lines.append(f"- short_id={sid} text={text}")

    return "\n".join(lines) if lines else "未找到用户画像。"

async def _impl_search_user_profile_info(
    long_memory: LongMemoryContainer,
    query: str,
    max_users: int = 5,
    limit: int = 10,
    mode: Literal["expand", "direct"] = "direct",
    return_content: str = "user_id,short_id,text",
    similarity_threshold: float = 0.0,
) -> str:
    """跨用户搜索用户画像（支持 direct / expand 两种返回方式）。    

    该工具会在全库用户画像中执行向量检索，并将命中结果按 `user_id` 聚合：

    - 先找出与 `query` 最相近的“最多 N 个用户”（以该用户命中中的最高 `similarity` 作为排序依据）。
    - 对每个用户返回最多 `limit` 条最相近的画像描述。
    - 返回内容包含 `similarity`/`distance` 作为参考，并将数据库 `doc_id` 映射为 `short_id`，避免泄露真实 ID。

    Args:
        long_memory: LongMemory 服务容器。
        query: 查询文本。
        max_users: 最大返回用户数。
        limit: 每个用户返回的画像条目数量上限。
        mode: 检索模式（默认：direct）。
            - direct：仅返回全库相似度最高的命中条目列表；不做按用户聚合。该模式下 `max_users` 无效，只有 `limit` 生效。
            - expand：先向量检索，再按 user_id 聚合并返回每个用户的多条命中。
        return_content: 返回字段列表（逗号分隔）。可选字段：user_id、short_id、similarity、text、distance、threshold。
            默认："user_id,short_id,text"（精简模式）。
            完整："user_id,short_id,similarity,distance,text,threshold"。
        similarity_threshold: 相似度阈值（低于该值的命中会被丢弃）。

    Returns:
        str: 聚合后的检索结果文本。
    """

    q = str(query).strip()
    if not q:
        return "未检索：query 为空。"

    if limit <= 0:
        return f"未检索：limit 必须为正数，当前={limit}。"

    mode_value = str(mode).strip().lower()
    if mode_value not in {"expand", "direct"}:
        return f"未检索：mode 必须为 expand/direct，当前={mode!r}。"

    # 解析返回字段列表
    fields_raw = str(return_content).strip().lower()
    if not fields_raw:
        return_fields = {"user_id", "short_id", "text"}
    else:
        return_fields = set(x.strip() for x in fields_raw.split(",") if x.strip())
    
    valid_fields = {"user_id", "short_id", "similarity", "text", "distance", "threshold"}
    invalid_fields = return_fields - valid_fields
    if invalid_fields:
        return f"未检索：return_content 包含无效字段={invalid_fields}，有效字段={valid_fields}。"

    max_user_count = int(max_users)
    if mode_value == "expand" and max_user_count <= 0:
        return f"未检索：max_users 必须为正数，当前={max_user_count}。"

    threshold = float(similarity_threshold)
    if threshold < 0.0 or threshold > 1.0:
        return f"未检索：similarity_threshold 必须在 [0, 1]，当前={threshold}。"

    def _format_item(
        uid: int,
        sid: str,
        sim: float,
        dist: float,
        text: str,
        threshold_val: float,
        fields: set[str],
    ) -> str:
        """根据选中的字段格式化单条命中。"""
        parts = []
        if "user_id" in fields:
            parts.append(f"user_id={uid}")
        if "short_id" in fields:
            parts.append(f"short_id={sid}")
        if "similarity" in fields:
            parts.append(f"similarity={sim:.6f}")
        if "distance" in fields:
            parts.append(f"distance={dist:.6f}")
        if "text" in fields:
            parts.append(f"text={text}")
        if "threshold" in fields:
            parts.append(f"threshold={threshold_val}")
        return " ".join(parts)

    def _as_short_id_list(raw: Any, *, expected: int) -> list[str]:
        if isinstance(raw, list):
            out = [str(s) for s in raw]
        else:
            out = [str(raw)]
        if len(out) < expected:
            out = out + ["<unknown>"] * (expected - len(out))
        elif len(out) > expected:
            out = out[:expected]
        return out

    if mode_value == "direct":
        # direct 模式：仅返回相似度最高的若干条命中（不按用户分组）。
        # 为了在阈值过滤后仍能返回足够条目，取一定放大倍数。
        n_results = max(1, int(limit) * 5)
        hits = await long_memory.user_profile_manager.search_user_profiles(
            q,
            n_results=n_results,
            order_by="similarity",
            order="desc",
        )
        if not hits:
            return f"未检索到用户画像：query={q}。"

        filtered: list[Any] = []
        for h in hits:
            sim = float(getattr(h, "similarity", 0.0) or 0.0)
            if sim < threshold:
                continue
            filtered.append(h)
            if len(filtered) >= int(limit):
                break

        if not filtered:
            return f"未检索到有效命中：query={q}。"

        lines: list[str] = []
        for h in filtered:
            uid = int(getattr(h, "user_id", 0) or 0)
            did = str(getattr(h, "doc_id", ""))
            
            # 直接对单条命中映射 short_id
            short_id = mapping_manager.get_short_id(layer="user_profile", group=str(uid), long_id=did)
            sid = str(short_id) if short_id else "<unknown>"
            
            sim = float(getattr(h, "similarity", 0.0) or 0.0)
            dist = float(getattr(h, "distance", 0.0) or 0.0)
            text = str(getattr(h, "text", "")).replace("\n", " ").strip()
            
            line = _format_item(uid, sid, sim, dist, text, threshold, return_fields)
            lines.append(line)
        
        return "\n".join(lines)

    # expand（当前逻辑）：向量检索后按 user_id 聚合返回。
    n_results = max(1, int(max_user_count) * int(limit) * 3)
    hits = await long_memory.user_profile_manager.search_user_profiles(
        q,
        n_results=n_results,
        order_by="similarity",
        order="desc",
    )

    if not hits:
        return f"未检索到用户画像：query={q}。"

    hits_by_user: dict[int, list[Any]] = defaultdict(list)
    for h in hits:
        uid = int(getattr(h, "user_id", 0) or 0)
        if uid <= 0:
            continue
        sim = float(getattr(h, "similarity", 0.0) or 0.0)
        if sim < threshold:
            continue
        hits_by_user[uid].append(h)

    if not hits_by_user:
        return f"未检索到有效命中：query={q}。"

    user_rank: list[tuple[int, float]] = []
    for uid, user_hits in hits_by_user.items():
        best = max(float(getattr(x, "similarity", 0.0) or 0.0) for x in user_hits)
        user_rank.append((uid, best))
    user_rank.sort(key=lambda x: x[1], reverse=True)
    top_users = [uid for uid, _ in user_rank[: int(max_user_count)]]

    lines: list[str] = []
    for uid in top_users:
        user_hits = hits_by_user.get(uid, [])
        user_hits.sort(key=lambda x: float(getattr(x, "similarity", 0.0) or 0.0), reverse=True)
        user_hits = user_hits[: int(limit)]

        doc_ids: list[str] = [str(getattr(x, "doc_id", "")) for x in user_hits]
        short_ids = mapping_manager.get_short_id(layer="user_profile", group=str(uid), long_id=doc_ids)
        short_ids_list = _as_short_id_list(short_ids, expected=len(user_hits))

        for sid, h in zip(short_ids_list, user_hits, strict=False):
            sim = float(getattr(h, "similarity", 0.0) or 0.0)
            dist = float(getattr(h, "distance", 0.0) or 0.0)
            text = str(getattr(h, "text", "")).replace("\n", " ").strip()
            line = _format_item(uid, sid, sim, dist, text, threshold, return_fields)
            lines.append(line)

    return "\n".join(lines)


# ========= 群画像部分（SQLite，无相似度检索） =========


async def _impl_add_group_profile_info(
    long_memory: LongMemoryContainer,
    group_id: int,
    info: list[str] | str,
    category: str | None = None,
) -> str:
    """向群画像中添加信息，并返回 short_id。

    说明：
        - 群画像使用 SQLite 存储，`doc_id` 为自增主键。
        - 本工具会把数据库 `doc_id` 映射为短 ID（short_id），避免对 LLM 暴露真实 ID。

    Args:
        long_memory: LongMemory 服务容器。
        group_id: 群 ID（QQ群号）。
        info: 要添加的信息或信息列表。
        category: 可选分类。

    Returns:
        str: 操作结果描述（包含新增条目的 `short_id` 列表）。
    """

    current: int = await long_memory.group_profile_manager.count_group_profile_descriptions(int(group_id))
    add_count: int = len(info) if isinstance(info, list) else 1
    if current + add_count > MAX_GROUP_PROFILE_NUMBER:
        return f"未写入：群 {group_id} 的画像已达上限（{MAX_GROUP_PROFILE_NUMBER} 条）。"

    doc_ids: list[str] = await long_memory.group_profile_manager.add_group_profile(
        group_id=int(group_id),
        profile_texts=info,
        category=category,
    )
    if not doc_ids:
        return f"未写入：群 {group_id} 的画像内容为空。"

    short_ids = mapping_manager.get_short_id(
        layer="group_profile",
        group=str(group_id),
        long_id=doc_ids,
    )
    return f"已向群 {group_id} 的画像中添加信息。short_id={short_ids}"


async def _impl_delete_group_profile_info(
    long_memory: LongMemoryContainer,
    group_id: int,
    short_id: str | list[str],
) -> str:
    """删除群画像中的指定信息（仅接收 short_id）。

    注意：
        - 为了避免“部分成功、部分失败”造成状态不一致，只要传入的 short_id 中
          任意一个不存在/无法解析，本工具将直接拒绝并不执行任何删除。

    Args:
        long_memory: LongMemory 服务容器。
        group_id: 群 ID（QQ群号）。
        short_id: 要删除的画像条目 short_id（单条或列表）。

            - short_id 由 `add_group_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。

    Returns:
        str: 操作结果描述。
    """

    input_ids: list[str] = [short_id] if isinstance(short_id, str) else [str(x) for x in short_id]
    input_ids = [str(x).strip() for x in input_ids if str(x).strip()]
    if not input_ids:
        return "未删除：short_id 为空。"

    mapped = mapping_manager.get_long_id(
        layer="group_profile",
        group=str(group_id),
        short_id=input_ids,
    )

    mapped_list: list[str | None]
    if isinstance(mapped, list):
        mapped_list = mapped
    else:
        mapped_list = [mapped]

    if len(mapped_list) < len(input_ids):
        mapped_list = mapped_list + [None] * (len(input_ids) - len(mapped_list))
    elif len(mapped_list) > len(input_ids):
        mapped_list = mapped_list[: len(input_ids)]

    resolved_doc_ids: list[str] = []
    missing_short_ids: list[str] = []
    for raw_id, doc_id in zip(input_ids, mapped_list, strict=False):
        if doc_id is None:
            missing_short_ids.append(raw_id)
            continue
        resolved_doc_ids.append(str(doc_id))

    if missing_short_ids:
        return (
            f"已拒绝删除：short_id 中存在未找到的条目={missing_short_ids}。"
            f"为避免部分删除，未执行任何删除操作。"
        )

    if not resolved_doc_ids:
        return f"未删除：未找到对应的群画像条目，short_id={input_ids}。"

    # 二次校验：只允许删除当前群内真实存在的 doc_id。
    existing = await long_memory.group_profile_manager.get_existing_doc_ids(
        int(group_id),
        resolved_doc_ids,
    )
    missing_doc_ids = [d for d in resolved_doc_ids if d not in existing]
    if missing_doc_ids:
        return (
            f"已拒绝删除：存在已失效/不存在的条目（doc_id）={missing_doc_ids}。"
            f"为避免部分删除，未执行任何删除操作。"
        )

    deleted_count = await long_memory.group_profile_manager.delete_many_by_doc_id(
        int(group_id),
        resolved_doc_ids,
    )
    return f"已删除群 {group_id} 的画像条目，short_id={input_ids}，删除数={deleted_count}。"


async def _impl_update_group_profile_info(
    long_memory: LongMemoryContainer,
    group_id: int,
    short_id: str,
    new_info: str,
    category: str | None = None,
) -> str:
    """更新群画像中的指定信息（仅接收 short_id）。

    Args:
        long_memory: LongMemory 服务容器。
        group_id: 群 ID（QQ群号）。
        short_id: 要更新的画像条目 short_id。

            - short_id 由 `add_group_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。
        new_info: 新的信息内容。
        category: 可选分类；
            - None：不更新分类；
            - 空字符串：清空分类；
            - 其他：更新为指定分类。

    Returns:
        str: 操作结果描述。
    """

    mapped = mapping_manager.get_long_id(
        layer="group_profile",
        group=str(group_id),
        short_id=str(short_id).strip(),
    )
    if mapped is None:
        return f"未更新：未找到对应的群画像条目，short_id={short_id}。"

    updated = await long_memory.group_profile_manager.update_by_doc_id(
        str(mapped),
        description=str(new_info),
        category=category,
        last_updated=time.time(),
    )
    if not updated:
        return f"未更新：群 {group_id} 的画像条目 short_id={short_id} 不存在或内容无变化。"

    return f"已更新群 {group_id} 的画像条目，short_id={short_id}。"


async def _impl_get_group_profile_info(
    long_memory: LongMemoryContainer,
    group_id: int | list[int],
    limit: int = MAX_GROUP_PROFILE_NUMBER,
) -> str:
    """获取群画像（返回 short_id，不泄露 doc_id）。

    Args:
        long_memory: LongMemory 服务容器。
        group_id: 群 ID（QQ群号）或群 ID 列表。
        limit: 每个群最多返回的条目数。

    Returns:
        str: 群画像信息（每条包含 short_id 与文本内容）。
    """

    group_ids = [int(group_id)] if isinstance(group_id, int) else [int(x) for x in group_id]
    group_ids = [gid for gid in group_ids if gid > 0]
    if not group_ids:
        return "未获取：group_id 为空。"

    lines: list[str] = []
    for gid in group_ids:
        profile = await long_memory.group_profile_manager.get_group_profiles(
            int(gid),
            limit=int(limit) if int(limit) > 0 else MAX_GROUP_PROFILE_NUMBER,
            sort_by="last_updated",
            sort_order="desc",
            touch_read_time=False,
        )

        doc_ids = [x.doc_id for x in profile.description]
        short_ids = mapping_manager.get_short_id(
            layer="group_profile",
            group=str(gid),
            long_id=doc_ids,
        )
        short_ids_list = short_ids if isinstance(short_ids, list) else [short_ids]

        lines.append(f"群 {gid} 画像条目数={len(profile.description)}")
        for sid, item in zip(short_ids_list, profile.description, strict=False):
            text = str(item.text).replace("\n", " ").strip()
            lines.append(f"- short_id={sid} text={text}")

    return "\n".join(lines) if lines else "未找到群画像。"



# ========================
# 直接暴露给 LangChain 的工具
# ========================


def _get_long_memory_from_runtime(runtime: ToolRuntime[LongMemoryToolContext]) -> LongMemoryContainer:
    """从 runtime 中获取 LongMemory 容器。

    Args:
        runtime: LangChain 工具运行时。

    Returns:
        LongMemoryContainer: LongMemory 服务容器。

    Raises:
        ValueError: runtime.context 未提供 long_memory。
    """

    long_memory = getattr(getattr(runtime, "context", None), "long_memory", None)
    if long_memory is None:
        raise ValueError("runtime.context.long_memory 为空，无法使用 LongMemory 工具")
    return long_memory


def _get_session_id_from_runtime(runtime: ToolRuntime[LongMemoryToolContext]) -> str:
    """从 runtime 中推断会话 ID。

    Args:
        runtime: LangChain 工具运行时。

    Returns:
        str: 会话 ID（例如 group_123 / private_456）。
    """

    return _infer_session_id(runtime.context)


@tool("add_user_profile_info")
async def add_user_profile_info(user_id: int, info: list[str] | str, runtime: ToolRuntime[LongMemoryToolContext]) -> str:
    """向用户画像中添加信息，并返回 short_id。

    Args:
        user_id: 用户 ID（QQ 号）。
        info: 要添加的信息或信息列表。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_add_user_profile_info(_get_long_memory_from_runtime(runtime), user_id=user_id, info=info)


@tool("delete_user_profile_info")
async def delete_user_profile_info(
    user_id: int,
    short_id: str | list[str],
    runtime: ToolRuntime[LongMemoryToolContext],
) -> str:
    """删除用户画像中的指定信息（仅接收 short_id）。

    Args:
        user_id: 用户 ID（QQ 号）。
        short_id: 要删除的画像条目 short_id（单条或列表）。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_delete_user_profile_info(
        _get_long_memory_from_runtime(runtime),
        user_id=user_id,
        short_id=short_id,
    )


@tool("update_user_profile_info")
async def update_user_profile_info(
    user_id: int,
    short_id: str,
    new_info: str,
    runtime: ToolRuntime[LongMemoryToolContext],
) -> str:
    """更新用户画像中的指定信息（仅接收 short_id）。

    Args:
        user_id: 用户 ID（QQ 号）。
        short_id: 要更新的画像条目 short_id。
        new_info: 新的信息内容。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_update_user_profile_info(
        _get_long_memory_from_runtime(runtime),
        user_id=user_id,
        short_id=short_id,
        new_info=new_info,
    )


@tool("get_user_profile_info")
async def get_user_profile_info(user_id: int | list[int], runtime: ToolRuntime[LongMemoryToolContext]) -> str:
    """获取用户画像（返回 short_id，不泄露 doc_id）。

    Args:
        user_id: 用户 ID（QQ 号）或用户 ID 列表。

    Returns:
        str: 用户画像信息。
    """

    return await _impl_get_user_profile_info(_get_long_memory_from_runtime(runtime), user_id=user_id)


@tool("search_user_profile_info")
async def search_user_profile_info(
    query: str,
    runtime: ToolRuntime[LongMemoryToolContext],
    max_users: int = 5,
    limit: int = 10,
    mode: Literal["expand", "direct"] = "direct",
    return_content: str = "user_id,short_id,text",
    similarity_threshold: float = 0.0,
) -> str:
    """跨用户搜索用户画像（支持 direct / expand 两种返回方式）。

    Args:
        query: 查询文本。
        max_users: 最大返回用户数。
        limit: 每个用户返回的画像条目数量上限。
        mode: 检索模式（expand/direct）。
        return_content: 返回字段列表（逗号分隔）。
        similarity_threshold: 相似度阈值（低于该值的命中会被丢弃）。

    Returns:
        str: 检索结果文本。
    """

    return await _impl_search_user_profile_info(
        _get_long_memory_from_runtime(runtime),
        query=query,
        max_users=max_users,
        limit=limit,
        mode=mode,
        return_content=return_content,
        similarity_threshold=similarity_threshold,
    )


@tool("add_group_profile_info")
async def add_group_profile_info(
    group_id: int,
    info: list[str] | str,
    runtime: ToolRuntime[LongMemoryToolContext],
    category: str | None = None,
) -> str:
    """向群画像中添加信息，并返回 short_id。

    Args:
        group_id: 群号。
        info: 要添加的信息或信息列表。
        category: 可选的分类字段。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_add_group_profile_info(
        _get_long_memory_from_runtime(runtime),
        group_id=group_id,
        info=info,
        category=category,
    )


@tool("delete_group_profile_info")
async def delete_group_profile_info(
    group_id: int,
    short_id: str | list[str],
    runtime: ToolRuntime[LongMemoryToolContext],
) -> str:
    """删除群画像中的指定信息（仅接收 short_id）。

    Args:
        group_id: 群号。
        short_id: 要删除的画像条目 short_id（单条或列表）。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_delete_group_profile_info(
        _get_long_memory_from_runtime(runtime),
        group_id=group_id,
        short_id=short_id,
    )


@tool("update_group_profile_info")
async def update_group_profile_info(
    group_id: int,
    short_id: str,
    new_info: str,
    runtime: ToolRuntime[LongMemoryToolContext],
    category: str | None = None,
) -> str:
    """更新群画像中的指定信息（仅接收 short_id）。

    Args:
        group_id: 群号。
        short_id: 要更新的画像条目 short_id。
        new_info: 新的信息内容。
        category: 可选分类字段。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_update_group_profile_info(
        _get_long_memory_from_runtime(runtime),
        group_id=group_id,
        short_id=short_id,
        new_info=new_info,
        category=category,
    )


@tool("get_group_profile_info")
async def get_group_profile_info(
    group_id: int | list[int],
    runtime: ToolRuntime[LongMemoryToolContext],
    limit: int = MAX_GROUP_PROFILE_NUMBER,
) -> str:
    """获取群画像（返回 short_id，不泄露 doc_id）。

    Args:
        group_id: 群号或群号列表。
        limit: 每个群返回条目数量上限。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 群画像信息。
    """

    return await _impl_get_group_profile_info(
        _get_long_memory_from_runtime(runtime),
        group_id=group_id,
        limit=limit,
    )


@tool("add_event_log_info")
async def add_event_log_info(
    details: str,
    runtime: ToolRuntime[LongMemoryToolContext],
    event_name: str = "",
    relevant_members: list[int] | None = None,
    time_slots: list[dict[str, float]] | None = None,
) -> str:
    """新增事件日志，并返回 short_id。

    说明：
        会话 ID 会从 `runtime.context` 推断：优先使用 `session_id`（若存在），否则使用 group_id/user_id。

    Args:
        details: 事件详情。
        event_name: 可选事件名称。
        relevant_members: 相关成员 ID 列表。
        time_slots: 可选的时间片列表。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_add_event_log_info(
        _get_long_memory_from_runtime(runtime),
        session_id=_get_session_id_from_runtime(runtime),
        details=details,
        event_name=event_name,
        relevant_members=relevant_members,
        time_slots=time_slots,
    )


@tool("get_event_log_info")
async def get_event_log_info(
    short_id: str | list[str],
    runtime: ToolRuntime[LongMemoryToolContext],
) -> str:
    """按 short_id 获取事件日志内容。

    Args:
        short_id: 事件日志条目 short_id（单条或列表）。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 事件日志内容。
    """

    return await _impl_get_event_log_info(
        _get_long_memory_from_runtime(runtime),
        session_id=_get_session_id_from_runtime(runtime),
        short_id=short_id,
    )


@tool("search_event_log_info")
async def search_event_log_info(
    query: str,
    runtime: ToolRuntime[LongMemoryToolContext],
    relevant_members_any: list[int] | None = None,
    limit: int = 5,
    min_similarity: float | None = None,
    return_content: str = "short_id,event_name,similarity,distance,details",
) -> str:
    """按相似度检索事件日志（返回 short_id）。

    Args:
        query: 查询文本。
        relevant_members_any: 相关成员（任意命中）。
        limit: 返回条目数量。
        min_similarity: 最低相似度阈值。
        return_content: 返回字段列表（逗号分隔）。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 检索结果。
    """

    return await _impl_search_event_log_info(
        _get_long_memory_from_runtime(runtime),
        session_id=_get_session_id_from_runtime(runtime),
        query=query,
        relevant_members_any=relevant_members_any,
        limit=limit,
        min_similarity=min_similarity,
        return_content=return_content,
    )


@tool("update_event_log_info")
async def update_event_log_info(
    short_id: str,
    runtime: ToolRuntime[LongMemoryToolContext],
    details: str | None = None,
    event_name: str | None = None,
    relevant_members: list[int] | None = None,
    time_slots: list[dict[str, float]] | None = None,
) -> str:
    """更新事件日志（仅接收 short_id）。

    Args:
        short_id: 要更新的事件日志条目 short_id。
        details: 可选的新详情。
        event_name: 可选的新事件名。
        relevant_members: 可选的新相关成员列表。
        time_slots: 可选的新时间片。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_update_event_log_info(
        _get_long_memory_from_runtime(runtime),
        session_id=_get_session_id_from_runtime(runtime),
        short_id=short_id,
        details=details,
        event_name=event_name,
        relevant_members=relevant_members,
        time_slots=time_slots,
    )


@tool("delete_event_log_info")
async def delete_event_log_info(
    short_id: str | list[str],
    runtime: ToolRuntime[LongMemoryToolContext],
) -> str:
    """删除事件日志（仅接收 short_id）。

    Args:
        short_id: 要删除的事件日志条目 short_id（单条或列表）。
        runtime: LangChain 工具运行时上下文。

    Returns:
        str: 操作结果描述。
    """

    return await _impl_delete_event_log_info(
        _get_long_memory_from_runtime(runtime),
        session_id=_get_session_id_from_runtime(runtime),
        short_id=short_id,
    )