import time
from collections import defaultdict
from typing import Any, Literal

from langchain.tools import tool, ToolRuntime



from plugins.GTBot.services.LongMemory import LongMemoryManager
from .MappingManager import mapping_manager

from ...GroupChatContext import GroupChatContext


MAX_USER_PROFILE_NUMBER = 15

# ========= 用户画像部分 =========

@tool()
async def add_user_profile_info(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int,
    info: list[str] | str,
) -> str:
    """向用户画像中添加信息，并返回 short_id。

    说明：
        - 本工具会把真实数据库 doc_id（Qdrant point id）映射为短 ID（short_id）。
        - 返回给 LLM 的只会是 short_id；数据库 doc_id 对 LLM 隐藏。

    Args:
        runtime: 工具运行时上下文。
        user_id: 用户 ID（QQ 号）。
        info: 要添加的信息或信息列表。

    Returns:
        str: 操作结果描述（包含新增画像条目的 `short_id` 列表）。
    """
    long_memory: LongMemoryManager = runtime.context.long_memory
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

@tool()
async def delete_user_profile_info(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int,
    short_id: str | list[str],
) -> str:
    """删除用户画像中的指定信息（仅接收 short_id）。

    注意：
        - 为了避免“部分成功、部分失败”造成状态不一致，只要传入的 short_id 中
          任意一个不存在/无法解析，本工具将直接拒绝并不执行任何删除。

    Args:
        runtime: 工具运行时上下文。
        user_id: 用户 ID（QQ 号）。
        short_id: 要删除的画像条目 short_id（单条或列表）。

            - short_id 由 `add_user_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。
    Returns:
        str: 操作结果描述。
    """
    long_memory: LongMemoryManager = runtime.context.long_memory

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

@tool()
async def update_user_profile_info(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int,
    short_id: str,
    new_info: str,
) -> str:
    """更新用户画像中的指定信息（仅接收 short_id）。

    Args:
        runtime: 工具运行时上下文。
        user_id: 用户 ID（QQ 号）。
        short_id: 要更新的画像条目 short_id。

            - short_id 由 `add_user_profile_info` 返回。
            - 为了避免数据库 ID 泄露，本工具不支持直接传数据库 doc_id。
        new_info: 新的信息内容。
    Returns:
        str: 操作结果描述。
    """
    long_memory: LongMemoryManager = runtime.context.long_memory

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

@tool()
async def get_user_profile_info(
    runtime: ToolRuntime[GroupChatContext],
    user_id: int | list[int],
) -> str:
    """获取用户画像（返回 short_id，不泄露 doc_id）。

    Args:
        runtime: 工具运行时上下文。
        user_id: 用户 ID（QQ 号）或用户 ID 列表。

    Returns:
        str: 用户画像信息（每条包含 short_id 与文本内容）。
    """

    long_memory: LongMemoryManager = runtime.context.long_memory

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

@tool()
async def search_user_profile_info(
    runtime: ToolRuntime[GroupChatContext],
    query: str,
    max_users: int = 5,
    limit: int = 10,
    mode: Literal["expand", "direct"] = "direct",
    return_content: str = "user_id,short_id,similarity,text",
    similarity_threshold: float = 0.0,
) -> str:
    """跨用户搜索用户画像（按用户聚合返回，附带相似度）。

    该工具会在全库用户画像中执行向量检索，并将命中结果按 `user_id` 聚合：

    - 先找出与 `query` 最相近的“最多 N 个用户”（以该用户命中中的最高 `similarity` 作为排序依据）。
    - 对每个用户返回最多 `limit` 条最相近的画像描述。
    - 返回内容包含 `similarity`/`distance` 作为参考，并将数据库 `doc_id` 映射为 `short_id`，避免泄露真实 ID。

    Args:
        runtime: 工具运行时上下文。
        query: 查询文本。
        max_users: 最大返回用户数。
        limit: 每个用户返回的画像条目数量上限。
        mode: 检索模式。
            - direct：仅返回相似度最高的命中条目列表，不做按用户聚合扩展。该模式下 `max_users` 无效，只有 `limit` 生效。
            - expand：当前默认逻辑。先向量检索，再按 user_id 聚合并返回每个用户的多条命中。
        return_content: 返回字段列表（逗号分隔）。可选字段：user_id、short_id、similarity、text、distance、threshold。
            默认："user_id,short_id,similarity,text"（精简模式）。
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
        return_fields = {"user_id", "short_id", "similarity", "text"}
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

    long_memory: LongMemoryManager = runtime.context.long_memory

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
        # direct 模式：仅返回相似度最高的若干条命中（不按用户扩展）。
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