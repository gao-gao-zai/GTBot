"""长记忆召回管理器。

负责根据对话上下文从向量数据库中召回相关的长期记忆，包括事件日志、公共知识、
用户画像和群画像等。采用增量刷新策略，在消息累积或空闲时自动更新召回结果。
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Protocol

import aiohttp

from nonebot import logger

from plugins.GTBot.model import Message

from .MappingManager import mapping_manager


_PUBLIC_KNOWLEDGE_GROUP: str = "global"


def _normalize_session_id(session_id: str | None) -> str:
    """规范化会话 ID。

    Args:
        session_id: 上游传入的会话标识。
            - 为空/全空白时会被替换为默认值，避免下游存储/检索因空 key 出错。
            - 用于将不同对话上下文隔离在各自的召回空间（例如事件日志按 session 过滤）。

    Returns:
        规范化后的 session id（非空字符串）。
    """
    sid = (session_id or "").strip()
    return sid or "test_session"


def _safe_one_line(text: str) -> str:
    return str(text or "").replace("\n", " ").replace("\r", " ").strip()


def _clip_total_chars(text: str, *, budget: int) -> str:
    """按总字符预算裁剪文本。

    Args:
        text: 需要裁剪的文本。
        budget: 剩余字符预算。
            - `<= 0` 时直接返回空串。
            - 以 *字符数*（非 token 数）控制输出长度。

    Returns:
        在预算内的一行文本。
    """
    if budget <= 0:
        return ""
    s = _safe_one_line(text)
    if len(s) <= budget:
        return s
    return s[:budget]


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿]", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    """将文本分词为 token 集合，支持中文二元组增强。

    对于连续汉字，额外生成相邻二元组以提升中文召回效果。
    例如 "你好世界" 会生成 {"你", "好", "世", "界", "你好", "好世", "世界"}。

    Args:
        text: 待分词的文本。

    Returns:
        token 集合，全小写。
    """
    tokens = _TOKEN_RE.findall(_safe_one_line(text).lower())
    if not tokens:
        return set()

    out: set[str] = set(tokens)

    han = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
    if len(han) >= 2:
        for a, b in zip(han, han[1:], strict=False):
            out.add(a + b)

    return out


def _lexical_overlap_ratio(*, query_tokens: set[str], doc_tokens: set[str]) -> float:
    """计算查询与文档的词汇重叠率。

    Args:
        query_tokens: 查询 token 集合。
        doc_tokens: 文档 token 集合。

    Returns:
        重叠率，范围 [0, 1]，基于查询 token 数归一化。
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    inter = query_tokens & doc_tokens
    return float(len(inter)) / float(max(1, len(query_tokens)))


def _is_high_signal_text(text: str, *, min_alnum_or_han: int = 2) -> bool:
    """判断文本是否具有足够信息量以参与构建查询。

    Args:
        text: 待检测文本。
        min_alnum_or_han: 最少需要出现的“有效字符”数量。
            有效字符指数字/英文字母/汉字，用于过滤表情、标点、空白等低信号内容。

    Returns:
        是否为高信号文本。
    """
    s = _safe_one_line(text)
    if not s:
        return False

    count = 0
    for ch in s:
        if "0" <= ch <= "9" or "a" <= ch.lower() <= "z" or "\u4e00" <= ch <= "\u9fff":
            count += 1
            if count >= int(min_alnum_or_han):
                return True
    return False


def _strip_note_and_msg_tags(text: str) -> str:
    """清理 `<note>`/`<msg>` 标签，得到适合召回的纯文本。

    Args:
        text: 原始文本（可能包含模型提示或结构化标签）。

    Returns:
        清理后的单行文本。
    """
    s = str(text or "")
    s = re.sub(r"<note>.*?</note>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<msg>(.*?)</msg>", r"\1", s, flags=re.DOTALL | re.IGNORECASE)
    return _safe_one_line(s)


class TEIReranker:
    """基于 OpenAI 风格 `/v1/rerank` 接口的重排客户端。

    该客户端面向长期记忆召回阶段使用，负责把候选文本列表提交给外部
    rerank 服务，并将响应标准化为内部可消费的 `index/score` 列表。

    当前实现只接受 OpenAI 风格扩展协议：
    - 请求体使用 `model`、`query`、`documents`
    - 响应体使用 `results[].index`、`results[].relevance_score`

    不再兼容旧的 TEI `texts/raw_scores` 请求格式，也不再解析旧的列表式返回。
    """

    def __init__(
        self,
        *,
        api_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_url = str(api_url or "").strip()
        self.model_name = str(model_name or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = float(timeout_seconds)

    async def rerank(self, *, query: str, texts: list[str]) -> list[dict[str, Any]] | None:
        """调用 OpenAI 风格 rerank 接口并返回标准化后的排序结果。

        该方法会在入参缺失、HTTP 非 2xx、响应体不符合 `results` 结构时返回
        `None`，让上层回退到原始召回排序，避免因为重排服务短暂不可用而中断主流程。

        Args:
            query: 用于重排候选文本的查询语句。
            texts: 待重排的候选文本列表，顺序即原始候选顺序。

        Returns:
            标准化后的结果列表。每个元素至少包含 `index` 和 `score` 字段；
            当接口不可用或响应不合法时返回 `None`。
        """
        if not self.api_url or not self.model_name or not str(query or "").strip() or not texts:
            return None

        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": str(query),
            "documents": [str(x) for x in texts],
        }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = aiohttp.ClientTimeout(total=max(1.0, float(self.timeout_seconds)))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                if resp.status < 200 or resp.status >= 300:
                    return None
                data = await resp.json()

        if not isinstance(data, dict):
            return None

        results = data.get("results")
        if not isinstance(results, list):
            return None

        normalized: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("index")
            raw_score = item.get("relevance_score")
            if raw_idx is None or raw_score is None:
                continue
            try:
                normalized.append(
                    {
                        "index": int(raw_idx),
                        "score": float(raw_score),
                    }
                )
            except Exception:
                continue

        return normalized or None


@dataclass(frozen=True, slots=True)
class LongMemoryRecallConfig:
    """长记忆召回配置。

    Attributes:
        context_capacity: 上下文消息队列容量。
        query_max_messages: 构建查询时使用的最大消息数。
        query_max_chars: 查询文本最大字符数。
        refresh_message_threshold: 触发刷新的累积消息阈值。
        idle_refresh_seconds: 空闲后触发刷新的等待秒数。
        max_concurrent_refreshes: 最大并发刷新任务数。
        event_log_*: 事件日志召回的条目数、字符预算和相似度阈值。
        public_knowledge_*: 公共知识召回的条目数、字符预算和相似度阈值。
        user_profile_*: 用户画像召回的条目数、字符预算和相似度阈值。
        group_profile_*: 群画像召回的条目数、字符预算和相似度阈值。
        group_profile_stable_*: 群画像稳定条目的条目数和字符预算。
    """
    context_capacity: int = 80
    query_max_messages: int = 12
    query_max_chars: int = 800

    refresh_message_threshold: int = 3
    idle_refresh_seconds: float = 2.0
    max_concurrent_refreshes: int = 3

    event_log_max_items: int = 2
    event_log_total_chars: int = 300
    event_log_min_similarity: float | None = None

    public_knowledge_max_items: int = 10
    public_knowledge_total_chars: int = 200
    public_knowledge_min_similarity: float | None = None

    user_profile_max_items: int = 10
    user_profile_total_chars: int = 300
    user_profile_min_similarity: float | None = None

    group_profile_max_items: int = 5
    group_profile_total_chars: int = 300
    group_profile_min_similarity: float | None = None

    group_profile_stable_max_items: int = 5
    group_profile_stable_total_chars: int = 300

    def validate(self) -> "LongMemoryRecallConfig":
        if self.context_capacity <= 0:
            raise ValueError("context_capacity 必须 > 0")
        if self.query_max_messages <= 0:
            raise ValueError("query_max_messages 必须 > 0")
        if self.query_max_chars <= 0:
            raise ValueError("query_max_chars 必须 > 0")

        if self.refresh_message_threshold <= 0:
            raise ValueError("refresh_message_threshold 必须 > 0")
        if self.idle_refresh_seconds <= 0:
            raise ValueError("idle_refresh_seconds 必须 > 0")
        if self.max_concurrent_refreshes <= 0:
            raise ValueError("max_concurrent_refreshes 必须 > 0")

        for name in (
            "event_log_max_items",
            "public_knowledge_max_items",
            "user_profile_max_items",
            "group_profile_max_items",
            "group_profile_stable_max_items",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须 > 0")

        for name in (
            "event_log_total_chars",
            "public_knowledge_total_chars",
            "user_profile_total_chars",
            "group_profile_total_chars",
            "group_profile_stable_total_chars",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} 必须 >= 0")

        for name in (
            "event_log_min_similarity",
            "public_knowledge_min_similarity",
            "user_profile_min_similarity",
            "group_profile_min_similarity",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} 必须在 [0, 1] 范围内")

        return self


@dataclass(frozen=True, slots=True)
class EventLogRecallItem:
    short_id: str
    event_name: str
    session_id: str
    relevant_members: list[int]
    details: str
    similarity: float


@dataclass(frozen=True, slots=True)
class PublicKnowledgeRecallItem:
    short_id: str
    title: str
    content: str
    similarity: float


@dataclass(frozen=True, slots=True)
class UserProfileRecallItem:
    user_id: int
    short_id: str
    text: str
    similarity: float


@dataclass(frozen=True, slots=True)
class GroupProfileRecallItem:
    short_id: str
    group_id: int
    category: str | None
    text: str
    similarity: float


@dataclass(frozen=True, slots=True)
class GroupProfileStableItem:
    short_id: str
    group_id: int
    category: str | None
    text: str


@dataclass(frozen=True, slots=True)
class RelatedLongMemories:
    """召回结果容器，包含各类相关的长期记忆。"""
    session_id: str
    group_id: int | None
    user_id: int | None
    query: str

    event_logs: list[EventLogRecallItem]
    public_knowledge: list[PublicKnowledgeRecallItem]
    user_profiles: list[UserProfileRecallItem]
    group_profile_hits: list[GroupProfileRecallItem]
    group_profile: list[GroupProfileStableItem] | None

    timings: dict[str, float] | None = None


class RelatedMemoryResolver(Protocol):
    """记忆解析器协议，定义如何从上下文中召回相关记忆。"""
    async def resolve(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        context_messages: list[Message],
    ) -> RelatedLongMemories:
        ...


@dataclass(slots=True)
class _QueryVariant:
    text: str
    weight: float
    tokens: set[str]


class VectorSearchResolver:
    """基于向量搜索的记忆解析器。

    通过构建多权重查询变体，并行搜索多个记忆库，并合并去重后返回最佳结果。
    """
    def __init__(self, *, config: LongMemoryRecallConfig, long_memory: Any) -> None:
        """创建向量搜索解析器。

        Args:
            config: 召回相关的阈值、条目数与字符预算配置。
            long_memory: 长记忆服务聚合对象。
                期望包含（按需）:
                - `event_log_manager`
                - `public_knowledge`
                - `user_profile_manager`
                - `group_profile_manager`
        """
        self.config = config
        self.long_memory = long_memory

    def _build_query_variants(self, messages: list[Message]) -> list[_QueryVariant]:
        """构建多权重查询变体。

        生成单条、近三条、全量三种查询，权重依次递减，
        以平衡精确召回与上下文覆盖。

        Args:
            messages: 上下文消息列表。

        Returns:
            查询变体列表，按权重降序排列。
        """
        cleaned: list[str] = []
        for m in messages:
            content = _strip_note_and_msg_tags(getattr(m, "content", "") or "")
            if not _is_high_signal_text(content):
                continue

            uid = getattr(m, "user_id", None)
            uname = getattr(m, "user_name", None)
            if uname:
                line = f"{uname}({uid}): {content}"
            else:
                line = f"{uid}: {content}" if uid is not None else content
            cleaned.append(_safe_one_line(line))

        if not cleaned:
            content = _safe_one_line(getattr(messages[-1], "content", "") or "") if messages else ""
            fallback = content[: int(self.config.query_max_chars)]
            v = _QueryVariant(text=fallback, weight=1.0, tokens=_tokenize(fallback))
            return [v] if v.text else []

        cleaned = cleaned[-max(1, int(self.config.query_max_messages)) :]

        def _join_last(n: int) -> str:
            text = "\n".join(cleaned[-n:]).strip()
            return text[: int(self.config.query_max_chars)]

        q1 = _join_last(1)
        q2 = _join_last(min(3, len(cleaned)))
        q3 = _join_last(len(cleaned))

        variants: list[_QueryVariant] = []
        for text, w in ((q1, 1.0), (q2, 0.92), (q3, 0.85)):
            t = _safe_one_line(text)
            if not t:
                continue
            if any(v.text == t for v in variants):
                continue
            variants.append(_QueryVariant(text=t, weight=float(w), tokens=_tokenize(t)))

        return variants

    @staticmethod
    def _merge_best_by_doc_id(
        scored_items: list[tuple[str, float, Any]],
    ) -> list[tuple[str, float, Any]]:
        best: dict[str, tuple[float, Any]] = {}
        for doc_id, score, item in scored_items:
            prev = best.get(doc_id)
            if prev is None or float(score) > float(prev[0]):
                best[doc_id] = (float(score), item)

        out: list[tuple[str, float, Any]] = [(doc_id, v[0], v[1]) for doc_id, v in best.items()]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    @staticmethod
    def _apply_public_knowledge_budget(
        items: list[PublicKnowledgeRecallItem],
        *,
        total_budget: int,
    ) -> list[PublicKnowledgeRecallItem]:
        budget = max(0, int(total_budget))
        if budget == 0:
            return []

        out: list[PublicKnowledgeRecallItem] = []
        for item in items:
            if budget <= 0:
                break

            title = _safe_one_line(item.title)
            content = _safe_one_line(item.content)

            title_clipped = _clip_total_chars(title, budget=budget)
            budget -= len(title_clipped)
            if budget <= 0:
                out.append(
                    PublicKnowledgeRecallItem(
                        short_id=item.short_id,
                        title=title_clipped,
                        content="",
                        similarity=item.similarity,
                    )
                )
                break

            content_clipped = _clip_total_chars(content, budget=budget)
            if content_clipped:
                budget -= len(content_clipped)

            if not title_clipped and not content_clipped:
                continue

            out.append(
                PublicKnowledgeRecallItem(
                    short_id=item.short_id,
                    title=title_clipped,
                    content=content_clipped,
                    similarity=item.similarity,
                )
            )

        return out

    @staticmethod
    def _apply_total_char_budget(
        items: list[Any],
        *,
        total_budget: int,
        get_text: Any,
        replace_text: Any,
    ) -> list[Any]:
        budget = max(0, int(total_budget))
        if budget == 0:
            return []

        out: list[Any] = []
        for item in items:
            if budget <= 0:
                break
            text = str(get_text(item) or "")
            clipped = _clip_total_chars(text, budget=budget)
            if not clipped:
                continue
            out.append(replace_text(item, clipped))
            budget -= len(clipped)
        return out

    async def resolve(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        context_messages: list[Message],
    ) -> RelatedLongMemories:
        """解析并召回相关的长期记忆。

        并行搜索事件日志、公共知识、用户画像和群画像，
        合并去重后应用字符预算裁剪。

        Args:
            session_id: 会话 ID。
                - 用于隔离不同会话的召回空间（事件日志会按 session 过滤）。
                - 会先经过 `_normalize_session_id` 处理。
            group_id: 群组 ID。
                - 为 `None` 或 `<= 0` 时不会检索群画像相关内容。
                - 用于群画像（group profile）检索与稳定画像读取。
            user_id: 用户 ID。
                - 当前请求的用户标识，主要用于回填到返回值，方便上游做展示/缓存。
                - 具体检索时会额外基于 `context_messages` 推断参与者集合做加权。
            context_messages: 上下文消息列表。
                - 用于构建查询变体（最近 1/3/N 条）。
                - 同时用于提取参与者（`user_id`）集合以做轻量加权。

        Returns:
            召回结果容器。
        """
        timings: dict[str, float] = {}
        t_resolve0 = time.perf_counter()

        sid = _normalize_session_id(session_id)
        variants = self._build_query_variants(context_messages)
        query_text = variants[0].text if variants else ""

        if not variants or not _is_high_signal_text(variants[-1].text):
            variants = []
            query_text = ""

        participants: set[int] = set()
        for m in context_messages:
            uid = getattr(m, "user_id", None)
            if isinstance(uid, int) and uid > 0:
                participants.add(int(uid))

        event_log_manager = getattr(self.long_memory, "event_log_manager", None)
        public_knowledge = getattr(self.long_memory, "public_knowledge", None)
        user_profile_manager = getattr(self.long_memory, "user_profile_manager", None)
        group_profile_manager = getattr(self.long_memory, "group_profile_manager", None)

        qs = [v.text for v in variants] if variants else [""]

        async def _timed(label: str, coro: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await coro
            finally:
                timings[label] = float(time.perf_counter() - t0)

        async def _search_event_logs() -> list[Any]:
            if event_log_manager is None:
                timings["search_event_logs"] = 0.0
                return []
            if not variants:
                timings["search_event_logs"] = 0.0
                return []
            hits = await event_log_manager.search_events(
                qs,
                n_results=int(self.config.event_log_max_items) * 6,
                session_id=sid,
                min_similarity=self.config.event_log_min_similarity,
                order_by="similarity",
                order="desc",
                touch_last_called=True,
            )
            return hits if isinstance(hits, list) else []

        async def _search_public_knowledge() -> list[Any]:
            if public_knowledge is None:
                timings["search_public_knowledge"] = 0.0
                return []
            if not variants:
                timings["search_public_knowledge"] = 0.0
                return []
            hits = await public_knowledge.search_public_knowledge(
                qs,
                n_results=int(self.config.public_knowledge_max_items) * 6,
                min_similarity=self.config.public_knowledge_min_similarity,
                order_by="similarity",
                order="desc",
                touch_last_called=True,
            )
            return hits if isinstance(hits, list) else []

        async def _search_user_profiles() -> list[Any]:
            if user_profile_manager is None:
                timings["search_user_profiles"] = 0.0
                return []
            if not variants:
                timings["search_user_profiles"] = 0.0
                return []
            hits = await user_profile_manager.search_user_profiles(
                qs,
                n_results=int(self.config.user_profile_max_items) * 8,
                order_by="similarity",
                order="desc",
            )
            return hits if isinstance(hits, list) else []

        async def _search_group_profiles() -> tuple[list[Any], Any | None]:
            if group_id is None or int(group_id) <= 0 or group_profile_manager is None:
                timings["search_group_profiles"] = 0.0
                timings["search_group_profiles_search"] = 0.0
                timings["search_group_profiles_stable"] = 0.0
                return [], None

            q = variants[-1].text if variants else ""
            hits: list[Any] = []
            if variants:
                t0 = time.perf_counter()
                hits = await group_profile_manager.search_group_profiles(
                    q,
                    group_id=int(group_id),
                    n_results=int(self.config.group_profile_max_items) * 6,
                    min_similarity=self.config.group_profile_min_similarity,
                    order_by="similarity",
                    order="desc",
                    touch_last_called=True,
                )
                timings["search_group_profiles_search"] = float(time.perf_counter() - t0)
            else:
                timings["search_group_profiles_search"] = 0.0

            timings["search_group_profiles_stable"] = 0.0
            return (hits if isinstance(hits, list) else []), None

        event_hits_raw, knowledge_hits_raw, user_hits_raw, (group_hits_raw, group_profile_raw) = await asyncio.gather(
            _timed("search_event_logs", _search_event_logs()),
            _timed("search_public_knowledge", _search_public_knowledge()),
            _timed("search_user_profiles", _search_user_profiles()),
            _timed("search_group_profiles", _search_group_profiles()),
        )

        t_merge0 = time.perf_counter()

        def _score(
            *,
            similarity: float,
            variant: _QueryVariant,
            doc_text: str,
            participant_boost: float = 0.0,
        ) -> float:
            ratio = _lexical_overlap_ratio(query_tokens=variant.tokens, doc_tokens=_tokenize(doc_text))
            lexical_boost = min(0.12, 0.12 * ratio)
            return float(similarity) * float(variant.weight) + lexical_boost + float(participant_boost)

        event_scored: list[tuple[str, float, Any]] = []
        if variants and event_hits_raw:
            for v, hit_list in zip(variants, event_hits_raw, strict=False):
                items = hit_list if isinstance(hit_list, list) else []
                for h in items:
                    doc_id = str(getattr(h, "doc_id", "") or "").strip()
                    if not doc_id:
                        continue
                    details = str(getattr(h, "details", "") or "")
                    members = getattr(h, "relevant_members", []) or []
                    boost = 0.05 if any(int(x) in participants for x in members if isinstance(x, int)) else 0.0
                    s = _score(similarity=float(getattr(h, "similarity", 0.0) or 0.0), variant=v, doc_text=details, participant_boost=boost)
                    event_scored.append((doc_id, s, h))

        knowledge_scored: list[tuple[str, float, Any]] = []
        if variants and knowledge_hits_raw:
            for v, hit_list in zip(variants, knowledge_hits_raw, strict=False):
                items = hit_list if isinstance(hit_list, list) else []
                for h in items:
                    doc_id = str(getattr(h, "doc_id", "") or "").strip()
                    if not doc_id:
                        continue
                    content = str(getattr(h, "content", "") or "")
                    title = str(getattr(h, "title", "") or "")
                    s = _score(
                        similarity=float(getattr(h, "similarity", 0.0) or 0.0),
                        variant=v,
                        doc_text=f"{title}\n{content}",
                    )
                    knowledge_scored.append((doc_id, s, h))

        user_scored: list[tuple[str, float, Any]] = []
        if variants and user_hits_raw:
            for v, hit_list in zip(variants, user_hits_raw, strict=False):
                items = hit_list if isinstance(hit_list, list) else []
                for h in items:
                    doc_id = str(getattr(h, "doc_id", "") or "").strip()
                    if not doc_id:
                        continue
                    text = str(getattr(h, "text", "") or "")
                    uid = int(getattr(h, "user_id", 0) or 0)
                    boost = 0.06 if uid in participants else 0.0
                    s = _score(
                        similarity=float(getattr(h, "similarity", 0.0) or 0.0),
                        variant=v,
                        doc_text=text,
                        participant_boost=boost,
                    )
                    user_scored.append((doc_id, s, h))

        group_scored: list[tuple[str, float, Any]] = []
        if variants and group_hits_raw:
            v = variants[-1]
            for h in group_hits_raw:
                doc_id = str(getattr(h, "doc_id", "") or "").strip()
                if not doc_id:
                    continue
                text = str(getattr(h, "text", "") or "")
                cat = getattr(h, "category", None)
                s = _score(
                    similarity=float(getattr(h, "similarity", 0.0) or 0.0),
                    variant=v,
                    doc_text=f"{cat or ''}\n{text}",
                )
                group_scored.append((doc_id, s, h))

        reranker = getattr(self.long_memory, "reranker", None)

        async def _maybe_rerank(
            *,
            label: str,
            merged: list[tuple[str, float, Any]],
            max_items: int,
            get_text: Any,
            candidate_mul: int = 6,
            candidate_cap: int = 40,
        ) -> list[tuple[str, float, Any]]:
            if not merged:
                timings[label] = 0.0
                return []
            if reranker is None or not query_text.strip():
                timings[label] = 0.0
                return merged[: int(max_items)]

            candidate_n = min(len(merged), max(int(max_items) * int(candidate_mul), int(max_items)), int(candidate_cap))
            candidate = merged[:candidate_n]
            texts = [str(get_text(x[2]) or "") for x in candidate]
            if not any(t.strip() for t in texts):
                timings[label] = 0.0
                return merged[: int(max_items)]

            t0 = time.perf_counter()
            try:
                ranks = await reranker.rerank(query=query_text, texts=texts)
            except Exception:
                timings[label] = float(time.perf_counter() - t0)
                return merged[: int(max_items)]
            timings[label] = float(time.perf_counter() - t0)

            if not ranks:
                return merged[: int(max_items)]

            idx2score: dict[int, float] = {}
            for r in ranks:
                raw_idx = getattr(r, "index", None) if not isinstance(r, dict) else r.get("index")
                raw_sc = getattr(r, "score", None) if not isinstance(r, dict) else r.get("score")
                if raw_idx is None or raw_sc is None:
                    continue
                try:
                    idx = int(raw_idx)
                    sc = float(raw_sc)
                except Exception:
                    continue
                if 0 <= idx < len(candidate):
                    idx2score[idx] = sc

            if not idx2score:
                return merged[: int(max_items)]

            order = sorted(idx2score.items(), key=lambda x: x[1], reverse=True)
            seen: set[int] = set()
            reranked: list[tuple[str, float, Any]] = []
            for idx, sc in order:
                if idx in seen:
                    continue
                seen.add(idx)
                doc_id, _, item = candidate[idx]
                reranked.append((doc_id, float(sc), item))

            for i, (doc_id, sc, item) in enumerate(candidate):
                if i in seen:
                    continue
                reranked.append((doc_id, float(sc), item))

            reranked.extend(merged[candidate_n:])
            return reranked[: int(max_items)]

        event_merged = self._merge_best_by_doc_id(event_scored)
        knowledge_merged = self._merge_best_by_doc_id(knowledge_scored)
        user_merged = self._merge_best_by_doc_id(user_scored)
        group_merged = self._merge_best_by_doc_id(group_scored)

        event_best, knowledge_best, user_best, group_best = await asyncio.gather(
            _maybe_rerank(
                label="rerank_event_logs",
                merged=event_merged,
                max_items=int(self.config.event_log_max_items),
                get_text=lambda h: str(getattr(h, "details", "") or ""),
            ),
            _maybe_rerank(
                label="rerank_public_knowledge",
                merged=knowledge_merged,
                max_items=int(self.config.public_knowledge_max_items),
                get_text=lambda h: f"{getattr(h, 'title', '') or ''}\n{getattr(h, 'content', '') or ''}",
            ),
            _maybe_rerank(
                label="rerank_user_profiles",
                merged=user_merged,
                max_items=int(self.config.user_profile_max_items),
                get_text=lambda h: str(getattr(h, "text", "") or ""),
            ),
            _maybe_rerank(
                label="rerank_group_profiles",
                merged=group_merged,
                max_items=int(self.config.group_profile_max_items),
                get_text=lambda h: f"{getattr(h, 'category', '') or ''}\n{getattr(h, 'text', '') or ''}",
            ),
        )

        events = [
            EventLogRecallItem(
                short_id=str(mapping_manager.get_short_id(layer="event_log", group=sid, long_id=doc_id)),
                event_name=str(getattr(h, "event_name", "") or ""),
                session_id=str(getattr(h, "session_id", "") or sid),
                relevant_members=[int(x) for x in (getattr(h, "relevant_members", []) or []) if int(x) > 0],
                details=_safe_one_line(str(getattr(h, "details", "") or "")),
                similarity=float(getattr(h, "similarity", 0.0) or 0.0),
            )
            for doc_id, _, h in event_best
        ]

        knowledge = [
            PublicKnowledgeRecallItem(
                short_id=str(mapping_manager.get_short_id(layer="public_knowledge", group=_PUBLIC_KNOWLEDGE_GROUP, long_id=doc_id)),
                title=_safe_one_line(str(getattr(h, "title", "") or ""))[:80],
                content=_safe_one_line(str(getattr(h, "content", "") or "")),
                similarity=float(getattr(h, "similarity", 0.0) or 0.0),
            )
            for doc_id, _, h in knowledge_best
        ]

        user_profiles = []
        user_profile_doc_ids: list[str] = []
        for doc_id, _, h in user_best:
            uid = int(getattr(h, "user_id", 0) or 0)
            sid2 = str(mapping_manager.get_short_id(layer="user_profile", group=str(uid), long_id=doc_id))
            user_profiles.append(
                UserProfileRecallItem(
                    user_id=uid,
                    short_id=sid2,
                    text=_safe_one_line(str(getattr(h, "text", "") or "")),
                    similarity=float(getattr(h, "similarity", 0.0) or 0.0),
                )
            )
            user_profile_doc_ids.append(doc_id)

        if user_profile_manager is not None and user_profile_doc_ids:
            try:
                await user_profile_manager.touch_read_time_by_doc_id(user_profile_doc_ids)
            except Exception:
                pass

        group_profile_hits = [
            GroupProfileRecallItem(
                short_id=str(mapping_manager.get_short_id(layer="group_profile", group=str(group_id or 0), long_id=doc_id)),
                group_id=int(getattr(h, "group_id", 0) or 0),
                category=getattr(h, "category", None),
                text=_safe_one_line(str(getattr(h, "text", "") or "")),
                similarity=float(getattr(h, "similarity", 0.0) or 0.0),
            )
            for doc_id, _, h in group_best
        ]

        group_profile: list[GroupProfileStableItem] | None = None
        if group_profile_raw is not None and getattr(group_profile_raw, "description", None):
            desc = getattr(group_profile_raw, "description", [])
            out_profile: list[GroupProfileStableItem] = []
            for item in desc:
                doc_id = str(getattr(item, "doc_id", "") or "").strip()
                if not doc_id:
                    continue
                out_profile.append(
                    GroupProfileStableItem(
                        short_id=str(
                            mapping_manager.get_short_id(
                                layer="group_profile",
                                group=str(group_id or 0),
                                long_id=doc_id,
                            )
                        ),
                        group_id=int(group_id or 0),
                        category=None,
                        text=_safe_one_line(str(getattr(item, "text", "") or getattr(item, "description", "") or "")),
                    )
                )
                if len(out_profile) >= int(self.config.group_profile_stable_max_items):
                    break
            group_profile = out_profile

        events = self._apply_total_char_budget(
            events,
            total_budget=int(self.config.event_log_total_chars),
            get_text=lambda x: x.details,
            replace_text=lambda x, t: EventLogRecallItem(
                short_id=x.short_id,
                event_name=x.event_name,
                session_id=x.session_id,
                relevant_members=x.relevant_members,
                details=t,
                similarity=x.similarity,
            ),
        )

        knowledge = self._apply_public_knowledge_budget(
            knowledge,
            total_budget=int(self.config.public_knowledge_total_chars),
        )

        user_profiles = self._apply_total_char_budget(
            user_profiles,
            total_budget=int(self.config.user_profile_total_chars),
            get_text=lambda x: x.text,
            replace_text=lambda x, t: UserProfileRecallItem(
                user_id=x.user_id,
                short_id=x.short_id,
                text=t,
                similarity=x.similarity,
            ),
        )

        min_user_profile_items = 3
        if (
            user_profile_manager is not None
            and len(user_profiles) < int(min_user_profile_items)
            and int(self.config.user_profile_total_chars) > 0
        ):
            used_chars = sum(len(str(getattr(x, "text", "") or "")) for x in user_profiles)
            remaining_chars = max(0, int(self.config.user_profile_total_chars) - int(used_chars))
            if remaining_chars > 0:
                exclude_doc_ids: set[str] = set(user_profile_doc_ids)
                needed = int(min_user_profile_items) - len(user_profiles)

                candidate_user_ids: list[int] = []
                for m in reversed(context_messages):
                    uid = getattr(m, "user_id", None)
                    if isinstance(uid, int) and uid > 0 and uid not in candidate_user_ids:
                        candidate_user_ids.append(int(uid))
                if isinstance(user_id, int) and user_id > 0 and user_id not in candidate_user_ids:
                    candidate_user_ids.append(int(user_id))

                extra_doc_ids: list[str] = []
                for uid in candidate_user_ids:
                    if needed <= 0 or remaining_chars <= 0:
                        break
                    try:
                        profiles = await user_profile_manager.get_user_profiles(user_id=int(uid), limit=20)
                    except Exception:
                        continue

                    desc = getattr(profiles, "description", None) or []
                    for item in desc:
                        if needed <= 0 or remaining_chars <= 0:
                            break
                        doc_id = str(getattr(item, "doc_id", "") or "").strip()
                        if not doc_id or doc_id in exclude_doc_ids:
                            continue
                        text = _safe_one_line(str(getattr(item, "text", "") or ""))
                        if not text:
                            continue
                        clipped = _clip_total_chars(text, budget=int(remaining_chars))
                        if not clipped:
                            continue

                        exclude_doc_ids.add(doc_id)
                        extra_doc_ids.append(doc_id)
                        sid2 = str(mapping_manager.get_short_id(layer="user_profile", group=str(int(uid)), long_id=doc_id))
                        user_profiles.append(
                            UserProfileRecallItem(
                                user_id=int(uid),
                                short_id=sid2,
                                text=clipped,
                                similarity=0.0,
                            )
                        )
                        remaining_chars -= len(clipped)
                        needed -= 1

                if extra_doc_ids:
                    try:
                        await user_profile_manager.touch_read_time_by_doc_id(extra_doc_ids)
                    except Exception:
                        pass

        group_profile_hits = self._apply_total_char_budget(
            group_profile_hits,
            total_budget=int(self.config.group_profile_total_chars),
            get_text=lambda x: x.text,
            replace_text=lambda x, t: GroupProfileRecallItem(
                short_id=x.short_id,
                group_id=x.group_id,
                category=x.category,
                text=t,
                similarity=x.similarity,
            ),
        )

        if group_profile is not None:
            group_profile = self._apply_total_char_budget(
                group_profile,
                total_budget=int(self.config.group_profile_stable_total_chars),
                get_text=lambda x: x.text,
                replace_text=lambda x, t: GroupProfileStableItem(
                    short_id=x.short_id,
                    group_id=x.group_id,
                    category=x.category,
                    text=t,
                ),
            )

        result = RelatedLongMemories(
            session_id=sid,
            group_id=group_id,
            user_id=user_id,
            query=query_text,
            event_logs=events,
            public_knowledge=knowledge,
            user_profiles=user_profiles,
            group_profile_hits=group_profile_hits,
            group_profile=group_profile,
            timings=timings,
        )

        timings["merge_and_budget"] = float(time.perf_counter() - t_merge0)
        timings["resolve_total"] = float(time.perf_counter() - t_resolve0)
        return result


@dataclass(slots=True)
class _SessionState:
    """会话状态，跟踪上下文、召回结果和刷新任务。"""
    context: Deque[Message]
    related: RelatedLongMemories | None = None
    dirty_count: int = 0
    seq: int = 0
    last_msg_at: float = 0.0
    last_refresh_at: float = 0.0
    group_id: int | None = None
    user_id: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle_task: asyncio.Task[None] | None = None
    refresh_task: asyncio.Task[RelatedLongMemories] | None = None


class LongMemoryRecallManager:
    """长记忆召回管理器。

    维护各会话的上下文队列，在消息累积达到阈值或空闲时触发增量刷新，
    通过信号量控制全局并发刷新数。
    """
    def __init__(
        self,
        *,
        config: LongMemoryRecallConfig,
        long_memory: Any,
        resolver: RelatedMemoryResolver | None = None,
    ) -> None:
        """创建长记忆召回管理器。

        Args:
            config: 召回与刷新策略配置（阈值、预算、并发等），会在初始化时校验。
            long_memory: 长记忆服务聚合对象，透传给底层 resolver 做实际检索。
            resolver: 自定义解析器。
                - 为空时默认使用 `VectorSearchResolver`。
                - 便于在不同向量库/检索策略间切换或做单测替身。
        """
        self.config = config.validate()
        self.long_memory = long_memory
        self._sessions: dict[str, _SessionState] = {}
        self._global_sem = asyncio.Semaphore(int(self.config.max_concurrent_refreshes))
        self._resolver = resolver or VectorSearchResolver(config=self.config, long_memory=long_memory)

    def _get_or_create_session(self, session_id: str) -> _SessionState:
        """获取或创建会话状态。

        Args:
            session_id: 会话 ID（允许未规范化），内部会调用 `_normalize_session_id`。

        Returns:
            对应会话的 `_SessionState`。
        """
        sid = _normalize_session_id(session_id)
        state = self._sessions.get(sid)
        if state is not None:
            return state

        state = _SessionState(context=deque(maxlen=int(self.config.context_capacity)))
        self._sessions[sid] = state
        return state

    async def add_message(
        self,
        *,
        session_id: str,
        message: Message | None = None,
        messages: list[Message] | None = None,
        group_id: int | None = None,
        user_id: int | None = None,
    ) -> None:
        """添加消息到会话上下文。

        累积消息数达到阈值时立即触发刷新，否则启动空闲刷新任务。

        Args:
            session_id: 会话 ID。
                - 决定消息被追加到哪个会话上下文队列。
                - 同时影响后续召回（事件日志按 session 过滤）。
            message: 新消息对象（可选）。
                - `message` 与 `messages` 可同时传入，会按 `messages` 先、`message` 后拼接。
                - 主要使用 `content` 构建查询。
                - `user_id`/`user_name` 会参与参与者推断与 query 构造。
            messages: 批量新消息列表（可选）。
                - 用于一次性追加多条上下文消息。
            group_id: 群组 ID（可选）。
                - 提供时会写入会话状态，后续刷新时用于检索群画像。
                - 不提供则沿用会话中已记录的 group_id。
            user_id: 用户 ID（可选）。
                - 提供时会写入会话状态，后续刷新时回填到返回值。
                - 不提供则沿用会话中已记录的 user_id。
        """
        sid = _normalize_session_id(session_id)
        state = self._get_or_create_session(sid)

        batch: list[Message] = []
        if messages:
            batch.extend([m for m in messages if m is not None])
        if message is not None:
            batch.append(message)

        if not batch:
            return

        async with state.lock:
            existing_message_ids: set[int] = set()
            for x in state.context:
                mid = int(getattr(x, "message_id", 0) or 0)
                if mid > 0:
                    existing_message_ids.add(mid)

            appended = 0
            seen_in_batch: set[int] = set()
            for m in batch:
                mid = int(getattr(m, "message_id", 0) or 0)
                if mid > 0:
                    if mid in existing_message_ids or mid in seen_in_batch:
                        continue
                    seen_in_batch.add(mid)
                state.context.append(m)
                appended += 1

            if appended <= 0:
                return

            now = float(time.time())
            state.last_msg_at = now
            state.group_id = group_id if group_id is not None else state.group_id
            state.user_id = user_id if user_id is not None else state.user_id

            state.dirty_count += appended
            state.seq += appended

            if state.idle_task is not None and not state.idle_task.done():
                state.idle_task.cancel()

            state.idle_task = asyncio.create_task(
                self._idle_refresh_after(session_id=sid, delay_seconds=float(self.config.idle_refresh_seconds))
            )

            if state.dirty_count >= int(self.config.refresh_message_threshold):
                asyncio.create_task(self.refresh_session(session_id=sid, reason="threshold"))

    async def get_current_related_memories(
        self,
        *,
        session_id: str,
        group_id: int | None = None,
        user_id: int | None = None,
        force_refresh: bool = False,
    ) -> RelatedLongMemories:
        """获取当前召回结果，必要时触发刷新。

        Args:
            session_id: 会话 ID。
                - 用于定位会话状态与上下文队列。
            group_id: 群组 ID（可选）。
                - 提供时会覆盖/写入会话状态，影响后续刷新时的群画像检索范围。
            user_id: 用户 ID（可选）。
                - 提供时会覆盖/写入会话状态，主要用于回填到返回值。
            force_refresh: 是否强制刷新。
                - 为 `True` 时无论是否 dirty，都会触发一次刷新。
                - 常用于上游希望“立刻看到最新召回结果”的场景。

        Returns:
            最新的召回结果。
        """
        sid = _normalize_session_id(session_id)
        state = self._get_or_create_session(sid)

        async with state.lock:
            if group_id is not None:
                state.group_id = group_id
            if user_id is not None:
                state.user_id = user_id

            need = bool(force_refresh or state.related is None or state.dirty_count > 0)
            cached = state.related

        if not need and cached is not None:
            return cached

        last: RelatedLongMemories | None = None
        for _ in range(2):
            last = await self.refresh_session(session_id=sid, reason="get")
            async with state.lock:
                if state.dirty_count <= 0:
                    return last

        return last if last is not None else await self.refresh_session(session_id=sid, reason="get")

    async def _idle_refresh_after(self, *, session_id: str, delay_seconds: float) -> None:
        """在会话空闲一段时间后触发刷新。

        Args:
            session_id: 会话 ID。
            delay_seconds: 延迟秒数。
                在 `add_message` 中会重置该任务，用于“用户停止输入后再刷新”，减少频繁检索。
        """
        try:
            await asyncio.sleep(float(delay_seconds))
            await self.refresh_session(session_id=session_id, reason="idle")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                f"LongMemory recall idle refresh 任务异常（session={session_id}）: {type(exc).__name__}: {exc!r}"
            )

    async def refresh_session(self, *, session_id: str, reason: str) -> RelatedLongMemories:
        """刷新会话的召回结果。

        若已有刷新任务在进行则等待其完成，避免重复刷新。

        Args:
            session_id: 会话 ID。
                - 用于读取会话上下文并将刷新结果写回会话缓存。
            reason: 刷新原因。
                - 仅用于日志/观测（例如 "threshold"/"idle"/"get"）。
                - 便于定位刷新触发来源。

        Returns:
            召回结果。
        """
        sid = _normalize_session_id(session_id)
        state = self._get_or_create_session(sid)

        async with state.lock:
            if state.refresh_task is not None and not state.refresh_task.done():
                task = state.refresh_task
            else:
                group_id = state.group_id
                user_id = state.user_id
                ctx = list(state.context)
                start_seq = int(state.seq)
                task = asyncio.create_task(
                    self._do_refresh_and_commit(
                        state=state,
                        session_id=sid,
                        group_id=group_id,
                        user_id=user_id,
                        context_messages=ctx,
                        reason=reason,
                        start_seq=start_seq,
                    )
                )
                state.refresh_task = task

        result = await task
        return result

    async def _do_refresh_and_commit(
        self,
        *,
        state: _SessionState,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        context_messages: list[Message],
        reason: str,
        start_seq: int,
    ) -> RelatedLongMemories:
        """执行刷新并提交结果到会话状态。

        Args:
            state: 会话状态对象。
            session_id: 会话 ID。
            group_id: 群组 ID，透传给 resolver 用于群画像检索。
            user_id: 用户 ID，透传给 resolver 用于回填/扩展。
            context_messages: 刷新时使用的上下文快照。
            reason: 刷新原因（日志观测）。
            start_seq: 刷新启动时的序列号快照。
                用于在 commit 时计算刷新期间新增消息数，从而更新 `dirty_count`。

        Returns:
            刷新的召回结果。
        """
        result = await self._do_refresh(
            session_id=session_id,
            group_id=group_id,
            user_id=user_id,
            context_messages=context_messages,
            reason=reason,
        )

        async with state.lock:
            state.related = result
            state.last_refresh_at = float(time.time())
            state.dirty_count = max(0, int(state.seq) - int(start_seq))

            current = asyncio.current_task()
            if current is not None and state.refresh_task is current:
                state.refresh_task = None

        return result

    async def _do_refresh(
        self,
        *,
        session_id: str,
        group_id: int | None,
        user_id: int | None,
        context_messages: list[Message],
        reason: str,
    ) -> RelatedLongMemories:
        """实际执行一次召回刷新（带全局并发限制与降级策略）。

        Args:
            session_id: 会话 ID。
            group_id: 群组 ID。
            user_id: 用户 ID。
            context_messages: 上下文消息快照。
            reason: 刷新原因（仅用于日志）。

        Returns:
            召回结果。若 resolver 失败，会返回上一次成功结果或空结果（降级）。
        """
        async with self._global_sem:
            try:
                return await self._resolver.resolve(
                    session_id=session_id,
                    group_id=group_id,
                    user_id=user_id,
                    context_messages=context_messages,
                )
            except Exception as exc:
                logger.error(
                    f"LongMemory recall 刷新失败（session={session_id} reason={reason} ctx={len(context_messages)}）: "
                    f"{type(exc).__name__}: {exc!r}"
                )

                state = self._sessions.get(session_id)
                if state is not None and state.related is not None:
                    return state.related

                return RelatedLongMemories(
                    session_id=session_id,
                    group_id=group_id,
                    user_id=user_id,
                    query="",
                    event_logs=[],
                    public_knowledge=[],
                    user_profiles=[],
                    group_profile_hits=[],
                    group_profile=None,
                    timings=None,
                )
