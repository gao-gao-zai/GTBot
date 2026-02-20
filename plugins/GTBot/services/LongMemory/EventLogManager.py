from __future__ import annotations

import time
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .model import EventLog
from .model import EventLogSearchHit
from .model import EventLogWithId
from .model import TimeSlot
from .VectorGenerator import VectorGenerator


MAX_N_RESULTS = 100

PayloadValue = str | int | float | bool | None
Payload = dict[str, Any]


class QdrantEventLogManager:
    """事件日志的存储与检索服务（Qdrant 版本）。

    该类用于管理“事件日志（Event Logs）”条目：
        - 支持新增/获取/更新/删除。
        - 支持向量相似度检索（Qdrant `query_points`）。
        - 支持按会话 ID、相关成员等 payload 条件过滤。
        - 命中后可更新 `last_called_time`（可选）。

    重要约束（按你的最新决定）：
        - `event_name` 允许重复：不作为唯一键，仅用于展示/检索辅助。
        - Active Event List 由上游负责；本模块不维护任何“活跃列表”的状态。

    数据约定（payload 字段）：
        - `type`: 固定为 "event_log"。
        - `event_name`: 事件名（可重复）。
        - `session_id`: 会话 ID（例如 `group_<群号>` 或 `private_<QQ号>`）。
        - `details`: 事件详情（自然语言）。
        - `relevant_members`: 相关成员 ID 列表（int）。
        - `time_slots`: 时间段列表（list[dict]，包含 start_time/end_time）。
        - `creation_time` / `last_updated` / `last_called_time`: Unix 时间戳（秒）。

    Attributes:
        collection_name: Qdrant collection 名称。
        client: Qdrant 异步客户端。
        vector_generator: 向量生成器对象，用于生成文本向量。
    """

    def __init__(
        self,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
    ) -> None:
        """初始化事件日志管理器。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
        """

        self.collection_name: str = collection_name
        self.client: AsyncQdrantClient = client
        self.vector_generator: VectorGenerator = vector_generator
        self.payload_type_value: str = "event_log"

    @classmethod
    async def create(
        cls,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> "QdrantEventLogManager":
        """创建服务实例并确保 collection 存在。

        说明：
            - 若 collection 不存在，会创建一个单向量 collection，并使用 `VectorParams` 设置维度与度量。
            - 若未显式传入 `vector_size`，会通过一次 embedding 探测维度（会产生额外一次向量生成开销）。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
            vector_size: 向量维度；不传时会通过一次 embedding 探测维度。
            distance: 距离度量方式，默认 `COSINE`。

        Returns:
            QdrantEventLogManager: 服务实例。

        Raises:
            ValueError: collection_name 为空，或 vector_size 非法。
        """

        if not collection_name:
            raise ValueError("collection_name 不能为空")

        size = vector_size
        if size is None:
            probe = await vector_generator.embed_query("dimension probe")
            size = int(probe.shape[0])

        if size <= 0:
            raise ValueError(f"vector_size 非法: {size}")

        if not await client.collection_exists(collection_name=collection_name):
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=distance),
            )

        return cls(collection_name=collection_name, client=client, vector_generator=vector_generator)

    def _build_base_filter(
        self,
        *,
        session_id: str | None = None,
        relevant_members_any: list[int] | None = None,
    ) -> Filter:
        """构建事件日志过滤器。

        Args:
            session_id: 限定会话 ID。
            relevant_members_any: 若传入，则要求 `relevant_members` 中包含任意一个成员 ID。

        Returns:
            Filter: Qdrant 过滤器。
        """

        must: list[Any] = [
            FieldCondition(key="type", match=MatchValue(value=self.payload_type_value)),
        ]

        if session_id is not None:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=str(session_id))))

        if relevant_members_any:
            members = [int(x) for x in relevant_members_any]
            must.append(FieldCondition(key="relevant_members", match=MatchAny(any=members)))

        return Filter(must=must)

    @staticmethod
    def _normalize_point_id(doc_id: str) -> tuple[Any, str]:
        """将外部传入的 doc_id 规范化为 Qdrant 可接受的 PointId。

        Args:
            doc_id: 外部传入的 doc_id 字符串。

        Returns:
            tuple[Any, str]:
                - 第 1 项：可传给 Qdrant API 的 point id（int 或 UUID）。
                - 第 2 项：规范化字符串（`str(id)`）。

        Raises:
            ValueError: doc_id 为空或格式不合法。
        """

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id 不能为空")

        value = doc_id.strip()
        if value.isdigit():
            int_id = int(value)
            return int_id, str(int_id)

        try:
            uid = UUID(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"doc_id 非法（需为 int 或 UUID 字符串）: {doc_id!r}") from exc

        return uid, str(uid)

    @classmethod
    def _normalize_point_ids(cls, doc_id: str | list[str]) -> tuple[list[Any], list[str]]:
        """批量规范化 doc_id。"""

        raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not raw:
            raise ValueError("doc_id 不能为空")

        qdrant_ids: list[Any] = []
        keys: list[str] = []
        for item in raw:
            qid, key = cls._normalize_point_id(str(item))
            qdrant_ids.append(qid)
            keys.append(key)
        return qdrant_ids, keys

    @staticmethod
    def _as_vector_list(vectors: NDArray[np.float32]) -> list[list[float]]:
        """将二维 ndarray 转为 Qdrant 需要的 list[list[float]]。

        Args:
            vectors: shape 为 `(n, dim)` 的向量矩阵（dtype=float32）。

        Returns:
            list[list[float]]: Qdrant 可接受的向量列表。

        Raises:
            ValueError: vectors 不是二维数组。
        """

        if vectors.ndim != 2:
            raise ValueError(f"vectors 维度错误: ndim={vectors.ndim}")
        return [
            [float(x) for x in np.asarray(vectors[i], dtype=np.float32).tolist()]
            for i in range(int(vectors.shape[0]))
        ]

    @staticmethod
    def _payload_get_str(payload: dict[str, Any] | None, key: str, default: str = "") -> str:
        """从 payload 中读取字符串字段。"""

        if not payload:
            return default
        value = payload.get(key)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _payload_get_int(payload: dict[str, Any] | None, key: str, default: int = 0) -> int:
        """从 payload 中读取整数字段。"""

        if not payload:
            return default
        value = payload.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _payload_get_members(payload: dict[str, Any] | None) -> list[int]:
        """读取 relevant_members 列表。"""

        if not payload:
            return []
        value = payload.get("relevant_members")
        if not isinstance(value, list):
            return []
        out: list[int] = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    @staticmethod
    def _payload_get_time_slots(payload: dict[str, Any] | None) -> list[TimeSlot]:
        """读取 time_slots 列表。"""

        if not payload:
            return []
        value = payload.get("time_slots")
        if not isinstance(value, list):
            return []

        slots: list[TimeSlot] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                start_time = float(item.get("start_time", 0.0))
                end_time = float(item.get("end_time", 0.0))
            except Exception:
                continue
            if start_time <= 0 or end_time <= 0:
                continue
            slots.append(TimeSlot(start_time=start_time, end_time=end_time))
        return slots

    @staticmethod
    def _time_slots_to_payload(slots: list[TimeSlot]) -> list[dict[str, float]]:
        """将 TimeSlot 列表转为可 JSON 序列化的 payload。"""

        out: list[dict[str, float]] = []
        for s in slots:
            out.append({"start_time": float(s.start_time), "end_time": float(s.end_time)})
        return out

    def _build_document_for_embedding(self, *, event_name: str, event: EventLog) -> str:
        """将事件条目拼接为用于向量化的文档文本。"""

        name = str(event_name).strip()
        details = str(event.details).strip()

        members = ",".join(str(int(x)) for x in event.relevant_members)
        members_part = f"members:{members}" if members else ""

        parts = [p for p in (name, members_part, details) if p]
        return "\n".join(parts)

    async def add_event(
        self,
        event: EventLog,
        *,
        event_name: str = "",
    ) -> str:
        """新增事件日志条目。

        Args:
            event: 事件日志数据。
            event_name: 事件名（可重复）。

        Returns:
            str: 新写入的 doc_id（UUID 字符串）。

        Raises:
            ValueError: event 内容非法。
        """

        if not isinstance(event, EventLog):
            raise ValueError("event 类型错误")

        details = str(event.details).strip()
        if not details:
            raise ValueError("details 不能为空")

        now_ts = float(time.time())
        ev = EventLog(
            time_slots=list(event.time_slots or []),
            details=details,
            relevant_members=[int(x) for x in (event.relevant_members or [])],
            session_id=str(event.session_id),
        )

        doc = self._build_document_for_embedding(event_name=str(event_name), event=ev)
        vec = await self.vector_generator.embed_query(doc)
        vector_list = [float(x) for x in np.asarray(vec, dtype=np.float32).tolist()]

        doc_uuid = uuid4()
        payload: Payload = {
            "type": self.payload_type_value,
            "event_name": str(event_name),
            "session_id": str(ev.session_id),
            "details": ev.details,
            "relevant_members": list(ev.relevant_members),
            "time_slots": self._time_slots_to_payload(ev.time_slots),
            "creation_time": now_ts,
            "last_updated": now_ts,
            "last_called_time": 0.0,
        }

        point = PointStruct(id=doc_uuid, vector=vector_list, payload=payload)
        await self.client.upsert(collection_name=self.collection_name, points=[point])
        return str(doc_uuid)

    async def get_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        with_vectors: bool = False,
    ) -> EventLogWithId | list[EventLogWithId]:
        """按 doc_id 获取事件日志条目。"""

        qdrant_ids, keys = self._normalize_point_ids(doc_id)
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=with_vectors,
        )

        id_to_item: dict[str, EventLogWithId] = {}
        for p in points:
            payload = p.payload or {}
            if str(payload.get("type", "")) != self.payload_type_value:
                continue
            key = str(p.id)
            id_to_item[key] = EventLogWithId(
                doc_id=key,
                event_name=self._payload_get_str(payload, "event_name"),
                session_id=self._payload_get_str(payload, "session_id"),
                relevant_members=self._payload_get_members(payload),
                time_slots=self._payload_get_time_slots(payload),
                details=self._payload_get_str(payload, "details"),
            )

        missing = [k for k in keys if k not in id_to_item]
        if missing:
            raise ValueError(f"以下 doc_id 不存在或不属于事件日志: {missing}")

        if isinstance(doc_id, str):
            return id_to_item[keys[0]]
        return [id_to_item[k] for k in keys]

    async def search_events(
        self,
        query: str | list[str],
        *,
        n_results: int = 5,
        session_id: str | None = None,
        relevant_members_any: list[int] | None = None,
        min_similarity: float | None = None,
        order_by: Literal["distance", "similarity"] = "similarity",
        order: Literal["asc", "desc"] = "desc",
        touch_last_called: bool = True,
    ) -> list[EventLogSearchHit] | list[list[EventLogSearchHit]]:
        """检索事件日志（Qdrant 向量检索）。

        说明：
            - Qdrant 返回 `score`（越大越相似）。
            - 本实现将：`similarity = score`，`distance = 1 - score`（便捷展示）。

        Args:
            query: 查询文本或文本列表。
            n_results: 每条查询返回的最大命中数。
            session_id: 可选；限定会话范围。
            relevant_members_any: 可选；限定相关成员（任意命中）。
            min_similarity: 最小相似度阈值。
            order_by: 排序依据。
            order: 排序顺序。
            touch_last_called: 是否更新命中条目的 last_called_time。

        Returns:
            命中项列表或列表列表。
        """

        if n_results <= 0:
            return [] if isinstance(query, str) else [[] for _ in query]

        reverse = order == "desc"

        base_filter = self._build_base_filter(
            session_id=session_id,
            relevant_members_any=relevant_members_any,
        )

        async def _search_one(q: str) -> list[EventLogSearchHit]:
            qv = await self.vector_generator.embed_query(q)
            resp = await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in np.asarray(qv, dtype=np.float32).tolist()],
                limit=min(int(n_results), MAX_N_RESULTS),
                query_filter=base_filter,
                with_payload=True,
                with_vectors=False,
            )

            hits: list[EventLogSearchHit] = []
            for p in resp.points:
                payload = p.payload or {}
                score = float(p.score or 0.0)
                hit = EventLogSearchHit(
                    doc_id=str(p.id),
                    event_name=self._payload_get_str(payload, "event_name"),
                    session_id=self._payload_get_str(payload, "session_id"),
                    relevant_members=self._payload_get_members(payload),
                    details=self._payload_get_str(payload, "details"),
                    similarity=float(score),
                    distance=float(1.0 - score),
                )
                if min_similarity is not None and hit.similarity < float(min_similarity):
                    continue
                hits.append(hit)

            if order_by == "distance":
                hits.sort(key=lambda x: x.distance, reverse=reverse)
            elif order_by == "similarity":
                hits.sort(key=lambda x: x.similarity, reverse=reverse)
            else:
                raise ValueError(f"不支持的 order_by: {order_by}")

            if touch_last_called and hits:
                now_ts = float(time.time())
                await self.touch_called_time_by_doc_id([h.doc_id for h in hits], timestamp=now_ts)

            return hits

        if isinstance(query, str):
            return await _search_one(query)

        out: list[list[EventLogSearchHit]] = []
        for q in query:
            out.append(await _search_one(str(q)))
        return out

    async def update_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        event_name: str | list[str] | None = None,
        session_id: str | list[str] | None = None,
        details: str | list[str] | None = None,
        relevant_members: list[int] | list[list[int]] | None = None,
        time_slots: list[TimeSlot] | list[list[TimeSlot]] | None = None,
        creation_time: float | list[float] | None = None,
        last_updated: float | list[float] | None = None,
        last_called_time: float | list[float] | None = None,
    ) -> int:
        """按 doc_id 更新事件日志记录。

        规则：
            - 只更新显式传入的字段；未传入字段保持不变。
            - 若更新了 `details/event_name/relevant_members`（影响语义）：会重算向量并 upsert。
            - 若仅更新状态/时间戳等：不重算向量（复用原向量）。

        Args:
            doc_id: Qdrant point id（单个或列表）。

        Returns:
            int: 实际更新条数。

        Raises:
            ValueError: 参数长度不匹配或 doc_id 不存在。
        """

        doc_ids_raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids_raw:
            return 0

        qdrant_ids, doc_keys = self._normalize_point_ids(doc_ids_raw)

        if (
            event_name is None
            and session_id is None
            and details is None
            and relevant_members is None
            and time_slots is None
            and creation_time is None
            and last_updated is None
            and last_called_time is None
        ):
            return 0

        def normalize_update_value(
            value: Any,
            n: int,
            *,
            name: str,
            scalar_types: tuple[type, ...],
        ) -> list[Any] | None:
            if value is None:
                return None
            if isinstance(value, scalar_types):
                return [value for _ in range(n)]
            if isinstance(value, list):
                if len(value) != n:
                    raise ValueError(f"参数 {name} 长度({len(value)})与 doc_id 长度({n})不一致")
                return value
            raise ValueError(f"参数 {name} 类型不支持: {type(value)!r}")

        n = len(doc_keys)
        event_name_list = normalize_update_value(event_name, n, name="event_name", scalar_types=(str,))
        session_id_list = normalize_update_value(session_id, n, name="session_id", scalar_types=(str,))
        details_list = normalize_update_value(details, n, name="details", scalar_types=(str,))

        # relevant_members / time_slots 比较特殊：允许 list[int] 作为“标量”广播
        relevant_members_list: list[list[int]] | None = None
        if relevant_members is not None:
            if isinstance(relevant_members, list) and (not relevant_members or isinstance(relevant_members[0], int)):
                relevant_members_list = [cast(list[int], relevant_members) for _ in range(n)]
            elif isinstance(relevant_members, list) and isinstance(relevant_members[0], list):
                if len(relevant_members) != n:
                    raise ValueError(
                        f"参数 relevant_members 长度({len(relevant_members)})与 doc_id 长度({n})不一致"
                    )
                relevant_members_list = [
                    [int(x) for x in cast(list[int], item)] for item in cast(list[list[int]], relevant_members)
                ]
            else:
                raise ValueError("参数 relevant_members 类型不支持")

        time_slots_list: list[list[TimeSlot]] | None = None
        if time_slots is not None:
            if isinstance(time_slots, list) and (not time_slots or isinstance(time_slots[0], TimeSlot)):
                time_slots_list = [cast(list[TimeSlot], time_slots) for _ in range(n)]
            elif isinstance(time_slots, list) and isinstance(time_slots[0], list):
                if len(time_slots) != n:
                    raise ValueError(f"参数 time_slots 长度({len(time_slots)})与 doc_id 长度({n})不一致")
                time_slots_list = [cast(list[TimeSlot], item) for item in cast(list[list[TimeSlot]], time_slots)]
            else:
                raise ValueError("参数 time_slots 类型不支持")

        creation_time_list = normalize_update_value(
            creation_time, n, name="creation_time", scalar_types=(int, float)
        )
        last_updated_list = normalize_update_value(
            last_updated, n, name="last_updated", scalar_types=(int, float)
        )
        last_called_list = normalize_update_value(
            last_called_time, n, name="last_called_time", scalar_types=(int, float)
        )

        existing = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=True,
        )
        existing_map: dict[str, Any] = {str(p.id): p for p in existing}

        missing = [x for x in doc_keys if x not in existing_map]
        if missing:
            raise ValueError(f"以下 doc_id 不存在: {missing}")

        new_points: list[PointStruct] = []
        new_docs_for_embedding: list[str] = []
        new_docs_index: list[int] = []

        for i, (qid, key) in enumerate(zip(qdrant_ids, doc_keys, strict=False)):
            p = existing_map[key]
            payload = dict(p.payload or {})

            if str(payload.get("type", "")) != self.payload_type_value:
                raise ValueError(f"doc_id 不属于事件日志: {key}")

            vector = p.vector

            if event_name_list is not None:
                payload["event_name"] = str(event_name_list[i])
            if session_id_list is not None:
                payload["session_id"] = str(session_id_list[i])

            semantic_changed = False
            if details_list is not None:
                text = str(details_list[i]).strip()
                if not text:
                    raise ValueError("details 不能为空")
                payload["details"] = text
                semantic_changed = True

            if relevant_members_list is not None:
                payload["relevant_members"] = [int(x) for x in relevant_members_list[i]]
                semantic_changed = True

            if time_slots_list is not None:
                payload["time_slots"] = self._time_slots_to_payload(time_slots_list[i])

            if creation_time_list is not None:
                payload["creation_time"] = float(creation_time_list[i])
            if last_updated_list is not None:
                payload["last_updated"] = float(last_updated_list[i])
            if last_called_list is not None:
                payload["last_called_time"] = float(last_called_list[i])

            if semantic_changed or event_name_list is not None:
                # 语义向量以 event_name + members + details 为主
                ev = EventLog(
                    time_slots=self._payload_get_time_slots(payload),
                    details=str(payload.get("details", "")),
                    relevant_members=self._payload_get_members(payload),
                    session_id=str(payload.get("session_id", "")),
                )
                doc = self._build_document_for_embedding(
                    event_name=str(payload.get("event_name", "")),
                    event=ev,
                )
                new_docs_for_embedding.append(doc)
                new_docs_index.append(i)

            new_points.append(PointStruct(id=qid, vector=vector, payload=payload))

        if new_docs_for_embedding:
            embeddings = await self.vector_generator.embed_documents(new_docs_for_embedding)
            vector_list = self._as_vector_list(embeddings)
            for j, i in enumerate(new_docs_index):
                new_points[i] = PointStruct(
                    id=new_points[i].id,
                    vector=vector_list[j],
                    payload=new_points[i].payload,
                )

        await self.client.upsert(collection_name=self.collection_name, points=new_points)
        return len(doc_keys)

    async def delete_by_doc_id(self, doc_id: str | list[str]) -> int:
        """按 doc_id 删除事件日志条目。"""

        qdrant_ids, _ = self._normalize_point_ids(doc_id)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )
        return len(qdrant_ids)

    async def touch_called_time_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        timestamp: float | None = None,
    ) -> int:
        """快捷更新命中时间（last_called_time）。"""

        ts = time.time() if timestamp is None else float(timestamp)
        return await self.update_by_doc_id(doc_id, last_called_time=ts)




# 兼容导出（便于未来统一 import 路径）
EventLogManager = QdrantEventLogManager


__all__ = [
    "QdrantEventLogManager",
    "EventLogManager",
]
