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
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .model import PublicKnowledge
from .model import PublicKnowledgeSearchHit
from .model import PublicKnowledgeWithId
from .VectorGenerator import VectorGenerator


PayloadValue = str | int | float | bool | None
Payload = dict[str, PayloadValue]

MAX_N_RESULTS = 100


class QdrantPublicKnowledge:
    """公共知识的存储与检索服务（Qdrant 版本）。

    该类用于管理“公共知识（Public Knowledge）”，与用户画像不同：
        - 不绑定用户/群；全局一份。
        - 支持相似度检索（向量检索）与 CRUD。
        - 命中后会更新 `last_called_time`，用于后续降权/清理。

    数据约定（payload 字段）：
        - `type`: 固定为 "public_knowledge"，用于与其它类型数据隔离。
        - `title`: 知识标题。
        - `content`: 知识正文。
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
        """初始化公共知识服务。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
        """

        self.collection_name: str = collection_name
        self.client: AsyncQdrantClient = client
        self.vector_generator: VectorGenerator = vector_generator
        self.payload_type_value: str = "public_knowledge"

    @classmethod
    async def create(
        cls,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> "QdrantPublicKnowledge":
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
            QdrantPublicKnowledge: 服务实例。

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

    def _build_type_filter(self) -> Filter:
        """构建仅匹配公共知识的过滤器。"""

        return Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value=self.payload_type_value)),
            ],
        )

    @staticmethod
    def _normalize_point_id(doc_id: str) -> tuple[Any, str]:
        """将外部传入的 doc_id 规范化为 Qdrant 可接受的 PointId。

        Qdrant 的 point id 通常只支持：
            - 整数（int）
            - UUID（uuid.UUID / UUID 字符串）

        本项目默认使用 UUID。

        Args:
            doc_id: 外部传入的 doc_id 字符串。

        Returns:
            tuple[Any, str]:
                - 第 1 项：可传给 Qdrant API 的 point id（int 或 UUID）。
                - 第 2 项：用于本地比对/映射的规范化字符串（`str(id)`）。

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

    def _build_document_for_embedding(self, knowledge: PublicKnowledge) -> str:
        """将公共知识条目拼接为用于向量化的文档文本。"""

        title = str(knowledge.title).strip()
        content = str(knowledge.content).strip()
        if title and content:
            return f"{title}\n{content}"
        return title or content

    async def add_public_knowledge(self, knowledge: PublicKnowledge | list[PublicKnowledge]) -> list[str]:
        """添加公共知识条目。

        Args:
            knowledge: 单条或多条公共知识条目。

        Returns:
            list[str]: 写入成功的 doc_id 列表。
        """

        items: list[PublicKnowledge] = [knowledge] if isinstance(knowledge, PublicKnowledge) else list(knowledge)
        if not items:
            return []

        docs = [self._build_document_for_embedding(x) for x in items]
        vectors = await self.vector_generator.embed_documents(docs)
        if vectors.ndim != 2 or int(vectors.shape[0]) != len(docs):
            raise ValueError(f"向量数量({int(vectors.shape[0])})与文档数量({len(docs)})不一致")

        vector_list = self._as_vector_list(vectors)
        now_ts = float(time.time())

        out_ids: list[str] = []
        points: list[PointStruct] = []
        for i, item in enumerate(items):
            doc_uuid = uuid4()
            payload: Payload = {
                "type": self.payload_type_value,
                "title": str(item.title),
                "content": str(item.content),
                "creation_time": now_ts,
                "last_updated": now_ts,
                "last_called_time": 0.0,
            }
            points.append(PointStruct(id=doc_uuid, vector=vector_list[i], payload=payload))
            out_ids.append(str(doc_uuid))

        await self.client.upsert(collection_name=self.collection_name, points=points)
        return out_ids

    async def get_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        with_vectors: bool = False,
    ) -> PublicKnowledgeWithId | list[PublicKnowledgeWithId]:
        """按 doc_id 获取公共知识条目。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            with_vectors: 是否同时拉取向量；默认 False。

        Returns:
            PublicKnowledgeWithId 或其列表。

        Raises:
            ValueError: doc_id 为空或记录不存在。
        """

        qdrant_ids, keys = self._normalize_point_ids(doc_id)
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=with_vectors,
        )

        id_to_item: dict[str, PublicKnowledgeWithId] = {}
        for p in points:
            payload = p.payload or {}
            if str(payload.get("type", "")) != self.payload_type_value:
                continue
            id_to_item[str(p.id)] = PublicKnowledgeWithId(
                doc_id=str(p.id),
                title=self._payload_get_str(payload, "title"),
                content=self._payload_get_str(payload, "content"),
            )

        missing = [k for k in keys if k not in id_to_item]
        if missing:
            raise ValueError(f"以下 doc_id 不存在或不属于公共知识: {missing}")

        if isinstance(doc_id, str):
            return id_to_item[keys[0]]
        return [id_to_item[k] for k in keys]

    async def search_public_knowledge(
        self,
        query: str | list[str],
        *,
        n_results: int = 5,
        min_similarity: float | None = None,
        order_by: Literal["distance", "similarity"] = "similarity",
        order: Literal["asc", "desc"] = "desc",
        touch_last_called: bool = True,
    ) -> list[PublicKnowledgeSearchHit] | list[list[PublicKnowledgeSearchHit]]:
        """检索公共知识（Qdrant 向量检索）。

        说明：
            - Qdrant 返回 `score`（越大越相似）。
            - 本实现将：`similarity = score`，`distance = 1 - score`（便捷展示）。
            - 默认会更新命中条目的 `last_called_time`。

        Args:
            query: 查询文本或文本列表。
            n_results: 每条查询返回的最大命中数。
            min_similarity: 最小相似度阈值；不传表示不过滤。
            order_by: 排序依据字段。
            order: 排序顺序。
            touch_last_called: 是否在命中后更新 `last_called_time`。

        Returns:
            命中项列表或列表列表。
        """

        if n_results <= 0:
            return [] if isinstance(query, str) else [[] for _ in query]

        reverse = order == "desc"

        async def _search_one(q: str) -> list[PublicKnowledgeSearchHit]:
            qv = await self.vector_generator.embed_query(q)
            resp = await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in np.asarray(qv, dtype=np.float32).tolist()],
                limit=min(int(n_results), MAX_N_RESULTS),
                query_filter=self._build_type_filter(),
                with_payload=True,
                with_vectors=False,
            )
            results = resp.points

            hits: list[PublicKnowledgeSearchHit] = []
            for p in results:
                payload = p.payload or {}
                score = float(p.score or 0.0)
                hit = PublicKnowledgeSearchHit(
                    doc_id=str(p.id),
                    title=self._payload_get_str(payload, "title"),
                    content=self._payload_get_str(payload, "content"),
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

        out: list[list[PublicKnowledgeSearchHit]] = []
        for q in query:
            out.append(await _search_one(str(q)))
        return out

    async def update_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        title: str | list[str] | None = None,
        content: str | list[str] | None = None,
        creation_time: float | list[float] | None = None,
        last_updated: float | list[float] | None = None,
        last_called_time: float | list[float] | None = None,
    ) -> int:
        """按 doc_id 更新公共知识记录。

        规则：
            - 只会更新显式传入的字段；未传入的字段保持不变。
            - 若更新了 `title/content`：会重新生成向量，并通过 `upsert` 覆盖写回。
            - 若仅更新 `*_time`：不会重新生成向量。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            title: 新标题。
            content: 新正文。
            creation_time: 创建时间。
            last_updated: 更新时间。
            last_called_time: 最近命中时间。

        Returns:
            int: 实际更新条数。

        Raises:
            ValueError: 参数列表长度不匹配或 doc_id 不存在。
        """

        doc_ids_raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids_raw:
            return 0

        qdrant_ids, doc_keys = self._normalize_point_ids(doc_ids_raw)

        if (
            title is None
            and content is None
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
        title_list = normalize_update_value(title, n, name="title", scalar_types=(str,))
        content_list = normalize_update_value(content, n, name="content", scalar_types=(str,))
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
                raise ValueError(f"doc_id 不属于公共知识: {key}")

            vector = p.vector

            if title_list is not None:
                payload["title"] = str(title_list[i])
            if content_list is not None:
                payload["content"] = str(content_list[i])

            if creation_time_list is not None:
                payload["creation_time"] = float(creation_time_list[i])
            if last_updated_list is not None:
                payload["last_updated"] = float(last_updated_list[i])
            if last_called_list is not None:
                payload["last_called_time"] = float(last_called_list[i])

            if title_list is not None or content_list is not None:
                doc_text = f"{payload.get('title','')}\n{payload.get('content','')}".strip()
                new_docs_for_embedding.append(doc_text)
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
        """按 doc_id 删除公共知识条目。

        Args:
            doc_id: Qdrant point id（单个或列表）。

        Returns:
            int: 删除请求的条目数（按输入数量计算）。
        """

        qdrant_ids, _ = self._normalize_point_ids(doc_id)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )
        return len(qdrant_ids)

    async def touch_called_time_by_doc_id(self, doc_id: str | list[str], *, timestamp: float | None = None) -> int:
        """快捷更新命中时间（last_called_time）。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            timestamp: 命中时间戳；不传则使用 `time.time()`。

        Returns:
            int: 实际更新的记录数。
        """

        ts = time.time() if timestamp is None else float(timestamp)
        return await self.update_by_doc_id(doc_id, last_called_time=ts)

    async def upsert_by_similarity(
        self,
        knowledge: PublicKnowledge,
        *,
        threshold: float = 0.85,
        n_results: int = 1,
    ) -> str:
        """按相似度进行 upsert（优先更新旧条目，避免重复）。

        逻辑：
            - 对输入知识进行相似度检索；若最相似命中项的相似度 >= threshold，则更新该条；否则新建。

        Args:
            knowledge: 待写入的公共知识。
            threshold: 相似度阈值；默认 0.85。
            n_results: 检索返回条数；默认 1。

        Returns:
            str: 被更新或新建的 doc_id。
        """

        query = self._build_document_for_embedding(knowledge)
        hits = await self.search_public_knowledge(
            query,
            n_results=int(n_results),
            min_similarity=None,
            touch_last_called=False,
        )

        now_ts = float(time.time())

        if hits and isinstance(hits, list) and isinstance(hits[0], PublicKnowledgeSearchHit):
            top = cast(list[PublicKnowledgeSearchHit], hits)[0]
            if float(top.similarity) >= float(threshold):
                await self.update_by_doc_id(
                    str(top.doc_id),
                    title=str(knowledge.title),
                    content=str(knowledge.content),
                    last_updated=now_ts,
                )
                return str(top.doc_id)

        ids = await self.add_public_knowledge(knowledge)
        if not ids:
            raise RuntimeError("写入失败：未返回 doc_id")
        return ids[0]
