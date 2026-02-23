from __future__ import annotations

import time
from typing import Any, Literal
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

from .model import GroupProfileDescriptionWithId, GroupProfileSearchHit, GroupProfileWithDescriptionIds
from .VectorGenerator import VectorGenerator


MAX_N_RESULTS = 100

PayloadValue = str | int | float | bool | None
Payload = dict[str, Any]


class QdrantGroupProfileManager:
    """群画像的存储与检索服务（Qdrant 版本）。"""

    def __init__(
        self,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
    ) -> None:
        self.collection_name: str = collection_name
        self.client: AsyncQdrantClient = client
        self.vector_generator: VectorGenerator = vector_generator
        self.payload_type_value: str = "group_profile"

    @classmethod
    async def create(
        cls,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> "QdrantGroupProfileManager":
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
        return Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value=self.payload_type_value)),
            ],
        )

    def _build_group_filter(self, group_id: int) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value=self.payload_type_value)),
                FieldCondition(key="group_id", match=MatchValue(value=int(group_id))),
            ],
        )

    @staticmethod
    def _normalize_point_id(doc_id: str) -> tuple[Any, str]:
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
        if vectors.ndim != 2:
            raise ValueError(f"vectors 维度错误: ndim={vectors.ndim}")
        return [
            [float(x) for x in np.asarray(vectors[i], dtype=np.float32).tolist()]
            for i in range(int(vectors.shape[0]))
        ]

    @staticmethod
    def _normalize_vector(vector: Any) -> list[float]:
        if vector is None:
            raise ValueError("向量为空，无法写回 Qdrant")

        if isinstance(vector, list):
            return [float(x) for x in vector]

        if isinstance(vector, dict):
            if "default" in vector and isinstance(vector.get("default"), list):
                return [float(x) for x in vector["default"]]

            for v in vector.values():
                if isinstance(v, list):
                    return [float(x) for x in v]

        raise ValueError(f"不支持的向量结构: {type(vector)!r}")

    @staticmethod
    def _payload_get_str(payload: dict[str, Any] | None, key: str, default: str = "") -> str:
        if not payload:
            return default
        value = payload.get(key)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _payload_get_int(payload: dict[str, Any] | None, key: str, default: int = 0) -> int:
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
    def _payload_get_category(payload: dict[str, Any] | None) -> str | None:
        if not payload:
            return None
        raw = payload.get("category")
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None

    def _build_document_for_embedding(self, *, description: str, category: str | None) -> str:
        text = str(description).strip()
        cat = None if category is None else str(category).strip() or None
        if cat:
            return f"{cat}\n{text}"
        return text

    async def add_group_profile(
        self,
        group_id: int,
        profile_texts: str | list[str],
        *,
        category: str | None = None,
    ) -> list[str]:
        texts = [profile_texts] if isinstance(profile_texts, str) else list(profile_texts)
        cleaned = [str(x).strip() for x in texts if str(x).strip()]
        if not cleaned:
            return []

        docs = [self._build_document_for_embedding(description=t, category=category) for t in cleaned]
        vectors = await self.vector_generator.embed_documents(docs)
        if vectors.ndim != 2 or int(vectors.shape[0]) != len(docs):
            raise ValueError(f"向量数量({int(vectors.shape[0])})与文档数量({len(docs)})不一致")

        vector_list = self._as_vector_list(vectors)
        now_ts = float(time.time())
        cat = None if category is None else str(category).strip() or None

        out_ids: list[str] = []
        points: list[PointStruct] = []
        for i, text in enumerate(cleaned):
            doc_uuid = uuid4()
            payload: Payload = {
                "type": self.payload_type_value,
                "group_id": int(group_id),
                "description": str(text),
                "category": cat,
                "creation_time": now_ts,
                "last_updated": now_ts,
                "last_read": now_ts,
                "last_called_time": 0.0,
            }
            points.append(PointStruct(id=doc_uuid, vector=vector_list[i], payload=payload))
            out_ids.append(str(doc_uuid))

        await self.client.upsert(collection_name=self.collection_name, points=points)
        return out_ids

    async def get_group_profiles(
        self,
        group_id: int,
        *,
        limit: int = 50,
        sort_by: Literal["creation_time", "last_updated", "last_read"] = "last_updated",
        sort_order: Literal["asc", "desc"] = "desc",
        touch_read_time: bool = True,
    ) -> GroupProfileWithDescriptionIds:
        if limit <= 0:
            return GroupProfileWithDescriptionIds(id=int(group_id), description=[])

        if sort_by not in ("creation_time", "last_updated", "last_read"):
            raise ValueError(f"不支持的 sort_by: {sort_by}")

        flt = self._build_group_filter(int(group_id))

        items: list[tuple[str, str, float, float, float]] = []
        offset: Any = None
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=min(256, 10_000),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in batch:
                payload = p.payload or {}
                if str(payload.get("type", "")) != self.payload_type_value:
                    continue
                key = str(p.id)
                text = self._payload_get_str(payload, "description")
                creation_time = float(payload.get("creation_time", 0.0) or 0.0)
                last_updated = float(payload.get("last_updated", 0.0) or 0.0)
                last_read = float(payload.get("last_read", 0.0) or 0.0)
                items.append((key, text, creation_time, last_updated, last_read))

            if next_offset is None:
                break
            offset = next_offset

        reverse = sort_order != "asc"
        if sort_by == "creation_time":
            items.sort(key=lambda x: x[2], reverse=reverse)
        elif sort_by == "last_read":
            items.sort(key=lambda x: x[4], reverse=reverse)
        else:
            items.sort(key=lambda x: x[3], reverse=reverse)

        items = items[: int(limit)]

        if touch_read_time and items:
            await self.touch_read_time_by_doc_id([x[0] for x in items])

        desc_items = [GroupProfileDescriptionWithId(doc_id=doc_id, text=text) for doc_id, text, *_ in items]
        return GroupProfileWithDescriptionIds(id=int(group_id), description=desc_items)

    async def count_group_profile_descriptions(self, group_id: int, *, category: str | None = None) -> int:
        cat = None if category is None else str(category).strip() or None
        flt = self._build_group_filter(int(group_id))
        offset: Any = None
        total = 0
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=min(256, 10_000),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in batch:
                payload = p.payload or {}
                if str(payload.get("type", "")) != self.payload_type_value:
                    continue
                if cat is not None and self._payload_get_category(payload) != cat:
                    continue
                total += 1

            if next_offset is None:
                break
            offset = next_offset

        return total

    async def update_by_doc_id(
        self,
        doc_id: str,
        *,
        description: str | None = None,
        category: str | None = None,
        last_updated: float | None = None,
    ) -> bool:
        key = str(doc_id).strip()
        if not key:
            raise ValueError("doc_id 不能为空")

        qid, doc_key = self._normalize_point_id(key)

        if description is None and category is None and last_updated is None:
            return False

        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[qid],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return False

        p = points[0]
        payload = dict(p.payload or {})
        if str(payload.get("type", "")) != self.payload_type_value:
            return False

        old_description = self._payload_get_str(payload, "description")
        old_category = self._payload_get_category(payload)

        semantic_changed = False
        if description is not None:
            text = str(description).strip()
            if not text:
                raise ValueError("description 不能为空")
            if text != old_description:
                payload["description"] = text
                semantic_changed = True

        if category is not None:
            cat = str(category).strip()
            new_cat = cat or None
            if new_cat != old_category:
                payload["category"] = new_cat
                semantic_changed = True

        payload["last_updated"] = float(time.time()) if last_updated is None else float(last_updated)

        vector = self._normalize_vector(p.vector)
        if semantic_changed:
            doc = self._build_document_for_embedding(
                description=self._payload_get_str(payload, "description"),
                category=self._payload_get_category(payload),
            )
            vec = await self.vector_generator.embed_query(doc)
            vector = [float(x) for x in np.asarray(vec, dtype=np.float32).tolist()]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=qid, vector=vector, payload=payload)],
        )
        return True

    async def touch_read_time_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        timestamp: float | None = None,
    ) -> int:
        ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if not ids:
            return 0

        ts = float(time.time()) if timestamp is None else float(timestamp)

        qids, keys = self._normalize_point_ids(ids)
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qids,
            with_payload=True,
            with_vectors=True,
        )

        existing_map: dict[str, Any] = {str(p.id): p for p in points}
        new_points: list[PointStruct] = []
        for qid, key in zip(qids, keys, strict=False):
            p = existing_map.get(key)
            if p is None:
                continue
            payload = dict(p.payload or {})
            if str(payload.get("type", "")) != self.payload_type_value:
                continue
            payload["last_read"] = ts
            new_points.append(PointStruct(id=qid, vector=self._normalize_vector(p.vector), payload=payload))

        if not new_points:
            return 0

        await self.client.upsert(collection_name=self.collection_name, points=new_points)
        return len(new_points)

    async def touch_called_time_by_doc_id(self, doc_id: str | list[str], *, timestamp: float | None = None) -> int:
        ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        ids = [str(x).strip() for x in ids if str(x).strip()]
        if not ids:
            return 0

        ts = float(time.time()) if timestamp is None else float(timestamp)

        qids, keys = self._normalize_point_ids(ids)
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qids,
            with_payload=True,
            with_vectors=True,
        )

        existing_map: dict[str, Any] = {str(p.id): p for p in points}
        new_points: list[PointStruct] = []
        for qid, key in zip(qids, keys, strict=False):
            p = existing_map.get(key)
            if p is None:
                continue
            payload = dict(p.payload or {})
            if str(payload.get("type", "")) != self.payload_type_value:
                continue
            payload["last_called_time"] = ts
            new_points.append(PointStruct(id=qid, vector=self._normalize_vector(p.vector), payload=payload))

        if not new_points:
            return 0

        await self.client.upsert(collection_name=self.collection_name, points=new_points)
        return len(new_points)

    async def get_existing_doc_ids(self, group_id: int, doc_id: list[str]) -> set[str]:
        ids = [str(x).strip() for x in doc_id if str(x).strip()]
        if not ids:
            return set()

        qids, keys = self._normalize_point_ids(ids)
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qids,
            with_payload=True,
            with_vectors=False,
        )
        out = set()
        for p in points:
            payload = p.payload or {}
            if str(payload.get("type", "")) != self.payload_type_value:
                continue
            if int(payload.get("group_id", 0) or 0) != int(group_id):
                continue
            out.add(str(p.id))
        return out

    async def delete_many_by_doc_id(self, group_id: int, doc_id: list[str]) -> int:
        existing = await self.get_existing_doc_ids(int(group_id), doc_id)
        if not existing:
            return 0

        qids, _ = self._normalize_point_ids(list(existing))
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qids),
        )
        return len(qids)

    async def delete_all_by_group_id(self, group_id: int) -> int:
        flt = self._build_group_filter(int(group_id))
        offset: Any = None
        all_ids: list[str] = []
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=min(256, 10_000),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in batch:
                all_ids.append(str(p.id))

            if next_offset is None:
                break
            offset = next_offset

        if not all_ids:
            return 0

        qids, _ = self._normalize_point_ids(all_ids)
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qids),
        )
        return len(qids)

    async def search_group_profiles(
        self,
        query: str,
        *,
        group_id: int,
        n_results: int = 5,
        min_similarity: float | None = None,
        order_by: Literal["distance", "similarity"] = "similarity",
        order: Literal["asc", "desc"] = "desc",
        touch_last_called: bool = True,
    ) -> list[GroupProfileSearchHit]:
        if n_results <= 0:
            return []

        reverse = order == "desc"
        flt = self._build_group_filter(int(group_id))

        qv = await self.vector_generator.embed_query(str(query))
        resp = await self.client.query_points(
            collection_name=self.collection_name,
            query=[float(x) for x in np.asarray(qv, dtype=np.float32).tolist()],
            limit=min(int(n_results), MAX_N_RESULTS),
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[GroupProfileSearchHit] = []
        for p in resp.points:
            payload = p.payload or {}
            score = float(p.score or 0.0)
            if min_similarity is not None and score < float(min_similarity):
                continue

            hit = GroupProfileSearchHit(
                doc_id=str(p.id),
                group_id=self._payload_get_int(payload, "group_id"),
                category=self._payload_get_category(payload),
                text=self._payload_get_str(payload, "description"),
                similarity=float(score),
                distance=float(1.0 - score),
            )
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


GroupProfileManager = QdrantGroupProfileManager


__all__ = [
    "QdrantGroupProfileManager",
    "GroupProfileManager",
]
