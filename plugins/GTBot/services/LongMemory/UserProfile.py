from __future__ import annotations

import time
from typing import Any, Literal, overload
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .model import UserProfile as UserProfileModel
from .model import UserProfileDescriptionWithId
from .model import UserProfileSearchHit
from .model import UserProfileWithDescriptionIds
from .VectorGenerator import VectorGenerator

MAX_N_RESULTS = 100


PayloadValue = str | int | float | bool | None
Payload = dict[str, PayloadValue]


class QdrantUserProfile:
    """用户画像的存储与检索服务（Qdrant 版本）。

    该类负责将“用户画像文本”写入 Qdrant，并提供以下能力：
        - 基于文本相似度的检索（向量检索，优先 `client.query_points`）
        - 基于 payload 过滤拉取全量（`client.scroll`），并在本地排序
        - 支持按 doc_id 更新/删除/反查归属

    与 Chroma 版本的关系：
        - 上层方法名与返回结构尽量与 Chroma 版 `UserProfile` 对齐，方便替换后端。
        - 初始化不同：Qdrant 需要显式创建/检查 collection，并配置向量维度与距离度量。
        - 相似度语义不同：Qdrant 通常返回 `score`（越大越相似），而 Chroma 常见 `distance`（越小越相似）。

    数据约定（payload 字段）：
        - `id`: 用户 ID (QQ号)
        - `description`: 描述文本
        - `creation_time` / `last_updated` / `last_read`: Unix 时间戳（秒）

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
        """初始化 Qdrant 用户画像服务。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
        """

        self.collection_name: str = collection_name
        self.client: AsyncQdrantClient = client
        self.vector_generator: VectorGenerator = vector_generator

    @classmethod
    async def create(
        cls,
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> "QdrantUserProfile":
        """创建 Qdrant 用户画像服务实例，并确保 collection 存在。

        说明：
            - 若 collection 不存在，会创建一个单向量 collection，并使用 `VectorParams` 设置维度与度量。
            - 若未显式传入 `vector_size`，会通过一次 embedding 探测维度（会产生额外一次向量生成开销）。
            - `distance` 会影响 `search` 返回的 `score` 语义；项目默认使用 `COSINE`。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
            vector_size: 向量维度；不传时会通过一次 embedding 探测维度。
            distance: 距离度量方式，默认 `COSINE`。

        Returns:
            QdrantUserProfile: 服务实例。

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

        return cls(
            collection_name=collection_name,
            client=client,
            vector_generator=vector_generator,
        )

    @staticmethod
    def _build_user_filter(user_id: int) -> Filter:
        """构建按用户 ID 过滤的 Qdrant Filter。

        Args:
            user_id: 用户 ID (QQ号)。

        Returns:
            Filter: 可用于 Qdrant `search/scroll` 的过滤器。
        """

        return Filter(
            must=[FieldCondition(key="id", match=MatchValue(value=int(user_id)))],
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
        """从 payload 中读取字符串字段。

        Args:
            payload: Qdrant point payload。
            key: 字段名。
            default: 缺失时返回的默认值。

        Returns:
            str: 读取到的字符串值。
        """

        if not payload:
            return default
        value = payload.get(key)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _payload_get_float(payload: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
        """从 payload 中读取浮点字段。

        Args:
            payload: Qdrant point payload。
            key: 字段名。
            default: 缺失时返回的默认值。

        Returns:
            float: 读取到的浮点值。
        """

        if not payload:
            return default
        value = payload.get(key)
        if value is None:
            return default
        return float(value)

    @overload
    async def add_user_profile(self, user_id: int, profile_texts: list[str] | str) -> None:
        """添加用户画像文本。

        Args:
            user_id: 用户 ID (QQ号)。
            profile_texts: 用户画像文本或文本列表。
        """

    @overload
    async def add_user_profile(self, user_id: UserProfileModel, profile_texts: None = None) -> None:
        """添加用户画像（以模型形式传入）。

        Args:
            user_id: 用户画像数据模型（见 `plugins.GTBot.services.LongMemory.model.UserProfile`）。
            profile_texts: 必须为 None。
        """

    async def add_user_profile(
        self,
        user_id: int | UserProfileModel,
        profile_texts: list[str] | str | None = None,
    ) -> None:
        """添加用户画像文本。

        写入 Qdrant 时：
            - point id 使用 `uuid4()` 生成（UUID）。
            - vector 来自 `vector_generator.embed`（二维矩阵）。
            - payload 写入 `id/description/creation_time/last_updated/last_read`。

        Args:
            user_id: 用户 ID (QQ号) 或用户画像数据模型。
            profile_texts: 用户画像文本或文本列表；当第一个参数为用户画像数据模型时必须为 None。

        Raises:
            TypeError: 传参组合不合法。
            ValueError: 向量数量与文档数量不一致。
        """

        if isinstance(user_id, UserProfileModel):
            if profile_texts is not None:
                raise TypeError("传入 UserProfile 时不允许同时传入 profile_texts")
            user_id_value = int(user_id.id)
            documents: list[str] = list(user_id.description)
        else:
            if profile_texts is None:
                raise TypeError("传入 user_id 时必须同时传入 profile_texts")
            if not profile_texts:
                return
            user_id_value = int(user_id)
            documents = [profile_texts] if isinstance(profile_texts, str) else list(profile_texts)

        if not documents:
            return

        vectors: NDArray[np.float32] = await self.vector_generator.embed(documents)
        if vectors.ndim != 2 or int(vectors.shape[0]) != len(documents):
            raise ValueError(
                f"向量数量({int(vectors.shape[0])})与文档数量({len(documents)})不一致"
            )

        vector_list = self._as_vector_list(vectors)
        now_ts = float(time.time())

        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            doc_id = uuid4()
            payload: Payload = {
                "id": int(user_id_value),
                "description": str(doc),
                "creation_time": now_ts,
                "last_updated": now_ts,
                "last_read": now_ts,
            }
            points.append(PointStruct(id=doc_id, vector=vector_list[i], payload=payload))

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def get_user_profiles(
        self,
        user_id: int,
        limit: int = 10,
        text: str | None = None,
        sort_by: Literal["creation_time", "last_updated", "last_read", "text", "auto"] = "auto",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> UserProfileWithDescriptionIds:
        """检索用户画像文本。

        当 `sort_by='text'`（或 `sort_by='auto'` 且提供了 `text`）时：
            - 使用向量相似度检索：`client.search(..., query_filter=Filter(id=user_id))`
            - Qdrant 返回按相似度排序的 points（受 collection 的 distance 影响）

        当 `sort_by!='text'` 时：
            - 使用 `client.scroll` 按 payload 过滤拉取全量
            - 然后在本地按 `creation_time/last_updated/last_read` 排序并截断到 `limit`

        Args:
            user_id: 用户 ID (QQ号)。
            limit: 最大返回数量；小于等于 0 时返回空画像。
            text: 可选的查询文本；仅在 `sort_by='text'` 时生效。
            sort_by: 排序字段。
            sort_order: 排序顺序。

        Returns:
            UserProfileWithDescriptionIds: 用户画像（带 doc_id）。

        Raises:
            ValueError: 参数不合法。
        """

        if limit <= 0:
            return UserProfileWithDescriptionIds(id=int(user_id), description=[])

        if sort_by == "auto":
            sort_by = "text" if text is not None else "last_updated"

        if sort_by == "text" and not text:
            raise ValueError("sort_by='text' 时必须提供 text")

        reverse = sort_order == "desc"

        # 1) 相似度检索：Qdrant score 越大越相似（COSINE 下）。
        if sort_by == "text":
            if text is None:
                raise ValueError("sort_by='text' 时必须提供 text")
            query_vector = await self.vector_generator.embed_query(text)
            resp = await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in np.asarray(query_vector, dtype=np.float32).tolist()],
                limit=min(int(limit), MAX_N_RESULTS),
                query_filter=self._build_user_filter(int(user_id)),
                with_payload=True,
                with_vectors=False,
            )
            results = resp.points

            descriptions: list[UserProfileDescriptionWithId] = []
            for p in results:
                payload = p.payload or {}
                descriptions.append(
                    UserProfileDescriptionWithId(doc_id=str(p.id), text=self._payload_get_str(payload, "description"))
                )

            return UserProfileWithDescriptionIds(id=int(user_id), description=descriptions)

        # 2) 非相似度：scroll 拉全量，本地排序
        if sort_by not in ("creation_time", "last_updated", "last_read"):
            raise ValueError(f"不支持的 sort_by: {sort_by}")

        points: list[tuple[str, str, dict[str, Any]]] = []
        offset: Any = None
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_user_filter(int(user_id)),
                limit=min(256, 10_000),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for p in batch:
                payload = p.payload or {}
                points.append((str(p.id), self._payload_get_str(payload, "description"), dict(payload)))

            if not batch or next_offset is None:
                break
            offset = next_offset

        points.sort(key=lambda x: float(x[2].get(sort_by, 0.0)), reverse=reverse)
        descriptions = [
            UserProfileDescriptionWithId(doc_id=doc_id, text=text_value)
            for doc_id, text_value, _ in points[: int(limit)]
        ]
        return UserProfileWithDescriptionIds(id=int(user_id), description=descriptions)

    @overload
    async def search_user_profiles(
        self,
        query: str,
        *,
        n_results: int = 10,
        order_by: Literal["distance", "similarity"] = "distance",
        order: Literal["asc", "desc"] = "asc",
    ) -> list[UserProfileSearchHit]:
        """检索全库用户画像（单条查询）。

        Args:
            query: 查询文本。
            n_results: 返回的最大命中数。
            order_by: 排序依据字段。
            order: 排序顺序。
        """

    @overload
    async def search_user_profiles(
        self,
        query: list[str],
        *,
        n_results: int = 10,
        order_by: Literal["distance", "similarity"] = "distance",
        order: Literal["asc", "desc"] = "asc",
    ) -> list[list[UserProfileSearchHit]]:
        """检索全库用户画像（批量查询）。

        Args:
            query: 查询文本列表。
            n_results: 每条查询返回的最大命中数。
            order_by: 排序依据字段。
            order: 排序顺序。
        """

    async def search_user_profiles(
        self,
        query: str | list[str],
        *,
        n_results: int = 10,
        order_by: Literal["distance", "similarity"] = "distance",
        order: Literal["asc", "desc"] = "asc",
    ) -> list[UserProfileSearchHit] | list[list[UserProfileSearchHit]]:
        """检索全库用户画像（Qdrant）。

        该方法用于“跨用户”的向量检索：会在整个 collection 范围内查找最相似的画像描述，
        并返回每条命中对应的描述文本、所属用户 ID、数据库记录 ID 以及距离/相似度分数。

        说明：
            - Qdrant 返回 `score`（越大越相似）。
            - 本实现将：`similarity = score`，`distance = 1 - score`（仅作便捷展示）。

        Args:
            query: 查询文本或文本列表。
            n_results: 每条查询返回的最大命中数。
            order_by: 排序依据字段。
            order: 排序顺序。

        Returns:
            命中项列表或列表列表。
        """

        if n_results <= 0:
            return [] if isinstance(query, str) else [[] for _ in query]

        reverse = order == "desc"

        async def _search_one(q: str) -> list[UserProfileSearchHit]:
            qv = await self.vector_generator.embed_query(q)
            resp = await self.client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in np.asarray(qv, dtype=np.float32).tolist()],
                limit=min(int(n_results), MAX_N_RESULTS),
                with_payload=True,
                with_vectors=False,
            )
            results = resp.points

            hits: list[UserProfileSearchHit] = []
            for p in results:
                payload = p.payload or {}
                score = float(p.score or 0.0)
                hits.append(
                    UserProfileSearchHit(
                        doc_id=str(p.id),
                        user_id=int(payload.get("id", 0)),
                        text=self._payload_get_str(payload, "description"),
                        distance=float(1.0 - score),
                        similarity=float(score),
                    )
                )

            if order_by == "distance":
                hits.sort(key=lambda x: x.distance, reverse=reverse)
            elif order_by == "similarity":
                hits.sort(key=lambda x: x.similarity, reverse=reverse)
            else:
                raise ValueError(f"不支持的 order_by: {order_by}")

            return hits

        if isinstance(query, str):
            return await _search_one(query)

        out: list[list[UserProfileSearchHit]] = []
        for q in query:
            out.append(await _search_one(str(q)))
        return out

    async def update_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        descriptions: str | list[str] | None = None,
        user_id: int | list[int] | None = None,
        creation_time: float | list[float] | None = None,
        last_updated: float | list[float] | None = None,
        last_read: float | list[float] | None = None,
    ) -> int:
        """按数据库记录 ID 更新画像描述记录（Qdrant）。

        只会更新显式传入的字段；未传入的字段保持不变。
        记录 ID（`doc_id` / point id）不允许更新。

        约定：
            - 若更新了 `descriptions`：
                - 会更新 payload 的 `description`
                - 并重新生成向量，随后使用 `upsert` 覆盖写回
            - 若仅更新 payload 字段（如 `user_id/xxx_time`）：
                - 不会重新生成向量

        Args:
            doc_id: Qdrant point id（单个或列表）。
            descriptions: 新的描述文本。
            user_id: 新的所属用户 ID。
            creation_time: 创建时间。
            last_updated: 更新时间。
            last_read: 读取时间。

        Returns:
            int: 实际更新条数。

        Raises:
            ValueError: 参数列表长度不匹配或 doc_id 不存在。
        """

        doc_ids_raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids_raw:
            return 0

        qdrant_ids: list[Any] = []
        doc_keys: list[str] = []
        for item in doc_ids_raw:
            qid, key = self._normalize_point_id(str(item))
            qdrant_ids.append(qid)
            doc_keys.append(key)

        if (
            descriptions is None
            and user_id is None
            and creation_time is None
            and last_updated is None
            and last_read is None
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
        descriptions_list = normalize_update_value(descriptions, n, name="descriptions", scalar_types=(str,))
        user_id_list = normalize_update_value(user_id, n, name="user_id", scalar_types=(int,))
        creation_time_list = normalize_update_value(
            creation_time, n, name="creation_time", scalar_types=(int, float)
        )
        last_updated_list = normalize_update_value(
            last_updated, n, name="last_updated", scalar_types=(int, float)
        )
        last_read_list = normalize_update_value(last_read, n, name="last_read", scalar_types=(int, float))

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
        # 需要重算向量的 doc
        new_docs_for_embedding: list[str] = []
        new_docs_index: list[int] = []

        for i, (qid, key) in enumerate(zip(qdrant_ids, doc_keys, strict=False)):
            p = existing_map[key]
            payload = dict(p.payload or {})
            vector = p.vector

            if descriptions_list is not None:
                new_text = str(descriptions_list[i])
                payload["description"] = new_text
                new_docs_for_embedding.append(new_text)
                new_docs_index.append(i)

            if user_id_list is not None:
                payload["id"] = int(user_id_list[i])
            if creation_time_list is not None:
                payload["creation_time"] = float(creation_time_list[i])
            if last_updated_list is not None:
                payload["last_updated"] = float(last_updated_list[i])
            if last_read_list is not None:
                payload["last_read"] = float(last_read_list[i])

            # 先占位，向量可能稍后补
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

    async def touch_read_time_by_doc_id(self, doc_id: str | list[str], timestamp: float | None = None) -> int:
        """快捷更新读取时间（last_read）。

        说明：
            - 等价于 `update_by_doc_id(..., last_read=timestamp)`。
            - 未提供 `timestamp` 时自动写入当前时间戳。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            timestamp: 读取时间戳；不传则使用 `time.time()`。

        Returns:
            int: 实际更新的记录数。
        """

        ts = time.time() if timestamp is None else float(timestamp)
        return await self.update_by_doc_id(doc_id, last_read=ts)

    async def touch_write_time_by_doc_id(self, doc_id: str | list[str], timestamp: float | None = None) -> int:
        """快捷更新写入时间（last_updated）。

        说明：
            - 等价于 `update_by_doc_id(..., last_updated=timestamp)`。
            - 未提供 `timestamp` 时自动写入当前时间戳。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            timestamp: 写入时间戳；不传则使用 `time.time()`。

        Returns:
            int: 实际更新的记录数。
        """

        ts = time.time() if timestamp is None else float(timestamp)
        return await self.update_by_doc_id(doc_id, last_updated=ts)

    async def get_owner_by_doc_id(
        self,
        doc_id: str | list[str],
        *,
        return_type: Literal["user_id", "profile"] = "user_id",
    ) -> int | list[int] | UserProfileWithDescriptionIds | list[UserProfileWithDescriptionIds]:
        """通过 doc_id（point id）反查所属用户。

        该方法用于在已知某条记录 ID 的情况下，读取 payload 并反推出它属于哪个用户。

        Args:
            doc_id: Qdrant point id（单个或列表）。
            return_type: 返回内容类型。
                - `user_id`: 返回所属用户 ID。
                - `profile`: 返回所属用户画像（会再次调用 `get_user_profiles` 获取全量描述）。

        Returns:
            - `return_type='user_id'`：返回 `int` 或 `list[int]`。
            - `return_type='profile'`：返回 `UserProfileWithDescriptionIds` 或其列表。

        Raises:
            ValueError: doc_id 为空、记录不存在或 payload 缺少必要字段。
        """

        doc_ids_raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids_raw:
            raise ValueError("doc_id 不能为空")

        qdrant_ids: list[Any] = []
        doc_keys: list[str] = []
        for item in doc_ids_raw:
            qid, key = self._normalize_point_id(str(item))
            qdrant_ids.append(qid)
            doc_keys.append(key)

        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=False,
        )
        id_to_owner: dict[str, int] = {}
        for p in points:
            payload = p.payload or {}
            if "id" not in payload:
                raise ValueError(f"doc_id 的 payload 缺少 id 字段: {p.id}")
            id_to_owner[str(p.id)] = int(payload.get("id", 0))

        missing = [x for x in doc_keys if x not in id_to_owner]
        if missing:
            raise ValueError(f"未找到 doc_id 对应的记录: {missing}")

        owners = [id_to_owner[x] for x in doc_keys]
        if return_type == "profile":
            profile_cache: dict[int, UserProfileWithDescriptionIds] = {}
            profiles: list[UserProfileWithDescriptionIds] = []
            for owner_user_id in owners:
                if owner_user_id not in profile_cache:
                    profile_cache[owner_user_id] = await self.get_user_profiles(user_id=owner_user_id)
                profiles.append(profile_cache[owner_user_id])

            if isinstance(doc_id, str):
                return profiles[0]
            return profiles

        if isinstance(doc_id, str):
            return owners[0]
        return owners

    async def delete_by_doc_id(self, doc_id: str | list[str]) -> int:
        """按 doc_id（point id）删除记录。

        Args:
            doc_id: Qdrant point id（单个或列表）。

        Returns:
            int: 预计删除的记录条数（等于输入 doc_id 数量）。

        Raises:
            ValueError: doc_id 为空。
        """

        doc_ids_raw: list[str] = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids_raw:
            raise ValueError("doc_id 不能为空")

        qdrant_ids: list[Any] = []
        doc_keys: list[str] = []
        for item in doc_ids_raw:
            qid, key = self._normalize_point_id(str(item))
            qdrant_ids.append(qid)
            doc_keys.append(key)

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )
        return len(doc_keys)

    async def delete_all_by_user_id(self, user_id: int, *, page_size: int = 1000) -> int:
        """删除指定用户 ID 下的所有画像数据。

        说明：
            - Qdrant 的 `scroll` 支持分页，本方法会先分页拿到所有 point id，
              再分批调用 `delete`，避免一次性请求过大。

        Args:
            user_id: 用户 ID (QQ号)。
            page_size: 分页大小（同时用于删除批大小）。

        Returns:
            int: 实际删除的记录条数。

        Raises:
            ValueError: page_size 非法。
        """

        if page_size <= 0:
            raise ValueError("page_size 必须为正整数")

        all_ids: list[str] = []
        offset: Any = None
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_user_filter(int(user_id)),
                limit=min(int(page_size), 10_000),
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            if not batch:
                break

            all_ids.extend([str(p.id) for p in batch])

            if next_offset is None:
                break
            offset = next_offset

        if not all_ids:
            return 0

        deleted = 0
        for i in range(0, len(all_ids), int(page_size)):
            chunk = all_ids[i : i + int(page_size)]
            qdrant_ids: list[Any] = []
            for item in chunk:
                qid, _ = self._normalize_point_id(str(item))
                qdrant_ids.append(qid)
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=qdrant_ids),
            )
            deleted += len(chunk)

        return deleted
