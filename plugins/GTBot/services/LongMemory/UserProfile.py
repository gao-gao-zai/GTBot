from __future__ import annotations

from typing import TypeVar, cast

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance

from .qdrant_user_profile import QdrantUserProfile
from .VectorGenerator import VectorGenerator


TUserProfile = TypeVar("TUserProfile", bound="UserProfile")


class UserProfile(QdrantUserProfile):
    """用户画像的存储与检索服务（默认 Qdrant 后端）。

    说明：
        本项目早期版本提供过基于 Chroma 的 `UserProfile` 实现；目前已切换为 Qdrant。
        为了保持上层 import 路径稳定（仍可 `from ...UserProfile import UserProfile`），
        这里直接继承 `QdrantUserProfile` 并复用其全部能力。

    备份：
        Chroma 版本源码已移动到：
        `plugins/GTBot/services/LongMemory/chroma_backup/20260207/UserProfile.py`

    使用：
        - 创建：`await UserProfile.create(...)`
        - 写入/检索/更新/删除：与 `QdrantUserProfile` 一致。
    """

    @classmethod
    async def create(
        cls: type[TUserProfile],
        *,
        collection_name: str,
        client: AsyncQdrantClient,
        vector_generator: VectorGenerator,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
    ) -> TUserProfile:
        """创建用户画像服务实例，并确保 Qdrant collection 存在。

        Args:
            collection_name: Qdrant collection 名称。
            client: Qdrant 异步客户端。
            vector_generator: 向量生成器。
            vector_size: 向量维度；不传时会通过一次 embedding 探测维度。
            distance: 距离度量方式，默认余弦距离（COSINE）。

        Returns:
            UserProfile: 服务实例。
        """

        instance = await super().create(
            collection_name=collection_name,
            client=client,
            vector_generator=vector_generator,
            vector_size=vector_size,
            distance=distance,
        )

        return cast(TUserProfile, instance)


__all__ = ["UserProfile", "QdrantUserProfile"]

