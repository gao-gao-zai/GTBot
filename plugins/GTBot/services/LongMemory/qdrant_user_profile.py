"""Qdrant 用户画像管理器兼容模块。

该模块用于对齐历史文档/代码中提到的路径：
`plugins/GTBot/services/LongMemory/qdrant_user_profile.py`。

当前项目的 Qdrant 实现实际位于 [UserProfile.py](UserProfile.py) 中。
这里仅做重导出（re-export），避免出现“文档说有这个文件但实际不存在”的问题。

Attributes:
    QdrantUserProfile: Qdrant 后端的用户画像服务实现类。
    UserProfile: 兼容别名，等同于 `QdrantUserProfile`。
"""

from __future__ import annotations

from .UserProfile import QdrantUserProfile, UserProfile

__all__ = [
    "QdrantUserProfile",
    "UserProfile",
]
