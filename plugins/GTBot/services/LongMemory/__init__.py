
"""LongMemory 服务。

对外导出中短期记忆记事本相关工具，并提供基于配置的
`SessionNotepadManager` 单例获取入口。
"""

from __future__ import annotations
from typing import Literal
from pathlib import Path
import os


from numpy import long
from qdrant_client import AsyncQdrantClient



from . import notepad
from . import VectorGenerator
from . import UserProfile
from . import GroupProfileSQLite
from . import EventLogManager
from . import PublicKnowledge



# 群/长期记忆相关的本地 SQLite 数据库存放路径（与本模块同目录）。
GROUP_PROFILE_DB_PATH: Path = Path(__file__).parent.resolve() / "long_memory.db"


class LongMemoryContainer:
	"""LongMemory 服务容器。

	该类作为 LongMemory 服务的统一访问入口，封装了相关子服务的实例。
	"""
	def __init__(
        self, 
        vector_generator: VectorGenerator.BaseVectorGenerator,
        notepad_manager: notepad.SessionNotepadManager,
		user_profile_manager: UserProfile.QdrantUserProfile,
		group_profile_manager: GroupProfileSQLite.SQLiteGroupProfileStore,
		event_log_manager: EventLogManager.QdrantEventLogManager,
		public_knowledge: PublicKnowledge.QdrantPublicKnowledge,
    ) -> None:
		"""初始化 LongMemory 服务容器。

		Args:
			vector_generator: 向量生成器实例（负责把文本转为向量）。
			notepad_manager: 会话记事本管理器实例。
			user_profile_manager: 用户画像管理器实例（Qdrant 后端）。
			group_profile_manager: 群画像管理器实例（SQLite 后端）。
			event_log_manager: 事件日志管理器实例（Qdrant 后端）。
			public_knowledge: 公共知识库实例（Qdrant 后端）。
		"""
		self.vector_generator: VectorGenerator.BaseVectorGenerator = vector_generator
		self.notepad_manager: notepad.SessionNotepadManager = notepad_manager
		self.user_profile_manager: UserProfile.QdrantUserProfile = user_profile_manager
		self.group_profile_manager: GroupProfileSQLite.SQLiteGroupProfileStore = group_profile_manager
		self.event_log_manager: EventLogManager.QdrantEventLogManager = event_log_manager
		self.public_knowledge: PublicKnowledge.QdrantPublicKnowledge = public_knowledge

	@classmethod
	def create(
		cls,
		qdrant_server_url: str,
		embed_service_url: str,
		embed_model: str,
		embed_api_key: str | None = None,
        qdrant_api_key: str | None = None,
		embed_service_type: Literal["openai"] = "openai",
		notepad_max_length: int = 15,
		max_sessions: int|None = None,
		session_timeout_seconds: float | None = None,
        qdrant_collection_name: str = "long_memory",
	) -> "LongMemoryContainer":
		"""创建并初始化 LongMemoryContainer。

		该工厂方法会完成：
		- 初始化向量生成器（当前仅支持 OpenAI 兼容嵌入接口）
		- 创建 Qdrant 客户端
		- 构造用户画像 / 事件日志 / 公共知识库等子服务

		Args:
			qdrant_server_url: Qdrant 服务地址。
			embed_service_url: 嵌入（Embedding）服务地址。
			embed_model: 嵌入模型名称。
			embed_api_key: 嵌入服务的 API Key（可选）。
			qdrant_api_key: Qdrant 的 API Key（可选）。
			embed_service_type: 嵌入服务类型（当前仅支持 "openai"）。
			notepad_max_length: 会话记事本最大条目数。
			max_sessions: 允许缓存的最大会话数（None 表示不限制）。
			session_timeout_seconds: 会话过期时间（秒），None 表示不过期。
			qdrant_collection_name: Qdrant collection 名称。

		Returns:
			LongMemoryContainer: 初始化完成的服务管理器。

		Raises:
			ValueError: 当 embed_service_type 不受支持时抛出。
		"""
		if embed_api_key is None:
			embed_api_key = ""

		if embed_service_type == "openai":
			vector_generator = VectorGenerator.OpenaiVectorGenerator(
				model_name=embed_model,
				api_url=embed_service_url,
				api_key=embed_api_key,
			)
		else:
			raise ValueError(f"不支持的嵌入服务类型：{embed_service_type}")

		qdrant_client = AsyncQdrantClient(
			url=qdrant_server_url,
			api_key=qdrant_api_key,
		)

		notepad_manager = notepad.SessionNotepadManager(
			notepad_max_length=notepad_max_length,
			max_sessions=max_sessions,
			session_timeout_seconds=session_timeout_seconds
		)

		user_profile_manager = UserProfile.QdrantUserProfile(
            collection_name=qdrant_collection_name,
			client=qdrant_client,
			vector_generator=vector_generator,
		)
		group_profile_manager = GroupProfileSQLite.SQLiteGroupProfileStore(db_path=GROUP_PROFILE_DB_PATH)
		event_log_manager = EventLogManager.QdrantEventLogManager(
			collection_name=qdrant_collection_name,
			client=qdrant_client,
			vector_generator=vector_generator,
		)
		public_knowledge = PublicKnowledge.QdrantPublicKnowledge(
			collection_name=qdrant_collection_name,
			client=qdrant_client,
			vector_generator=vector_generator,
		)
		return cls(
			vector_generator=vector_generator,
			notepad_manager=notepad_manager, 
			user_profile_manager=user_profile_manager,
			group_profile_manager=group_profile_manager,
			event_log_manager=event_log_manager,
			public_knowledge=public_knowledge,
		)





_AUTO_INIT = os.getenv("GTBOT_LONGMEMORY_AUTOINIT", "1").strip() not in {"0", "false", "False"}

# 兼容现有行为：默认导入即初始化 long_memory_manager。
# CLI/脚本测试可通过设置环境变量 `GTBOT_LONGMEMORY_AUTOINIT=0` 来关闭。
if _AUTO_INIT:
    long_memory_manager = LongMemoryContainer.create(
        qdrant_server_url="http://172.26.226.57:6333",
        embed_service_url="http://172.26.226.57:30020/v1/embeddings",
		embed_model="qwen3-embedding-0.6b",
		qdrant_api_key="",
		embed_api_key="",
		embed_service_type="openai",
		notepad_max_length=20,
		max_sessions=1000,
		session_timeout_seconds=3600,
		qdrant_collection_name="long_memory",
	)
