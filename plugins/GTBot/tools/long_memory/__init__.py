
"""LongMemory 服务。

对外导出中短期记忆记事本相关工具，并提供基于配置的
`SessionNotepadManager` 单例获取入口。
"""

from __future__ import annotations
import re
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING, cast
from pathlib import Path
import asyncio
import time

from importlib import import_module

from langchain.tools import ToolRuntime, tool as lc_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context

from .tool import (
	_impl_search_event_log_info,
	_impl_search_group_profile_info,
	_impl_search_public_knowledge,
	_impl_search_user_profile_info,
	normalize_session_id,
)


lc_tool = cast(Any, lc_tool)


LongMemoryRecallConfig = Any
LongMemoryRecallManager = Any
LongMemoryIngestConfig = Any
LongMemoryIngestManager = Any


from qdrant_client import AsyncQdrantClient



from . import notepad
from . import VectorGenerator
from . import UserProfile
from . import GroupProfileQdrant
from . import EventLogManager
from . import PublicKnowledge
from .config import get_long_memory_plugin_config


# ============================================================================
# 长期记忆入库管理器（消息缓冲 -> 触发 -> LLM 工具整理）
# ============================================================================


_ingest_manager = None


_recall_manager = None


_recall_seen_message_keys_by_session: dict[str, set[tuple[int, int, float]]] = {}


_recall_seen_message_key_order_by_session: dict[str, list[tuple[int, int, float]]] = {}


_ingest_last_db_id_by_session: dict[str, int] = {}


_post_llm_ingest_tasks: dict[str, asyncio.Task[Any]] = {}


_POST_LLM_INGEST_RECENT_N = 20


_POST_LLM_INGEST_DELAY_SECONDS = 0.3

def get_long_memory_ingest_manager(
	*,
	config: "LongMemoryIngestConfig | None" = None,
) -> "LongMemoryIngestManager | None":
	"""获取长期记忆入库管理器单例（懒加载）。

	说明：
		- 为避免循环导入，函数内部再导入 IngestManager。
		- 入库管理器仅在显式提供 `config` 时创建，避免隐式读取配置或产生副作用。

	Returns:
		LongMemoryIngestManager | None: 入库管理器；若 LongMemory 未初始化则返回 None。
	"""

	global _ingest_manager
	if _ingest_manager is not None:
		return _ingest_manager

	if config is None:
		return None

	long_memory = globals().get("long_memory_manager", None)
	if long_memory is None:
		return None

	from .IngestManager import LongMemoryIngestManager

	try:
		_ingest_manager = LongMemoryIngestManager(config=config, long_memory=long_memory)
		return _ingest_manager
	except Exception as exc:
		from nonebot import logger as _nb_logger
		_nb_logger.error(f"LongMemory 入库管理器初始化失败: {exc}")
		return None


def get_long_memory_recall_manager(
	*,
	config: "LongMemoryRecallConfig | None" = None,
) -> "LongMemoryRecallManager | None":
	"""获取长期记忆出库（召回）管理器单例（懒加载）。

	说明：
		- 为避免循环导入，函数内部再导入 RecallManager。
		- 召回管理器仅在显式提供 `config` 时创建，避免隐式读取配置或产生副作用。

	Returns:
		LongMemoryRecallManager | None: 召回管理器；若 LongMemory 未初始化则返回 None。
	"""

	global _recall_manager
	if _recall_manager is not None:
		return _recall_manager

	if config is None:
		return None

	long_memory = globals().get("long_memory_manager", None)
	if long_memory is None:
		return None

	module = import_module(".RecallManager", package=__name__)
	LongMemoryRecallManager = getattr(module, "LongMemoryRecallManager")

	try:
		_recall_manager = LongMemoryRecallManager(config=config, long_memory=long_memory)
		return _recall_manager
	except Exception as exc:
		from nonebot import logger as _nb_logger
		_nb_logger.error(f"LongMemory 召回管理器初始化失败: {exc}")
		return None


def _format_related_long_memories(related: Any) -> str:
	if related is None:
		return ""

	lines: list[str] = []
	lines.append("<long_term_memory_retrieval_hit>")

	query = getattr(related, "query", "")
	if isinstance(query, str) and query.strip():
		lines.append(f"- query: {query.strip()}")

	def _take(items: Any) -> list[Any]:
		return items if isinstance(items, list) else []

	events = _take(getattr(related, "event_logs", None))
	if events:
		lines.append("<event>")
		for it in events:
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			details = getattr(it, "details", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- [{sid}] (similarity={float(sim):.3f}) {details}")
			else:
				lines.append(f"- [{sid}] {details}")
		lines.append("</event>")

	knowledge = _take(getattr(related, "public_knowledge", None))
	if knowledge:
		lines.append("<knowledge>")
		for it in knowledge:
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			title = getattr(it, "title", "")
			content = getattr(it, "content", "")
			text = f"{title}: {content}".strip(": ")
			if isinstance(sim, (int, float)):
				lines.append(f"- [{sid}] (similarity={float(sim):.3f}) {text}")
			else:
				lines.append(f"- [{sid}] {text}")
		lines.append("</knowledge>")

	user_profiles = _take(getattr(related, "user_profiles", None))
	if user_profiles:
		lines.append("<user_profile>")
		for it in user_profiles:
			uid = getattr(it, "user_id", "")
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			text = getattr(it, "text", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- user_id={uid} [{sid}] (similarity={float(sim):.3f}) {text}")
			else:
				lines.append(f"- user_id={uid} [{sid}] {text}")
		lines.append("</user_profile>")

	group_hits = _take(getattr(related, "group_profile_hits", None))
	if group_hits:
		lines.append("<group_porfile>")
		for it in group_hits:
			gid = getattr(it, "group_id", "")
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			text = getattr(it, "text", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- group_id={gid} [{sid}] (similarity={float(sim):.3f}) {text}")
			else:
				lines.append(f"- group_id={gid} [{sid}] {text}")
		lines.append("</group_porfile>")

	if not query.strip() and not events and not knowledge and not user_profiles and not group_hits:
		return ""

	lines.append("</long_term_memory_retrieval_hit>")
	return "\n".join([str(x) for x in lines if str(x).strip()])


async def prepare_long_memory_recall(plugin_ctx: Any) -> None:
	"""在 agent 启动前准备当前会话的长期记忆召回结果。

	Args:
		plugin_ctx: 当前请求对应的插件上下文，用于写入共享缓存。
		runtime_context: 当前请求的运行时上下文，用于推断会话 ID 与读取消息快照。

	Returns:
		None: 该函数通过 `plugin_ctx.extra` 输出召回结果，不直接返回值。
	"""

	recall_manager = None
	session_id: str | None = None
	try:
		if plugin_ctx is None:
			logger.debug("LongMemory recall skipped: plugin_ctx is None")
			return
		if plugin_ctx.extra.get("_long_memory_recall_prepared"):
			logger.debug("LongMemory recall skipped: already prepared")
			return
		runtime_context = getattr(plugin_ctx, "runtime_context", None)
		if runtime_context is None:
			logger.debug("LongMemory recall skipped: runtime_context is None")
			return

		session_id = _infer_session_id_from_runtime_context(runtime_context)
		if not session_id:
			logger.debug("LongMemory recall skipped: cannot infer session_id")
			return

		raw_messages = getattr(runtime_context, "raw_messages", [])
		raw_messages_list = raw_messages if isinstance(raw_messages, list) else []
		if not raw_messages_list:
			logger.debug(f"LongMemory recall skipped: no raw_messages (session={session_id})")
			return

		module = import_module(".RecallManager", package=__name__)
		LongMemoryRecallConfigCls = getattr(module, "LongMemoryRecallConfig", None)
		if LongMemoryRecallConfigCls is None:
			return
		try:
			cfg = get_long_memory_plugin_config()
			recall_cfg = getattr(cfg, "recall", None)
			payload = dict(recall_cfg.model_dump()) if recall_cfg is not None else {}
			config_obj = LongMemoryRecallConfigCls(**payload)
		except Exception:
			config_obj = LongMemoryRecallConfigCls()

		recall_manager = get_long_memory_recall_manager(config=config_obj)
		if recall_manager is None:
			logger.warning(f"LongMemory recall skipped: recall_manager initialization failed (session={session_id})")
			return

		max_k = int(getattr(config_obj, "query_max_messages", 12) or 12)
		candidates = raw_messages_list[-max_k:]
		logger.debug(f"LongMemory recall starting: session={session_id} raw_messages={len(raw_messages_list)} candidates={len(candidates)}")

		seen = _recall_seen_message_keys_by_session.get(session_id)
		if seen is None:
			seen = set()
			_recall_seen_message_keys_by_session[session_id] = seen
		order = _recall_seen_message_key_order_by_session.get(session_id)
		if order is None:
			order = []
			_recall_seen_message_key_order_by_session[session_id] = order

		new_messages: list[Any] = []
		for m in candidates:
			mid = int(getattr(m, "message_id", 0) or 0)
			uid = int(getattr(m, "user_id", 0) or 0)
			ts = float(getattr(m, "send_time", 0.0) or 0.0)
			key = (mid, uid, ts)
			if key in seen:
				continue
			seen.add(key)
			order.append(key)
			new_messages.append(m)

		max_seen = 500
		if len(order) > max_seen:
			drop = len(order) - max_seen
			for _ in range(drop):
				old = order.pop(0)
				seen.discard(old)

		if not new_messages:
			new_messages = raw_messages_list[-1:]

		group_id = getattr(runtime_context, "group_id", None)
		group_id = int(group_id) if isinstance(group_id, int) and group_id > 0 else None
		user_id = getattr(runtime_context, "user_id", None)
		user_id = int(user_id) if isinstance(user_id, int) and user_id > 0 else None

		await recall_manager.add_message(
			session_id=session_id,
			messages=new_messages,
			group_id=group_id,
			user_id=user_id,
		)

		related = await recall_manager.get_current_related_memories(
			session_id=session_id,
			group_id=group_id,
			user_id=user_id,
			force_refresh=False,
		)

		plugin_ctx.extra["long_memory_recall_config"] = config_obj
		plugin_ctx.extra["long_memory_related_memories"] = related
		plugin_ctx.extra["_long_memory_recall_prepared"] = True
		logger.info(f"LongMemory recall completed: session={session_id} has_related={related is not None}")
		return
	except Exception as exc:
		logger.error(f"LongMemory recall failed: session={session_id} error={exc}", exc_info=True)
		if plugin_ctx is not None and isinstance(session_id, str) and recall_manager is not None:
			cached_related = _get_recall_manager_cached_related_memories(
				recall_manager=recall_manager,
				session_id=session_id,
			)
			if cached_related is not None:
				plugin_ctx.extra["long_memory_related_memories"] = cached_related
				plugin_ctx.extra["_long_memory_recall_prepared"] = True
		return

class LongMemoryContainer:
	"""LongMemory 服务容器。

	该类作为 LongMemory 服务的统一访问入口，封装了相关子服务的实例。
	"""
	def __init__(
        self, 
        vector_generator: VectorGenerator.BaseVectorGenerator,
        notepad_manager: notepad.SessionNotepadManager,
		user_profile_manager: UserProfile.QdrantUserProfile,
		group_profile_manager: Any,
		event_log_manager: EventLogManager.QdrantEventLogManager,
		public_knowledge: PublicKnowledge.QdrantPublicKnowledge,
		reranker: Any = None,
    ) -> None:
		"""初始化 LongMemory 服务容器。

		Args:
			vector_generator: 向量生成器实例（负责把文本转为向量）。
			notepad_manager: 会话记事本管理器实例。
			user_profile_manager: 用户画像管理器实例（Qdrant 后端）。
			group_profile_manager: 群画像管理器实例（Qdrant 后端）。
			event_log_manager: 事件日志管理器实例（Qdrant 后端）。
			public_knowledge: 公共知识库实例（Qdrant 后端）。
		"""
		self.vector_generator: VectorGenerator.BaseVectorGenerator = vector_generator
		self.notepad_manager: notepad.SessionNotepadManager = notepad_manager
		self.user_profile_manager: UserProfile.QdrantUserProfile = user_profile_manager
		self.group_profile_manager: Any = group_profile_manager
		self.event_log_manager: EventLogManager.QdrantEventLogManager = event_log_manager
		self.public_knowledge: PublicKnowledge.QdrantPublicKnowledge = public_knowledge
		self.reranker: Any = reranker

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
		enable_rerank: bool = False,
		rerank_service_url: str | None = None,
		rerank_model: str | None = None,
		rerank_api_key: str | None = None,
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

		normalized_qdrant_api_key = str(qdrant_api_key).strip() if qdrant_api_key is not None else ""
		qdrant_client = AsyncQdrantClient(
			url=qdrant_server_url,
			api_key=normalized_qdrant_api_key or None,
			timeout=60,
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
		group_profile_cls = getattr(GroupProfileQdrant, "QdrantGroupProfileManager", None)
		if group_profile_cls is None:
			raise RuntimeError("无法加载 QdrantGroupProfileManager")
		group_profile_manager = group_profile_cls(
			collection_name=qdrant_collection_name,
			client=qdrant_client,
			vector_generator=vector_generator,
		)
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

		reranker = None
		if bool(enable_rerank) and isinstance(rerank_service_url, str) and rerank_service_url.strip():
			try:
				module = import_module(".RecallManager", package=__name__)
				RerankerCls = getattr(module, "TEIReranker", None)
				if RerankerCls is not None:
					reranker = RerankerCls(
						api_url=str(rerank_service_url).strip(),
						model_name=str(rerank_model or "").strip(),
						api_key=rerank_api_key,
					)
			except Exception:
				reranker = None
		return cls(
			vector_generator=vector_generator,
			notepad_manager=notepad_manager, 
			user_profile_manager=user_profile_manager,
			group_profile_manager=group_profile_manager,
			event_log_manager=event_log_manager,
			public_knowledge=public_knowledge,
			reranker=reranker,
		)





try:
	_AUTO_INIT = bool(get_long_memory_plugin_config().auto_init)
except Exception:
	_AUTO_INIT = True

# 兼容现有行为：默认导入即初始化 long_memory_manager。
# CLI/脚本测试可通过设置配置 `auto_init=false` 来关闭。
if _AUTO_INIT:
	_cfg = get_long_memory_plugin_config()
	_container_cfg = getattr(_cfg, "container", None)
	_rerank_cfg = getattr(_cfg, "rerank", None)
	long_memory_manager = LongMemoryContainer.create(
		qdrant_server_url=str(getattr(_container_cfg, "qdrant_server_url", "http://localhost:6333/")),
		embed_service_url=str(getattr(_container_cfg, "embed_service_url", "http://localhost:30020/v1/embeddings")),
		embed_model=str(getattr(_container_cfg, "embed_model", "qwen3-embedding-0.6b")),
		qdrant_api_key=getattr(_container_cfg, "qdrant_api_key", ""),
		embed_api_key=getattr(_container_cfg, "embed_api_key", ""),
		embed_service_type=cast(Any, getattr(_container_cfg, "embed_service_type", "openai")),
		notepad_max_length=int(getattr(_container_cfg, "notepad_max_length", 20)),
		max_sessions=getattr(_container_cfg, "max_sessions", 1000),
		session_timeout_seconds=getattr(_container_cfg, "session_timeout_seconds", 3600),
		qdrant_collection_name=str(getattr(_container_cfg, "qdrant_collection_name", "long_memory")),
		enable_rerank=bool(getattr(_rerank_cfg, "enable", False)),
		rerank_service_url=str(getattr(_rerank_cfg, "service_url", "http://localhost:30021/v1/rerank")),
		rerank_model=str(getattr(_rerank_cfg, "model", "")),
		rerank_api_key=getattr(_rerank_cfg, "api_key", None),
	)

	# 可选：自动初始化入库管理器单例。
	# 注意：此处不自动初始化入库管理器，避免在导入 LongMemory 时产生额外副作用。


_NOTE_TAG_PATTERN = re.compile(r"<note>(.*?)</note>", flags=re.IGNORECASE | re.DOTALL)


def _extract_note_tags(content: str) -> tuple[list[str], str]:
	if not content:
		return [], ""
	notes = [n.strip() for n in _NOTE_TAG_PATTERN.findall(content) if n and n.strip()]
	remaining = _NOTE_TAG_PATTERN.sub("", content).strip()
	return notes, remaining


def _normalize_notepad_session_id(raw_session_id: str, *, default_group_id: int | None) -> str:
	raw = (raw_session_id or "").strip()
	if not raw:
		if default_group_id is None:
			raise ValueError("缺少默认群号，无法推断会话 ID")
		return normalize_session_id(f"group_{default_group_id}")
	if raw.isdigit():
		return normalize_session_id(f"group_{int(raw)}")
	return normalize_session_id(raw)


def _infer_session_id_from_runtime_context(context: Any) -> str | None:
	"""从运行时上下文中推断并规范化会话 ID。

	Args:
		context: GTBot 当前请求的运行时上下文，通常为 `GroupChatContext`。

	Returns:
		str | None: 规范化后的会话 ID；若上下文缺失必要信息则返回 `None`。
	"""

	if context is None:
		return None
	session_id = getattr(context, "session_id", None)
	if isinstance(session_id, str) and session_id.strip():
		return normalize_session_id(session_id.strip())
	group_id = getattr(context, "group_id", None)
	if isinstance(group_id, int) and group_id > 0:
		return normalize_session_id(f"group_{group_id}")
	user_id = getattr(context, "user_id", None)
	if isinstance(user_id, int) and user_id > 0:
		return normalize_session_id(f"private_{user_id}")
	return None


def _infer_session_id_from_runtime(runtime: Any) -> str | None:
	"""从 LangChain runtime 中推断并规范化会话 ID。

	Args:
		runtime: Agent middleware / tool 运行时对象，要求包含 `context` 字段。

	Returns:
		str | None: 规范化后的会话 ID；若无法推断则返回 `None`。
	"""

	return _infer_session_id_from_runtime_context(getattr(runtime, "context", None))


def _find_primary_human_message_index(messages: list[Any]) -> int:
	insert_at = 0
	while insert_at < len(messages) and isinstance(messages[insert_at], SystemMessage):
		insert_at += 1
	for index in range(insert_at, len(messages)):
		if isinstance(messages[index], HumanMessage):
			return index
	return -1


def _merge_text_into_primary_human_message(
	*,
	messages: list[Any],
	text: str,
	prepend: bool,
) -> list[Any]:
	"""将文本合并到主 HumanMessage 中。

	Args:
		messages: 当前最终会送入 LLM 的消息列表。
		text: 需要注入的文本片段。
		prepend: 是否前置到主 HumanMessage 内容前面。

	Returns:
		list[Any]: 合并后的消息列表；若不存在主 HumanMessage，则插入到起始
		`SystemMessage` 连续块之后。
	"""

	merged_text = str(text or "").strip()
	if not merged_text:
		return list(messages)

	updated_messages = list(messages)
	index = _find_primary_human_message_index(updated_messages)
	if index < 0:
		insert_at = 0
		while insert_at < len(updated_messages) and isinstance(updated_messages[insert_at], SystemMessage):
			insert_at += 1
		return (
			list(updated_messages[:insert_at])
			+ [HumanMessage(merged_text)]
			+ list(updated_messages[insert_at:])
		)

	current = updated_messages[index]
	content = getattr(current, "content", "")
	if isinstance(content, str) and content.strip():
		new_content = merged_text + "\n\n" + content if prepend else content.rstrip() + "\n\n" + merged_text
	else:
		new_content = merged_text
	updated_messages[index] = HumanMessage(new_content)
	return updated_messages


def _get_recall_manager_cached_related_memories(
	*,
	recall_manager: Any,
	session_id: str,
) -> Any | None:
	"""最佳努力读取 recall manager 中当前会话的旧缓存结果。

	Args:
		recall_manager: 当前使用的 `LongMemoryRecallManager` 实例。
		session_id: 规范化后的会话 ID。

	Returns:
		Any | None: 缓存的 `RelatedLongMemories`；若无缓存或结构不可用则返回 `None`。
	"""

	try:
		sessions = getattr(recall_manager, "_sessions", None)
		if not isinstance(sessions, dict):
			return None
		state = sessions.get(session_id)
		return getattr(state, "related", None) if state is not None else None
	except Exception:
		return None


def _build_long_memory_notepad_context(*, runtime_context: Any) -> str:
	"""构造当前会话的记事本注入文本。

	Args:
		runtime_context: 当前请求的运行时上下文，用于推断会话 ID。

	Returns:
		str: `<note>...</note>` 文本；若当前会话没有可注入记事本则返回空串。
	"""

	session_id = _infer_session_id_from_runtime_context(runtime_context)
	if not session_id:
		return ""

	manager = globals().get("long_memory_manager", None)
	if manager is None:
		return ""

	notepad_manager = getattr(manager, "notepad_manager", None)
	if notepad_manager is None or not callable(getattr(notepad_manager, "has_session", None)):
		return ""
	if not notepad_manager.has_session(session_id):
		return ""

	get_notes = getattr(notepad_manager, "get_notes", None)
	if not callable(get_notes):
		return ""
	notes_text = str(get_notes(session_id) or "").strip()
	if not notes_text:
		return ""
	return "<note>\n" + notes_text + "\n</note>"


@lc_tool("take_notes")
async def take_notes(note: str, runtime: ToolRuntime[Any]) -> str:
	"""
	用于将文本记录到当前会话的记事本中。
	工具参数:
		note: 要记录的文本内容。
	返回值:
		str: 记录结果的描述信息。
	"""

	context = getattr(runtime, "context", None)
	if context is None:
		return "记事本工具缺少运行上下文。"

	session_id = _infer_session_id_from_runtime(runtime)
	if not session_id:
		return "记事本工具无法推断会话 ID。"

	manager = globals().get("long_memory_manager", None)
	if manager is None:
		return "LongMemory 未初始化，无法写入记事本。"

	manager.notepad_manager.add_note(session_id, str(note))
	return "已添加记事本记录。"


async def inject_long_memory_context(plugin_ctx: Any, messages: list[Any]) -> list[Any]:
	"""将 recall 与记事本内容一次性合并进主 HumanMessage。

	Args:
		plugin_ctx: 当前请求的插件上下文，需提供运行时上下文与 recall 缓存。
		messages: 当前最终会送入 LLM 的消息列表。

	Returns:
		list[Any]: 注入长期记忆内容后的消息列表；若没有可注入内容则返回原列表副本。
	"""

	try:
		sections: list[str] = []
		related = plugin_ctx.extra.get("long_memory_related_memories")
		recall_text = _format_related_long_memories(related)
		if recall_text:
			sections.append(recall_text)
			plugin_ctx.extra["_long_memory_recall_injected"] = True

		runtime_context = getattr(plugin_ctx, "runtime_context", None)
		notepad_text = _build_long_memory_notepad_context(runtime_context=runtime_context)
		if notepad_text:
			sections.append(notepad_text)
			plugin_ctx.extra["_long_memory_notepad_injected"] = True

		if not sections:
			return list(messages)

		return _merge_text_into_primary_human_message(
			messages=list(messages),
			text="\n\n".join(sections),
			prepend=True,
		)
	except Exception:
		return list(messages)


async def _post_llm_ingest_recent_messages(*, session_id: str, runtime_context: Any) -> None:
	"""在响应完成后延迟拉取最新消息并写入长期记忆。

	Args:
		session_id: 当前会话的规范化 ID。
		runtime_context: 当前请求的运行时上下文快照。

	Returns:
		None: 该函数只产生异步副作用，不返回结果。
	"""

	try:
		try:
			cfg = get_long_memory_plugin_config()
			post_cfg = getattr(cfg, "post_llm_ingest", None)
			recent_n = int(getattr(post_cfg, "recent_n", _POST_LLM_INGEST_RECENT_N))
			delay_seconds = float(getattr(post_cfg, "delay_seconds", _POST_LLM_INGEST_DELAY_SECONDS))
		except Exception:
			recent_n = int(_POST_LLM_INGEST_RECENT_N)
			delay_seconds = float(_POST_LLM_INGEST_DELAY_SECONDS)

		await asyncio.sleep(float(delay_seconds))

		context = runtime_context
		if context is None:
			return

		manager = globals().get("long_memory_manager", None)
		if manager is None:
			return

		message_manager = getattr(context, "message_manager", None)
		get_recent_messages = getattr(message_manager, "get_recent_messages", None)
		if not callable(get_recent_messages):
			return

		group_id = getattr(context, "group_id", None)
		group_id = int(group_id) if isinstance(group_id, int) and group_id > 0 else None
		user_id = getattr(context, "user_id", None)
		user_id = int(user_id) if isinstance(user_id, int) and user_id > 0 else None

		kwargs: dict[str, Any] = {
			"limit": int(recent_n),
			"asc_order": False,
		}
		if group_id is not None:
			kwargs["group_id"] = group_id
		elif user_id is not None:
			kwargs["user_id"] = user_id
		else:
			return

		get_recent_messages_fn = cast(Callable[..., Awaitable[list[Any]]], get_recent_messages)
		recent = await get_recent_messages_fn(**kwargs)
		recent_list = list(recent) if isinstance(recent, list) else []
		if not recent_list:
			return

		recent_list.reverse()

		last_db_id = int(_ingest_last_db_id_by_session.get(session_id, 0))
		batch: list[Any] = []
		max_db_id = last_db_id
		seen_keys: set[tuple[Any, ...]] = set()

		for m in recent_list:
			db_id_raw = getattr(m, "db_id", None)
			db_id = int(db_id_raw) if isinstance(db_id_raw, int) and db_id_raw > 0 else 0
			if db_id > 0:
				if db_id <= last_db_id:
					continue
				key = ("db", db_id)
				if key in seen_keys:
					continue
				seen_keys.add(key)
				batch.append(m)
				if db_id > max_db_id:
					max_db_id = db_id
				continue

			mid = int(getattr(m, "message_id", 0) or 0)
			uid = int(getattr(m, "user_id", 0) or 0)
			send_time = float(getattr(m, "send_time", 0.0) or 0.0)
			key2 = ("msg", mid, uid, send_time)
			if key2 in seen_keys:
				continue
			seen_keys.add(key2)
			batch.append(m)

		if not batch:
			return

		module = import_module(".IngestManager", package=__name__)
		LongMemoryIngestConfigCls = getattr(module, "LongMemoryIngestConfig", None)
		if LongMemoryIngestConfigCls is None:
			return
		try:
			cfg = get_long_memory_plugin_config()
			ingest_cfg = getattr(cfg, "ingest", None)
			payload = dict(ingest_cfg.model_dump()) if ingest_cfg is not None else {}
			config_obj = LongMemoryIngestConfigCls(**payload)
		except Exception:
			config_obj = LongMemoryIngestConfigCls()
		ingest_manager = get_long_memory_ingest_manager(config=config_obj)
		if ingest_manager is None:
			return

		for m in batch:
			await ingest_manager.add_message(
				session_id=session_id,
				message=m,
				group_id=group_id,
				user_id=user_id,
			)

		if max_db_id > last_db_id:
			_ingest_last_db_id_by_session[session_id] = max_db_id
		return
	except asyncio.CancelledError:
		return
	except Exception:
		return


def _schedule_post_llm_ingest_task(*, session_id: str, runtime_context: Any, event_loop: Any | None = None) -> None:
	"""为当前会话调度 post-LLM ingest 任务，并清理旧任务。

	Args:
		session_id: 当前会话的规范化 ID。
		runtime_context: 当前请求的运行时上下文快照。

	Returns:
		None: 该函数仅更新任务表并调度后台任务。
	"""

	async def _runner() -> None:
		try:
			await _post_llm_ingest_recent_messages(
				session_id=session_id,
				runtime_context=runtime_context,
			)
		finally:
			current_task = asyncio.current_task()
			if current_task is not None and _post_llm_ingest_tasks.get(session_id) is current_task:
				_post_llm_ingest_tasks.pop(session_id, None)

	def _start_task() -> None:
		old_task = _post_llm_ingest_tasks.get(session_id)
		if old_task is not None and not old_task.done():
			old_task.cancel()
		_post_llm_ingest_tasks[session_id] = asyncio.create_task(_runner())

	try:
		asyncio.get_running_loop()
	except RuntimeError:
		if event_loop is None or bool(getattr(event_loop, "is_closed", lambda: True)()):
			logger.warning(f"long_memory post-llm ingest skipped: no running event loop for session={session_id}")
			return
		event_loop.call_soon_threadsafe(_start_task)
		return

	_start_task()


class LongMemoryPostLLMIngestCallback(BaseCallbackHandler):
	"""在链路结束后调度长期记忆入库任务。"""

	def __init__(self) -> None:
		super().__init__()
		self._run_to_session: dict[str, str] = {}
		self._run_to_runtime_context: dict[str, Any] = {}
		self._run_to_event_loop: dict[str, Any] = {}

	def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> Any:
		if not isinstance(inputs, dict) or inputs.get("messages") is None:
			return None

		plugin_ctx = get_current_plugin_context()
		runtime_context = getattr(plugin_ctx, "runtime_context", None) if plugin_ctx is not None else None
		session_id = _infer_session_id_from_runtime_context(runtime_context)
		run_id = kwargs.get("run_id")
		if runtime_context is None or not session_id or run_id is None:
			return None

		key = str(run_id)
		try:
			event_loop = asyncio.get_running_loop()
		except RuntimeError:
			event_loop = getattr(runtime_context, "event_loop", None)
		self._run_to_session[key] = session_id
		self._run_to_runtime_context[key] = runtime_context
		self._run_to_event_loop[key] = event_loop
		return None

	def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
		run_id = kwargs.get("run_id")
		key = str(run_id) if run_id is not None else ""
		session_id = self._run_to_session.pop(key, "")
		runtime_context = self._run_to_runtime_context.pop(key, None)
		event_loop = self._run_to_event_loop.pop(key, None)
		if event_loop is None and runtime_context is not None:
			event_loop = getattr(runtime_context, "event_loop", None)
		if session_id and runtime_context is not None:
			_schedule_post_llm_ingest_task(
				session_id=session_id,
				runtime_context=runtime_context,
				event_loop=event_loop,
			)
		return None

	def on_chain_error(self, error: BaseException, **kwargs: Any) -> Any:
		run_id = kwargs.get("run_id")
		key = str(run_id) if run_id is not None else ""
		if key:
			self._run_to_session.pop(key, None)
			self._run_to_runtime_context.pop(key, None)
			self._run_to_event_loop.pop(key, None)
		return None


def register(registry) -> None:  # noqa: ANN001
	"""将长期记忆能力注册到 GTBot 插件系统。

	Args:
		registry: GTBot 插件注册表。

	Returns:
		None: 通过注册表挂载工具、前置处理器、消息注入器与 callback。
	"""

	registry.add_tool(take_notes)
	registry.add_pre_agent_processor(prepare_long_memory_recall, priority=-20, wait_until_complete=True)
	registry.add_pre_agent_message_injector(inject_long_memory_context, priority=20)
	registry.add_callback(LongMemoryPostLLMIngestCallback(), priority=-20)



from nonebot import on_command, logger
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.params import CommandArg, Depends

from plugins.GTBot.services.message import GroupMessageManager, get_message_manager
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, get_permission_manager


async def _ensure_long_memory_admin(user_id: int) -> None:
	permission_manager = get_permission_manager()
	await permission_manager.require_role(user_id, PermissionRole.ADMIN)


QueryRelatedLongMemories = on_command(
	"相关记忆",
	aliases={"查看相关记忆", "相关记忆列表", "longmem", "lm"},
	priority=-5,
	block=True,
)


@QueryRelatedLongMemories.handle()
async def handle_query_related_long_memories(
	bot: Bot,
	event: GroupMessageEvent,
	args: Message = CommandArg(),
) -> None:
	try:
		await _ensure_long_memory_admin(int(event.user_id))
	except PermissionError as exc:
		await QueryRelatedLongMemories.finish(str(exc))

	arg_text = args.extract_plain_text().strip()
	if arg_text == "列表":
		arg_text = ""
	default_group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
	try:
		session_id = _normalize_notepad_session_id(arg_text, default_group_id=default_group_id)
	except ValueError:
		await QueryRelatedLongMemories.finish(
			"该命令仅支持群聊默认会话；请显式提供会话ID，例如：/相关记忆 group_123"
		)

	module = import_module(".RecallManager", package=__name__)
	LongMemoryRecallConfigCls = getattr(module, "LongMemoryRecallConfig", None)
	if LongMemoryRecallConfigCls is None:
		await QueryRelatedLongMemories.finish("LongMemoryRecallConfig 不可用。")
	try:
		cfg = get_long_memory_plugin_config()
		recall_cfg = getattr(cfg, "recall", None)
		payload = dict(recall_cfg.model_dump()) if recall_cfg is not None else {}
		config_obj = LongMemoryRecallConfigCls(**payload)
	except Exception:
		config_obj = LongMemoryRecallConfigCls()

	recall_manager = get_long_memory_recall_manager(config=config_obj)
	if recall_manager is None:
		await QueryRelatedLongMemories.finish("LongMemory 未初始化或召回管理器不可用。")

	related = await recall_manager.get_current_related_memories(
		session_id=session_id,
		group_id=default_group_id,
		force_refresh=True,
	)

	text = _format_related_long_memories(related)
	if not text:
		await QueryRelatedLongMemories.finish("暂无相关记忆。")

	forward_threshold = 200
	forward_chunks = _split_long_text(text, max_total_chars=9000, max_chunk_chars=700)
	if len(text) >= forward_threshold:
		try:
			await _send_as_forward(bot=bot, event=event, chunks=forward_chunks, name="GTBot")
			return
		except ActionFailed as exc:
			logger.warning(f"合并转发发送失败，降级为分段发送: {exc!s}")
		except Exception as exc:
			logger.warning(f"合并转发发送异常，降级为分段发送: {type(exc).__name__}: {exc!s}")

	chunks = _split_long_text(text, max_total_chars=3000, max_chunk_chars=900)
	try:
		for c in chunks:
			await QueryRelatedLongMemories.send(c)
	except ActionFailed as exc:
		await QueryRelatedLongMemories.finish(f"发送失败（可能输出过长或风控超时）：{exc!s}")


SearchLongMemory = on_command(
	"记忆搜索",
	aliases={"搜索记忆", "searchmem"},
	priority=-5,
	block=True,
)


def _split_long_text(text: str, *, max_total_chars: int, max_chunk_chars: int) -> list[str]:
	s = (text or "").strip()
	if not s:
		return []
	if len(s) > max_total_chars:
		s = s[: max_total_chars - 20].rstrip() + "\n...（已截断）"

	chunks: list[str] = []
	while s:
		cut = min(len(s), max_chunk_chars)
		i = s.rfind("\n", 0, cut)
		if i <= 0:
			i = cut
		chunks.append(s[:i].rstrip())
		s = s[i:].lstrip("\n")
	return [c for c in chunks if c]


async def _send_as_forward(
	*,
	bot: Bot,
	event: GroupMessageEvent,
	chunks: list[str],
	name: str,
) -> None:
	if not chunks:
		return

	try:
		uin = int(str(getattr(bot, "self_id", "") or "").strip() or 0)
	except Exception:
		uin = 0

	nodes = [
		{
			"type": "node",
			"data": {
				"uin": uin,
				"name": name,
				"content": [{"type": "text", "data": {"text": c}}],
			},
		}
		for c in chunks
	]

	await bot.call_api("send_group_forward_msg", group_id=int(event.group_id), messages=nodes)


@SearchLongMemory.handle()
async def handle_search_long_memory(
	bot: Bot,
	event: GroupMessageEvent,
	args: Message = CommandArg(),
) -> None:
	try:
		await _ensure_long_memory_admin(int(event.user_id))
	except PermissionError as exc:
		await SearchLongMemory.finish(str(exc))

	query = args.extract_plain_text().strip()
	if not query:
		await SearchLongMemory.finish("用法：/记忆搜索 <关键词或短句>")

	manager = globals().get("long_memory_manager", None)
	if manager is None:
		await SearchLongMemory.finish("LongMemory 未初始化。")

	group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
	session_id = f"group_{group_id}" if isinstance(group_id, int) and group_id > 0 else "test_session"

	results = await asyncio.gather(
		_impl_search_event_log_info(manager, session_id=session_id, query=query, limit=5),
		_impl_search_public_knowledge(manager, query=query, limit=5),
		_impl_search_user_profile_info(manager, query=query, max_users=3, limit=5, mode="expand"),
		_impl_search_group_profile_info(
			manager,
			group_id=int(group_id) if isinstance(group_id, int) else 0,
			query=query,
			limit=5,
			similarity_threshold=0.0,
		),
	)

	out_lines = [
		"[向量检索] 记忆搜索结果：",
		"",
		"### 事件日志",
		str(results[0]).strip() or "无",
		"",
		"### 公共知识",
		str(results[1]).strip() or "无",
		"",
		"### 用户画像",
		str(results[2]).strip() or "无",
		"",
		"### 群画像",
		str(results[3]).strip() or "无",
	]

	text = "\n".join(out_lines).strip()

	forward_threshold = 900
	forward_chunks = _split_long_text(text, max_total_chars=9000, max_chunk_chars=700)
	if len(text) >= forward_threshold and len(forward_chunks) >= 2:
		try:
			await _send_as_forward(bot=bot, event=event, chunks=forward_chunks, name="GTBot")
			return
		except ActionFailed as exc:
			logger.warning(f"合并转发发送失败，降级为分段发送: {exc!s}")
		except Exception as exc:
			logger.warning(f"合并转发发送异常，降级为分段发送: {type(exc).__name__}: {exc!s}")

	chunks = _split_long_text(text, max_total_chars=3000, max_chunk_chars=900)
	try:
		for c in chunks:
			await SearchLongMemory.send(c)
	except ActionFailed as exc:
		await SearchLongMemory.finish(f"发送失败（可能输出过长或风控超时）：{exc!s}")


BenchmarkLongMemoryRecall = on_command(
	"召回测速",
	aliases={"测召回", "recallbench"},
	priority=-5,
	block=True,
)


ForceIngestLongMemory = on_command(
	"整理记忆",
	aliases={"立即整理记忆", "强制整理记忆", "记忆整理"},
	priority=-5,
	block=True,
)


@ForceIngestLongMemory.handle()
async def handle_force_ingest_long_memory(
	event: GroupMessageEvent,
	args: Message = CommandArg(),
	msg_mg: GroupMessageManager = Depends(get_message_manager),
) -> None:
	try:
		await _ensure_long_memory_admin(int(event.user_id))
	except PermissionError as exc:
		await ForceIngestLongMemory.finish(str(exc))

	arg_text = args.extract_plain_text().strip()
	default_group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
	default_user_id = int(event.user_id) if isinstance(event.user_id, int) and event.user_id > 0 else None

	try:
		cfg = get_long_memory_plugin_config()
		post_cfg = getattr(cfg, "post_llm_ingest", None)
		default_n = int(getattr(post_cfg, "recent_n", _POST_LLM_INGEST_RECENT_N))
	except Exception:
		default_n = int(_POST_LLM_INGEST_RECENT_N)

	n = default_n
	session_arg = ""
	parts = [p for p in arg_text.split() if p]
	if len(parts) == 1:
		if parts[0].isdigit():
			n = int(parts[0])
		else:
			session_arg = parts[0]
	elif len(parts) >= 2:
		if parts[0].isdigit():
			n = int(parts[0])
			session_arg = parts[1]
		elif parts[1].isdigit():
			session_arg = parts[0]
			n = int(parts[1])
		else:
			await ForceIngestLongMemory.finish(
				"用法：/整理记忆 [N] [session_id]（N 为整数；session_id 省略则默认当前群会话）"
			)

	if n <= 0:
		await ForceIngestLongMemory.finish("N 必须是正整数。")
	if n > 200:
		n = 200

	try:
		session_id = _normalize_notepad_session_id(session_arg, default_group_id=default_group_id)
	except ValueError:
		await ForceIngestLongMemory.finish(
			"该命令仅支持群聊默认会话；请显式提供会话ID，例如：/整理记忆 20 group_123"
		)

	ingest_manager = get_long_memory_ingest_manager()
	if ingest_manager is None:
		await ForceIngestLongMemory.finish("LongMemory 未初始化或入库管理器不可用。")

	group_id = int(default_group_id) if isinstance(default_group_id, int) and default_group_id > 0 else None
	if group_id is None:
		await ForceIngestLongMemory.finish("该命令仅支持群聊。")

	try:
		recent = await msg_mg.get_recent_messages(limit=int(n), group_id=group_id, asc_order=False)
		recent_list = list(recent) if isinstance(recent, list) else []
	except Exception as exc:
		logger.warning(f"整理记忆：读取最近消息失败: {type(exc).__name__}: {exc!s}")
		await ForceIngestLongMemory.finish("读取最近消息失败。")

	if not recent_list:
		await ForceIngestLongMemory.finish("没有可用的历史消息。")

	recent_list.reverse()

	last_db_id = int(_ingest_last_db_id_by_session.get(session_id, 0))
	added = 0
	max_db_id = last_db_id
	seen_keys: set[tuple[Any, ...]] = set()

	for m in recent_list:
		db_id_raw = getattr(m, "db_id", None)
		db_id = int(db_id_raw) if isinstance(db_id_raw, int) and db_id_raw > 0 else 0
		if db_id > 0:
			if db_id <= last_db_id:
				continue
			key = ("db", db_id)
			if key in seen_keys:
				continue
			seen_keys.add(key)
			await ingest_manager.add_message(
				session_id=session_id,
				message=m,
				group_id=group_id,
				user_id=default_user_id,
			)
			added += 1
			if db_id > max_db_id:
				max_db_id = db_id
			continue

		mid = int(getattr(m, "message_id", 0) or 0)
		uid = int(getattr(m, "user_id", 0) or 0)
		send_time = float(getattr(m, "send_time", 0.0) or 0.0)
		key2 = ("msg", mid, uid, send_time)
		if key2 in seen_keys:
			continue
		seen_keys.add(key2)
		await ingest_manager.add_message(
			session_id=session_id,
			message=m,
			group_id=group_id,
			user_id=default_user_id,
		)
		added += 1

	if max_db_id > last_db_id:
		_ingest_last_db_id_by_session[session_id] = max_db_id

	try:
		await ingest_manager.flush_session(
			session_id=session_id,
			group_id=group_id,
			user_id=default_user_id,
			reason="manual_command",
		)
	except Exception as exc:
		logger.warning(f"整理记忆：flush_session 失败: {type(exc).__name__}: {exc!s}")
		await ForceIngestLongMemory.finish("已添加消息，但触发整理失败。")

	await ForceIngestLongMemory.finish(f"已添加 {added} 条消息并触发整理（session_id={session_id}）。")


@BenchmarkLongMemoryRecall.handle()
async def handle_benchmark_long_memory_recall(
	event: GroupMessageEvent,
	args: Message = CommandArg(),
) -> None:
	try:
		await _ensure_long_memory_admin(int(event.user_id))
	except PermissionError as exc:
		await BenchmarkLongMemoryRecall.finish(str(exc))

	arg_text = args.extract_plain_text().strip()
	default_group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
	try:
		session_id = _normalize_notepad_session_id(arg_text, default_group_id=default_group_id)
	except ValueError:
		await BenchmarkLongMemoryRecall.finish(
			"该命令仅支持群聊默认会话；请显式提供会话ID，例如：/召回测速 group_123"
		)

	module = import_module(".RecallManager", package=__name__)
	LongMemoryRecallConfigCls = getattr(module, "LongMemoryRecallConfig", None)
	if LongMemoryRecallConfigCls is None:
		await BenchmarkLongMemoryRecall.finish("LongMemoryRecallConfig 不可用。")
	try:
		cfg = get_long_memory_plugin_config()
		recall_cfg = getattr(cfg, "recall", None)
		payload = dict(recall_cfg.model_dump()) if recall_cfg is not None else {}
		config_obj = LongMemoryRecallConfigCls(**payload)
	except Exception:
		config_obj = LongMemoryRecallConfigCls()

	recall_manager = get_long_memory_recall_manager(config=config_obj)
	if recall_manager is None:
		await BenchmarkLongMemoryRecall.finish("LongMemory 未初始化或召回管理器不可用。")

	n = 10
	agg: dict[str, float] = {}
	t0 = time.perf_counter()
	for _ in range(n):
		related = await recall_manager.get_current_related_memories(
			session_id=session_id,
			group_id=default_group_id,
			force_refresh=True,
		)
		timings = getattr(related, "timings", None) or {}
		for k, v in timings.items():
			try:
				agg[k] = float(agg.get(k, 0.0)) + float(v)
			except Exception:
				continue
	t1 = time.perf_counter()

	total = t1 - t0
	avg = total / float(n)
	manager = globals().get("long_memory_manager", None)
	reranker = getattr(manager, "reranker", None) if manager is not None else None
	rerank_enabled = bool(reranker is not None)
	rerank_url = getattr(reranker, "api_url", None) if reranker is not None else None
	lines = [
		"召回测速（force_refresh=True，不添加聊天记录）：",
		f"- session_id={session_id}",
		f"- 次数={n}",
		f"- 总耗时={total:.4f}s",
		f"- 平均耗时={avg*1000:.2f}ms",
		f"- rerank_enabled={rerank_enabled}",
		f"- rerank_url={rerank_url}" if rerank_enabled and isinstance(rerank_url, str) and rerank_url.strip() else "- rerank_url=",
	]

	if agg:
		lines.append("")
		rerank_keys = [k for k in agg.keys() if str(k).startswith("rerank_")]
		if rerank_keys:
			lines.append("rerank 分项平均耗时（ms）:")
			for k in sorted(rerank_keys):
				ms = (agg[k] / float(n)) * 1000.0
				lines.append(f"- {k}: {ms:.2f}")
			lines.append("")
		lines.append("分项平均耗时（ms）:")
		for k in sorted(agg.keys()):
			ms = (agg[k] / float(n)) * 1000.0
			lines.append(f"- {k}: {ms:.2f}")

	await BenchmarkLongMemoryRecall.send("\n".join(lines))

QueryNotepad = on_command(
	"记事本",
	aliases={"查看记事本", "notepad", "notebook"},
	priority=-5,
	block=True,
)

@QueryNotepad.handle()
async def handle_query_notepad(
	event: GroupMessageEvent,
	args: Message = CommandArg(),
) -> None:
	try:
		await _ensure_long_memory_admin(int(event.user_id))
	except PermissionError as exc:
		await QueryNotepad.finish(str(exc))

	logger.debug(f"收到记事本查询命令，参数: {args.extract_plain_text().strip()}")
    
	arg_text = args.extract_plain_text().strip()
	default_group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
	try:
		session_id = _normalize_notepad_session_id(arg_text, default_group_id=default_group_id)
	except ValueError:
		await QueryNotepad.finish("该命令仅支持群聊默认会话；请显式提供会话ID，例如：/记事本 group_123")

	manager: LongMemoryContainer = long_memory_manager
	if manager is None:
		await QueryNotepad.finish(" LongMemory 未初始化。")

	notepad_manager: notepad.SessionNotepadManager = manager.notepad_manager
	if not notepad_manager.has_session(session_id):
		await QueryNotepad.finish(f"会话 {session_id} 暂无记事本记录。")

	notes_text = notepad_manager.get_notes(session_id).strip()
	if not notes_text:
		await QueryNotepad.finish(f"会话 {session_id} 暂无记事本记录。")

	header = f"会话记事本（{session_id}）：\n"
	await QueryNotepad.send(header + notes_text)


from . import memory_editor as _memory_editor  # noqa: F401,E402
