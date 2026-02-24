
"""LongMemory 服务。

对外导出中短期记忆记事本相关工具，并提供基于配置的
`SessionNotepadManager` 单例获取入口。
"""

from __future__ import annotations
import re
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING, cast
from pathlib import Path
import os
import asyncio
import time

from importlib import import_module

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool as lc_tool
from langchain_core.messages import AIMessage, HumanMessage

from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context

from .tool import (
	_impl_search_event_log_info,
	_impl_search_group_profile_info,
	_impl_search_public_knowledge,
	_impl_search_user_profile_info,
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


# ============================================================================
# 长期记忆入库管理器（消息缓冲 -> 触发 -> LLM 工具整理）
# ============================================================================


_ingest_manager = None


_recall_manager = None


_recall_last_message_id_by_session: dict[str, int] = {}

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
	lines.append("## [系统提示] 以下是可能相关的记忆（仅供参考，可能不准确）：")

	query = getattr(related, "query", "")
	if isinstance(query, str) and query.strip():
		lines.append(f"- query: {query.strip()}")

	def _take(items: Any) -> list[Any]:
		return items if isinstance(items, list) else []

	events = _take(getattr(related, "event_logs", None))
	if events:
		lines.append("")
		lines.append("### 事件日志")
		for it in events:
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			details = getattr(it, "details", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- [{sid}] (similarity={float(sim):.3f}) {details}")
			else:
				lines.append(f"- [{sid}] {details}")

	knowledge = _take(getattr(related, "public_knowledge", None))
	if knowledge:
		lines.append("")
		lines.append("### 公共知识")
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

	user_profiles = _take(getattr(related, "user_profiles", None))
	if user_profiles:
		lines.append("")
		lines.append("### 用户画像")
		for it in user_profiles:
			uid = getattr(it, "user_id", "")
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			text = getattr(it, "text", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- user_id={uid} [{sid}] (similarity={float(sim):.3f}) {text}")
			else:
				lines.append(f"- user_id={uid} [{sid}] {text}")

	group_hits = _take(getattr(related, "group_profile_hits", None))
	if group_hits:
		lines.append("")
		lines.append("### 群画像（检索命中）")
		for it in group_hits:
			gid = getattr(it, "group_id", "")
			sid = getattr(it, "short_id", "")
			sim = getattr(it, "similarity", None)
			text = getattr(it, "text", "")
			if isinstance(sim, (int, float)):
				lines.append(f"- group_id={gid} [{sid}] (similarity={float(sim):.3f}) {text}")
			else:
				lines.append(f"- group_id={gid} [{sid}] {text}")

	if not query.strip() and not events and not knowledge and not user_profiles and not group_hits:
		return ""

	return "\n".join([str(x) for x in lines if str(x).strip()])


async def prepare_long_memory_recall(*, plugin_ctx: Any, runtime_context: Any) -> None:
	try:
		if plugin_ctx is None:
			return
		if plugin_ctx.extra.get("_long_memory_recall_prepared"):
			return

		session_id = getattr(runtime_context, "session_id", None)
		if not (isinstance(session_id, str) and session_id.strip()):
			group_id = getattr(runtime_context, "group_id", None)
			user_id = getattr(runtime_context, "user_id", None)
			if isinstance(group_id, int) and group_id > 0:
				session_id = f"group_{group_id}"
			elif isinstance(user_id, int) and user_id > 0:
				session_id = f"private_{user_id}"
			else:
				return
		session_id = str(session_id).strip()
		if not session_id:
			return

		raw_messages = getattr(runtime_context, "raw_messages", [])
		raw_messages_list = raw_messages if isinstance(raw_messages, list) else []
		if not raw_messages_list:
			return

		module = import_module(".RecallManager", package=__name__)
		LongMemoryRecallConfigCls = getattr(module, "LongMemoryRecallConfig", None)
		if LongMemoryRecallConfigCls is None:
			return
		config_obj = LongMemoryRecallConfigCls()

		recall_manager = get_long_memory_recall_manager(config=config_obj)
		if recall_manager is None:
			return

		last_seen = int(_recall_last_message_id_by_session.get(session_id, 0))
		new_messages = [m for m in raw_messages_list if int(getattr(m, "message_id", 0) or 0) > last_seen]
		if not new_messages:
			new_messages = raw_messages_list[-1:]

		max_mid = last_seen
		for m in new_messages:
			mid = int(getattr(m, "message_id", 0) or 0)
			if mid > max_mid:
				max_mid = mid
		_recall_last_message_id_by_session[session_id] = max_mid

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
			force_refresh=True,
		)

		plugin_ctx.extra["long_memory_recall_config"] = config_obj
		plugin_ctx.extra["long_memory_related_memories"] = related
		plugin_ctx.extra["_long_memory_recall_prepared"] = True
		return
	except Exception:
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
		qdrant_server_url="http://localhost:6333/",
		embed_service_url="http://localhost:30020/v1/embeddings",
		embed_model="qwen3-embedding-0.6b",
		qdrant_api_key="",
		embed_api_key="",
		embed_service_type="openai",
		notepad_max_length=20,
		max_sessions=1000,
		session_timeout_seconds=3600,
		qdrant_collection_name="long_memory",
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
		return f"group_{default_group_id}"
	if raw.isdigit():
		return f"group_{int(raw)}"
	return raw


def _infer_session_id_from_runtime(runtime: Any) -> str | None:
	context = getattr(runtime, "context", None)
	if context is None:
		return None
	session_id = getattr(context, "session_id", None)
	if isinstance(session_id, str) and session_id.strip():
		return session_id.strip()
	group_id = getattr(context, "group_id", None)
	if isinstance(group_id, int) and group_id > 0:
		return f"group_{group_id}"
	user_id = getattr(context, "user_id", None)
	if isinstance(user_id, int) and user_id > 0:
		return f"private_{user_id}"
	return None


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


class LongMemoryNotepadMiddleware(AgentMiddleware[AgentState, Any]):
	def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
		try:
			plugin_ctx = get_current_plugin_context()
			if plugin_ctx is not None and plugin_ctx.extra.get("_long_memory_notepad_injected"):
				return None

			session_id = _infer_session_id_from_runtime(runtime)
			if not session_id:
				return None

			manager = globals().get("long_memory_manager", None)
			if manager is None:
				return None

			notepad_manager = manager.notepad_manager
			if not notepad_manager.has_session(session_id):
				return None
			notes_text = notepad_manager.get_notes(session_id).strip()
			if not notes_text:
				return None

			notepad_context = (
				"[系统提示] 以下是当前会话的记事本记录（用于补充短中期记忆，可能与当前问题无关）：\n\n"
				+ notes_text
			)

			messages = list(state.get("messages", []) or [])
			if messages:
				messages.insert(1, HumanMessage(notepad_context))
			else:
				messages = [HumanMessage(notepad_context)]
			state["messages"] = cast(Any, messages)

			if plugin_ctx is not None:
				plugin_ctx.extra["_long_memory_notepad_injected"] = True

			return None
		except Exception:
			return None


class LongMemoryRecallMiddleware(AgentMiddleware[AgentState, Any]):
	def wrap_model_call(self, request: Any, handler: Any) -> Any:
		try:
			plugin_ctx = get_current_plugin_context()
			if plugin_ctx is None:
				return handler(request)
			if plugin_ctx.extra.get("_long_memory_recall_injected"):
				return handler(request)

			related = plugin_ctx.extra.get("long_memory_related_memories")
			prefix = _format_related_long_memories(related)
			if not prefix:
				return handler(request)

			messages = list(getattr(request, "messages", []) or [])
			idx = -1
			for i in range(len(messages) - 1, -1, -1):
				if isinstance(messages[i], HumanMessage):
					idx = i
					break
			if idx >= 0:
				last = messages[idx]
				content = getattr(last, "content", "")
				if isinstance(content, str) and content.strip():
					messages[idx] = HumanMessage(prefix + "\n\n" + content)
					plugin_ctx.extra["_long_memory_recall_injected"] = True

			override = getattr(request, "override", None)
			if callable(override):
				request = override(messages=messages)
			else:
				setattr(request, "messages", messages)
			return handler(request)
		except Exception:
			return handler(request)

	async def awrap_model_call(self, request: Any, handler: Any) -> Any:
		try:
			plugin_ctx = get_current_plugin_context()
			if plugin_ctx is None:
				result = handler(request)
				if isinstance(result, Awaitable):
					return await result
				return result

			runtime_context = getattr(plugin_ctx, "runtime_context", None)
			await prepare_long_memory_recall(plugin_ctx=plugin_ctx, runtime_context=runtime_context)

			if plugin_ctx.extra.get("_long_memory_recall_injected"):
				result = handler(request)
				if isinstance(result, Awaitable):
					return await result
				return result

			related = plugin_ctx.extra.get("long_memory_related_memories")
			prefix = _format_related_long_memories(related)
			if prefix:
				messages = list(getattr(request, "messages", []) or [])
				idx = -1
				for i in range(len(messages) - 1, -1, -1):
					if isinstance(messages[i], HumanMessage):
						idx = i
						break
				if idx >= 0:
					last = messages[idx]
					content = getattr(last, "content", "")
					if isinstance(content, str) and content.strip():
						messages[idx] = HumanMessage(prefix + "\n\n" + content)
						plugin_ctx.extra["_long_memory_recall_injected"] = True

				override = getattr(request, "override", None)
				if callable(override):
					request = override(messages=messages)
				else:
					setattr(request, "messages", messages)

			result = handler(request)
			if isinstance(result, Awaitable):
				return await result
			return result
		except Exception:
			result = handler(request)
			if isinstance(result, Awaitable):
				return await result
			return result


def register(registry) -> None:  # noqa: ANN001
	registry.add_tool(take_notes)
	registry.add_agent_middleware(LongMemoryRecallMiddleware(), priority=-10)
	registry.add_agent_middleware(LongMemoryNotepadMiddleware())



from nonebot import on_command, logger
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.params import CommandArg


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


@BenchmarkLongMemoryRecall.handle()
async def handle_benchmark_long_memory_recall(
	event: GroupMessageEvent,
	args: Message = CommandArg(),
) -> None:
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
	lines = [
		"召回测速（force_refresh=True，不添加聊天记录）：",
		f"- session_id={session_id}",
		f"- 次数={n}",
		f"- 总耗时={total:.4f}s",
		f"- 平均耗时={avg*1000:.2f}ms",
	]

	if agg:
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

