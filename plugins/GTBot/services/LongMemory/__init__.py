
"""LongMemory 服务。

对外导出中短期记忆记事本相关工具，并提供基于配置的
`SessionNotepadManager` 单例获取入口。
"""

from __future__ import annotations

from .notepad import Arecord, Notepad, SessionNotepadManager
from .EventLogManager import QdrantEventLogManager
from .PublicKnowledge import QdrantPublicKnowledge

_session_notepad_manager: SessionNotepadManager | None = None


def _create_session_notepad_manager_from_config() -> SessionNotepadManager:
	"""根据当前配置初始化会话记事本管理器。

	Returns:
		SessionNotepadManager: 初始化后的会话记事本管理器。
	"""
	from ...ConfigManager import total_config

	memory_cfg = (
		total_config.processed_configuration.current_config_group.chat_model.memory
	)
	retention_seconds = float(memory_cfg.notepad_retention_seconds)
	return SessionNotepadManager(
		notepad_max_length=int(memory_cfg.notepad_max_entries),
		session_timeout_seconds=None if retention_seconds <= 0 else retention_seconds,
	)


def get_session_notepad_manager() -> SessionNotepadManager:
	"""获取全局会话记事本管理器单例（懒加载）。

	该函数用于确保项目内所有模块共享同一个 `SessionNotepadManager` 实例，
	避免在不同模块导入时各自创建导致记事本不共享。

	Returns:
		SessionNotepadManager: 全局会话记事本管理器。
	"""
	global _session_notepad_manager
	if _session_notepad_manager is None:
		_session_notepad_manager = _create_session_notepad_manager_from_config()
	return _session_notepad_manager

__all__ = [
	"Arecord",
	"Notepad",
	"SessionNotepadManager",
	"get_session_notepad_manager",
	"QdrantEventLogManager",
	"QdrantPublicKnowledge",
]

