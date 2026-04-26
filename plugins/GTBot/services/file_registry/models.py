from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ManagedFileRecord:
    """描述文件注册表中持久化保存的一条文件记录。

    该结构对应 SQLite 中的单行记录，负责承载文件的稳定标识、物理落盘路径、
    来源元信息以及与会话相关的辅助上下文。`local_path` 在进入该结构前应已经被
    规范化为绝对路径；调用方不应依赖数据库中的原始字符串格式。

    Args:
        file_id: 供 Agent 与工具之间传递的稳定文件句柄。
        kind: 文件业务类型，例如 `avatar`、`draw_result`、`meme`。
        source_type: 文件来源类型，例如 `avatar_download`、`openai_draw`。
        local_path: 文件在当前机器上的绝对路径。
        original_name: 注册时的原始文件名；若未知则为空。
        mime_type: 推断得到的 MIME 类型；若未知则为空。
        size_bytes: 物理文件大小，单位为字节。
        sha256: 文件内容的 SHA-256 哈希，用于校验与排障。
        session_id: 与文件相关的会话标识；若无则为空。
        group_id: 与文件相关的群号；若无则为 `None`。
        user_id: 与文件相关的用户号；若无则为 `None`。
        created_at: 注册记录创建时间戳。
        expires_at: 预留的过期时间戳；第一版可为空。
        extra: 调用方附带的扩展元信息。
    """

    file_id: str
    kind: str
    source_type: str
    local_path: Path
    original_name: str | None
    mime_type: str | None
    size_bytes: int
    sha256: str
    session_id: str | None
    group_id: int | None
    user_id: int | None
    created_at: float
    expires_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManagedFileHandle:
    """描述已解析、可直接供上层工具消费的文件句柄。

    与 `ManagedFileRecord` 相比，该结构强调“可直接使用”的视角：一旦成功解析，
    调用方可以立即读取 `local_path`、`mime_type` 和 `extra` 等字段，而无需继续
    关心底层存储表结构。第一版中该结构与持久化记录字段几乎等价，以便保持
    实现简单、减少转换损耗。

    Args:
        file_id: 稳定文件句柄。
        local_path: 已校验存在的本地绝对路径。
        mime_type: 文件 MIME 类型；若未知则为空。
        size_bytes: 文件大小，单位为字节。
        sha256: 文件内容哈希。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        original_name: 原始文件名。
        created_at: 注册记录创建时间戳。
        extra: 扩展元信息。
    """

    file_id: str
    local_path: Path
    mime_type: str | None
    size_bytes: int
    sha256: str
    kind: str
    source_type: str
    original_name: str | None
    created_at: float
    extra: dict[str, Any] = field(default_factory=dict)

