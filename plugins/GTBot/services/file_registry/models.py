from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CleanupPolicy = Literal[
    "delete_file_with_mapping",
    "keep_file",
    "managed_by_plugin",
]


@dataclass(frozen=True, slots=True)
class ManagedFileRecord:
    """描述文件注册表中持久化保存的一条文件记录。

    该结构与 SQLite 中的单行记录一一对应，负责保存稳定 GT 文件引用、可选语义
    别名、物理文件位置、生命周期约束以及业务元信息。`file_id` 永远是主引用，
    `display_name` 只是按需启用的可读别名；调用方不得假设每条记录都一定存在
    `gf:` 形式的名称。

    Args:
        file_id: 稳定机器句柄，格式固定为 `gfid:...`。
        display_name: 可选的语义别名，格式固定为 `gf:层级:文件名`。
        kind: 文件业务类型，例如 `avatar`、`draw_result`、`meme`。
        source_type: 文件来源类型，例如 `avatar_download`、`onebot_image`。
        local_path: 文件在当前机器上的绝对路径。
        original_name: 注册时记录的原始文件名。
        mime_type: 推断或显式提供的 MIME 类型。
        size_bytes: 物理文件大小，单位为字节。
        sha256: 文件内容的 SHA-256 哈希。
        session_id: 与文件相关的会话标识。
        group_id: 与文件相关的群号。
        user_id: 与文件相关的用户号。
        created_at: 注册记录创建时间戳。
        expires_at: 映射过期时间戳；为 `None` 表示永久映射。
        cleanup_policy: 映射删除时的清理策略。
        cleanup_ref: 当清理责任交由插件处理时的责任标识。
        extra: 调用方附带的扩展元信息。
    """

    file_id: str
    display_name: str | None
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
    expires_at: float | None
    cleanup_policy: CleanupPolicy
    cleanup_ref: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManagedFileHandle:
    """描述已解析、可直接供上层工具消费的文件句柄。

    该结构是 `ManagedFileRecord` 的上层视图，强调“可直接使用”的语义。解析成功
    后，调用方可以直接读取 `local_path`、`display_name`、`extra` 等字段，而无需
    继续关心底层数据库结构与主键索引策略。

    Args:
        file_id: 稳定机器句柄。
        display_name: 可选语义别名。
        local_path: 已校验存在的本地绝对路径。
        mime_type: 文件 MIME 类型。
        size_bytes: 文件大小，单位为字节。
        sha256: 文件内容哈希。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        original_name: 原始文件名。
        created_at: 注册记录创建时间戳。
        expires_at: 映射过期时间戳。
        cleanup_policy: 当前记录的清理策略。
        cleanup_ref: 插件级清理责任标识。
        extra: 扩展元信息。
    """

    file_id: str
    display_name: str | None
    local_path: Path
    mime_type: str | None
    size_bytes: int
    sha256: str
    kind: str
    source_type: str
    original_name: str | None
    created_at: float
    expires_at: float | None
    cleanup_policy: CleanupPolicy
    cleanup_ref: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """描述一次过期映射清理任务的统计结果。

    第一版仅提供显式调用的清理入口，因此返回值需要同时覆盖“删除了多少映射”
    与“物理文件删除是否成功”两类信息，便于调用方在调试或后台任务中做日志记录。

    Args:
        scanned_count: 本次扫描到的过期记录数。
        deleted_mapping_count: 成功删除的映射记录数。
        deleted_file_count: 成功删除的物理文件数。
        failed_refs: 清理失败的引用及其失败原因。
    """

    scanned_count: int
    deleted_mapping_count: int
    deleted_file_count: int
    failed_refs: dict[str, str] = field(default_factory=dict)
