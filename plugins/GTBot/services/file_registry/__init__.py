from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import CleanupPolicy, ManagedFileHandle, ManagedFileRecord, PurgeResult
from .service import ManagedFileService, get_managed_file_service


def register_local_file(
    path: str | Path,
    *,
    kind: str,
    source_type: str,
    session_id: str | None = None,
    group_id: int | None = None,
    user_id: int | None = None,
    mime_type: str | None = None,
    original_name: str | None = None,
    display_name: str | None = None,
    extra: dict[str, Any] | None = None,
    expires_at: float | None = None,
    cleanup_policy: CleanupPolicy = "delete_file_with_mapping",
    cleanup_ref: str | None = None,
) -> str:
    """使用全局文件映射服务注册一个已存在的本地文件。

    Args:
        path: 已存在的本地文件路径。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        session_id: 可选会话标识。
        group_id: 可选群号。
        user_id: 可选用户号。
        mime_type: 可选 MIME 类型。
        original_name: 可选原始文件名。
        display_name: 可选 `gf:` 语义别名。
        extra: 可选扩展元信息。
        expires_at: 映射过期时间戳；`None` 表示永久映射。
        cleanup_policy: 映射删除时的清理策略。
        cleanup_ref: 插件级清理责任标识。

    Returns:
        新生成的稳定 `gfid:`。
    """

    return get_managed_file_service().register_local_file(
        path,
        kind=kind,
        source_type=source_type,
        session_id=session_id,
        group_id=group_id,
        user_id=user_id,
        mime_type=mime_type,
        original_name=original_name,
        display_name=display_name,
        extra=extra,
        expires_at=expires_at,
        cleanup_policy=cleanup_policy,
        cleanup_ref=cleanup_ref,
    )


def register_bytes(
    content: bytes,
    *,
    suffix: str,
    kind: str,
    source_type: str,
    session_id: str | None = None,
    group_id: int | None = None,
    user_id: int | None = None,
    mime_type: str | None = None,
    original_name: str | None = None,
    display_name: str | None = None,
    extra: dict[str, Any] | None = None,
    expires_at: float | None = None,
    cleanup_policy: CleanupPolicy = "delete_file_with_mapping",
    cleanup_ref: str | None = None,
) -> ManagedFileHandle:
    """保留旧接口，但拒绝由 GTFile 直接负责字节落盘。

    Args:
        content: 兼容旧接口保留的原始字节。
        suffix: 兼容旧接口保留的扩展名。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        session_id: 可选会话标识。
        group_id: 可选群号。
        user_id: 可选用户号。
        mime_type: 可选 MIME 类型。
        original_name: 可选原始文件名。
        display_name: 可选 `gf:` 语义别名。
        extra: 可选扩展元信息。
        expires_at: 映射过期时间戳。
        cleanup_policy: 映射删除时的清理策略。
        cleanup_ref: 插件级清理责任标识。

    Returns:
        此接口不会返回可用文件句柄。

    Raises:
        NotImplementedError: 始终抛出，提示调用方先自行落盘再注册映射。
    """

    return get_managed_file_service().register_bytes(
        content,
        suffix=suffix,
        kind=kind,
        source_type=source_type,
        session_id=session_id,
        group_id=group_id,
        user_id=user_id,
        mime_type=mime_type,
        original_name=original_name,
        display_name=display_name,
        extra=extra,
        expires_at=expires_at,
        cleanup_policy=cleanup_policy,
        cleanup_ref=cleanup_ref,
    )


def resolve_file(file_id: str) -> ManagedFileHandle:
    """使用全局文件映射服务解析稳定 `gfid:`。

    Args:
        file_id: 目标机器句柄。

    Returns:
        可直接消费的文件句柄。
    """

    return get_managed_file_service().resolve_file(file_id)


def resolve_file_ref(file_ref: str) -> ManagedFileHandle:
    """使用全局文件映射服务解析 GTFile 引用。

    Args:
        file_ref: `gfid:` 或 `gf:` 形式的文件引用。

    Returns:
        可直接消费的文件句柄。
    """

    return get_managed_file_service().resolve_file_ref(file_ref)


def get_local_path(file_ref: str) -> Path:
    """使用全局文件映射服务获取物理文件路径。

    Args:
        file_ref: `gfid:` 或 `gf:` 形式的文件引用。

    Returns:
        已校验存在的本地绝对路径。
    """

    return get_managed_file_service().get_local_path(file_ref)


def delete_file(file_ref: str, *, remove_physical_file: bool = False) -> None:
    """使用全局文件映射服务删除文件映射。

    Args:
        file_ref: `gfid:` 或 `gf:` 形式的文件引用。
        remove_physical_file: 是否同时删除物理文件。
    """

    get_managed_file_service().delete_file(file_ref, remove_physical_file=remove_physical_file)


def purge_expired_mappings(now_ts: float | None = None) -> PurgeResult:
    """使用全局文件映射服务显式清理已过期映射。

    Args:
        now_ts: 可选的当前时间戳覆盖值。

    Returns:
        清理统计结果。
    """

    return get_managed_file_service().purge_expired_mappings(now_ts=now_ts)


__all__ = [
    "CleanupPolicy",
    "ManagedFileHandle",
    "ManagedFileRecord",
    "ManagedFileService",
    "PurgeResult",
    "delete_file",
    "get_local_path",
    "get_managed_file_service",
    "purge_expired_mappings",
    "register_bytes",
    "register_local_file",
    "resolve_file",
    "resolve_file_ref",
]
