from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ManagedFileHandle, ManagedFileRecord
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
    extra: dict[str, Any] | None = None,
    expires_at: float | None = None,
) -> str:
    """使用全局文件服务注册一个本地文件。

    Args:
        path: 待注册的本地文件路径。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        session_id: 可选会话标识。
        group_id: 可选群号。
        user_id: 可选用户号。
        mime_type: 可选 MIME 类型。
        original_name: 可选原始文件名。
        extra: 可选扩展元信息。
        expires_at: 可选过期时间戳。

    Returns:
        新生成的稳定 `file_id`。
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
        extra=extra,
        expires_at=expires_at,
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
    extra: dict[str, Any] | None = None,
    expires_at: float | None = None,
) -> ManagedFileHandle:
    """使用全局文件服务落盘并注册原始字节。

    Args:
        content: 待落盘的字节内容。
        suffix: 文件扩展名。
        kind: 文件业务类型。
        source_type: 文件来源类型。
        session_id: 可选会话标识。
        group_id: 可选群号。
        user_id: 可选用户号。
        mime_type: 可选 MIME 类型。
        original_name: 可选原始文件名。
        extra: 可选扩展元信息。
        expires_at: 可选过期时间戳。

    Returns:
        已落盘并注册完成的文件句柄。
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
        extra=extra,
        expires_at=expires_at,
    )


def resolve_file(file_id: str) -> ManagedFileHandle:
    """使用全局文件服务解析稳定 `file_id`。

    Args:
        file_id: 目标文件句柄。

    Returns:
        可直接消费的文件句柄。
    """

    return get_managed_file_service().resolve_file(file_id)


def get_local_path(file_id: str) -> Path:
    """使用全局文件服务获取物理文件路径。

    Args:
        file_id: 目标文件句柄。

    Returns:
        已校验存在的本地绝对路径。
    """

    return get_managed_file_service().get_local_path(file_id)


def delete_file(file_id: str, *, remove_physical_file: bool = False) -> None:
    """使用全局文件服务删除文件映射。

    Args:
        file_id: 目标文件句柄。
        remove_physical_file: 是否同时删除物理文件。
    """

    get_managed_file_service().delete_file(file_id, remove_physical_file=remove_physical_file)


__all__ = [
    "ManagedFileHandle",
    "ManagedFileRecord",
    "ManagedFileService",
    "delete_file",
    "get_local_path",
    "get_managed_file_service",
    "register_bytes",
    "register_local_file",
    "resolve_file",
]
