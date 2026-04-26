from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ManagedFileHandle, ManagedFileRecord
from .store import ManagedFileStore


class ManagedFileService:
    """为 GTBot 提供统一文件注册、解析与删除能力。

    该服务面向 Agent tool 与产物型插件，负责把“本地文件路径”转换为稳定 `file_id`，
    并在消费阶段把 `file_id` 恢复为可直接读取的标准化句柄。第一版仅处理单实例本地
    文件，不涉及跨机器同步、权限隔离或自动过期清理。
    """

    def __init__(self, store: ManagedFileStore | None = None) -> None:
        """初始化文件服务。

        Args:
            store: 自定义底层存储实现。未传入时使用默认 SQLite 存储。
        """

        self._store = store or ManagedFileStore()

    def register_local_file(
        self,
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
        """将已有本地文件注册为稳定 `file_id`。

        该方法不会移动或复制文件，只会校验目标路径、计算摘要并写入映射记录。
        若调用方随后替换或删除原文件，解析该 `file_id` 时会在消费阶段显式报错，
        避免数据库记录与物理文件状态 silently 偏离。

        Args:
            path: 待注册的本地文件路径。
            kind: 文件业务类型。
            source_type: 文件来源类型。
            session_id: 可选会话标识。
            group_id: 可选群号。
            user_id: 可选用户号。
            mime_type: 显式提供的 MIME 类型；为空时自动推断。
            original_name: 记录用原始文件名；为空时使用目标文件名。
            extra: 扩展元信息。
            expires_at: 预留过期时间戳。

        Returns:
            新生成的稳定 `file_id`。

        Raises:
            FileNotFoundError: 当目标文件不存在时抛出。
            ValueError: 当 `kind`、`source_type` 非法或目标路径不是文件时抛出。
        """

        local_path = Path(path).resolve()
        if not str(kind or "").strip():
            raise ValueError("kind 不能为空")
        if not str(source_type or "").strip():
            raise ValueError("source_type 不能为空")
        if not local_path.exists():
            raise FileNotFoundError(f"文件不存在: {local_path}")
        if not local_path.is_file():
            raise ValueError(f"目标路径不是文件: {local_path}")

        content = local_path.read_bytes()
        stat = local_path.stat()
        file_id = self._new_file_id()
        guessed_mime = mime_type or mimetypes.guess_type(local_path.name)[0]
        record = ManagedFileRecord(
            file_id=file_id,
            kind=str(kind).strip(),
            source_type=str(source_type).strip(),
            local_path=local_path,
            original_name=str(original_name).strip() if str(original_name or "").strip() else local_path.name,
            mime_type=guessed_mime,
            size_bytes=int(stat.st_size),
            sha256=hashlib.sha256(content).hexdigest(),
            session_id=str(session_id).strip() if str(session_id or "").strip() else None,
            group_id=int(group_id) if group_id is not None else None,
            user_id=int(user_id) if user_id is not None else None,
            created_at=float(time.time()),
            expires_at=float(expires_at) if expires_at is not None else None,
            extra=dict(extra or {}),
        )
        self._store.insert_record(record)
        return file_id

    def register_bytes(
        self,
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
        """将原始字节落盘并注册为受管文件。

        第一版统一把该类文件保存到 GTBot 数据目录下的 `managed_files` 目录，并使用
        `file_id` 作为主文件名。这样可保证调用方无需自行管理临时目录，也不会因原始
        路径来源不稳定而影响后续解析。

        Args:
            content: 待落盘的原始文件内容。
            suffix: 目标文件扩展名，允许传入不带点的值。
            kind: 文件业务类型。
            source_type: 文件来源类型。
            session_id: 可选会话标识。
            group_id: 可选群号。
            user_id: 可选用户号。
            mime_type: 显式指定的 MIME 类型。
            original_name: 记录用原始文件名。
            extra: 扩展元信息。
            expires_at: 预留过期时间戳。

        Returns:
            已落盘并注册完成的标准文件句柄。

        Raises:
            ValueError: 当字节内容为空、扩展名为空或类型信息非法时抛出。
        """

        if not content:
            raise ValueError("content 不能为空")
        if not str(suffix or "").strip():
            raise ValueError("suffix 不能为空")
        file_id = self._new_file_id()
        normalized_suffix = str(suffix).strip()
        if not normalized_suffix.startswith("."):
            normalized_suffix = f".{normalized_suffix}"
        target_dir = self._store.db_path.parent / "managed_files"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_file_name = file_id.replace(":", "_")
        target_path = target_dir / f"{safe_file_name}{normalized_suffix}"
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        tmp_path.write_bytes(content)
        tmp_path.replace(target_path)
        registered_id = self.register_local_file(
            target_path,
            kind=kind,
            source_type=source_type,
            session_id=session_id,
            group_id=group_id,
            user_id=user_id,
            mime_type=mime_type,
            original_name=original_name or target_path.name,
            extra=extra,
            expires_at=expires_at,
        )
        return self.resolve_file(registered_id)

    def resolve_file(self, file_id: str) -> ManagedFileHandle:
        """解析稳定 `file_id` 并返回可直接消费的文件句柄。

        Args:
            file_id: 目标文件句柄。

        Returns:
            已校验物理文件存在性的标准文件句柄。

        Raises:
            ValueError: 当 `file_id` 格式非法时抛出。
            FileNotFoundError: 当记录不存在或物理文件已经丢失时抛出。
        """

        normalized = str(file_id or "").strip()
        if not normalized.startswith("gtfile:"):
            raise ValueError("file_id 必须以 gtfile: 开头")
        record = self._store.get_record(normalized)
        if record is None:
            raise FileNotFoundError(f"未找到文件映射: {normalized}")
        local_path = record.local_path.resolve()
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"文件映射存在但物理文件缺失: {local_path}")
        return ManagedFileHandle(
            file_id=record.file_id,
            local_path=local_path,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            kind=record.kind,
            source_type=record.source_type,
            original_name=record.original_name,
            created_at=record.created_at,
            extra=dict(record.extra),
        )

    def get_local_path(self, file_id: str) -> Path:
        """根据 `file_id` 获取文件本地绝对路径。

        Args:
            file_id: 目标文件句柄。

        Returns:
            已校验存在的本地绝对路径对象。
        """

        return self.resolve_file(file_id).local_path

    def delete_file(self, file_id: str, *, remove_physical_file: bool = False) -> None:
        """删除文件映射，必要时同步删除物理文件。

        Args:
            file_id: 目标文件句柄。
            remove_physical_file: 是否同时删除映射指向的物理文件。

        Raises:
            ValueError: 当 `file_id` 格式非法时抛出。
        """

        normalized = str(file_id or "").strip()
        if not normalized.startswith("gtfile:"):
            raise ValueError("file_id 必须以 gtfile: 开头")
        removed = self._store.delete_record(normalized)
        if removed is None:
            return
        if remove_physical_file:
            removed.local_path.unlink(missing_ok=True)

    def _new_file_id(self) -> str:
        """生成新的稳定文件句柄。

        Returns:
            以 `gtfile:` 为前缀的唯一文件标识。
        """

        return f"gtfile:{uuid4().hex}"


_managed_file_service: ManagedFileService | None = None


def get_managed_file_service() -> ManagedFileService:
    """返回全局共享的文件注册服务实例。

    Returns:
        单例模式的文件注册服务。
    """

    global _managed_file_service
    if _managed_file_service is None:
        _managed_file_service = ManagedFileService()
    return _managed_file_service
