from __future__ import annotations

import hashlib
import mimetypes
import secrets
import string
import time
from pathlib import Path
from typing import Any

from .models import CleanupPolicy, ManagedFileHandle, ManagedFileRecord, PurgeResult
from .store import ManagedFileStore

_GFID_PREFIX = "gfid:"
_GF_PREFIX = "gf:"
_GFID_ALPHABET = string.ascii_letters + string.digits
_GFID_LENGTH = 12
_VALID_CLEANUP_POLICIES: set[str] = {"delete_file_with_mapping", "keep_file", "managed_by_plugin"}


class ManagedFileService:
    """提供 GTFile 映射的注册、解析与显式清理能力。

    这个服务只负责“把已有文件位置映射为稳定 GTFile 引用”，不负责决定物理
    文件应该落到哪里，也不负责替插件托管二进制内容。插件如果需要处理下载、
    缓存或临时文件，必须先在自己的目录中完成落盘，再把最终路径注册到这里。
    """

    def __init__(self, store: ManagedFileStore | None = None) -> None:
        """初始化文件映射服务。

        Args:
            store: 可选的底层存储实现。未提供时使用默认 SQLite 存储。
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
        display_name: str | None = None,
        extra: dict[str, Any] | None = None,
        expires_at: float | None = None,
        cleanup_policy: CleanupPolicy = "delete_file_with_mapping",
        cleanup_ref: str | None = None,
    ) -> str:
        """把一个已存在的本地文件注册为 GTFile 映射。

        该方法不会移动、复制或重命名目标文件，只会校验路径、计算摘要并写入
        映射记录。所有生命周期约束也在这里统一校验，避免不同插件出现互相不兼容
        的“永久文件”或“过期清理”规则。

        Args:
            path: 已存在的本地文件路径。
            kind: 文件业务类型。
            source_type: 文件来源类型。
            session_id: 可选会话标识。
            group_id: 可选群号。
            user_id: 可选用户号。
            mime_type: 可选 MIME 类型；未提供时按文件名推断。
            original_name: 记录用原始文件名；为空时使用目标文件名。
            display_name: 可选 `gf:` 语义别名。
            extra: 可选扩展元信息。
            expires_at: 映射过期时间戳；`None` 表示永久映射。
            cleanup_policy: 映射删除时的清理策略。
            cleanup_ref: 插件级清理责任标识。

        Returns:
            新生成的稳定 `gfid:`。

        Raises:
            FileNotFoundError: 当目标文件不存在时抛出。
            ValueError: 当参数、别名或生命周期配置非法时抛出。
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

        normalized_display_name = self._normalize_display_name(display_name)
        normalized_cleanup_policy = self._normalize_cleanup_policy(cleanup_policy)
        normalized_cleanup_ref = self._normalize_cleanup_ref(cleanup_ref)
        normalized_expires_at = self._normalize_lifecycle(
            expires_at=expires_at,
            cleanup_policy=normalized_cleanup_policy,
            cleanup_ref=normalized_cleanup_ref,
        )

        content = local_path.read_bytes()
        stat = local_path.stat()
        file_id = self._new_file_id()
        guessed_mime = mime_type or mimetypes.guess_type(local_path.name)[0]
        record = ManagedFileRecord(
            file_id=file_id,
            display_name=normalized_display_name,
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
            expires_at=normalized_expires_at,
            cleanup_policy=normalized_cleanup_policy,
            cleanup_ref=normalized_cleanup_ref,
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
        display_name: str | None = None,
        extra: dict[str, Any] | None = None,
        expires_at: float | None = None,
        cleanup_policy: CleanupPolicy = "delete_file_with_mapping",
        cleanup_ref: str | None = None,
    ) -> ManagedFileHandle:
        """拒绝由 GTFile 直接负责字节落盘。

        这个兼容入口被保留，仅用于给旧调用方一个明确错误。GTFile 只处理映射，
        不处理物理文件存放位置；调用方必须先自行落盘，再调用
        `register_local_file`。

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
            此接口不会返回可用句柄。

        Raises:
            NotImplementedError: 始终抛出，提示调用方先自行落盘再注册映射。
        """

        _ = (
            content,
            suffix,
            kind,
            source_type,
            session_id,
            group_id,
            user_id,
            mime_type,
            original_name,
            display_name,
            extra,
            expires_at,
            cleanup_policy,
            cleanup_ref,
        )
        raise NotImplementedError("GTFile 只负责文件映射；请先由插件自行落盘，再调用 register_local_file")

    def resolve_file(self, file_id: str) -> ManagedFileHandle:
        """解析稳定 `gfid:` 并返回可直接消费的文件句柄。

        Args:
            file_id: 稳定机器句柄。

        Returns:
            已校验物理文件存在性的标准文件句柄。

        Raises:
            ValueError: 当 `file_id` 格式非法时抛出。
            FileNotFoundError: 当映射不存在或物理文件已丢失时抛出。
        """

        normalized = str(file_id or "").strip()
        if not normalized.startswith(_GFID_PREFIX):
            raise ValueError("file_id 必须以 gfid: 开头")
        record = self._resolve_record_by_file_id(normalized)
        if record is None:
            raise FileNotFoundError(f"未找到文件映射: {normalized}")
        return self._record_to_handle(record)

    def resolve_file_ref(self, ref: str) -> ManagedFileHandle:
        """解析 GTFile 引用并返回可直接消费的文件句柄。

        Args:
            ref: `gfid:` 或 `gf:` 形式的文件引用。

        Returns:
            已校验物理文件存在性的标准文件句柄。

        Raises:
            ValueError: 当引用格式非法时抛出。
            FileNotFoundError: 当映射不存在或物理文件已丢失时抛出。
        """

        normalized = str(ref or "").strip()
        if normalized.startswith(_GFID_PREFIX):
            return self.resolve_file(normalized)
        if not normalized.startswith(_GF_PREFIX):
            raise ValueError("file_ref 必须以 gfid: 或 gf: 开头")
        record = self._store.get_record_by_display_name(normalized)
        if record is None:
            raise FileNotFoundError(f"未找到文件映射: {normalized}")
        return self._record_to_handle(record)

    def get_local_path(self, file_ref: str) -> Path:
        """根据 GTFile 引用获取本地绝对路径。

        Args:
            file_ref: `gfid:` 或 `gf:` 形式的文件引用。

        Returns:
            已校验存在的本地绝对路径对象。
        """

        return self.resolve_file_ref(file_ref).local_path

    def _resolve_record_by_file_id(self, file_id: str) -> ManagedFileRecord | None:
        """按 `gfid:` 解析文件记录，并兼容误带扩展名的历史引用。

        某些旧链路会把 GTFile 机器句柄当作“文件名”继续拼接图片扩展名，例如
        `gfid:abc123.jpg`。标准 GTFile 主键本身不包含扩展名，因此这里会先按原值
        查询，再在未命中时尝试去掉尾部扩展名重新查询，以兼容已经写入上下文或
        缓存中的历史坏引用。

        Args:
            file_id: 待解析的 `gfid:` 机器句柄。

        Returns:
            命中时返回对应文件记录；否则返回 `None`。
        """

        record = self._store.get_record(file_id)
        if record is not None:
            return record

        normalized_candidate = self._strip_accidental_file_suffix(file_id)
        if normalized_candidate == file_id:
            return None
        return self._store.get_record(normalized_candidate)

    def _strip_accidental_file_suffix(self, file_id: str) -> str:
        """移除误拼接到 `gfid:` 末尾的文件扩展名。

        标准 `gfid:` 只包含前缀和随机 token，不应自带 `.jpg`、`.png` 等扩展名。
        这里仅在后缀看起来像常见文件扩展名时裁掉最后一段，避免把正常包含点号的
        其他业务标识误处理。

        Args:
            file_id: 原始 `gfid:` 字符串。

        Returns:
            去掉疑似扩展名后的 `gfid:`；如果原值不需要处理则原样返回。
        """

        suffix = Path(file_id).suffix.lower()
        if not suffix:
            return file_id
        if not suffix.startswith("."):
            return file_id
        if not suffix[1:].isalnum():
            return file_id
        stem = Path(file_id).stem
        return stem if stem.startswith(_GFID_PREFIX) else file_id

    def delete_file(self, file_ref: str, *, remove_physical_file: bool = False) -> None:
        """删除文件映射，并按需删除对应物理文件。

        Args:
            file_ref: `gfid:` 或 `gf:` 形式的文件引用。
            remove_physical_file: 是否同时删除映射指向的物理文件。
        """

        handle = self.resolve_file_ref(file_ref)
        removed = self._store.delete_record(handle.file_id)
        if removed is None:
            return
        if remove_physical_file:
            removed.local_path.unlink(missing_ok=True)

    def purge_expired_mappings(self, now_ts: float | None = None) -> PurgeResult:
        """显式清理所有已过期的映射记录。

        Args:
            now_ts: 可选的当前时间戳覆盖值；为空时使用系统当前时间。

        Returns:
            本次清理任务的统计结果。
        """

        effective_now = float(time.time()) if now_ts is None else float(now_ts)
        records = self._store.list_expired_records(effective_now)
        deleted_mapping_count = 0
        deleted_file_count = 0
        failed_refs: dict[str, str] = {}

        for record in records:
            try:
                removed = self._store.delete_record(record.file_id)
                if removed is None:
                    continue
                deleted_mapping_count += 1
                if removed.cleanup_policy == "delete_file_with_mapping":
                    try:
                        removed.local_path.unlink(missing_ok=True)
                        deleted_file_count += 1
                    except Exception as exc:  # noqa: BLE001
                        failed_refs[removed.file_id] = f"删除物理文件失败: {type(exc).__name__}: {exc!s}"
                elif removed.cleanup_policy == "managed_by_plugin":
                    failed_refs[removed.file_id] = "记录已过期，但物理文件需由插件自行清理"
            except Exception as exc:  # noqa: BLE001
                failed_refs[record.file_id] = f"删除映射失败: {type(exc).__name__}: {exc!s}"

        return PurgeResult(
            scanned_count=len(records),
            deleted_mapping_count=deleted_mapping_count,
            deleted_file_count=deleted_file_count,
            failed_refs=failed_refs,
        )

    def _record_to_handle(self, record: ManagedFileRecord) -> ManagedFileHandle:
        """把持久化记录转换为可消费的文件句柄。

        Args:
            record: 已从存储层读取出的文件记录。

        Returns:
            已校验物理文件存在性的标准文件句柄。

        Raises:
            FileNotFoundError: 当映射指向的物理文件不存在时抛出。
        """

        local_path = record.local_path.resolve()
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"文件映射存在但物理文件缺失: {local_path}")
        return ManagedFileHandle(
            file_id=record.file_id,
            display_name=record.display_name,
            local_path=local_path,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            kind=record.kind,
            source_type=record.source_type,
            original_name=record.original_name,
            created_at=record.created_at,
            expires_at=record.expires_at,
            cleanup_policy=record.cleanup_policy,
            cleanup_ref=record.cleanup_ref,
            extra=dict(record.extra),
        )

    def _normalize_display_name(self, display_name: str | None) -> str | None:
        """规范化并校验可选 `gf:` 语义别名。

        Args:
            display_name: 调用方传入的可读别名。

        Returns:
            规范化后的别名；未提供时返回 `None`。

        Raises:
            ValueError: 当别名格式不符合 `gf:层级:文件名` 约束时抛出。
        """

        normalized = str(display_name or "").strip()
        if not normalized:
            return None
        if not normalized.startswith(_GF_PREFIX):
            raise ValueError("display_name 必须以 gf: 开头")
        parts = normalized.split(":")
        if len(parts) < 3 or any(not str(part).strip() for part in parts[1:]):
            raise ValueError("display_name 必须为 gf:层级1:层级n:文件名 格式")
        return ":".join(parts)

    def _normalize_cleanup_policy(self, cleanup_policy: CleanupPolicy | str) -> CleanupPolicy:
        """规范化清理策略字符串。

        Args:
            cleanup_policy: 调用方传入的清理策略。

        Returns:
            受支持的清理策略字面量。

        Raises:
            ValueError: 当清理策略不在支持集合内时抛出。
        """

        normalized = str(cleanup_policy or "").strip()
        if normalized not in _VALID_CLEANUP_POLICIES:
            raise ValueError("cleanup_policy 不合法")
        return normalized  # type: ignore[return-value]

    def _normalize_cleanup_ref(self, cleanup_ref: str | None) -> str | None:
        """规范化插件级清理责任标识。

        Args:
            cleanup_ref: 调用方传入的清理责任标识。

        Returns:
            去掉首尾空白后的标识；为空时返回 `None`。
        """

        normalized = str(cleanup_ref or "").strip()
        return normalized or None

    def _normalize_lifecycle(
        self,
        *,
        expires_at: float | None,
        cleanup_policy: CleanupPolicy,
        cleanup_ref: str | None,
    ) -> float | None:
        """校验映射生命周期与清理责任组合是否合法。

        Args:
            expires_at: 调用方传入的过期时间戳。
            cleanup_policy: 清理策略。
            cleanup_ref: 插件级清理责任标识。

        Returns:
            规范化后的过期时间戳；永久映射时返回 `None`。

        Raises:
            ValueError: 当生命周期与清理策略组合不满足约束时抛出。
        """

        if expires_at is None:
            if cleanup_policy != "managed_by_plugin" or cleanup_ref is None:
                raise ValueError("永久映射必须声明 managed_by_plugin 且提供 cleanup_ref")
            return None
        normalized_expires_at = float(expires_at)
        if normalized_expires_at <= 0:
            raise ValueError("expires_at 必须大于 0")
        return normalized_expires_at

    def _new_file_id(self) -> str:
        """生成新的稳定机器句柄。

        Returns:
            以 `gfid:` 为前缀的短随机唯一文件标识。
        """

        while True:
            token = "".join(secrets.choice(_GFID_ALPHABET) for _ in range(_GFID_LENGTH))
            file_id = f"{_GFID_PREFIX}{token}"
            if self._store.get_record(file_id) is None:
                return file_id


_managed_file_service: ManagedFileService | None = None


def get_managed_file_service() -> ManagedFileService:
    """返回全局共享的文件映射服务实例。

    Returns:
        单例模式的文件映射服务。
    """

    global _managed_file_service
    if _managed_file_service is None:
        _managed_file_service = ManagedFileService()
    return _managed_file_service
