from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from .client import AnythingLLMWorkspaceDocument
from .config import get_anythingllm_docs_plugin_config


class StoredDocumentRecord(BaseModel):
    """本地保存的已上传文档记录。"""

    record_id: str
    title: str
    file_name: str
    location: str
    document_name: str
    workspace_slug: str
    uploaded_by: int
    uploaded_at: str
    source_url: str = ""
    doc_source: str = ""
    published: str = ""
    word_count: int | None = None
    token_count_estimate: int | None = None


class StoredDocumentCollection(BaseModel):
    """本地文档记录集合。"""

    documents: list[StoredDocumentRecord] = Field(default_factory=list)


class AnythingLLMDocumentStore:
    """AnythingLLM 文档上传记录存储。"""

    def __init__(self, path: Path) -> None:
        """初始化本地存储对象。

        Args:
            path: JSON 状态文件路径。
        """

        self.path = path
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        """确保状态文件所在目录存在。"""

        await asyncio.to_thread(self.path.parent.mkdir, parents=True, exist_ok=True)

    async def list_documents(self) -> list[StoredDocumentRecord]:
        """读取全部文档记录。"""

        collection = await self._load()
        return sorted(collection.documents, key=lambda item: item.uploaded_at, reverse=True)

    async def add_document(
        self,
        *,
        title: str,
        file_name: str,
        location: str,
        document_name: str,
        workspace_slug: str,
        uploaded_by: int,
        source_url: str = "",
        doc_source: str = "",
        published: str = "",
        word_count: int | None = None,
        token_count_estimate: int | None = None,
    ) -> StoredDocumentRecord:
        """新增一条文档记录。"""

        async with self._lock:
            collection = await self._load_unlocked()
            normalized_document_name = str(document_name or "").strip()
            record = StoredDocumentRecord(
                record_id=uuid.uuid4().hex[:8],
                title=str(title or file_name).strip() or file_name,
                file_name=file_name,
                location=location,
                document_name=normalized_document_name,
                workspace_slug=workspace_slug,
                uploaded_by=int(uploaded_by),
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                source_url=source_url,
                doc_source=doc_source,
                published=published,
                word_count=word_count,
                token_count_estimate=token_count_estimate,
            )
            collection.documents = [
                item
                for item in collection.documents
                if item.location != record.location
                and (
                    not normalized_document_name
                    or not str(item.document_name or "").strip()
                    or item.document_name != normalized_document_name
                )
            ]
            collection.documents.append(record)
            await self._save_unlocked(collection)
            return record

    async def remove_by_record_id(self, record_id: str) -> StoredDocumentRecord | None:
        """按本地记录 ID 删除一条记录。"""

        target = str(record_id).strip()
        if not target:
            return None

        async with self._lock:
            collection = await self._load_unlocked()
            removed: StoredDocumentRecord | None = None
            remained: list[StoredDocumentRecord] = []
            for item in collection.documents:
                if item.record_id == target and removed is None:
                    removed = item
                    continue
                remained.append(item)
            if removed is None:
                return None
            collection.documents = remained
            await self._save_unlocked(collection)
            return removed

    async def find_by_record_id(self, record_id: str) -> StoredDocumentRecord | None:
        """按本地记录 ID 查找记录。"""

        target = str(record_id).strip()
        if not target:
            return None

        collection = await self._load()
        for item in collection.documents:
            if item.record_id == target:
                return item
        return None

    async def search_by_keyword(self, keyword: str) -> list[StoredDocumentRecord]:
        """按标题、文件名或文档名模糊查找记录。"""

        text = str(keyword).strip().lower()
        if not text:
            return []

        collection = await self._load()
        matched: list[StoredDocumentRecord] = []
        for item in collection.documents:
            haystacks = [item.title, item.file_name, item.document_name]
            if any(text in str(value).lower() for value in haystacks):
                matched.append(item)
        return sorted(matched, key=lambda item: item.uploaded_at, reverse=True)

    async def sync_with_workspace_documents(
        self,
        *,
        workspace_slug: str,
        api_documents: list[AnythingLLMWorkspaceDocument],
    ) -> list[StoredDocumentRecord]:
        """按 API 真源把本地记录整体对齐到当前工作区文档清单。

        Args:
            workspace_slug: 目标工作区 slug。
            api_documents: 当前工作区从 AnythingLLM API 拉取到的文档清单。

        Returns:
            对齐后的全部本地记录，已按上传时间倒序排序。
        """

        async with self._lock:
            collection = await self._load_unlocked()
            target_slug = str(workspace_slug).strip()
            api_by_location = {
                str(item.location).strip(): item
                for item in api_documents
                if str(item.location).strip()
            }

            preserved_other_workspaces = [
                item for item in collection.documents if item.workspace_slug != target_slug
            ]
            existing_current = {
                item.location: item
                for item in collection.documents
                if item.workspace_slug == target_slug and str(item.location).strip()
            }

            reconciled_current: list[StoredDocumentRecord] = []
            for location, api_item in api_by_location.items():
                existed = existing_current.get(location)
                uploaded_at = self._normalize_uploaded_at(
                    existed.uploaded_at if existed is not None else (api_item.created_at or api_item.published)
                )
                record = StoredDocumentRecord(
                    record_id=existed.record_id if existed is not None else uuid.uuid4().hex[:8],
                    title=str(api_item.title or api_item.document_name or Path(location).name).strip() or Path(location).name,
                    file_name=(
                        existed.file_name
                        if existed is not None and str(existed.file_name).strip()
                        else (api_item.title or api_item.document_name or Path(location).name)
                    ),
                    location=location,
                    document_name=str(api_item.document_name or Path(location).name).strip(),
                    workspace_slug=target_slug,
                    uploaded_by=existed.uploaded_by if existed is not None else 0,
                    uploaded_at=uploaded_at,
                    source_url=str(api_item.source_url or (existed.source_url if existed is not None else "")),
                    doc_source=str(api_item.doc_source or (existed.doc_source if existed is not None else "")),
                    published=str(api_item.published or (existed.published if existed is not None else "")),
                    word_count=api_item.word_count,
                    token_count_estimate=api_item.token_count_estimate,
                )
                reconciled_current.append(record)

            collection.documents = preserved_other_workspaces + reconciled_current
            await self._save_unlocked(collection)
            return sorted(collection.documents, key=lambda item: item.uploaded_at, reverse=True)

    async def _load(self) -> StoredDocumentCollection:
        """在锁外读取状态文件。"""

        async with self._lock:
            return await self._load_unlocked()

    async def _load_unlocked(self) -> StoredDocumentCollection:
        """在已持有锁的情况下读取状态文件。"""

        await self.ensure_ready()
        if not self.path.exists():
            return StoredDocumentCollection()

        raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        if not raw.strip():
            return StoredDocumentCollection()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError("anythingllm_docs documents.json must be a JSON object")
        return cast(StoredDocumentCollection, StoredDocumentCollection.model_validate(parsed))

    async def _save_unlocked(self, collection: StoredDocumentCollection) -> None:
        """在已持有锁的情况下写回状态文件。"""

        await self.ensure_ready()
        payload = json.dumps(collection.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        await asyncio.to_thread(tmp.write_text, payload, encoding="utf-8")
        await asyncio.to_thread(tmp.replace, self.path)

    @staticmethod
    def _normalize_uploaded_at(value: str) -> str:
        """规范化上传时间，无法解析时回退到当前 UTC 时间。"""

        text = str(value or "").strip()
        if not text:
            return datetime.now(timezone.utc).isoformat()
        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).astimezone(timezone.utc).isoformat()
        except ValueError:
            return datetime.now(timezone.utc).isoformat()


_store_cache: AnythingLLMDocumentStore | None = None


def get_anythingllm_document_store() -> AnythingLLMDocumentStore:
    """获取 AnythingLLM 文档状态存储单例。"""

    global _store_cache
    if _store_cache is None:
        cfg = get_anythingllm_docs_plugin_config()
        _store_cache = AnythingLLMDocumentStore(cfg.store_file_path)
    return _store_cache
