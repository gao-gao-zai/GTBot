from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import AnythingLLMDocsPluginConfig, get_anythingllm_docs_plugin_config


class AnythingLLMClientError(RuntimeError):
    """AnythingLLM 接口调用失败异常。"""


@dataclass(frozen=True)
class AnythingLLMUploadedDocument:
    """上传到 AnythingLLM 后返回的文档元数据。"""

    location: str
    name: str
    title: str
    url: str = ""
    doc_source: str = ""
    published: str = ""
    word_count: int | None = None
    token_count_estimate: int | None = None


@dataclass(frozen=True)
class AnythingLLMWorkspace:
    """AnythingLLM 工作区摘要信息。"""

    id: int | None
    name: str
    slug: str
    documents: list[Any]


@dataclass(frozen=True)
class AnythingLLMChatResult:
    """AnythingLLM 问答结果。"""

    answer: str
    sources: list[dict[str, Any]]
    chunks: list[dict[str, Any]]


@dataclass(frozen=True)
class AnythingLLMWorkspaceDocument:
    """工作区中已挂载文档的标准化摘要。"""

    location: str
    document_name: str
    title: str
    source_url: str = ""
    doc_source: str = ""
    published: str = ""
    word_count: int | None = None
    token_count_estimate: int | None = None
    created_at: str = ""


class AnythingLLMClient:
    """AnythingLLM HTTP 客户端。"""

    def __init__(self, cfg: AnythingLLMDocsPluginConfig) -> None:
        """初始化客户端。

        Args:
            cfg: 插件配置对象。
        """

        self.cfg = cfg
        self._workspace_checked_slug: str | None = None

    async def ensure_workspace(self) -> AnythingLLMWorkspace:
        """确保全局共享工作区存在且配置已同步。"""

        self._ensure_api_key()
        target_slug = self.cfg.anythingllm.workspace_slug
        if self._workspace_checked_slug == target_slug:
            workspace = await self.get_workspace(target_slug)
            if workspace is not None:
                return workspace

        workspace = await self.get_workspace(target_slug)
        if workspace is None:
            created = await self.create_workspace()
            if created.slug != target_slug:
                raise AnythingLLMClientError(
                    f"工作区已创建，但 slug={created.slug} 与配置 workspace_slug={target_slug} 不一致，请调整 config.json"
                )
            workspace = created

        await self.update_workspace_settings(workspace.slug)
        self._workspace_checked_slug = workspace.slug
        refreshed = await self.get_workspace(workspace.slug)
        return refreshed or workspace

    async def list_workspaces(self) -> list[AnythingLLMWorkspace]:
        """列出实例中的全部工作区。"""

        payload = await self._request("GET", "/api/v1/workspaces")
        raw_items = payload.get("workspaces")
        if not isinstance(raw_items, list):
            return []
        return [self._workspace_from_dict(item) for item in raw_items if isinstance(item, dict)]

    async def get_workspace(self, slug: str) -> AnythingLLMWorkspace | None:
        """按 slug 查询工作区。"""

        text = str(slug).strip()
        if not text:
            return None

        try:
            payload = await self._request("GET", f"/api/v1/workspace/{text}")
        except AnythingLLMClientError as exc:
            if "404" in str(exc):
                return None
            raise

        workspace_value = payload.get("workspace")
        if isinstance(workspace_value, list) and workspace_value:
            workspace_value = workspace_value[0]
        if not isinstance(workspace_value, dict):
            return None
        return self._workspace_from_dict(workspace_value)

    async def list_workspace_documents(self, slug: str) -> list[AnythingLLMWorkspaceDocument]:
        """列出指定工作区当前真正挂载的文档。"""

        workspace = await self.get_workspace(slug)
        if workspace is None:
            return []

        documents: list[AnythingLLMWorkspaceDocument] = []
        for item in workspace.documents:
            if not isinstance(item, dict):
                continue
            documents.append(self._workspace_document_from_dict(item))
        return documents

    async def create_workspace(self) -> AnythingLLMWorkspace:
        """创建全局文档工作区。"""

        cfg = self.cfg.anythingllm
        payload = await self._request(
            "POST",
            "/api/v1/workspace/new",
            json_body={
                "name": cfg.workspace_name,
                "similarityThreshold": cfg.chat.similarity_threshold,
                "chatMode": cfg.chat.mode,
                "topN": cfg.chat.top_n,
            },
        )
        workspace = payload.get("workspace")
        if not isinstance(workspace, dict):
            raise AnythingLLMClientError("AnythingLLM 创建工作区失败：响应缺少 workspace")
        return self._workspace_from_dict(workspace)

    async def update_workspace_settings(self, slug: str) -> None:
        """同步工作区的基础检索设置。"""

        cfg = self.cfg.anythingllm
        await self._request(
            "POST",
            f"/api/v1/workspace/{slug}/update",
            json_body={
                "name": cfg.workspace_name,
                "similarityThreshold": cfg.chat.similarity_threshold,
                "chatMode": cfg.chat.mode,
                "topN": cfg.chat.top_n,
            },
        )

    async def upload_document(self, file_path: Path, *, add_to_workspace_slug: str) -> list[AnythingLLMUploadedDocument]:
        """上传文件到 AnythingLLM，并直接加入指定工作区。"""

        self._ensure_api_key()
        if not file_path.exists() or not file_path.is_file():
            raise AnythingLLMClientError(f"待上传文件不存在: {file_path}")

        files = {
            "file": (file_path.name, file_path.read_bytes(), "application/octet-stream"),
        }
        data = {
            "addToWorkspaces": add_to_workspace_slug,
        }
        payload = await self._request("POST", "/api/v1/document/upload", files=files, data=data)
        raw_docs = payload.get("documents")
        if not isinstance(raw_docs, list):
            raise AnythingLLMClientError("AnythingLLM 上传成功但未返回 documents")

        documents: list[AnythingLLMUploadedDocument] = []
        for item in raw_docs:
            if not isinstance(item, dict):
                continue
            documents.append(
                AnythingLLMUploadedDocument(
                    location=str(item.get("location") or ""),
                    name=str(item.get("name") or ""),
                    title=str(item.get("title") or file_path.name),
                    url=str(item.get("url") or ""),
                    doc_source=str(item.get("docSource") or ""),
                    published=str(item.get("published") or ""),
                    word_count=self._coerce_int(item.get("wordCount")),
                    token_count_estimate=self._coerce_int(item.get("token_count_estimate")),
                )
            )
        return documents

    async def remove_documents(self, *, names: list[str]) -> None:
        """从 AnythingLLM 系统中永久删除文档文件。"""

        cleaned = [str(item).strip() for item in names if str(item).strip()]
        if not cleaned:
            return
        await self._request("DELETE", "/api/v1/system/remove-documents", json_body={"names": cleaned})

    async def remove_documents_from_workspace(self, *, workspace_slug: str, names: list[str]) -> None:
        """将文档从指定工作区关联中移除。"""

        cleaned = [str(item).strip() for item in names if str(item).strip()]
        if not cleaned:
            return
        await self._request(
            "POST",
            f"/api/v1/workspace/{workspace_slug}/update-embeddings",
            json_body={"adds": [], "deletes": cleaned},
        )

    async def query_workspace(
        self,
        *,
        workspace_slug: str,
        question: str,
        session_id: str,
    ) -> AnythingLLMChatResult:
        """向指定工作区发起 SSE 文档问答请求。"""

        body = {
            "message": question,
            "mode": self.cfg.anythingllm.chat.mode,
            "sessionId": session_id,
            "reset": bool(self.cfg.anythingllm.chat.reset),
        }
        chunks = await self._stream_events(
            "POST",
            f"/api/v1/workspace/{workspace_slug}/stream-chat",
            json_body=body,
        )
        answer_parts: list[str] = []
        final_sources: list[dict[str, Any]] = []
        for item in chunks:
            text = str(item.get("textResponse") or "")
            if text:
                answer_parts.append(text)
            sources = item.get("sources")
            if isinstance(sources, list) and sources:
                final_sources = [source for source in sources if isinstance(source, dict)]
            error_text = str(item.get("error") or "").strip()
            if error_text and error_text.lower() != "null":
                raise AnythingLLMClientError(f"AnythingLLM 文档问答失败: {error_text}")

        return AnythingLLMChatResult(
            answer="".join(answer_parts).strip(),
            sources=final_sources,
            chunks=chunks,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送标准 JSON/表单请求。"""

        async with httpx.AsyncClient(
            timeout=self.cfg.anythingllm.timeout_sec,
            follow_redirects=True,
        ) as client:
            try:
                response = await client.request(
                    method=method,
                    url=self._url(path),
                    headers=self._headers(),
                    json=json_body,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip()
                raise AnythingLLMClientError(
                    f"AnythingLLM 请求失败: {method} {path} -> {exc.response.status_code} {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnythingLLMClientError(f"AnythingLLM 网络请求失败: {method} {path}: {exc!s}") from exc

        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise AnythingLLMClientError(f"AnythingLLM 返回了非 JSON 响应: {method} {path}") from exc
        if not isinstance(parsed, dict):
            raise AnythingLLMClientError(f"AnythingLLM 返回体不是 JSON 对象: {method} {path}")
        error_value = parsed.get("error")
        if error_value not in (None, "", False):
            raise AnythingLLMClientError(f"AnythingLLM 返回错误: {error_value}")
        return parsed

    async def _stream_events(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """发送 SSE 请求并解析事件流。"""

        events: list[dict[str, Any]] = []
        async with httpx.AsyncClient(
            timeout=self.cfg.anythingllm.timeout_sec,
            follow_redirects=True,
        ) as client:
            try:
                async with client.stream(
                    method=method,
                    url=self._url(path),
                    headers={
                        **self._headers(),
                        "Accept": "text/event-stream",
                    },
                    json=json_body,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        parsed = self._parse_sse_line(line)
                        if parsed is not None:
                            events.append(parsed)
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip()
                raise AnythingLLMClientError(
                    f"AnythingLLM 流式请求失败: {method} {path} -> {exc.response.status_code} {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnythingLLMClientError(f"AnythingLLM 流式网络请求失败: {method} {path}: {exc!s}") from exc
        return events

    def _parse_sse_line(self, line: str) -> dict[str, Any] | None:
        """解析单行 SSE 事件。"""

        text = str(line).strip()
        if not text:
            return None
        if text.startswith("event:"):
            return None
        if text.startswith("data:"):
            text = text[5:].strip()
        if not text or text == "[DONE]":
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _workspace_from_dict(self, data: dict[str, Any]) -> AnythingLLMWorkspace:
        """将 API 工作区对象转换为内部模型。"""

        return AnythingLLMWorkspace(
            id=self._coerce_int(data.get("id")),
            name=str(data.get("name") or ""),
            slug=str(data.get("slug") or ""),
            documents=list(data.get("documents") or []),
        )

    def _workspace_document_from_dict(self, data: dict[str, Any]) -> AnythingLLMWorkspaceDocument:
        """将工作区文档对象转换为内部模型。"""

        metadata_raw = data.get("metadata")
        metadata: dict[str, Any] = {}
        if isinstance(metadata_raw, str) and metadata_raw.strip():
            try:
                parsed = json.loads(metadata_raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                metadata = parsed
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw

        location = str(data.get("docpath") or data.get("location") or "").strip()
        document_name = str(data.get("filename") or Path(location).name or "").strip()
        title = str(metadata.get("title") or document_name or location).strip()

        return AnythingLLMWorkspaceDocument(
            location=location,
            document_name=document_name,
            title=title,
            source_url=str(metadata.get("url") or ""),
            doc_source=str(metadata.get("docSource") or ""),
            published=str(metadata.get("published") or ""),
            word_count=self._coerce_int(metadata.get("wordCount")),
            token_count_estimate=self._coerce_int(metadata.get("token_count_estimate")),
            created_at=str(data.get("createdAt") or ""),
        )

    def _headers(self) -> dict[str, str]:
        """构造 HTTP 请求头。"""

        headers = {
            "Accept": "application/json",
        }
        api_key = str(self.cfg.anythingllm.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _url(self, path: str) -> str:
        """拼接完整请求 URL。"""

        base_url = str(self.cfg.anythingllm.base_url).rstrip("/")
        return f"{base_url}{path}"

    def _ensure_api_key(self) -> None:
        """校验 API Key 已配置。"""

        if not str(self.cfg.anythingllm.api_key or "").strip():
            raise AnythingLLMClientError("AnythingLLM API Key 未配置，请先编辑插件根目录 config.json")

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        """将未知数值转换为整数。"""

        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


_client_cache: AnythingLLMClient | None = None


def get_anythingllm_client() -> AnythingLLMClient:
    """获取 AnythingLLM 客户端单例。"""

    global _client_cache
    if _client_cache is None:
        _client_cache = AnythingLLMClient(get_anythingllm_docs_plugin_config())
    return _client_cache
