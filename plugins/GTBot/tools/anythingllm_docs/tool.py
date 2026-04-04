from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool

from plugins.GTBot.services.chat.context import GroupChatContext

from .client import AnythingLLMClientError, get_anythingllm_client
from .config import get_anythingllm_docs_plugin_config
from .store import StoredDocumentRecord, get_anythingllm_document_store


def _build_session_id(ctx: GroupChatContext, prefix: str) -> str:
    """构造发送到 AnythingLLM 的会话标识。

    Args:
        ctx: GTBot 当前会话上下文。
        prefix: 配置中定义的会话前缀。

    Returns:
        适用于 AnythingLLM `sessionId` 的稳定字符串。
    """

    chat_type = str(getattr(ctx, "chat_type", "group") or "group")
    group_id = int(getattr(ctx, "group_id", 0) or 0)
    user_id = int(getattr(ctx, "user_id", 0) or 0)
    if chat_type == "group" and group_id > 0:
        return f"{prefix}-group-{group_id}"
    return f"{prefix}-private-{user_id}"


def _format_sources(sources: list[dict[str, Any]]) -> str:
    """格式化 AnythingLLM 返回的来源列表。

    Args:
        sources: AnythingLLM 返回的来源对象列表。

    Returns:
        适合直接回传给 Agent 的来源摘要文本。
    """

    if not sources:
        return "(无来源)"

    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        title = str(source.get("title") or source.get("source") or f"来源{index}").strip()
        lines.append(f"{index}. {title}")
    return "\n".join(lines)


def _format_document_line(record: StoredDocumentRecord) -> str:
    """格式化单条文档记录，便于在工具输出中复用。

    Args:
        record: 本地文档记录。

    Returns:
        包含记录 ID、标题、文件名和上传人信息的一行摘要。
    """

    uploaded_at = str(record.uploaded_at).replace("T", " ").replace("+00:00", " UTC")
    return (
        f"[{record.record_id}] {record.title} | file={record.file_name} | "
        f"doc={record.document_name} | by={record.uploaded_by} | at={uploaded_at}"
    )


async def list_anythingllm_documents_impl(limit: int = 20) -> str:
    """列出当前可供 Agent 查询的文档。

    Args:
        limit: 最多返回多少条文档摘要。会自动限制在 1 到 100 之间。

    Returns:
        适合直接回传给 Agent 的文档列表文本，包含记录 ID、标题、文件名和上传信息。
    """

    normalized_limit = max(1, min(int(limit), 100))
    client = get_anythingllm_client()
    workspace = await client.ensure_workspace()
    api_documents = await client.list_workspace_documents(workspace.slug)
    store = get_anythingllm_document_store()
    documents = await store.sync_with_workspace_documents(
        workspace_slug=workspace.slug,
        api_documents=api_documents,
    )
    if not documents:
        return "当前没有可查询的文档。管理员可先给机器人发送文件完成入库。"

    lines = [
        "当前可查询文档列表：",
        "这些记录仅用于查看当前文档库中有哪些文档，不代表查询工具可以对单份文档做硬过滤。",
    ]
    for item in documents[:normalized_limit]:
        lines.append(_format_document_line(item))
    if len(documents) > normalized_limit:
        lines.append(f"... 其余 {len(documents) - normalized_limit} 条未展示")
    return "\n".join(lines)


async def query_anythingllm_documents_impl(
    question: str,
    runtime: ToolRuntime[GroupChatContext],
) -> str:
    """查询全局文档库。

    Args:
        question: 要向文档库提出的自然语言问题。
        runtime: GTBot 当前工具运行时，用于获取群聊或私聊上下文并构造会话 ID。

    Returns:
        包含 `[scope]`、`[answer]`、`[sources]` 三段结构的文本。
        如果未命中文档、插件被禁用或 AnythingLLM 请求失败，也会返回明确的中文提示。
        当前工具只支持对整个工作区查询，不支持对单份文档做硬过滤。
    """

    text = str(question).strip()
    if not text:
        raise ValueError("question 不能为空")

    cfg = get_anythingllm_docs_plugin_config()
    if not bool(cfg.enabled):
        return "anythingllm_docs 插件当前已禁用。"

    client = get_anythingllm_client()
    ctx = runtime.context
    session_id = _build_session_id(ctx, cfg.anythingllm.chat.session_prefix)

    try:
        workspace = await client.ensure_workspace()
        result = await client.query_workspace(
            workspace_slug=workspace.slug,
            question=text,
            session_id=session_id,
        )
    except AnythingLLMClientError as exc:
        return f"文档查询失败: {exc!s}"

    answer = str(result.answer or "").strip()
    if not answer and not result.sources:
        return "未找到相关文档内容。"

    lines: list[str] = []
    lines.append("[scope]")
    lines.append("全局文档库")
    lines.append("[answer]")
    lines.append(answer or "未找到相关文档内容。")
    lines.append("[sources]")
    lines.append(_format_sources(result.sources))
    return "\n".join(lines)


@tool("list_anythingllm_documents")
async def list_anythingllm_documents_tool(limit: int = 20) -> str:
    """列出当前 AnythingLLM 全局文档库中可查询的文档摘要。

    Args:
        limit: 可选，限制最多返回多少条文档摘要。默认 20，最大 100。

    Returns:
        一个适合 GTBot Agent 直接阅读和后续引用的文档列表字符串。
        每条记录都会包含本地记录 ID、标题、文件名、AnythingLLM 文档名、上传人和上传时间。
        本工具主要用于帮助 Agent 判断当前文档库中有哪些资料可用，不代表后续问答可以对单份文档做硬过滤。

    Example:
        先调用 `list_anythingllm_documents(limit=10)` 查看最近的可查询文档，
        再决定是否继续对整个文档库发起问答。
    """

    return await list_anythingllm_documents_impl(limit=limit)


@tool("query_anythingllm_documents")
async def query_anythingllm_documents_tool(
    question: str,
    runtime: ToolRuntime[GroupChatContext],
) -> str:
    """向 AnythingLLM 全局文档库发起问答。

    Args:
        question: 必填，自然语言问题，例如“退货规则是什么”或“总结第三章重点”。
        runtime: GTBot 当前工具运行时。工具会使用其中的群号、用户号等上下文生成稳定的会话 ID。

    Returns:
        固定包含 `[scope]`、`[answer]`、`[sources]` 三段内容的字符串。
        当前工具始终对整个工作区发起查询，不支持通过参数把检索范围硬性限定到某一份文档。
        当文档库无结果时，也会返回明确提示，便于 Agent 决定是改问法还是继续向用户追问。

    Example:
        `query_anythingllm_documents(question="总结安装步骤")`
        表示对当前 AnythingLLM 全局文档库发起一次普通问答。
    """

    return await query_anythingllm_documents_impl(question=question, runtime=runtime)
