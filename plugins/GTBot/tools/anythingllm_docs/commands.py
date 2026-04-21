from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from ...services.help import HelpArgumentSpec, HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, require_admin
from .client import AnythingLLMClientError, AnythingLLMUploadedDocument, get_anythingllm_client
from .config import AnythingLLMDocsPluginConfig, get_anythingllm_docs_plugin_config
from .store import StoredDocumentRecord, get_anythingllm_document_store

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


class DocumentCommandError(RuntimeError):
    """表示文档插件管理员命令执行失败的异常。"""


def _pick_first_string(data: dict[str, Any], keys: list[str]) -> str | None:
    """按优先级提取字典中的首个非空字符串。

    Args:
        data: 待读取的字典。
        keys: 候选键名列表，按优先级排序。

    Returns:
        第一个非空字符串；如果没有匹配值则返回 `None`。
    """

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_remote_url(value: str | None) -> str | None:
    """规范化可能来自 OneBot 的远程文件 URL。

    Args:
        value: 原始 URL 字符串。

    Returns:
        标准化后的 `http` 或 `https` URL；无法识别时返回 `None`。
    """

    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith(("http://", "https://")):
        return text
    return None


def _suffix_from_url(url: str, default: str = ".bin") -> str:
    """根据 URL 推断文件后缀。

    Args:
        url: 远程下载地址。
        default: 无法推断时使用的默认后缀。

    Returns:
        URL 路径中的文件后缀，若为空则返回默认值。
    """

    suffix = Path(urlparse(url).path).suffix
    return suffix or default


async def _download_remote_file(url: str, target: Path, *, timeout_sec: float) -> Path:
    """下载远程文件到本地路径。

    Args:
        url: 文件下载地址。
        target: 本地目标文件路径。
        timeout_sec: 下载超时时间，单位秒。

    Returns:
        下载完成后的本地文件路径。
    """

    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        await asyncio.to_thread(target.write_bytes, response.content)
    return target


async def _try_fetch_remote_file_via_api(bot: Bot, data: dict[str, Any]) -> tuple[str | None, str | None]:
    """尝试通过 OneBot 文件相关 API 获取下载地址。

    Args:
        bot: 当前 OneBot Bot 实例。
        data: 文件段数据。

    Returns:
        一个二元组。
        第一个元素是远程 URL 或本地临时路径。
        第二个元素是解析出的文件 ID。
    """

    file_id = _pick_first_string(data, ["file_id", "fid", "id"])
    busid = data.get("busid")
    if not file_id:
        return None, None

    candidates = [
        ("get_file", {"file_id": file_id}),
        ("get_file", {"fid": file_id}),
        ("get_file_url", {"file_id": file_id}),
        ("get_file_url", {"fid": file_id}),
    ]
    if busid is not None:
        candidates.extend(
            [
                ("get_file", {"file_id": file_id, "busid": busid}),
                ("get_file", {"fid": file_id, "busid": busid}),
                ("get_file_url", {"file_id": file_id, "busid": busid}),
                ("get_file_url", {"fid": file_id, "busid": busid}),
            ]
        )

    for action, params in candidates:
        try:
            response = await bot.call_api(action, **params)
        except Exception:
            continue

        if isinstance(response, str) and response.startswith(("http://", "https://")):
            return response, file_id
        if not isinstance(response, dict):
            continue

        url = _pick_first_string(response, ["url", "download_url", "file_url"])
        path = _pick_first_string(response, ["file", "path", "temp_file"])
        if url:
            return url, file_id
        if path:
            return path, file_id
    return None, file_id


def _coerce_int(value: Any) -> int | None:
    """将任意值安全转换为整数。

    Args:
        value: 待转换的原始值。

    Returns:
        成功时返回整数，失败时返回 `None`。
    """

    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_reply_message_id_from_event(event: MessageEvent) -> int | None:
    """从当前事件中提取被回复消息的 ID。

    Args:
        event: 当前消息事件。

    Returns:
        被回复消息的 `message_id`；若当前消息未引用其他消息则返回 `None`。
    """

    reply = getattr(event, "reply", None)
    if reply is not None:
        message_id = getattr(reply, "message_id", None)
        if isinstance(message_id, int):
            return message_id

    message = getattr(event, "message", None)
    if message is None:
        return None

    try:
        for segment in message:
            if getattr(segment, "type", "") != "reply":
                continue
            raw_id = getattr(segment, "data", {}).get("id")
            if raw_id is None:
                continue
            return int(raw_id)
    except Exception:
        return None
    return None


def _message_segments(raw_message: dict[str, Any]) -> list[dict[str, Any]]:
    """把 OneBot 原始消息对象转换为统一的消息段列表。

    Args:
        raw_message: `bot.get_msg()` 返回的原始消息对象。

    Returns:
        统一格式的消息段列表。
    """

    message = raw_message.get("message")
    if isinstance(message, list):
        segments: list[dict[str, Any]] = []
        for segment in message:
            if isinstance(segment, dict):
                segments.append(segment)
            elif hasattr(segment, "type") and hasattr(segment, "data"):
                segments.append({"type": segment.type, "data": dict(segment.data)})
        return segments

    if message is not None and hasattr(message, "__iter__") and not isinstance(message, (str, bytes, dict)):
        segments = []
        for segment in message:
            if hasattr(segment, "type") and hasattr(segment, "data"):
                segments.append({"type": segment.type, "data": dict(segment.data)})
        if segments:
            return segments

    return []


def _extract_file_segment_from_raw_message(raw_message: dict[str, Any]) -> dict[str, Any] | None:
    """从被回复的原始消息中提取首个文件段。

    Args:
        raw_message: `bot.get_msg()` 返回的原始消息对象。

    Returns:
        文件段字典；若没有文件段则返回 `None`。
    """

    for segment in _message_segments(raw_message):
        if segment.get("type") == "file":
            return segment
    return None


async def _download_file_from_segment_data(
    bot: Bot,
    data: dict[str, Any],
    *,
    event_time: int,
    cfg: AnythingLLMDocsPluginConfig,
) -> tuple[Path, str, int | None]:
    """根据文件段数据下载或复制出本地临时文件。

    Args:
        bot: 当前 OneBot Bot 实例。
        data: 文件段中的 `data` 字典。
        event_time: 文件消息时间戳，用于构造临时文件名。
        cfg: 插件配置。

    Returns:
        一个三元组，依次为本地临时文件路径、原始文件名、声明的文件大小。

    Raises:
        DocumentCommandError: 当无法解析或下载文件时抛出。
    """

    file_name = _pick_first_string(data, ["name", "file_name", "file"]) or "upload.bin"
    declared_size = _coerce_int(data.get("file_size") or data.get("size"))
    file_path_raw = _pick_first_string(data, ["file", "path", "temp_file"])
    remote_url = _normalize_remote_url(_pick_first_string(data, ["url", "file_url", "download_url"]))

    if not remote_url:
        fetched_value, _file_id = await _try_fetch_remote_file_via_api(bot, data)
        if fetched_value:
            normalized = _normalize_remote_url(fetched_value)
            if normalized:
                remote_url = normalized
            else:
                file_path_raw = fetched_value

    suffix = Path(file_name).suffix
    if not suffix and remote_url:
        suffix = _suffix_from_url(remote_url)
    target = cfg.temp_dir_path / f"{Path(file_name).stem}-{int(event_time or 0)}{suffix or '.bin'}"
    target.parent.mkdir(parents=True, exist_ok=True)

    if remote_url:
        try:
            await _download_remote_file(remote_url, target, timeout_sec=cfg.anythingllm.timeout_sec)
            return target, file_name, declared_size
        except Exception as exc:  # noqa: BLE001
            raise DocumentCommandError(f"下载 QQ 文件失败: {exc!s}") from exc

    if file_path_raw:
        candidate = Path(file_path_raw)
        if candidate.exists() and candidate.is_file():
            await asyncio.to_thread(target.write_bytes, candidate.read_bytes())
            return target, file_name, declared_size
        if file_path_raw.startswith(("http://", "https://")):
            try:
                await _download_remote_file(file_path_raw, target, timeout_sec=cfg.anythingllm.timeout_sec)
                return target, file_name, declared_size
            except Exception as exc:  # noqa: BLE001
                raise DocumentCommandError(f"下载 QQ 文件失败: {exc!s}") from exc

    raise DocumentCommandError("无法解析该文件消息的下载地址，请确认 OneBot 侧支持文件下载 API。")


async def _resolve_referenced_file(
    bot: Bot,
    event: MessageEvent,
    cfg: AnythingLLMDocsPluginConfig,
) -> tuple[Path, str, int | None]:
    """把当前命令回复引用的文件消息解析为本地临时文件。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前命令消息事件。
        cfg: 插件配置。

    Returns:
        一个三元组，依次为本地临时文件路径、原始文件名、声明的文件大小。

    Raises:
        DocumentCommandError: 当当前命令没有回复文件消息，或被回复消息无法解析时抛出。
    """

    reply_message_id = _extract_reply_message_id_from_event(event)
    if reply_message_id is None:
        raise DocumentCommandError(
            "请先回复一条文件消息后，再发送 /上传文档、#上传文档 或 #发送文件上传文档。"
        )

    raw_message = await bot.get_msg(message_id=int(reply_message_id))
    if not isinstance(raw_message, dict):
        raise DocumentCommandError("无法获取被回复的消息内容。")

    file_segment = _extract_file_segment_from_raw_message(raw_message)
    if file_segment is None:
        raise DocumentCommandError("被回复消息中没有文件段，请回复一条文件消息。")

    message_time = _coerce_int(raw_message.get("time")) or int(getattr(event, "time", 0) or 0)
    return await _download_file_from_segment_data(
        bot,
        dict(file_segment.get("data") or {}),
        event_time=message_time,
        cfg=cfg,
    )


def _is_extension_allowed(file_name: str, cfg: AnythingLLMDocsPluginConfig) -> bool:
    """检查文件扩展名是否在白名单中。

    Args:
        file_name: 原始文件名。
        cfg: 插件配置。

    Returns:
        当扩展名在允许列表中时返回 `True`。
    """

    suffix = Path(file_name).suffix.lower()
    allowed = {str(item).lower() for item in cfg.anythingllm.allowed_extensions}
    return bool(suffix) and suffix in allowed


def _check_file_size(file_path: Path, cfg: AnythingLLMDocsPluginConfig, declared_size: int | None) -> None:
    """校验文件大小是否超出限制。

    Args:
        file_path: 本地临时文件路径。
        cfg: 插件配置。
        declared_size: 来自消息段的声明大小；若为空则回退到本地文件大小。

    Raises:
        DocumentCommandError: 当文件超过大小限制时抛出。
    """

    max_bytes = int(cfg.anythingllm.max_file_size_mb) * 1024 * 1024
    effective_size = declared_size if declared_size is not None and declared_size > 0 else file_path.stat().st_size
    if effective_size > max_bytes:
        raise DocumentCommandError(
            f"文件大小超出限制：{effective_size / 1024 / 1024:.2f}MB > {cfg.anythingllm.max_file_size_mb}MB"
        )


def _format_uploaded_document(record: StoredDocumentRecord) -> str:
    """格式化单条文档记录，供列表展示使用。

    Args:
        record: 本地保存的文档记录。

    Returns:
        一行可读的文档摘要。
    """

    count_text = ""
    if record.word_count is not None:
        count_text = f" words={record.word_count}"
    return f"[{record.record_id}] {record.title} file={record.file_name} by={record.uploaded_by}{count_text}"


def _help_role() -> PermissionRole:
    """返回帮助系统中管理员命令使用的权限等级。"""

    return PermissionRole.ADMIN


def _register_help_items() -> None:
    """注册文档插件的管理员命令和 Agent 工具帮助信息。"""

    register_help(
        HelpCommandSpec(
            name="查看文档列表",
            category="文档知识库",
            summary="查看机器人通过 QQ 上传到全局文档库的文件列表。",
            description="仅管理员可用，列出当前全局共享文档工作区中由机器人记录的文档，以及后续删除或人工核对时可使用的本地记录 ID。",
            examples=("/查看文档列表",),
            required_role=_help_role(),
            audience="管理员命令",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="删除文档",
            category="文档知识库",
            summary="从全局文档库移除指定文档。",
            description="仅管理员可用。优先传入文档列表中的本地记录 ID；也可传标题关键字，但关键字命中多个文档时需要先缩小范围。",
            arguments=(
                HelpArgumentSpec(
                    name="<文档ID或关键字>",
                    description="文档列表中的本地记录 ID，或用于模糊匹配的标题关键字。",
                ),
            ),
            examples=("/删除文档 ab12cd34", "/删除文档 产品手册"),
            required_role=_help_role(),
            audience="管理员命令",
            sort_key=11,
        )
    )
    register_help(
        HelpCommandSpec(
            name="上传文档",
            category="文档知识库",
            summary="回复一条文件消息后执行该命令，将文件上传到全局文档库。",
            description=(
                "仅管理员可用。请先回复一条文件消息，再发送 `#上传文档` 或 `#发送文件上传文档`。"
                "校验通过后，机器人会先提示“已开始上传文档，正在下载并写入 AnythingLLM，请稍候”。"
                "随后插件会解析被回复的文件并上传到 AnythingLLM 全局共享工作区。"
            ),
            examples=(
                "先回复文件消息，再发送 #上传文档",
                "先回复文件消息，再发送 #发送文件上传文档",
            ),
            required_role=_help_role(),
            audience="管理员命令",
            sort_key=12,
        )
    )
    register_help(
        HelpCommandSpec(
            name="可查询文档列表工具",
            category="文档知识库",
            summary="供 GTBot Agent 列出当前可查询文档的工具。",
            description=(
                "该工具名为 `list_anythingllm_documents`。"
                "Agent 在不知道文档库里有哪些文档时，应先调用这个工具查看文档清单。"
                "工具输出中的记录 ID 主要用于文档管理和人工确认，不代表后续问答支持按单份文档硬过滤。"
            ),
            examples=(
                "当用户说“先看看现在能查哪些文档”时，Agent 应调用 `list_anythingllm_documents`。",
                "当用户怀疑某份文档是否已经入库时，Agent 应先列出文档再继续回答。",
            ),
            required_role=PermissionRole.USER,
            audience="Agent 工具",
            sort_key=13,
        )
    )
    register_help(
        HelpCommandSpec(
            name="文档查询工具",
            category="文档知识库",
            summary="供 GTBot Agent 调用的全局文档问答工具。",
            description=(
                "该工具名为 `query_anythingllm_documents`。"
                "它不是用户手打命令，而是提供给 GTBot 内部 Agent 使用。"
                "当用户问题依赖管理员上传的 PDF、Word、Excel、Markdown 或文本资料时，"
                "Agent 应优先调用该工具到 AnythingLLM 全局工作区检索并总结答案。"
                "工具返回内容只包含总结答案和来源标题，不会直接返回原文片段。"
                "当前工具对整个工作区查询，不支持通过参数把检索范围硬性限定到某一份文档。"
            ),
            examples=(
                "用户提问“产品手册里退货规则是什么”时，Agent 应调用 `query_anythingllm_documents`。",
                "用户提问“总结当前知识库里和退货有关的规则”时，Agent 应调用 `query_anythingllm_documents(question=...)`。",
            ),
            required_role=PermissionRole.USER,
            audience="Agent 工具",
            sort_key=14,
        )
    )
    register_help(
        HelpCommandSpec(
            name="文档插件配置",
            category="文档知识库",
            summary="AnythingLLM 文档插件的配置文件位置与关键字段说明。",
            description=(
                "插件配置固定写在 `plugins/GTBot/tools/anythingllm_docs/config.json`。"
                "至少需要配置 AnythingLLM 的 `base_url`、`api_key`、`workspace_name`、"
                "`workspace_slug`、允许上传的扩展名和文件大小限制。"
                "首次接入时可参考同目录下的 `config.json.example`，详细说明可查看 `使用指南.md`。"
            ),
            examples=(
                "配置文件路径：plugins/GTBot/tools/anythingllm_docs/config.json",
                "示例文件路径：plugins/GTBot/tools/anythingllm_docs/config.json.example",
                "详细说明：plugins/GTBot/tools/anythingllm_docs/使用指南.md",
            ),
            required_role=PermissionRole.ADMIN,
            audience="管理员与维护者",
            sort_key=15,
        )
    )


UploadDocumentCommand = on_command("上传文档", aliases={"发送文件上传文档"}, priority=4, block=True)
ListDocumentsCommand = on_command("查看文档列表", priority=4, block=True)
DeleteDocumentCommand = on_command("删除文档", priority=4, block=True)

_register_help_items()


@UploadDocumentCommand.handle()
async def handle_upload_document(bot: Bot, event: MessageEvent) -> None:
    """处理管理员通过“回复文件 + 上传命令”触发的入库流程。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前命令消息事件。
    """

    cfg = get_anythingllm_docs_plugin_config()
    if not bool(cfg.enabled):
        await UploadDocumentCommand.finish("anythingllm_docs 插件当前已禁用。")

    try:
        await require_admin(int(event.user_id))
    except PermissionError:
        await UploadDocumentCommand.finish("你没有上传文档到全局知识库的权限。")

    temp_path: Path | None = None
    try:
        temp_path, file_name, declared_size = await _resolve_referenced_file(bot, event, cfg)
        if not _is_extension_allowed(file_name, cfg):
            raise DocumentCommandError(
                f"不支持的文件类型：{Path(file_name).suffix or '(无扩展名)'}。允许类型：{', '.join(cfg.anythingllm.allowed_extensions)}"
            )
        _check_file_size(temp_path, cfg, declared_size)
        await UploadDocumentCommand.send("已开始上传文档，正在下载并写入 AnythingLLM，请稍候。")

        client = get_anythingllm_client()
        workspace = await client.ensure_workspace()
        uploaded = await client.upload_document(temp_path, add_to_workspace_slug=workspace.slug)
        if not uploaded:
            raise DocumentCommandError("AnythingLLM 未返回任何上传文档信息。")
        refreshed_workspace = await client.get_workspace(workspace.slug)
        if refreshed_workspace is None or not _workspace_has_uploaded_documents(refreshed_workspace, uploaded):
            raise DocumentCommandError(
                "文件已上传到 AnythingLLM 文档池，但没有成功加入目标工作区。"
                "请检查 AnythingLLM 的 embeddings / workspace 挂载状态。"
            )

        saved_records: list[StoredDocumentRecord] = []
        for item in uploaded:
            saved_records.append(
                await _save_uploaded_record(
                    item=item,
                    file_name=file_name,
                    uploaded_by=int(event.user_id),
                    workspace_slug=workspace.slug,
                )
            )

        first = saved_records[0]
        lines: list[str] = [
            "文档上传成功。",
            f"工作区: {workspace.name} ({workspace.slug})",
            f"文档数: {len(saved_records)}",
            f"首条记录ID: {first.record_id}",
            f"标题: {first.title}",
        ]
        if first.word_count is not None:
            lines.append(f"词数: {first.word_count}")
        if first.token_count_estimate is not None:
            lines.append(f"Token估算: {first.token_count_estimate}")
        await UploadDocumentCommand.finish("\n".join(lines))
    except (DocumentCommandError, AnythingLLMClientError) as exc:
        logger.warning("anythingllm_docs upload failed: user_id=%s error=%s", event.user_id, exc)
        await UploadDocumentCommand.finish(f"文档上传失败: {exc!s}")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


async def _save_uploaded_record(
    *,
    item: AnythingLLMUploadedDocument,
    file_name: str,
    uploaded_by: int,
    workspace_slug: str,
) -> StoredDocumentRecord:
    """将上传成功的文档写入本地状态文件。

    Args:
        item: AnythingLLM 返回的上传结果。
        file_name: 原始文件名。
        uploaded_by: 上传者 QQ 号。
        workspace_slug: 文档所在工作区 slug。

    Returns:
        新保存的本地文档记录。
    """

    store = get_anythingllm_document_store()
    return await store.add_document(
        title=item.title,
        file_name=file_name,
        location=item.location,
        document_name=item.name or Path(item.location).name,
        workspace_slug=workspace_slug,
        uploaded_by=uploaded_by,
        source_url=item.url,
        doc_source=item.doc_source,
        published=item.published,
        word_count=item.word_count,
        token_count_estimate=item.token_count_estimate,
    )


def _workspace_has_uploaded_documents(workspace: Any, uploaded: list[AnythingLLMUploadedDocument]) -> bool:
    """检查上传结果中的文档是否真的已经挂到目标工作区。

    Args:
        workspace: AnythingLLM 返回的工作区对象，要求至少包含 `documents` 字段。
        uploaded: 本次上传接口返回的文档列表。

    Returns:
        只要上传结果里任一文档已经出现在工作区文档列表中，就返回 `True`。
    """

    workspace_documents = list(getattr(workspace, "documents", []) or [])
    if not workspace_documents or not uploaded:
        return False

    known_paths: set[str] = set()
    for item in workspace_documents:
        if not isinstance(item, dict):
            continue
        for key in ("docpath", "location", "filename"):
            value = str(item.get(key) or "").strip()
            if value:
                known_paths.add(value)

    for item in uploaded:
        candidates = {
            str(item.location or "").strip(),
            Path(str(item.location or "")).name,
            str(item.name or "").strip(),
        }
        if any(candidate and candidate in known_paths for candidate in candidates):
            return True
    return False


async def _sync_local_store_from_api(workspace_slug: str) -> list[StoredDocumentRecord]:
    """从 AnythingLLM API 拉取工作区文档并对齐本地记录。"""

    client = get_anythingllm_client()
    api_documents = await client.list_workspace_documents(workspace_slug)
    store = get_anythingllm_document_store()
    return await store.sync_with_workspace_documents(
        workspace_slug=workspace_slug,
        api_documents=api_documents,
    )


@ListDocumentsCommand.handle()
async def handle_list_documents(event: MessageEvent) -> None:
    """处理管理员的文档列表命令。

    Args:
        event: 当前消息事件。
    """

    try:
        await require_admin(int(event.user_id))
    except PermissionError:
        await ListDocumentsCommand.finish("你没有查看文档列表的权限。")

    try:
        workspace = await get_anythingllm_client().ensure_workspace()
        documents = await _sync_local_store_from_api(workspace.slug)
    except AnythingLLMClientError as exc:
        await ListDocumentsCommand.finish(f"同步工作区文档失败: {exc!s}")
    if not documents:
        await ListDocumentsCommand.finish("当前全局文档库还没有通过机器人上传的文档。")

    lines: list[str] = ["当前机器人已记录的文档列表："]
    for item in documents[:50]:
        lines.append(_format_uploaded_document(item))
    if len(documents) > 50:
        lines.append(f"... 其余 {len(documents) - 50} 条未展示")
    await ListDocumentsCommand.finish("\n".join(lines))


@DeleteDocumentCommand.handle()
async def handle_delete_document(event: MessageEvent, args: Message = CommandArg()) -> None:
    """处理管理员删除文档命令。

    Args:
        event: 当前消息事件。
        args: 命令参数消息。
    """

    try:
        await require_admin(int(event.user_id))
    except PermissionError:
        await DeleteDocumentCommand.finish("你没有删除文档的权限。")

    raw_arg = args.extract_plain_text().strip()
    if not raw_arg:
        await DeleteDocumentCommand.finish("用法: /删除文档 <文档ID或关键字>")

    client = get_anythingllm_client()
    try:
        workspace = await client.ensure_workspace()
        await _sync_local_store_from_api(workspace.slug)
    except AnythingLLMClientError as exc:
        await DeleteDocumentCommand.finish(f"同步工作区文档失败: {exc!s}")

    store = get_anythingllm_document_store()
    target = await store.find_by_record_id(raw_arg)
    if target is None:
        matched = await store.search_by_keyword(raw_arg)
        if not matched:
            await DeleteDocumentCommand.finish(f"未找到与 “{raw_arg}” 对应的文档记录。")
        if len(matched) > 1:
            lines = [f"关键字 “{raw_arg}” 命中了多个文档，请改用记录ID删除："]
            for item in matched[:10]:
                lines.append(_format_uploaded_document(item))
            await DeleteDocumentCommand.finish("\n".join(lines))
        target = matched[0]

    try:
        await client.remove_documents_from_workspace(
            workspace_slug=target.workspace_slug,
            names=[target.location],
        )
        await client.remove_documents(
            names=[target.location or target.document_name],
        )
    except AnythingLLMClientError as exc:
        await DeleteDocumentCommand.finish(f"删除文档失败: {exc!s}")
    documents = await _sync_local_store_from_api(target.workspace_slug)
    if any(item.record_id == target.record_id for item in documents):
        await DeleteDocumentCommand.finish("文档已请求删除，但 API 同步后仍显示存在，请检查 AnythingLLM 侧状态。")

    await DeleteDocumentCommand.finish(f"已删除文档: [{target.record_id}] {target.title}")
