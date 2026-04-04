from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import aiosqlite
import httpx
import requests
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import BaseMessage, HumanMessage
from nonebot import logger

from plugins.GTBot.services.shared import fun as Fun
from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.services.chat.context import GroupChatContext


_DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024
_DB_INIT_LOCK = asyncio.Lock()
_TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_DESCRIPTION_TAG_RE = re.compile(r"<description>(.*?)</description>", re.DOTALL)
_VLM_IMAGE_PAYLOAD_CACHE_KEY = "vlm_image_prefetched_payload_cache"
_VLM_IMAGE_HASH_CACHE_KEY = "vlm_image_prefetched_hash_cache"
_MIGRATION_HINT = (
    "vlm_image_cache.sqlite3 缺少 title 列，请先运行 "
    "`python -m plugins.GTBot.tools.vlm_image.migrate_cache` 完成一次性迁移"
)


@dataclass(frozen=True)
class ImageAnalysisResult:
    """图片识别结果。

    Attributes:
        title: 图片标题。
        description: 图片描述。
    """

    title: str
    description: str
    image_size_bytes: int | None = None


@dataclass(frozen=True)
class CachedImageRecord:
    """缓存中的图片记录。"""

    image_hash: str
    title: str
    image_size_bytes: int | None = None


def _get_vlm_chat_completions_url(base_url: str) -> str:
    """规范化 VLM chat completions 接口地址。

    Args:
        base_url: 配置中的基础地址。

    Returns:
        可直接用于请求的 `chat/completions` URL。

    Raises:
        ValueError: 当 `base_url` 为空时抛出。
    """
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("base_url 不能为空")

    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _get_cache_db_path() -> Path:
    """返回识图缓存数据库路径。

    Returns:
        GTBot 数据目录下的 `vlm_image_cache.sqlite3` 路径。
    """
    data_dir = total_config.get_data_dir_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "vlm_image_cache.sqlite3"


def _get_request_image_caches(plugin_ctx: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    extra = getattr(plugin_ctx, "extra", None)
    if not isinstance(extra, dict):
        return {}, {}

    payload_cache = extra.get(_VLM_IMAGE_PAYLOAD_CACHE_KEY)
    if not isinstance(payload_cache, dict):
        payload_cache = {}
        extra[_VLM_IMAGE_PAYLOAD_CACHE_KEY] = payload_cache

    image_hash_cache = extra.get(_VLM_IMAGE_HASH_CACHE_KEY)
    if not isinstance(image_hash_cache, dict):
        image_hash_cache = {}
        extra[_VLM_IMAGE_HASH_CACHE_KEY] = image_hash_cache

    return payload_cache, image_hash_cache


async def _init_cache_db(db_path: Path) -> None:
    """初始化图片缓存数据库。

    Args:
        db_path: 缓存数据库文件路径。
    """
    async with _DB_INIT_LOCK:
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS image_cache (
                    image_hash TEXT PRIMARY KEY,
                    title TEXT,
                    image_size_bytes INTEGER,
                    description TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()


def _handle_cache_schema_error(exc: Exception) -> None:
    """将旧版缓存表结构错误转换为可操作的迁移提示。

    Args:
        exc: 原始异常。

    Raises:
        RuntimeError: 当判断为缺少 `title` 列时抛出带迁移提示的异常。
    """
    message = str(exc)
    if "no such column: title" in message or "no such column: image_size_bytes" in message:
        raise RuntimeError(_MIGRATION_HINT) from exc


async def _execute_cache_query_fetchone(
    db_path: Path,
    query: str,
    params: tuple[Any, ...],
) -> Sequence[Any] | None:
    """执行缓存查询并返回单行结果。

    Args:
        db_path: 缓存数据库文件路径。
        query: SQL 查询语句。
        params: SQL 参数。

    Returns:
        查询得到的单行元组；无结果时返回 `None`。
    """
    try:
        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute(query, params) as cur:
                row = await cur.fetchone()
    except aiosqlite.OperationalError as exc:
        _handle_cache_schema_error(exc)
        raise
    return row


async def _execute_cache_query_fetchall(
    db_path: Path,
    query: str,
    params: tuple[Any, ...],
) -> list[Sequence[Any]]:
    """执行缓存查询并返回所有结果。"""
    try:
        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
    except aiosqlite.OperationalError as exc:
        _handle_cache_schema_error(exc)
        raise
    return list(rows)


def _normalize_positive_int(value: Any) -> int | None:
    """将输入归一化为正整数。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _collect_image_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    pattern = r"(\[CQ:(?:\\.|[^\]])+\])"
    for part in re.split(pattern, str(text)):
        if not (part.startswith("[CQ:") and part.endswith("]")):
            continue
        try:
            cq_dict = Fun.parse_single_cq(part)
        except Exception:
            continue
        if str(cq_dict.get("CQ") or "").strip() != "image":
            continue
        image_name = str(cq_dict.get("file") or "").strip()
        if image_name and image_name not in names:
            names.append(image_name)
    return names


def _extend_image_names_from_message_value(target: list[str], message_value: Any) -> None:
    if isinstance(message_value, list):
        for segment in message_value:
            if not isinstance(segment, dict):
                continue
            if str(segment.get("type") or "").strip() != "image":
                continue
            data = segment.get("data")
            if not isinstance(data, dict):
                continue
            image_name = str(data.get("file") or "").strip()
            if image_name and image_name not in target:
                target.append(image_name)
        return

    if message_value is None or isinstance(message_value, (str, bytes, dict)):
        return

    try:
        iterator = iter(message_value)
    except TypeError:
        return

    for segment in iterator:
        if getattr(segment, "type", "") != "image":
            continue
        data = getattr(segment, "data", None)
        if not isinstance(data, dict):
            continue
        image_name = str(data.get("file") or "").strip()
        if image_name and image_name not in target:
            target.append(image_name)


def _collect_image_names_from_raw_messages(raw_messages: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for raw_message in raw_messages:
        if isinstance(raw_message, dict):
            _extend_image_names_from_message_value(names, raw_message.get("message"))
            for key in ("content", "raw_message"):
                value = raw_message.get(key)
                if not isinstance(value, str):
                    continue
                for image_name in _collect_image_names_from_text(value):
                    if image_name not in names:
                        names.append(image_name)
            continue

        _extend_image_names_from_message_value(names, getattr(raw_message, "message", None))
        for attr in ("content", "raw_message"):
            value = getattr(raw_message, attr, None)
            if not isinstance(value, str):
                continue
            for image_name in _collect_image_names_from_text(value):
                if image_name not in names:
                    names.append(image_name)
    return names


async def _get_cached_result(db_path: Path, image_hash: str) -> ImageAnalysisResult | None:
    """读取缓存中的图片识别结果。

    Args:
        db_path: 缓存数据库文件路径。
        image_hash: 图片内容哈希。

    Returns:
        当缓存存在且描述有效时返回结果；不存在时返回 `None`。

    Raises:
        RuntimeError: 当缓存仍是旧表结构、尚未执行迁移脚本时抛出。
    """
    await _init_cache_db(db_path)
    row = await _execute_cache_query_fetchone(
        db_path,
        "SELECT title, image_size_bytes, description FROM image_cache WHERE image_hash = ?",
        (str(image_hash),),
    )
    if not row:
        return None

    title = row[0] if len(row) > 0 else ""
    image_size_bytes = int(row[1]) if len(row) > 1 and row[1] is not None else None
    description = row[2] if len(row) > 2 else ""
    description_text = str(description or "").strip()
    if not description_text:
        return None

    return ImageAnalysisResult(
        title=str(title or "").strip(),
        description=description_text,
        image_size_bytes=image_size_bytes,
    )


async def _upsert_cached_result(db_path: Path, image_hash: str, result: ImageAnalysisResult) -> None:
    """写入或更新图片识别缓存。

    Args:
        db_path: 缓存数据库文件路径。
        image_hash: 图片内容哈希。
        result: 待写入的标题与描述。
    """
    await _init_cache_db(db_path)
    now = float(time.time())
    try:
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                INSERT INTO image_cache(image_hash, title, image_size_bytes, description, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    title=excluded.title,
                    image_size_bytes=excluded.image_size_bytes,
                    description=excluded.description,
                    updated_at=excluded.updated_at
                """,
                (
                    str(image_hash),
                    str(result.title),
                    int(result.image_size_bytes) if result.image_size_bytes is not None else None,
                    str(result.description),
                    now,
                    now,
                ),
            )
            await db.commit()
    except aiosqlite.OperationalError as exc:
        _handle_cache_schema_error(exc)
        raise


async def _find_cached_records_by_size(db_path: Path, image_size_bytes: int) -> list[CachedImageRecord]:
    """按图片大小查找已缓存且带标题的记录。"""
    await _init_cache_db(db_path)
    rows = await _execute_cache_query_fetchall(
        db_path,
        """
        SELECT image_hash, title, image_size_bytes
        FROM image_cache
        WHERE image_size_bytes = ? AND COALESCE(TRIM(title), '') <> ''
        """,
        (int(image_size_bytes),),
    )
    out: list[CachedImageRecord] = []
    for row in rows:
        image_hash = str(row[0] or "").strip() if len(row) > 0 else ""
        title = str(row[1] or "").strip() if len(row) > 1 else ""
        size = int(row[2]) if len(row) > 2 and row[2] is not None else None
        if image_hash and title:
            out.append(CachedImageRecord(image_hash=image_hash, title=title, image_size_bytes=size))
    return out


async def _read_image_bytes_from_path(image_path: Path, *, max_size_bytes: int) -> bytes:
    """从本地路径读取图片字节并校验大小。

    Args:
        image_path: 本地图片路径。
        max_size_bytes: 允许的最大图片大小。

    Returns:
        图片二进制内容。

    Raises:
        FileNotFoundError: 当图片文件不存在时抛出。
        ValueError: 当图片超过大小限制时抛出。
    """
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    stat = image_path.stat()
    if stat.st_size > int(max_size_bytes):
        raise ValueError(f"图片过大: {stat.st_size} bytes > {max_size_bytes} bytes")

    return image_path.read_bytes()


async def _download_image_bytes(url: str, *, max_size_bytes: int) -> bytes:
    """下载图片并限制最大体积。

    Args:
        url: 图片下载地址。
        max_size_bytes: 允许的最大图片大小。

    Returns:
        下载得到的图片二进制内容。

    Raises:
        ValueError: 当 URL 为空或下载内容超限时抛出。
        httpx.HTTPError: 当下载请求失败时抛出。
    """
    u = str(url or "").strip()
    if not u:
        raise ValueError("图片 URL 为空")

    buf = bytearray()
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        async with client.stream("GET", u) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > int(max_size_bytes):
                    raise ValueError(f"图片过大: > {max_size_bytes} bytes")

    return bytes(buf)


def _extract_onebot_image_data(resp: Any) -> dict[str, Any] | None:
    """从 OneBot `get_image` 响应中提取图片信息。

    Args:
        resp: OneBot API 原始响应。

    Returns:
        提取后的字典；若响应格式无法识别则返回 `None`。
    """
    if not isinstance(resp, dict):
        return None

    data = resp.get("data")
    if isinstance(data, dict):
        return data

    if any(k in resp for k in ("file", "url", "file_size", "file_name")):
        return resp

    return None


async def _call_onebot_get_image(bot: Any, image_name: str) -> dict[str, Any]:
    """调用 OneBot `get_image` 并解析返回。"""
    try:
        onebot_resp = await bot.call_api("get_image", file=str(image_name))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"调用 OneBot get_image 失败: {type(exc).__name__}: {exc!s}") from exc

    data = _extract_onebot_image_data(onebot_resp)
    if not isinstance(data, dict):
        if isinstance(onebot_resp, dict):
            keys = ",".join(sorted([str(k) for k in onebot_resp.keys()]))
            raise RuntimeError(f"OneBot get_image 返回格式异常：无法解析图片数据 keys=[{keys}]")
        raise RuntimeError("OneBot get_image 返回格式异常：响应不是 dict")
    return data


def _extract_image_size_from_cq_data(cq_data: dict[str, Any]) -> int | None:
    """从图片 CQ 参数中提取图片大小。"""
    for key in ("file_size", "fileSize", "size"):
        value = _normalize_positive_int(cq_data.get(key))
        if value is not None:
            return value
    return None


def _extract_image_size_from_onebot_data(data: dict[str, Any]) -> int | None:
    """从 OneBot 图片信息中提取图片大小。"""
    for key in ("file_size", "fileSize", "size"):
        value = _normalize_positive_int(data.get(key))
        if value is not None:
            return value

    local_file = data.get("file")
    if isinstance(local_file, str) and local_file:
        try:
            path = Path(local_file)
            if path.exists():
                return int(path.stat().st_size)
        except Exception:
            return None
    return None


def _extract_local_image_path_from_cq_data(cq_data: dict[str, Any]) -> Path | None:
    """从图片 CQ 参数中提取可直接访问的本地图片路径。

    Args:
        cq_data: 已解析的图片 CQ 参数字典。

    Returns:
        当 `file` 字段指向本地且存在的图片文件时返回对应路径，否则返回 `None`。
    """
    image_name = str(cq_data.get("file") or "").strip()
    if not image_name:
        return None

    try:
        image_path = Path(image_name)
    except (TypeError, ValueError):
        return None

    if image_path.exists() and image_path.is_file():
        return image_path
    return None


async def _resolve_image_bytes_from_onebot_data(data: dict[str, Any], *, max_size_bytes: int) -> tuple[bytes, Path | None]:
    """根据 OneBot 图片信息读取图片字节。"""
    local_file = data.get("file")
    url = data.get("url")
    image_path = Path(str(local_file)) if isinstance(local_file, str) and local_file else None

    if image_path is not None and image_path.exists():
        image_bytes = await _read_image_bytes_from_path(image_path, max_size_bytes=int(max_size_bytes))
        return image_bytes, image_path
    if isinstance(url, str) and url.strip():
        image_bytes = await _download_image_bytes(url.strip(), max_size_bytes=int(max_size_bytes))
        return image_bytes, image_path
    raise FileNotFoundError(f"无法定位图片文件: {local_file or url or ''}")


def _guess_mime_type(image_name: str, image_path: str | None) -> str:
    """推断图片 MIME 类型。

    Args:
        image_name: 图片文件名。
        image_path: 本地图片路径。

    Returns:
        推断出的图片 MIME 类型；无法识别时回退为 `image/png`。
    """
    for candidate in (image_path, image_name):
        if not candidate:
            continue
        mime, _ = mimetypes.guess_type(candidate)
        if mime and mime.startswith("image/"):
            return mime
    return "image/png"


def _extract_openai_content(data: Any) -> str:
    """提取 OpenAI 兼容响应中的文本内容。

    Args:
        data: VLM 接口返回的 JSON 对象。

    Returns:
        抽取后的文本内容。

    Raises:
        RuntimeError: 当响应结构不符合预期时抛出。
    """
    if not isinstance(data, dict):
        raise RuntimeError("VLM 响应不是 JSON 对象")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("VLM 响应缺少 choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("VLM 响应 choices[0] 不是对象")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("VLM 响应缺少 message")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
        return "\n".join([p for p in parts if p]).strip()

    raise RuntimeError("VLM 响应 message.content 格式不受支持")


def _merge_custom_dict(*, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """合并两个字典，后者覆盖前者。

    Args:
        base: 基础字典。
        override: 覆盖字典。

    Returns:
        合并后的新字典。
    """
    out = dict(base)
    out.update(dict(override))
    return out


def _filter_reserved(*, data: dict[str, Any], reserved: set[str], allow_override: bool) -> dict[str, Any]:
    """过滤保留字段，避免调用方覆盖核心请求参数。

    Args:
        data: 待过滤字典。
        reserved: 不允许覆盖的字段集合。
        allow_override: 是否允许覆盖保留字段。

    Returns:
        过滤后的字典。
    """
    if allow_override:
        return dict(data)
    return {k: v for k, v in dict(data).items() if str(k) not in reserved}


def _build_vlm_prompt(question: str | None) -> str:
    """构造要求返回 XML 的 VLM 提示词。

    Args:
        question: 可选的补充问题。

    Returns:
        要求模型只返回固定 XML 标签的提示词。
    """
    cfg_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".config")
    cfg = getattr(cfg_mod, "get_vlm_image_plugin_config")()
    q = str(question or "").strip()
    if q:
        user_requirement = f"在描述图片时，还要回答这个补充要求：{q}"
    else:
        user_requirement = "先概括图片主题，再客观描述图片主要内容。"

    raw_title_recommended_chars = getattr(cfg, "title_recommended_chars", None)
    title_max_chars = int(getattr(cfg, "title_max_chars", 16) or 16)
    raw_description_recommended_chars = getattr(cfg, "description_recommended_chars", None)
    description_max_chars = int(getattr(cfg, "description_max_chars", 120) or 120)
    title_recommended_chars = (
        int(raw_title_recommended_chars) if raw_title_recommended_chars is not None else None
    )
    description_recommended_chars = (
        int(raw_description_recommended_chars) if raw_description_recommended_chars is not None else None
    )

    title_requirement = f"title 简短准确，尽量不超过 {title_max_chars} 个字"
    if title_recommended_chars is not None:
        title_requirement = (
            f"title 简短准确，推荐 {title_recommended_chars} 个字左右，尽量不超过 {title_max_chars} 个字"
        )

    description_requirement = f"description 客观简洁，尽量不超过 {description_max_chars} 个字"
    if description_recommended_chars is not None:
        description_requirement = (
            "description 客观简洁，"
            f"推荐 {description_recommended_chars} 个字左右，尽量不超过 {description_max_chars} 个字"
        )
    description_requirement += (
        "；信息密度尽量高，若能识别具体角色、人物、作品、地点、物品或品牌，"
        "优先直接写出其名称，不要仅用外貌、服饰、颜色或泛化称呼代指"
    )

    return (
        "请分析这张图片，并先完整、客观地描述图片内容，再根据描述提炼一个简短准确的标题。\n"
        "严格只返回以下 XML，不要输出任何额外文字、Markdown、代码块或解释。\n"
        "<description>图片描述</description>\n"
        "<title>图片标题</title>\n"
        f"{user_requirement}\n"
        f"要求：{description_requirement}；"
        f"{title_requirement}；"
        "两个标签都必须非空。"
    )


def _parse_vlm_xml_result(raw_text: str) -> ImageAnalysisResult:
    """解析 VLM 返回的固定 XML 结果。

    Args:
        raw_text: VLM 返回的原始文本。

    Returns:
        解析后的标题与描述。

    Raises:
        RuntimeError: 当 XML 缺标签、标签重复、内容为空或含额外文本时抛出。
    """
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("VLM 返回内容为空")

    title_matches = _TITLE_TAG_RE.findall(text)
    if len(title_matches) != 1:
        raise RuntimeError("VLM 返回的 XML 格式异常：title 标签缺失或重复")

    description_matches = _DESCRIPTION_TAG_RE.findall(text)
    if len(description_matches) != 1:
        raise RuntimeError("VLM 返回的 XML 格式异常：description 标签缺失或重复")

    leftover = _TITLE_TAG_RE.sub("", text, count=1)
    leftover = _DESCRIPTION_TAG_RE.sub("", leftover, count=1)
    if leftover.strip():
        raise RuntimeError("VLM 返回的 XML 格式异常：包含额外文本")

    title = str(title_matches[0]).strip()
    description = str(description_matches[0]).strip()
    if not title:
        raise RuntimeError("VLM 返回的 XML 格式异常：title 为空")
    if not description:
        raise RuntimeError("VLM 返回的 XML 格式异常：description 为空")

    return ImageAnalysisResult(title=title, description=description)


def _format_analysis_result(result: ImageAnalysisResult) -> str:
    """将识图结果格式化为工具对外返回文本。

    Args:
        result: 已解析的标题与描述。

    Returns:
        面向上层调用的组合文本。
    """
    return f"标题：{result.title}\n描述：{result.description}"


def _copy_message_with_text(message: Any, text: str) -> Any:
    """复制消息对象并替换文本内容。"""
    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": text})
    return HumanMessage(content=text)


async def _inject_title_into_image_cq(
    *,
    cq_data: dict[str, Any],
    bot: Any,
    db_path: Path,
    max_size_bytes: int,
    image_payload_cache: dict[str, dict[str, Any]],
    image_hash_cache: dict[str, str | None],
) -> str | None:
    """若缓存中存在同图标题，则将其注入图片 CQ。"""
    image_name = str(cq_data.get("file") or "").strip()
    if not image_name:
        return None

    local_image_path = _extract_local_image_path_from_cq_data(cq_data)
    image_size = _extract_image_size_from_cq_data(cq_data)
    payload: dict[str, Any] | None = None
    if image_size is None:
        if local_image_path is not None:
            try:
                image_size = int(local_image_path.stat().st_size)
            except OSError:
                image_size = None
        else:
            logger.warning("vlm_image middleware: CQ:image 缺少 file_size，回退调用 get_image 获取大小: file=%s", image_name)
            payload = image_payload_cache.get(image_name)
            if payload is None:
                payload = await _call_onebot_get_image(bot, image_name)
                image_payload_cache[image_name] = payload
            image_size = _extract_image_size_from_onebot_data(payload)
    if image_size is None:
        return None

    candidates = await _find_cached_records_by_size(db_path, image_size)
    if not candidates:
        return None

    image_hash = image_hash_cache.get(image_name)
    if image_hash is None:
        if local_image_path is not None:
            image_bytes = await _read_image_bytes_from_path(local_image_path, max_size_bytes=max_size_bytes)
        else:
            if payload is None:
                payload = image_payload_cache.get(image_name)
            if payload is None:
                payload = await _call_onebot_get_image(bot, image_name)
                image_payload_cache[image_name] = payload
            image_bytes, _ = await _resolve_image_bytes_from_onebot_data(payload, max_size_bytes=max_size_bytes)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_hash_cache[image_name] = image_hash

    for candidate in candidates:
        if candidate.image_hash == image_hash:
            title = str(candidate.title).strip()
            if not title:
                return None
            if not cq_data.get("title"):
                cq_data["title"] = title
            if not cq_data.get("file_size"):
                cq_data["file_size"] = str(image_size)
            return Fun.generate_cq_string("image", cq_data)
    return None


async def _inject_titles_into_text(
    *,
    text: str,
    bot: Any,
    db_path: Path,
    max_size_bytes: int,
    image_payload_cache: dict[str, dict[str, Any]],
    image_hash_cache: dict[str, str | None],
) -> str:
    """扫描文本中的图片 CQ，并按缓存标题补充 `title` 参数。"""
    pattern = r"(\[CQ:(?:\\.|[^\]])+\])"
    parts = re.split(pattern, str(text))
    out: list[str] = []
    changed = False

    for part in parts:
        if not part:
            continue
        if not (part.startswith("[CQ:") and part.endswith("]")):
            out.append(part)
            continue

        try:
            cq_dict = Fun.parse_single_cq(part)
        except Exception:
            out.append(part)
            continue

        cq_type = str(cq_dict.get("CQ") or "").strip()
        if cq_type != "image":
            out.append(part)
            continue

        cq_data = dict(cq_dict)
        cq_data.pop("CQ", None)
        replaced = await _inject_title_into_image_cq(
            cq_data=cq_data,
            bot=bot,
            db_path=db_path,
            max_size_bytes=max_size_bytes,
            image_payload_cache=image_payload_cache,
            image_hash_cache=image_hash_cache,
        )
        if replaced is not None:
            out.append(replaced)
            changed = True
        else:
            out.append(part)

    return "".join(out) if changed else str(text)


async def prewarm_vlm_image_cq_titles(plugin_ctx: Any) -> None:
    runtime_context = getattr(plugin_ctx, "runtime_context", None)
    bot = getattr(runtime_context, "bot", None) if runtime_context is not None else None
    if bot is None:
        return

    image_names = _collect_image_names_from_raw_messages(getattr(plugin_ctx, "raw_messages", []))
    if not image_names:
        return

    cfg_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".config")
    cfg = getattr(cfg_mod, "get_vlm_image_plugin_config")()
    max_size_bytes = int(getattr(cfg, "max_image_size_bytes", _DEFAULT_MAX_SIZE_BYTES) or _DEFAULT_MAX_SIZE_BYTES)
    db_path = _get_cache_db_path()
    image_payload_cache, image_hash_cache = _get_request_image_caches(plugin_ctx)

    for image_name in image_names:
        try:
            payload = image_payload_cache.get(image_name)
            if payload is None:
                payload = await _call_onebot_get_image(bot, image_name)
                image_payload_cache[image_name] = payload

            image_size = _extract_image_size_from_onebot_data(payload)
            if image_size is None:
                continue

            candidates = await _find_cached_records_by_size(db_path, image_size)
            if not candidates or image_hash_cache.get(image_name):
                continue

            image_bytes, _ = await _resolve_image_bytes_from_onebot_data(
                payload,
                max_size_bytes=max_size_bytes,
            )
            image_hash_cache[image_name] = hashlib.sha256(image_bytes).hexdigest()
        except Exception:
            logger.debug("vlm_image prewarm skipped for %s", image_name, exc_info=True)


async def inject_vlm_image_cq_titles(plugin_ctx: Any, messages: list[BaseMessage]) -> list[BaseMessage]:
    """在 Agent 启动前为最终消息中的图片 CQ 注入缓存标题。

    Args:
        plugin_ctx: 当前请求的插件上下文，用于读取运行时 `bot` 与插件配置。
        messages: 已经转换完成、即将发送给 LLM 的 LangChain 消息列表。

    Returns:
        list[BaseMessage]: 注入标题后的消息列表；若发生异常则返回原消息副本。
    """

    updated_messages = list(messages)

    try:
        runtime_context = getattr(plugin_ctx, "runtime_context", None)
        bot = getattr(runtime_context, "bot", None) if runtime_context is not None else None
        if bot is None:
            return updated_messages

        cfg_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".config")
        cfg = getattr(cfg_mod, "get_vlm_image_plugin_config")()
        max_size_bytes = int(
            getattr(cfg, "max_image_size_bytes", _DEFAULT_MAX_SIZE_BYTES) or _DEFAULT_MAX_SIZE_BYTES
        )
        db_path = _get_cache_db_path()
        image_payload_cache, image_hash_cache = _get_request_image_caches(plugin_ctx)

        for idx, message in enumerate(updated_messages):
            if not isinstance(message, HumanMessage):
                continue

            content = getattr(message, "content", None)
            if not isinstance(content, str) or "[CQ:image" not in content:
                continue

            new_content = await _inject_titles_into_text(
                text=content,
                bot=bot,
                db_path=db_path,
                max_size_bytes=max_size_bytes,
                image_payload_cache=image_payload_cache,
                image_hash_cache=image_hash_cache,
            )
            if new_content != content:
                updated_messages[idx] = _copy_message_with_text(message, new_content)

        return updated_messages
    except Exception:
        logger.warning("vlm_image injector: 注入图片标题失败", exc_info=True)
        return updated_messages


async def _call_vlm_api(
    *,
    prompt: str,
    image_data_url: str,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """调用 VLM 接口并返回原始文本结果。

    Args:
        prompt: 发送给 VLM 的提示词。
        image_data_url: 图片 data URL。
        extra_body: 附加请求体字段。
        extra_headers: 附加请求头字段。

    Returns:
        VLM 返回的原始文本。

    Raises:
        ValueError: 当配置缺失时抛出。
        requests.RequestException: 当 HTTP 请求失败时抛出。
        RuntimeError: 当响应结构不符合预期时抛出。
    """
    cfg_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".config")
    cfg = getattr(cfg_mod, "get_vlm_image_plugin_config")()
    base_url = str(getattr(cfg, "base_url", "") or "").strip()
    api_key = str(getattr(cfg, "api_key", "") or "").strip()
    model = str(getattr(cfg, "model", "") or "").strip()
    allow_override_reserved = bool(getattr(cfg, "allow_override_reserved", False))

    if not base_url:
        raise ValueError("VLMImage 配置缺少 base_url")
    if not model:
        raise ValueError("VLMImage 配置缺少 model")

    url = _get_vlm_chat_completions_url(base_url)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    default_extra_headers = getattr(cfg, "extra_headers", None)
    if isinstance(default_extra_headers, dict):
        for k, v in default_extra_headers.items():
            if isinstance(k, str) and isinstance(v, str):
                headers[k] = v
    call_extra_headers = extra_headers or {}
    for k, v in dict(call_extra_headers).items():
        if isinstance(k, str) and isinstance(v, str):
            headers[k] = v

    reserved_keys = {"model", "messages"}

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(prompt)},
                    {"type": "image_url", "image_url": {"url": str(image_data_url)}},
                ],
            }
        ],
        "max_tokens": int(getattr(cfg, "max_tokens", 512) or 512),
    }

    default_extra_body = getattr(cfg, "extra_body", None)
    if isinstance(default_extra_body, dict):
        payload = _merge_custom_dict(
            base=payload,
            override=_filter_reserved(
                data=default_extra_body,
                reserved=reserved_keys,
                allow_override=allow_override_reserved,
            ),
        )
    if extra_body is not None:
        payload = _merge_custom_dict(
            base=payload,
            override=_filter_reserved(
                data=extra_body,
                reserved=reserved_keys,
                allow_override=allow_override_reserved,
            ),
        )

    timeout_sec = float(getattr(cfg, "timeout_sec", 60.0) or 60.0)

    def _do_request() -> dict[str, Any]:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("VLM 响应 JSON 解析失败")
        return data

    data = await asyncio.to_thread(_do_request)
    return _extract_openai_content(data)


@tool("vlm_describe_image")
async def vlm_describe_image(
    image_name: str,
    runtime: ToolRuntime[GroupChatContext],
    question: str | None = None,
    use_cache: bool = True,
    max_size_bytes: int | None = None,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """调用 VLM 识别图片，并返回标题与描述的组合文本。

    Args:
        image_name: 图片名，通常来自 CQ 码的 `file` 字段。
        runtime: LangChain ToolRuntime，内部会使用 `runtime.context.bot`
            调用 OneBot `get_image` 获取图片路径或 URL。
        question: 可选的补充问题。传入后本次调用不直接复用普通描述缓存，
            但仍要求模型返回 XML 格式的标题与描述。
        use_cache: 是否优先使用缓存。仅当未传入 `question` 时才会直接命中缓存。
        max_size_bytes: 本次调用允许的最大图片大小，会与插件配置中的上限取更小值。
        extra_body: 透传给 VLM 接口的附加请求体字段。
        extra_headers: 透传给 VLM 接口的附加请求头字段。

    Returns:
        组合文本，格式固定为 `标题：...\n描述：...`。

    Raises:
        ValueError: 当参数非法、图片过大或 VLM 配置缺失时抛出。
        RuntimeError: 当 OneBot/VLM 调用失败或 XML 解析失败时抛出。
        FileNotFoundError: 当无法定位到图片文件且没有可用 URL 时抛出。
    """
    name = str(image_name or "").strip()
    if not name:
        raise ValueError("image_name 不能为空")

    cfg_mod = importlib.import_module(__name__.rsplit(".", 1)[0] + ".config")
    cfg = getattr(cfg_mod, "get_vlm_image_plugin_config")()
    cfg_max_size = getattr(cfg, "max_image_size_bytes", _DEFAULT_MAX_SIZE_BYTES)

    effective_max_size = cfg_max_size
    if max_size_bytes is not None:
        if int(max_size_bytes) <= 0:
            raise ValueError("max_size_bytes 必须为正整数")
        effective_max_size = min(int(effective_max_size), int(max_size_bytes))

    if int(effective_max_size) <= 0:
        raise ValueError("max_image_size_bytes 配置必须为正整数")

    q = (question or "").strip() or None
    should_use_cache = bool(use_cache) and q is None

    data = await _call_onebot_get_image(runtime.context.bot, name)
    image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
        data,
        max_size_bytes=int(effective_max_size),
    )
    image_size_bytes = len(image_bytes)

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    db_path = _get_cache_db_path()
    if should_use_cache:
        cached = await _get_cached_result(db_path, image_hash)
        if cached:
            if cached.title:
                return _format_analysis_result(cached)
            logger.info("vlm_image cache hit without title, regenerate and backfill: image_hash=%s", image_hash)

    mime = _guess_mime_type(name, str(image_path) if image_path is not None else None)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{mime};base64,{b64}"

    prompt = _build_vlm_prompt(q)

    try:
        raw_result = await _call_vlm_api(
            prompt=prompt,
            image_data_url=image_data_url,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLM 调用失败: {type(exc).__name__}: {exc!s}")
        raise

    parsed_result = _parse_vlm_xml_result(raw_result)
    result = ImageAnalysisResult(
        title=parsed_result.title,
        description=parsed_result.description,
        image_size_bytes=image_size_bytes,
    )

    if should_use_cache:
        try:
            await _upsert_cached_result(db_path, image_hash, result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"写入图片缓存失败: {type(exc).__name__}: {exc!s}")

    return _format_analysis_result(result)
