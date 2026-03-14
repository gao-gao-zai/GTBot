from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import requests
from langchain.tools import ToolRuntime, tool
from nonebot import logger

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.GroupChatContext import GroupChatContext

import importlib


_DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024
_DB_INIT_LOCK = asyncio.Lock()


def _get_vlm_chat_completions_url(base_url: str) -> str:
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
    data_dir = total_config.get_data_dir_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "vlm_image_cache.sqlite3"


async def _init_cache_db(db_path: Path) -> None:
    async with _DB_INIT_LOCK:
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS image_cache (
                    image_hash TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()


async def _get_cached_description(db_path: Path, image_hash: str) -> str | None:
    await _init_cache_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute(
            "SELECT description FROM image_cache WHERE image_hash = ?",
            (str(image_hash),),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            value = row[0]
            return value if isinstance(value, str) and value else None


async def _upsert_cached_description(db_path: Path, image_hash: str, description: str) -> None:
    await _init_cache_db(db_path)
    now = float(time.time())
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            INSERT INTO image_cache(image_hash, description, created_at, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(image_hash) DO UPDATE SET
                description=excluded.description,
                updated_at=excluded.updated_at
            """,
            (str(image_hash), str(description), now, now),
        )
        await db.commit()


async def _read_image_bytes_from_path(image_path: Path, *, max_size_bytes: int) -> bytes:
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    stat = image_path.stat()
    if stat.st_size > int(max_size_bytes):
        raise ValueError(f"图片过大: {stat.st_size} bytes > {max_size_bytes} bytes")

    return image_path.read_bytes()


async def _download_image_bytes(url: str, *, max_size_bytes: int) -> bytes:
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
    if not isinstance(resp, dict):
        return None

    data = resp.get("data")
    if isinstance(data, dict):
        return data

    if any(k in resp for k in ("file", "url", "file_size", "file_name")):
        return resp

    return None


def _guess_mime_type(image_name: str, image_path: str | None) -> str:
    for candidate in (image_path, image_name):
        if not candidate:
            continue
        mime, _ = mimetypes.guess_type(candidate)
        if mime and mime.startswith("image/"):
            return mime
    return "image/png"


def _extract_openai_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise RuntimeError("VLM 响应不是 JSON 对象")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("VLM 响应缺少 choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("VLM 响应 choices[0] 非对象")

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

    raise RuntimeError("VLM 响应 message.content 格式不支持")


def _merge_custom_dict(*, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(dict(override))
    return out


def _filter_reserved(*, data: dict[str, Any], reserved: set[str], allow_override: bool) -> dict[str, Any]:
    if allow_override:
        return dict(data)
    return {k: v for k, v in dict(data).items() if str(k) not in reserved}


async def _call_vlm_api(
    *,
    prompt: str,
    image_data_url: str,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
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
    """调用 VLM 识别图片并返回描述（支持 SQLite 缓存）。

    Args:
        image_name: 图片名（通常来自 CQ 码的 `file` 字段）。
        question: 可选的提问；传入后强制不使用缓存。
        use_cache: 是否使用缓存（默认 True）。
        max_size_bytes: 允许的最大图片大小（字节）。

    Returns:
        图片描述文本。

    Raises:
        ValueError: 当参数非法、图片过大或 VLM 配置缺失时抛出。
        RuntimeError: 当调用 OneBot/VLM 失败或响应格式异常时抛出。
        FileNotFoundError: 当无法获取到本地图片文件且无可用 URL 时抛出。
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

    try:
        onebot_resp = await runtime.context.bot.call_api("get_image", file=name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"调用 OneBot get_image 失败: {type(exc).__name__}: {exc!s}") from exc

    data = _extract_onebot_image_data(onebot_resp)
    if not isinstance(data, dict):
        if isinstance(onebot_resp, dict):
            keys = ",".join(sorted([str(k) for k in onebot_resp.keys()]))
            raise RuntimeError(f"OneBot get_image 返回格式异常：无法解析图片数据 keys=[{keys}]")
        raise RuntimeError("OneBot get_image 返回格式异常：响应不是 dict")

    local_file = data.get("file")
    url = data.get("url")

    image_path = Path(str(local_file)) if isinstance(local_file, str) and local_file else None
    image_bytes: bytes

    if image_path is not None and image_path.exists():
        image_bytes = await _read_image_bytes_from_path(image_path, max_size_bytes=int(effective_max_size))
    elif isinstance(url, str) and url.strip():
        image_bytes = await _download_image_bytes(url.strip(), max_size_bytes=int(effective_max_size))
    else:
        raise FileNotFoundError(f"无法定位图片文件: {name}")

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    db_path = _get_cache_db_path()
    if should_use_cache:
        cached = await _get_cached_description(db_path, image_hash)
        if cached:
            return cached

    mime = _guess_mime_type(name, str(image_path) if image_path is not None else None)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{mime};base64,{b64}"

    prompt = q or "请描述这张图片的主要内容，尽量客观简洁。"

    try:
        description = await _call_vlm_api(
            prompt=prompt,
            image_data_url=image_data_url,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"VLM 调用失败: {type(exc).__name__}: {exc!s}")
        raise

    if not description:
        raise RuntimeError("VLM 返回内容为空")

    if should_use_cache:
        try:
            await _upsert_cached_description(db_path, image_hash, description)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"写入图片缓存失败: {type(exc).__name__}: {exc!s}")

    return description
