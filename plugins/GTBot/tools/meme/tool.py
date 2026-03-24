from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from nonebot import logger

from plugins.GTBot import Fun
from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.GroupChatContext import GroupChatContext
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context


_DB_INIT_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class MemeRecord:
    """表情包记录。"""

    title: str
    image_hash: str
    stored_path: str
    file_size_bytes: int
    created_at: float
    updated_at: float
    last_used_at: float | None = None


def _get_meme_db_path() -> Path:
    """返回表情包索引数据库路径。"""

    data_dir = total_config.get_data_dir_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "meme_store.sqlite3"


def _get_meme_files_dir() -> Path:
    """返回表情包文件目录。"""

    data_dir = total_config.get_data_dir_path()
    data_dir.mkdir(parents=True, exist_ok=True)
    meme_dir = data_dir / "memes"
    meme_dir.mkdir(parents=True, exist_ok=True)
    return meme_dir


async def _init_meme_db(db_path: Path) -> None:
    """初始化表情包索引数据库。"""

    async with _DB_INIT_LOCK:
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS meme_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL UNIQUE,
                    image_hash TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used_at REAL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_meme_store_image_hash ON meme_store(image_hash)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_meme_store_last_used_at ON meme_store(last_used_at)"
            )
            await db.commit()


def _copy_message_with_text(message: Any, text: str) -> Any:
    """复制消息对象并替换文本内容。"""

    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": text})
    return HumanMessage(content=text)


def _normalize_title(raw_title: str, *, max_title_chars: int) -> str:
    """清洗并校验表情包标题。"""

    title = str(raw_title or "").strip()
    if not title:
        raise ValueError("title 不能为空")
    if "\n" in title or "\r" in title:
        raise ValueError("title 不能包含换行")
    if "<" in title or ">" in title:
        raise ValueError("title 不能包含尖括号")
    if len(title) > int(max_title_chars):
        raise ValueError(f"title 不能超过 {int(max_title_chars)} 个字符")
    return title


def _guess_file_extension(*, image_name: str, local_path: str | None) -> str:
    """推断图片文件扩展名。"""

    for candidate in (local_path, image_name):
        if not candidate:
            continue
        mime, _ = mimetypes.guess_type(candidate)
        if mime and mime.startswith("image/"):
            ext = mimetypes.guess_extension(mime) or ""
            if ext:
                return ext
        suffix = Path(str(candidate)).suffix
        if suffix:
            return suffix
    return ".png"


async def _read_image_source(
    *,
    bot: Any,
    image_name: str,
    max_size_bytes: int,
) -> tuple[bytes, str | None, int]:
    """通过 OneBot 读取图片字节与大小。"""

    from plugins.GTBot.tools.vlm_image.tool import (
        _call_onebot_get_image,
        _extract_image_size_from_onebot_data,
        _resolve_image_bytes_from_onebot_data,
    )

    payload = await _call_onebot_get_image(bot, image_name)
    image_bytes, image_path = await _resolve_image_bytes_from_onebot_data(
        payload,
        max_size_bytes=int(max_size_bytes),
    )
    image_size_bytes = _extract_image_size_from_onebot_data(payload) or len(image_bytes)
    return image_bytes, (str(image_path) if image_path is not None else None), int(image_size_bytes)


async def _fetch_meme_count(db_path: Path) -> int:
    """统计当前表情包数量。"""

    await _init_meme_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute("SELECT COUNT(1) FROM meme_store") as cur:
            row = await cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


async def _get_meme_by_title(db_path: Path, title: str) -> MemeRecord | None:
    """按标题查询表情包记录。"""

    await _init_meme_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute(
            """
            SELECT title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at
            FROM meme_store
            WHERE title = ?
            """,
            (str(title),),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return MemeRecord(
        title=str(row[0]),
        image_hash=str(row[1]),
        stored_path=str(row[2]),
        file_size_bytes=int(row[3]),
        created_at=float(row[4]),
        updated_at=float(row[5]),
        last_used_at=float(row[6]) if row[6] is not None else None,
    )


async def _get_meme_by_hash(db_path: Path, image_hash: str) -> MemeRecord | None:
    """按图片哈希查询一条已有记录。"""

    await _init_meme_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute(
            """
            SELECT title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at
            FROM meme_store
            WHERE image_hash = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (str(image_hash),),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return MemeRecord(
        title=str(row[0]),
        image_hash=str(row[1]),
        stored_path=str(row[2]),
        file_size_bytes=int(row[3]),
        created_at=float(row[4]),
        updated_at=float(row[5]),
        last_used_at=float(row[6]) if row[6] is not None else None,
    )


async def _insert_meme_record(
    *,
    db_path: Path,
    title: str,
    image_hash: str,
    stored_path: str,
    file_size_bytes: int,
) -> None:
    """插入新的表情包记录。"""

    await _init_meme_db(db_path)
    now = float(time.time())
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            INSERT INTO meme_store(title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at)
            VALUES(?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                str(title),
                str(image_hash),
                str(stored_path),
                int(file_size_bytes),
                now,
                now,
            ),
        )
        await db.commit()


async def _touch_meme_last_used(db_path: Path, title: str) -> None:
    """更新表情包最近使用时间。"""

    await _init_meme_db(db_path)
    now = float(time.time())
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            "UPDATE meme_store SET last_used_at = ?, updated_at = ? WHERE title = ?",
            (now, now, str(title)),
        )
        await db.commit()


async def _list_memes_for_prompt(db_path: Path, limit: int) -> list[str]:
    """按展示优先级列出可注入到提示词中的表情包标题。"""

    await _init_meme_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        async with db.execute(
            """
            SELECT title
            FROM meme_store
            ORDER BY COALESCE(last_used_at, 0) DESC, created_at DESC, title ASC
            LIMIT ?
            """,
            (int(limit),),
        ) as cur:
            rows = await cur.fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


async def _store_image_file_if_needed(
    *,
    image_hash: str,
    image_bytes: bytes,
    file_ext: str,
) -> Path:
    """将图片文件按哈希落盘，已存在则直接复用。"""

    meme_dir = _get_meme_files_dir()
    suffix = file_ext if str(file_ext).startswith(".") else f".{file_ext}"
    target_path = meme_dir / f"{image_hash}{suffix}"
    if target_path.exists():
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    def _write() -> None:
        tmp_path.write_bytes(image_bytes)
        tmp_path.replace(target_path)

    await asyncio.to_thread(_write)
    return target_path


async def build_meme_context_prompt(*, limit: int | None = None) -> str:
    """构造注入给 LLM 的表情包上下文文本。"""

    from plugins.GTBot.tools.meme.config import get_meme_plugin_config

    cfg = get_meme_plugin_config()
    effective_limit = int(limit or cfg.max_injected_memes)
    titles = await _list_memes_for_prompt(_get_meme_db_path(), effective_limit)
    if not titles:
        return (
            "表情包能力说明：\n"
            "1. 你可以调用 save_meme(image_name, title) 把当前会话中的某张图片收藏为表情包。\n"
            "2. 你应当常用、善用表情包；只要当前语气和场景合适，就应主动考虑使用表情包增强表达，而不是只有用户要求时才用。\n"
            "3. 给表情包起名时，要先看图片本身表达的情绪、动作、语气和梗点，不要只照抄用户提到的人名、角色名或作品名。\n"
            "4. 优先使用“角色或主体 + 情绪/动作”或直接使用“情绪/动作/梗点”命名，例如“白子歪头”“猫猫震惊”，不要只命名为“白子”或“猫猫”；标题不必刻意很短，只要清晰好找，十几个字也可以。\n"
            "5. 发送已收藏表情包时，可以输出 <meme>表情包标题</meme>。\n"
            "6. 当前没有可用表情包。"
        )

    lines = [
        "表情包能力说明：",
        "1. 你可以调用 save_meme(image_name, title) 把当前会话中的某张图片收藏为表情包。",
        "2. 你应当常用、善用表情包；只要当前语气和场景合适，就应主动考虑使用表情包增强表达，而不是只有用户要求时才用。",
        "3. 给表情包起名时，要先看图片本身表达的情绪、动作、语气和梗点，不要只照抄用户提到的人名、角色名或作品名。",
        "4. 优先使用“角色或主体 + 情绪/动作”或直接使用“情绪/动作/梗点”命名，例如“白子歪头”“猫猫震惊”，不要只命名为“白子”或“猫猫”；标题不必刻意很短，只要清晰好找，十几个字也可以。",
        "5. 发送已收藏表情包时，可以输出 <meme>表情包标题</meme>。",
        "6. 当前可用表情包标题如下：",
    ]
    lines.extend([f"- {title}" for title in titles])
    return "\n".join(lines)


async def resolve_meme_title_to_cq(title: str) -> str | None:
    """将 `<meme>` 标题解析为可发送的图片 CQ 码。"""

    normalized_title = str(title or "").strip()
    if not normalized_title:
        return None

    record = await _get_meme_by_title(_get_meme_db_path(), normalized_title)
    if record is None:
        return None

    path = Path(record.stored_path)
    if not path.exists():
        logger.warning("meme file missing: title=%s path=%s", normalized_title, path)
        return None

    await _touch_meme_last_used(_get_meme_db_path(), normalized_title)
    return Fun.generate_cq_string(
        "image",
        {
            "file": str(path),
            "file_size": int(record.file_size_bytes),
        },
    )


@tool("save_meme")
async def save_meme(
    image_name: str,
    title: str,
    runtime: ToolRuntime[GroupChatContext],
) -> str:
    """收藏一张图片为表情包。

    Args:
        image_name: 待收藏图片的 OneBot 图片标识，通常来自 CQ:image 的 `file` 字段。
        title: 表情包标题，后续可通过 `<meme>标题</meme>` 发送。
        runtime: LangChain ToolRuntime，内部会通过 `runtime.context.bot` 读取图片。

    Returns:
        收藏结果说明文本。

    Raises:
        ValueError: 当标题非法、超过大小限制或数量上限时抛出。
        RuntimeError: 当图片读取、落盘或数据库写入失败时抛出。
    """

    from plugins.GTBot.tools.meme.config import get_meme_plugin_config

    cfg = get_meme_plugin_config()
    normalized_title = _normalize_title(title, max_title_chars=cfg.max_title_chars)
    normalized_image_name = str(image_name or "").strip()
    if not normalized_image_name:
        raise ValueError("image_name 不能为空")

    bot = getattr(runtime.context, "bot", None)
    if bot is None:
        raise RuntimeError("当前上下文缺少 bot，无法读取图片")

    db_path = _get_meme_db_path()
    existing_by_title = await _get_meme_by_title(db_path, normalized_title)

    image_bytes, local_path, image_size_bytes = await _read_image_source(
        bot=bot,
        image_name=normalized_image_name,
        max_size_bytes=int(cfg.max_meme_size_bytes),
    )
    if int(image_size_bytes) > int(cfg.max_meme_size_bytes):
        raise ValueError(f"图片大小超过限制：{image_size_bytes} > {cfg.max_meme_size_bytes}")

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    if existing_by_title is not None:
        if existing_by_title.image_hash == image_hash:
            return f"表情包已存在：{normalized_title}"
        raise ValueError(f"表情包标题已存在：{normalized_title}")

    current_count = await _fetch_meme_count(db_path)
    if current_count >= int(cfg.max_meme_count):
        raise ValueError(f"表情包数量已达上限：{cfg.max_meme_count}")

    existing_by_hash = await _get_meme_by_hash(db_path, image_hash)
    file_ext = _guess_file_extension(image_name=normalized_image_name, local_path=local_path)
    created_new_file = existing_by_hash is None
    stored_path = (
        Path(existing_by_hash.stored_path)
        if existing_by_hash is not None
        else await _store_image_file_if_needed(
            image_hash=image_hash,
            image_bytes=image_bytes,
            file_ext=file_ext,
        )
    )

    try:
        await _insert_meme_record(
            db_path=db_path,
            title=normalized_title,
            image_hash=image_hash,
            stored_path=str(stored_path),
            file_size_bytes=int(image_size_bytes),
        )
    except Exception as exc:  # noqa: BLE001
        if created_new_file:
            try:
                Path(stored_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("meme file cleanup failed after insert error: path=%s", stored_path, exc_info=True)
        raise RuntimeError(f"保存表情包失败：{type(exc).__name__}: {exc!s}") from exc

    return f"已收藏表情包：{normalized_title}"


class MemeContextMiddleware(AgentMiddleware[AgentState, GroupChatContext]):
    """在 LLM 上下文中注入可用表情包说明。"""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        plugin_ctx = get_current_plugin_context()
        if plugin_ctx is None or plugin_ctx.extra.get("_meme_context_injected") is True:
            result = handler(request)
            return await result if asyncio.iscoroutine(result) else result

        try:
            prompt = await build_meme_context_prompt()
            messages = list(getattr(request, "messages", []) or [])
            messages.append(HumanMessage(content=prompt))

            override = getattr(request, "override", None)
            if callable(override):
                request = override(messages=messages)
            else:
                setattr(request, "messages", messages)
            plugin_ctx.extra["_meme_context_injected"] = True
        except Exception:
            logger.warning("meme middleware: 注入表情包上下文失败", exc_info=True)

        result = handler(request)
        return await result if asyncio.iscoroutine(result) else result
