from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


_TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)


@dataclass(frozen=True)
class VLMClientConfig:
    """迁移脚本使用的 VLM 配置。"""

    base_url: str
    api_key: str
    model: str
    timeout_sec: float
    max_tokens: int
    title_recommended_chars: int | None
    title_max_chars: int
    description_max_chars: int
    extra_body: dict[str, Any]
    extra_headers: dict[str, str]
    allow_override_reserved: bool


def _default_db_path() -> Path:
    """返回默认的识图缓存数据库路径。"""
    return Path(__file__).resolve().parents[2] / "data" / "vlm_image_cache.sqlite3"


def _default_config_path() -> Path:
    """返回默认的 VLMImage 配置路径。"""
    return Path(__file__).resolve().with_name("config.json")


def _get_vlm_chat_completions_url(base_url: str) -> str:
    """规范化 VLM chat completions 接口地址。"""
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("base_url 不能为空")

    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _has_column(conn: sqlite3.Connection, *, table: str, column: str) -> bool:
    """检查 SQLite 表是否包含指定列。"""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(len(row) > 1 and str(row[1]) == column for row in rows)


def _load_vlm_config(config_path: Path) -> VLMClientConfig:
    """从配置文件加载迁移脚本所需的 VLM 参数。"""
    path = config_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"VLM 配置文件不存在: {path}")

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"VLM 配置格式异常: {path}")

    cfg = VLMClientConfig(
        base_url=str(parsed.get("base_url") or "").strip(),
        api_key=str(parsed.get("api_key") or "").strip(),
        model=str(parsed.get("model") or "").strip(),
        timeout_sec=float(parsed.get("timeout_sec") or 60.0),
        max_tokens=int(parsed.get("max_tokens") or 512),
        title_recommended_chars=(
            int(parsed["title_recommended_chars"])
            if parsed.get("title_recommended_chars") is not None
            else None
        ),
        title_max_chars=int(parsed.get("title_max_chars") or 16),
        description_max_chars=int(parsed.get("description_max_chars") or 120),
        extra_body=dict(parsed.get("extra_body") or {}),
        extra_headers={str(k): str(v) for k, v in dict(parsed.get("extra_headers") or {}).items()},
        allow_override_reserved=bool(parsed.get("allow_override_reserved", False)),
    )
    if not cfg.base_url:
        raise ValueError(f"VLM 配置缺少 base_url: {path}")
    if not cfg.model:
        raise ValueError(f"VLM 配置缺少 model: {path}")
    return cfg


def _build_title_prompt(
    description: str,
    *,
    title_recommended_chars: int | None,
    title_max_chars: int,
) -> str:
    """根据已有描述构造标题补齐提示词。"""
    title_requirement = f"title 尽量不超过 {int(title_max_chars)} 个字。"
    if title_recommended_chars is not None:
        title_requirement = (
            f"title 推荐 {int(title_recommended_chars)} 个字左右，尽量不超过 {int(title_max_chars)} 个字。"
        )
    return (
        "下面是一张图片已有的描述，请先认真理解描述内容，再只根据这段描述为图片生成一个简短准确的标题。"
        "严格只返回以下 XML，不要输出任何额外文字、Markdown、代码块或解释。\n"
        "<title>图片标题</title>\n"
        f"要求：{title_requirement}\n"
        f"图片描述：{description.strip()}"
    )


def _filter_reserved(*, data: dict[str, Any], reserved: set[str], allow_override: bool) -> dict[str, Any]:
    """过滤保留字段，避免覆盖核心请求参数。"""
    if allow_override:
        return dict(data)
    return {k: v for k, v in dict(data).items() if str(k) not in reserved}


def _extract_openai_content(data: Any) -> str:
    """提取 OpenAI 兼容响应中的文本内容。"""
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


def _parse_title_xml(raw_text: str) -> str:
    """解析仅包含 `<title>` 的 XML 结果。"""
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("VLM 返回内容为空")

    matches = _TITLE_TAG_RE.findall(text)
    if len(matches) != 1:
        raise RuntimeError("VLM 返回的 XML 格式异常：title 标签缺失或重复")

    leftover = _TITLE_TAG_RE.sub("", text, count=1)
    if leftover.strip():
        raise RuntimeError("VLM 返回的 XML 格式异常：包含额外文本")

    title = str(matches[0]).strip()
    if not title:
        raise RuntimeError("VLM 返回的 XML 格式异常：title 为空")
    return title


def _call_vlm_for_title(*, cfg: VLMClientConfig, description: str) -> str:
    """调用 VLM，根据描述生成标题。"""
    url = _get_vlm_chat_completions_url(cfg.base_url)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    headers.update(cfg.extra_headers)

    reserved_keys = {"model", "messages"}
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": _build_title_prompt(
                    description,
                    title_recommended_chars=cfg.title_recommended_chars,
                    title_max_chars=cfg.title_max_chars,
                ),
            }
        ],
        "max_tokens": int(cfg.max_tokens),
    }
    payload.update(
        _filter_reserved(
            data=cfg.extra_body,
            reserved=reserved_keys,
            allow_override=cfg.allow_override_reserved,
        )
    )

    resp = requests.post(url, json=payload, headers=headers, timeout=float(cfg.timeout_sec))
    resp.raise_for_status()
    data = resp.json()
    return _parse_title_xml(_extract_openai_content(data))


def migrate_cache_schema(db_path: Path) -> None:
    """为识图缓存数据库补充 `title` 与 `image_size_bytes` 列。"""
    path = db_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {path}")

    with sqlite3.connect(str(path)) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_cache' LIMIT 1"
        ).fetchone()
        if not table_exists:
            raise RuntimeError(f"数据库中不存在 image_cache 表: {path}")

        changed: list[str] = []
        if not _has_column(conn, table="image_cache", column="title"):
            conn.execute("ALTER TABLE image_cache ADD COLUMN title TEXT")
            changed.append("title")
        if not _has_column(conn, table="image_cache", column="image_size_bytes"):
            conn.execute("ALTER TABLE image_cache ADD COLUMN image_size_bytes INTEGER")
            changed.append("image_size_bytes")

        if not changed:
            print(f"无需补列，title / image_size_bytes 列已存在: {path}")
            return

        conn.commit()
        print(f"补列完成，已新增列 {', '.join(changed)}: {path}")


def backfill_empty_titles(*, db_path: Path, config_path: Path, limit: int | None = None) -> None:
    """用 VLM 根据已有描述补齐空标题。

    Args:
        db_path: SQLite 缓存库路径。
        config_path: VLMImage 配置文件路径。
        limit: 最多处理多少条；为 `None` 时处理全部。
    """
    cfg = _load_vlm_config(config_path)
    path = db_path.resolve()

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        sql = (
            "SELECT image_hash, description FROM image_cache "
            "WHERE COALESCE(TRIM(title), '') = '' AND COALESCE(TRIM(description), '') <> '' "
            "ORDER BY updated_at ASC, created_at ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = conn.execute(sql, params).fetchall()

        if not rows:
            print(f"无需补齐标题，没有发现 title 为空的记录: {path}")
            return

        print(f"开始补齐空标题，共 {len(rows)} 条: {path}")
        ok_count = 0
        for idx, row in enumerate(rows, start=1):
            image_hash = str(row["image_hash"])
            description = str(row["description"] or "").strip()
            if not description:
                continue

            print(f"[{idx}/{len(rows)}] 正在生成标题: {image_hash}")
            title = _call_vlm_for_title(cfg=cfg, description=description)
            conn.execute(
                "UPDATE image_cache SET title = ?, updated_at = CAST(strftime('%s','now') AS REAL) WHERE image_hash = ?",
                (title, image_hash),
            )
            conn.commit()
            ok_count += 1

        print(f"标题补齐完成，成功 {ok_count}/{len(rows)} 条: {path}")


def main() -> None:
    """执行识图缓存数据库迁移与标题补齐脚本。"""
    parser = argparse.ArgumentParser(description="为 vlm_image 缓存库补充 title 列并回填空标题")
    parser.add_argument(
        "--db",
        dest="db_path",
        default=str(_default_db_path()),
        help="待迁移的 SQLite 数据库路径，默认使用 plugins/GTBot/data/vlm_image_cache.sqlite3",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(_default_config_path()),
        help="VLMImage 配置文件路径，默认使用 plugins/GTBot/tools/vlm_image/config.json",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="只补 title 列，不调用 VLM 回填空标题",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多回填多少条空标题记录，默认处理全部",
    )
    args = parser.parse_args()

    db_path = Path(str(args.db_path))
    migrate_cache_schema(db_path)
    if not args.skip_backfill:
        backfill_empty_titles(
            db_path=db_path,
            config_path=Path(str(args.config_path)),
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
