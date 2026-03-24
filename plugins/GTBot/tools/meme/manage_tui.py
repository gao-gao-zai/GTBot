from __future__ import annotations

import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.tools.meme.config import get_meme_plugin_config


console = Console()


@dataclass(frozen=True)
class MemeRow:
    """表情包行数据。

    Attributes:
        meme_id: 数据库主键。
        title: 表情包标题。
        image_hash: 图片内容哈希。
        stored_path: 图片存储路径。
        file_size_bytes: 图片字节大小。
        created_at: 创建时间戳。
        updated_at: 更新时间戳。
        last_used_at: 最近使用时间戳。
    """

    meme_id: int
    title: str
    image_hash: str
    stored_path: str
    file_size_bytes: int
    created_at: float
    updated_at: float
    last_used_at: float | None = None


def _get_meme_db_path() -> Path:
    """返回表情包数据库路径。"""

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


def _ensure_db(conn: sqlite3.Connection) -> None:
    """确保表情包数据库表存在。"""

    conn.execute(
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meme_store_image_hash ON meme_store(image_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meme_store_last_used_at ON meme_store(last_used_at)")
    conn.commit()


def _open_conn(db_path: Path) -> sqlite3.Connection:
    """打开数据库连接并初始化表结构。"""

    conn = sqlite3.connect(str(db_path))
    _ensure_db(conn)
    return conn


def _row_to_meme(row: tuple[Any, ...]) -> MemeRow:
    """将数据库行转换为表情包对象。"""

    return MemeRow(
        meme_id=int(row[0]),
        title=str(row[1]),
        image_hash=str(row[2]),
        stored_path=str(row[3]),
        file_size_bytes=int(row[4]),
        created_at=float(row[5]),
        updated_at=float(row[6]),
        last_used_at=float(row[7]) if row[7] is not None else None,
    )


def _format_ts(ts: float | None) -> str:
    """格式化时间戳。"""

    if ts is None:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "-"


def _format_size(size_bytes: int) -> str:
    """格式化字节大小。"""

    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{int(size_bytes)} B"


def _fetch_memes(conn: sqlite3.Connection, *, keyword: str = "") -> list[MemeRow]:
    """查询表情包列表。

    Args:
        conn: SQLite 连接。
        keyword: 可选标题关键字。

    Returns:
        按最近使用时间与创建时间排序的表情包列表。
    """

    normalized_keyword = str(keyword or "").strip()
    if normalized_keyword:
        rows = conn.execute(
            """
            SELECT id, title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at
            FROM meme_store
            WHERE title LIKE ?
            ORDER BY COALESCE(last_used_at, 0) DESC, created_at DESC, title ASC
            """,
            (f"%{normalized_keyword}%",),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at
            FROM meme_store
            ORDER BY COALESCE(last_used_at, 0) DESC, created_at DESC, title ASC
            """
        ).fetchall()
    return [_row_to_meme(row) for row in rows]


def _get_meme_by_id(conn: sqlite3.Connection, meme_id: int) -> MemeRow | None:
    """按 ID 查询表情包。"""

    row = conn.execute(
        """
        SELECT id, title, image_hash, stored_path, file_size_bytes, created_at, updated_at, last_used_at
        FROM meme_store
        WHERE id = ?
        """,
        (int(meme_id),),
    ).fetchone()
    return _row_to_meme(row) if row is not None else None


def _rename_meme(conn: sqlite3.Connection, *, meme_id: int, new_title: str) -> None:
    """重命名表情包标题。"""

    normalized_title = str(new_title or "").strip()
    if not normalized_title:
        raise ValueError("新标题不能为空")

    existing = conn.execute(
        "SELECT id FROM meme_store WHERE title = ? AND id != ?",
        (normalized_title, int(meme_id)),
    ).fetchone()
    if existing is not None:
        raise ValueError(f"标题已存在：{normalized_title}")

    updated = conn.execute(
        "UPDATE meme_store SET title = ?, updated_at = ? WHERE id = ?",
        (normalized_title, float(time.time()), int(meme_id)),
    )
    conn.commit()
    if updated.rowcount <= 0:
        raise ValueError(f"未找到 ID={meme_id} 的表情包")


def _delete_meme(conn: sqlite3.Connection, *, meme_id: int) -> tuple[str, Path | None]:
    """删除表情包记录，并在无引用时删除图片文件。

    Args:
        conn: SQLite 连接。
        meme_id: 要删除的表情包 ID。

    Returns:
        被删除的标题与可能被一并删除的图片路径。
    """

    row = _get_meme_by_id(conn, meme_id)
    if row is None:
        raise ValueError(f"未找到 ID={meme_id} 的表情包")

    conn.execute("DELETE FROM meme_store WHERE id = ?", (int(meme_id),))
    conn.commit()

    remain = conn.execute(
        "SELECT COUNT(1) FROM meme_store WHERE image_hash = ?",
        (row.image_hash,),
    ).fetchone()
    remain_count = int(remain[0]) if remain and remain[0] is not None else 0

    removed_file: Path | None = None
    if remain_count <= 0:
        path = Path(row.stored_path)
        if path.exists():
            path.unlink()
            removed_file = path

    return row.title, removed_file


def _collect_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """汇总表情包统计信息。"""

    total_row = conn.execute("SELECT COUNT(1), COALESCE(SUM(file_size_bytes), 0) FROM meme_store").fetchone()
    missing_files = 0
    existing_files = 0
    for meme in _fetch_memes(conn):
        if Path(meme.stored_path).exists():
            existing_files += 1
        else:
            missing_files += 1

    return {
        "count": int(total_row[0]) if total_row else 0,
        "total_size_bytes": int(total_row[1]) if total_row else 0,
        "existing_files": existing_files,
        "missing_files": missing_files,
    }


def _render_header(conn: sqlite3.Connection, *, db_path: Path, files_dir: Path) -> None:
    """渲染顶部摘要。"""

    cfg = get_meme_plugin_config()
    stats = _collect_stats(conn)
    lines = [
        f"DB: {db_path}",
        f"Files: {files_dir}",
        f"Count: {stats['count']}/{cfg.max_meme_count}",
        f"Size: {_format_size(stats['total_size_bytes'])}",
        f"Missing files: {stats['missing_files']}",
        f"Limits: max_size={_format_size(cfg.max_meme_size_bytes)} title_max={cfg.max_title_chars}",
    ]
    console.print(Panel("\n".join(lines), title="Meme Manager", border_style="cyan"))


def _render_table(memes: list[MemeRow], *, keyword: str = "") -> None:
    """渲染表情包列表。"""

    title = "当前表情包"
    if keyword:
        title = f"当前表情包 - 搜索: {keyword}"

    table = Table(title=title, show_lines=False)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("标题", style="bold")
    table.add_column("大小", justify="right")
    table.add_column("最近使用")
    table.add_column("创建时间")
    table.add_column("文件", overflow="fold")

    for meme in memes:
        path = Path(meme.stored_path)
        file_label = str(path.name)
        if not path.exists():
            file_label = f"[red]{path.name} (missing)[/red]"
        table.add_row(
            str(meme.meme_id),
            meme.title,
            _format_size(meme.file_size_bytes),
            _format_ts(meme.last_used_at),
            _format_ts(meme.created_at),
            file_label,
        )

    if not memes:
        console.print(Panel("当前没有匹配的表情包。", border_style="yellow"))
        return

    console.print(table)


def _render_help() -> None:
    """渲染帮助信息。"""

    help_table = Table(title="操作说明")
    help_table.add_column("命令", style="cyan", no_wrap=True)
    help_table.add_column("说明")
    help_table.add_row("r", "刷新列表")
    help_table.add_row("s", "按标题搜索")
    help_table.add_row("v", "查看某个表情包详情")
    help_table.add_row("o", "打开表情包图片")
    help_table.add_row("n", "重命名表情包")
    help_table.add_row("d", "删除表情包")
    help_table.add_row("c", "查看当前配置")
    help_table.add_row("x", "检查丢失文件")
    help_table.add_row("q", "退出")
    console.print(help_table)


def _show_meme_detail(conn: sqlite3.Connection) -> None:
    """展示单个表情包详情。"""

    meme_id = IntPrompt.ask("输入要查看的表情包 ID")
    meme = _get_meme_by_id(conn, meme_id)
    if meme is None:
        console.print(f"[red]未找到 ID={meme_id} 的表情包[/red]")
        return

    path = Path(meme.stored_path)
    lines = [
        f"ID: {meme.meme_id}",
        f"标题: {meme.title}",
        f"哈希: {meme.image_hash}",
        f"大小: {_format_size(meme.file_size_bytes)}",
        f"创建时间: {_format_ts(meme.created_at)}",
        f"更新时间: {_format_ts(meme.updated_at)}",
        f"最近使用: {_format_ts(meme.last_used_at)}",
        f"文件路径: {path}",
        f"文件存在: {'是' if path.exists() else '否'}",
    ]
    console.print(Panel("\n".join(lines), title="表情包详情", border_style="green"))


def _rename_meme_interactive(conn: sqlite3.Connection) -> None:
    """交互式重命名表情包。"""

    meme_id = IntPrompt.ask("输入要重命名的表情包 ID")
    meme = _get_meme_by_id(conn, meme_id)
    if meme is None:
        console.print(f"[red]未找到 ID={meme_id} 的表情包[/red]")
        return

    new_title = Prompt.ask("输入新标题", default=meme.title).strip()
    if new_title == meme.title:
        console.print("[yellow]标题未变化，已跳过。[/yellow]")
        return

    _rename_meme(conn, meme_id=meme_id, new_title=new_title)
    console.print(f"[green]已重命名：{meme.title} -> {new_title}[/green]")


def _open_meme_file(conn: sqlite3.Connection) -> None:
    """在系统文件管理器中打开表情包图片。"""

    meme_id = IntPrompt.ask("输入要打开的表情包 ID")
    meme = _get_meme_by_id(conn, meme_id)
    if meme is None:
        console.print(f"[red]未找到 ID={meme_id} 的表情包[/red]")
        return

    path = Path(meme.stored_path)
    if not path.exists():
        console.print(f"[red]文件不存在：{path}[/red]")
        return

    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        import subprocess

        subprocess.Popen(["open", str(path)])
    else:
        import subprocess

        subprocess.Popen(["xdg-open", str(path)])
    console.print(f"[green]已打开：{meme.title}[/green]")


def _delete_meme_interactive(conn: sqlite3.Connection) -> None:
    """交互式删除表情包。"""

    meme_id = IntPrompt.ask("输入要删除的表情包 ID")
    meme = _get_meme_by_id(conn, meme_id)
    if meme is None:
        console.print(f"[red]未找到 ID={meme_id} 的表情包[/red]")
        return

    confirmed = Confirm.ask(f"确认删除表情包 [{meme.title}] 吗？", default=False)
    if not confirmed:
        console.print("[yellow]已取消删除。[/yellow]")
        return

    deleted_title, removed_file = _delete_meme(conn, meme_id=meme_id)
    if removed_file is not None:
        console.print(f"[green]已删除表情包：{deleted_title}，并删除图片文件 {removed_file.name}[/green]")
        return
    console.print(f"[green]已删除表情包：{deleted_title}[/green]")


def _show_config() -> None:
    """展示当前配置。"""

    cfg = get_meme_plugin_config()
    lines = [
        f"max_meme_count = {cfg.max_meme_count}",
        f"max_meme_size_bytes = {cfg.max_meme_size_bytes} ({_format_size(cfg.max_meme_size_bytes)})",
        f"max_title_chars = {cfg.max_title_chars}",
        f"max_injected_memes = {cfg.max_injected_memes}",
    ]
    console.print(Panel("\n".join(lines), title="当前配置", border_style="blue"))


def _check_missing_files(conn: sqlite3.Connection) -> None:
    """检查缺失的图片文件。"""

    missing: list[MemeRow] = []
    for meme in _fetch_memes(conn):
        if not Path(meme.stored_path).exists():
            missing.append(meme)

    if not missing:
        console.print("[green]没有发现丢失的表情包文件。[/green]")
        return

    table = Table(title="缺失文件")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("标题", style="bold")
    table.add_column("路径", overflow="fold")
    for meme in missing:
        table.add_row(str(meme.meme_id), meme.title, meme.stored_path)
    console.print(table)


def run_tui() -> int:
    """运行表情包管理 TUI 主循环。

    Returns:
        进程退出码。
    """

    db_path = _get_meme_db_path()
    files_dir = _get_meme_files_dir()
    keyword = ""

    with _open_conn(db_path) as conn:
        while True:
            console.clear()
            memes = _fetch_memes(conn, keyword=keyword)
            _render_header(conn, db_path=db_path, files_dir=files_dir)
            _render_table(memes, keyword=keyword)
            _render_help()

            command = Prompt.ask("输入命令").strip().lower()
            console.print()

            try:
                if command in {"q", "quit", "exit"}:
                    return 0
                if command in {"r", "refresh", ""}:
                    continue
                if command in {"s", "search"}:
                    keyword = Prompt.ask("输入标题关键字，留空表示清空搜索", default="").strip()
                    continue
                if command in {"v", "view"}:
                    _show_meme_detail(conn)
                elif command in {"o", "open"}:
                    _open_meme_file(conn)
                elif command in {"n", "rename"}:
                    _rename_meme_interactive(conn)
                elif command in {"d", "delete"}:
                    _delete_meme_interactive(conn)
                elif command in {"c", "config"}:
                    _show_config()
                elif command in {"x", "check"}:
                    _check_missing_files(conn)
                else:
                    console.print(f"[yellow]未知命令：{command}[/yellow]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]操作失败：{type(exc).__name__}: {exc!s}[/red]")

            Prompt.ask("按回车继续", default="")


def main() -> None:
    """脚本入口。"""

    raise SystemExit(run_tui())


if __name__ == "__main__":
    sys.exit(run_tui())
