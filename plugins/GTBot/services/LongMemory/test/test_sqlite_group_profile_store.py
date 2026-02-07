#!/usr/bin/env python3
"""SQLite 群画像存储（SQLiteGroupProfileStore）快速自测脚本。

使用方式：

    python plugins/GTBot/services/LongMemory/test/test_sqlite_group_profile_store.py

该脚本不会依赖 NoneBot，不会联网。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4


async def main() -> int:
    # 避免导入 plugins.GTBot 触发 NoneBot 初始化：按路径加载 LongMemory 模块。
    longmemory_dir = Path(__file__).resolve().parents[1]

    def load_longmemory_package(longmemory_path: Path) -> str:
        package_name = f"_longmemory_sqlite_testpkg_{uuid4().hex}"
        pkg = ModuleType(package_name)
        pkg.__path__ = [str(longmemory_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = pkg
        return package_name

    def load_module_from_path(module_qualname: str, file_path: Path) -> ModuleType:
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_qualname, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法创建 spec: {module_qualname} -> {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_qualname] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg = load_longmemory_package(longmemory_dir)
    load_module_from_path(f"{pkg}.model", longmemory_dir / "model.py")
    sqlite_mod = load_module_from_path(f"{pkg}.GroupProfileSQLite", longmemory_dir / "GroupProfileSQLite.py")
    SQLiteGroupProfileStore = getattr(sqlite_mod, "SQLiteGroupProfileStore")

    tmp_db = Path(__file__).parent / "_tmp_group_profiles.db"
    if tmp_db.exists():
        os.remove(tmp_db)

    store = SQLiteGroupProfileStore(db_path=tmp_db)
    await store.create_tables()

    group_id = 123456
    doc_ids = await store.add_group_profile(group_id, ["群规：不要刷屏", "偏好：叫我群主"], category="rule")
    assert len(doc_ids) == 2

    p = await store.get_group_profiles(group_id, limit=10)
    assert p.id == group_id
    assert len(p.description) == 2

    ok = await store.update_by_doc_id(p.description[0].doc_id, description="群规：禁止刷屏")
    assert ok is True

    deleted = await store.delete_all_by_group_id(group_id)
    assert deleted >= 1

    await store.close()
    if tmp_db.exists():
        os.remove(tmp_db)

    print("[OK] SQLiteGroupProfileStore basic flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
