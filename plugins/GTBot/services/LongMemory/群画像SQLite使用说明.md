# SQLite 群画像存储使用说明（LongMemory）

本文档介绍如何使用 `SQLiteGroupProfileStore` 在本地 SQLite 中存储“群画像”（不包含向量/相似度检索）。

## 适用场景

- 你只需要：新增/查询/更新/删除 群画像条目。
- 不需要：按语义相似度检索（向量检索）。

## 依赖

- Python 包：`aiosqlite`
- 本项目已在依赖中加入 `aiosqlite>=0.19.0`。

## 数据结构

群画像按“条目”存储：每条画像是一行记录。

返回结构使用 LongMemory 的数据模型：

- `GroupProfileWithDescriptionIds`
  - `id`: 群号
  - `description`: `list[GroupProfileDescriptionWithId]`
- `GroupProfileDescriptionWithId`
  - `doc_id`: 数据库记录 ID（自增主键，字符串形式）
  - `text`: 描述文本

## 默认数据库位置与表名

- 默认库文件路径：`plugins/GTBot/data/group_profiles.db`
  - 可通过 `SQLiteGroupProfileStore.default_db_path()` 获取。
- 默认表名：`group_profiles`

表结构（自动创建）：

- `doc_id INTEGER PRIMARY KEY AUTOINCREMENT`
- `group_id INTEGER NOT NULL`
- `description TEXT NOT NULL`
- `category TEXT`（可选）
- `creation_time REAL NOT NULL`
- `last_updated REAL NOT NULL`
- `last_read REAL NOT NULL`

同时会创建索引：

- `group_id`
- `last_updated`

## 快速开始（脚本/调试）

在任何异步环境中（脚本、测试）都可以直接使用：

```python
import asyncio

from plugins.GTBot.services.LongMemory.GroupProfileSQLite import SQLiteGroupProfileStore


async def demo() -> None:
    store = SQLiteGroupProfileStore(db_path=SQLiteGroupProfileStore.default_db_path())
    await store.create_tables()

    group_id = 123456

    # 1) 新增
    doc_ids = await store.add_group_profile(group_id, ["群规：不要刷屏", "偏好：叫我群主"], category="rule")
    print("inserted:", doc_ids)

    # 2) 查询（按 last_updated 倒序）
    profile = await store.get_group_profiles(group_id, limit=50)
    for item in profile.description:
        print(item.doc_id, item.text)

    # 3) 更新（按 doc_id 更新）
    await store.update_by_doc_id(profile.description[0].doc_id, description="群规：禁止刷屏")

    # 4) 删除单条
    await store.delete_by_doc_id(profile.description[0].doc_id)

    # 5) 清空整群
    deleted = await store.delete_all_by_group_id(group_id)
    print("deleted rows:", deleted)

    await store.close()


if __name__ == "__main__":
    asyncio.run(demo())
```

## 在 NoneBot 中推荐的接入方式（单例 + 生命周期托管）

建议：

- 全局只创建一个 `SQLiteGroupProfileStore` 实例（避免多连接/并发写冲突）。
- 在 `on_startup` 里 `connect()` + `create_tables()`。
- 在 `on_shutdown` 里 `close()`。

示例（你可以放到自己的服务模块中）：

```python
from nonebot import get_driver

from plugins.GTBot.services.LongMemory.GroupProfileSQLite import SQLiteGroupProfileStore


group_profile_store = SQLiteGroupProfileStore(
    db_path=SQLiteGroupProfileStore.default_db_path(),
)


@get_driver().on_startup
async def _init_group_profile_store() -> None:
    await group_profile_store.connect()
    await group_profile_store.create_tables()


@get_driver().on_shutdown
async def _close_group_profile_store() -> None:
    await group_profile_store.close()
```

业务代码中直接使用 `group_profile_store`：

- 新增：`await group_profile_store.add_group_profile(group_id, "...")`
- 列表：`await group_profile_store.get_group_profiles(group_id)`
- 更新：`await group_profile_store.update_by_doc_id(doc_id, description="...")`
- 删除：`await group_profile_store.delete_by_doc_id(doc_id)`

## API 说明

### `SQLiteGroupProfileStore(db_path, table_name="group_profiles")`

- `db_path`: SQLite 文件路径（`str | Path`）
- `table_name`: 表名，默认 `group_profiles`

### `await create_tables()`

- 连接数据库并创建表/索引（幂等，可重复调用）。

### `await add_group_profile(group_id, profile_texts, category=None) -> list[str]`

- `profile_texts` 支持 `str` 或 `Sequence[str]`
- 返回新增记录的 `doc_id` 列表（字符串）

### `await get_group_profiles(group_id, limit=50, sort_by="last_updated", sort_order="desc", touch_read_time=True)`

- `sort_by`: 仅支持 `creation_time | last_updated | last_read`
- `touch_read_time=True` 时，会把“本次返回的记录”的 `last_read` 更新为当前时间

### `await update_by_doc_id(doc_id, description=None, category=None, last_updated=None) -> bool`

- `doc_id` 必须是数字字符串（SQLite 自增主键）
- `description=None` 表示不更新文本；但仍会更新 `last_updated`
- `category=""`（空字符串）表示清空分类

### `await delete_by_doc_id(doc_id) -> bool`

- 删除单条记录。

### `await delete_all_by_group_id(group_id) -> int`

- 清空某群所有画像条目，返回删除行数。

## 并发与注意事项

- 本实现内置 `_write_lock`，保证同一进程内写入串行化（SQLite 更稳定）。
- 如果你在多进程部署（多个 bot 进程共享同一个 db 文件），仍可能出现锁竞争；此时建议改为单进程写入或使用独立数据库服务。
- 本实现不做“去重/合并”。如果需要避免重复条目，建议在写入前自行做字符串去重，或按业务规则合并。

## 相关文件

- 实现：plugins/GTBot/services/LongMemory/GroupProfileSQLite.py
- 数据模型：plugins/GTBot/services/LongMemory/model.py
- 自测脚本：plugins/GTBot/services/LongMemory/test/test_sqlite_group_profile_store.py
