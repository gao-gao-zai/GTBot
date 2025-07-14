import aiosqlite
import asyncio
from typing import List, Dict, Optional, Tuple, Union, Any, cast
from pathlib import Path
import time
import sys
from nonebot import get_driver
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from config_manager import global_config_manager as gcm


class SQLiteManager:
    def __init__(self, db_path: str | Path = "database.db"):
        self.db_path = db_path
        self.connection: Optional[aiosqlite.Connection] = None
        self._tables = {}
        # 写入锁和状态跟踪
        self._write_lock = asyncio.Lock()
        self._write_pending = False
        self._last_write_time = 0.0
        self._write_lock_holder = None
        self._close_requested = False  # 关闭请求标志

    async def connect(self) -> None:
        """创建数据库连接"""
        if self.connection is None:
            self.connection = await aiosqlite.connect(self.db_path)
            self.connection.row_factory = aiosqlite.Row
            # 启用 WAL 模式提高并发性能
            await self.connection.execute("PRAGMA journal_mode=WAL;")
            await self.connection.commit()
    
    async def close(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        关闭数据库连接
        :param wait: 是否等待写入操作完成
        :param timeout: 等待超时时间（秒）
        """
        if self.connection is None:
            return
            
        self._close_requested = True
        
        if wait:
            await self.wait_for_write_completion(timeout)
        
        # 即使写入未完成也强制关闭（如果wait=False）
        if self.connection:
            await self.connection.close()
            self.connection = None
            
    async def __aenter__(self) -> "SQLiteManager":
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close(wait=True)
    
    def table(self, table_name: str) -> "TableManager":
        """获取表管理器"""
        if table_name not in self._tables:
            self._tables[table_name] = TableManager(self, table_name)
        return self._tables[table_name]
    
    async def _ensure_connection(self) -> aiosqlite.Connection:
        """确保数据库连接已建立"""
        if self._close_requested:
            raise RuntimeError("Database is closing or has been closed")
            
        if self.connection is None:
            await self.connect()
        # 使用类型断言确保connection不为None
        assert self.connection is not None
        return self.connection
    
    async def execute(
        self, 
        query: str, 
        params: tuple = (), 
        wait: bool = True, 
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> aiosqlite.Cursor:
        """
        执行SQL查询并提交
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间（秒）
        :param write_timeout: 写入操作本身的超时时间（秒）
        """
        if self._close_requested:
            raise RuntimeError("Database is closing, cannot execute queries")
        
        start_time = time.monotonic()
        
        # 检查是否已经有写入操作在进行
        if self._write_pending:
            if not wait:
                raise TimeoutError("Write operation in progress and wait=False")
            
            # 等待锁释放
            while self._write_pending:
                elapsed = time.monotonic() - start_time
                if acquire_timeout is not None and elapsed >= acquire_timeout:
                    raise TimeoutError(f"Timeout waiting for write lock after {elapsed:.2f}s")
                await asyncio.sleep(0.05)  # 短暂等待避免忙循环
        
        # 获取写入锁
        self._write_pending = True
        self._write_lock_holder = asyncio.current_task()
        
        try:
            # 执行写入操作
            connection = await self._ensure_connection()
            
            # 设置写入超时
            if write_timeout is not None:
                try:
                    cursor = await asyncio.wait_for(
                        connection.execute(query, params),
                        timeout=write_timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Write operation timed out after {write_timeout}s")
            else:
                cursor = await connection.execute(query, params)
            
            await connection.commit()
            self._last_write_time = time.monotonic()
            return cursor
        finally:
            # 释放写入锁
            self._write_pending = False
            self._write_lock_holder = None
    
    async def fetch_all(self, query: str, params: tuple = ()) -> List[Dict]:
        """获取所有结果"""
        connection = await self._ensure_connection()
        async with connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """获取单条结果"""
        connection = await self._ensure_connection()
        async with connection.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    @property
    def is_writing(self) -> bool:
        """检查是否有写入操作在进行"""
        return self._write_pending
    
    @property
    def last_write_time(self) -> float:
        """获取上次写入完成的时间戳"""
        return self._last_write_time
    
    async def wait_for_write_completion(self, timeout: Optional[float] = None) -> None:
        """等待当前写入操作完成"""
        start_time = time.monotonic()
        while self._write_pending:
            elapsed = time.monotonic() - start_time
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(f"Timeout waiting for write completion after {elapsed:.2f}s")
            await asyncio.sleep(0.05)


    async def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表的所有索引信息"""
        connection = await self._ensure_connection()
        query = """
            SELECT 
                m.name AS index_name,
                ii.name AS column_name,
                m.tbl_name AS table_name,
                m.sql AS sql_statement,
                m.unique AS is_unique
            FROM 
                sqlite_master AS m
            JOIN 
                pragma_index_info(m.name) AS ii
            WHERE 
                m.type = 'index' 
                AND m.tbl_name = ?
            ORDER BY 
                m.name, ii.seqno
        """
        return await self.fetch_all(query, (table_name,))

    async def create_index(
        self, 
        table_name: str, 
        columns: Union[str, List[str]], 
        index_name: Optional[str] = None,
        unique: bool = False,
        if_not_exists: bool = True,
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> None:
        """
        创建索引
        :param table_name: 表名
        :param columns: 列名或列名列表
        :param index_name: 索引名称（可选）
        :param unique: 是否创建唯一索引
        :param if_not_exists: 如果索引已存在是否跳过
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间
        :param write_timeout: 写入操作超时时间
        """
        if not index_name:
            if isinstance(columns, str):
                columns_str = columns
            else:
                columns_str = "_".join(columns)
            index_name = f"idx_{table_name}_{columns_str}"
        
        if isinstance(columns, list):
            columns_str = ", ".join(columns)
        else:
            columns_str = columns
        
        unique_str = "UNIQUE " if unique else ""
        if_not_exists_str = "IF NOT EXISTS " if if_not_exists else ""
        
        query = f"""
            CREATE {unique_str}INDEX {if_not_exists_str}{index_name} 
            ON {table_name} ({columns_str})
        """
        
        await self.execute(
            query, 
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )

    async def drop_index(
        self, 
        index_name: str, 
        if_exists: bool = True,
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> None:
        """
        删除索引
        :param index_name: 索引名称
        :param if_exists: 如果索引不存在是否跳过
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间
        :param write_timeout: 写入操作超时时间
        """
        if_exists_str = "IF EXISTS " if if_exists else ""
        query = f"DROP INDEX {if_exists_str}{index_name}"
        await self.execute(
            query, 
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
    
    async def index_exists(self, table_name: str, index_name: str) -> bool:
        """检查索引是否存在"""
        indexes = await self.get_indexes(table_name)
        return any(index['index_name'] == index_name for index in indexes)

    async def optimize_indexes_for_table(
        self, 
        table_name: str,
        suggested_indexes: Dict[str, Union[str, List[str]]],
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> None:
        """
        优化表的索引配置
        :param table_name: 表名
        :param suggested_indexes: 建议的索引配置 {索引名: 列名或列名列表}
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间
        :param write_timeout: 写入操作超时时间
        """
        existing_indexes = [index['index_name'] for index in await self.get_indexes(table_name)]
        
        # 创建缺失的索引
        for index_name, columns in suggested_indexes.items():
            if index_name not in existing_indexes:
                await self.create_index(
                    table_name, 
                    columns,
                    index_name=index_name,
                    if_not_exists=True,
                    wait=wait,
                    acquire_timeout=acquire_timeout,
                    write_timeout=write_timeout
                )
                print(f"已创建索引: {index_name} ON {table_name}({columns})")
        
        # 删除不在建议列表中的索引（除了主键索引）
        for index in existing_indexes:
            if index not in suggested_indexes and not index.startswith("sqlite_autoindex"):
                await self.drop_index(
                    index, 
                    if_exists=True,
                    wait=wait,
                    acquire_timeout=acquire_timeout,
                    write_timeout=write_timeout
                )
                print(f"已删除未使用的索引: {index}")



class TableManager:
    def __init__(self, db: SQLiteManager, table_name: str):
        self.db = db
        self.table_name = table_name

    async def insert(
        self, 
        *, 
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None,
        **data
    ) -> Optional[int]:
        """
        插入数据
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间（秒）
        :param write_timeout: 写入操作本身的超时时间（秒）
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        cursor = await self.db.execute(
            query, 
            tuple(data.values()),
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
        return cursor.lastrowid

    async def get(self, id: int, columns: str = "*") -> Optional[Dict]:
        """通过ID获取单条记录"""
        result = await self.query(
            columns=columns, 
            where="id = ?", 
            params=(id,), 
            limit=1
        )
        return result[0] if result else None

    async def query(self, columns: str = "*", 
              where: Optional[str] = None, 
              params: tuple = (), 
              order_by: Optional[str] = None, 
              limit: Optional[int] = None) -> List[Dict]:
        """查询数据"""
        query = f"SELECT {columns} FROM {self.table_name}"
        
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        
        return await self.db.fetch_all(query, params)
    
    async def get_nearby_rows(
        self,
        target_id: int,
        before: int = 2,
        after: int = 2,
        order_column: str = "id",
        order_direction: str = "ASC",
        include_target: bool = True,
        columns: str = "*",
        where: Optional[str] = None,
        where_params: tuple = ()
    ) -> List[Dict]:
        """
        获取指定行附近的若干行（支持筛选条件和列选择）
        """
        # 验证排序方向
        order_direction = order_direction.upper()
        if order_direction not in ["ASC", "DESC"]:
            raise ValueError("order_direction 必须是 'ASC' 或 'DESC'")
        
        # 构建基础WHERE条件（包含目标行）
        base_where = " AND ".join(filter(None, [where, "id = ?"]))
        base_params = where_params + (target_id,)
        
        # 获取目标行的排序值
        target_value_query = f"SELECT {order_column} FROM {self.table_name} WHERE {base_where}"
        target_value_result = await self.db.fetch_one(target_value_query, base_params)
        
        if not target_value_result:
            return []
        
        target_value = target_value_result[order_column]
        
        # 构建筛选条件（如果有）
        filter_clause = f" AND ({where})" if where else ""
        filter_params = where_params if where else ()
        
        # 获取之前的行
        before_query = f"""
            SELECT {columns} 
            FROM {self.table_name}
            WHERE {order_column} {'<' if order_direction == 'ASC' else '>'} ? 
            {filter_clause}
            ORDER BY {order_column} {'DESC' if order_direction == 'ASC' else 'ASC'}
            LIMIT ?
        """
        before_rows = await self.db.fetch_all(
            before_query, 
            (target_value,) + filter_params + (before,)
        )
        
        if order_direction == "ASC":
            before_rows = before_rows[::-1]  # 反转顺序
        
        # 获取目标行
        target_row = await self.get(target_id, columns=columns)
        
        # 获取之后的行
        after_query = f"""
            SELECT {columns} 
            FROM {self.table_name}
            WHERE {order_column} {'>' if order_direction == 'ASC' else '<'} ? 
            {filter_clause}
            ORDER BY {order_column} {order_direction}
            LIMIT ?
        """
        after_rows = await self.db.fetch_all(
            after_query, 
            (target_value,) + filter_params + (after,)
        )
        
        # 组合结果
        result: List[Dict] = []
        if before_rows:
            result.extend(before_rows)
        if include_target and target_row:
            result.append(target_row)
        if after_rows:
            result.extend(after_rows)
        
        return result
    
    async def get_nearby_rows_window(
        self,
        target_id: int,
        before: int = 2,
        after: int = 2,
        order_column: str = "id",
        order_direction: str = "ASC",
        include_target: bool = True,
        columns: str = "*",
        where: Optional[str] = None,
        where_params: tuple = ()
    ) -> List[Dict]:
        """
        使用窗口函数获取附近行（支持筛选条件和列选择）
        """
        # 验证排序方向
        order_direction = order_direction.upper()
        if order_direction not in ["ASC", "DESC"]:
            raise ValueError("order_direction 必须是 'ASC' 或 'DESC'")
        
        # 构建筛选条件（如果有）
        where_clause = f"WHERE {where}" if where else ""
        
        # 获取目标行的位置
        position_query = f"""
            SELECT COUNT(*) AS position
            FROM (
                SELECT {order_column} 
                FROM {self.table_name}
                {where_clause}
            ) AS filtered
            WHERE {order_column} {'<' if order_direction == 'ASC' else '>'} 
                (SELECT {order_column} FROM {self.table_name} WHERE id = ?)
        """
        position_result = await self.db.fetch_one(
            position_query, 
            where_params + (target_id,)
        )
        
        if not position_result:
            return []
        
        position = position_result["position"]
        
        # 计算查询范围
        start_row = max(0, position - before)
        total_rows = before + after + (1 if include_target else 0)
        
        # 获取附近行
        nearby_query = f"""
            SELECT {columns} 
            FROM (
                SELECT {columns}, 
                       ROW_NUMBER() OVER (
                           ORDER BY {order_column} {order_direction}
                       ) AS row_num
                FROM {self.table_name}
                {where_clause}
            ) 
            WHERE row_num BETWEEN ? AND ?
            ORDER BY row_num {order_direction}
        """
        return await self.db.fetch_all(
            nearby_query, 
            where_params + (start_row + 1, start_row + total_rows)
        )
    
    async def get_next_row(
        self,
        target_id: int,
        order_column: str = "id",
        order_direction: str = "ASC",
        columns: str = "*",
        where: Optional[str] = None,
        where_params: tuple = ()
    ) -> Optional[Dict]:
        """
        获取下一行（支持筛选条件和列选择）
        """
        result = await self.get_nearby_rows(
            target_id=target_id,
            before=0,
            after=1,
            order_column=order_column,
            order_direction=order_direction,
            include_target=False,
            columns=columns,
            where=where,
            where_params=where_params
        )
        return result[0] if result else None
    
    async def get_previous_row(
        self,
        target_id: int,
        order_column: str = "id",
        order_direction: str = "ASC",
        columns: str = "*",
        where: Optional[str] = None,
        where_params: tuple = ()
    ) -> Optional[Dict]:
        """
        获取上一行（支持筛选条件和列选择）
        """
        result = await self.get_nearby_rows(
            target_id=target_id,
            before=1,
            after=0,
            order_column=order_column,
            order_direction=order_direction,
            include_target=False,
            columns=columns,
            where=where,
            where_params=where_params
        )
        return result[0] if result else None


class GroupChatRecordManager(TableManager):
    """群聊消息记录管理器"""
    def __init__(self, db: SQLiteManager):
        super().__init__(db, "group_chat_record")
        self._indexes_created = False  # 添加索引创建状态标志
  
    async def ensure_indexes(self) -> None:
        """确保必要的索引存在"""
        if self._indexes_created:
            return
            
        # 等待数据库连接就绪
        await self.db.connect()
      
        # 检查并创建常用查询的索引
        if not await self.db.index_exists(self.table_name, "idx_group_id_id"):
            await self.db.create_index(
                self.table_name,
                ["group_id", "id"],
                index_name="idx_group_id_id"
            )
      
        if not await self.db.index_exists(self.table_name, "idx_group_id_send_time"):
            await self.db.create_index(
                self.table_name,
                ["group_id", "send_time"],
                index_name="idx_group_id_send_time"
            )
        
        self._indexes_created = True  # 标记索引已创建


    async def optimize_indexes(self) -> None:
        """优化索引配置"""
        suggested_indexes: Dict = {
            "idx_group_id_id": ["group_id", "id"],
            "idx_group_id_send_time": ["group_id", "send_time"],
            "idx_user_id": ["user_id"],
            "idx_send_time": ["send_time"]
        }
        await self.db.optimize_indexes_for_table(self.table_name, suggested_indexes)
    
    async def insert_record(
        self,
        msg_id: int,
        group_id: int,
        user_id: int,
        content: str,
        send_time: float,
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> Optional[int]:
        """插入群聊消息记录"""
        if not self._indexes_created:
            await self.ensure_indexes()  # 确保索引已创建
        return await self.insert(
            msg_id=msg_id,
            group_id=group_id,
            user_id=user_id,
            content=content,
            send_time=send_time,
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
    
    async def get_by_msg_id(self, msg_id: int) -> Optional[Dict]:
        """通过消息ID获取群聊消息记录"""
        if not self._indexes_created:
            await self.ensure_indexes()
        return await self.get(id=msg_id)
    
    async def get_recent_records(
        self,
        group_id: int,
        limit: int = 10,
        before: Optional[int] = None,
        after: Optional[int] = None
    ) -> List[Dict]:
        """
        获取群组最近的消息记录
        :param group_id: 群组ID
        :param limit: 获取的消息数量
        :param before: 获取早于指定ID的消息
        :param after: 获取晚于指定ID的消息
        """
        if not self._indexes_created:
            await self.ensure_indexes()
        where_clause = "group_id = ?"
        params = (group_id,)
        
        if before is not None:
            where_clause += " AND id < ?"
            params += (before,)
        elif after is not None:
            where_clause += " AND id > ?"
            params += (after,)
        
        return await self.query(
            where=where_clause,
            params=params,
            order_by="id DESC",
            limit=limit
        )
    


class PrivateChatRecordManager(TableManager):
    """私聊消息记录管理器"""
    def __init__(self, db: SQLiteManager):
        super().__init__(db, "private_chat_record")
        self._indexes_created = False
  
    async def ensure_indexes(self) -> None:
        """确保必要的索引存在"""
        if self._indexes_created:
            return
            
        await self.db.connect()
      
        if not await self.db.index_exists(self.table_name, "idx_user_id_id"):
            await self.db.create_index(
                self.table_name,
                ["user_id", "id"],
                index_name="idx_user_id_id"
            )
      
        if not await self.db.index_exists(self.table_name, "idx_user_id_send_time"):
            await self.db.create_index(
                self.table_name,
                ["user_id", "send_time"],
                index_name="idx_user_id_send_time"
            )
        
        self._indexes_created = True
    
    async def optimize_indexes(self) -> None:
        """优化索引配置"""
        suggested_indexes: Dict = {
            "idx_user_id_id": ["user_id", "id"],
            "idx_user_id_send_time": ["user_id", "send_time"],
            "idx_send_time": ["send_time"]
        }
        await self.db.optimize_indexes_for_table(self.table_name, suggested_indexes)
    
    async def insert_record(
        self,
        msg_id: int,
        user_id: int,
        content: str,
        send_time: float,
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None
    ) -> Optional[int]:
        """插入私聊消息记录"""
        if not self._indexes_created:
            await self.ensure_indexes()
        return await self.insert(
            msg_id=msg_id,
            user_id=user_id,
            content=content,
            send_time=send_time,
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
    
    async def get_by_msg_id(self, msg_id: int) -> Optional[Dict]:
        """通过消息ID获取私聊消息记录"""
        if not self._indexes_created:
            await self.ensure_indexes()
        return await self.get(id=msg_id)
    
    async def get_recent_records(
        self,
        user_id: int,
        limit: int = 10,
        before: Optional[int] = None,
        after: Optional[int] = None
    ) -> List[Dict]:
        """
        获取用户最近的私聊消息记录
        :param user_id: 用户ID
        :param limit: 获取的消息数量
        :param before: 获取早于指定ID的消息
        :param after: 获取晚于指定ID的消息
        """
        if not self._indexes_created:
            await self.ensure_indexes()
        where_clause = "user_id = ?"
        params = (user_id,)
        
        if before is not None:
            where_clause += " AND id < ?"
            params += (before,)
        elif after is not None:
            where_clause += " AND id > ?"
            params += (after,)
        
        return await self.query(
            where=where_clause,
            params=params,
            order_by="id DESC",
            limit=limit
        )
    


chat_record_db = SQLiteManager(gcm.message_handling.chat_record_db_path)
image_description_cache_db = SQLiteManager(gcm.image_recognition.cache_db_path)
group_chat_manager = GroupChatRecordManager(chat_record_db)
private_chat_manager = PrivateChatRecordManager(chat_record_db)



@get_driver().on_shutdown
async def on_shutdown():
    await chat_record_db.close()
    await image_description_cache_db.close()





# 异步使用示例
async def main():
    async with SQLiteManager("example.db") as db:
        # 创建示例表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                category TEXT,
                author TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                views INTEGER DEFAULT 0
            )
        """)
        
        # 获取文章表管理器
        articles = db.table("articles")
        
        # 插入示例数据
        categories = ["技术", "生活", "科技", "艺术"]
        authors = ["Alice", "Bob", "Charlie", "Diana"]
        
        for i in range(1, 51):
            category = categories[i % len(categories)]
            author = authors[i % len(authors)]
            await articles.insert(
                title=f"文章标题 {i}",
                content=f"这是第 {i} 篇文章的内容",
                category=category,
                author=author,
                views=i * 10  # 模拟浏览量
            )
        
        # 示例1：获取ID为25的文章附近的内容
        print("\n示例1：获取ID为25的文章附近的内容（仅标题和分类）")
        target_id = 25
        nearby_rows = await articles.get_nearby_rows(
            target_id,
            before=2,
            after=2,
            columns="id, title, category"
        )
        for row in nearby_rows:
            print(f"ID: {row['id']}, 标题: {row['title']}, 分类: {row['category']}")
        
        # 示例2：在"技术"分类中获取附近行
        print("\n示例2：在'技术'分类中获取附近行")
        tech_nearby = await articles.get_nearby_rows(
            target_id,
            before=3,
            after=3,
            where="category = ?",
            where_params=("技术",),
            columns="id, title, category"
        )
        for row in tech_nearby:
            print(f"ID: {row['id']}, 标题: {row['title']}, 分类: {row['category']}")
        
        # 示例3：按浏览量排序获取附近行
        print("\n示例3：按浏览量排序获取附近行")
        views_nearby = await articles.get_nearby_rows(
            target_id,
            before=3,
            after=3,
            order_column="views",
            columns="id, title, views"
        )
        for row in views_nearby:
            print(f"ID: {row['id']}, 标题: {row['title']}, 浏览量: {row['views']}")
        
        # 示例4：在特定作者的文章中获取上一篇/下一篇
        print("\n示例4：在特定作者的文章中获取上一篇/下一篇")
        author = "Bob"
        prev_article = await articles.get_previous_row(
            target_id,
            where="author = ?",
            where_params=(author,),
            columns="id, title, author"
        )
        next_article = await articles.get_next_row(
            target_id,
            where="author = ?",
            where_params=(author,),
            columns="id, title, author"
        )
        
        print(f"当前文章作者: {author}")
        print(f"上一篇: {prev_article['title'] if prev_article else '无'}")
        print(f"下一篇: {next_article['title'] if next_article else '无'}")
        
        # 示例5：使用窗口函数版本（需要SQLite 3.25+）
        try:
            print("\n示例5：使用窗口函数获取筛选后的附近行")
            window_nearby = await articles.get_nearby_rows_window(
                target_id,
                before=3,
                after=3,
                where="views > 200",
                where_params=(),
                columns="id, title, views"
            )
            for row in window_nearby:
                print(f"ID: {row['id']}, 标题: {row['title']}, 浏览量: {row['views']}")
        except Exception as e:
            print(f"窗口函数出错: {e}")
        
        # 示例6：测试写入锁机制
        print("\n示例6：测试写入锁机制")
        
        # 创建并发写入任务
        async def concurrent_insert(task_id):
            try:
                print(f"任务 {task_id} 尝试写入...")
                await articles.insert(
                    title=f"并发文章 {task_id}",
                    content="并发写入测试",
                    category="测试",
                    author="并发测试员",
                    wait=True,  # 等待锁释放
                    acquire_timeout=3.0,  # 获取锁超时3秒
                    write_timeout=2.0  # 写入操作超时2秒
                )
                print(f"任务 {task_id} 写入成功")
            except Exception as e:
                print(f"任务 {task_id} 写入失败: {str(e)}")
        
        # 启动多个并发写入任务
        tasks = [asyncio.create_task(concurrent_insert(i)) for i in range(1, 6)]
        await asyncio.gather(*tasks)
        
        # 示例7：非阻塞写入
        print("\n示例7：测试非阻塞写入")
        try:
            # 模拟长时间写入
            async def long_write():
                print("开始长时间写入...")
                await articles.insert(
                    title="长时间写入测试",
                    content="这个写入需要5秒",
                    category="测试",
                    author="长时间测试员",
                    wait=True,
                    write_timeout=10.0
                )
                await asyncio.sleep(5)  # 模拟长时间操作
                print("长时间写入完成")
            
            # 启动长时间写入任务
            long_task = asyncio.create_task(long_write())
            
            # 等待写入开始
            await asyncio.sleep(0.1)
            
            # 尝试非阻塞写入
            print("尝试非阻塞写入...")
            try:
                await articles.insert(
                    title="非阻塞写入测试",
                    content="应该失败",
                    category="测试",
                    author="非阻塞测试员",
                    wait=False
                )
            except TimeoutError as e:
                print(f"非阻塞写入失败（符合预期）: {str(e)}")
            
            # 等待长时间写入完成
            await long_task
        except Exception as e:
            print(f"非阻塞写入测试出错: {str(e)}")
        
        # 示例8：测试带超时的关闭
        print("\n示例8：测试带超时的关闭")
        try:
            # 启动长时间写入
            async def blocking_write():
                print("开始阻塞写入...")
                await articles.insert(
                    title="阻塞写入测试",
                    content="这个写入会阻塞关闭操作",
                    category="测试",
                    author="关闭测试员",
                    wait=True
                )
                await asyncio.sleep(10)  # 长时间操作
                print("阻塞写入完成")
            
            write_task = asyncio.create_task(blocking_write())
            await asyncio.sleep(0.1)  # 确保写入开始
            
            # 尝试带超时的关闭
            print("尝试关闭数据库（等待2秒）...")
            await db.close(wait=True, timeout=2.0)
            print("数据库关闭成功")
        except TimeoutError as e:
            print(f"关闭超时（符合预期）: {str(e)}")
            # 强制关闭
            print("强制关闭数据库...")
            await db.close(wait=False)
            print("数据库已强制关闭")
        except Exception as e:
            print(f"关闭出错: {str(e)}")
        finally:
            # 确保任务完成
            if 'write_task' in locals() and not write_task.done():
                write_task.cancel()



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

