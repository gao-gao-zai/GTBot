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

class GroupMessage:
    db_id: int 
    """数据库ID"""
    group_id: int 
    """群号"""
    user_id: int 
    """发起消息的用户ID"""
    msg_id: int 
    """消息ID"""
    content: str
    """消息内容"""
    send_time: float 
    """发送时间"""
    is_recalled: bool 
    """是否被撤回"""

class PrivateMessage:
    db_id: int # 数据库ID
    user_id: int # 用户ID
    msg_id: int # 消息ID
    content: str # 消息内容
    send_time: float # 发送时间
    is_recalled: bool # 是否被撤回




ID = "id"
"""数据库自增主键列名"""
GROUP_CHAT_RECORD_TABLE_NAME = "group_chat_record"
"""群聊消息记录表名"""
PRIVATE_CHAT_RECORD_TABLE_NAME = "private_chat_record"
"""私聊消息记录表名"""


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
    
    async def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> List[Dict]:
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
              params: tuple[Any, ...] = (), 
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

    async def update(
        self,
        *,
        where: str,
        params: tuple[Any, ...] = (),
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None,
        **data
    ) -> int:
        """
        更新符合条件的记录
        :param where: 更新条件（WHERE子句）
        :param params: 条件参数
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间（秒）
        :param write_timeout: 写入操作超时时间（秒）
        :param data: 要更新的字段键值对
        :return: 受影响的行数
        """
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {where}"
        
        # 合并参数：更新值 + 条件参数
        all_params = tuple(data.values()) + params
        
        cursor = await self.db.execute(
            query,
            all_params,
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
        return cursor.rowcount

    async def delete(
        self,
        *,
        where: str,
        params: tuple[Any, ...] = (),
        wait: bool = True,
        acquire_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None,
    ) -> int:
        """
        删除符合条件的记录
        :param where: 删除条件（WHERE子句）
        :param params: 条件参数
        :param wait: 是否等待锁释放
        :param acquire_timeout: 获取锁的超时时间（秒）
        :param write_timeout: 写入操作超时时间（秒）
        :return: 受影响的行数
        """
        query = f"DELETE FROM {self.table_name} WHERE {where}"
        cursor = await self.db.execute(
            query,
            params,
            wait=wait,
            acquire_timeout=acquire_timeout,
            write_timeout=write_timeout
        )
        return cursor.rowcount




class GroupMessageManager:
    """
    群聊消息管理器
    
    用于管理群聊消息的增删改查操作，提供消息检索、附近消息获取等功能。
    """

    def __init__(
        self, 
        db: str|Path|SQLiteManager = gcm.message_handling.chat_record_db_path, 
        table_name: str = GROUP_CHAT_RECORD_TABLE_NAME
    ):
        """
        初始化群聊消息管理器
        
        Args:
            db_path (str|Path|SQLiteManager): 数据库路径或SQLiteManager实例，
                默认使用gcm.message_handling.chat_record_db_path
            table_name (str): 数据表名称，默认为"group_chat_record"
        """
        if isinstance(db, (str, Path)):
            self.db = SQLiteManager(db_path=db)
        else:
            self.db = db
        self.table = self.db.table(table_name)

    async def get_message_by_msg_id(self, msg_id: int) -> GroupMessage:
        """
        根据消息ID获取群聊消息
        
        Args:
            msg_id (int): 消息ID
            
        Returns:
            GroupMessage: 群聊消息对象
            
        Raises:
            ValueError: 当消息ID不存在时抛出异常
        """
        rows = await self.table.query(where="msg_id = ?", params=(msg_id,))
        if not rows:
            raise ValueError("消息ID不存在")
        row = rows[0]
        msg = self._convert_row_to_group_message(row)
        return msg

    async def get_message_by_db_id(self, id: int) -> GroupMessage:
        """
        根据数据库ID获取群聊消息
        
        Args:
            id (int): 数据库记录ID
            
        Returns:
            GroupMessage: 群聊消息对象
            
        Raises:
            ValueError: 当数据库ID不存在时抛出异常
        """
        rows = await self.table.query(where="id = ?", params=(id,))
        if not rows:
            raise ValueError("数据库ID不存在")
        row = rows[0]
        msg = self._convert_row_to_group_message(row)
        return msg

    def _convert_row_to_group_message(self, row: Dict) -> GroupMessage:
        """
        将数据库行转换为GroupMessage对象
        
        Args:
            row (Dict): 数据库查询结果行
            
        Returns:
            GroupMessage: 转换后的群聊消息对象
        """
        msg = GroupMessage()
        msg.db_id = row["id"]
        msg.msg_id = row["msg_id"]
        msg.group_id = row["group_id"]
        msg.user_id = row["user_id"]
        msg.content = row["content"]
        msg.send_time = row["send_time"]
        msg.is_recalled = row["is_recalled"]
        return msg

    async def get_recent_messages(
        self, 
        limit: int = 10, 
        group_id: Optional[int] = None, 
        user_id: Optional[int] = None,
        order_by: str = f"{ID} DESC"
    ) -> List[GroupMessage]:
        """
        获取最近的消息
        
        Args:
            limit (int): 获取消息数量限制，默认为10
            group_id (Optional[int]): 群组ID，为None时不限制群组
            user_id (Optional[int]): 用户ID，为None时不限制用户
            order_by (str): 排序方式，默认为"id DESC"（按ID降序）
            
        Returns:
            List[GroupMessage]: 群聊消息列表，按指定顺序排列
            
        Note:
            - 当同时提供group_id和user_id时，获取指定群组中指定用户的消息
            - 当只提供group_id时，获取指定群组的所有消息
            - 当只提供user_id时，获取指定用户在所有群组的消息
            - 都不提供时，获取所有群组的所有消息
        """
        if group_id and user_id:
            where_clause = "group_id = ? AND user_id = ?"
            params = (group_id, user_id)
        elif group_id:
            where_clause = "group_id = ?"
            params = (group_id,)
        elif user_id:
            where_clause = "user_id = ?"
            params = (user_id,)
        else:
            where_clause = None
            params = ()

        rows = await self.table.query(
            limit=limit,
            where=where_clause,
            params=params,
            order_by=order_by
        )
        return [self._convert_row_to_group_message(row) for row in rows]

    async def get_nearby_messages(
        self,
        msg: Optional[GroupMessage] = None,
        db_id: Optional[int] = None,
        msg_id: Optional[int] = None,
        group_id: Optional[int] = None,
        user_id: Optional[int] = None,
        include_target: bool = True,
        before: int = 10,
        after: int = 0,
        order_column: str = ID,
        order_direction: str = "ASC",
    ) -> List[GroupMessage]:
        """
        获取某条消息附近的消息
        
        Args:
            msg (Optional[GroupMessage]): 目标消息对象
            db_id (Optional[int]): 目标消息的数据库ID
            msg_id (Optional[int]): 目标消息的消息ID
            group_id (Optional[int]): 限制在指定群组内，为None时不限制
            user_id (Optional[int]): 限制为指定用户的消息，为None时不限制
            before (int): 获取目标消息之前的消息数量，默认为10
            after (int): 获取目标消息之后的消息数量，默认为0
            order_column (str): 排序列名，默认为ID
            order_direction (str): 排序方向，"ASC"或"DESC"，默认为"ASC"
            
        Returns:
            List[GroupMessage]: 目标消息附近的消息列表
            
        Raises:
            ValueError: 当未提供或提供多个目标消息标识符时抛出异常
            ValueError: 当消息ID不存在时抛出异常
            
        Note:
            - 必须提供且仅提供一个目标消息标识符：msg、db_id或msg_id
            - 返回的消息列表包含目标消息(及其)前后的消息
            - 可以通过group_id和user_id进一步筛选结果
        """
        if (msg is not None) + (db_id is not None) + (msg_id is not None) != 1:
            raise ValueError("必须提供且仅提供一个参数：msg, db_id, msg_id")

        if msg:
            db_id = msg.db_id
        elif msg_id:
            db_id = await self._get_db_id_by_msg_id(msg_id)

        if group_id and user_id:
            where_clause = "group_id = ? AND user_id = ?"
            params = (group_id, user_id)
        elif group_id:
            where_clause = "group_id = ?"
            params = (group_id,)
        elif user_id:
            where_clause = "user_id = ?"
            params = (user_id,)
        else:
            where_clause = None
            params = ()

        assert db_id is not None
        
        rows = await self.table.get_nearby_rows(
            target_id=db_id,
            before=before,
            after=after,
            order_column=order_column,
            order_direction=order_direction,
            columns="*",
            where=where_clause,
            where_params=params,
            include_target=include_target
        )
        return [self._convert_row_to_group_message(row) for row in rows]

    async def _get_db_id_by_msg_id(self, msg_id: int) -> int:
        """
        根据消息ID获取数据库ID
        
        Args:
            msg_id (int): 消息ID
            
        Returns:
            int: 对应的数据库记录ID
            
        Raises:
            ValueError: 当消息ID不存在时抛出异常
        """
        rows = await self.table.query(where="msg_id = ?", params=(msg_id,), columns=ID)
        if len(rows) != 1:
            raise ValueError(f"消息ID({msg_id})不存在或不唯一")
        return rows[0][ID]

    async def add_message(
        self, 
        msg: Optional[GroupMessage] = None, 
        msg_id: Optional[int] = None,
        user_id: Optional[int] = None,
        content: Optional[str] = None,
        group_id: Optional[int] = None,
        send_time: Optional[float] = None,
        is_recalled: bool = False,
    ):
        """
        添加一条消息记录

        Args:
            msg (GroupMessage): 要添加的消息对象
            msg_id (Optional[int]): 消息ID
            user_id (Optional[int]): 用户ID
            content (Optional[str]): 消息内容
            group_id (Optional[int]): 群组ID
            send_time (Optional[float]): 发送时间戳
            is_recalled (bool): 是否已撤回，默认为False

        Raises:
            ValueError: 当既没提供消息对象也没提供完整消息信息时抛出
        """
        # 验证参数：必须提供消息对象或完整的消息信息
        if not msg and not (msg_id and user_id and content and group_id and send_time):
            raise ValueError("必须提供消息对象或完整的消息信息")
        
        # 如果提供了消息对象，从中提取各个字段
        if msg:
            msg_id = msg.msg_id
            group_id = msg.group_id
            user_id = msg.user_id
            content = msg.content
            send_time = msg.send_time
            is_recalled = msg.is_recalled

        # 将消息记录插入数据库
        await self.table.insert(
            msg_id=msg_id,
            group_id=group_id,
            user_id=user_id,
            content=content,
            send_time=send_time,
            is_recalled=is_recalled
        )

    async def update_message(
        self,
        old_msg: Optional[GroupMessage] = None,
        old_db_id: Optional[int] = None,
        old_msg_id: Optional[int] = None,
        msg_id: Optional[int] = None,
        user_id: Optional[int] = None,
        content: Optional[str] = None,
        group_id: Optional[int] = None,
        send_time: Optional[float] = None,
        is_recalled: Optional[bool] = None,
    ):
        """
        更新一条消息记录

        Args:
            old_msg (GroupMessage): 要更新的消息对象
            old_db_id (Optional[int]): 旧消息的数据库ID
            old_msg_id (Optional[int]): 旧消息的消息ID
            msg_id (Optional[int]): 新的消息ID
            user_id (Optional[int]): 新的用户ID
            content (Optional[str]): 新的消息内容
            group_id (Optional[int]): 新的群组ID
            send_time (Optional[float]): 新的发送时间戳
            is_recalled (Optional[bool]): 新的撤回状态

        Raises:
            ValueError: 当没有提供定位旧消息的参数或没有要更新的数据时抛出
        """
        # 验证参数：必须提供用于定位旧消息的参数
        if not old_msg and not old_db_id and not old_msg_id:
            raise ValueError("必须提供旧消息对象或数据库ID或消息ID")
        
        # 获取旧消息的数据库ID
        if old_msg:
            old_db_id = old_msg.db_id  # 从消息对象中获取数据库ID
        elif old_msg_id:
            old_db_id = await self._get_db_id_by_msg_id(old_msg_id)  # 通过消息ID查询数据库ID

        # 构建更新数据字典，只包含非None值
        update_data = {}
        if msg_id is not None:
            update_data['msg_id'] = msg_id
        if user_id is not None:
            update_data['user_id'] = user_id
        if content is not None:
            update_data['content'] = content
        if group_id is not None:
            update_data['group_id'] = group_id
        if send_time is not None:
            update_data['send_time'] = send_time
        if is_recalled is not None:
            update_data['is_recalled'] = is_recalled
        
        # 如果没有要更新的数据，抛出异常
        if not update_data:
            raise ValueError("没有要更新的数据")

        # 执行数据库更新操作
        await self.table.update(
            where=f"{ID} = ?",
            params=(old_db_id,),
            **update_data
        )

    async def delete_message(self, msg: Optional[GroupMessage] = None, db_id: Optional[int] = None, msg_id: Optional[int] = None):
        """
        删除一条消息记录

        Args:
            msg (GroupMessage): 要删除的消息对象
            db_id (Optional[int]): 要删除消息的数据库ID
            msg_id (Optional[int]): 要删除消息的消息ID

        Raises:
            ValueError: 当没有提供定位消息的参数时抛出
        """
        # 验证参数：必须提供用于定位要删除消息的参数
        if not msg and not db_id and not msg_id:
            raise ValueError("必须提供消息对象或数据库ID或消息ID")
        
        # 获取要删除消息的数据库ID
        if msg:
            db_id = msg.db_id  # 从消息对象中获取数据库ID
        elif msg_id:
            db_id = await self._get_db_id_by_msg_id(msg_id)  # 通过消息ID查询数据库ID

        # 执行数据库删除操作
        await self.table.delete(where=f"{ID} = ?", params=(db_id,))



chat_record_db = SQLiteManager(db_path=gcm.message_handling.chat_record_db_path)
image_description_cache_db = SQLiteManager(db_path=gcm.image_recognition.cache_db_path)
group_message_manager = GroupMessageManager(chat_record_db)



@get_driver().on_shutdown
async def close_db():
    await chat_record_db.close()
    await image_description_cache_db.close()