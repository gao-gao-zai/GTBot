import time
import asyncio
from pathlib import Path
from tkinter import NO
from typing import List, Optional, Union, Dict, Any

from click import group
from sqlalchemy import select, String, INTEGER, BOOLEAN, FLOAT, update, delete, desc, asc, and_, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from pydantic import BaseModel, Field


try:
    from .model import Base, GroupMessages, GroupMessage
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from model import Base, GroupMessages, GroupMessage


class GroupMessageManager:
    """
    群聊消息管理器 (SQLAlchemy AsyncIO 版)
    """

    def __init__(self, engine: AsyncEngine):
        """
        初始化管理器
        
        Args:
            engine: SQLAlchemy AsyncEngine 实例
        """
        self.engine = engine
        self.session_maker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def create_tables(self):
        """创建数据库表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_message(
        self, 
        msg: Optional[GroupMessage] = None, 
        **kwargs
    ) -> GroupMessage:
        """
        添加一条消息
        
        Args:
            msg: Pydantic 模型对象
            **kwargs: 如果未提供 msg，则使用关键字参数 (message_id, group_id, user_id, etc.)
            
        Returns:
            GroupMessage: 保存后的 Pydantic 对象（包含生成的 db_id）
        """
        if msg:
            data = msg.model_dump(exclude={'db_id'})
        else:
            # 检查必要参数
            required = {'message_id', 'user_id'}
            if not required.issubset(kwargs.keys()):
                raise ValueError(f"缺少必要参数: {required - kwargs.keys()}")
            data = kwargs
            # 设置默认值
            if 'send_time' not in data:
                data['send_time'] = time.time()
            if 'group_id' not in data:
                data['group_id'] = 0

        orm_msg = GroupMessages(**data)

        async with self.session_maker() as session:
            async with session.begin():
                session.add(orm_msg)
                await session.flush() # 获取自增ID
                # 提交事务由 session.begin() 上下文自动处理
                
            # 刷新对象以确保数据同步
            await session.refresh(orm_msg)
            return orm_msg.to_pydantic()

    async def get_message_by_msg_id(self, message_id: int) -> GroupMessage:
        """根据平台消息ID获取消息"""
        async with self.session_maker() as session:
            stmt = select(GroupMessages).where(GroupMessages.message_id == message_id)
            result = await session.execute(stmt)
            orm_msg = result.scalar_one_or_none()
            
            if not orm_msg:
                raise ValueError(f"消息ID {message_id} 不存在")
            return orm_msg.to_pydantic()

    async def get_message_by_db_id(self, db_id: int) -> GroupMessage:
        """根据数据库ID获取消息"""
        async with self.session_maker() as session:
            stmt = select(GroupMessages).where(GroupMessages.id == db_id)
            result = await session.execute(stmt)
            orm_msg = result.scalar_one_or_none()
            
            if not orm_msg:
                raise ValueError(f"数据库ID {db_id} 不存在")
            return orm_msg.to_pydantic()

    async def get_recent_messages(
        self, 
        limit: int = 10, 
        group_id: Optional[int] = None, 
        user_id: Optional[int] = None,
        asc_order: bool = False
    ) -> List[GroupMessage]:
        """
        获取最近的消息
        
        Args:
            limit: 数量限制
            group_id: 筛选群号
            user_id: 筛选用户
            asc_order: 是否按时间正序排列 (默认False，即最新的在前面)
        """
        async with self.session_maker() as session:
            stmt = select(GroupMessages)
            
            # 构建查询条件
            conditions = []
            if group_id is not None:
                conditions.append(GroupMessages.group_id == group_id)
            if user_id is not None:
                conditions.append(GroupMessages.user_id == user_id)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            # 排序
            if asc_order:
                stmt = stmt.order_by(asc(GroupMessages.id))
            else:
                stmt = stmt.order_by(desc(GroupMessages.id))
                
            stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_pydantic() for row in rows]

    async def get_nearby_messages(
        self,
        target_msg: Optional[GroupMessage] = None,
        db_id: Optional[int] = None,
        message_id: Optional[int] = None,
        group_id: Optional[int] = None,
        user_id: Optional[int] = None,
        before: int = 10,
        after: int = 0,
        include_target: bool = True
    ) -> List[GroupMessage]:
        """
        获取上下文消息
        
        Args:
            target_msg/db_id/message_id: 定位目标消息 (三选一)
            group_id/user_id: 额外的筛选条件 (通常需要传入 group_id 以保证上下文在同一个群)
            before: 获取目标之前的条数
            after: 获取目标之后的条数
        """
        # 1. 确定目标消息的 DB ID
        target_db_id = None
        if target_msg:
            target_db_id = target_msg.db_id
        elif db_id:
            target_db_id = db_id
        elif message_id:
            try:
                msg = await self.get_message_by_msg_id(message_id)
                target_db_id = msg.db_id
            except ValueError:
                raise ValueError("未找到目标消息")
        
        if target_db_id is None:
            raise ValueError("必须提供定位目标消息的参数")

        async with self.session_maker() as session:
            # 基础筛选条件
            base_conditions = []
            if group_id is not None:
                base_conditions.append(GroupMessages.group_id == group_id)
            if user_id is not None:
                base_conditions.append(GroupMessages.user_id == user_id)

            # 2. 获取之前的消息 (ID < target, 倒序取 limit)
            prev_msgs = []
            if before > 0:
                stmt_prev = select(GroupMessages).where(
                    and_(GroupMessages.id < target_db_id, *base_conditions)
                ).order_by(desc(GroupMessages.id)).limit(before)
                res_prev = await session.execute(stmt_prev)
                # 结果是倒序的 (target-1, target-2...), 需要反转回正序
                prev_msgs = list(reversed(res_prev.scalars().all()))

            # 3. 获取之后的消息 (ID > target, 正序取 limit)
            next_msgs = []
            if after > 0:
                stmt_next = select(GroupMessages).where(
                    and_(GroupMessages.id > target_db_id, *base_conditions)
                ).order_by(asc(GroupMessages.id)).limit(after)
                res_next = await session.execute(stmt_next)
                next_msgs = list(res_next.scalars().all())

            # 4. 获取目标消息本身 (如果需要)
            target_list = []
            if include_target:
                stmt_target = select(GroupMessages).where(GroupMessages.id == target_db_id)
                res_target = await session.execute(stmt_target)
                target_obj = res_target.scalar_one_or_none()
                if target_obj:
                    target_list = [target_obj]

            # 5. 组合结果
            combined = prev_msgs + target_list + next_msgs
            return [m.to_pydantic() for m in combined]

    async def update_message(
        self,
        identify_by_msg_id: Optional[int] = None,
        identify_by_db_id: Optional[int] = None,
        **update_fields
    ):
        """
        更新消息
        
        Args:
            identify_by_msg_id: 通过 message_id 定位
            identify_by_db_id: 通过 db_id 定位
            **update_fields: 要更新的字段 (content, is_withdrawn 等)
        """
        if not update_fields:
            return

        stmt = update(GroupMessages)
        
        if identify_by_db_id is not None:
            stmt = stmt.where(GroupMessages.id == identify_by_db_id)
        elif identify_by_msg_id is not None:
            stmt = stmt.where(GroupMessages.message_id == identify_by_msg_id)
        else:
            raise ValueError("必须提供 message_id 或 db_id 以定位要更新的消息")

        stmt = stmt.values(**update_fields)

        async with self.session_maker() as session:
            async with session.begin():
                result = await session.execute(stmt)
                if result.rowcount == 0: # type: ignore
                    raise ValueError("未找到要更新的消息")

    async def delete_message(
        self, 
        message_id: Optional[int] = None, 
        db_id: Optional[int] = None
    ):
        """删除消息"""
        stmt = delete(GroupMessages)
        
        if db_id is not None:
            stmt = stmt.where(GroupMessages.id == db_id)
        elif message_id is not None:
            stmt = stmt.where(GroupMessages.message_id == message_id)
        else:
            raise ValueError("必须提供 message_id 或 db_id 以定位要删除的消息")

        async with self.session_maker() as session:
            async with session.begin():
                result = await session.execute(stmt)
                if result.rowcount == 0: # type: ignore
                    raise ValueError("未找到要删除的消息")

    async def mark_message_withdrawn(
        self,
        message_id: int
    ) -> None:
        """
        将消息标记为已撤回。
        
        Args:
            message_id: 要标记为撤回的消息 ID
            
        Raises:
            ValueError: 当未找到消息时抛出
        """
        await self.update_message(
            identify_by_msg_id=message_id,
            is_withdrawn=True
        )

NONEBOT_ENV: bool = False
try:
    from nonebot import get_driver
    NONEBOT_ENV = True
except ImportError:
    NONEBOT_ENV = False

if NONEBOT_ENV:
    from nonebot import get_driver
    from .ConfigManager import total_config
    
    # 添加异步锁和初始化标志
    _init_lock = asyncio.Lock()
    _manager_ready = asyncio.Event()
    
    @get_driver().on_startup
    async def _init_message_manager():
        """NoneBot 启动时初始化消息管理器"""
        global message_manager
        async with _init_lock:
            DATA_DIR = total_config.processed_configuration.config.data_dir_path
            DB_PATH = DATA_DIR / "data.db"
            ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"
            engine = create_async_engine(ASYNC_DB_URL, echo=False)
            message_manager = GroupMessageManager(engine)
            await message_manager.create_tables()
            _manager_ready.set()  # 标记初始化完成

    # 创建依赖注入函数
    async def get_message_manager() -> GroupMessageManager:
        """获取消息管理器实例 (依赖注入)"""
        await _manager_ready.wait()  # 等待初始化完成
        return message_manager