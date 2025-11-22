import time
import asyncio
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from sqlalchemy import select, String, INTEGER, BOOLEAN, FLOAT, update, delete, desc, asc, and_, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from pydantic import BaseModel, Field
from sympy import im

# --- 配置初始化和数据库路径配置 ---
try:
    from .ConfigManager import total_config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from ConfigManager import total_config

# 初始化配置
try:
    DATA_DIR = total_config.processed_configuration.config.data_dir_path
    DB_PATH = DATA_DIR / "data.db"
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"
except Exception as e:
    # 如果配置加载失败，使用默认路径
    print(f"警告：配置加载失败，使用默认数据库路径: {e}")
    DB_PATH = Path("./data.db")
    ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# --- 基础模型 ---
class Base(DeclarativeBase):
    pass

class GroupMessages(Base):
    """群聊消息模型 (ORM)"""
    __tablename__ = "group_messages"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    """数据库自增ID"""
    
    message_id: Mapped[int] = mapped_column(INTEGER, index=True, unique=True) 
    """消息ID (平台侧ID)"""
    
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, default=0)
    """群组ID"""
    
    user_id: Mapped[int] = mapped_column(INTEGER, index=True)
    """用户ID"""
    
    user_name: Mapped[str] = mapped_column(String, default="")
    """用户昵称"""
    
    content: Mapped[str] = mapped_column(String, default="")
    """消息内容"""
    
    send_time: Mapped[float] = mapped_column(FLOAT, index=True, default=0.0)
    """发送时间戳"""
    
    is_withdrawn: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    """是否已被撤回"""

    def to_pydantic(self) -> "GroupMessage":
        """转换为Pydantic模型"""
        return GroupMessage(
            db_id=self.id,
            message_id=self.message_id,
            group_id=self.group_id,
            user_id=self.user_id,
            user_name=self.user_name,
            content=self.content,
            send_time=self.send_time,
            is_withdrawn=self.is_withdrawn
        )

class GroupMessage(BaseModel):
    """群聊消息 (Pydantic)"""
    db_id: Optional[int] = None
    """数据库内部ID"""
    message_id: int
    """消息ID"""
    group_id: int = 0
    """群号"""
    user_id: int
    """发送者QQ号"""
    user_name: str = ""
    """发送者昵称"""
    content: str = ""
    """消息内容"""
    send_time: float = Field(default_factory=lambda: time.time())
    """发送时间"""
    is_withdrawn: bool = False
    """是否撤回"""
