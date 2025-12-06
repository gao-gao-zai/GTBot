
import json
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from nonebot import logger
from sqlalchemy import select, String, INTEGER, BOOLEAN, FLOAT, update, delete, desc, asc, and_, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from pydantic import BaseModel, Field

from nonebot.adapters.onebot.v11 import Bot
from nonebot import get_driver

from .model import UserProfile, GroupProfile, UserProfileModel, GroupProfileModel, engine
from .ConfigManager import total_config
config = total_config.processed_configuration.current_config_group.user_profile

class UserProfileManager:
    """用户资料管理器，负责用户资料的增删改查操作。"""
    
    def __init__(self, engine: AsyncEngine) -> None:
        """初始化用户资料管理器。
        
        Args:
            engine: 异步数据库引擎。
        """
        self._engine: AsyncEngine = engine
        self._session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """创建数据库表。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(UserProfileModel.metadata.create_all)
            await conn.run_sync(GroupProfileModel.metadata.create_all)

    def _validate_descriptions(self, descriptions: list[str]) -> None:
        """验证描述列表的有效性。
        
        Args:
            descriptions: 描述列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        for desc_item in descriptions:
            if len(desc_item) > config.max_description_char_length:
                raise ValueError(
                    f"单条描述长度超过限制（{len(desc_item)} > {config.max_description_char_length}）"
                )
        if len(descriptions) > config.max_descriptions:
            raise ValueError(
                f"用户画像描述条数超过限制（{len(descriptions)} > {config.max_descriptions}）"
            )

    def _parse_description_data(self, data_str: str, user_id: int) -> list[str]:
        """解析描述数据字符串。
        
        Args:
            data_str: JSON格式的描述数据字符串。
            user_id: 用户ID，用于日志记录。
            
        Returns:
            解析后的描述列表。
        """
        try:
            data = json.loads(data_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        logger.error(f"用户 {user_id} 的描述数据格式错误")
        return []

    async def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """获取用户资料。
        
        Args:
            user_id: 用户ID。
            
        Returns:
            用户资料对象，如果不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            if user_model:
                return UserProfile(
                    user_id=user_model.user_id,
                    description=self._parse_description_data(user_model.data, user_id)
                )
            return None

    async def get_user_descriptions_with_index(self, user_id: int) -> Optional[Dict[int, str]]:
        """获取带有序号的用户描述。
        
        Args:
            user_id: 用户ID。
            
        Returns:
            带序号的描述字典，键为序号（从1开始），值为描述内容。
            如果用户不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            if user_model:
                descriptions = self._parse_description_data(user_model.data, user_id)
                return {i + 1: desc_item for i, desc_item in enumerate(descriptions)}
            return None

    async def edit_user_description_by_index(self, user_id: int, index: int, new_description: str) -> None:
        """编辑用户对应序号的描述。
        
        Args:
            user_id: 用户ID。
            index: 描述序号（从1开始）。
            new_description: 新的描述内容。
            
        Raises:
            ValueError: 当用户不存在、序号无效或描述超过长度限制时抛出。
        """
        if len(new_description) > config.max_description_char_length:
            raise ValueError(
                f"描述长度超过限制（{len(new_description)} > {config.max_description_char_length}）"
            )
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                raise ValueError(f"用户 {user_id} 的资料不存在")
            
            descriptions = self._parse_description_data(user_model.data, user_id)
            
            if index < 1 or index > len(descriptions):
                raise ValueError(f"序号 {index} 无效，有效范围为 1-{len(descriptions)}")
            
            descriptions[index - 1] = new_description
            user_model.data = json.dumps(descriptions)
            await session.commit()

    async def add_user_profile(self, user_id: int, description: Union[list[str], str]) -> None:
        """添加用户资料。
        
        Args:
            user_id: 用户ID。
            description: 描述内容，可以是字符串或字符串列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        descriptions = [description] if isinstance(description, str) else description
        self._validate_descriptions(descriptions)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if user_model:
                existing_data = self._parse_description_data(user_model.data, user_id)
                existing_data.extend(descriptions)
                self._validate_descriptions(existing_data)
                user_model.data = json.dumps(existing_data)
            else:
                session.add(UserProfileModel(user_id=user_id, data=json.dumps(descriptions)))
            
            await session.commit()

    async def delete_user_description_by_index(self, user_id: int, indices: Union[int, list[int]]) -> None:
        """按序号删除用户的指定描述。
        
        Args:
            user_id: 用户ID。
            indices: 要删除的描述序号（从1开始），可以是单个整数或整数列表。
                    删除后其他描述的序号会自动递减，请重新获取后确认新序号。
            
        Raises:
            ValueError: 当用户不存在、序号无效或删除后无描述时抛出。
            
        Note:
            删除描述后，剩余描述的序号会发生变动。例如删除序号2后，
            原来的序号3会变成2，序号4会变成3，依此类推。
            若需要继续操作，请重新调用 get_user_descriptions_with_index() 获取最新序号。
        """
        # 统一转换为列表格式
        delete_indices = [indices] if isinstance(indices, int) else indices
        
        # 验证所有序号都是正整数
        if not all(isinstance(idx, int) and idx > 0 for idx in delete_indices):
            raise ValueError("所有序号必须为正整数")
        
        # 去重并排序（倒序删除以避免序号变动）
        delete_indices = sorted(set(delete_indices), reverse=True)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                raise ValueError(f"用户 {user_id} 的资料不存在")
            
            descriptions = self._parse_description_data(user_model.data, user_id)
            
            # 验证序号有效性
            for idx in delete_indices:
                if idx < 1 or idx > len(descriptions):
                    raise ValueError(f"序号 {idx} 无效，有效范围为 1-{len(descriptions)}")
            
            # 倒序删除以避免序号变动
            for idx in delete_indices:
                descriptions.pop(idx - 1)
            
            if not descriptions:
                raise ValueError(f"删除后用户 {user_id} 没有任何描述")
            
            user_model.data = json.dumps(descriptions)
            await session.commit()

    async def update_user_profile(self, user_id: int, description: list[str]) -> None:
        """更新用户资料。
        
        Args:
            user_id: 用户ID。
            description: 新的描述列表。
            
        Raises:
            ValueError: 当描述超过限制或用户不存在时抛出。
        """
        self._validate_descriptions(description)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            user_model = result.scalar_one_or_none()
            
            if not user_model:
                raise ValueError(f"用户 {user_id} 的资料不存在，无法更新")
            
            user_model.data = json.dumps(description)
            await session.commit()

    async def delete_user_profile(self, user_id: int) -> None:
        """删除用户的所有描述（仅在资料完全删除时使用）。
        
        Args:
            user_id: 用户ID。
            
        Note:
            此方法直接删除整个用户资料。通常不建议使用，
            建议使用 delete_user_description_by_index() 删除指定描述。
        """
        async with self._session_maker() as session:
            await session.execute(
                delete(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            await session.commit()

class GroupProfileManager:
    """群聊资料管理器，负责群聊资料的增删改查操作。"""
    
    def __init__(self, engine: AsyncEngine) -> None:
        """初始化群聊资料管理器。
        
        Args:
            engine: 异步数据库引擎。
        """
        self._engine: AsyncEngine = engine
        self._session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """创建数据库表。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(GroupProfileModel.metadata.create_all)

    def _validate_descriptions(self, descriptions: list[str]) -> None:
        """验证描述列表的有效性。
        
        Args:
            descriptions: 描述列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        for desc_item in descriptions:
            if len(desc_item) > config.max_description_char_length:
                raise ValueError(
                    f"单条描述长度超过限制（{len(desc_item)} > {config.max_description_char_length}）"
                )
        if len(descriptions) > config.max_descriptions:
            raise ValueError(
                f"群聊画像描述条数超过限制（{len(descriptions)} > {config.max_descriptions}）"
            )

    def _parse_description_data(self, data_str: str, group_id: int) -> list[str]:
        """解析描述数据字符串。
        
        Args:
            data_str: JSON格式的描述数据字符串。
            group_id: 群组ID，用于日志记录。
            
        Returns:
            解析后的描述列表。
        """
        try:
            data = json.loads(data_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        logger.error(f"群聊 {group_id} 的描述数据格式错误")
        return []

    async def get_group_profile(self, group_id: int) -> Optional[GroupProfile]:
        """获取群聊资料。
        
        Args:
            group_id: 群组ID。
            
        Returns:
            群聊资料对象，如果不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            if group_model:
                return GroupProfile(
                    group_id=group_model.group_id,
                    description=self._parse_description_data(group_model.data, group_id)
                )
            return None

    async def get_group_descriptions_with_index(self, group_id: int) -> Optional[Dict[int, str]]:
        """获取带有序号的群聊描述。
        
        Args:
            group_id: 群组ID。
            
        Returns:
            带序号的描述字典，键为序号（从1开始），值为描述内容。
            如果群聊不存在则返回None。
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            if group_model:
                descriptions = self._parse_description_data(group_model.data, group_id)
                return {i + 1: desc_item for i, desc_item in enumerate(descriptions)}
            return None

    async def edit_group_description_by_index(self, group_id: int, index: int, new_description: str) -> None:
        """编辑群聊对应序号的描述。
        
        Args:
            group_id: 群组ID。
            index: 描述序号（从1开始）。
            new_description: 新的描述内容。
            
        Raises:
            ValueError: 当群聊不存在、序号无效或描述超过长度限制时抛出。
        """
        if len(new_description) > config.max_description_char_length:
            raise ValueError(
                f"描述长度超过限制（{len(new_description)} > {config.max_description_char_length}）"
            )
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            
            if not group_model:
                raise ValueError(f"群聊 {group_id} 的资料不存在")
            
            descriptions = self._parse_description_data(group_model.data, group_id)
            
            if index < 1 or index > len(descriptions):
                raise ValueError(f"序号 {index} 无效，有效范围为 1-{len(descriptions)}")
            
            descriptions[index - 1] = new_description
            group_model.data = json.dumps(descriptions)
            await session.commit()

    async def add_group_profile(self, group_id: int, description: Union[list[str], str]) -> None:
        """添加群聊资料。
        
        Args:
            group_id: 群组ID。
            description: 描述内容，可以是字符串或字符串列表。
            
        Raises:
            ValueError: 当描述超过长度或数量限制时抛出。
        """
        descriptions = [description] if isinstance(description, str) else description
        self._validate_descriptions(descriptions)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            
            if group_model:
                existing_data = self._parse_description_data(group_model.data, group_id)
                existing_data.extend(descriptions)
                self._validate_descriptions(existing_data)
                group_model.data = json.dumps(existing_data)
            else:
                session.add(GroupProfileModel(group_id=group_id, data=json.dumps(descriptions)))
            
            await session.commit()

    async def delete_group_description_by_index(self, group_id: int, indices: Union[int, list[int]]) -> None:
        """按序号删除群聊的指定描述。
        
        Args:
            group_id: 群组ID。
            indices: 要删除的描述序号（从1开始），可以是单个整数或整数列表。
                    删除后其他描述的序号会自动递减，请重新获取后确认新序号。
            
        Raises:
            ValueError: 当群聊不存在、序号无效或删除后无描述时抛出。
            
        Note:
            删除描述后，剩余描述的序号会发生变动。例如删除序号2后，
            原来的序号3会变成2，序号4会变成3，依此类推。
            若需要继续操作，请重新调用 get_group_descriptions_with_index() 获取最新序号。
        """
        # 统一转换为列表格式
        delete_indices = [indices] if isinstance(indices, int) else indices
        
        # 验证所有序号都是正整数
        if not all(isinstance(idx, int) and idx > 0 for idx in delete_indices):
            raise ValueError("所有序号必须为正整数")
        
        # 去重并排序（倒序删除以避免序号变动）
        delete_indices = sorted(set(delete_indices), reverse=True)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            
            if not group_model:
                raise ValueError(f"群聊 {group_id} 的资料不存在")
            
            descriptions = self._parse_description_data(group_model.data, group_id)
            
            # 验证序号有效性
            for idx in delete_indices:
                if idx < 1 or idx > len(descriptions):
                    raise ValueError(f"序号 {idx} 无效，有效范围为 1-{len(descriptions)}")
            
            # 倒序删除以避免序号变动
            for idx in delete_indices:
                descriptions.pop(idx - 1)
            
            if not descriptions:
                raise ValueError(f"删除后群聊 {group_id} 没有任何描述")
            
            group_model.data = json.dumps(descriptions)
            await session.commit()

    async def delete_group_profile(self, group_id: int) -> None:
        """删除群聊的所有描述（仅在资料完全删除时使用）。
        
        Args:
            group_id: 群组ID。
            
        Note:
            此方法直接删除整个群聊资料。通常不建议使用，
            建议使用 delete_group_description_by_index() 删除指定描述。
        """
        async with self._session_maker() as session:
            await session.execute(
                delete(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            await session.commit()

    async def update_group_profile(self, group_id: int, description: list[str]) -> None:
        """更新群聊资料。
        
        Args:
            group_id: 群组ID。
            description: 新的描述列表。
            
        Raises:
            ValueError: 当描述超过限制或群聊不存在时抛出。
        """
        self._validate_descriptions(description)
        
        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupProfileModel).where(GroupProfileModel.group_id == group_id)
            )
            group_model = result.scalar_one_or_none()
            
            if not group_model:
                raise ValueError(f"群聊 {group_id} 的资料不存在，无法更新")
            
            group_model.data = json.dumps(description)
            await session.commit()

user_profile_manager: Optional[UserProfileManager] = None
group_profile_manager: Optional[GroupProfileManager] = None
_initialized_event = asyncio.Event()

@get_driver().on_startup
async def _initialize_user_profile_manager() -> None:
    """在 NoneBot 启动时初始化用户和群聊资料管理器。"""
    global user_profile_manager, group_profile_manager
    user_profile_manager = UserProfileManager(engine)
    group_profile_manager = GroupProfileManager(engine)
    await user_profile_manager.create_tables()
    await group_profile_manager.create_tables()
    logger.info("用户和群聊资料管理器已初始化并创建数据库表")
    _initialized_event.set()

async def get_user_profile_manager() -> UserProfileManager:
    """获取全局用户资料管理器实例。
    
    等待初始化完成后返回管理器实例。
    
    Returns:
        用户资料管理器实例。
    """
    await _initialized_event.wait()
    return user_profile_manager  # type: ignore[return-value]

async def get_group_profile_manager() -> GroupProfileManager:
    """获取全局群聊资料管理器实例。
    
    等待初始化完成后返回管理器实例。
    
    Returns:
        群聊资料管理器实例。
    """
    await _initialized_event.wait()
    return group_profile_manager  # type: ignore[return-value]


class ProfileManager:
    """用户和群聊资料管理器包装类。
    
    统一管理用户画像和群聊画像，方便作为单个参数传递。
    """
    
    def __init__(self, user_manager: UserProfileManager, group_manager: GroupProfileManager) -> None:
        """初始化资料管理器包装类。
        
        Args:
            user_manager: 用户资料管理器实例。
            group_manager: 群聊资料管理器实例。
        """
        self.user: UserProfileManager = user_manager
        """用户资料管理器"""
        
        self.group: GroupProfileManager = group_manager
        """群聊资料管理器"""


profile_manager: Optional[ProfileManager] = None


@get_driver().on_startup
async def _initialize_profile_manager() -> None:
    """在 NoneBot 启动时初始化全局资料管理器包装类。"""
    global profile_manager
    await _initialized_event.wait()
    if user_profile_manager is not None and group_profile_manager is not None:
        profile_manager = ProfileManager(user_profile_manager, group_profile_manager)
        logger.info("资料管理器包装类已初始化")


async def get_profile_manager() -> ProfileManager:
    """获取全局资料管理器包装类实例。
    
    等待初始化完成后返回管理器实例。
    
    Returns:
        资料管理器包装类实例。
        
    Raises:
        RuntimeError: 当管理器未正确初始化时抛出。
    """
    await _initialized_event.wait()
    if profile_manager is None:
        raise RuntimeError("资料管理器包装类未被正确初始化")
    return profile_manager













