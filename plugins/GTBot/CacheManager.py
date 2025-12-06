import asyncio
import json
import time
from typing import Any, Dict, List, Tuple

from nonebot.adapters.onebot.v11 import Bot
from sqlalchemy import select, delete

from .ConfigManager import total_config
from .Logger import logger
from .model import (
    CachedGroupInfo,
    CachedGroupMemberInfo,
    CachedStrangerInfo,
    GroupInfo,
    GroupMemberInfo,
    StrangerInfo,
    async_session_maker,
    init_all_tables,
)


class UserCacheManager:
    """管理 QQ 用户相关接口的缓存生命周期。"""

    def __init__(self) -> None:
        """根据配置初始化刷新/过期策略与数据库句柄。"""
        config = total_config.processed_configuration.config
        self.update_interval = getattr(
            config, "user_cache_update_interval_sec", 3600
        )
        self.expire_interval = getattr(config, "user_cache_expire_sec", 86400)
        self._session_maker = async_session_maker
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._bot_self_id: int | None = None
        """bot 自身的 QQ 号"""

    async def _ensure_initialized(self) -> None:
        """懒加载方式创建数据库表结构。"""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await init_all_tables()
            self._initialized = True

    async def ensure_ready(self) -> None:
        """对外公开的初始化接口。"""
        await self._ensure_initialized()

    async def set_bot_self_id(self, bot: Bot) -> None:
        """设置 bot 自身信息并将其加入缓存。

        Args:
            bot: 已连接的 OneBot V11 Bot。
        """
        self._bot_self_id = int(bot.self_id)
        await self._ensure_initialized()
        
        # 将 bot 本身的信息作为陌生人信息缓存
        try:
            await self.get_stranger_info(bot, self._bot_self_id)
            logger.info(f"bot 信息已加入缓存: {self._bot_self_id}")
        except Exception as exc:
            logger.warning(f"缓存 bot 信息失败: {exc}")

    @staticmethod
    def _serialize(data: Dict[str, Any]) -> str:
        """将 QQ 接口返回的数据转为持久化字符串。"""
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize(data_str: str) -> Dict[str, Any]:
        """将数据库中的 JSON 字符串还原为原始字典。"""
        return json.loads(data_str)

    def _is_stale(self, last_update: float, now_ts: float) -> bool:
        """判断缓存是否超过刷新间隔。"""
        return now_ts - last_update >= self.update_interval

    async def get_group_info(
        self, bot: Bot, group_id: int, force_refresh: bool = False
    ) -> GroupInfo:
        """获取群信息，必要时刷新缓存。

        Args:
            bot: 已连接的 OneBot V11 Bot。
            group_id: 目标群号。
            force_refresh: 是否忽略缓存直接刷新。
        
        Returns:
            GroupInfo: 群信息 Pydantic 对象
        """
        await self._ensure_initialized()
        now = time.time()
        async with self._session_maker() as session:
            # 优先命中本地缓存，缺失时立即创建空记录方便后续字段统一更新
            cache = await session.get(CachedGroupInfo, group_id)
            needs_refresh = force_refresh
            if cache is None:
                cache = CachedGroupInfo(group_id=group_id)
                session.add(cache)
                needs_refresh = True
            elif self._is_stale(cache.last_update_time, now):
                needs_refresh = True

            if needs_refresh:
                data = await bot.get_group_info(group_id=group_id, no_cache=True)
                cache.data = self._serialize(data)
                cache.last_update_time = now

            cache.last_access_time = now
            await session.commit()
            await session.refresh(cache)
            return cache.to_pydantic()

    async def get_group_member_info(
        self, bot: Bot, group_id: int, user_id: int, force_refresh: bool = False
    ) -> GroupMemberInfo:
        """获取指定群成员信息，必要时刷新缓存。

        Args:
            bot: OneBot V11 Bot。
            group_id: 群号。
            user_id: 成员 QQ。
            force_refresh: 是否强制刷新。
        
        Returns:
            GroupMemberInfo: 群成员信息 Pydantic 对象
        """
        await self._ensure_initialized()
        now = time.time()
        async with self._session_maker() as session:
            # 群成员缓存需要根据 (group_id, user_id) 联合定位
            stmt = select(CachedGroupMemberInfo).where(
                CachedGroupMemberInfo.group_id == group_id,
                CachedGroupMemberInfo.user_id == user_id,
            )
            result = await session.execute(stmt)
            cache = result.scalar_one_or_none()

            needs_refresh = force_refresh
            if cache is None:
                cache = CachedGroupMemberInfo(group_id=group_id, user_id=user_id)
                session.add(cache)
                needs_refresh = True
            elif self._is_stale(cache.last_update_time, now):
                needs_refresh = True

            if needs_refresh:
                data = await bot.get_group_member_info(
                    group_id=group_id, user_id=user_id, no_cache=True
                )
                cache.data = self._serialize(data)
                cache.last_update_time = now

            cache.last_access_time = now
            await session.commit()
            await session.refresh(cache)
            return cache.to_pydantic()

    async def get_stranger_info(
        self, bot: Bot, user_id: int, force_refresh: bool = False
    ) -> StrangerInfo:
        """获取陌生人信息，必要时刷新缓存。

        Args:
            bot: OneBot V11 Bot。
            user_id: QQ 号。
            force_refresh: 是否跳过缓存。
        
        Returns:
            StrangerInfo: 陌生人信息 Pydantic 对象
        """
        await self._ensure_initialized()
        now = time.time()
        async with self._session_maker() as session:
            # 陌生人缓存只依赖 user_id，逻辑与群缓存保持一致
            cache = await session.get(CachedStrangerInfo, user_id)
            needs_refresh = force_refresh
            if cache is None:
                cache = CachedStrangerInfo(user_id=user_id)
                session.add(cache)
                needs_refresh = True
            elif self._is_stale(cache.last_update_time, now):
                needs_refresh = True

            if needs_refresh:
                data = await bot.get_stranger_info(user_id=user_id, no_cache=True)
                cache.data = self._serialize(data)
                cache.last_update_time = now

            cache.last_access_time = now
            await session.commit()
            await session.refresh(cache)
            return cache.to_pydantic()

    async def refresh_due_entries(self, bot: Bot) -> None:
        """批量刷新超时的缓存记录。"""
        await self._ensure_initialized()
        cutoff = time.time() - self.update_interval
        group_ids = await self._collect_stale_group_ids(cutoff)
        member_pairs = await self._collect_stale_member_pairs(cutoff)
        stranger_ids = await self._collect_stale_strangers(cutoff)

        for group_id in group_ids:
            try:
                await self.get_group_info(bot, group_id, force_refresh=True)
            except Exception as exc:
                logger.warning(f"刷新群信息缓存失败 group_id={group_id}: {exc}")

        for group_id, user_id in member_pairs:
            try:
                await self.get_group_member_info(
                    bot, group_id, user_id, force_refresh=True
                )
            except Exception as exc:
                logger.warning(
                    f"刷新群成员缓存失败 group_id={group_id}, user_id={user_id}: {exc}"
                )

        for user_id in stranger_ids:
            try:
                await self.get_stranger_info(bot, user_id, force_refresh=True)
            except Exception as exc:
                logger.warning(f"刷新陌生人缓存失败 user_id={user_id}: {exc}")

    @staticmethod
    def _rowcount(result: Any) -> int:
        """兼容 SQLAlchemy Result 不同版本的 `rowcount` 表现。"""
        value = getattr(result, "rowcount", None)
        return int(value) if isinstance(value, int) else 0

    async def cleanup_expired_entries(self) -> None:
        """删除长期未访问的缓存记录。"""
        await self._ensure_initialized()
        cutoff = time.time() - self.expire_interval
        async with self._session_maker() as session:
            group_res = await session.execute(
                delete(CachedGroupInfo).where(
                    CachedGroupInfo.last_access_time <= cutoff
                )
            )
            member_res = await session.execute(
                delete(CachedGroupMemberInfo).where(
                    CachedGroupMemberInfo.last_access_time <= cutoff
                )
            )
            stranger_res = await session.execute(
                delete(CachedStrangerInfo).where(
                    CachedStrangerInfo.last_access_time <= cutoff
                )
            )
            await session.commit()

        removed = {
            "group": self._rowcount(group_res),
            "member": self._rowcount(member_res),
            "stranger": self._rowcount(stranger_res),
        }
        logger.info(f"用户缓存清理完成: {removed}")

    async def _collect_stale_group_ids(self, cutoff: float) -> List[int]:
        """收集需要刷新的群号列表。"""
        async with self._session_maker() as session:
            stmt = select(CachedGroupInfo.group_id).where(
                CachedGroupInfo.last_update_time <= cutoff
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def _collect_stale_member_pairs(self, cutoff: float) -> List[Tuple[int, int]]:
        """收集需要刷新的群成员键。"""
        async with self._session_maker() as session:
            stmt = select(
                CachedGroupMemberInfo.group_id, CachedGroupMemberInfo.user_id
            ).where(CachedGroupMemberInfo.last_update_time <= cutoff)
            result = await session.execute(stmt)
            return [(row[0], row[1]) for row in result.all()]

    async def _collect_stale_strangers(self, cutoff: float) -> List[int]:
        """收集需要刷新的陌生人用户 ID 列表。"""
        async with self._session_maker() as session:
            stmt = select(CachedStrangerInfo.user_id).where(
                CachedStrangerInfo.last_update_time <= cutoff
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def get_group_name(
        self, bot: Bot, group_id: int, force_refresh: bool = False
    ) -> str:
        """快捷获取群名称。

        Args:
            bot: 已连接的 OneBot V11 Bot。
            group_id: 群号。
            force_refresh: 是否强制刷新缓存。

        Returns:
            str: 群名称，失败时返回空字符串。
        """
        try:
            info = await self.get_group_info(bot, group_id, force_refresh)
            return info.group_name or str(group_id)
        except Exception as exc:
            logger.warning(f"获取群名称失败 group_id={group_id}: {exc}")
            return str(group_id)

    async def get_user_name(
        self, bot: Bot, user_id: int, force_refresh: bool = False
    ) -> str:
        """快捷获取用户名称。

        Args:
            bot: 已连接的 OneBot V11 Bot。
            user_id: 用户 QQ 号。
            force_refresh: 是否强制刷新缓存。

        Returns:
            str: 用户昵称，失败时返回空字符串。
        """
        try:
            info = await self.get_stranger_info(bot, user_id, force_refresh)
            return info.nickname or str(user_id)
        except Exception as exc:
            logger.warning(f"获取用户名称失败 user_id={user_id}: {exc}")
            return str(user_id)

    async def get_group_member_name(
        self, bot: Bot, group_id: int, user_id: int, force_refresh: bool = False
    ) -> str:
        """快捷获取群成员名称。

        Args:
            bot: 已连接的 OneBot V11 Bot。
            group_id: 群号。
            user_id: 成员 QQ 号。
            force_refresh: 是否强制刷新缓存。

        Returns:
            str: 群成员名称（优先群名片，其次昵称），失败时返回用户 ID。
        """
        try:
            info = await self.get_group_member_info(
                bot, group_id, user_id, force_refresh
            )
            return info.display_name or str(user_id)
        except Exception as exc:
            logger.warning(
                f"获取群成员名称失败 group_id={group_id}, user_id={user_id}: {exc}"
            )
            return str(user_id)


user_cache_manager = UserCacheManager()

async def get_user_cache_manager() -> UserCacheManager:
    """NoneBot 依赖注入接口。"""
    await user_cache_manager.ensure_ready()
    return user_cache_manager