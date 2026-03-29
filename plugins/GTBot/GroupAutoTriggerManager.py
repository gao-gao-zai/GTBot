from __future__ import annotations

import asyncio
import time
from enum import Enum

from sqlalchemy import FLOAT, INTEGER, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from .DBmodel import Base, async_session_maker, engine
from .Logger import logger


_DEFAULT_GROUP_ID = 0
_DEFAULT_TRIGGER_PROBABILITY = 100.0
_DEFAULT_COOLDOWN_SECONDS = 3600.0


class GroupAutoTriggerListType(str, Enum):
    """群聊自动触发当前支持的名单类型。"""

    WHITELIST = "whitelist"


class GroupAutoTriggerEntryModel(Base):
    """保存允许自动触发的群白名单条目。"""

    __tablename__ = "group_auto_trigger_entries"
    __table_args__ = (
        UniqueConstraint("group_id", "list_type", name="uq_group_auto_trigger_entry"),
    )

    id: Mapped[int] = mapped_column(INTEGER, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    list_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    created_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupAutoTriggerSettingModel(Base):
    """保存默认值或指定群的自动触发设置。"""

    __tablename__ = "group_auto_trigger_settings"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    probability: Mapped[float | None] = mapped_column(FLOAT, nullable=True, default=None)
    cooldown_seconds: Mapped[float | None] = mapped_column(FLOAT, nullable=True, default=None)
    updated_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    updated_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupAutoTriggerStateModel(Base):
    """保存每个群最近一次自动触发时间。"""

    __tablename__ = "group_auto_trigger_states"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    last_triggered_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    updated_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)


class GroupAutoTriggerManager:
    """统一管理群聊自动触发的白名单、配置和触发状态。"""

    def __init__(
        self,
        *,
        session_maker: async_sessionmaker | None = None,
        engine_obj: AsyncEngine | None = None,
    ) -> None:
        """初始化自动触发管理器。

        Args:
            session_maker: 可选的异步会话工厂，单测可注入独立数据库。
            engine_obj: 可选的异步引擎，单测可注入独立数据库。
        """

        self._session_maker = session_maker or async_session_maker
        self._engine = engine_obj or engine
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_tables(self) -> None:
        """确保自动触发相关数据表已经创建。"""

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self._engine.begin() as conn:
                await conn.run_sync(GroupAutoTriggerEntryModel.metadata.create_all)
            self._initialized = True

    def _normalize_group_id(self, group_id: int | None, *, allow_default: bool = False) -> int:
        """将群号或默认槽位转换为内部存储值。"""

        if group_id is None:
            if allow_default:
                return _DEFAULT_GROUP_ID
            raise ValueError("群号不能为空")

        normalized_group_id = int(group_id)
        if normalized_group_id <= 0:
            raise ValueError(f"无效的群号: {group_id}")
        return normalized_group_id

    def _normalize_list_type(self, list_type: GroupAutoTriggerListType | str) -> GroupAutoTriggerListType:
        """将名单类型标准化为枚举值。"""

        if isinstance(list_type, GroupAutoTriggerListType):
            return list_type
        return GroupAutoTriggerListType(str(list_type).strip().lower())

    def _normalize_probability(self, probability: float | int | str) -> float:
        """把概率输入解析为 0 到 100 之间的浮点数。"""

        raw = str(probability).strip()
        if raw.endswith("%"):
            raw = raw[:-1].strip()
        value = float(raw)
        if value < 0 or value > 100:
            raise ValueError("自动触发概率必须在 0 到 100 之间")
        return round(value, 4)

    def _normalize_cooldown_seconds(self, cooldown_seconds: float | int | str) -> float:
        """把冷却时间输入解析为大于等于 0 的秒数。"""

        value = float(str(cooldown_seconds).strip())
        if value < 0:
            raise ValueError("自动触发冷却时间必须大于等于 0 秒")
        return round(value, 4)

    async def list_entries(
        self,
        list_type: GroupAutoTriggerListType | str = GroupAutoTriggerListType.WHITELIST,
    ) -> list[int]:
        """列出自动触发白名单中的所有群号。"""

        await self._ensure_tables()
        normalized_list_type = self._normalize_list_type(list_type)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupAutoTriggerEntryModel.group_id)
                .where(GroupAutoTriggerEntryModel.list_type == normalized_list_type.value)
                .order_by(GroupAutoTriggerEntryModel.group_id.asc())
            )
            return [int(group_id) for group_id in result.scalars().all()]

    async def add_entry(
        self,
        *,
        group_id: int,
        operator_user_id: int,
        list_type: GroupAutoTriggerListType | str = GroupAutoTriggerListType.WHITELIST,
    ) -> bool:
        """将群加入自动触发白名单。"""

        await self._ensure_tables()
        normalized_group_id = self._normalize_group_id(group_id)
        normalized_list_type = self._normalize_list_type(list_type)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupAutoTriggerEntryModel)
                .where(
                    GroupAutoTriggerEntryModel.group_id == normalized_group_id,
                    GroupAutoTriggerEntryModel.list_type == normalized_list_type.value,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return False

            session.add(
                GroupAutoTriggerEntryModel(
                    group_id=normalized_group_id,
                    list_type=normalized_list_type.value,
                    created_at=time.time(),
                    created_by=int(operator_user_id),
                )
            )
            await session.commit()

        logger.info(
            "用户 %s 已将群 %s 加入群聊自动触发白名单",
            int(operator_user_id),
            normalized_group_id,
        )
        return True

    async def remove_entry(
        self,
        *,
        group_id: int,
        operator_user_id: int,
        list_type: GroupAutoTriggerListType | str = GroupAutoTriggerListType.WHITELIST,
    ) -> bool:
        """将群从自动触发白名单中移除。"""

        await self._ensure_tables()
        normalized_group_id = self._normalize_group_id(group_id)
        normalized_list_type = self._normalize_list_type(list_type)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupAutoTriggerEntryModel)
                .where(
                    GroupAutoTriggerEntryModel.group_id == normalized_group_id,
                    GroupAutoTriggerEntryModel.list_type == normalized_list_type.value,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return False

            await session.delete(existing)
            await session.commit()

        logger.info(
            "用户 %s 已将群 %s 从群聊自动触发白名单中移除",
            int(operator_user_id),
            normalized_group_id,
        )
        return True

    async def is_group_enabled(self, group_id: int) -> bool:
        """判断指定群当前是否允许参与自动触发。"""

        normalized_group_id = self._normalize_group_id(group_id)
        whitelist = await self.list_entries()
        return normalized_group_id in set(whitelist)

    async def set_probability(
        self,
        *,
        group_id: int | None,
        probability: float | int | str,
        operator_user_id: int,
    ) -> float:
        """设置默认值或指定群的自动触发概率。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)
        normalized_probability = self._normalize_probability(probability)
        now = time.time()

        async with self._session_maker() as session:
            row = await session.get(GroupAutoTriggerSettingModel, storage_group_id)
            if row is None:
                row = GroupAutoTriggerSettingModel(
                    group_id=storage_group_id,
                    probability=normalized_probability,
                    cooldown_seconds=None,
                    updated_at=now,
                    updated_by=int(operator_user_id),
                )
                session.add(row)
            else:
                row.probability = normalized_probability
                row.updated_at = now
                row.updated_by = int(operator_user_id)
            await session.commit()

        logger.info(
            "用户 %s 已将群聊自动触发概率设置为 %.4f（group_id=%s）",
            int(operator_user_id),
            normalized_probability,
            storage_group_id,
        )
        return normalized_probability

    async def set_cooldown_seconds(
        self,
        *,
        group_id: int | None,
        cooldown_seconds: float | int | str,
        operator_user_id: int,
    ) -> float:
        """设置默认值或指定群的自动触发冷却时间。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)
        normalized_cooldown = self._normalize_cooldown_seconds(cooldown_seconds)
        now = time.time()

        async with self._session_maker() as session:
            row = await session.get(GroupAutoTriggerSettingModel, storage_group_id)
            if row is None:
                row = GroupAutoTriggerSettingModel(
                    group_id=storage_group_id,
                    probability=None,
                    cooldown_seconds=normalized_cooldown,
                    updated_at=now,
                    updated_by=int(operator_user_id),
                )
                session.add(row)
            else:
                row.cooldown_seconds = normalized_cooldown
                row.updated_at = now
                row.updated_by = int(operator_user_id)
            await session.commit()

        logger.info(
            "用户 %s 已将群聊自动触发冷却设置为 %.4f 秒（group_id=%s）",
            int(operator_user_id),
            normalized_cooldown,
            storage_group_id,
        )
        return normalized_cooldown

    async def _get_configured_setting(self, group_id: int | None) -> GroupAutoTriggerSettingModel | None:
        """获取默认值或指定群显式配置的设置行。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)

        async with self._session_maker() as session:
            return await session.get(GroupAutoTriggerSettingModel, storage_group_id)

    async def get_configured_probability(self, group_id: int | None) -> float | None:
        """获取默认值或指定群显式配置的概率。"""

        row = await self._get_configured_setting(group_id)
        if row is None or row.probability is None:
            return None
        return float(row.probability)

    async def get_configured_cooldown_seconds(self, group_id: int | None) -> float | None:
        """获取默认值或指定群显式配置的冷却秒数。"""

        row = await self._get_configured_setting(group_id)
        if row is None or row.cooldown_seconds is None:
            return None
        return float(row.cooldown_seconds)

    async def get_effective_probability(self, group_id: int) -> float:
        """获取指定群最终生效的自动触发概率。"""

        normalized_group_id = self._normalize_group_id(group_id)
        group_probability = await self.get_configured_probability(normalized_group_id)
        if group_probability is not None:
            return group_probability

        default_probability = await self.get_configured_probability(None)
        if default_probability is not None:
            return default_probability
        return _DEFAULT_TRIGGER_PROBABILITY

    async def get_effective_cooldown_seconds(self, group_id: int) -> float:
        """获取指定群最终生效的自动触发冷却时间。"""

        normalized_group_id = self._normalize_group_id(group_id)
        group_cooldown = await self.get_configured_cooldown_seconds(normalized_group_id)
        if group_cooldown is not None:
            return group_cooldown

        default_cooldown = await self.get_configured_cooldown_seconds(None)
        if default_cooldown is not None:
            return default_cooldown
        return _DEFAULT_COOLDOWN_SECONDS

    async def get_default_probability(self) -> float:
        """获取默认槽位当前生效的自动触发概率。"""

        default_probability = await self.get_configured_probability(None)
        if default_probability is not None:
            return default_probability
        return _DEFAULT_TRIGGER_PROBABILITY

    async def get_default_cooldown_seconds(self) -> float:
        """获取默认槽位当前生效的自动触发冷却时间。"""

        default_cooldown = await self.get_configured_cooldown_seconds(None)
        if default_cooldown is not None:
            return default_cooldown
        return _DEFAULT_COOLDOWN_SECONDS

    async def get_last_triggered_at(self, group_id: int) -> float | None:
        """获取指定群最近一次自动触发时间戳。"""

        await self._ensure_tables()
        normalized_group_id = self._normalize_group_id(group_id)

        async with self._session_maker() as session:
            row = await session.get(GroupAutoTriggerStateModel, normalized_group_id)
            if row is None:
                return None
            return float(row.last_triggered_at)

    async def mark_triggered(self, group_id: int, *, triggered_at: float | None = None) -> float:
        """记录指定群最近一次自动触发时间。"""

        await self._ensure_tables()
        normalized_group_id = self._normalize_group_id(group_id)
        now = float(triggered_at if triggered_at is not None else time.time())

        async with self._session_maker() as session:
            row = await session.get(GroupAutoTriggerStateModel, normalized_group_id)
            if row is None:
                row = GroupAutoTriggerStateModel(
                    group_id=normalized_group_id,
                    last_triggered_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.last_triggered_at = now
                row.updated_at = now
            await session.commit()

        return now

    async def is_cooldown_ready(
        self,
        group_id: int,
        *,
        now_ts: float | None = None,
        cooldown_seconds: float | None = None,
    ) -> bool:
        """判断指定群当前是否已通过冷却时间检查。"""

        normalized_group_id = self._normalize_group_id(group_id)
        effective_now = float(now_ts if now_ts is not None else time.time())
        effective_cooldown = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else await self.get_effective_cooldown_seconds(normalized_group_id)
        )
        last_triggered_at = await self.get_last_triggered_at(normalized_group_id)
        if last_triggered_at is None:
            return True
        return (effective_now - float(last_triggered_at)) >= effective_cooldown

    async def describe_overview(self) -> str:
        """生成群聊自动触发的全局配置摘要。"""

        whitelist = await self.list_entries()
        default_probability = await self.get_default_probability()
        default_cooldown = await self.get_default_cooldown_seconds()

        lines = [
            "群聊自动触发配置",
            f"白名单: {', '.join(str(x) for x in whitelist) if whitelist else '(空)'}",
            f"默认触发概率: {default_probability:.2f}%",
            f"默认冷却时间: {default_cooldown:.2f} 秒",
        ]
        return "\n".join(lines)

    async def describe_group(self, group_id: int) -> str:
        """生成指定群的自动触发生效配置摘要。"""

        normalized_group_id = self._normalize_group_id(group_id)
        enabled = await self.is_group_enabled(normalized_group_id)
        default_probability = await self.get_default_probability()
        default_cooldown = await self.get_default_cooldown_seconds()
        group_probability = await self.get_configured_probability(normalized_group_id)
        group_cooldown = await self.get_configured_cooldown_seconds(normalized_group_id)
        effective_probability = await self.get_effective_probability(normalized_group_id)
        effective_cooldown = await self.get_effective_cooldown_seconds(normalized_group_id)
        last_triggered_at = await self.get_last_triggered_at(normalized_group_id)

        lines = [
            f"群 {normalized_group_id} 自动触发配置",
            f"是否启用: {'是' if enabled else '否'}",
            f"默认触发概率: {default_probability:.2f}%",
            f"当前群单独概率: {f'{group_probability:.2f}%' if group_probability is not None else '(未设置)'}",
            f"当前群生效概率: {effective_probability:.2f}%",
            f"默认冷却时间: {default_cooldown:.2f} 秒",
            f"当前群单独冷却: {f'{group_cooldown:.2f} 秒' if group_cooldown is not None else '(未设置)'}",
            f"当前群生效冷却: {effective_cooldown:.2f} 秒",
            f"最近触发时间: {f'{float(last_triggered_at):.3f}' if last_triggered_at is not None else '(无)'}",
        ]
        return "\n".join(lines)


_group_auto_trigger_manager = GroupAutoTriggerManager()


def get_group_auto_trigger_manager() -> GroupAutoTriggerManager:
    """返回全局共享的群聊自动触发管理器实例。"""

    return _group_auto_trigger_manager
