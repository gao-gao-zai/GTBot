from __future__ import annotations

import asyncio
import time
from enum import Enum

from sqlalchemy import FLOAT, INTEGER, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column

from ...DBmodel import Base, async_session_maker, engine
from ...Logger import logger


_DEFAULT_GROUP_ID = 0
_GLOBAL_SCOPE_KEY = "group"
_DEFAULT_TRIGGER_PROBABILITY = 100.0


class GroupKeywordTriggerMode(str, Enum):
    """群聊关键词触发的群范围控制模式。"""

    OFF = "off"
    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"


class GroupKeywordTriggerListType(str, Enum):
    """群聊关键词触发的名单类型。"""

    BLACKLIST = "blacklist"
    WHITELIST = "whitelist"


class GroupKeywordTriggerModeModel(Base):
    """保存群聊关键词触发的全局模式配置。"""

    __tablename__ = "group_keyword_trigger_modes"

    scope: Mapped[str] = mapped_column(String, primary_key=True)
    mode: Mapped[str] = mapped_column(String, nullable=False, default=GroupKeywordTriggerMode.OFF.value)
    updated_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    updated_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupKeywordTriggerEntryModel(Base):
    """保存允许或禁止关键词触发的群名单项。"""

    __tablename__ = "group_keyword_trigger_entries"
    __table_args__ = (
        UniqueConstraint("group_id", "list_type", name="uq_group_keyword_trigger_entry"),
    )

    id: Mapped[int] = mapped_column(INTEGER, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False)
    list_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    created_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupKeywordTriggerProbabilityModel(Base):
    """保存默认值或指定群的触发概率配置。"""

    __tablename__ = "group_keyword_trigger_probabilities"

    group_id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    probability: Mapped[float] = mapped_column(FLOAT, nullable=False, default=_DEFAULT_TRIGGER_PROBABILITY)
    updated_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    updated_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupKeywordTriggerKeywordModel(Base):
    """保存默认值或指定群的关键词列表。"""

    __tablename__ = "group_keyword_trigger_keywords"
    __table_args__ = (
        UniqueConstraint("group_id", "keyword", name="uq_group_keyword_trigger_keyword"),
    )

    id: Mapped[int] = mapped_column(INTEGER, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(INTEGER, index=True, nullable=False, default=_DEFAULT_GROUP_ID)
    keyword: Mapped[str] = mapped_column(String, index=True, nullable=False, default="")
    created_at: Mapped[float] = mapped_column(FLOAT, nullable=False, default=0.0)
    created_by: Mapped[int] = mapped_column(INTEGER, nullable=False, default=0)


class GroupKeywordTriggerManager:
    """统一管理群聊关键词触发的配置与判定。"""

    def __init__(self) -> None:
        """初始化管理器并准备异步数据库访问依赖。"""

        self._session_maker = async_session_maker
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_tables(self) -> None:
        """确保群聊关键词触发相关的数据表已经创建。"""

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with engine.begin() as conn:
                await conn.run_sync(GroupKeywordTriggerModeModel.metadata.create_all)
            self._initialized = True

    def _normalize_mode(self, mode: GroupKeywordTriggerMode | str) -> GroupKeywordTriggerMode:
        """将触发模式标准化为枚举值。"""

        if isinstance(mode, GroupKeywordTriggerMode):
            return mode
        return GroupKeywordTriggerMode(str(mode).strip().lower())

    def _normalize_list_type(self, list_type: GroupKeywordTriggerListType | str) -> GroupKeywordTriggerListType:
        """将名单类型标准化为枚举值。"""

        if isinstance(list_type, GroupKeywordTriggerListType):
            return list_type
        return GroupKeywordTriggerListType(str(list_type).strip().lower())

    def _normalize_group_id(self, group_id: int | None, *, allow_default: bool = False) -> int:
        """将群号或默认槽位标准化为内部存储值。

        Args:
            group_id: 目标群号；当允许默认槽位时可传 `None`。
            allow_default: 是否允许把 `None` 视为默认配置槽位。

        Returns:
            int: 有效群号，或默认配置使用的 `_DEFAULT_GROUP_ID`。

        Raises:
            ValueError: 当群号为空或不是正整数时抛出。
        """

        if group_id is None:
            if allow_default:
                return _DEFAULT_GROUP_ID
            raise ValueError("群号不能为空")

        normalized_group_id = int(group_id)
        if normalized_group_id <= 0:
            raise ValueError(f"无效的群号: {group_id}")
        return normalized_group_id

    def _normalize_probability(self, probability: float | int | str) -> float:
        """把概率输入解析为 0 到 100 之间的浮点数。"""

        raw = str(probability).strip()
        if raw.endswith("%"):
            raw = raw[:-1].strip()
        value = float(raw)
        if value < 0 or value > 100:
            raise ValueError("触发概率必须在 0 到 100 之间")
        return round(value, 4)

    def _normalize_keyword(self, keyword: str) -> str:
        """校验并清洗关键词文本。"""

        normalized = str(keyword).strip()
        if not normalized:
            raise ValueError("关键词不能为空")
        return normalized

    @staticmethod
    def _dedupe_keywords(keywords: list[str]) -> list[str]:
        """按出现顺序去重关键词，避免默认词与群专属词重复。"""

        result: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            if keyword in seen:
                continue
            seen.add(keyword)
            result.append(keyword)
        return result

    async def get_mode(self) -> GroupKeywordTriggerMode:
        """获取当前群聊关键词触发的全局范围模式。"""

        await self._ensure_tables()
        async with self._session_maker() as session:
            row = await session.get(GroupKeywordTriggerModeModel, _GLOBAL_SCOPE_KEY)
            if row is None:
                return GroupKeywordTriggerMode.OFF
            return self._normalize_mode(row.mode)

    async def set_mode(
        self,
        mode: GroupKeywordTriggerMode | str,
        operator_user_id: int,
    ) -> GroupKeywordTriggerMode:
        """设置群聊关键词触发的全局范围模式。"""

        await self._ensure_tables()
        normalized_mode = self._normalize_mode(mode)
        now = time.time()

        async with self._session_maker() as session:
            row = await session.get(GroupKeywordTriggerModeModel, _GLOBAL_SCOPE_KEY)
            if row is None:
                row = GroupKeywordTriggerModeModel(
                    scope=_GLOBAL_SCOPE_KEY,
                    mode=normalized_mode.value,
                    updated_at=now,
                    updated_by=int(operator_user_id),
                )
                session.add(row)
            else:
                row.mode = normalized_mode.value
                row.updated_at = now
                row.updated_by = int(operator_user_id)
            await session.commit()

        logger.info(
            "用户 %s 已将群聊关键词触发模式设置为 %s",
            int(operator_user_id),
            normalized_mode.value,
        )
        return normalized_mode

    async def list_entries(self, list_type: GroupKeywordTriggerListType | str) -> list[int]:
        """列出指定名单类型中的所有群号。"""

        await self._ensure_tables()
        normalized_list_type = self._normalize_list_type(list_type)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerEntryModel.group_id)
                .where(GroupKeywordTriggerEntryModel.list_type == normalized_list_type.value)
                .order_by(GroupKeywordTriggerEntryModel.group_id.asc())
            )
            return [int(group_id) for group_id in result.scalars().all()]

    async def add_entry(
        self,
        list_type: GroupKeywordTriggerListType | str,
        group_id: int,
        operator_user_id: int,
    ) -> bool:
        """向黑名单或白名单中添加群号。"""

        await self._ensure_tables()
        normalized_list_type = self._normalize_list_type(list_type)
        normalized_group_id = self._normalize_group_id(group_id)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerEntryModel)
                .where(
                    GroupKeywordTriggerEntryModel.group_id == normalized_group_id,
                    GroupKeywordTriggerEntryModel.list_type == normalized_list_type.value,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return False

            session.add(
                GroupKeywordTriggerEntryModel(
                    group_id=normalized_group_id,
                    list_type=normalized_list_type.value,
                    created_at=time.time(),
                    created_by=int(operator_user_id),
                )
            )
            await session.commit()

        logger.info(
            "用户 %s 已将群 %s 加入群聊关键词触发 %s",
            int(operator_user_id),
            normalized_group_id,
            normalized_list_type.value,
        )
        return True

    async def remove_entry(
        self,
        list_type: GroupKeywordTriggerListType | str,
        group_id: int,
        operator_user_id: int,
    ) -> bool:
        """从黑名单或白名单中移除群号。"""

        await self._ensure_tables()
        normalized_list_type = self._normalize_list_type(list_type)
        normalized_group_id = self._normalize_group_id(group_id)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerEntryModel)
                .where(
                    GroupKeywordTriggerEntryModel.group_id == normalized_group_id,
                    GroupKeywordTriggerEntryModel.list_type == normalized_list_type.value,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return False

            await session.delete(existing)
            await session.commit()

        logger.info(
            "用户 %s 已将群 %s 从群聊关键词触发 %s 中移除",
            int(operator_user_id),
            normalized_group_id,
            normalized_list_type.value,
        )
        return True

    async def is_group_enabled(self, group_id: int) -> bool:
        """判断某个群是否允许使用关键词触发。"""

        normalized_group_id = self._normalize_group_id(group_id)
        mode = await self.get_mode()

        if mode == GroupKeywordTriggerMode.OFF:
            return False

        if mode == GroupKeywordTriggerMode.BLACKLIST:
            blacklist = await self.list_entries(GroupKeywordTriggerListType.BLACKLIST)
            return normalized_group_id not in set(blacklist)

        whitelist = await self.list_entries(GroupKeywordTriggerListType.WHITELIST)
        return normalized_group_id in set(whitelist)

    async def set_probability(
        self,
        *,
        group_id: int | None,
        probability: float | int | str,
        operator_user_id: int,
    ) -> float:
        """设置默认值或指定群的触发概率。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)
        normalized_probability = self._normalize_probability(probability)
        now = time.time()

        async with self._session_maker() as session:
            row = await session.get(GroupKeywordTriggerProbabilityModel, storage_group_id)
            if row is None:
                row = GroupKeywordTriggerProbabilityModel(
                    group_id=storage_group_id,
                    probability=normalized_probability,
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
            "用户 %s 已将群聊关键词触发概率设置为 %.4f（group_id=%s）",
            int(operator_user_id),
            normalized_probability,
            storage_group_id,
        )
        return normalized_probability

    async def get_configured_probability(self, group_id: int | None) -> float | None:
        """获取默认值或指定群显式配置的概率。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)

        async with self._session_maker() as session:
            row = await session.get(GroupKeywordTriggerProbabilityModel, storage_group_id)
            if row is None:
                return None
            return float(row.probability)

    async def get_effective_probability(self, group_id: int) -> float:
        """获取指定群最终生效的触发概率。"""

        normalized_group_id = self._normalize_group_id(group_id)
        group_probability = await self.get_configured_probability(normalized_group_id)
        if group_probability is not None:
            return group_probability

        default_probability = await self.get_configured_probability(None)
        if default_probability is not None:
            return default_probability
        return _DEFAULT_TRIGGER_PROBABILITY

    async def get_default_probability(self) -> float:
        """获取默认槽位当前生效的关键词触发概率。"""

        default_probability = await self.get_configured_probability(None)
        if default_probability is not None:
            return default_probability
        return _DEFAULT_TRIGGER_PROBABILITY

    async def list_keywords(self, group_id: int | None) -> list[str]:
        """列出默认值或指定群直接配置的关键词。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerKeywordModel.keyword)
                .where(GroupKeywordTriggerKeywordModel.group_id == storage_group_id)
                .order_by(GroupKeywordTriggerKeywordModel.keyword.asc())
            )
            return [str(keyword) for keyword in result.scalars().all()]

    async def add_keyword(
        self,
        *,
        group_id: int | None,
        keyword: str,
        operator_user_id: int,
    ) -> bool:
        """向默认值或指定群添加一个可用关键词。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)
        normalized_keyword = self._normalize_keyword(keyword)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerKeywordModel)
                .where(
                    GroupKeywordTriggerKeywordModel.group_id == storage_group_id,
                    GroupKeywordTriggerKeywordModel.keyword == normalized_keyword,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return False

            session.add(
                GroupKeywordTriggerKeywordModel(
                    group_id=storage_group_id,
                    keyword=normalized_keyword,
                    created_at=time.time(),
                    created_by=int(operator_user_id),
                )
            )
            await session.commit()

        logger.info(
            "用户 %s 已添加群聊关键词触发关键词 %r（group_id=%s）",
            int(operator_user_id),
            normalized_keyword,
            storage_group_id,
        )
        return True

    async def remove_keyword(
        self,
        *,
        group_id: int | None,
        keyword: str,
        operator_user_id: int,
    ) -> bool:
        """从默认值或指定群移除一个关键词。"""

        await self._ensure_tables()
        storage_group_id = self._normalize_group_id(group_id, allow_default=True)
        normalized_keyword = self._normalize_keyword(keyword)

        async with self._session_maker() as session:
            result = await session.execute(
                select(GroupKeywordTriggerKeywordModel)
                .where(
                    GroupKeywordTriggerKeywordModel.group_id == storage_group_id,
                    GroupKeywordTriggerKeywordModel.keyword == normalized_keyword,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                return False

            await session.delete(existing)
            await session.commit()

        logger.info(
            "用户 %s 已移除群聊关键词触发关键词 %r（group_id=%s）",
            int(operator_user_id),
            normalized_keyword,
            storage_group_id,
        )
        return True

    async def get_effective_keywords(self, group_id: int) -> list[str]:
        """获取某个群最终生效的关键词集合。

        规则为“默认关键词 + 群专属关键词”按顺序合并去重。
        """

        normalized_group_id = self._normalize_group_id(group_id)
        default_keywords = await self.list_keywords(None)
        group_keywords = await self.list_keywords(normalized_group_id)
        return self._dedupe_keywords([*default_keywords, *group_keywords])

    async def find_matching_keyword(self, group_id: int, text: str) -> str | None:
        """在给定文本中查找某个群当前可用的命中关键词。"""

        normalized_text = str(text).strip()
        if not normalized_text:
            return None

        effective_keywords = await self.get_effective_keywords(group_id)
        if not effective_keywords:
            return None

        lowered_text = normalized_text.casefold()
        for keyword in sorted(effective_keywords, key=lambda item: len(item), reverse=True):
            if keyword.casefold() in lowered_text:
                return keyword
        return None

    async def describe_overview(self) -> str:
        """生成全局关键词触发配置摘要。"""

        mode = await self.get_mode()
        whitelist = await self.list_entries(GroupKeywordTriggerListType.WHITELIST)
        blacklist = await self.list_entries(GroupKeywordTriggerListType.BLACKLIST)
        default_probability = await self.get_default_probability()
        default_keywords = await self.list_keywords(None)

        mode_labels = {
            GroupKeywordTriggerMode.OFF: "关闭",
            GroupKeywordTriggerMode.BLACKLIST: "黑名单",
            GroupKeywordTriggerMode.WHITELIST: "白名单",
        }

        lines = [
            f"群聊关键词触发模式: {mode_labels[mode]}",
            f"白名单: {', '.join(str(x) for x in whitelist) if whitelist else '(空)'}",
            f"黑名单: {', '.join(str(x) for x in blacklist) if blacklist else '(空)'}",
            f"默认触发概率: {default_probability:.2f}%",
            f"默认关键词: {', '.join(default_keywords) if default_keywords else '(空)'}",
        ]
        return "\n".join(lines)

    async def describe_group(self, group_id: int) -> str:
        """生成某个群的生效关键词触发配置摘要。"""

        normalized_group_id = self._normalize_group_id(group_id)
        mode = await self.get_mode()
        enabled = await self.is_group_enabled(normalized_group_id)
        default_probability = await self.get_default_probability()
        group_probability = await self.get_configured_probability(normalized_group_id)
        effective_probability = await self.get_effective_probability(normalized_group_id)
        default_keywords = await self.list_keywords(None)
        group_keywords = await self.list_keywords(normalized_group_id)
        effective_keywords = await self.get_effective_keywords(normalized_group_id)

        mode_labels = {
            GroupKeywordTriggerMode.OFF: "关闭",
            GroupKeywordTriggerMode.BLACKLIST: "黑名单",
            GroupKeywordTriggerMode.WHITELIST: "白名单",
        }

        lines = [
            f"群 {normalized_group_id} 关键词触发配置:",
            f"全局模式: {mode_labels[mode]}",
            f"当前群是否允许触发: {'是' if enabled else '否'}",
            f"默认触发概率: {default_probability:.2f}%",
            f"当前群单独概率: {f'{group_probability:.2f}%' if group_probability is not None else '(未设置)'}",
            f"当前群生效概率: {effective_probability:.2f}%",
            f"默认关键词: {', '.join(default_keywords) if default_keywords else '(空)'}",
            f"当前群单独关键词: {', '.join(group_keywords) if group_keywords else '(空)'}",
            f"当前群生效关键词: {', '.join(effective_keywords) if effective_keywords else '(空)'}",
        ]
        return "\n".join(lines)


_group_keyword_trigger_manager = GroupKeywordTriggerManager()


def get_group_keyword_trigger_manager() -> GroupKeywordTriggerManager:
    """返回全局共享的群聊关键词触发管理器实例。"""

    return _group_keyword_trigger_manager
