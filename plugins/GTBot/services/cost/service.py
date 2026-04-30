from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage

from plugins.GTBot.services.plugin_api.permissions import PermissionError, PermissionRole, get_role
from plugins.GTBot.services.plugin_system.runtime import get_current_plugin_context

from ..cache import get_user_cache_manager
from .models import CostBillingMode, CostLeaderboardEntry, CostRecord, CostSourceType, CostSummary
from .store import CostLedgerStore


@dataclass(frozen=True, slots=True)
class UsagePathSet:
    """描述一组可用于提取 usage 的路径配置。

    该结构既可表达供应商默认路径，也可表达流式或非流式模式下的覆盖路径。
    当 `input_tokens_include_cache_read` 为 `True` 时，会在计费前自动从输入 token 中扣除
    缓存命中部分，避免同一批 token 同时按普通输入和缓存读取两种单价重复计费。
    """

    input_tokens_path: str
    output_tokens_path: str
    cache_read_tokens_path: str = ""
    request_id_path: str = ""
    input_tokens_include_cache_read: bool = False


@dataclass(frozen=True, slots=True)
class ProviderUsageRule:
    """描述单个供应商在响应体中的 usage 字段提取规则。

    不同供应商即使都走 OpenAI 兼容协议，也可能把缓存读取 token 或请求 ID
    放在不同路径下。该结构用于把“如何读取 usage”的差异从具体计费逻辑中抽离。

    Attributes:
        input_tokens_path: 输入 token 路径。
        output_tokens_path: 输出 token 路径。
        cache_read_tokens_path: 缓存读取 token 路径；允许为空字符串表示未配置。
        request_id_path: 请求 ID 路径；允许为空字符串表示未配置。
    """

    input_tokens_path: str
    output_tokens_path: str
    cache_read_tokens_path: str = ""
    request_id_path: str = ""
    input_tokens_include_cache_read: bool = False
    non_streaming: UsagePathSet | None = None
    streaming: UsagePathSet | None = None


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """描述单个模型的输入、输出和缓存读取价格。

    Attributes:
        enabled: 是否启用该模型的自动计费。
        input_price_per_million: 输入 token 单价，单位为金额/百万 token。
        output_price_per_million: 输出 token 单价，单位为金额/百万 token。
        cache_read_price_per_million: 缓存读取 token 单价，单位为金额/百万 token。
        currency: 币种。当前版本固定为 `CNY`。
    """

    enabled: bool
    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float
    currency: str


class CostLedgerService:
    """提供统一消费账本的记账、计费和查询能力。

    该服务作为 GTBot 消费统计功能的中心入口，负责：

    - 将聊天 API 的成功响应转换为可持久化账单。
    - 将插件主动上报的消费写入统一账本。
    - 提供个人总览、明细列表和排行榜查询。
    - 校验“谁可以查询谁”的权限边界。

    存储层采用同步 SQLite，实现层统一通过 `asyncio.to_thread()` 调用，
    避免在 NoneBot 主事件循环中直接阻塞。
    """

    def __init__(self, store: CostLedgerStore | None = None) -> None:
        """初始化消费账本服务。

        Args:
            store: 可选自定义存储实现。未传入时使用默认 SQLite 账本。
        """

        self._store = store if store is not None else CostLedgerStore()

    @property
    def database_path(self) -> str:
        """返回当前账本数据库路径，便于命令和日志展示。

        Returns:
            账本数据库文件的绝对路径字符串。
        """

        return str(self._store.db_path)

    async def record_cost(self, record: CostRecord) -> bool:
        """写入一条消费账单记录。

        Args:
            record: 已完成字段归一化和金额计算的消费明细记录。

        Returns:
            `True` 表示成功写入新账单，`False` 表示命中幂等去重。
        """

        return await asyncio.to_thread(self._store.insert_record, record)

    async def record_plugin_cost(
        self,
        *,
        source_name: str,
        category: str,
        billing_mode: CostBillingMode,
        quantity: float,
        unit_price: float | None,
        amount: float,
        owner_user_id: int,
        actor_user_id: int | None = None,
        occurred_at: float | None = None,
        group_id: int | None = None,
        session_id: str | None = None,
        response_id: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        extra: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> bool:
        """按显式参数构造并写入一条插件账单。

        Args:
            source_name: 账单来源名称。
            category: 账单分类。
            billing_mode: 计费模式。
            quantity: 原始计费数量。
            unit_price: 单价；`direct_amount` 模式允许为空。
            amount: 最终金额。
            owner_user_id: 账单归属用户。
            actor_user_id: 实际触发用户；未传入时默认与归属用户一致。
            occurred_at: 可选消费发生时间；未传入时使用当前时间。
            group_id: 可选群号。
            session_id: 可选会话 ID。
            response_id: 可选响应 ID。
            provider: 可选供应商名称。
            model_name: 可选模型名或资源名。
            extra: 可选扩展字段。
            event_id: 可选稳定事件 ID；未传入时自动生成。

        Returns:
            `True` 表示成功写入新账单，`False` 表示命中幂等去重。
        """

        normalized_owner_user_id = int(owner_user_id)
        normalized_actor_user_id = int(actor_user_id if actor_user_id is not None else owner_user_id)
        record = CostRecord(
            event_id=str(event_id or f"plugin_cost_{uuid4().hex}"),
            occurred_at=float(occurred_at if occurred_at is not None else time()),
            source_type="plugin",
            source_name=str(source_name),
            category=str(category),
            owner_user_id=normalized_owner_user_id,
            actor_user_id=normalized_actor_user_id,
            group_id=int(group_id) if group_id is not None else None,
            session_id=str(session_id) if session_id else None,
            response_id=str(response_id) if response_id else None,
            provider=str(provider) if provider else None,
            model_name=str(model_name) if model_name else None,
            billing_mode=billing_mode,
            quantity=float(quantity),
            unit_price=float(unit_price) if unit_price is not None else None,
            amount=self._normalize_money(amount),
            currency="CNY",
            extra=dict(extra or {}),
        )
        return await self.record_cost(record)

    async def record_cost_for_current_request(
        self,
        *,
        source_name: str,
        category: str,
        billing_mode: CostBillingMode,
        quantity: float,
        unit_price: float | None,
        amount: float,
        owner_user_id: int | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        extra: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> bool:
        """基于当前请求上下文写入一条插件账单。

        该入口优先从当前 `PluginContext` 关联的 `runtime_context` 中自动补全用户、
        会话、群号和响应 ID，适合插件在 tool 或后台回调中直接调用，避免每个插件
        重新拼接同一套上下文字段。

        Args:
            source_name: 账单来源名称。
            category: 账单分类。
            billing_mode: 计费模式。
            quantity: 原始计费数量。
            unit_price: 单价；`direct_amount` 模式允许为空。
            amount: 最终金额。
            owner_user_id: 可选显式归属用户；未传入时默认等于当前请求用户。
            provider: 可选供应商名称。
            model_name: 可选模型名或资源名。
            extra: 可选扩展字段。
            event_id: 可选稳定事件 ID；未传入时自动生成。

        Returns:
            `True` 表示成功写入新账单，`False` 表示命中幂等去重。

        Raises:
            RuntimeError: 当前协程不在插件请求上下文内，或上下文中缺失用户信息时抛出。
        """

        plugin_ctx = get_current_plugin_context()
        if plugin_ctx is None:
            raise RuntimeError("当前协程不存在可用的 GTBot 插件上下文，无法自动记账")
        runtime_context = getattr(plugin_ctx, "runtime_context", None)
        actor_user_id = int(getattr(runtime_context, "user_id", 0) or 0)
        if actor_user_id <= 0:
            raise RuntimeError("当前请求上下文缺少有效 user_id，无法自动记账")
        normalized_owner_user_id = int(owner_user_id if owner_user_id is not None else actor_user_id)
        return await self.record_plugin_cost(
            source_name=source_name,
            category=category,
            billing_mode=billing_mode,
            quantity=quantity,
            unit_price=unit_price,
            amount=amount,
            owner_user_id=normalized_owner_user_id,
            actor_user_id=actor_user_id,
            occurred_at=float(time()),
            group_id=self._normalize_optional_int(getattr(runtime_context, "group_id", None)),
            session_id=self._normalize_optional_str(getattr(runtime_context, "session_id", None)),
            response_id=self._normalize_optional_str(getattr(runtime_context, "response_id", None)),
            provider=provider,
            model_name=model_name,
            extra=extra,
            event_id=event_id,
        )

    async def record_chat_cost_from_response(
        self,
        *,
        response: dict[str, Any],
        runtime_context: Any,
        chat_model_config: Any,
    ) -> bool:
        """从一轮成功聊天响应中提取 usage 并自动写入账单。

        该方法只应在聊天主链路成功拿到最终响应后调用。若未配置供应商 usage 规则、
        未配置模型价格或响应中取不到 usage，则直接返回 `False`，并将责任交给调用方
        决定是否记录诊断日志。若同一轮响应中包含多条 `AIMessage`，则会把每条
        可提取到 usage 的消息视为一次独立上游请求并分别落账。

        Args:
            response: 智能体最终响应字典。
            runtime_context: 当前聊天请求的运行时上下文。
            chat_model_config: 当前配置组中已解析完成的 `chat_model` 配置对象。

        Returns:
            `True` 表示至少成功写入一条新账单，`False` 表示本次未形成账单或全部命中幂等去重。
        """

        provider_name = str(getattr(chat_model_config, "provider_name", "") or "").strip()
        model_name = str(getattr(chat_model_config, "model_id", "") or "").strip()
        if not provider_name or not model_name:
            return False

        usage_rule = self._resolve_provider_usage_rule(chat_model_config=chat_model_config, provider_name=provider_name)
        pricing = self._resolve_model_pricing(
            chat_model_config=chat_model_config,
            provider_name=provider_name,
            model_name=model_name,
        )
        if usage_rule is None or pricing is None or not pricing.enabled:
            return False

        ai_messages = self._iter_ai_messages(response)
        if not ai_messages:
            return False

        response_id = self._normalize_optional_str(getattr(runtime_context, "response_id", None))
        owner_user_id = int(getattr(runtime_context, "user_id", 0) or 0)
        if owner_user_id <= 0:
            return False
        recorded_any = False
        for message_index, ai_message in ai_messages:
            input_tokens, output_tokens, cache_read_tokens, request_id, usage_snapshot = (
                self._extract_usage_and_request_context(
                    message=ai_message,
                    usage_rule=usage_rule,
                )
            )
            if input_tokens <= 0 and output_tokens <= 0 and cache_read_tokens <= 0:
                continue

            input_cost = input_tokens / 1_000_000.0 * pricing.input_price_per_million
            output_cost = output_tokens / 1_000_000.0 * pricing.output_price_per_million
            cache_read_cost = cache_read_tokens / 1_000_000.0 * pricing.cache_read_price_per_million
            total_amount = self._normalize_money(input_cost + output_cost + cache_read_cost)
            record = CostRecord(
                event_id=self._build_chat_cost_event_id(
                    response_id=response_id,
                    request_id=request_id,
                    message_index=message_index,
                ),
                occurred_at=float(time()),
                source_type="chat_api",
                source_name="gtbot_chat",
                category="llm_tokens",
                owner_user_id=owner_user_id,
                actor_user_id=owner_user_id,
                group_id=self._normalize_optional_int(getattr(runtime_context, "group_id", None)),
                session_id=self._normalize_optional_str(getattr(runtime_context, "session_id", None)),
                response_id=response_id,
                provider=provider_name,
                model_name=model_name,
                billing_mode="per_million_tokens",
                quantity=float(input_tokens + output_tokens + cache_read_tokens),
                unit_price=None,
                amount=total_amount,
                currency="CNY",
                extra={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "input_cost": self._normalize_money(input_cost),
                    "output_cost": self._normalize_money(output_cost),
                    "cache_read_cost": self._normalize_money(cache_read_cost),
                    "request_id": request_id,
                    "message_index": message_index,
                    "usage_snapshot": usage_snapshot,
                },
            )
            if await self.record_cost(record):
                recorded_any = True
        return recorded_any

    @staticmethod
    def _build_chat_cost_event_id(
        *,
        response_id: str | None,
        request_id: str | None,
        message_index: int,
    ) -> str:
        """为单次聊天请求构造稳定的消费事件 ID。

        同一轮对话中可能包含多次上游模型请求。若供应商返回了 `request_id`，则优先
        使用它作为幂等键；否则回退到“当前轮 `response_id` + 消息索引”的组合，
        以便重复处理同一份 response 时仍能避免重复记账。

        Args:
            response_id: 当前整轮聊天的响应 ID，可为空。
            request_id: 上游单次模型请求 ID，可为空。
            message_index: 当前 AIMessage 在 `response["messages"]` 中的索引。

        Returns:
            对当前单次模型请求稳定唯一的事件 ID。
        """

        normalized_request_id = str(request_id or "").strip()
        if normalized_request_id:
            return f"chat_cost_request:{normalized_request_id}"
        normalized_response_id = str(response_id or "").strip()
        if normalized_response_id:
            return f"chat_cost:{normalized_response_id}:{int(message_index)}"
        return f"chat_cost:{uuid4().hex}:{int(message_index)}"

    async def ensure_can_query_user(self, *, requester_user_id: int, target_user_id: int) -> None:
        """校验请求方是否有权查询目标用户账单。

        规则固定为：

        - 用户查自己永远允许。
        - `ADMIN`、`OWNER` 可查任意用户。
        - 其他角色禁止查他人。

        Args:
            requester_user_id: 发起查询的 QQ 用户 ID。
            target_user_id: 被查询账单归属的 QQ 用户 ID。

        Raises:
            PermissionError: 当请求方无权查询目标用户账单时抛出。
        """

        normalized_requester_user_id = int(requester_user_id)
        normalized_target_user_id = int(target_user_id)
        if normalized_requester_user_id == normalized_target_user_id:
            return
        role = await get_role(normalized_requester_user_id)
        if role in {PermissionRole.ADMIN, PermissionRole.OWNER}:
            return
        raise PermissionError("你无权查看其他用户的消费账单。")

    async def get_user_summary(
        self,
        *,
        requester_user_id: int,
        target_user_id: int,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
    ) -> CostSummary:
        """返回指定用户在给定范围内的消费汇总。

        Args:
            requester_user_id: 发起查询的 QQ 用户 ID。
            target_user_id: 需要汇总的账单归属用户 ID。
            start_at: 可选时间范围起点，包含该时间。
            end_at: 可选时间范围终点，不包含该时间。
            group_id: 可选群范围；用于“本群榜”或群内明细过滤。

        Returns:
            指定口径下的消费汇总结果。

        Raises:
            PermissionError: 请求方无权查看目标用户账单时抛出。
        """

        await self.ensure_can_query_user(
            requester_user_id=int(requester_user_id),
            target_user_id=int(target_user_id),
        )
        return await asyncio.to_thread(
            self._store.summarize,
            owner_user_id=int(target_user_id),
            start_at=start_at,
            end_at=end_at,
            group_id=group_id,
        )

    async def get_user_breakdown_by_source(
        self,
        *,
        requester_user_id: int,
        target_user_id: int,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
    ) -> list[tuple[str, CostSummary]]:
        """返回指定用户按来源拆分的消费统计。

        Args:
            requester_user_id: 发起查询的 QQ 用户 ID。
            target_user_id: 需要统计的账单归属用户 ID。
            start_at: 可选时间范围起点。
            end_at: 可选时间范围终点。
            group_id: 可选群范围过滤。

        Returns:
            按消费来源拆分的统计结果列表。

        Raises:
            PermissionError: 请求方无权查看目标用户账单时抛出。
        """

        await self.ensure_can_query_user(
            requester_user_id=int(requester_user_id),
            target_user_id=int(target_user_id),
        )
        return await asyncio.to_thread(
            self._store.summarize_by_source,
            owner_user_id=int(target_user_id),
            start_at=start_at,
            end_at=end_at,
            group_id=group_id,
        )

    async def list_user_records(
        self,
        *,
        requester_user_id: int,
        target_user_id: int,
        start_at: float | None = None,
        end_at: float | None = None,
        source_name: str | None = None,
        response_id: str | None = None,
        group_id: int | None = None,
        limit: int = 20,
    ) -> list[CostRecord]:
        """返回指定用户在给定范围内的消费明细。

        Args:
            requester_user_id: 发起查询的 QQ 用户 ID。
            target_user_id: 需要查询的账单归属用户 ID。
            start_at: 可选时间范围起点。
            end_at: 可选时间范围终点。
            source_name: 可选来源名称过滤。
            response_id: 可选响应 ID 过滤。
            group_id: 可选群范围过滤。
            limit: 最多返回的明细条数。

        Returns:
            按时间倒序排列的消费明细列表。

        Raises:
            PermissionError: 请求方无权查看目标用户账单时抛出。
        """

        await self.ensure_can_query_user(
            requester_user_id=int(requester_user_id),
            target_user_id=int(target_user_id),
        )
        return await asyncio.to_thread(
            self._store.list_records,
            owner_user_id=int(target_user_id),
            start_at=start_at,
            end_at=end_at,
            source_name=source_name,
            response_id=response_id,
            group_id=group_id,
            limit=limit,
        )

    async def get_leaderboard(
        self,
        *,
        start_at: float | None = None,
        end_at: float | None = None,
        group_id: int | None = None,
        limit: int = 20,
    ) -> list[CostLeaderboardEntry]:
        """返回指定范围内的消费排行榜。

        普通成员允许查看完整排行榜，因此该接口不对查询方施加额外的“仅自己可见”限制。
        真正的隐私边界体现在“明细查询必须经过 `ensure_can_query_user()`”。

        Args:
            start_at: 可选时间范围起点。
            end_at: 可选时间范围终点。
            group_id: 可选群范围过滤；用于“本群榜”。
            limit: 最多返回的排行榜条数。

        Returns:
            按消费总额倒序排列的排行榜条目列表。
        """

        return await asyncio.to_thread(
            self._store.leaderboard,
            start_at=start_at,
            end_at=end_at,
            group_id=group_id,
            limit=limit,
        )

    async def resolve_user_display_name(
        self,
        *,
        bot: Any,
        user_id: int,
        group_id: int | None = None,
    ) -> str:
        """解析排行榜或账单展示时的用户名。

        当存在群上下文时优先展示群名片，其次退回到陌生人昵称，再退回到纯 QQ 号。

        Args:
            bot: 当前 Bot 实例。
            user_id: 目标 QQ 用户 ID。
            group_id: 可选群号；存在时优先按群成员名称解析。

        Returns:
            适合展示在命令输出中的用户名字符串。
        """

        cache = await get_user_cache_manager()
        if group_id is not None and int(group_id) > 0:
            return str(await cache.get_group_member_name(bot, int(group_id), int(user_id)))
        return str(await cache.get_user_name(bot, int(user_id)))

    @staticmethod
    def resolve_named_range(name: str) -> tuple[float | None, float | None]:
        """将固定时间范围名称转换为时间戳区间。

        当前支持：

        - `今日`
        - `本周`
        - `本月`
        - `全部`

        其中起点包含、终点不包含，所有边界均按服务器本地时区计算。

        Args:
            name: 时间范围名称。

        Returns:
            对应的 `(start_at, end_at)` 元组；`全部` 返回 `(None, None)`。

        Raises:
            ValueError: 传入不支持的时间范围名称时抛出。
        """

        normalized = str(name or "").strip()
        if normalized in {"", "全部"}:
            return None, None
        now = datetime.fromtimestamp(time()).astimezone()
        if normalized == "今日":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start.timestamp(), (start + timedelta(days=1)).timestamp()
        if normalized == "本周":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday())
            return start.timestamp(), (start + timedelta(days=7)).timestamp()
        if normalized == "本月":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
            return start.timestamp(), end.timestamp()
        raise ValueError(f"不支持的时间范围: {name}")

    @staticmethod
    def parse_date_range(start_text: str, end_text: str) -> tuple[float, float]:
        """将命令中的起止日期解析为闭开区间时间戳。

        日期格式固定为 `YYYY-MM-DD`。起点按当天零点计算，终点按结束日期次日零点计算，
        以保证用户输入“2026-04-01 2026-04-30”时能覆盖完整的 4 月 30 日。

        Args:
            start_text: 起始日期文本。
            end_text: 结束日期文本。

        Returns:
            对应的 `(start_at, end_at)` 时间戳区间。

        Raises:
            ValueError: 日期格式非法或结束日期早于开始日期时抛出。
        """

        try:
            start_dt = datetime.strptime(str(start_text), "%Y-%m-%d").astimezone()
            end_dt = datetime.strptime(str(end_text), "%Y-%m-%d").astimezone()
        except ValueError as exc:
            raise ValueError("日期格式必须为 YYYY-MM-DD") from exc
        if end_dt < start_dt:
            raise ValueError("结束日期不能早于开始日期")
        end_exclusive = end_dt + timedelta(days=1)
        return start_dt.timestamp(), end_exclusive.timestamp()

    def _resolve_provider_usage_rule(self, *, chat_model_config: Any, provider_name: str) -> ProviderUsageRule | None:
        """从聊天模型配置中解析指定供应商的 usage 提取规则。

        Args:
            chat_model_config: 当前配置组中的 `chat_model` 对象。
            provider_name: 当前生效的供应商名称。

        Returns:
            找到且字段有效时返回解析后的规则对象，否则返回 `None`。
        """

        cost_cfg = getattr(chat_model_config, "cost", None)
        if cost_cfg is None:
            return None
        provider_rules = getattr(cost_cfg, "provider_usage_rules", {}) or {}
        raw_rule = provider_rules.get(provider_name)
        if raw_rule is None:
            return None
        return ProviderUsageRule(
            input_tokens_path=str(getattr(raw_rule, "input_tokens_path", "") or ""),
            output_tokens_path=str(getattr(raw_rule, "output_tokens_path", "") or ""),
            cache_read_tokens_path=str(getattr(raw_rule, "cache_read_tokens_path", "") or ""),
            request_id_path=str(getattr(raw_rule, "request_id_path", "") or ""),
            input_tokens_include_cache_read=bool(
                getattr(raw_rule, "input_tokens_include_cache_read", False)
            ),
            non_streaming=self._resolve_usage_path_set(getattr(raw_rule, "non_streaming", None)),
            streaming=self._resolve_usage_path_set(getattr(raw_rule, "streaming", None)),
        )

    @staticmethod
    def _resolve_usage_path_set(raw_path_set: Any) -> UsagePathSet | None:
        """将配置对象安全转换为运行时使用的路径集合。"""

        if raw_path_set is None:
            return None
        input_tokens_path = str(getattr(raw_path_set, "input_tokens_path", "") or "")
        output_tokens_path = str(getattr(raw_path_set, "output_tokens_path", "") or "")
        cache_read_tokens_path = str(getattr(raw_path_set, "cache_read_tokens_path", "") or "")
        request_id_path = str(getattr(raw_path_set, "request_id_path", "") or "")
        input_tokens_include_cache_read = bool(
            getattr(raw_path_set, "input_tokens_include_cache_read", False)
        )
        if not any((input_tokens_path, output_tokens_path, cache_read_tokens_path, request_id_path)):
            return None
        return UsagePathSet(
            input_tokens_path=input_tokens_path,
            output_tokens_path=output_tokens_path,
            cache_read_tokens_path=cache_read_tokens_path,
            request_id_path=request_id_path,
            input_tokens_include_cache_read=input_tokens_include_cache_read,
        )

    def _resolve_model_pricing(
        self,
        *,
        chat_model_config: Any,
        provider_name: str,
        model_name: str,
    ) -> ModelPricing | None:
        """从聊天模型配置中解析指定模型的价格配置。

        Args:
            chat_model_config: 当前配置组中的 `chat_model` 对象。
            provider_name: 当前生效的供应商名称。
            model_name: 当前生效的上游模型名。

        Returns:
            找到且字段有效时返回价格配置，否则返回 `None`。
        """

        cost_cfg = getattr(chat_model_config, "cost", None)
        if cost_cfg is None:
            return None
        model_pricing = getattr(cost_cfg, "model_pricing", {}) or {}
        provider_pricing = model_pricing.get(provider_name)
        if not isinstance(provider_pricing, dict):
            return None
        raw_pricing = provider_pricing.get(model_name)
        if raw_pricing is None:
            return None
        return ModelPricing(
            enabled=bool(getattr(raw_pricing, "enabled", False)),
            input_price_per_million=float(getattr(raw_pricing, "input_price_per_million", 0.0) or 0.0),
            output_price_per_million=float(getattr(raw_pricing, "output_price_per_million", 0.0) or 0.0),
            cache_read_price_per_million=float(
                getattr(raw_pricing, "cache_read_price_per_million", 0.0) or 0.0
            ),
            currency=str(getattr(raw_pricing, "currency", "CNY") or "CNY"),
        )

    @staticmethod
    def _iter_ai_messages(response: dict[str, Any]) -> list[tuple[int, AIMessage]]:
        """按原始顺序枚举响应中的全部 AIMessage。

        一轮聊天在发生工具调用时，通常会先产生一条携带 `tool_calls` 的 AIMessage，
        再在工具返回后产生新的 AIMessage。消费统计若采用“一次上游请求记一条账”
        的口径，就必须保留这些消息的顺序和索引，而不是只取最后一条消息。

        Args:
            response: 智能体最终响应字典。

        Returns:
            由 `(message_index, message)` 组成的列表；若不存在 AIMessage 则返回空列表。
        """

        messages = list(response.get("messages", []) or [])
        result: list[tuple[int, AIMessage]] = []
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage):
                result.append((index, message))
        return result

    @staticmethod
    def _extract_raw_payload(message: AIMessage) -> dict[str, Any] | None:
        """提取 AIMessage 上挂载的原始响应摘要。

        Args:
            message: 需要提取原始响应的 AIMessage。

        Returns:
            原始响应摘要字典；不存在时返回 `None`。
        """

        additional_kwargs = getattr(message, "additional_kwargs", {})
        if not isinstance(additional_kwargs, dict):
            return None
        raw_payload = additional_kwargs.get("raw_response")
        return raw_payload if isinstance(raw_payload, dict) else None

    def _extract_usage_and_request_context(
        self,
        *,
        message: AIMessage,
        usage_rule: ProviderUsageRule,
    ) -> tuple[float, float, float, str | None, dict[str, Any] | None]:
        """从 AIMessage 的多个候选位置提取 usage 与请求 ID。

        当前项目同时支持流式与非流式调用，而不同路径下 LangChain 挂载 token 用量
        的位置并不一致。这里按 `raw_response.body_json`、`response_metadata.token_usage`
        和 `usage_metadata` 的顺序回退，尽量避免因单一路径缺失导致整笔账单丢失。

        Args:
            message: 当前请求最终生成的 AIMessage。
            usage_rule: 当前供应商的 usage 提取规则。

        Returns:
            `(input_tokens, output_tokens, cache_read_tokens, request_id, usage_snapshot)`。
            若未找到任何可用 usage，则返回全 0 与 `None`。
        """

        non_streaming_rule = self._select_usage_path_set(usage_rule=usage_rule, mode="non_streaming")
        streaming_rule = self._select_usage_path_set(usage_rule=usage_rule, mode="streaming")

        raw_payload = self._extract_raw_payload(message)
        if raw_payload is not None:
            body_json = raw_payload.get("body_json")
            if isinstance(body_json, dict):
                extracted = self._extract_usage_from_payload(
                    payload=body_json,
                    raw_payload=raw_payload,
                    path_set=non_streaming_rule,
                )
                if extracted is not None:
                    input_tokens, output_tokens, cache_read_tokens, request_id = extracted
                    return input_tokens, output_tokens, cache_read_tokens, request_id, body_json

        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                wrapped_payload = {
                    "id": response_metadata.get("id"),
                    "usage": token_usage,
                }
                extracted = self._extract_usage_from_payload(
                    payload=wrapped_payload,
                    raw_payload=response_metadata,
                    path_set=non_streaming_rule,
                )
                if extracted is not None:
                    input_tokens, output_tokens, cache_read_tokens, request_id = extracted
                    if request_id is None:
                        request_id = self._normalize_optional_str(response_metadata.get("id"))
                    return input_tokens, output_tokens, cache_read_tokens, request_id, wrapped_payload

        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            extracted = self._extract_usage_from_payload(
                payload=usage_metadata,
                raw_payload=usage_metadata,
                path_set=streaming_rule,
            )
            if extracted is not None:
                input_tokens, output_tokens, cache_read_tokens, request_id = extracted
                return input_tokens, output_tokens, cache_read_tokens, request_id, dict(usage_metadata)

            input_total_tokens = self._read_path_as_non_negative_number(usage_metadata, "input_tokens")
            output_tokens = self._read_path_as_non_negative_number(usage_metadata, "output_tokens")
            cache_read_tokens = self._read_path_as_non_negative_number(
                usage_metadata,
                "input_token_details.cache_read",
            )
            input_tokens = max(0.0, input_total_tokens - cache_read_tokens)
            if input_tokens > 0 or output_tokens > 0 or cache_read_tokens > 0:
                return input_tokens, output_tokens, cache_read_tokens, None, dict(usage_metadata)

        return 0.0, 0.0, 0.0, None, None

    @staticmethod
    def _select_usage_path_set(*, usage_rule: ProviderUsageRule, mode: str) -> UsagePathSet:
        """根据流式或非流式模式选出当前应使用的路径集合。"""

        if mode == "streaming" and usage_rule.streaming is not None:
            return usage_rule.streaming
        if mode == "non_streaming" and usage_rule.non_streaming is not None:
            return usage_rule.non_streaming
        return UsagePathSet(
            input_tokens_path=usage_rule.input_tokens_path,
            output_tokens_path=usage_rule.output_tokens_path,
            cache_read_tokens_path=usage_rule.cache_read_tokens_path,
            request_id_path=usage_rule.request_id_path,
            input_tokens_include_cache_read=usage_rule.input_tokens_include_cache_read,
        )

    def _extract_usage_from_payload(
        self,
        *,
        payload: dict[str, Any],
        raw_payload: dict[str, Any],
        path_set: UsagePathSet,
    ) -> tuple[float, float, float, str | None] | None:
        """按给定路径集合从指定 payload 中提取 token 用量与请求 ID。"""

        input_tokens = self._read_path_as_non_negative_number(payload, path_set.input_tokens_path)
        output_tokens = self._read_path_as_non_negative_number(payload, path_set.output_tokens_path)
        cache_read_tokens = self._read_path_as_non_negative_number(payload, path_set.cache_read_tokens_path)
        if path_set.input_tokens_include_cache_read and input_tokens > 0 and cache_read_tokens > 0:
            input_tokens = max(0.0, input_tokens - cache_read_tokens)
        request_id = self._extract_request_id(
            raw_payload=raw_payload,
            body_json=payload,
            request_id_path=path_set.request_id_path,
        )
        if input_tokens <= 0 and output_tokens <= 0 and cache_read_tokens <= 0:
            return None
        return input_tokens, output_tokens, cache_read_tokens, request_id

    @staticmethod
    def _read_path_as_non_negative_number(payload: Any, path: str) -> float:
        """按点号路径从嵌套字典中读取非负数值。

        路径缺失、类型不符或值为负数时统一按 `0.0` 处理，避免因为供应商字段差异让
        账本写入抛出异常。

        Args:
            payload: 待读取的嵌套对象。
            path: 以点号分隔的字段路径。

        Returns:
            解析出的非负浮点数；异常或缺失时返回 `0.0`。
        """

        if not path:
            return 0.0
        current = payload
        for token in str(path).split("."):
            if not token:
                return 0.0
            if not isinstance(current, dict) or token not in current:
                return 0.0
            current = current[token]
        try:
            value = float(current)
        except Exception:
            return 0.0
        return value if value > 0 else 0.0

    @staticmethod
    def _extract_request_id(
        *,
        raw_payload: dict[str, Any],
        body_json: dict[str, Any],
        request_id_path: str,
    ) -> str | None:
        """按配置规则和兼容回退顺序提取请求 ID。

        Args:
            raw_payload: 挂载在 AIMessage 上的原始响应摘要。
            body_json: 原始响应体 JSON。
            usage_rule: 当前供应商的 usage 提取规则。

        Returns:
            请求 ID；若无法确定则返回 `None`。
        """

        request_id = None
        if request_id_path:
            current: Any = body_json
            for token in request_id_path.split("."):
                if not token or not isinstance(current, dict) or token not in current:
                    current = None
                    break
                current = current[token]
            if isinstance(current, str) and current.strip():
                request_id = current.strip()
        if request_id:
            return request_id
        fallback = raw_payload.get("request_id")
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return None

    @staticmethod
    def _normalize_optional_int(value: Any) -> int | None:
        """将可选值安全归一化为正整数或 `None`。

        Args:
            value: 待归一化的原始值。

        Returns:
            正整数时返回对应值，否则返回 `None`。
        """

        try:
            normalized = int(value)
        except Exception:
            return None
        return normalized if normalized > 0 else None

    @staticmethod
    def _normalize_optional_str(value: Any) -> str | None:
        """将可选值安全归一化为非空字符串或 `None`。

        Args:
            value: 待归一化的原始值。

        Returns:
            非空字符串时返回对应值，否则返回 `None`。
        """

        normalized = str(value or "").strip()
        return normalized or None

    @staticmethod
    def _normalize_money(value: float) -> float:
        """将金额归一化到固定精度，避免日志和排行榜中出现明显浮点噪声。

        Args:
            value: 待归一化的原始金额。

        Returns:
            保留 8 位小数后的金额。
        """

        return round(float(value), 8)


_cost_ledger_service: CostLedgerService | None = None


def get_cost_ledger_service() -> CostLedgerService:
    """返回全局共享的消费账本服务实例。

    Returns:
        单例模式的消费账本服务对象。
    """

    global _cost_ledger_service
    if _cost_ledger_service is None:
        _cost_ledger_service = CostLedgerService()
    return _cost_ledger_service
