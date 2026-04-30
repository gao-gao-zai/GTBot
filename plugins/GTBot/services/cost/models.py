from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


CostSourceType: TypeAlias = Literal["chat_api", "plugin", "manual_adjustment"]
"""消费来源类型。"""

CostBillingMode: TypeAlias = Literal[
    "per_million_tokens",
    "per_image",
    "per_request",
    "direct_amount",
]
"""消费计费模式。"""


@dataclass(frozen=True, slots=True)
class CostRecord:
    """描述一条已经持久化或即将持久化的消费明细记录。

    该模型是 GTBot 统一账本的最小公共语义载体，既服务于聊天 API 自动计费，
    也服务于各类插件的主动记账。字段设计优先保证追溯、聚合和幂等去重能力，
    而不是面向某个特定插件做强耦合建模。

    Attributes:
        event_id: 业务稳定唯一 ID，用于幂等去重。
        occurred_at: 消费实际发生时间戳。
        source_type: 消费来源类型。
        source_name: 消费来源名称，例如 `gtbot_chat` 或 `openai_draw`。
        category: 业务分类，例如 `llm_tokens` 或 `image_generation`。
        owner_user_id: 账单归属的 QQ 用户 ID，排行榜与个人汇总均按该字段聚合。
        actor_user_id: 实际触发本次消费的 QQ 用户 ID，默认与归属用户一致。
        group_id: 可选群号；全局消费或私聊消费时允许为空。
        session_id: 可选会话 ID，便于按会话回溯。
        response_id: 可选响应 ID，便于按单轮聊天回溯。
        provider: 可选供应商标识，例如 GTBot 配置中的 provider 名称。
        model_name: 可选模型名或资源名。
        billing_mode: 本条消费采用的计费模式。
        quantity: 原始计费数量，例如 token 数、图片张数或请求次数。
        unit_price: 单价；对 `direct_amount` 模式可为空。
        amount: 本条账单的最终金额。
        currency: 币种。当前版本固定为 `CNY`。
        extra: 额外扩展数据，会序列化保存到 JSON 字段中。
    """

    event_id: str
    occurred_at: float
    source_type: CostSourceType
    source_name: str
    category: str
    owner_user_id: int
    actor_user_id: int
    group_id: int | None
    session_id: str | None
    response_id: str | None
    provider: str | None
    model_name: str | None
    billing_mode: CostBillingMode
    quantity: float
    unit_price: float | None
    amount: float
    currency: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CostSummary:
    """描述某个时间范围内的消费汇总结果。

    该结果用于个人总览、按来源拆分统计和排行榜汇总等场景，因此只保留
    汇总所需的最小字段，并显式带上统计口径中的计数值，便于后续扩展展示。

    Attributes:
        total_amount: 汇总后的总金额。
        currency: 汇总币种。当前版本固定为 `CNY`。
        record_count: 纳入本次统计的账单条数。
    """

    total_amount: float
    currency: str
    record_count: int


@dataclass(frozen=True, slots=True)
class CostLeaderboardEntry:
    """描述排行榜中的单个用户聚合结果。

    Attributes:
        owner_user_id: 上榜用户的 QQ 用户 ID。
        total_amount: 指定统计口径下的消费总金额。
        currency: 汇总币种。当前版本固定为 `CNY`。
        record_count: 聚合的账单条数。
    """

    owner_user_id: int
    total_amount: float
    currency: str
    record_count: int
