from __future__ import annotations

from typing import Any

from ..cost import CostBillingMode, CostLedgerService, get_cost_ledger_service


def get_cost_service() -> CostLedgerService:
    """返回 GTBot 统一消费账本服务实例。

    Returns:
        插件可复用的消费账本服务对象。
    """

    return get_cost_ledger_service()


async def record_cost(
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
    """按显式参数写入一条插件消费记录。

    该接口适合后台任务、定时任务或与当前会话无直接上下文绑定的插件逻辑。

    Args:
        source_name: 账单来源名称。
        category: 账单分类。
        billing_mode: 计费模式。
        quantity: 原始计费数量。
        unit_price: 单价；`direct_amount` 模式允许为空。
        amount: 最终金额。
        owner_user_id: 账单归属用户。
        actor_user_id: 实际触发用户；未传入时默认与归属用户一致。
        occurred_at: 可选消费发生时间。
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

    return await get_cost_service().record_plugin_cost(
        source_name=source_name,
        category=category,
        billing_mode=billing_mode,
        quantity=quantity,
        unit_price=unit_price,
        amount=amount,
        owner_user_id=owner_user_id,
        actor_user_id=actor_user_id,
        occurred_at=occurred_at,
        group_id=group_id,
        session_id=session_id,
        response_id=response_id,
        provider=provider,
        model_name=model_name,
        extra=extra,
        event_id=event_id,
    )


async def record_cost_for_current_request(
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
    """基于当前插件请求上下文写入一条消费记录。

    该接口会自动补齐当前请求的 `user_id`、`group_id`、`session_id` 和 `response_id`，
    适合在 GTBot 插件 tool 或插件内部异步流程中直接调用。

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
    """

    return await get_cost_service().record_cost_for_current_request(
        source_name=source_name,
        category=category,
        billing_mode=billing_mode,
        quantity=quantity,
        unit_price=unit_price,
        amount=amount,
        owner_user_id=owner_user_id,
        provider=provider,
        model_name=model_name,
        extra=extra,
        event_id=event_id,
    )


__all__ = [
    "CostBillingMode",
    "CostLedgerService",
    "get_cost_service",
    "record_cost",
    "record_cost_for_current_request",
]
