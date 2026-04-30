from __future__ import annotations

from datetime import datetime

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from plugins.GTBot.services.cost import CostLeaderboardEntry, CostRecord, CostSummary, get_cost_ledger_service
from plugins.GTBot.services.help import HelpArgumentSpec, HelpCommandSpec, register_help
from plugins.GTBot.services.plugin_api.permissions import PermissionError, PermissionRole


CostCommand = on_command("消费", priority=4, block=True)

register_help(
    HelpCommandSpec(
        name="消费",
        category="消费统计",
        summary="查看个人消费总览、明细和排行榜。",
        description=(
            "普通成员可查询自己的总消费与明细，并查看完整排行榜；GTBot 管理员及以上可查询任意用户。"
            "排行榜默认查看全部时间全局榜，也支持按本群和时间范围过滤。"
        ),
        arguments=(
            HelpArgumentSpec(
                name="[今日|本周|本月|全部]",
                description="查看自己的汇总消费；留空时默认查看全部时间。",
                required=False,
            ),
            HelpArgumentSpec(
                name="[明细]",
                description="查看自己的最近消费明细，可继续跟时间范围。",
                required=False,
            ),
            HelpArgumentSpec(
                name="[排行]",
                description="查看消费排行榜，可继续跟本群或时间范围。",
                required=False,
            ),
        ),
        examples=(
            "/消费",
            "/消费 今日",
            "/消费 明细 本周",
            "/消费 排行",
            "/消费 排行 本群",
            "/消费 用户 123456 今日",
        ),
        required_role=PermissionRole.USER,
        audience="群聊和私聊",
        sort_key=30,
    )
)


def _format_ts(timestamp: float) -> str:
    """将时间戳格式化为便于 QQ 文本阅读的本地时间字符串。

    Args:
        timestamp: 需要格式化的 Unix 时间戳。

    Returns:
        形如 `YYYY-MM-DD HH:MM` 的本地时间文本。
    """

    return datetime.fromtimestamp(float(timestamp)).astimezone().strftime("%Y-%m-%d %H:%M")


def _format_amount(amount: float) -> str:
    """统一格式化人民币金额，避免输出过长的小数。 

    Args:
        amount: 待展示的金额。

    Returns:
        去除多余尾随零后的金额文本。
    """

    text = f"{float(amount):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_range_label(start_at: float | None, end_at: float | None) -> str:
    """将时间范围转换为展示标题中的简短标签。

    Args:
        start_at: 起始时间戳，包含该时刻；为 `None` 表示无下界。
        end_at: 结束时间戳，不包含该时刻；为 `None` 表示无上界。

    Returns:
        适合直接展示在命令回复标题中的范围描述。
    """

    if start_at is None and end_at is None:
        return "全部时间"
    if start_at is not None and end_at is not None:
        display_end_at = max(float(start_at), float(end_at) - 1.0)
        end_text = datetime.fromtimestamp(display_end_at).astimezone().strftime("%Y-%m-%d")
        start_text = datetime.fromtimestamp(start_at).astimezone().strftime("%Y-%m-%d")
        return f"{start_text} ~ {end_text}"
    if start_at is not None:
        return f"{datetime.fromtimestamp(start_at).astimezone():%Y-%m-%d} 起"
    return f"截至 {datetime.fromtimestamp(end_at or 0).astimezone():%Y-%m-%d}"


def _parse_range_tokens(tokens: list[str]) -> tuple[float | None, float | None, int]:
    """从命令参数中解析命名时间范围或自定义日期区间。

    Args:
        tokens: 尚未消费的参数列表。

    Returns:
        `(start_at, end_at, consumed)`，其中 `consumed` 表示消耗的 token 数。

    Raises:
        ValueError: 当时间范围格式不合法时抛出。
    """

    if not tokens:
        return None, None, 0
    service = get_cost_ledger_service()
    first = str(tokens[0]).strip()
    if first in {"今日", "本周", "本月", "全部"}:
        start_at, end_at = service.resolve_named_range(first)
        return start_at, end_at, 1
    if len(tokens) >= 2:
        start_at, end_at = service.parse_date_range(tokens[0], tokens[1])
        return start_at, end_at, 2
    raise ValueError("时间范围格式不正确，请使用 今日/本周/本月/全部 或 YYYY-MM-DD YYYY-MM-DD")


async def _resolve_target_label(bot: Bot, *, target_user_id: int, group_id: int | None) -> str:
    """解析消费统计中显示的用户标签。

    Args:
        bot: 当前 OneBot Bot 实例。
        target_user_id: 目标 QQ 用户 ID。
        group_id: 当前群号；存在时优先使用群名片解析。

    Returns:
        形如 `昵称(123456)` 的展示文本。
    """

    service = get_cost_ledger_service()
    display_name = await service.resolve_user_display_name(bot=bot, user_id=target_user_id, group_id=group_id)
    return f"{display_name}({target_user_id})"


def _render_summary(
    *,
    title: str,
    summary: CostSummary,
    breakdown: list[tuple[str, CostSummary]],
) -> str:
    """将用户汇总和来源拆分统计渲染为多行文本。

    Args:
        title: 回复标题。
        summary: 汇总统计结果。
        breakdown: 按来源拆分的汇总统计。

    Returns:
        适合直接发送到 QQ 的多行摘要文本。
    """

    lines = [
        title,
        f"总金额: {_format_amount(summary.total_amount)} {summary.currency}",
        f"记录数: {summary.record_count}",
    ]
    if breakdown:
        lines.append("来源拆分:")
        for source_name, item in breakdown:
            lines.append(
                f"- {source_name}: {_format_amount(item.total_amount)} {item.currency} / {item.record_count} 条"
            )
    else:
        lines.append("来源拆分: 暂无记录")
    return "\n".join(lines)


def _render_records(*, title: str, records: list[CostRecord]) -> str:
    """将消费明细列表渲染为文本。

    Args:
        title: 回复标题。
        records: 已按时间倒序排列的账单明细。

    Returns:
        适合直接发送到 QQ 的明细文本。
    """

    lines = [title]
    if not records:
        lines.append("暂无消费记录。")
        return "\n".join(lines)
    for item in records:
        scope = f" 群={item.group_id}" if item.group_id is not None else ""
        model = f" 模型={item.model_name}" if item.model_name else ""
        response = f" 响应={item.response_id}" if item.response_id else ""
        lines.append(
            f"- {_format_ts(item.occurred_at)} {item.source_name}/{item.category} "
            f"{_format_amount(item.amount)} {item.currency}{scope}{model}{response}"
        )
    return "\n".join(lines)


def _render_leaderboard(
    *,
    title: str,
    entries: list[tuple[CostLeaderboardEntry, str]],
) -> str:
    """将排行榜条目渲染为文本。

    Args:
        title: 回复标题。
        entries: 已补齐展示名的排行榜条目。

    Returns:
        适合直接发送到 QQ 的排行榜文本。
    """

    lines = [title]
    if not entries:
        lines.append("暂无消费记录。")
        return "\n".join(lines)
    for index, (entry, display_name) in enumerate(entries, start=1):
        lines.append(
            f"{index}. {display_name} - {_format_amount(entry.total_amount)} {entry.currency} "
            f"({entry.record_count} 条)"
        )
    return "\n".join(lines)


async def _handle_summary(
    *,
    bot: Bot,
    requester_user_id: int,
    target_user_id: int,
    start_at: float | None,
    end_at: float | None,
    group_id: int | None,
) -> str:
    """执行个人消费汇总查询并生成回复文本。

    Args:
        bot: 当前 OneBot Bot 实例。
        requester_user_id: 发起查询的用户 ID。
        target_user_id: 被查询账单归属用户 ID。
        start_at: 起始时间戳。
        end_at: 结束时间戳。
        group_id: 可选群范围过滤。

    Returns:
        汇总查询的回复文本。
    """

    service = get_cost_ledger_service()
    summary = await service.get_user_summary(
        requester_user_id=requester_user_id,
        target_user_id=target_user_id,
        start_at=start_at,
        end_at=end_at,
        group_id=group_id,
    )
    breakdown = await service.get_user_breakdown_by_source(
        requester_user_id=requester_user_id,
        target_user_id=target_user_id,
        start_at=start_at,
        end_at=end_at,
        group_id=group_id,
    )
    target_label = await _resolve_target_label(bot, target_user_id=target_user_id, group_id=group_id)
    scope_label = "本群" if group_id is not None else "全局"
    range_label = _format_range_label(start_at, end_at)
    return _render_summary(
        title=f"{target_label} 的消费汇总 [{scope_label} / {range_label}]",
        summary=summary,
        breakdown=breakdown,
    )


async def _handle_details(
    *,
    bot: Bot,
    requester_user_id: int,
    target_user_id: int,
    start_at: float | None,
    end_at: float | None,
    group_id: int | None,
) -> str:
    """执行个人消费明细查询并生成回复文本。

    Args:
        bot: 当前 OneBot Bot 实例。
        requester_user_id: 发起查询的用户 ID。
        target_user_id: 被查询账单归属用户 ID。
        start_at: 起始时间戳。
        end_at: 结束时间戳。
        group_id: 可选群范围过滤。

    Returns:
        明细查询的回复文本。
    """

    service = get_cost_ledger_service()
    records = await service.list_user_records(
        requester_user_id=requester_user_id,
        target_user_id=target_user_id,
        start_at=start_at,
        end_at=end_at,
        group_id=group_id,
        limit=20,
    )
    target_label = await _resolve_target_label(bot, target_user_id=target_user_id, group_id=group_id)
    scope_label = "本群" if group_id is not None else "全局"
    range_label = _format_range_label(start_at, end_at)
    return _render_records(
        title=f"{target_label} 的消费明细 [{scope_label} / {range_label}]",
        records=records,
    )


async def _handle_leaderboard(
    *,
    bot: Bot,
    start_at: float | None,
    end_at: float | None,
    group_id: int | None,
) -> str:
    """执行排行榜查询并生成回复文本。

    Args:
        bot: 当前 OneBot Bot 实例。
        start_at: 起始时间戳。
        end_at: 结束时间戳。
        group_id: 可选群范围过滤。

    Returns:
        排行榜回复文本。
    """

    service = get_cost_ledger_service()
    leaderboard = await service.get_leaderboard(
        start_at=start_at,
        end_at=end_at,
        group_id=group_id,
        limit=20,
    )
    display_entries: list[tuple[CostLeaderboardEntry, str]] = []
    for item in leaderboard:
        display_name = await service.resolve_user_display_name(
            bot=bot,
            user_id=item.owner_user_id,
            group_id=group_id,
        )
        display_entries.append((item, f"{display_name}({item.owner_user_id})"))
    scope_label = "本群" if group_id is not None else "全局"
    range_label = _format_range_label(start_at, end_at)
    return _render_leaderboard(
        title=f"消费排行榜 [{scope_label} / {range_label}]",
        entries=display_entries,
    )


async def _parse_and_execute(bot: Bot, event: MessageEvent, tokens: list[str]) -> str:
    """解析 `/消费` 命令参数并执行对应查询。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 触发命令的消息事件。
        tokens: 纯文本分词后的参数列表。

    Returns:
        最终需要回复给用户的文本。

    Raises:
        PermissionError: 当用户越权查询他人账单时抛出。
        ValueError: 当命令参数不合法时抛出。
    """

    requester_user_id = int(event.user_id)
    current_group_id_raw = getattr(event, "group_id", None)
    current_group_id = int(current_group_id_raw) if current_group_id_raw is not None else None

    if not tokens:
        return await _handle_summary(
            bot=bot,
            requester_user_id=requester_user_id,
            target_user_id=requester_user_id,
            start_at=None,
            end_at=None,
            group_id=None,
        )

    if tokens[0] == "排行":
        remaining = tokens[1:]
        group_scope = None
        if remaining and remaining[0] == "本群":
            if current_group_id is None:
                raise ValueError("当前不在群聊中，无法查询本群排行榜")
            group_scope = current_group_id
            remaining = remaining[1:]
        start_at, end_at, consumed = _parse_range_tokens(remaining)
        if remaining and consumed == 0:
            raise ValueError("排行参数不正确，请使用 /消费 排行 [本群] [今日|本周|本月|全部|起始日期 结束日期]")
        if consumed != len(remaining):
            raise ValueError("排行参数过多，请检查输入格式")
        return await _handle_leaderboard(bot=bot, start_at=start_at, end_at=end_at, group_id=group_scope)

    target_user_id = requester_user_id
    remaining_tokens = list(tokens)
    if remaining_tokens[0] == "用户":
        if len(remaining_tokens) < 2:
            raise ValueError("请在 /消费 用户 后提供 QQ 号")
        try:
            target_user_id = int(remaining_tokens[1])
        except ValueError as exc:
            raise ValueError("用户 QQ 号必须为整数") from exc
        remaining_tokens = remaining_tokens[2:]

    detail_mode = False
    if remaining_tokens and remaining_tokens[0] == "明细":
        detail_mode = True
        remaining_tokens = remaining_tokens[1:]

    start_at, end_at, consumed = _parse_range_tokens(remaining_tokens)
    if consumed != len(remaining_tokens):
        raise ValueError("消费命令参数格式不正确")

    if detail_mode:
        return await _handle_details(
            bot=bot,
            requester_user_id=requester_user_id,
            target_user_id=target_user_id,
            start_at=start_at,
            end_at=end_at,
            group_id=None,
        )
    return await _handle_summary(
        bot=bot,
        requester_user_id=requester_user_id,
        target_user_id=target_user_id,
        start_at=start_at,
        end_at=end_at,
        group_id=None,
    )


@CostCommand.handle()
async def handle_cost_command(bot: Bot, event: MessageEvent, args: Message = CommandArg()) -> None:
    """处理统一消费查询命令。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 触发命令的消息事件。
        args: 命令后的剩余参数。
    """

    arg_text = args.extract_plain_text().strip()
    tokens = [item for item in arg_text.split() if item]
    try:
        rendered = await _parse_and_execute(bot, event, tokens)
    except PermissionError as exc:
        await CostCommand.finish(str(exc))
    except ValueError as exc:
        await CostCommand.finish(str(exc))
    await CostCommand.finish(rendered)
