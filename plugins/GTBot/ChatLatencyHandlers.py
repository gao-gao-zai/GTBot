from __future__ import annotations

from typing import Any

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import MessageEvent

from local_plugins.nonebot_plugin_gt_help import HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import (
    PermissionError,
    PermissionRole,
    get_permission_manager,
)

from .services.chat.latency_chart import GanttRenderResult, prepare_last_completed_latency_gantt
from .services.chat.latency_monitor import get_chat_latency_monitor


ChatLatencyCommand = on_command("查看聊天延迟", aliases={"聊天延迟", "响应耗时"}, priority=4, block=True)
ChatLatencyGanttCommand = on_command("查看聊天甘特图", aliases={"聊天甘特图", "甘特图"}, priority=4, block=True)

_STAGE_LABELS: dict[str, str] = {
    "lock_wait_or_reject_check": "锁检查/拒绝判定",
    "processing_emoji": "处理中表情",
    "load_turn_messages": "读取上下文消息",
    "parse_streaming_settings": "解析流式配置",
    "build_runtime_context": "构建运行时上下文",
    "build_plugin_context": "构建插件上下文",
    "build_history_text": "格式化历史文本",
    "build_plugin_bundle": "构建插件集合",
    "create_agent": "创建智能体",
    "agent_prep_total": "智能体准备总耗时",
    "pre_agent_processors": "前置处理器",
    "agent_first_token_wait": "等待首字延迟",
    "agent_tool_execution": "工具执行时间",
    "agent_model_output": "模型输出时间",
    "inject_messages": "注入消息",
    "agent_invoke": "模型/智能体执行",
    "completion_emoji": "完成表情",
    "continuation_window": "续聊窗口处理",
    "send_messages_dispatch": "回复分发",
    "total": "总耗时",
    "idle": "空闲",
}

_OUTCOME_LABELS: dict[str, str] = {
    "completed": "成功",
    "failed": "失败",
    "timed_out": "超时",
    "access_denied": "权限拒绝",
    "lock_rejected": "锁拒绝",
}

_TRIGGER_MODE_LABELS: dict[str, str] = {
    "group_at": "群聊@触发",
    "private": "私聊触发",
    "group_keyword": "群聊关键词触发",
    "group_auto": "群聊自动触发",
    "group_continuation": "群聊续聊触发",
    "unknown": "未知触发",
}

_CHAT_TYPE_LABELS: dict[str, str] = {"group": "群聊", "private": "私聊"}


def _register_chat_latency_help_items() -> None:
    """注册聊天延迟和甘特图命令的帮助信息。"""

    register_help(
        HelpCommandSpec(
            name="查看聊天延迟",
            aliases=("聊天延迟", "响应耗时"),
            category="聊天监控",
            summary="查看当前聊天请求的实时阶段耗时与最近窗口平均耗时。",
            description="展示当前执行中的聊天请求、最近 100 次平均耗时，以及最近一次完成请求的摘要。",
            examples=("/查看聊天延迟", "/聊天延迟"),
            required_role=PermissionRole.ADMIN,
            audience="管理员运维命令",
            sort_key=50,
        )
    )
    register_help(
        HelpCommandSpec(
            name="查看聊天甘特图",
            aliases=("聊天甘特图", "甘特图"),
            category="聊天监控",
            summary="查看最近一次完成聊天请求的甘特图图片。",
            description="仅输出最近一次完成请求的阶段耗时甘特图，适合在 QQ 中直接查看时间线。",
            examples=("/查看聊天甘特图", "/甘特图"),
            required_role=PermissionRole.ADMIN,
            audience="管理员运维命令",
            sort_key=51,
        )
    )


_register_chat_latency_help_items()


async def _ensure_admin(user_id: int) -> None:
    """确保当前命令调用者具备管理员权限。

    Args:
        user_id: 触发命令的 QQ 用户 ID。

    Raises:
        PermissionError: 当用户不具备管理员权限时抛出。
    """

    permission_manager = get_permission_manager()
    await permission_manager.require_role(int(user_id), PermissionRole.ADMIN)


def _localize_stage_name(stage_name: str) -> str:
    """将内部阶段名转换为中文展示名称。"""

    normalized = str(stage_name or "").strip()
    if not normalized:
        return "未知阶段"
    if normalized.startswith("pre_agent_processor:"):
        processor_name = normalized.split(":", 1)[1].strip()
        if not processor_name:
            return "前置处理器/未知处理器"
        return f"前置处理器/{processor_name}"
    return _STAGE_LABELS.get(normalized, normalized)


def _localize_outcome_name(outcome: str) -> str:
    """将内部结果状态转换为中文展示名称。"""

    normalized = str(outcome or "").strip()
    if not normalized:
        return "未知结果"
    return _OUTCOME_LABELS.get(normalized, normalized)


def _localize_trigger_mode(trigger_mode: str) -> str:
    """将触发模式转换为中文展示名称。"""

    normalized = str(trigger_mode or "").strip()
    if not normalized:
        return "未知触发"
    return _TRIGGER_MODE_LABELS.get(normalized, normalized)


def _localize_chat_type(chat_type: str) -> str:
    """将会话类型转换为中文展示名称。"""

    normalized = str(chat_type or "").strip()
    if not normalized:
        return "未知会话"
    return _CHAT_TYPE_LABELS.get(normalized, normalized)


def _format_average_stage_lines(stages_ms: dict[str, Any]) -> list[str]:
    """将阶段平均耗时字典格式化为展示文本行。"""

    if not stages_ms:
        return ["- 暂无阶段样本"]

    normalized: list[tuple[str, float]] = []
    for stage_name, value in stages_ms.items():
        try:
            normalized.append((_localize_stage_name(str(stage_name)), float(value)))
        except Exception:
            continue

    if not normalized:
        return ["- 暂无阶段样本"]
    normalized.sort(key=lambda item: (-item[1], item[0]))
    return [f"- {stage}: {duration:.2f}ms" for stage, duration in normalized]


def _format_outcome_counts(outcome_counts: dict[str, Any]) -> str:
    """将结果分布映射格式化为中文文本。"""

    parts: list[str] = []
    for key, value in sorted(outcome_counts.items()):
        try:
            parts.append(f"{_localize_outcome_name(str(key))}={int(value)}")
        except Exception:
            continue
    return ", ".join(parts) or "暂无"


def _render_latency_summary(snapshot: dict[str, Any], *, gantt_status: str | None = None) -> str:
    """将聊天延迟监控快照渲染为命令文字摘要。

    Args:
        snapshot: `ChatLatencyMonitor.snapshot()` 返回的完整快照。
        gantt_status: 当前甘特图渲染状态。仅当成功时才在摘要中提示图片已附带发送。

    Returns:
        str: 适合直接发送到 QQ 的多行摘要文本。
    """

    inflight = list(snapshot.get("inflight") or [])
    recent = dict(snapshot.get("recent") or {})
    last_completed = snapshot.get("last_completed")

    lines: list[str] = [f"当前运行中请求: {int(snapshot.get('inflight_count') or 0)}"]
    if inflight:
        for item in inflight[:10]:
            response_id = str(item.get("response_id") or "")
            session_id = str(item.get("session_id") or "")
            trigger_mode = _localize_trigger_mode(str(item.get("trigger_mode") or ""))
            chat_type = _localize_chat_type(str(item.get("chat_type") or ""))
            current_stage = _localize_stage_name(str(item.get("current_stage") or "idle"))
            elapsed_ms = float(item.get("elapsed_ms") or 0.0)
            lines.append(
                f"- {response_id} 会话={session_id} 类型={chat_type} 触发={trigger_mode} "
                f"当前阶段={current_stage} 已运行={elapsed_ms:.2f}ms"
            )
    else:
        lines.append("- 暂无进行中请求")

    sample_count = int(recent.get("sample_count") or 0)
    average_total_ms = float(recent.get("average_total_ms") or 0.0)
    outcome_counts = dict(recent.get("outcome_counts") or {})

    lines.append("")
    lines.append(f"最近100次平均总耗时: {average_total_ms:.2f}ms (样本数={sample_count})")
    lines.append(f"最近100次结果分布: {_format_outcome_counts(outcome_counts)}")
    lines.append("最近100次阶段平均耗时:")
    lines.extend(_format_average_stage_lines(dict(recent.get("average_stages_ms") or {})))

    lines.append("")
    lines.append("最近一次完成请求:")
    if isinstance(last_completed, dict):
        response_id = str(last_completed.get("response_id") or "")
        outcome = _localize_outcome_name(str(last_completed.get("outcome") or ""))
        total_ms = float(last_completed.get("total_ms") or 0.0)
        trigger_mode = _localize_trigger_mode(str(last_completed.get("trigger_mode") or ""))
        chat_type = _localize_chat_type(str(last_completed.get("chat_type") or ""))
        lines.append(f"- {response_id} 结果={outcome} 类型={chat_type} 触发={trigger_mode} 总耗时={total_ms:.2f}ms")
        lines.extend(_format_average_stage_lines(dict(last_completed.get("stages_ms") or {})))
        if gantt_status == "success":
            lines.append("- 甘特图已作为图片附带发送")
    else:
        lines.append("- 暂无已完成请求样本")

    return "\n".join(lines)


def _prepare_gantt(snapshot: dict[str, Any]) -> GanttRenderResult:
    """准备最近一次完成请求的甘特图渲染结果。"""

    return prepare_last_completed_latency_gantt(
        snapshot,
        stage_name_resolver=_localize_stage_name,
        outcome_name_resolver=_localize_outcome_name,
        trigger_mode_resolver=_localize_trigger_mode,
        chat_type_resolver=_localize_chat_type,
    )


def _build_gantt_reply(result: GanttRenderResult) -> Message | None:
    """将甘特图渲染结果转换为图片消息。"""

    if result.status != "success" or result.image_path is None:
        return None
    return Message(MessageSegment.image(file=result.image_path.as_posix()))


def _render_gantt_error_message(result: GanttRenderResult) -> str:
    """将甘特图失败结果转换为可读提示。"""

    if result.status in {"no_last_completed", "no_stage_spans"}:
        return "暂无可用的聊天甘特图，请先等待至少一条聊天请求完成。"
    if result.status == "build_failed":
        return f"聊天甘特图构建失败：{result.detail or '请检查阶段时间线数据。'}"
    if result.status == "write_failed":
        return f"聊天甘特图导出失败：{result.detail or '请检查 Plotly/Kaleido 环境。'}"
    return "聊天甘特图暂时不可用。"


def _render_latency_reply(snapshot: dict[str, Any]) -> Message:
    """生成聊天延迟摘要命令的回复消息。"""

    gantt_result = _prepare_gantt(snapshot)
    summary_text = _render_latency_summary(snapshot, gantt_status=gantt_result.status)
    reply = Message(MessageSegment.text(summary_text))
    gantt_reply = _build_gantt_reply(gantt_result)
    if gantt_reply is None:
        return reply
    reply.append(MessageSegment.text("\n"))
    for segment in gantt_reply:
        reply.append(segment)
    return reply


@ChatLatencyCommand.handle()
async def handle_chat_latency_command(event: MessageEvent) -> None:
    """处理聊天延迟监控摘要命令。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        await ChatLatencyCommand.finish("你没有查看聊天延迟监控的权限。")
        return

    snapshot = get_chat_latency_monitor().snapshot()
    await ChatLatencyCommand.finish(_render_latency_reply(snapshot))


@ChatLatencyGanttCommand.handle()
async def handle_chat_latency_gantt_command(event: MessageEvent) -> None:
    """处理聊天甘特图查询命令。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        await ChatLatencyGanttCommand.finish("你没有查看聊天甘特图的权限。")
        return

    snapshot = get_chat_latency_monitor().snapshot()
    gantt_result = _prepare_gantt(snapshot)
    gantt_reply = _build_gantt_reply(gantt_result)
    if gantt_reply is None:
        await ChatLatencyGanttCommand.finish(_render_gantt_error_message(gantt_result))
        return
    await ChatLatencyGanttCommand.finish(gantt_reply)
