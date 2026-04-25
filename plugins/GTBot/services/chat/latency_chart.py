from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ...ConfigManager import total_config


LabelResolver = Callable[[str], str]

_TITLE_COLOR = "#111827"
_TEXT_COLOR = "#374151"
_SUBTLE_TEXT_COLOR = "#6B7280"
_GRID_COLOR = "#D7E0EA"
_BAR_FILL = "#2F7CFF"
_BAR_STROKE = "#1E4FBF"
_TOTAL_LINE = "#DC2626"
_BG_COLOR = "#FFFFFF"
_PLOT_BG_COLOR = "#F8FAFC"


@dataclass(slots=True)
class GanttRenderResult:
    """描述单次聊天请求甘特图的渲染结果。

    当前实现只使用 Pillow 生成 PNG，因此状态只区分数据缺失、绘图失败和绘图成功，
    不再依赖 Plotly/Kaleido 或外部浏览器进程。

    Attributes:
        status: 渲染状态。常见取值为 `success`、`no_last_completed`、
            `no_stage_spans`、`build_failed`。
        image_path: 成功导出时的本地图片路径。
        detail: 失败时的补充说明，便于命令层定位问题。
    """

    status: str
    image_path: Path | None = None
    detail: str = ""


@dataclass(slots=True)
class _NormalizedSpan:
    """表示已标准化的单个阶段时间片段。"""

    stage_name: str
    start_ms: float
    end_ms: float
    duration_ms: float


def _get_latency_chart_output_dir() -> Path:
    """返回聊天延迟图表的落盘目录。

    Returns:
        Path: 图表输出目录的绝对路径。
    """

    data_dir = total_config.get_data_dir_path()
    output_dir = data_dir / "chat_latency"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_stage_spans(
    last_completed: dict[str, Any],
    *,
    stage_name_resolver: LabelResolver,
) -> list[_NormalizedSpan]:
    """将快照中的阶段时间线标准化为可绘图结构。

    Args:
        last_completed: 最近一次完成请求的快照字典。
        stage_name_resolver: 阶段名中文映射函数。

    Returns:
        list[_NormalizedSpan]: 已过滤无效值并完成排序的时间片段列表。
    """

    normalized_rows: list[_NormalizedSpan] = []
    for item in list(last_completed.get("stage_spans") or []):
        try:
            stage_name = stage_name_resolver(str(item.get("name") or ""))
            start_ms = max(0.0, float(item.get("start_ms") or 0.0))
            duration_ms = max(0.0, float(item.get("duration_ms") or 0.0))
            end_ms = max(start_ms, float(item.get("end_ms") or (start_ms + duration_ms)))
        except Exception:
            continue
        normalized_rows.append(
            _NormalizedSpan(
                stage_name=stage_name,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
            )
        )
    normalized_rows.sort(key=lambda item: (item.start_ms, item.stage_name))
    return normalized_rows


def _load_pillow_font(size: int):
    """加载 Pillow 用于绘图的字体。

    Args:
        size: 目标字号。

    Returns:
        ImageFont.FreeTypeFont | ImageFont.ImageFont: 可直接用于绘制文本的字体对象。
    """

    from PIL import ImageFont

    candidate_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in candidate_paths:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure_text(draw, text: str, font) -> tuple[int, int]:
    """测量文本在当前字体下的像素尺寸。"""

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left), max(0, bottom - top)


def _draw_pillow_gantt(
    last_completed: dict[str, Any],
    *,
    stage_name_resolver: LabelResolver,
    outcome_name_resolver: LabelResolver,
    trigger_mode_resolver: LabelResolver,
    chat_type_resolver: LabelResolver,
    target: Path,
) -> None:
    """使用 Pillow 渲染最近一次完成请求的甘特图。

    Args:
        last_completed: 最近一次完成请求的快照字典。
        stage_name_resolver: 阶段名中文映射函数。
        outcome_name_resolver: 结果状态中文映射函数。
        trigger_mode_resolver: 触发模式中文映射函数。
        chat_type_resolver: 会话类型中文映射函数。
        target: 输出 PNG 的目标路径。

    Raises:
        ValueError: 当阶段时间线为空或没有可用片段时抛出。
    """

    from PIL import Image, ImageDraw

    normalized_rows = _normalize_stage_spans(
        last_completed,
        stage_name_resolver=stage_name_resolver,
    )
    if not normalized_rows:
        raise ValueError("最近一次完成请求缺少可用阶段时间线")

    total_ms = max(float(last_completed.get("total_ms") or 0.0), 1.0)
    response_id = str(last_completed.get("response_id") or "unknown")
    outcome = outcome_name_resolver(str(last_completed.get("outcome") or ""))
    trigger_mode = trigger_mode_resolver(str(last_completed.get("trigger_mode") or ""))
    chat_type = chat_type_resolver(str(last_completed.get("chat_type") or ""))

    title_font = _load_pillow_font(28)
    subtitle_font = _load_pillow_font(18)
    axis_font = _load_pillow_font(16)
    label_font = _load_pillow_font(18)
    meta_font = _load_pillow_font(15)

    width = 1280
    left_margin = 250
    right_margin = 80
    top_margin = 140
    bottom_margin = 90
    row_height = 54
    plot_height = len(normalized_rows) * row_height
    height = max(420, top_margin + plot_height + bottom_margin)
    plot_left = left_margin
    plot_top = top_margin
    plot_right = width - right_margin
    plot_bottom = plot_top + plot_height
    plot_width = max(1, plot_right - plot_left)

    image = Image.new("RGB", (width, height), _BG_COLOR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill=_PLOT_BG_COLOR)

    title = "GTBot 单次聊天请求耗时甘特图"
    subtitle = (
        f"请求={response_id} | 结果={outcome} | 类型={chat_type} | "
        f"触发={trigger_mode} | 总耗时={total_ms:.2f}ms"
    )
    title_width, _ = _measure_text(draw, title, title_font)
    subtitle_width, _ = _measure_text(draw, subtitle, subtitle_font)
    draw.text(((width - title_width) / 2, 28), title, fill=_TITLE_COLOR, font=title_font)
    draw.text(((width - subtitle_width) / 2, 68), subtitle, fill=_SUBTLE_TEXT_COLOR, font=subtitle_font)

    tick_count = 6
    for index in range(tick_count + 1):
        ratio = index / tick_count
        x = plot_left + int(round(plot_width * ratio))
        value_ms = total_ms * ratio
        draw.line((x, plot_top, x, plot_bottom), fill=_GRID_COLOR, width=1)
        tick_label = f"{value_ms:.0f} ms"
        tick_width, _ = _measure_text(draw, tick_label, axis_font)
        draw.text((x - tick_width / 2, plot_bottom + 14), tick_label, fill=_SUBTLE_TEXT_COLOR, font=axis_font)

    total_x = plot_right
    draw.line((total_x, plot_top - 8, total_x, plot_bottom), fill=_TOTAL_LINE, width=3)
    total_label = f"总耗时 {total_ms:.2f}ms"
    total_label_width, _ = _measure_text(draw, total_label, axis_font)
    draw.rounded_rectangle(
        (total_x - total_label_width - 18, plot_top - 44, total_x - 6, plot_top - 12),
        radius=8,
        fill="#FEE2E2",
        outline="#FCA5A5",
        width=1,
    )
    draw.text((total_x - total_label_width - 12, plot_top - 39), total_label, fill=_TOTAL_LINE, font=axis_font)

    for row_index, row in enumerate(normalized_rows):
        row_top = plot_top + row_index * row_height
        row_center_y = row_top + row_height / 2
        bar_left = plot_left + int(round((row.start_ms / total_ms) * plot_width))
        bar_right = plot_left + int(round((row.end_ms / total_ms) * plot_width))
        bar_right = max(bar_left + 2, min(plot_right, bar_right))
        bar_top = row_top + 11
        bar_bottom = row_top + row_height - 11

        stage_width, stage_height = _measure_text(draw, row.stage_name, label_font)
        draw.text(
            (left_margin - stage_width - 18, row_center_y - stage_height / 2),
            row.stage_name,
            fill=_TEXT_COLOR,
            font=label_font,
        )

        draw.rounded_rectangle(
            (bar_left, bar_top, bar_right, bar_bottom),
            radius=8,
            fill=_BAR_FILL,
            outline=_BAR_STROKE,
            width=2,
        )

        meta_text = f"{row.start_ms:.1f} - {row.end_ms:.1f}ms | {row.duration_ms:.1f}ms"
        meta_width, meta_height = _measure_text(draw, meta_text, meta_font)
        meta_x = min(plot_right - meta_width, bar_right + 10)
        meta_x = max(plot_left + 6, meta_x)
        draw.text((meta_x, row_center_y - meta_height / 2), meta_text, fill=_SUBTLE_TEXT_COLOR, font=meta_font)

    axis_label = "耗时（毫秒）"
    axis_label_width, _ = _measure_text(draw, axis_label, axis_font)
    draw.text((plot_left + (plot_width - axis_label_width) / 2, height - 42), axis_label, fill=_TEXT_COLOR, font=axis_font)
    image.save(target, format="PNG")


def prepare_last_completed_latency_gantt(
    snapshot: dict[str, Any],
    *,
    stage_name_resolver: LabelResolver,
    outcome_name_resolver: LabelResolver,
    trigger_mode_resolver: LabelResolver,
    chat_type_resolver: LabelResolver,
) -> GanttRenderResult:
    """准备最近一次完成请求的甘特图，并返回结构化结果。

    Args:
        snapshot: `ChatLatencyMonitor.snapshot()` 返回的完整快照。
        stage_name_resolver: 阶段名中文映射函数。
        outcome_name_resolver: 结果状态中文映射函数。
        trigger_mode_resolver: 触发模式中文映射函数。
        chat_type_resolver: 会话类型中文映射函数。

    Returns:
        GanttRenderResult: 包含渲染状态、图片路径和补充说明的结果对象。
    """

    last_completed = snapshot.get("last_completed")
    if not isinstance(last_completed, dict):
        return GanttRenderResult(status="no_last_completed", detail="当前还没有已完成请求样本。")
    if not list(last_completed.get("stage_spans") or []):
        return GanttRenderResult(status="no_stage_spans", detail="最近一次完成请求没有阶段时间线。")

    response_id = str(last_completed.get("response_id") or "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = _get_latency_chart_output_dir() / f"{timestamp}_{response_id}_gantt.png"

    try:
        _draw_pillow_gantt(
            last_completed,
            stage_name_resolver=stage_name_resolver,
            outcome_name_resolver=outcome_name_resolver,
            trigger_mode_resolver=trigger_mode_resolver,
            chat_type_resolver=chat_type_resolver,
            target=target,
        )
    except Exception as exc:
        return GanttRenderResult(status="build_failed", detail=str(exc))
    return GanttRenderResult(status="success", image_path=target)


def render_last_completed_latency_gantt(
    snapshot: dict[str, Any],
    *,
    stage_name_resolver: LabelResolver,
    outcome_name_resolver: LabelResolver,
    trigger_mode_resolver: LabelResolver,
    chat_type_resolver: LabelResolver,
) -> Path | None:
    """兼容旧调用方，返回最近一次完成请求的甘特图路径。

    Args:
        snapshot: `ChatLatencyMonitor.snapshot()` 返回的完整快照。
        stage_name_resolver: 阶段名中文映射函数。
        outcome_name_resolver: 结果状态中文映射函数。
        trigger_mode_resolver: 触发模式中文映射函数。
        chat_type_resolver: 会话类型中文映射函数。

    Returns:
        Path | None: 成功时返回图片路径，否则返回 `None`。
    """

    result = prepare_last_completed_latency_gantt(
        snapshot,
        stage_name_resolver=stage_name_resolver,
        outcome_name_resolver=outcome_name_resolver,
        trigger_mode_resolver=trigger_mode_resolver,
        chat_type_resolver=chat_type_resolver,
    )
    return result.image_path
