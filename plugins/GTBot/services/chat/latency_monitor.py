from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter, time
from typing import Any


@dataclass(slots=True)
class StageSpan:
    """描述单次请求里一个阶段片段的起止时间。

    当同一阶段在一次请求中出现多次，监控器会为每次出现分别记录一个片段，
    供命令层绘制甘特图或供日志层输出完整时间线。

    Attributes:
        name: 阶段名称。
        start_ms: 相对请求起点的开始偏移，单位为毫秒。
        end_ms: 相对请求起点的结束偏移，单位为毫秒。
        duration_ms: 当前片段持续时间，单位为毫秒。
    """

    name: str
    start_ms: float
    end_ms: float
    duration_ms: float

    def to_snapshot(self) -> dict[str, float | str]:
        """导出片段的只读快照。

        Returns:
            dict[str, float | str]: 适合命令和日志读取的可序列化字典。
        """

        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(slots=True)
class StageAggregate:
    """聚合单个阶段的累计耗时统计。

    该结构只保存阶段累计耗时和样本数，不持有单次请求明细。
    调用方可以基于它计算平均耗时，并将其序列化给命令层或日志层。

    Attributes:
        total_ms: 当前阶段累计耗时，单位为毫秒。
        count: 当前阶段出现的样本数。
    """

    total_ms: float = 0.0
    count: int = 0

    def add(self, duration_ms: float) -> None:
        """累加一次阶段耗时样本。

        Args:
            duration_ms: 本次阶段耗时，单位为毫秒。调用方应保证传入值非负。
        """

        self.total_ms += max(0.0, float(duration_ms))
        self.count += 1

    def to_snapshot(self) -> dict[str, float | int]:
        """导出当前聚合状态，供命令和日志读取。

        Returns:
            dict[str, float | int]: 包含累计耗时、样本数和平均耗时的只读快照。
        """

        average_ms = self.total_ms / self.count if self.count > 0 else 0.0
        return {
            "total_ms": round(self.total_ms, 3),
            "count": self.count,
            "average_ms": round(average_ms, 3),
        }


@dataclass(slots=True)
class RunningLatencyRecord:
    """表示一个仍在执行中的聊天请求延迟记录。

    该结构维护请求的基础元数据、阶段耗时累计值和当前正在执行的阶段。
    阶段既支持 `start/end` 成对记录，也支持直接累加一个已测得的持续时间，
    以适配并行任务、流式 flush 或外部队列等不方便显式包裹的场景。

    Attributes:
        response_id: 当前请求的唯一响应 ID。
        session_id: 当前请求所属会话 ID。
        trigger_mode: 当前请求的触发模式。
        chat_type: 当前请求的会话类型，如 `group` 或 `private`。
        started_at: 当前请求的墙钟开始时间，单位为 Unix 时间戳秒。
        started_perf: 当前请求的单调时钟开始时间。
        stage_started_perf: 各阶段最近一次开始时间。
        stages_ms: 已累计完成的各阶段耗时，单位为毫秒。
        current_stage: 当前最近一次进入但尚未结束的阶段名称。
    """

    response_id: str
    session_id: str
    trigger_mode: str
    chat_type: str
    started_at: float
    started_perf: float
    stage_started_perf: dict[str, float] = field(default_factory=dict)
    stage_started_offset_ms: dict[str, float] = field(default_factory=dict)
    stages_ms: dict[str, float] = field(default_factory=dict)
    stage_spans: list[StageSpan] = field(default_factory=list)
    current_stage: str = ""

    def start_stage(self, stage_name: str) -> None:
        """记录某个阶段的开始时间。

        如果同名阶段尚未结束，则本次调用会覆盖它的起始时间。
        该行为用于容忍重复埋点或异常恢复场景，避免单次请求因状态不一致而报错。

        Args:
            stage_name: 要开始计时的阶段名称。
        """

        normalized = str(stage_name or "").strip()
        if not normalized:
            return
        self.stage_started_perf[normalized] = perf_counter()
        self.stage_started_offset_ms[normalized] = self.total_ms()
        self.current_stage = normalized

    def end_stage(self, stage_name: str) -> None:
        """结束某个阶段并累加耗时。

        如果该阶段之前没有被显式开始，则本次调用会被安全忽略。

        Args:
            stage_name: 要结束计时的阶段名称。
        """

        normalized = str(stage_name or "").strip()
        if not normalized:
            return
        started = self.stage_started_perf.pop(normalized, None)
        started_offset_ms = self.stage_started_offset_ms.pop(normalized, None)
        if started is None:
            return
        now_perf = perf_counter()
        duration_sec = now_perf - started
        duration_ms = max(0.0, float(duration_sec)) * 1000.0
        if started_offset_ms is None:
            end_offset_ms = self.total_ms(now_perf=now_perf)
            started_offset_ms = max(0.0, end_offset_ms - duration_ms)
        self._append_stage_span(
            normalized,
            start_ms=float(started_offset_ms),
            duration_ms=duration_ms,
        )
        self.stages_ms[normalized] = self.stages_ms.get(normalized, 0.0) + duration_ms
        if self.current_stage == normalized:
            self.current_stage = ""

    def record_stage_duration(self, stage_name: str, duration_sec: float) -> None:
        """直接累加一个已测得的阶段耗时。

        该接口用于并行任务或多次 flush 这类不适合成对 `start/end` 管理的路径。
        同名阶段会累计写入，便于统计一次请求内多次发生的发送或流式分片。

        Args:
            stage_name: 阶段名称。
            duration_sec: 阶段耗时，单位为秒。
        """

        normalized = str(stage_name or "").strip()
        if not normalized:
            return
        duration_ms = max(0.0, float(duration_sec)) * 1000.0
        end_offset_ms = self.total_ms()
        start_offset_ms = max(0.0, end_offset_ms - duration_ms)
        self._append_stage_span(
            normalized,
            start_ms=start_offset_ms,
            duration_ms=duration_ms,
        )
        self.stages_ms[normalized] = self.stages_ms.get(normalized, 0.0) + duration_ms

    def _append_stage_span(self, stage_name: str, *, start_ms: float, duration_ms: float) -> None:
        """追加一个阶段时间片段。

        Args:
            stage_name: 阶段名称。
            start_ms: 相对请求起点的开始偏移，单位为毫秒。
            duration_ms: 当前片段持续时间，单位为毫秒。
        """

        normalized = str(stage_name or "").strip()
        if not normalized:
            return
        safe_start_ms = max(0.0, float(start_ms))
        safe_duration_ms = max(0.0, float(duration_ms))
        self.stage_spans.append(
            StageSpan(
                name=normalized,
                start_ms=safe_start_ms,
                end_ms=safe_start_ms + safe_duration_ms,
                duration_ms=safe_duration_ms,
            )
        )

    def total_ms(self, *, now_perf: float | None = None) -> float:
        """计算请求自开始以来的总耗时。

        Args:
            now_perf: 可选的当前单调时钟值；未传入时会即时读取。

        Returns:
            float: 请求总耗时，单位为毫秒。
        """

        current_perf = perf_counter() if now_perf is None else float(now_perf)
        return max(0.0, current_perf - self.started_perf) * 1000.0

    def to_view(self, *, now_perf: float | None = None) -> dict[str, Any]:
        """导出进行中请求的只读视图。

        Args:
            now_perf: 可选的当前单调时钟值，用于批量生成快照时复用同一时刻。

        Returns:
            dict[str, Any]: 包含当前阶段、已运行时长和已完成阶段耗时的视图数据。
        """

        current_perf = perf_counter() if now_perf is None else float(now_perf)
        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "trigger_mode": self.trigger_mode,
            "chat_type": self.chat_type,
            "current_stage": self.current_stage,
            "elapsed_ms": round(self.total_ms(now_perf=current_perf), 3),
            "started_at": round(self.started_at, 3),
            "stages_ms": {
                key: round(value, 3)
                for key, value in sorted(self.stages_ms.items())
            },
            "stage_spans": [
                item.to_snapshot()
                for item in sorted(
                    self.stage_spans,
                    key=lambda span: (span.start_ms, span.name),
                )
            ],
        }


@dataclass(slots=True)
class CompletedLatencyRecord:
    """表示一个已结束请求的延迟结果。

    该结构用于最近窗口和结构化日志，保存请求最终 outcome、总耗时与各阶段耗时。

    Attributes:
        response_id: 当前请求的唯一响应 ID。
        session_id: 当前请求所属会话 ID。
        trigger_mode: 当前请求的触发模式。
        chat_type: 当前请求的会话类型。
        outcome: 本次请求结束状态，如 `completed`、`failed`、`timed_out`。
        started_at: 墙钟开始时间，单位为 Unix 时间戳秒。
        finished_at: 墙钟结束时间，单位为 Unix 时间戳秒。
        total_ms: 本次请求总耗时，单位为毫秒。
        stages_ms: 本次请求各阶段耗时明细，单位为毫秒。
    """

    response_id: str
    session_id: str
    trigger_mode: str
    chat_type: str
    outcome: str
    started_at: float
    finished_at: float
    total_ms: float
    stages_ms: dict[str, float]
    stage_spans: list[StageSpan]

    def to_snapshot(self) -> dict[str, Any]:
        """导出可序列化的完成态快照。

        Returns:
            dict[str, Any]: 适合命令展示和结构化日志记录的只读字典。
        """

        return {
            "response_id": self.response_id,
            "session_id": self.session_id,
            "trigger_mode": self.trigger_mode,
            "chat_type": self.chat_type,
            "outcome": self.outcome,
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3),
            "total_ms": round(self.total_ms, 3),
            "stages_ms": {
                key: round(value, 3)
                for key, value in sorted(self.stages_ms.items())
            },
            "stage_spans": [
                item.to_snapshot()
                for item in sorted(
                    self.stage_spans,
                    key=lambda span: (span.start_ms, span.name),
                )
            ],
        }


class ChatLatencyMonitor:
    """维护 GTBot 聊天主链路的进程内延迟监控状态。

    该监控器同时维护三类数据：
    1. 当前正在执行的请求；
    2. 最近固定窗口内已完成请求；
    3. 进程生命周期内的累计聚合统计。

    所有读写操作都使用线程锁保护，以兼容 `asyncio.to_thread`、命令查询和主链路并发读写。
    该监控器不做持久化，进程重启后统计会被重置。
    """

    def __init__(self, *, recent_window_size: int = 100) -> None:
        """初始化一个新的聊天延迟监控器。

        Args:
            recent_window_size: 最近完成请求窗口大小。达到上限后会自动淘汰最旧样本。
        """

        self._recent_window_size = max(1, int(recent_window_size))
        self._lock = Lock()
        self._inflight_requests: dict[str, RunningLatencyRecord] = {}
        self._recent_completed: deque[CompletedLatencyRecord] = deque(maxlen=self._recent_window_size)
        self._aggregate_total = StageAggregate()
        self._aggregate_stages: dict[str, StageAggregate] = {}
        self._outcome_counts: Counter[str] = Counter()
        self._last_completed: CompletedLatencyRecord | None = None

    def reset(self) -> None:
        """清空所有监控状态。

        该方法主要用于单元测试，避免不同测试之间共享进程内统计结果。
        正常运行时一般不应主动调用。
        """

        with self._lock:
            self._inflight_requests.clear()
            self._recent_completed.clear()
            self._aggregate_total = StageAggregate()
            self._aggregate_stages.clear()
            self._outcome_counts.clear()
            self._last_completed = None

    def start_request(
        self,
        response_id: str,
        session_id: str,
        trigger_mode: str,
        chat_type: str,
    ) -> None:
        """注册一个新的进行中请求。

        如果同一个 `response_id` 已存在，旧记录会被覆盖。这保证了重复开始调用不会抛错，
        但也意味着调用方应避免在真正的不同请求之间复用同一 `response_id`。

        Args:
            response_id: 当前请求的唯一响应 ID。
            session_id: 当前请求所属会话 ID。
            trigger_mode: 当前请求的触发模式。
            chat_type: 当前请求的会话类型。
        """

        with self._lock:
            self._inflight_requests[str(response_id)] = RunningLatencyRecord(
                response_id=str(response_id),
                session_id=str(session_id),
                trigger_mode=str(trigger_mode),
                chat_type=str(chat_type),
                started_at=time(),
                started_perf=perf_counter(),
            )

    def mark_stage_start(self, response_id: str, stage_name: str) -> None:
        """标记某个请求进入指定阶段。

        Args:
            response_id: 目标请求的响应 ID。
            stage_name: 阶段名称。
        """

        with self._lock:
            record = self._inflight_requests.get(str(response_id))
            if record is not None:
                record.start_stage(stage_name)

    def mark_stage_end(self, response_id: str, stage_name: str) -> None:
        """标记某个请求结束指定阶段。

        Args:
            response_id: 目标请求的响应 ID。
            stage_name: 阶段名称。
        """

        with self._lock:
            record = self._inflight_requests.get(str(response_id))
            if record is not None:
                record.end_stage(stage_name)

    def record_stage_duration(self, response_id: str, stage_name: str, duration_sec: float) -> None:
        """直接写入某个请求某个阶段的耗时。

        Args:
            response_id: 目标请求的响应 ID。
            stage_name: 阶段名称。
            duration_sec: 已测得的阶段耗时，单位为秒。
        """

        with self._lock:
            record = self._inflight_requests.get(str(response_id))
            if record is not None:
                record.record_stage_duration(stage_name, duration_sec)

    def finish_request(self, response_id: str, outcome: str) -> dict[str, Any] | None:
        """结束一个请求并将其归档到最近窗口和累计统计中。

        如果目标请求不存在，说明它要么从未注册，要么已经被归档。
        这类场景通常出现在异常恢复或重复结束调用中，此时方法会返回 `None`。

        Args:
            response_id: 目标请求的响应 ID。
            outcome: 当前请求的最终结果状态。

        Returns:
            dict[str, Any] | None: 成功归档时返回该请求的完成态快照；否则返回 `None`。
        """

        with self._lock:
            record = self._inflight_requests.pop(str(response_id), None)
            if record is None:
                return None

            finished_at = time()
            total_ms = record.total_ms()
            stages_ms = dict(record.stages_ms)
            stages_ms.setdefault("total", total_ms)
            completed = CompletedLatencyRecord(
                response_id=record.response_id,
                session_id=record.session_id,
                trigger_mode=record.trigger_mode,
                chat_type=record.chat_type,
                outcome=str(outcome),
                started_at=record.started_at,
                finished_at=finished_at,
                total_ms=total_ms,
                stages_ms=stages_ms,
                stage_spans=list(record.stage_spans),
            )

            self._recent_completed.append(completed)
            self._aggregate_total.add(total_ms)
            for stage_name, stage_ms in completed.stages_ms.items():
                aggregate = self._aggregate_stages.setdefault(stage_name, StageAggregate())
                aggregate.add(stage_ms)
            self._outcome_counts[completed.outcome] += 1
            self._last_completed = completed
            return completed.to_snapshot()

    def snapshot(self) -> dict[str, Any]:
        """返回当前监控器的整体快照。

        Returns:
            dict[str, Any]: 包含进行中请求、最近窗口统计、累计统计和最近一次完成请求。
        """

        with self._lock:
            now_perf = perf_counter()
            inflight = [
                item.to_view(now_perf=now_perf)
                for item in sorted(
                    self._inflight_requests.values(),
                    key=lambda record: record.started_perf,
                )
            ]

            recent_total_ms = 0.0
            recent_outcomes: Counter[str] = Counter()
            recent_stage_totals: dict[str, StageAggregate] = {}
            for item in self._recent_completed:
                recent_total_ms += item.total_ms
                recent_outcomes[item.outcome] += 1
                for stage_name, stage_ms in item.stages_ms.items():
                    aggregate = recent_stage_totals.setdefault(stage_name, StageAggregate())
                    aggregate.add(stage_ms)

            recent_count = len(self._recent_completed)
            recent_average_total_ms = recent_total_ms / recent_count if recent_count > 0 else 0.0

            return {
                "inflight_count": len(inflight),
                "inflight": inflight,
                "recent_window_size": self._recent_window_size,
                "recent": {
                    "sample_count": recent_count,
                    "average_total_ms": round(recent_average_total_ms, 3),
                    "average_stages_ms": {
                        key: value.to_snapshot()["average_ms"]
                        for key, value in sorted(
                            recent_stage_totals.items(),
                            key=lambda item: item[0],
                        )
                    },
                    "outcome_counts": dict(recent_outcomes),
                },
                "lifetime": {
                    "sample_count": self._aggregate_total.count,
                    "average_total_ms": self._aggregate_total.to_snapshot()["average_ms"],
                    "average_stages_ms": {
                        key: value.to_snapshot()["average_ms"]
                        for key, value in sorted(
                            self._aggregate_stages.items(),
                            key=lambda item: item[0],
                        )
                    },
                    "outcome_counts": dict(self._outcome_counts),
                },
                "last_completed": (
                    None
                    if self._last_completed is None
                    else self._last_completed.to_snapshot()
                ),
            }


_chat_latency_monitor = ChatLatencyMonitor()


def get_chat_latency_monitor() -> ChatLatencyMonitor:
    """返回进程级聊天延迟监控器单例。

    Returns:
        ChatLatencyMonitor: 当前进程共享的聊天延迟监控器。
    """

    return _chat_latency_monitor
