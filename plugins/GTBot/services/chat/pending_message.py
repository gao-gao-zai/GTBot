from __future__ import annotations

from asyncio import Event, TimeoutError as AsyncTimeoutError, wait_for
from typing import Literal

from ...Logger import logger
from .queue_payload import QueueMessageContent

PendingQueuedMessageStatus = Literal["pending", "fulfilled", "cancelled", "timed_out"]


class PendingQueuedMessageHandle:
    """表示一条已入队但尚未拿到最终内容的占位消息句柄。

    该句柄由 transport 在“预留发送位置”时创建，并随占位队列项一起进入消息队列。
    当队列真正轮到该位置时，会等待调用方后续通过 `fulfill()` 补入实际内容，或
    通过 `cancel()` 主动放弃；若直到超时仍未补入，则句柄会自动进入超时态，队列
    跳过该条目并继续后续发送。

    句柄状态只允许从 `pending` 进入一个终态，因此 `fulfill()`、`cancel()` 与
    内部超时结算都具备幂等行为。调用方可以根据返回值判断本次尝试是否真正生效。
    """

    def __init__(self, *, scope: str) -> None:
        """创建一个新的占位消息句柄。

        Args:
            scope: 人类可读的会话范围描述，用于日志定位，例如 `群组 123`。
        """

        self._scope = str(scope or "").strip() or "<unknown>"
        self._settled = Event()
        self._status: PendingQueuedMessageStatus = "pending"
        self._content: QueueMessageContent | None = None
        logger.info(f"queued placeholder created: scope={self._scope}")

    @property
    def status(self) -> PendingQueuedMessageStatus:
        """返回当前占位句柄的状态。

        Returns:
            PendingQueuedMessageStatus: 当前状态，便于诊断或测试断言。
        """

        return self._status

    def fulfill(self, content: QueueMessageContent) -> bool:
        """为占位消息补入最终消息内容。

        只有首次在 `pending` 状态下调用才会真正生效；一旦句柄已经被取消、超时或
        成功 fulfill，再次调用会被忽略并返回 `False`。

        Args:
            content: 最终要发出的实际消息内容，可为文本、消息段或完整消息对象。

        Returns:
            bool: 本次调用是否成功把句柄从 `pending` 结算为 `fulfilled`。
        """

        if self._status != "pending":
            return False
        self._content = content
        self._status = "fulfilled"
        self._settled.set()
        logger.info(f"queued placeholder fulfilled: scope={self._scope}")
        return True

    def cancel(self) -> bool:
        """主动取消占位消息，使队列在轮到该位置时直接跳过。

        Returns:
            bool: 本次调用是否成功把句柄从 `pending` 结算为 `cancelled`。
        """

        if self._status != "pending":
            return False
        self._status = "cancelled"
        self._settled.set()
        logger.info(f"queued placeholder cancelled: scope={self._scope}")
        return True

    async def wait_for_content(self, *, timeout_sec: float) -> QueueMessageContent | None:
        """等待占位消息获得最终内容，或在取消/超时后返回空。

        调用方通常是队列消费者。若句柄在等待期间被 `fulfill()`，则返回对应内容；
        若被 `cancel()` 或超时，则返回 `None`，调用方应把该占位视为“静默跳过”。
        当 `timeout_sec` 为 0 时，会执行一次非阻塞检查，未就绪则立即视为超时。

        Args:
            timeout_sec: 允许等待的最大秒数，必须为非负值。

        Returns:
            QueueMessageContent | None: 最终消息内容；取消或超时时返回 `None`。
        """

        normalized_timeout = max(0.0, float(timeout_sec))
        if self._status == "fulfilled":
            return self._content
        if self._status in {"cancelled", "timed_out"}:
            return None

        try:
            await wait_for(self._settled.wait(), timeout=normalized_timeout)
        except AsyncTimeoutError:
            self._expire()
            return None

        return self._content

    def _expire(self) -> None:
        """将仍处于等待中的句柄结算为超时状态。

        该方法仅供队列消费者内部在等待超时时调用；如果句柄已经被其他终态结算，
        则不会重复改写状态。
        """

        if self._status != "pending":
            return
        self._status = "timed_out"
        self._settled.set()
        logger.info(f"queued placeholder timed out and skipped: scope={self._scope}")
