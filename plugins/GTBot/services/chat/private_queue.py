from __future__ import annotations

from asyncio import Lock, Queue, create_task, sleep
from collections.abc import Sequence
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING

from nonebot.adapters.onebot.v11.message import Message

from ...Logger import logger
from ...model import QueuedMessageItem
from ...constants import DEFAULT_BOT_NAME_PLACEHOLDER
from ..message.segments import serialize_message_segments

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from .. import cache as CacheManager
    from ..message import GroupMessageManager


@dataclass(frozen=True, slots=True)
class PrivateMessageTask:
    """描述一次待发送的私聊消息任务。

    Attributes:
        messages: 已在入队前完成预处理的消息列表，每条都带有自己的后续等待时间。
        user_id: 目标私聊用户 ID。
        session_id: 私聊会话 ID；为空时会退化为 `private:{user_id}`。
    """

    messages: Sequence[QueuedMessageItem]
    user_id: int
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _QueuedPrivateMessageTask:
    """携带运行时依赖的私聊队列任务。"""

    task: PrivateMessageTask
    bot: "Bot"
    message_manager: "GroupMessageManager"
    cache: "CacheManager.UserCacheManager"


class PrivateMessageQueueManager:
    """按私聊会话串行发送消息的队列管理器。

    每个 `session_id` 维护独立队列，保证同一私聊中的多条消息不会并行发送。
    队列仅负责顺序发送与消息落库，不再承担文本清洗或 CQ 解析逻辑。
    """

    def __init__(self) -> None:
        """初始化私聊消息队列管理器。"""
        self._queues: dict[str, Queue[_QueuedPrivateMessageTask]] = {}
        self._consumers: dict[str, bool] = {}
        self._last_sent_at: dict[str, float | None] = {}
        self._lock = Lock()

    async def _get_or_create_queue(self, session_id: str) -> Queue[_QueuedPrivateMessageTask]:
        """获取或创建某个私聊会话的队列。

        Args:
            session_id: 私聊会话标识。

        Returns:
            Queue[_QueuedPrivateMessageTask]: 该会话对应的消息队列。
        """
        async with self._lock:
            if session_id not in self._queues:
                self._queues[session_id] = Queue()
                self._consumers[session_id] = False
                self._last_sent_at[session_id] = None
            return self._queues[session_id]

    async def _consumer(self, session_id: str) -> None:
        """消费指定私聊会话中的消息任务。

        Args:
            session_id: 需要处理的私聊会话 ID。
        """
        queue = self._queues.get(session_id)
        if queue is None:
            return

        try:
            while True:
                if queue.empty():
                    break

                queued = await queue.get()
                try:
                    last_sent_at = self._last_sent_at.get(session_id)
                    updated_last_sent_at = await self._process_task(
                        queued,
                        last_sent_at=last_sent_at,
                    )
                    self._last_sent_at[session_id] = updated_last_sent_at
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"处理私聊消息任务时发生错误（session {session_id}）: {exc}")
                finally:
                    queue.task_done()
        finally:
            async with self._lock:
                self._consumers[session_id] = False

    async def _process_task(
        self,
        queued: _QueuedPrivateMessageTask,
        *,
        last_sent_at: float | None,
    ) -> float | None:
        """发送单个私聊任务并回写消息记录。

        Args:
            queued: 包含任务数据、Bot、缓存和消息管理器的队列项。

        Returns:
            float | None: 本批任务最后一条消息实际发送完成的时间戳。
        """
        task = queued.task
        session_id = str(task.session_id or f"private:{int(task.user_id)}")
        current_last_sent_at = last_sent_at

        bot_user_name = await queued.cache.get_user_name(
            queued.bot,
            int(queued.bot.self_id),
        ) or DEFAULT_BOT_NAME_PLACEHOLDER

        for item in task.messages:
            await self._wait_until_sendable(item, last_sent_at=current_last_sent_at)
            processed_message = Message(item.message)
            result = await queued.bot.send_private_msg(
                user_id=task.user_id,
                message=processed_message,
            )
            sent_at = time()

            await queued.message_manager.add_chat_message(
                message_id=int(result["message_id"]),
                session_id=session_id,
                group_id=None,
                peer_user_id=int(task.user_id),
                sender_user_id=int(queued.bot.self_id),
                sender_name=bot_user_name,
                content=str(processed_message),
                serialized_segments=serialize_message_segments(processed_message),
                send_time=sent_at,
                is_withdrawn=False,
            )
            current_last_sent_at = sent_at

        return current_last_sent_at

    async def _wait_until_sendable(
        self,
        item: QueuedMessageItem,
        *,
        last_sent_at: float | None,
    ) -> None:
        """根据私聊会话历史发送时间和条目策略等待到可发送时刻。

        Args:
            item: 待发送的队列消息条目。
            last_sent_at: 当前私聊上一条消息的实际发送完成时间。
        """

        normalized_delay = max(0.0, float(item.delay_seconds))
        normalized_enqueued_at = float(item.enqueued_at)
        now_ts = time()

        if bool(item.force_wait):
            target_time = normalized_enqueued_at + normalized_delay
        elif last_sent_at is None:
            target_time = normalized_enqueued_at
        else:
            target_time = max(normalized_enqueued_at, float(last_sent_at) + normalized_delay)

        remaining_delay = float(target_time) - now_ts
        if remaining_delay > 0:
            await sleep(remaining_delay)

    async def enqueue(
        self,
        task: PrivateMessageTask,
        bot: "Bot",
        message_manager: "GroupMessageManager",
        cache: "CacheManager.UserCacheManager",
    ) -> None:
        """将私聊任务压入对应会话的发送队列。

        Args:
            task: 待发送的私聊消息任务。
            bot: 当前 Bot 实例。
            message_manager: 消息存储管理器。
            cache: 用户缓存管理器。
        """
        session_id = str(task.session_id or f"private:{int(task.user_id)}")
        queue = await self._get_or_create_queue(session_id)
        await queue.put(
            _QueuedPrivateMessageTask(
                task=task,
                bot=bot,
                message_manager=message_manager,
                cache=cache,
            )
        )

        async with self._lock:
            if not self._consumers.get(session_id, False):
                self._consumers[session_id] = True
                create_task(self._consumer(session_id))


private_message_queue_manager = PrivateMessageQueueManager()
