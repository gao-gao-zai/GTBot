from asyncio import Queue, Lock, create_task, sleep
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING

from nonebot.adapters.onebot.v11.message import Message

from ...Logger import logger
from ...constants import DEFAULT_BOT_NAME_PLACEHOLDER
from ..message.segments import serialize_message_segments
from ...model import GroupMessage, MessageTask, QueuedMessageItem
from .pending_message import PendingQueuedMessageHandle
from .queue_payload import prepare_queue_message

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from .. import cache as CacheManager
    from ..message import GroupMessageManager


@dataclass(frozen=True, slots=True)
class _QueuedMessageTask:
    """队列内部使用的任务封装。

    MessageTask 仅携带数据；运行时依赖在入队时单独传递，避免模型与业务对象耦合。

    Attributes:
        task: 纯数据任务。
        bot: OneBot 机器人实例。
        message_manager: 消息管理器实例。
        cache: 用户缓存管理器实例。
    """

    task: MessageTask
    bot: "Bot"
    message_manager: "GroupMessageManager"
    cache: "CacheManager.UserCacheManager"


class GroupMessageQueueManager:
    """群组消息队列管理器（生产者-消费者模型）。

    为每个群组维护独立的消息队列，确保发送给同一个群的消息按顺序发送，
    不会并行发送。不同群组之间的消息可以并行发送。

    队列本体只负责顺序发送和消息落库；聊天记录前缀清洗、CQ 解析等输入
    规范化逻辑必须在入队前完成。

    Example:
        >>> queue_manager = GroupMessageQueueManager()
        >>> await queue_manager.enqueue(MessageTask(...))
    """
    
    def __init__(self) -> None:
        """初始化群组消息队列管理器。"""
        self._queues: dict[int, Queue[_QueuedMessageTask]] = {}
        self._consumers: dict[int, bool] = {}  # 记录每个群是否有消费者在运行
        self._last_sent_at: dict[int, float | None] = {}  # 记录每个群最近一次实际发送完成时间
        self._lock = Lock()  # 保护队列创建的锁
    
    async def _get_or_create_queue(self, group_id: int) -> Queue[_QueuedMessageTask]:
        """获取或创建指定群组的消息队列。
        
        Args:
            group_id: 群组 ID。
        
        Returns:
            该群组的消息队列。
        """
        async with self._lock:
            if group_id not in self._queues:
                self._queues[group_id] = Queue()
                self._consumers[group_id] = False
                self._last_sent_at[group_id] = None
            return self._queues[group_id]
    
    async def _consumer(self, group_id: int) -> None:
        """消费者协程，处理指定群组的消息队列。
        
        从队列中取出消息任务并按顺序发送，确保同一群组的消息不会并行发送。
        当队列为空时，消费者协程结束。
        
        Args:
            group_id: 群组 ID。
        """
        queue = self._queues.get(group_id)
        if queue is None:
            return
        
        try:
            while True:
                # 非阻塞检查队列是否为空
                if queue.empty():
                    break
                
                task = await queue.get()
                try:
                    last_sent_at = self._last_sent_at.get(group_id)
                    updated_last_sent_at = await self._process_task(task, last_sent_at=last_sent_at)
                    self._last_sent_at[group_id] = updated_last_sent_at
                except Exception as e:
                    logger.error(f"处理消息任务时发生错误（群组 {group_id}）: {str(e)}")
                finally:
                    queue.task_done()
        finally:
            async with self._lock:
                self._consumers[group_id] = False

    async def _process_task(
        self,
        queued: _QueuedMessageTask,
        *,
        last_sent_at: float | None,
    ) -> float | None:
        """处理单个消息发送任务。

        Args:
            queued: 队列任务（包含运行时依赖）。
                其中 `task.messages` 应当已经是可直接发送的消息对象和延迟控制参数。
            last_sent_at: 当前群上一条消息实际发送完成的时间戳。

        Returns:
            float | None: 本批任务最后一条消息实际发送完成的时间戳。
        """
        task = queued.task
        current_last_sent_at = last_sent_at

        for item in task.messages:
            await self._wait_until_sendable(item, last_sent_at=current_last_sent_at)
            processed_message = await self._resolve_item_message(
                item,
                scope=f"群组 {task.group_id}",
            )
            if processed_message is None:
                continue
            result = await queued.bot.send_group_msg(
                group_id=task.group_id,
                message=processed_message
            )
            sent_at = time()

            bot_user_name = await queued.cache.get_user_name(
                queued.bot,
                int(queued.bot.self_id),
            ) or DEFAULT_BOT_NAME_PLACEHOLDER

            bot_msg = GroupMessage(
                message_id=result["message_id"],
                group_id=task.group_id,
                user_id=int(queued.bot.self_id),
                user_name=bot_user_name,
                content=str(processed_message),
                serialized_segments=serialize_message_segments(processed_message),
                send_time=sent_at,
                is_withdrawn=False,
            )
            
            # 将消息填回消息数据库
            await queued.message_manager.add_message(
                bot_msg
            )
            current_last_sent_at = sent_at

        return current_last_sent_at

    async def _resolve_item_message(
        self,
        item: QueuedMessageItem,
        *,
        scope: str,
    ) -> Message | None:
        """把普通消息或占位消息统一解析成可发送的 `Message`。

        普通消息会直接复用其已准备好的内容；占位消息则在真正轮到发送时等待插件
        补入最终内容，并复用统一的消息规范化逻辑做最后清洗。若占位被取消或超时，
        则返回 `None`，调用方应静默跳过该条目。

        Args:
            item: 当前待解析的队列条目。
            scope: 日志范围描述，用于透传给消息规范化逻辑。

        Returns:
            Message | None: 可直接发送的消息对象；当占位被跳过时返回 `None`。
        """

        if not item.is_placeholder():
            return item.message if isinstance(item.message, Message) else Message(item.message)

        handle = item.placeholder_handle
        if not isinstance(handle, PendingQueuedMessageHandle):
            logger.warning(f"queued placeholder handle is invalid and will be skipped: scope={scope}")
            return None

        resolved_content = await handle.wait_for_content(
            timeout_sec=float(item.placeholder_timeout_sec or 0.0),
        )
        if resolved_content is None:
            return None
        return await prepare_queue_message(resolved_content, scope=scope)

    async def _wait_until_sendable(
        self,
        item: QueuedMessageItem,
        *,
        last_sent_at: float | None,
    ) -> None:
        """根据队列历史发送时间和条目策略等待到可发送时刻。

        非强制等待模式下，如果当前条目的入队时间与上一条消息的发送时间间隔
        已经大于声明延迟，则当前条目会立即发送；否则只补足差值。强制等待模式
        下，则至少从入队时刻起等待 `delay_seconds`，不再额外参考上一条消息的
        发送时间来增加等待。

        Args:
            item: 待发送的队列消息条目。
            last_sent_at: 当前群上一条消息的实际发送完成时间。
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
        task: object,
        bot: "Bot",
        message_manager: "GroupMessageManager",
        cache: "CacheManager.UserCacheManager",
    ) -> None:
        """将消息任务加入队列。
        
        如果该群组没有运行中的消费者，会启动一个新的消费者协程。
        
        Args:
            task: 消息发送任务。调用方应传入可兼容 `MessageTask` 结构的对象。
            bot: OneBot 机器人实例。
            message_manager: 消息管理器实例。
            cache: 用户缓存管理器实例。
        """
        normalized_task = MessageTask.model_validate(task)
        queue = await self._get_or_create_queue(normalized_task.group_id)
        await queue.put(
            _QueuedMessageTask(
                task=normalized_task,
                bot=bot,
                message_manager=message_manager,
                cache=cache,
            )
        )
        
        # 检查是否需要启动消费者
        async with self._lock:
            if not self._consumers.get(normalized_task.group_id, False):
                self._consumers[normalized_task.group_id] = True
                create_task(self._consumer(normalized_task.group_id))


# 初始化全局消息队列管理器
group_message_queue_manager = GroupMessageQueueManager()
"""全局群组消息队列管理器，用于按群顺序发送消息。"""
