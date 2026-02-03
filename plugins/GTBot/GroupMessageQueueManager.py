
from asyncio import Queue, Lock, create_task, sleep
from time import time

from nonebot.adapters.onebot.v11.message import Message

from .Logger import logger
from . import Fun
from .constants import DEFAULT_BOT_NAME_PLACEHOLDER, SUPPORTED_CQ_CODES
from .model import GroupMessage, MessageTask


class GroupMessageQueueManager:
    """群组消息队列管理器（生产者-消费者模型）。
    
    为每个群组维护独立的消息队列，确保发送给同一个群的消息按顺序发送，
    不会并行发送。不同群组之间的消息可以并行发送。
    
    Example:
        >>> queue_manager = GroupMessageQueueManager()
        >>> await queue_manager.enqueue(MessageTask(...))
    """
    
    def __init__(self) -> None:
        """初始化群组消息队列管理器。"""
        self._queues: dict[int, Queue[MessageTask]] = {}
        self._consumers: dict[int, bool] = {}  # 记录每个群是否有消费者在运行
        self._lock = Lock()  # 保护队列创建的锁
    
    async def _get_or_create_queue(self, group_id: int) -> Queue[MessageTask]:
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
                    await self._process_task(task)
                except Exception as e:
                    logger.error(f"处理消息任务时发生错误（群组 {group_id}）: {str(e)}")
                finally:
                    queue.task_done()
        finally:
            async with self._lock:
                self._consumers[group_id] = False
    
    async def _process_task(self, task: MessageTask) -> None:
        """处理单个消息发送任务。
        
        Args:
            task: 消息发送任务。
        """
        for idx, msg_content in enumerate(task.messages):
            processed_message: Message = await Fun.text_to_message(
                msg_content, 
                whitelist=SUPPORTED_CQ_CODES
            )
            
            result = await task.bot.send_group_msg(
                group_id=task.group_id,
                message=processed_message
            )
            
            # 将消息填回消息数据库
            await task.message_manager.add_message(
                GroupMessage(
                    message_id=result["message_id"],
                    group_id=task.group_id,
                    user_id=int(task.bot.self_id),
                    user_name=await task.cache.get_user_name(
                        task.bot, 
                        int(task.bot.self_id)
                    ) or DEFAULT_BOT_NAME_PLACEHOLDER,
                    content=msg_content,
                    send_time=time(),
                    is_withdrawn=False
                )
            )
            
            # 如果不是最后一条消息，等待指定间隔
            if idx < len(task.messages) - 1:
                await sleep(task.interval)
    
    async def enqueue(self, task: MessageTask) -> None:
        """将消息任务加入队列。
        
        如果该群组没有运行中的消费者，会启动一个新的消费者协程。
        
        Args:
            task: 消息发送任务。
        """
        queue = await self._get_or_create_queue(task.group_id)
        await queue.put(task)
        
        # 检查是否需要启动消费者
        async with self._lock:
            if not self._consumers.get(task.group_id, False):
                self._consumers[task.group_id] = True
                create_task(self._consumer(task.group_id))


# 初始化全局消息队列管理器
group_message_queue_manager = GroupMessageQueueManager()
"""全局群组消息队列管理器，用于按群顺序发送消息。"""