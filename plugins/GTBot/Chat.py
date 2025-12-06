from ast import Dict
import re
from time import time
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List, Callable, Any, Union
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends, EventMessage
from nonebot import on, on_message, get_driver, on_notice
from nonebot.rule import to_me
from pathlib import Path
from asyncio import Semaphore, Queue, Lock, TimeoutError as AsyncTimeoutError, wait_for
from dataclasses import dataclass
from asyncio import sleep, create_task, Event

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI # TODO: 未来支持更多的提供商
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import SecretStr


from .model import GroupMessage
from .MassageManager import GroupMessageManager, get_message_manager, GroupMessageManager
from .ConfigManager import total_config, ProcessedConfiguration
from . import Fun
from . import CacheManager
from .UserProfileManager import ProfileManager, get_profile_manager


config = total_config.processed_configuration.current_config_group


# ============================================================================
# 消息发送队列管理器（生产者-消费者模型）
# ============================================================================

@dataclass
class MessageTask:
    """消息发送任务数据类。
    
    Attributes:
        messages: 要发送的消息列表。
        group_id: 目标群组 ID。
        bot: OneBot 机器人实例。
        message_manager: 消息管理器实例。
        cache: 用户缓存管理器实例。
        interval: 发送多条消息时的间隔时间（秒）。
    """
    messages: List[str]
    group_id: int
    bot: Bot
    message_manager: GroupMessageManager
    cache: CacheManager.UserCacheManager
    interval: float


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


# ============================================================================
# 响应锁管理器
# ============================================================================

class ResponseLockManager:
    """群聊响应事件锁管理器（非阻塞式拒绝）。
    
    用于控制同时进行的群聊响应事件数量，支持两个级别的限制：
    1. 单个群组级别：限制某个特定群组同时响应的事件数
    2. 全局级别：限制所有群组总共同时响应的事件数
    
    当锁满时会立即拒绝新请求，而不是等待。
    
    Example:
        >>> lock_mgr = ResponseLockManager(max_per_group=1, max_total=5)
        >>> if lock_mgr.try_acquire(group_id=123):
        ...     try:
        ...         # 处理聊群响应
        ...         pass
        ...     finally:
        ...         lock_mgr.release(group_id=123)
        ... else:
        ...     logger.info("锁满，拒绝本次请求")
    """
    
    def __init__(self, max_concurrent_responses_per_group: int = 1, max_total_concurrent_responses: int = 5) -> None:
        """初始化响应锁管理器。
        
        Args:
            max_concurrent_responses_per_group (int): 单个群组最多允许同时响应的事件数。
                                                       0 表示不限制。默认为 1。
            max_total_concurrent_responses (int): 全局最多允许同时响应的总事件数。
                                                   0 表示不限制。默认为 5。
        
        Note:
            如果两个参数都为 0，则不进行任何限制。
        """
        self.max_concurrent_responses_per_group = max_concurrent_responses_per_group
        self.max_total_concurrent_responses = max_total_concurrent_responses
        
        # 用于控制全局并发数
        self._global_semaphore: Semaphore | None = None
        if max_total_concurrent_responses > 0:
            self._global_semaphore = Semaphore(max_total_concurrent_responses)
        
        # 用于控制每个群组的并发数
        self._group_semaphores: dict[int, Semaphore] = {}
    
    def _get_group_semaphore(self, group_id: int) -> Semaphore | None:
        """获取指定群组的信号量。
        
        Args:
            group_id (int): 群组 ID。
        
        Returns:
            Semaphore | None: 群组的信号量，如果不需要限制返回 None。
        """
        if self.max_concurrent_responses_per_group == 0:
            return None
        
        if group_id not in self._group_semaphores:
            self._group_semaphores[group_id] = Semaphore(self.max_concurrent_responses_per_group)
        
        return self._group_semaphores[group_id]
    
    def try_acquire(self, group_id: int) -> bool:
        """尝试获取响应锁（非阻塞）。
        
        立即尝试同时获取全局锁和群组锁。如果任一锁满，则立即返回 False。
        
        Args:
            group_id (int): 群组 ID。
        
        Returns:
            bool: 如果成功获得锁返回 True，否则返回 False。
        
        Note:
            - 获得锁后必须调用 release() 来释放
            - 如果返回 False，不需要调用 release()
        
        Example:
            >>> if lock_mgr.try_acquire(group_id=123):
            ...     try:
            ...         # 处理聊群响应
            ...         pass
            ...     finally:
            ...         lock_mgr.release(group_id=123)
            ... else:
            ...     logger.info("锁满，拒绝请求")
        """
        # 尝试获取全局锁
        if self._global_semaphore is not None:
            if self._global_semaphore._value <= 0:
                return False
            self._global_semaphore._value -= 1
        
        # 尝试获取群组锁
        group_semaphore = self._get_group_semaphore(group_id)
        if group_semaphore is not None:
            if group_semaphore._value <= 0:
                # 回滚全局锁
                if self._global_semaphore is not None:
                    self._global_semaphore._value += 1
                return False
            group_semaphore._value -= 1
        
        return True
    
    def release(self, group_id: int) -> None:
        """释放响应锁。
        
        Args:
            group_id (int): 群组 ID。
        
        Note:
            只应该在 try_acquire() 返回 True 后调用。
        """
        # 释放群组锁
        group_semaphore = self._get_group_semaphore(group_id)
        if group_semaphore is not None:
            group_semaphore._value += 1
        
        # 释放全局锁
        if self._global_semaphore is not None:
            self._global_semaphore._value += 1


# 定义常量
NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES = 20
"""在获取附近消息时，额外获取的冗余消息数量，以确保上下文的完整性。"""

DEFAULT_BOT_NAME_PLACEHOLDER = "GTBot"
"""当找不到机器人名称时用的占位字符串。"""

SUPPORTED_CQ_CODES = [
    "at",
    "face",
    "image",
    "record",
    "reply",
]

SEND_MESSAGE_BLOCK_PATTERN = re.compile(
    r'```send_message\s*\n(.*?)\n```',
    re.DOTALL
)
"""用于匹配 markdown 风格的 send_message 代码块的正则表达式。

匹配格式:
```send_message
消息内容
```

每个代码块表示一条独立的消息。
"""

"""支持的CQ码白名单"""


def parse_send_message_blocks(content: str) -> List[str]:
    """解析文本中的 send_message 代码块，提取消息内容。
    
    使用 markdown 风格的代码块语法解析消息，每个代码块表示一条独立的消息。
    
    格式示例:
    ```send_message
    这是第一条消息
    ```
    
    ```send_message
    这是第二条消息
    ```
    
    Args:
        content: 包含 send_message 代码块的原始文本。
    
    Returns:
        解析出的消息列表。如果没有找到任何代码块，返回空列表。
    
    Example:
        >>> text = '''```send_message
        ... 你好！
        ... ```
        ... 
        ... ```send_message
        ... 再见！
        ... ```'''
        >>> parse_send_message_blocks(text)
        ['你好！', '再见！']
    """
    matches = SEND_MESSAGE_BLOCK_PATTERN.findall(content)
    # 过滤空消息并去除首尾空白
    return [msg.strip() for msg in matches if msg.strip()]

# 初始化全局响应锁管理器
response_lock_manager = ResponseLockManager(
    max_concurrent_responses_per_group=config.chat_model.max_concurrent_responses_per_group,
    max_total_concurrent_responses=config.chat_model.max_total_concurrent_responses
)
"""全局响应锁管理器，用于控制聊群响应的并发数。"""

class GroupChatContext(BaseModel):
    """群聊上下文类，用于存储群组聊天的相关信息。
    
    Attributes:
        bot (Bot): OneBot 机器人实例。
        group_id (int): 群组 ID。
        user_id (int): 用户 ID。
        message_id (int): 消息 ID。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bot: Bot
    group_id: int
    user_id: int
    message_id: int
    message_manager: GroupMessageManager
    cache: CacheManager.UserCacheManager
    profile_manager: ProfileManager


class TollCalls:
    """为语言模型提供的内置工具集合类。
    
    提供群聊消息发送等功能接口供 LLM 调用。
    """
    
    @staticmethod
    async def send_group_message(
        message: str | List[str], 
        runtime: ToolRuntime[GroupChatContext],
        group_id: int | None = None,
        interval: float = 0.2,
    ) -> str:
        """向指定群组发送消息。
        
        使用生产者-消费者模型，确保发送给同一个群的消息按顺序发送，不会并行发送。
        不同群组之间的消息可以并行发送。
        
        Args:
            message (str | List[str]): 要发送的消息内容，可以是单条消息或消息列表。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 目标群组 ID。不填则自动获取当前的聊群ID。
            interval (float): 发送多条消息时的间隔时间（秒），默认为 0.2。
        
        Returns:
            str: 发送结果信息。
        
        Note:
            - 使用消息队列确保同一群组的消息按顺序发送
            - 消息会被加入队列后异步发送，不会阻塞调用
        """
        
        if group_id is None:
            group_id = runtime.context.group_id

        # 将单条消息转换为列表
        messages: List[str] = [message] if isinstance(message, str) else message
        
        logger.info(
            f"工具调用: 向群组 {group_id} 发送 {len(messages)} 条消息（已加入队列）"
        )

        # 创建消息任务并加入队列
        task = MessageTask(
            messages=messages,
            group_id=group_id,
            bot=runtime.context.bot,
            message_manager=runtime.context.message_manager,
            cache=runtime.context.cache,
            interval=interval
        )
        
        # 将任务加入队列，由消费者按顺序处理
        await group_message_queue_manager.enqueue(task)
        
        return f"消息已提交发送至群组 {group_id}（共 {len(messages)} 条）"

    @staticmethod
    async def delete_message(
        message_id: int,
        runtime: ToolRuntime[GroupChatContext],
        delay: int = 0,
    ) -> str:
        """撤回指定的消息。
        
        Args:
            message_id (int): 要撤回的消息 ID。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            delay (int): 延迟撤回时间（秒），范围 0-60，默认为 0（立即撤回）。
        
        Returns:
            str: 撤回结果信息。
        
        Note:
            - 如果自己是管理员或群主则撤回无限制
            - 否则只能撤回自己的消息，且时间限制为2分钟内。
            - 延迟撤回时间最大为 60 秒
            - 延迟撤回会在后台进行，不会堵塞主流程
        """
        # 参数验证
        if not isinstance(delay, int) or delay < 0 or delay > 60:
            return f"撤回消息 {message_id} 失败: 延迟撤回时间必须在 0-60 秒之间"
        
        logger.info(f"工具调用: 撤回消息 {message_id}（延迟 {delay} 秒）")
        
        async def delete_message_async() -> None:
            """异步撤回消息，支持延迟撤回。"""
            try:
                # 如果需要延迟，则先等待
                if delay > 0:
                    await sleep(delay)
                
                result = await Fun.delete_message(runtime.context.bot, message_id, delay=0)
                
                # 更新消息数据库中的撤回状态
                msg_mg = runtime.context.message_manager
                await msg_mg.mark_message_withdrawn(message_id)
                
                logger.info(f"消息 {message_id} 已成功撤回")
            except Exception as e:
                logger.error(f"撤回消息 {message_id} 失败: {str(e)}")
        
        # 如果无延迟，立即同步执行
        if delay == 0:
            try:
                await delete_message_async()
                return f"消息 {message_id} 已成功撤回"
            except Exception as e:
                return f"撤回消息 {message_id} 失败: {str(e)}"
        else:
            # 有延迟时，开启新协程在后台执行，防止堵塞
            create_task(delete_message_async())
            return f"消息 {message_id} 将在 {delay} 秒后撤回"

    @staticmethod
    async def emoji_reaction(
        message_id: int,
        emoji_id: int,
        runtime: ToolRuntime[GroupChatContext],
    ) -> str:
        """对消息进行表情回应（表情贴）。
        
        Args:
            message_id (int): 要回应的消息 ID。
            emoji_id (int): 表情 ID，QQ 表情对应的数字编号。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            str: 回应结果信息。
        
        Note:
            只支持群聊消息。
        """
        logger.info(f"工具调用: 对消息 {message_id} 添加表情回应 {emoji_id}")
        
        try:
            await Fun.set_msg_emoji_like(
                runtime.context.bot, 
                message_id, 
                emoji_id
            )
            return f"已对消息 {message_id} 添加表情回应（表情ID: {emoji_id}）"
        except Exception as e:
            return f"表情回应失败: {str(e)}"

    @staticmethod
    async def poke_user(
        user_id: int,
        runtime: ToolRuntime[GroupChatContext],
        group_id: int | None = None,
    ) -> str:
        """在群聊中戳一戳指定用户（双击头像效果）。
        
        Args:
            user_id (int): 要戳的用户 QQ 号。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 群号。不填则使用当前群组。
        
        Returns:
            str: 戳一戳结果信息。
        """
        if group_id is None:
            group_id = runtime.context.group_id
            
        logger.info(f"工具调用: 在群组 {group_id} 戳一戳用户 {user_id}")
        
        try:
            await Fun.group_poke(runtime.context.bot, group_id, user_id)
            return f"已在群组 {group_id} 戳了用户 {user_id}"
        except Exception as e:
            return f"戳一戳失败: {str(e)}"

    @staticmethod
    async def send_like(
        user_id: int,
        runtime: ToolRuntime[GroupChatContext]
    ) -> str:

        """给指定用户发送点赞。
        
        Args:
            user_id (int): 要点赞的用户 QQ 号。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            str: 点赞结果信息。

        Note:
            一般每天只允许给同一个好友点赞10次，超出的会失败。
        """
        logger.info(f"工具调用: 给用户 {user_id} 发送点赞")
        
        try:
            await Fun.send_like(runtime.context.bot, user_id)
            return f"已给用户 {user_id} 发送点赞"
        except Exception as e:
            return f"发送点赞失败: {str(e)}"

    @staticmethod
    async def get_user_profile(
        user_id: int,
        runtime: ToolRuntime[GroupChatContext]
    ) -> dict[int, str] | str:
        """获取指定用户的画像描述。
        
        Args:
            user_id (int): 要获取画像的用户 QQ 号。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        returns:
            dict[int, str] | str: 用户画像描述字典，键为描述索引，值为描述内容。
                                 如果用户未设置画像，返回提示字符串。
        """
        logger.info(f"工具调用: 获取用户 {user_id} 的画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            description = await profile_manager.user.get_user_descriptions_with_index(user_id)
            if description:
                return description
            else:
                return f"用户 {user_id} 尚未设置画像描述。"
        except Exception as e:
            return f"获取用户画像失败: {str(e)}"

    @staticmethod
    async def add_user_profile(
        user_id: int,
        description: str,
        runtime: ToolRuntime[GroupChatContext]
    ) -> str:
        """为用户添加画像描述。
        
        Args:
            user_id (int): 用户 QQ 号。
            description (str): 要添加的画像描述内容。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            str: 操作结果信息。
        """
        logger.info(f"工具调用: 为用户 {user_id} 添加画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.user.add_user_profile(user_id, description)
            return f"已为用户 {user_id} 添加画像描述: {description}"
        except ValueError as e:
            return f"添加用户画像失败: {str(e)}"
        except Exception as e:
            return f"添加用户画像失败: {str(e)}"

    @staticmethod
    async def edit_user_profile(
        user_id: int,
        index: int,
        new_description: str,
        runtime: ToolRuntime[GroupChatContext]
    ) -> str:
        """编辑用户指定序号的画像描述。
        
        Args:
            user_id (int): 用户 QQ 号。
            index (int): 要编辑的描述序号（从1开始）。
            new_description (str): 新的描述内容。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            str: 操作结果信息。
        """
        logger.info(f"工具调用: 编辑用户 {user_id} 第 {index} 条画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.user.edit_user_description_by_index(user_id, index, new_description)
            return f"已编辑用户 {user_id} 第 {index} 条画像描述为: {new_description}"
        except ValueError as e:
            return f"编辑用户画像失败: {str(e)}"
        except Exception as e:
            return f"编辑用户画像失败: {str(e)}"

    @staticmethod
    async def delete_user_profile(
        user_id: int,
        indices: Union[int, list[int]],
        runtime: ToolRuntime[GroupChatContext]
    ) -> str:
        """删除用户指定序号的画像描述。
        
        Args:
            user_id (int): 用户 QQ 号。
            indices (int | list[int]): 要删除的描述序号（从1开始），可以是单个整数或整数列表。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            str: 操作结果信息。
            
        Note:
            删除后其他描述的序号会自动递减。例如删除序号2后，
            原来的序号3会变成2，序号4会变成3，依此类推。
            若需要继续操作，请重新调用获取用户画像工具以确认新序号。
        """
        logger.info(f"工具调用: 删除用户 {user_id} 的画像描述（序号: {indices}）")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.user.delete_user_description_by_index(user_id, indices)
            return f"已删除用户 {user_id} 的指定画像描述。"
        except ValueError as e:
            return f"删除用户画像失败: {str(e)}"
        except Exception as e:
            return f"删除用户画像失败: {str(e)}"

    @staticmethod
    async def get_group_profile(
        group_id: int,
        runtime: ToolRuntime[GroupChatContext]
    ) -> dict[int, str] | str:
        """获取指定群聊的画像描述。
        
        Args:
            group_id (int): 群号，不填则使用当前群组。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
        
        Returns:
            dict[int, str] | str: 群聊画像描述字典，键为描述索引，值为描述内容。
                                 如果群聊未设置画像，返回提示字符串。
        """
        if group_id is None:
            group_id = runtime.context.group_id
            
        logger.info(f"工具调用: 获取群聊 {group_id} 的画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            description = await profile_manager.group.get_group_descriptions_with_index(group_id)
            if description:
                return description
            else:
                return f"群聊 {group_id} 尚未设置画像描述。"
        except Exception as e:
            return f"获取群聊画像失败: {str(e)}"

    @staticmethod
    async def add_group_profile(
        description: str,
        runtime: ToolRuntime[GroupChatContext],
        group_id: int | None = None
    ) -> str:
        """为群聊添加画像描述。
        
        Args:
            description (str): 要添加的画像描述内容。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 群号，不填则使用当前群组。
        
        Returns:
            str: 操作结果信息。
        """
        if group_id is None:
            group_id = runtime.context.group_id
            
        logger.info(f"工具调用: 为群聊 {group_id} 添加画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.group.add_group_profile(group_id, description)
            return f"已为群聊 {group_id} 添加画像描述: {description}"
        except ValueError as e:
            return f"添加群聊画像失败: {str(e)}"
        except Exception as e:
            return f"添加群聊画像失败: {str(e)}"

    @staticmethod
    async def edit_group_profile(
        index: int,
        new_description: str,
        runtime: ToolRuntime[GroupChatContext],
        group_id: int | None = None
    ) -> str:
        """编辑群聊指定序号的画像描述。
        
        Args:
            index (int): 要编辑的描述序号（从1开始）。
            new_description (str): 新的描述内容。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 群号，不填则使用当前群组。
        
        Returns:
            str: 操作结果信息。
        """
        if group_id is None:
            group_id = runtime.context.group_id
            
        logger.info(f"工具调用: 编辑群聊 {group_id} 第 {index} 条画像描述")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.group.edit_group_description_by_index(group_id, index, new_description)
            return f"已编辑群聊 {group_id} 第 {index} 条画像描述为: {new_description}"
        except ValueError as e:
            return f"编辑群聊画像失败: {str(e)}"
        except Exception as e:
            return f"编辑群聊画像失败: {str(e)}"

    @staticmethod
    async def delete_group_profile(
        indices: Union[int, list[int]],
        runtime: ToolRuntime[GroupChatContext],
        group_id: int | None = None
    ) -> str:
        """删除群聊指定序号的画像描述。
        
        Args:
            indices (int | list[int]): 要删除的描述序号（从1开始），可以是单个整数或整数列表。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 群号，不填则使用当前群组。
        
        Returns:
            str: 操作结果信息。
            
        Note:
            删除后其他描述的序号会自动递减。例如删除序号2后，
            原来的序号3会变成2，序号4会变成3，依此类推。
            若需要继续操作，请重新调用获取群聊画像工具以确认新序号。
        """
        if group_id is None:
            group_id = runtime.context.group_id
            
        logger.info(f"工具调用: 删除群聊 {group_id} 的画像描述（序号: {indices}）")
        
        try:
            profile_manager = runtime.context.profile_manager
            await profile_manager.group.delete_group_description_by_index(group_id, indices)
            return f"已删除群聊 {group_id} 的指定画像描述。"
        except ValueError as e:
            return f"删除群聊画像失败: {str(e)}"
        except Exception as e:
            return f"删除群聊画像失败: {str(e)}"



# 将工具函数用 @tool 装饰并导出
send_group_message_tool = tool(TollCalls.send_group_message)
delete_message_tool = tool(TollCalls.delete_message)
emoji_reaction_tool = tool(TollCalls.emoji_reaction)
poke_user_tool = tool(TollCalls.poke_user)
send_like_tool = tool(TollCalls.send_like)
get_user_profile_tool = tool(TollCalls.get_user_profile)
add_user_profile_tool = tool(TollCalls.add_user_profile)
edit_user_profile_tool = tool(TollCalls.edit_user_profile)
delete_user_profile_tool = tool(TollCalls.delete_user_profile)
get_group_profile_tool = tool(TollCalls.get_group_profile)
add_group_profile_tool = tool(TollCalls.add_group_profile)
edit_group_profile_tool = tool(TollCalls.edit_group_profile)
delete_group_profile_tool = tool(TollCalls.delete_group_profile)

# 定义需要用到的辅助函数
def convert_openai_to_langchain_messages(openai_messages: List[dict]) -> List:
    """将 OpenAI 格式的消息列表转换为 LangChain 格式的消息对象列表。
    
    OpenAI 格式示例:
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    LangChain 格式示例:
    [
        SystemMessage("You are a helpful assistant"),
        HumanMessage("Hello"),
        AIMessage("Hi there!")
    ]
    
    Args:
        openai_messages (List[dict]): OpenAI 格式的消息列表，每个消息包含 "role" 和 "content" 字段。
    
    Returns:
        List: LangChain 格式的消息对象列表（SystemMessage、HumanMessage、AIMessage）。
    
    Raises:
        ValueError: 当消息角色不是 "system"、"user" 或 "assistant" 时抛出异常。
    
    Example:
        >>> openai_msgs = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello"}
        ... ]
        >>> langchain_msgs = convert_openai_to_langchain_messages(openai_msgs)
        >>> len(langchain_msgs)
        3
    """
    langchain_messages: List = []
    
    for msg in openai_messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "system":
            langchain_messages.append(SystemMessage(content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content))
        else:
            raise ValueError(f"未支持的消息角色: {role}。支持的角色有: 'system', 'user', 'assistant'")
    
    return langchain_messages


def create_group_chat_agent():
    """创建一个用于处理群聊消息的智能体。
    
    Returns:
        智能体实例，可用于处理群聊消息。
    
    Note:
        如果配置了 max_tool_calls_per_turn > 0，则会添加工具调用次数限制中间件。
        超过限制后，智能体将停止工具调用并返回错误信息。
    """

    tools = [
        send_group_message_tool,
        delete_message_tool,
        emoji_reaction_tool,
        poke_user_tool,
        send_like_tool,
        get_user_profile_tool,
        add_user_profile_tool,
        edit_user_profile_tool,
        delete_user_profile_tool,
        get_group_profile_tool,
        add_group_profile_tool,
        edit_group_profile_tool,
        delete_group_profile_tool,
    ]
    model = ChatOpenAI(
        model = config.chat_model.model_id,
        base_url=config.chat_model.base_url,
        api_key=SecretStr(config.chat_model.api_key),
        model_kwargs=config.chat_model.parameters
    )
    
    # 构建中间件列表
    middleware: list = []
    
    # 如果配置了工具调用次数限制，则添加工具调用限制中间件
    if config.chat_model.max_tool_calls_per_turn > 0:
        tool_call_limiter = ToolCallLimitMiddleware(
            run_limit=config.chat_model.max_tool_calls_per_turn,
            exit_behavior="continue"  # 超过限制后返回错误信息，让模型决定何时结束
        )
        middleware.append(tool_call_limiter)
        logger.debug(f"工具调用限制已启用: 单回合最多 {config.chat_model.max_tool_calls_per_turn} 次")
    
    # 根据是否有中间件决定如何创建智能体
    if middleware:
        agent = create_agent(
            model=model,
            tools=tools,
            context_schema=GroupChatContext,
            middleware=middleware,
        )
    else:
        agent = create_agent(
            model=model,
            tools=tools,
            context_schema=GroupChatContext,
        )

    return agent

def format_agent_response_for_logging(response: dict) -> str:
    """将智能体响应格式化为人类可读的日志格式。
    
    解析智能体返回的字典，提取关键信息并格式化为清晰的日志输出。
    
    Args:
        response (dict): 智能体的原始响应字典，包含 messages 等键。
    
    Returns:
        str: 格式化后的人类可读字符串。
    """
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("智能体响应摘要")
    lines.append("=" * 50)
    
    # 提取消息列表
    messages = response.get("messages", [])
    if messages:
        # 跳过第一条消息（通常是系统消息）
        display_messages = messages[1:] if len(messages) > 1 else messages
        lines.append(f"消息数量: {len(display_messages)}")
        
        # 遍历所有消息
        for i, msg in enumerate(display_messages):
            msg_type = type(msg).__name__
            
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            
            # 检查是否有工具调用
            tool_calls_info = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in msg.tool_calls]
                tool_calls_info = f" [工具调用: {', '.join(tool_names)}]"
            
            lines.append(f"  [{i+1}] {msg_type}{tool_calls_info}: {content}")
    
    # 提取结构化响应（如果有）
    if "structured_response" in response:
        lines.append("-" * 30)
        lines.append(f"结构化响应: {response['structured_response']}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


async def handle_direct_text_output(
    response: dict,
    bot: Bot,
    group_id: int,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    interval: float = 0.2
) -> None:
    """处理 AI 直接输出的文本内容。
    
    当 AI 没有调用 send_group_message 工具但有文本输出时，
    自动将最后一条 AIMessage 的内容发送到群组。
    
    支持两种格式：
    1. send_message 代码块格式（会被解析为多条消息）
    2. 普通文本（作为单条消息发送）
    
    Args:
        response: 智能体的响应字典。
        bot: OneBot 机器人实例。
        group_id: 目标群组 ID。
        message_manager: 消息管理器。
        cache: 缓存管理器。
        interval: 多条消息之间的发送间隔（秒）。
    """
    from langchain_core.messages import AIMessage
    
    messages = response.get("messages", [])
    if not messages:
        return
    
    # 获取最后一条消息
    last_message = messages[-1]
    
    # 只处理 AIMessage 类型
    if not isinstance(last_message, AIMessage):
        return
    
    # 如果最后一条 AIMessage 有工具调用，说明 AI 已经通过工具发送了消息
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return
    
    # 获取文本内容
    content = last_message.content if hasattr(last_message, 'content') else ""
    if not content or not isinstance(content, str):
        return
    
    content = content.strip()
    if not content:
        return
    
    # 尝试解析 send_message 代码块
    parsed_messages = parse_send_message_blocks(content)
    
    if parsed_messages:
        # 有代码块格式的消息
        messages_to_send = parsed_messages
        logger.debug(f"解析到 {len(messages_to_send)} 条 send_message 代码块消息")
    else:
        # 普通文本，作为单条消息
        messages_to_send = [content]
        logger.debug("AI 直接输出文本，将作为单条消息发送")
    
    # 创建消息任务并发送
    task = MessageTask(
        messages=messages_to_send,
        group_id=group_id,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
        interval=interval
    )
    
    await group_message_queue_manager.enqueue(task)
    logger.info(f"已将 AI 直接输出的 {len(messages_to_send)} 条消息加入发送队列（群组 {group_id}）")


def group_messages_by_role(
    messages: List[GroupMessage], 
    self_id: int
) -> List[tuple[str, List[GroupMessage]]]:
    """将消息按发送者角色分组，生成交替的 user/assistant 对话格式。
    
    遍历消息列表，根据消息发送者是否为机器人来判断角色类型，
    将连续相同角色的消息合并为一组，确保输出为交替的对话格式，
    同时保持消息的时间顺序。
    
    Args:
        messages (List[GroupMessage]): 消息对象列表，应按时间顺序排列。
        self_id (int): 机器人的用户 ID。
    
    Returns:
        List[tuple[str, List[GroupMessage]]]: 交替的对话分组列表。
            每个元素为 (角色, 消息对象列表)，角色为 "assistant" 或 "user"。
    
    Example:
        输入消息顺序: [用户A, 用户B, 机器人, 用户C, 机器人]
        输出结果:
        [
            ("user", [GroupMessage用户A, GroupMessage用户B]),
            ("assistant", [GroupMessage机器人]),
            ("user", [GroupMessage用户C]),
            ("assistant", [GroupMessage机器人])
        ]
    """
    if not messages:
        return []
    
    result: List[tuple[str, List[GroupMessage]]] = []
    current_role: str = ""
    current_messages: List[GroupMessage] = []
    
    for msg in messages:
        role = "assistant" if msg.user_id == self_id else "user"
        
        if role != current_role:
            # 角色切换，保存之前的分组
            if current_messages:
                result.append((current_role, current_messages))
            current_role = role
            current_messages = []
        
        current_messages.append(msg)
    
    # 保存最后一组
    if current_messages:
        result.append((current_role, current_messages))
    
    return result


async def create_group_chat_context(
    messages: List[GroupMessage], 
    self_id: int,
    profile_manager: ProfileManager,
    user_id: int,
    group_id: int
) -> List[dict]:
    """创建群聊消息上下文信息。
    
    将群消息列表转换为 LLM 可用的对话上下文格式，包含系统提示和按角色分组的消息。
    同时注入当前用户和群聊的画像信息作为独立的 user 消息段。
    
    Args:
        messages (List[GroupMessage]): 消息对象列表。
        self_id (int): 机器人的用户 ID。
        profile_manager (ProfileManager): 画像管理器。
        user_id (int): 当前交互用户的 ID。
        group_id (int): 当前群聊的 ID。
    
    Returns:
        List[dict]: LLM 对话上下文列表，每个元素包含 "role" 和 "content" 字段。
    """
    context: List[dict] = []

    # 添加系统提示
    context.append({
        "role": "system",
        "content": config.chat_model.prompt
    })

    # 获取当前用户和群聊的画像信息，作为独立的 user 消息段注入
    profile_info_parts: List[str] = []
    
    # 获取当前交互用户的画像
    try:
        user_profile = await profile_manager.user.get_user_descriptions_with_index(user_id)
        if user_profile:
            user_profile_text = f"[当前用户 {user_id} 的画像]\n"
            for idx, desc in user_profile.items():
                user_profile_text += f"  {idx}. {desc}\n"
            profile_info_parts.append(user_profile_text.strip())
    except Exception as e:
        logger.warning(f"获取用户 {user_id} 画像失败: {e}")
    
    # 获取当前群聊的画像
    try:
        group_profile = await profile_manager.group.get_group_descriptions_with_index(group_id)
        if group_profile:
            group_profile_text = f"[当前群聊 {group_id} 的画像]\n"
            for idx, desc in group_profile.items():
                group_profile_text += f"  {idx}. {desc}\n"
            profile_info_parts.append(group_profile_text.strip())
    except Exception as e:
        logger.warning(f"获取群聊 {group_id} 画像失败: {e}")
    
    # 如果有画像信息，作为独立的 user 消息段注入（放在聊天记录之前）
    if profile_info_parts:
        profile_context = "[系统提示] 以下是当前对话相关的画像信息：\n\n" + "\n\n".join(profile_info_parts)
        context.append({
            "role": "user",
            "content": profile_context
        })

    # 添加用户和助手消息（交替格式，保持时间顺序）
    grouped_messages = group_messages_by_role(messages, self_id)

    # 生成文本内容
    grouped_messages_text = [(i[0], await Fun.format_messages_to_text(i[1])) for i in grouped_messages]
    for role, content in grouped_messages_text:
        context.append({
            "role": role,
            "content": content
        })
    
    return context


async def handle_request_rejection(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """处理被拒绝的请求，发送拒绝表情回应。
    
    当响应被拒绝（锁已满）时，如果配置了拒绝表情ID，则发送表情回应。
    
    Args:
        bot (Bot): OneBot 机器人实例。
        msg_id (int): 被拒绝的消息 ID。
        group_id (int): 群组 ID。
    
    Note:
        - 如果 rejection_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.rejection_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.rejection_emoji_id
        )
        logger.debug(f"已发送拒绝表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.rejection_emoji_id}）")
    except Exception as e:
        logger.error(f"发送拒绝表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_processing_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在开始处理请求时发送处理中表情回应。
    
    当开始处理用户请求时，如果配置了处理中表情ID，则发送表情回应。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 如果 processing_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.processing_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.processing_emoji_id
        )
        logger.debug(f"已发送处理中表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.processing_emoji_id}）")
    except Exception as e:
        logger.error(f"发送处理中表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_completion_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在完成请求处理后发送完成表情回应。
    
    当成功完成用户请求处理后，如果配置了完成表情ID，则发送表情回应。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 如果 completion_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.completion_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.completion_emoji_id
        )
        logger.debug(f"已发送完成表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.completion_emoji_id}）")
    except Exception as e:
        logger.error(f"发送完成表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


async def handle_timeout_emoji(
    bot: Bot,
    msg_id: int,
    group_id: int
) -> None:
    """在API请求超时时发送拒绝表情回应。
    
    当API请求超时时，如果配置了拒绝表情ID，则发送表情回应。
    
    Args:
        bot: OneBot 机器人实例。
        msg_id: 消息 ID。
        group_id: 群组 ID。
    
    Note:
        - 复用 rejection_emoji_id 配置
        - 如果 rejection_emoji_id 为 -1，则不发送表情
        - 表情发送失败时只记录错误，不中断流程
    """
    if config.chat_model.rejection_emoji_id == -1:
        return
    
    try:
        await Fun.set_msg_emoji_like(
            bot=bot,
            message_id=msg_id,
            emoji_id=config.chat_model.rejection_emoji_id
        )
        logger.debug(f"已发送超时表情（群组 {group_id}，消息ID {msg_id}，表情ID {config.chat_model.rejection_emoji_id}）")
    except Exception as e:
        logger.error(f"发送超时表情失败（群组 {group_id}，消息ID {msg_id}）: {str(e)}")


GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)



@GroupChatProactiveRequest.handle()
async def handle_group_chat_request(
    event: GroupMessageEvent, 
    bot: Bot, 
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager),
    profile_manager: ProfileManager = Depends(get_profile_manager)
):
    """处理群聊中的主动请求消息。
    
    该函数实现了响应事件的锁机制，支持两个级别的并发控制：
    - 群组级别：限制单个群组同时进行的响应事件数
    - 全局级别：限制所有群组总共同时进行的响应事件数
    
    当锁满时会直接拒绝本次请求。
    
    新增功能：
    - 接收请求时发送处理中表情贴（processing_emoji_id）
    - 完成请求时发送完成表情贴（completion_emoji_id）
    - API 请求超时处理（api_timeout_sec）
    """
    
    msg_id: int = event.message_id
    group_id: int = event.group_id
    
    # 尝试获取响应锁（非阻塞）
    if not response_lock_manager.try_acquire(group_id):
        logger.warning(f"响应锁已满（群组 {group_id}，消息ID {msg_id}），拒绝本次请求")
        await handle_request_rejection(bot, msg_id, group_id)
        return
    
    try:
        logger.info(f"获得响应锁（群组 {group_id}，消息ID {msg_id}），开始处理请求")
        
        # 发送处理中表情贴
        await handle_processing_emoji(bot, msg_id, group_id)
        
        max_messages: int = config.chat_model.maximum_number_of_incoming_messages\
                            + NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES
        
        # 获取聊天记录信息
        messages: List[GroupMessage] = await msg_mg.get_nearby_messages(
            message_id=msg_id,
            group_id=group_id,
            before=max_messages
        )

        # 截取最后 N 条消息作为上下文
        relevant_messages: list[GroupMessage] = messages[-config.chat_model.maximum_number_of_incoming_messages:]
        logger.debug(
            f"处理群聊请求: 群号 {group_id}, 消息ID {msg_id}, 上下文消息数 {len(relevant_messages)}"
        )
        logger.debug(
            f"上下文消息列表: {await Fun.format_messages_to_text(relevant_messages, template=config.message_format_placeholder)}"
        )

        # 创建聊天上下文
        chat_context = await create_group_chat_context(
            messages=relevant_messages,
            self_id=int(bot.self_id),
            profile_manager=profile_manager,
            user_id=event.user_id,
            group_id=group_id
        )
        # 创建聊天智能体
        chat_agent = create_group_chat_agent()
        # 生成适配 LangChain 格式的消息
        chat_context = convert_openai_to_langchain_messages(chat_context)
        
        # 构建 API 调用的协程
        api_coro = chat_agent.ainvoke(
            input={"messages": chat_context},
            context=GroupChatContext(
                bot=bot,
                group_id=group_id,
                user_id=event.user_id,
                message_id=msg_id,
                message_manager=msg_mg,
                cache=cache,
                profile_manager=profile_manager
                )
            )
        
        # 根据配置决定是否使用超时控制
        timeout_sec = config.chat_model.api_timeout_sec
        if timeout_sec > 0:
            try:
                response = await wait_for(api_coro, timeout=timeout_sec)
            except AsyncTimeoutError:
                logger.error(f"API 请求超时（群组 {group_id}，消息ID {msg_id}，超时时间 {timeout_sec}秒）")
                await handle_timeout_emoji(bot, msg_id, group_id)
                await GroupChatProactiveRequest.send(
                    f"请求处理超时（{timeout_sec}秒），请稍后重试",
                    at_sender=True
                )
                return
        else:
            # 不设置超时
            response = await api_coro

        # 格式化并输出人类可读的响应日志
        formatted_response = format_agent_response_for_logging(response)
        logger.info(f"群聊智能体响应:\n{formatted_response}")
        
        # 处理 AI 直接输出的文本内容（未通过 send_group_message 工具发送的情况）
        await handle_direct_text_output(response, bot, group_id, msg_mg, cache)
        
        # 发送完成表情贴
        await handle_completion_emoji(bot, msg_id, group_id)
        
        logger.info(f"响应处理完成（群组 {group_id}，消息ID {msg_id}），释放响应锁")
    
    except AsyncTimeoutError:
        # 超时错误已在上面处理，这里作为备用捕获
        logger.error(f"API 请求超时（群组 {group_id}，消息ID {msg_id}）")
        await handle_timeout_emoji(bot, msg_id, group_id)
    
    except Exception as e:
        logger.error(f"处理群聊请求时发生错误（群组 {group_id}，消息ID {msg_id}）: {str(e)}", exc_info=True)
        await GroupChatProactiveRequest.send(
            f"处理请求时发生错误: {str(e)}",
            at_sender=True
        )
    
    finally:
        response_lock_manager.release(group_id)


# ============================================================================
# 管理员处理器 - 查看用户和群聊画像
# ============================================================================

# 管理员用户ID（写死）
ADMIN_USER_ID = 2126979584

# 查看用户画像处理器
AdminQueryUserProfile = on_message(priority=4, block=False)

@AdminQueryUserProfile.handle()
async def handle_query_user_profile(
    event: GroupMessageEvent,
    bot: Bot,
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    profile_manager: ProfileManager = Depends(get_profile_manager)
):
    """处理管理员查看用户画像的请求。
    
    命令格式: 查看用户画像 <用户QQ号>
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
        msg_mg: 消息管理器。
        profile_manager: 画像管理器。
    """
    # 仅允许管理员使用
    if event.user_id != ADMIN_USER_ID:
        return
    
    # 检查消息格式
    msg_text = event.get_plaintext().strip()
    if not msg_text.startswith("查看用户画像"):
        return
    
    # 提取用户QQ号
    parts = msg_text.split()
    if len(parts) < 2:
        await AdminQueryUserProfile.send("❌ 命令格式错误。用法: 查看用户画像 <用户QQ号>")
        return
    
    try:
        target_user_id = int(parts[1])
    except ValueError:
        await AdminQueryUserProfile.send(f"❌ 无效的用户QQ号: {parts[1]}")
        return
    
    try:
        logger.info(f"管理员 {event.user_id} 查询用户 {target_user_id} 的画像")
        
        # 获取用户画像
        user_profile = await profile_manager.user.get_user_descriptions_with_index(target_user_id)
        
        if not user_profile:
            await AdminQueryUserProfile.send(f"ℹ️ 用户 {target_user_id} 尚未设置画像描述。")
            return
        
        # 格式化输出
        profile_text = f"👤 用户 {target_user_id} 的画像描述：\n"
        profile_text += "=" * 40 + "\n"
        
        for idx, description in user_profile.items():
            profile_text += f"[{idx}] {description}\n"
        
        profile_text += "=" * 40
        
        await AdminQueryUserProfile.send(profile_text)
        logger.info(f"已为管理员 {event.user_id} 展示用户 {target_user_id} 的 {len(user_profile)} 条画像描述")
        
    except Exception as e:
        logger.error(f"查询用户 {target_user_id} 画像失败: {str(e)}")
        await AdminQueryUserProfile.send(f"❌ 查询失败: {str(e)}")


# 查看群聊画像处理器
AdminQueryGroupProfile = on_message(priority=4, block=False)

@AdminQueryGroupProfile.handle()
async def handle_query_group_profile(
    event: GroupMessageEvent,
    bot: Bot,
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    profile_manager: ProfileManager = Depends(get_profile_manager)
):
    """处理管理员查看群聊画像的请求。
    
    命令格式: 查看群聊画像 [群号]
    若不指定群号，则查看当前群聊的画像。
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
        msg_mg: 消息管理器。
        profile_manager: 画像管理器。
    """
    # 仅允许管理员使用
    if event.user_id != ADMIN_USER_ID:
        return
    
    # 检查消息格式
    msg_text = event.get_plaintext().strip()
    if not msg_text.startswith("查看群聊画像"):
        return
    
    # 提取群号（可选，默认为当前群）
    target_group_id = event.group_id
    parts = msg_text.split()
    
    if len(parts) >= 2:
        try:
            target_group_id = int(parts[1])
        except ValueError:
            await AdminQueryGroupProfile.send(f"❌ 无效的群号: {parts[1]}")
            return
    
    try:
        logger.info(f"管理员 {event.user_id} 查询群聊 {target_group_id} 的画像")
        
        # 获取群聊画像
        group_profile = await profile_manager.group.get_group_descriptions_with_index(target_group_id)
        
        if not group_profile:
            await AdminQueryGroupProfile.send(f"ℹ️ 群聊 {target_group_id} 尚未设置画像描述。")
            return
        
        # 格式化输出
        profile_text = f"👥 群聊 {target_group_id} 的画像描述：\n"
        profile_text += "=" * 40 + "\n"
        
        for idx, description in group_profile.items():
            profile_text += f"[{idx}] {description}\n"
        
        profile_text += "=" * 40
        
        await AdminQueryGroupProfile.send(profile_text)
        logger.info(f"已为管理员 {event.user_id} 展示群聊 {target_group_id} 的 {len(group_profile)} 条画像描述")
        
    except Exception as e:
        logger.error(f"查询群聊 {target_group_id} 画像失败: {str(e)}")
        await AdminQueryGroupProfile.send(f"❌ 查询失败: {str(e)}")