from time import time
from langchain_core.tools.base import BaseTool
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List, Callable, Any, Union
from typing import cast

from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends, EventMessage, CommandArg
from nonebot import on, on_message, get_driver, on_notice, on_command
from nonebot.rule import to_me
from pathlib import Path
from asyncio import Semaphore, Queue, Lock, TimeoutError as AsyncTimeoutError, wait_for
from dataclasses import dataclass
from asyncio import sleep, create_task, Event

from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI # TODO: 未来支持更多的提供商
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph
from pydantic import SecretStr


from .DBmodel import GroupMessage
from .MassageManager import GroupMessageManager, get_message_manager
from .ConfigManager import total_config, ProcessedConfiguration
from . import Fun
from . import CacheManager
from .UserProfileManager import ProfileManager, get_profile_manager
from .GroupChatContext import GroupChatContext
from .LLM_Tools import get_current_tools
from .constants import (
    DEFAULT_BOT_NAME_PLACEHOLDER,
    NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES,
    SEND_MESSAGE_BLOCK_PATTERN,
    SUPPORTED_CQ_CODES,
    NOTE_TAG_PATTERN
)
from .GroupMessageQueueManager import GroupMessageQueueManager, MessageTask, group_message_queue_manager
from .Internal_tools import (
    delete_message_tool,
    emoji_reaction_tool,
    poke_user_tool,
    send_group_message_tool,
    send_like_tool,
    take_notes,
)
from .services.LongMemory import long_memory_manager





GroupChatContext.model_rebuild()

config = total_config.processed_configuration.current_config_group


def _normalize_notepad_session_id(raw_session_id: str, *, default_group_id: int | None) -> str:
    """规范化记事本会话 ID。

    支持以下输入形式：
    - 空字符串：使用默认群聊会话（需要提供 default_group_id）。
    - 纯数字：视为群号，转换为 "group_<群号>"。
    - 其它字符串：原样使用（会做 strip）。

    Args:
        raw_session_id: 原始输入会话 ID 文本。
        default_group_id: 默认群号；当 raw_session_id 为空时使用。

    Returns:
        规范化后的会话 ID。

    Raises:
        ValueError: 当 raw_session_id 为空且 default_group_id 未提供时抛出。
    """
    raw = (raw_session_id or "").strip()
    if not raw:
        if default_group_id is None:
            raise ValueError("缺少默认群号，无法推断会话 ID")
        return f"group_{default_group_id}"

    if raw.isdigit():
        return f"group_{int(raw)}"

    return raw


# ============================================================================
# 消息发送队列管理器（生产者-消费者模型）
# ============================================================================



# 创建model缓存
agent_cache_info: dict = {
    "model": None,
    "model_id": None,
    "base_url": None,
    "api_key": None,
    "model_kwargs": None,
}


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
        >>> text = '''
        ... <msg>你好！</msg>
        ... <msg>再见！</msg>
        ... '''
        >>> parse_send_message_blocks(text)
        ['你好！', '再见！']
    """
    matches = SEND_MESSAGE_BLOCK_PATTERN.findall(content)
    # 过滤空消息并去除首尾空白
    return [msg.strip() for msg in matches if msg.strip()]




def extract_note_tags(content: str) -> tuple[list[str], str]:
    """提取 <note>...</note> 标签并返回剩余文本。

    标签用途：把标签内文本作为“记事本的一条记录”写入会话记事本。
    该函数只负责解析与剔除标签，不负责落库/写入记事本。

    Args:
        content: 原始文本。

    Returns:
        tuple[list[str], str]: (note 列表, 剩余文本)。
            - note 列表会按出现顺序返回，且每条会做 strip。
            - 剩余文本会移除所有 note 标签并做 strip。
    """
    if not content:
        return [], ""

    notes = [n.strip() for n in NOTE_TAG_PATTERN.findall(content) if n and n.strip()]
    remaining = NOTE_TAG_PATTERN.sub("", content).strip()
    return notes, remaining

# 初始化全局响应锁管理器
response_lock_manager = ResponseLockManager(
    max_concurrent_responses_per_group=config.chat_model.max_concurrent_responses_per_group,
    max_total_concurrent_responses=config.chat_model.max_total_concurrent_responses
)
"""全局响应锁管理器，用于控制聊群响应的并发数。"""



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

    tools: List[BaseTool] = [
        send_group_message_tool,
        delete_message_tool,
        emoji_reaction_tool,
        poke_user_tool,
        send_like_tool,
        take_notes,
    ]

    tools.extend(get_current_tools())

    global agent_cache_info

    # ===== 1) 检查缓存是否命中 =====
    t1 = time()
    api_key = config.chat_model.api_key

    cache_hit = (
        agent_cache_info.get("model") is not None
        and agent_cache_info.get("model_id") == config.chat_model.model_id
        and agent_cache_info.get("base_url") == config.chat_model.base_url
        and agent_cache_info.get("api_key") == api_key
        and agent_cache_info.get("model_kwargs") == config.chat_model.parameters
    )

    if cache_hit:
        model = agent_cache_info["model"]
        logger.debug("命中模型缓存")
    else:
        model = ChatOpenAI(
            model=config.chat_model.model_id,
            base_url=config.chat_model.base_url,
            api_key=SecretStr(api_key),
            model_kwargs=config.chat_model.parameters,
        )
        agent_cache_info.update({
            "model": model,
            "model_id": config.chat_model.model_id,
            "base_url": config.chat_model.base_url,
            "api_key": api_key,
            "model_kwargs": config.chat_model.parameters,
        })
        logger.debug("模型缓存已更新")

    logger.info(f"模型创建耗时: {time() - t1:.2f}")

    # ===== 2) 构建中间件 =====
    middleware = []
    t1 = time()
    if config.chat_model.max_tool_calls_per_turn > 0:
        middleware.append(
            ToolCallLimitMiddleware(
                run_limit=config.chat_model.max_tool_calls_per_turn,
                exit_behavior="continue"
            )
        )
        logger.debug(f"工具调用限制已启用: 单回合最多 {config.chat_model.max_tool_calls_per_turn} 次")

    logger.info(f"中间件创建耗时: {time() - t1:.2f}")

    # ===== 3) 创建智能体 =====
    t1 = time()
    agent = create_agent(
        model=model,
        tools=tools,
        context_schema=GroupChatContext,
        middleware=middleware,
    )

    logger.info(f"实际智能体创建耗时: {time() - t1:.2f}")
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
    msgs: list[AIMessage|HumanMessage|SystemMessage|ToolMessage] = response.get("messages", [])

    for i in msgs:
        lines.append(f"==== {i.type} ====")
        lines.append(cast(str, i.content))

    # return str(msgs)
    return "\n".join(lines)


async def process_assistant_direct_output(
    response: dict,
    bot: Bot,
    group_id: int,
    message_manager: GroupMessageManager,
    cache: CacheManager.UserCacheManager,
    interval: float = 0.2
) -> None:
    """处理智能体响应中的直接输出文本（包含工具调用场景）。

    无论智能体是否调用了工具，本函数都会读取最后一条 `AIMessage.content`，
    并按以下规则处理：

    1) 支持 `<note>...</note>` 标签：将标签内文本按出现顺序写入会话记事本，
       且从可发送文本中移除这些标签。
    2) 若剩余文本包含 ```send_message 代码块，则解析为多条消息发送。
    3) 否则将剩余文本作为单条消息发送。
    4) 若输出仅包含 `<note>` 且剔除后无可发送文本，则只记录不发送。

    Args:
        response (dict): 智能体响应字典，需包含 `messages`。
        bot (Bot): OneBot 机器人实例。
        group_id (int): 目标群组 ID。
        message_manager (GroupMessageManager): 消息管理器。
        cache (CacheManager.UserCacheManager): 缓存管理器。
        interval (float): 多条消息之间的发送间隔（秒）。默认 0.2。
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
    
    # 获取文本内容（可能为空；例如仅包含工具调用）
    content = last_message.content if hasattr(last_message, 'content') else ""
    if not content or not isinstance(content, str):
        return
    
    content = content.strip()
    if not content:
        return

    # 处理 <note>...</note>：写入记事本，并从可发送文本中移除
    notes, content = extract_note_tags(content)
    if notes:

        session_id = f"group_{group_id}"
        notepad_manager = long_memory_manager.notepad_manager
        for note in notes:
            notepad_manager.add_note(session_id, note)
        logger.info(f"已写入 {len(notes)} 条记事本记录（会话 {session_id}）")

    if not content:
        return
    
    # 尝试解析 send_message 代码块
    parsed_messages = parse_send_message_blocks(content)
    
    if parsed_messages:
        messages_to_send = parsed_messages
        logger.debug(f"解析到 {len(messages_to_send)} 条 send_message 代码块消息\n{messages_to_send}")
    else:
        # 普通文本：作为单条消息发送
        messages_to_send = [content]

    task = MessageTask(messages=messages_to_send, group_id=group_id, interval=interval)
    await group_message_queue_manager.enqueue(
        task,
        bot=bot,
        message_manager=message_manager,
        cache=cache,
    )
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

    # 剔除部分CQ码参数
    def replace_func(di: dict[str, str]):
        if di["CQ"] == "mface": # 表情
            summary = di.get("summary", "")
            return "[CQ:mface" + (f",summary={summary}" if summary else "") + "]" 
        elif di["CQ"] == "record": # 语音
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:record" + (f",file={file}" if file else "") + \
                    (f",file_size={file_size}" if file_size else "") + "]"
        elif di["CQ"] == "image": # 图片
            file = di.get("file", "")
            file_size = di.get("file_size", "")
            return "[CQ:image" + (f",file={file}" if file else "") + \
                    (f",file_size={file_size}" if file_size else "") + "]"
        else:
            text = Fun.generate_cq_code([di])[0]
            if len(text) > 100:
                return f"{text[:100]}...]"

    for message in messages:
        message.content = Fun.replace_cq_codes(message.content, replace_func=replace_func)



    
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

    # 注入当前会话的记事本信息（放在聊天记录之前）
    try:

        session_id = f"group_{group_id}"
        notepad_manager = long_memory_manager.notepad_manager
        if notepad_manager.has_session(session_id):
            notes_text = notepad_manager.get_notes(session_id).strip()
            if notes_text:
                notepad_context = (
                    "[系统提示] 以下是当前会话的记事本记录（用于补充短中期记忆，可能与当前问题无关）：\n\n"
                    + notes_text
                )
                context.append({
                    "role": "user",
                    "content": notepad_context,
                })
    except Exception as e:
        logger.warning(f"获取群聊 {group_id} 记事本失败: {e}")
    

    # 在AI自己的消息中的名字加上"(我)"后缀
    for i in messages:
        if i.user_id == self_id:
            i.user_name = i.user_name + "(我)"

    # 添加用户和助手消息：将整个历史信息合并为单个 HumanMessage（role="user"）
    history_text = (await Fun.format_messages_to_text(messages, template=config.message_format_placeholder)).strip()
    if history_text:
        context.append({
            "role": "user",
            "content": "[历史对话]\n\n" + history_text,
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


# ============================================================================
# 记事本查询处理器
# ============================================================================

QueryNotepad = on_command(
    "记事本",
    aliases={"查看记事本", "notepad", "notebook"},
    priority=-5,
    block=True,
)


@QueryNotepad.handle()
async def handle_query_notepad(
    event: MessageEvent,
    bot: Bot,
    args: Message = CommandArg(),
) -> None:
    """输出当前会话的记事本内容。

    命令格式:
        /记事本 [会话ID]

    参数规则:
        - 不填参数：默认使用当前群聊会话（"group_<群号>"）。
        - 参数为纯数字：视为群号，自动转换为 "group_<群号>"。
        - 其它字符串：作为会话 ID 原样使用。

    Args:
        event: 消息事件（群聊或私聊）。
        bot: 机器人实例。
        args: 命令参数。
    """


    arg_text = args.extract_plain_text().strip()
    default_group_id = event.group_id if isinstance(event, GroupMessageEvent) else None

    try:
        session_id = _normalize_notepad_session_id(arg_text, default_group_id=default_group_id)
    except ValueError:
        await QueryNotepad.finish("❌ 该命令仅支持群聊默认会话；请显式提供会话ID，例如：/记事本 group_123")

    notepad_manager = long_memory_manager.notepad_manager
    if not notepad_manager.has_session(session_id):
        await QueryNotepad.finish(f"ℹ️ 会话 {session_id} 暂无记事本记录。")

    notes_text = notepad_manager.get_notes(session_id).strip()
    if not notes_text:
        await QueryNotepad.finish(f"ℹ️ 会话 {session_id} 暂无记事本记录。")

    header = f"📝 会话记事本（{session_id}）：\n"
    await QueryNotepad.send(header + notes_text)



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

    first_time: float = time()

    get_lock_time: float = time()
    # 尝试获取响应锁（非阻塞）
    if not response_lock_manager.try_acquire(group_id):
        logger.warning(f"响应锁已满（群组 {group_id}，消息ID {msg_id}），拒绝本次请求")
        await handle_request_rejection(bot, msg_id, group_id)
        return
    logger.info(f"获取锁耗时: {time() - get_lock_time:.2f}s")
    
    try:
        logger.info(f"获得响应锁（群组 {group_id}，消息ID {msg_id}），开始处理请求")
        
        # 发送处理中表情贴
        await handle_processing_emoji(bot, msg_id, group_id)
        
        max_messages: int = config.chat_model.maximum_number_of_incoming_messages\
                            + NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES

        get_message_time = time()
        # 获取聊天记录信息
        messages: List[GroupMessage] = await msg_mg.get_nearby_messages(
            message_id=msg_id,
            group_id=group_id,
            before=max_messages
        )
        logger.info(f"获取聊天记录耗时: {time() - get_message_time:.2f}s")

        # 截取最后 N 条消息作为上下文
        relevant_messages: list[GroupMessage] = messages[-config.chat_model.maximum_number_of_incoming_messages:]
        logger.debug(
            f"处理群聊请求: 群号 {group_id}, 消息ID {msg_id}, 上下文消息数 {len(relevant_messages)}"
        )
        logger.debug(
            f"上下文消息列表: {await Fun.format_messages_to_text(relevant_messages, template=config.message_format_placeholder)}"
        )

        creat_agent_time = time()
        # 创建聊天上下文
        chat_context = await create_group_chat_context(
            messages=relevant_messages,
            self_id=int(bot.self_id),
            profile_manager=profile_manager,
            user_id=event.user_id,
            group_id=group_id
        )
        logger.info(f"创建聊天上下文耗时: {time() - creat_agent_time:.2f}s")
        t1 = time()
        # 创建聊天智能体
        chat_agent = create_group_chat_agent()
        logger.info(f"t1: {time() - t1:.2f}s")
        t1 = time()
        # 生成适配 LangChain 格式的消息
        chat_context = convert_openai_to_langchain_messages(chat_context)
        logger.info(f"t2: {time() - t1:.2f}s")
        
        logger.info(f"创建agent耗时: {time()-creat_agent_time:.2f}s")
        logger.info(f"总耗时: {time() - first_time:.2f}s")

        # LongMemory 工具通过 ToolRuntime.context 获取 long_memory 与会话信息，无需额外注入。
        
        # 构建 API 调用的协程
        api_coro = chat_agent.ainvoke(
            input={"messages": chat_context},
            context=GroupChatContext(
                bot=bot,
                event=event,
                message=msg,
                group_id=group_id,
                user_id=event.user_id,
                message_id=msg_id,
                message_manager=msg_mg,
                cache=cache,
                long_memory=long_memory_manager
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
        await process_assistant_direct_output(response, bot, group_id, msg_mg, cache)
        
        # 发送完成表情贴
        await handle_completion_emoji(bot, msg_id, group_id)
        
        logger.info(f"响应处理完成（群组 {group_id}，消息ID {msg_id}），释放响应锁")
    
    except AsyncTimeoutError:
        # 超时错误已在上面处理，这里作为备用捕获
        logger.error(f"API 请求超时（群组 {group_id}，消息ID {msg_id}）")
        await handle_timeout_emoji(bot, msg_id, group_id)
    
    except Exception as e:
        try:
            error_detail = repr(e) 
        except:
            error_detail = "无法解析的具体错误类型"

        logger.error(f"处理群聊请求时发生错误（群组 {group_id}，消息ID {msg_id}）")
        logger.error(f"错误堆栈: ", exc_info=True)

        await GroupChatProactiveRequest.send(
            f"处理请求时发生错误，请联系管理员或检查API状态。详情: {error_detail}",
            at_sender=True
        )
    
    finally:
        response_lock_manager.release(group_id)