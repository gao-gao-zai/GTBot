from time import time
from nb_cli import cache
from nonebot import logger
from pydantic import BaseModel, ConfigDict
from typing import List
from nonebot.adapters.onebot.v11.message import Message, MessageSegment
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.params import Depends, EventMessage
from nonebot import on, on_message, get_driver, on_notice
from nonebot.rule import to_me
from pathlib import Path

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI # TODO: 未来支持更多的提供商
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import SecretStr


from .model import GroupMessage
from .MassageManager import GroupMessageManager, get_message_manager, GroupMessageManager
from .ConfigManager import total_config, ProcessedConfiguration
from . import Fun
from . import CacheManager
from asyncio import sleep, create_task

config = total_config.processed_configuration.current_config_group


# 定义常量
NUMBER_OF_REDUNDANT_ACQUIRED_MESSAGES = 20
"""在获取附近消息时，额外获取的冗余消息数量，以确保上下文的完整性。"""

DEFAULT_BOT_NAME_PLACEHOLDER = "GTBot"
"""当找不到机器人名称时用的占位字符串。"""

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
        
        Args:
            message (str | List[str]): 要发送的消息内容，可以是单条消息或消息列表。
            runtime (ToolRuntime[GroupChatContext]): 工具运行时上下文。无需手动传入，由框架自动提供。
            group_id (int | None): 目标群组 ID。不填则自动获取当前的聊群ID。
            interval (float): 发送多条消息时的间隔时间（秒），默认为 0.2。
        
        Returns:
            str: 发送结果信息。
        
        Note:
            TODO: 完成 CQ 码转义处理。
        """
        
        if group_id is None:
            group_id = runtime.context.group_id

        # 将单条消息转换为列表
        messages: List[str] = [message] if isinstance(message, str) else message
        
        logger.info(
            f"工具调用: 向群组 {group_id} 发送 {len(messages)} 条消息"
        )

        async def send_messages_async() -> None:
            """异步发送消息列表。"""
            msg_mg = runtime.context.message_manager
            cache = runtime.context.cache
            bot = runtime.context.bot
            
            for idx, msg_content in enumerate(messages):
                # TODO: 完成CQ码转义
                processed_message: Message = Message(MessageSegment.text(msg_content))
                
                result = await bot.send_group_msg(
                    group_id=group_id,
                    message=processed_message
                )

                # 将消息填回消息数据库
                await msg_mg.add_message(
                    GroupMessage(
                        message_id=result["message_id"],
                        group_id=group_id,
                        user_id=int(bot.self_id),
                        user_name=await cache.get_user_name(bot, int(bot.self_id)) or DEFAULT_BOT_NAME_PLACEHOLDER,
                        content=msg_content,
                        send_time=time(),
                        is_withdrawn=False
                    )
                )
                
                # 如果不是最后一条消息，等待指定间隔
                if idx < len(messages) - 1:
                    await sleep(interval)

        # 开启新协程发送消息，防止堵塞
        create_task(send_messages_async())
        
        return f"消息已提交发送至群组 {group_id}（共 {len(messages)} 条）"


# 将工具函数用 @tool 装饰并导出
send_group_message_tool = tool(TollCalls.send_group_message)


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
    """

    tools = [send_group_message_tool]
    model = ChatOpenAI(
        model = config.chat_model.model_id,
        base_url=config.chat_model.base_url,
        api_key=SecretStr(config.chat_model.api_key),
        model_kwargs=config.chat_model.parameters
    )
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


async def create_group_chat_context(messages: List[GroupMessage], self_id: int) -> List[dict]:
    """创建群聊消息上下文信息。
    
    将群消息列表转换为 LLM 可用的对话上下文格式，包含系统提示和按角色分组的消息。
    
    Args:
        messages (List[GroupMessage]): 消息对象列表。
        self_id (int): 机器人的用户 ID。
    
    Returns:
        List[dict]: LLM 对话上下文列表，每个元素包含 "role" 和 "content" 字段。
    """
    context: List[dict] = []

    # 添加系统提示
    context.append({
        "role": "system",
        "content": config.chat_model.prompt    
    })

    # TODO: 添加其他元数据

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

GroupChatProactiveRequest = on_message(rule=to_me(), priority=5, block=False)



@GroupChatProactiveRequest.handle()
async def handle_group_chat_request(
    event: GroupMessageEvent, 
    bot: Bot, 
    msg: Message = EventMessage(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    cache: CacheManager.UserCacheManager = Depends(CacheManager.get_user_cache_manager)
):
    """处理群聊中的主动请求消息"""

    msg_id: int = event.message_id
    group_id: int = event.group_id
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

    # 创建聊天上下文
    chat_context = await create_group_chat_context(
        messages=relevant_messages,
        self_id=int(bot.self_id)
    )
    # 创建聊天智能体
    chat_agent = create_group_chat_agent()
    # 生成适配 LangChain 格式的消息
    chat_context = convert_openai_to_langchain_messages(chat_context)
    # 生成响应
    response = await chat_agent.ainvoke(
        input={"messages": chat_context},
        context=GroupChatContext(
            bot=bot,
            group_id=group_id,
            user_id=event.user_id,
            message_id=msg_id,
            message_manager=msg_mg,
            cache=cache
            )
        )

    # 格式化并输出人类可读的响应日志
    formatted_response = format_agent_response_for_logging(response)
    logger.info(f"群聊智能体响应:\n{formatted_response}")