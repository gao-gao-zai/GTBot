import random
from nonebot import on_message, get_driver, logger, on_type
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, ActionFailed
from nonebot.rule import to_me
import re
import asyncio
import time
import sys
import os
import json
import tomli
from pathlib import Path
from datetime import datetime
from typing import Any, TypeVar, Optional, Union, Literal
from collections import OrderedDict, defaultdict

# TODO: 添加observe_tool_result工具实现


# 要添加对撤回消息的适配，在数据库中添加一列标记是否撤回

# --- 常量配置区域 ---
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

BOT_SYSTEM_ID = -2
BOT_SYSTEM_NAME = "BOT_system"
SINGLE_SESSION_CONTINUOUS_CALL_TO_OBSERVE_TOOL_RESULT_LIMIT = 5


# --- 导入模块 ---
from nonebotSQL import chat_record_db, image_description_cache_db, group_message_manager
from SQLiteManager import Message, GroupMessage
from fun import toolbox, parse_cq_codes, file_to_sha256, replace_cq_codes
from chatgpt import ChatGPT
from config_manager import config_group_data as gcm
from SessionManager import SessionManager



# -----------------
# --- 全局变量 ---
driver = get_driver()
session_manager = SessionManager()
image_failure_tracker = defaultdict(lambda: {"count": 0, "last_fail_time": 0.0})
# 用户元数据内存缓存, 键为user_id, 值为包含数据和时间戳的字典
user_metadata_cache = {}
# 缓存有效期 (单位: 秒)
CACHE_TTL_SECONDS = 3600  # 1小时
# 用于保护用户元数据缓存的异步锁
user_cache_lock = asyncio.Lock()
# ----------------

# --- 数据库初始化 ---
@driver.on_startup
async def init_database():
    global private_chat_table, group_chat_table, gcm, image_description_cache_table, rag_manager, rag_loading_complete
    private_chat_table = chat_record_db.table("private_chat_record")
    group_chat_table = chat_record_db.table("group_chat_record")
    image_description_cache_table = image_description_cache_db.table("image_cache")
    rag_loading_complete = False
    if gcm.Retrieval_Augmented_Generation.enable:
        from NRAGmanager import rag_manager, rag_initialization_complete
        await rag_initialization_complete.wait()
        rag_loading_complete = True
# --------------------

class get_formatted_history_messages_return:
    text: str
    user_id: int

    def __init__(self, text: str, user_id: int):
        self.text = text
        self.user_id = user_id


class ToolCallManager:
    class Tools:
        def __init__(self, bot:Bot, event:GroupMessageEvent, bot_system_id: int, bot_system_name: str) -> None:
            self.bot = bot
            self.event = event
            self.bot_system_id = bot_system_id
            self.bot_system_name = bot_system_name

        async def like(self, user_id: int, times: int = 10):
            """
            给指定用户点赞
            """
            if not isinstance(user_id, int) or not isinstance(times, int):
                return "参数类型错误"
            try:
                await self.bot.send_like(
                    user_id=user_id,
                    times=times
                )
            except ActionFailed as e:
                return f"API请求错误: {e.info}"
            except Exception as e:
                return f"未知错误: {e}"
            
            return "点赞成功"

        async def poke(self, user_id: int, group_id: int|None = None):
            """
            群聊戳一戳
            """
            if not isinstance(user_id, int):
                return "参数类型错误"
            if group_id is None:
                group_id = self.event.group_id
            if not isinstance(group_id, int):
                return "参数类型错误"
            try:
                await self.bot.call_api("group_poke", group_id=self.event.group_id, user_id=user_id)
            except ActionFailed as e:
                return f"API请求错误: {e.info}"
            except Exception as e:
                return f"未知错误: {e}"

            return "戳一戳成功"

        async def recall(self, message_id: int):
            """
            撤回消息
            """
            if not isinstance(message_id, int):
                return "参数类型错误"
            try:
                await self.bot.delete_msg(message_id=message_id)
            except ActionFailed as e:
                return f"API请求错误: {e.info}"
            except Exception as e:
                return f"未知错误: {e}"

            return "撤回成功"

        async def record_intention(self, text: str):
            """
            记录意图
            """
            if not isinstance(text, str):
                return "参数类型错误"
            try:
                await self._record_to_db(f"{self.bot_system_name} 记录意图: {text}")
            except Exception as e:
                return f"记录意图时发生错误: {e}"
            return None

        async def observe_tool_result(self, record_intention: str):
            """
            观察工具结果
            """
            return None

        async def _record_to_db(self, text: str):
            """
            记录消息到数据库
            """
            await group_message_manager.add_message(
                msg_id=self.bot_system_id,
                user_id=self.bot_system_id,
                user_name=self.bot_system_name,
                content=text,
                group_id=self.event.group_id,
                send_time=time.time()
            )


    def __init__(self, bot: Bot, event: GroupMessageEvent, text: str):
        self.bot = bot
        self.event = event
        self.text = text
        self.bot_system_id = BOT_SYSTEM_ID
        self.bot_system_name = BOT_SYSTEM_NAME
        self.observe_tool_result = False
        self.tools = self.Tools(self.bot, self.event, self.bot_system_id, self.bot_system_name)

    async def record_to_db(self, text: str):
        """
        记录消息到数据库
        """
        await group_message_manager.add_message(
            msg_id=self.bot_system_id,
            user_id=self.bot_system_id,
            user_name=self.bot_system_name,
            content=text,
            group_id=self.event.group_id,
            send_time=time.time()
        )

    async def parse_text(self, text: str):
        """
        解析json字符串为动作列表
        """
        return json.loads(text)

    async def execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """
        执行工具
        """
        if tool_name.startswith("_"):
            return f"非法的工具名: {tool_name}"
        if not (hasattr(self.tools, tool_name) and callable(getattr(self.tools, tool_name))):
            return f"工具 {tool_name} 不存在"
        try:
            result = await getattr(self.tools, tool_name)(**tool_args)
        except Exception as e:
            return f"执行工具 {tool_name} 时发生错误: {e}"
        return result

    async def execute_tools(self, tools: list[dict], timeout: float = 20.0) -> list[dict]:
        """
        并行执行动作列表，并为每个工具调用设置超时
        """
        async def execute_with_timeout(index: int, tool: dict) -> tuple[int, Any]:
            """为单个工具调用添加超时控制"""
            tool_name = tool.get("tool_name")
            tool_args = tool.get("parameters", {})
            
            # 参数类型检查
            if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
                return (index, "参数类型错误")
                
            try:
                # 为工具调用添加超时控制
                result = await asyncio.wait_for(
                    self.execute_tool(tool_name, tool_args), 
                    timeout=timeout
                )
                # record_intention 有特殊处理逻辑
                if tool_name == "record_intention" and result is None:
                    return (index, None)  # 标记为需要移除
                elif tool_name == "observe_tool_result":
                    self.observe_tool_result = True
                    observe_tool_result = True
                return (index, result)
            except asyncio.TimeoutError:
                return (index, "工具调用超时, 任务可能未完成")
            except Exception as e:
                return (index, f"执行工具 {tool_name} 时发生错误: {e}")

        # 为每个工具调用创建任务
        tasks = [
            execute_with_timeout(index, tool) 
            for index, tool in enumerate(tools)
        ]
        
        # 并行执行所有任务
        results = await asyncio.gather(*tasks)
        
        # 按原始顺序构建结果
        final_results = []
        for index, result in results:
            if result is None:  # record_intention 且 result 为 None 的情况
                tools[index]["return"] = None
                tools[index]["parameters"] = {"text": "已记录, 不再返回"}
            else:
                # 直接修改原始工具对象（保持与原代码行为一致）
                tools[index]["return"] = result
            final_results.append(tools[index])
        
        return final_results

    async def execute(self, timeout: float = 20.0) -> str:
        """
        执行动作列表
        """
        try:
            tools = await self.parse_text(self.text)
        except json.JSONDecodeError as e:
            await self.record_to_db(f"工具调用返回: 解析JSON时发生错误: {e}")
            return f"解析JSON时发生错误: {e}"
        
        # 执行工具调用，支持自定义超时
        results = await self.execute_tools(tools, timeout=timeout)
        
        results = json.dumps(results, ensure_ascii=False, indent=2)
        logger.info(f"工具调用结果: {results}")
        await self.record_to_db(f"工具调用返回: {results}")
        return results

    async def get_observation_state(self) -> bool:
        """
        获取观察状态
        """
        try:
            data = json.loads(self.text)
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON时发生错误: {e}")
            return False

        if not isinstance(data, list):
            logger.error("观察状态数据格式错误")
            return False

        for item in data:
            if not isinstance(item, dict):
                logger.error("观察状态数据格式错误")
                return False

            if "tool_name" not in item :
                logger.error("观察状态数据格式错误")
                return False

            if item["tool_name"] == "observe_tool_result":
                return True

        return False










# -------------------

# --- 辅助函数 ---
async def refresh_user_metadata(bot: Bot, group_id: int, user_ids: set[int]):
    """
    立即从 API 获取并更新指定用户的元数据缓存，不使用旧缓存。
    """
    logger.info(f"开始立即刷新 {len(user_ids)} 个用户的元数据。")
    
    tasks = []
    async def fetch_user_info(user_id):
        try:
            member_info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=True)
            stranger_info = await bot.get_stranger_info(user_id=user_id, no_cache=True)
            return user_id, member_info, stranger_info
        except ActionFailed as e:
            logger.warning(f"获取用户 {user_id} 在群 {group_id} 的信息失败: {e.info}")
            return user_id, None, None
        except Exception as e:
            logger.error(f"获取用户 {user_id} 信息时发生未知错误: {e}")
            return user_id, None, None

    for uid in user_ids:
        # 排除已知的系统ID
        if uid == int(bot.self_id) or uid == int(gcm.message_handling.QQ_system_message_id) or uid == int(BOT_SYSTEM_ID):
            continue
        tasks.append(fetch_user_info(uid))
    
    results = await asyncio.gather(*tasks)
    
    # 处理API结果并更新缓存
    async with user_cache_lock:
        for uid, member_info, stranger_info in results:
            if member_info and stranger_info:
                combined_info = {**member_info, **stranger_info}
            elif member_info:
                combined_info = member_info
            elif stranger_info:
                combined_info = stranger_info
            else:
                continue
                
            # 立即更新全局缓存，并添加新的时间戳
            user_metadata_cache[uid] = {
                'data': combined_info,
                '_timestamp': time.time()
            }
            logger.debug(f"已更新用户 {uid} 的元数据缓存。")
    
    logger.info("用户元数据缓存刷新完成。")

async def get_all_relevant_user_ids(messages: list[GroupMessage], bot: Bot):
    """
    获取消息历史中所有相关的用户ID（发言者和被@者）。
    """
    user_ids = {
        int(bot.self_id),
        int(gcm.message_handling.QQ_system_message_id),
        int(BOT_SYSTEM_ID)
    }
    # 收集发言用户
    for msg in messages:
        user_ids.add(msg.user_id)

    # 收集被@用户
    msg_content = "".join([m.content for m in messages])
    for c in parse_cq_codes(msg_content):
        if c.get("CQ") == "at" and "qq" in c:
            try:
                user_ids.add(int(c["qq"]))
            except (ValueError, TypeError):
                logger.warning(f"无法从CQ码 'at' 中解析有效的用户ID: {c}")
    return user_ids

async def get_batch_user_metadata(bot: Bot, group_id: int, user_ids: set[int]) -> dict[int, dict]:
    """
    批量获取用户在群聊中的元数据，包括群成员信息和陌生人信息，并使用缓存。
    """
    # 最终返回结果
    user_metadata_results = {}
    # 需要从API获取的用户ID列表
    ids_to_fetch = []
    
    current_time = time.time()
    
    # 1. 检查缓存
    for uid in user_ids:
        cached_data = user_metadata_cache.get(uid)
        
        # 排除已知的系统ID，这些ID不需要查询
        if uid == int(bot.self_id) or uid == int(gcm.message_handling.QQ_system_message_id) or uid == int(BOT_SYSTEM_ID):
            continue

        if cached_data and (current_time - cached_data.get('_timestamp', 0) < CACHE_TTL_SECONDS):
            # 缓存有效，直接使用
            user_metadata_results[uid] = cached_data['data']
        else:
            # 缓存过期或不存在，标记为需要获取
            ids_to_fetch.append(uid)

    if not ids_to_fetch:
        logger.debug("所有用户元数据均来自缓存，跳过API调用。")
        return user_metadata_results

    logger.debug(f"缓存未命中，需要从API获取 {len(ids_to_fetch)} 个用户的元数据。")

    # 2. 从API获取未缓存或已过期的用户数据
    tasks = []
    async def fetch_user_info(user_id):
        try:
            member_info = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=False)
            stranger_info = await bot.get_stranger_info(user_id=user_id, no_cache=False)
            return user_id, member_info, stranger_info
        except ActionFailed as e:
            logger.warning(f"获取用户 {user_id} 在群 {group_id} 的信息失败: {e.info}")
            return user_id, None, None
        except Exception as e:
            logger.error(f"获取用户 {user_id} 信息时发生未知错误: {e}")
            return user_id, None, None

    for uid in ids_to_fetch:
        tasks.append(fetch_user_info(uid))
    
    results = await asyncio.gather(*tasks)
    
    # 3. 处理API结果并更新缓存
    async with user_cache_lock:
        for uid, member_info, stranger_info in results:
            if member_info and stranger_info:
                combined_info = {**member_info, **stranger_info}
            elif member_info:
                combined_info = member_info
            elif stranger_info:
                combined_info = stranger_info
            else:
                continue
                
            user_metadata_results[uid] = combined_info
            # 更新全局缓存，并添加时间戳
            user_metadata_cache[uid] = {
                'data': combined_info,
                '_timestamp': time.time()
            }
    
    return user_metadata_results

async def from_messages_map_ids_to_names(bot: Bot, user_metadata_cache: dict[int, dict]):
    """将消息中的QQ号映射为用户名字典"""
    user_name_map = {
        gcm.message_handling.QQ_system_message_id: gcm.message_handling.QQ_system_name,
    }
    if gcm.message_handling.replace_my_name_with_me:
        user_name_map[int(bot.self_id)] = "我"
    
    # 遍历所有相关的用户ID，从元数据字典中获取用户名
    for user_id in user_metadata_cache:
        user_info = user_metadata_cache[user_id]
        if user_id not in user_name_map:
            # 优先使用群昵称 (card)，其次是昵称 (nickname)
            name = user_info.get("card") or user_info.get("nickname") or f"用户{user_id}"
            user_name_map[user_id] = name
            
    return user_name_map

async def get_group_user_metadata(user_info: dict, lang: str = "zh") -> dict:
    """
    从预获取的用户信息字典中提取并格式化用户信息
    """
    if not user_info:
        return {}

    if lang == "zh":
        role_map = {"member": "成员", "admin": "管理员", "owner": "群主"}
        role = user_info.get("role", "member")
        user_info["role"] = role_map.get(role, role)
        
        birthday = f"{user_info.get('birthday_year', '')}年{user_info.get('birthday_month', '')}月{user_info.get('birthday_day', '')}日"
        
        return {
            "QQ号": user_info.get("user_id"),
            "昵称": user_info.get("nickname"),
            "群昵称": user_info.get("card"),
            "角色": user_info.get("role"),
            "头衔": user_info.get("title"),
            "群等级": user_info.get("level"),
            "加群时间": datetime.fromtimestamp(user_info.get("join_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if user_info.get("join_time") else "未知",
            "最后发言时间": datetime.fromtimestamp(user_info.get("last_sent_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if user_info.get("last_sent_time") else "未知",
            "性别": user_info.get("sex"),
            "年龄": user_info.get("age"),
            "生日": birthday,
            "城市": user_info.get("city"),
            "国家": user_info.get("country"),
        }
    else:
        birthday = f"{user_info.get('birthday_year', '')}-{user_info.get('birthday_month', '')}-{user_info.get('birthday_day', '')}"
        
        return {
            "QQ Number": user_info.get("user_id"),
            "Nickname": user_info.get("nickname"),
            "Group Nickname": user_info.get("card"),
            "Role": user_info.get("role"),
            "Title": user_info.get("title"),
            "Group Level": user_info.get("level"),
            "Join Time": datetime.fromtimestamp(user_info.get("join_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if user_info.get("join_time") else "unknown",
            "Last Message Time": datetime.fromtimestamp(user_info.get("last_sent_time", 0)).strftime("%Y-%m-%d %H:%M:%S") if user_info.get("last_sent_time") else "unknown",
            "Gender": user_info.get("sex"),
            "Age": user_info.get("age"),
            "Birthday": birthday,
            "City": user_info.get("city"),
            "Country": user_info.get("country"),
        }

async def get_group_user_metadatas_from_history(bot: Bot, user_metadata_cache: dict[int, dict], remove_itself: bool = True, lang: str = "zh") -> str:
    """
    从预先获取的元数据字典中构建用户信息字符串。
    """
    user_datas = {}
    
    self_id = int(bot.self_id)
    sys_id = int(gcm.message_handling.QQ_system_message_id)
    bot_sys_id = int(BOT_SYSTEM_ID)
    
    for uid, data in user_metadata_cache.items():
        if (not remove_itself or uid != self_id) and uid != sys_id and uid != bot_sys_id:
            user_datas[uid] = await get_group_user_metadata(data, lang=lang)
    
    if not user_datas:
        return ""
    
    return "用户信息:\n" + json.dumps(user_datas, ensure_ascii=False, indent=2)

async def record_group_chat(msg_id, group_id, user_id, content, send_time):
    """记录群聊消息到数据库"""
    await group_chat_table.insert(
        msg_id=msg_id, 
        group_id=group_id, 
        user_id=user_id, 
        content=content, 
        send_time=send_time
    )

async def set_msg_emoji(bot: Bot, msg_id: int, emoji_id: int, set_like=True):
    """设置或移除表情回应"""
    action = "set_msg_emoji_like" if set_like else "unset_msg_emoji_like"
    await bot.call_api(action, message_id=msg_id, emoji_id=str(emoji_id))

async def group_poke(bot: Bot, group_id: int, user_id: int):
    """群聊戳一戳"""
    await bot.call_api("group_poke", group_id=group_id, user_id=user_id)

async def update_processing_status(bot: Bot, event: GroupMessageEvent, status: str):
    """更新处理状态的表情回应"""
    if not gcm.emoji_response.enable_emoji_response:
        return
    
    emoji_map = {
        "start": gcm.emoji_response.processing_emoji_id,
        "recognize_picture": gcm.image_recognition.processing_emoji_id,
        "success": gcm.emoji_response.processing_success_emoji_id,
        "rejected": gcm.emoji_response.processing_rejected_emoji_id
    }
    
    emoji_id = emoji_map.get(status, -1)
    if emoji_id == -1:
        return
    
    if status == "start" and gcm.emoji_response.processing_emoji_id != -1:
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "recognize_picture" and gcm.image_recognition.processing_emoji_id != -1:
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "success":
        if gcm.emoji_response.remove_processing_emoji_after_complete:
            await set_msg_emoji(bot, event.message_id, gcm.emoji_response.processing_emoji_id, False)
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "rejected":
        await set_msg_emoji(bot, event.message_id, emoji_id)

async def get_formatted_history_messages(records: list[GroupMessage], user_name_map: dict) -> list[get_formatted_history_messages_return]:
    """获取格式化的历史消息记录"""

    datas: list[get_formatted_history_messages_return] = []
    for i in records:
        # 从映射中获取用户名
        user_name = user_name_map.get(i.user_id, f"未知用户")
        
        if i.send_time:
            send_time = datetime.fromtimestamp(i.send_time).strftime("%m-%d %H:%M:%S")
        else:
            send_time = "unknown"
        data = get_formatted_history_messages_return(f"[{send_time}] {user_name}({i.user_id}, {i.msg_id}): {i.content}\n", i.user_id)
        datas.append(data)

    # 为at消息添加名称支持
    def _(cq):
        if cq["CQ"] == "at":
            user_id = int(cq["qq"])
            if user_id in user_name_map:
                return f"[CQ:at, qq={user_id}, name={user_name_map[user_id]}]"

    for i in range(len(datas)):
        datas[i].text = replace_cq_codes(datas[i].text, _)
    return datas



async def process_ai_response(ai_response: str, bot: Bot, event: GroupMessageEvent) -> str:
    """
    处理AI回复字符串，应用CQ码规则
    1. 将戳一戳CQ码替换为实际操作
    2. 保留其他CQ码不变
    """
    # 提取响应部分
    response = ai_response.split("[RESPONSE]")[1].strip() if "[RESPONSE]" in ai_response else ai_response
    
    # 匹配CQ码
    cq_pattern = r'\[CQ:(\w+)((?:,[^=\]]+=[^\]]*)*)\]'
    poke_tasks = []
    
    def replace_cq(match):
        nonlocal poke_tasks
        cq_type, params_str = match.group(1), match.group(2)
        params = dict(re.findall(r',([^=]+)=([^,\]]+)', params_str)) if params_str else {}
        
        # 处理戳一戳
        if cq_type == "poke" and "qq" in params:
            poke_tasks.append((event.group_id, params["qq"]))
            return ""
        
        return match.group(0)
    
    # 替换CQ码
    processed_response = re.sub(cq_pattern, replace_cq, response)
    
    # 执行戳一戳任务
    for group_id, user_id in poke_tasks:
        try:
            await group_poke(bot, group_id, user_id)
        except Exception as e:
            logger.error(f"执行戳一戳失败: {group_id}, {user_id}, {str(e)}")
    
    return processed_response

async def send_ai_response(bot: Bot, group_id: int, response_text: str):
    """发送AI响应并记录到数据库"""
    output_message = await toolbox.text_to_message(response_text)
    send_result = await chat.send(output_message)
    output_msg_id = send_result["message_id"]
    
    await record_group_chat(
        msg_id=output_msg_id,
        group_id=group_id,
        user_id=bot.self_id,
        content=response_text,
        send_time=time.time()
    )
    
    return output_msg_id

async def get_filtered_history(group_id: int, current_row: dict, bot: Bot):
    """获取并筛选历史消息"""
    history = await group_chat_table.get_nearby_rows_window(
        current_row["id"], 
        before=gcm.message_handling.max_chat_history_limit + gcm.message_handling.max_extra_ai_messages + gcm.message_handling.max_extra_qq_system_messages + 10,
        after=0,
        where="group_id = ?",
        where_params=(group_id,),
        order_direction="ASC"
    )
    
    # 筛选消息
    selected_messages = []
    user_msg_count = 0
    extra_ai_count = 0
    extra_qq_system_count = 0
    
    for msg in reversed(history):
        is_ai = str(msg["user_id"]) == bot.self_id
        is_qq_system = str(msg["user_id"]) == str(gcm.message_handling.QQ_system_message_id)
            
        # 普通用户消息或首次AI消息
        if not is_ai and not is_qq_system:
            if user_msg_count < gcm.message_handling.max_chat_history_limit:
                selected_messages.append(msg)
                user_msg_count += 1
            else:
                continue
        # AI消息（现在只要是AI消息就记录到额外AI消息中）
        elif is_ai and extra_ai_count < gcm.message_handling.max_extra_ai_messages:
            selected_messages.append(msg)
            extra_ai_count += 1
        # QQ系统消息
        elif is_qq_system and extra_qq_system_count < gcm.message_handling.max_extra_qq_system_messages:
            selected_messages.append(msg)
            extra_qq_system_count += 1
    
    return list(reversed(selected_messages))

async def get_image_description(bot: Bot, event: GroupMessageEvent, history_messages: list[GroupMessage]) -> str:
    """获取图片描述，并跟踪识别失败的图片，在达到阈值时跳过。"""
    # 筛选未被撤回的消息并合并其内容
    valid_messages_content = "".join([msg.content for msg in history_messages if not msg.is_recalled])
    if not valid_messages_content:
        return ""

    # 解析消息中的CQ码
    try:
        cq_list = parse_cq_codes(valid_messages_content)
    except Exception as e:
        logger.error(f"解析CQ码时发生错误: {e}")
        return ""

    # 筛选有效图片
    img_list = []
    for i in cq_list:
        if i["CQ"] == "image":
            file_size = int(i.get("file_size", i.get("size", 0)))
            if not file_size:
                continue
            max_size = gcm.image_recognition.max_image_size_mb * 1024 * 1024
            if file_size <= max_size and i["file"] not in img_list:
                img_list.append(i["file"])

    if not img_list:
        return ""

    # 获取图片路径和哈希值
    semaphore = asyncio.Semaphore(gcm.image_recognition.max_concurrent_tasks)
    async def process_image(file_name):
        async with semaphore:
            try:
                file_path = (await bot.get_image(file=file_name))["file"]
                file_hash = await file_to_sha256(file_path)
                return file_name, file_path, file_hash
            except ActionFailed as e:
                logger.error(f"获取图片失败 (ActionFailed): {file_name}, {e.info}")
                return None, None, None
            except Exception as e:
                logger.error(f"处理图片时发生未知错误: {file_name}, {e}")
                return None, None, None

    tasks = [process_image(img) for img in img_list]
    results = await asyncio.gather(*tasks)

    uncached_images = OrderedDict()
    description_dict = {}
    current_time = time.time()

    # 检查缓存和失败状态
    for file_name, file_path, file_hash in results:
        if not all((file_name, file_path, file_hash)):
            continue

        # --- FAILURE TRACKING LOGIC ---
        failure_data = image_failure_tracker[file_hash]

        # 如果距离上次失败已经超过重置时间窗口，则清空失败次数
        if current_time - failure_data['last_fail_time'] > gcm.image_recognition.failure_reset_window_seconds:
            if failure_data['count'] > 0:
                logger.debug(f"重置图片 {file_hash} 的失败次数。")
                failure_data['count'] = 0

        # 如果失败次数达到上限，则跳过此图片
        if failure_data['count'] >= gcm.image_recognition.max_recognition_failures:
            logger.warning(f"图片 {file_name} (hash: {file_hash}) 已达到最大失败次数 ({failure_data['count']})，本次将跳过。")
            description_dict[file_name] = "图片识别失败次数过多，已跳过"
            continue
        # --- END OF FAILURE TRACKING LOGIC ---

        # 查询缓存
        try:
            row = await image_description_cache_table.query(where="hash = ?", params=(file_hash,))
            if row:
                description_dict[file_name] = row[0]["description"]
                logger.debug(f"使用缓存: {file_name}")
            else:
                uncached_images[file_name] = (file_path, file_hash)
        except Exception as e:
            logger.error(f"查询图片描述缓存时出错: {e}")

    if uncached_images:
        await update_processing_status(bot, event, "recognize_picture")

    # 限制API调用数量
    max_api_calls = gcm.image_recognition.max_api_calls_per_recognition
    if len(uncached_images) > max_api_calls:
        keys = list(uncached_images.keys())[-max_api_calls:]
        uncached_images = OrderedDict((k, uncached_images[k]) for k in keys)

    # 并发处理API调用
    api_semaphore = asyncio.Semaphore(gcm.image_recognition.max_concurrent_tasks)
    async def call_image_api(file_name, file_path, file_hash):
        async with api_semaphore:
            ai_instance = ChatGPT(
                api_key=gcm.api.image_ai_key, model=gcm.api.image_ai_model,
                base_url=gcm.api.image_ai_url, is_reasoning_model=gcm.image_recognition.is_reasoning_model
            )
            try:
                await ai_instance.add_local_file(file_path, gcm.prompt.image_recognition_prompt)
                response = await ai_instance.get_response(no_input=True, add_to_context=False)
                return file_name, response, file_hash
            except Exception as e:
                logger.error(f"调用图像识别API失败: {file_name}, 错误: {e}")
                return file_name, None, file_hash
            finally:
                del ai_instance

    api_tasks = [call_image_api(file_name, *data) for file_name, data in uncached_images.items()]
    api_results = await asyncio.gather(*api_tasks)

    # 处理API结果并更新缓存和失败计数器
    for file_name, description, file_hash in api_results:
        if description is not None:
            description_dict[file_name] = description
            await image_description_cache_table.insert(hash=file_hash, description=description)
        else:
            failure_data = image_failure_tracker[file_hash]
            failure_data['count'] += 1
            failure_data['last_fail_time'] = time.time()
            logger.warning(f"图片识别失败: {file_name} (hash: {file_hash})。当前失败次数: {failure_data['count']}。")
            description_dict[file_name] = "图片描述获取失败"
  
    # 按原始顺序生成描述
    return_str = ""
    for img in img_list:
        if img in description_dict:
            return_str += f"{img}: {description_dict[img]}\n"

    return return_str

async def get_group_metadata(bot: Bot, group_id: int, no_cache = True, lang: str = "zh") -> dict:
    """
    获取群组信息
    """
    g = await bot.get_group_info(group_id=group_id, no_cache=no_cache)
    if lang == "zh":
        return {
            "群号": g["group_id"],
            "群名": g["group_name"],
            "群创建时间": datetime.fromtimestamp(g["group_create_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "群人数": g["member_count"],
        }
    else:
        return {
            "Group ID": g["group_id"],
            "Group Name": g["group_name"],
            "Group Creation Time": datetime.fromtimestamp(g["group_create_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Group Members": g["member_count"],
        }



async def retrieve_message(history_context: list[str], group_id: int, selected_messages: list[GroupMessage], limit: int = 3) -> str:
    """
    从历史消息中检索与当前上下文相关的消息。

    该函数使用RAG（检索增强生成）管理器搜索与最新历史消息相关的消息，
    并对检索结果进行处理，包括去重、格式化和数量限制。

    Args:
        history_context (list[str]): 历史对话上下文列表，包含之前的对话内容
        group_id (int): 群组ID，用于限定检索范围
        selected_messages (list[GroupMessage]): 已选中的消息列表，用于去重判断
        limit (int): 返回消息的最大数量，默认为3

    Returns:
        str: 格式化后的检索结果消息，包含最多3条不重复的相关历史消息。
             如果没有找到相关消息，返回空字符串。

    Note:
        - 检索结果会排除与selected_messages中消息ID相关的内容
        - 最多返回3条相关消息
        - 返回的消息格式为Markdown格式，包含消息段编号和内容
        - 处理过程中会进行JSON解析和类型转换的健壮性检查
    """
    if limit > 10:
        limit = 10
    logger.debug("开始检索信息")
    retrieval_result = await rag_manager.search_messages(
        history_context[-1],
        group_id=group_id,
        limit=limit * 3
    )
    logger.debug(f"检索结果: {retrieval_result}")
    retrieval_message = ""
    
    if retrieval_result:
        # 在循环外创建上下文消息ID集合（全部转为字符串），用于快速、准确地查找
        context_msg_ids = {str(msg.msg_id) for msg in selected_messages}
        
        count = 0
        for i in retrieval_result:
            if count >= limit:
                break

            if not i.document or not i.metadata or "related_msg_id" not in i.metadata:
                logger.debug("检索结果无效或缺少元数据，忽略")
                continue

            related_ids_str = i.metadata["related_msg_id"]
            try:
                if not isinstance(related_ids_str, str):
                    related_ids_str = str(related_ids_str)
                related_ids = json.loads(related_ids_str)
                # 健壮性检查：如果解析出来的不是列表（例如，JSON字符串是 "123" 或 "\"abc\""）
                # 也统一包装成列表处理
                if not isinstance(related_ids, list):
                    related_ids = [related_ids]
            except json.JSONDecodeError:
                logger.warning(f"无法解析 related_msg_id JSON 字符串: '{related_ids_str}'，忽略此条检索结果。")
                continue

            # 将解析出的ID全部转为字符串，创建集合用于比较
            related_id_set = {str(rid) for rid in related_ids}

            # 使用集合的 isdisjoint() 方法高效判断是否有交集（即是否重复）
            if not context_msg_ids.isdisjoint(related_id_set):
                logger.debug(f"检索结果 {related_id_set} 与当前上下文 {context_msg_ids} 重复，忽略")
                continue
            
            # 开始构建消息
            count += 1
            if count == 1: 
                retrieval_message = "可能相关的历史消息:\n"

            retrieval_message += f"## 消息段 {count}:\n"
            retrieval_message += f"{i.document}\n"
    return retrieval_message


# --------------------

# --- 主消息处理器 ---
chat = on_message(rule=to_me(), priority=100, block=False)




@chat.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent):
    global chat_lock
    
    sender_id = None
    try:
        sender_id = event.user_id
    except Exception:
        sender_id = None

    session_id, reason = await session_manager.acquire(group_id=event.group_id, user_id=sender_id, is_private=False)

    if session_id is None:
        logger.warning(f"拒绝处理消息 {event.message_id}，原因: {reason}")
        await update_processing_status(bot, event, "rejected")
        return

    logger.debug(f"为消息 {event.message_id} 分配 session_id={session_id}")
    await update_processing_status(bot, event, "start")
    
    start_total_time = time.time()

    if "#" in event.message:
        await chat.send("忽略以#开头的消息")
        return
    
    chat_lock = True
    logger.debug(f"开始处理消息: {event.message_id}")
    await update_processing_status(bot, event, "start")
    
    try:
        start_history_retrieval = time.time()
        logger.debug("阶段1: 开始数据库与历史消息检索")
        
        start_db_query = time.time()
        logger.debug(f"查询数据库: group_id={event.group_id}, msg_id={event.message_id}")
        msg = await group_message_manager.get_message_by_msg_id(event.message_id)
        end_db_query = time.time()
        logger.debug(f"查询结果: {msg}，耗时: {end_db_query - start_db_query:.4f} 秒")
        
        if not msg:
            logger.error("无法从数据库获取消息内容")
            return
        
        group_id = event.group_id
        
        start_message_filter = time.time()
        logger.debug("开始筛选历史消息")
        selected_messages: list[GroupMessage] = await group_message_manager.get_nearby_messages(msg, group_id=group_id, before=gcm.message_handling.max_chat_history_limit)
        end_message_filter = time.time()
        logger.debug(f"筛选到 {len(selected_messages)} 条历史消息，耗时: {end_message_filter - start_message_filter:.4f} 秒")

        start_user_metadata = time.time()
        # 批量获取所有相关用户元数据
        all_relevant_user_ids = await get_all_relevant_user_ids(selected_messages, bot)
        user_metadata_cache = await get_batch_user_metadata(bot, group_id, all_relevant_user_ids)
        end_user_metadata = time.time()
        logger.debug(f"批量获取用户元数据完成，耗时: {end_user_metadata - start_user_metadata:.4f} 秒")

        start_name_mapping = time.time()
        user_name_map = await from_messages_map_ids_to_names( bot,  user_metadata_cache)
        end_name_mapping = time.time()
        logger.debug(f"用户ID映射到名称完成，耗时: {end_name_mapping - start_name_mapping:.4f} 秒")

        start_history_formatting = time.time()
        # 构建历史上下文
        history_data: list[get_formatted_history_messages_return] = await get_formatted_history_messages(
            selected_messages, user_name_map
        )
        history_context = [data.text for data in history_data]
        end_history_formatting = time.time()
        logger.debug(f"历史消息格式化完成，耗时: {end_history_formatting - start_history_formatting:.4f} 秒")
        
        end_history_retrieval = time.time()
        logger.debug(f"阶段1: 数据库与历史消息检索总耗时: {end_history_retrieval - start_history_retrieval:.4f} 秒")


        # 获取其它可并发信息
        start_concurrent_tasks = time.time()
        logger.debug("阶段2: 开始并发获取附加信息")

        tasks = {}
        images_description = ""
        group_metadata = ""
        user_metadata = ""
        retrieved_message = ""
        
        if gcm.image_recognition.enable:
            tasks["get_image_description"] = asyncio.create_task(get_image_description(bot, event, selected_messages))
        if gcm.message_handling.inject_group_metadata:
            tasks["get_group_metadata"] = asyncio.create_task(get_group_metadata(bot, group_id))
        if gcm.message_handling.inject_user_metadata:
            # 传递预获取的元数据字典
            tasks["get_group_user_metadatas_from_history"] = asyncio.create_task(get_group_user_metadatas_from_history(bot, user_metadata_cache))
        if rag_loading_complete:
            tasks["retrieve_message"] = asyncio.create_task(retrieve_message(history_context, group_id, selected_messages))

        # 等待所有并发任务完成
        for task_name, task in tasks.items():
            try:
                result = await task
                if task_name == "get_image_description":
                    images_description = result
                elif task_name == "get_group_metadata":
                    group_metadata = result
                    group_metadata = f"群聊信息:\n{group_metadata}"
                elif task_name == "get_group_user_metadatas_from_history":
                    user_metadata = result
                    user_metadata = f"你自己的QQ号为:{bot.self_id}\n" + user_metadata
                elif task_name == "retrieve_message":
                    retrieved_message = result
            except Exception as e:
                logger.error(f"并发任务 {task_name} 失败: {e}")
                await update_processing_status(bot, event, "error")
                chat_lock = False
                return

        end_concurrent_tasks = time.time()
        logger.debug(f"阶段2: 并发获取附加信息完成，耗时: {end_concurrent_tasks - start_concurrent_tasks:.4f} 秒")
        
        end_total_time = time.time()
        logger.debug(f"总信息获取耗时 (在调用AI之前): {end_total_time - start_total_time:.4f} 秒")
        

        main_chat_ai = ChatGPT(
            api_key=gcm.api.chat_ai_key,
            model=gcm.api.chat_ai_model,
            base_url=gcm.api.chat_ai_url,
            is_reasoning_model=gcm.api.chat_model_is_reasoning
        )
        # 与AI交互
        logger.debug("添加系统提示到AI上下文")
        await main_chat_ai.add_dialogue(gcm.prompt.chat_full_prompt, "system")
        if group_metadata:
            logger.debug("添加群聊元数据到AI上下文")
            await main_chat_ai.add_dialogue(group_metadata, "system")
        if user_metadata:
            logger.debug("添加用户元数据到AI上下文")
            await main_chat_ai.add_dialogue(user_metadata, "system")
        if images_description:
            logger.debug("添加图片描述到AI上下文")
            images_description = "历史消息中部分图片的描述:\n" + images_description
            await main_chat_ai.add_dialogue(images_description, "system")
        if retrieved_message:
            logger.debug("添加检索消息到AI上下文")
            await main_chat_ai.add_dialogue(retrieved_message, "system")
        logger.debug("添加历史记录到AI上下文")
        for j in history_data:
            await main_chat_ai.add_dialogue(j.text, "user" if j.user_id != bot.self_id else "assistant")
        logger.debug(f"系统提示词: {gcm.prompt.chat_full_prompt}")
        logger.debug(f"图片描述: {images_description}")
        
        ai_memory_context = "\n".join([i.text_content for i in (await main_chat_ai.get_context())])
        logger.debug(f"AI记忆: {ai_memory_context}")
        number_of_tool_calls = 0
        while True:
            tool_task = None
            need_observe_tool_result: bool = False
            if gcm.message_handling.enable_streaming:
                full_response = ""
                full_think_text = ""
                response_started = False
                line_buffer = ""
                last_send_time = 0  
                use_tool = False
                logger.debug("开始流式接收AI响应")
                async for chunk in main_chat_ai.stream_response(no_input=True):
                    if chunk.is_reasoning:
                        full_think_text += chunk.reasoning_text
                        logger.debug(f"处理推理内容: {chunk.reasoning_text}")
                    if not chunk.is_main_text:
                        continue
                    
                    full_response += chunk.main_text
                    line_buffer += chunk.main_text

                    if not response_started:
                        if "[ACTION]" in line_buffer:
                            if not use_tool:
                                logger.debug("检测到工具调用响应")
                            use_tool = True
                            
                        if "[RESPONSE]" in line_buffer:
                            response_started = True
                            if use_tool:
                                number_of_tool_calls += 1
                                tool_context = line_buffer.split("[ACTION]")[1].split("[RESPONSE]")[0].lstrip()
                                logger.debug(f"工具调用: {tool_context}")
                                tool_manager = ToolCallManager(bot, event, tool_context)
                                tool_task = asyncio.create_task(tool_manager.execute())
                                need_observe_tool_result = await tool_manager.get_observation_state()
                            line_buffer = line_buffer.split("[RESPONSE]")[1].lstrip()
                            logger.debug("检测到输出响应开始标记")
                        continue

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        if line.strip():
                            logger.debug(f"处理单行响应: {line}")
                            processed_line = await process_ai_response(line, bot, event)
                            logger.debug(f"处理后的响应: {processed_line}")
                            if processed_line.strip():
                                current_time = time.time()
                                elapsed = current_time - last_send_time
                                if elapsed < gcm.message_handling.min_send_interval:
                                    random_time = random.uniform(0, gcm.message_handling.send_interval_random_range)
                                    wait_time = gcm.message_handling.min_send_interval + random_time - elapsed
                                    logger.debug(f"等待 {wait_time:.2f} 秒以满足最小发送间隔")
                                    await asyncio.sleep(wait_time)
                                
                                logger.debug("发送单行响应")
                                await send_ai_response(bot, group_id, processed_line)
                                last_send_time = time.time()
                
                if response_started and line_buffer.strip():
                    full_response += line_buffer
                    logger.debug(f"处理剩余响应: {line_buffer}")
                    processed_response = await process_ai_response(line_buffer, bot, event)
                    logger.debug(f"处理后的剩余响应: {processed_response}")
                    if processed_response.strip():
                        current_time = time.time()
                        elapsed = current_time - last_send_time
                        if elapsed < gcm.message_handling.min_send_interval:
                            wait_time = gcm.message_handling.min_send_interval - elapsed
                            logger.debug(f"等待 {wait_time:.2f} 秒以满足最小发送间隔")
                            await asyncio.sleep(wait_time)
                        
                        logger.debug("发送剩余响应")
                        await send_ai_response(bot, group_id, processed_response)
                if full_think_text:
                    logger.info(f"AI思考内容: {full_think_text}")
                logger.info(f"AI输出内容: {full_response}")

            else:
                logger.debug("开始非流式接收AI响应")
                response = await main_chat_ai.get_response(no_input=True, return_reasoning=True)
                think_text = ""
                if isinstance(response, tuple):
                    response, think_text = response
                if think_text:
                    logger.info(f"AI思考内容: {think_text}")
                logger.debug(f"AI输出内容: {response}")
                if not isinstance(response, str):
                    logger.error("AI响应不是字符串类型")
                    raise ValueError("AI响应不是字符串类型")
                if "[RESPONSE]" not in response:
                    logger.error("AI响应中未找到响应内容")
                    raise ValueError("AI响应中未找到响应内容")
                output_text = response.split("[RESPONSE]")[1].lstrip()
                tool_context = ""
                if "[ACTION]" in response:
                    tool_context = response.split("[ACTION]")[1].split("[RESPONSE]")[0].lstrip()
                    if tool_context:
                        number_of_tool_calls += 1
                        tool_manager = ToolCallManager(bot, event, tool_context)
                        tool_task = asyncio.create_task(tool_manager.execute())
                        need_observe_tool_result = await tool_manager.get_observation_state()
                if tool_context:
                    logger.debug(f"工具调用: {tool_context}")
                    
                processed_response = await process_ai_response(output_text, bot, event)
                logger.debug(f"处理后的响应: {processed_response}")
                await send_ai_response(bot, group_id, processed_response)

            if need_observe_tool_result:
                if number_of_tool_calls > SINGLE_SESSION_CONTINUOUS_CALL_TO_OBSERVE_TOOL_RESULT_LIMIT:
                    logger.warning("达到连续调用工具的次数限制，强制退出")
                    raise ValueError("达到连续调用工具的次数限制，强制退出")
                if not isinstance(tool_task, asyncio.Task):
                    logger.error("tool_task 不是 asyncio.Task 类型")
                    break
                try:
                    tool_result = await tool_task
                except Exception as e:
                    raise ValueError(f"工具调用失败: {e}")
                text = f"[TOOL_RESULT]\n{tool_result}"
                await main_chat_ai.add_dialogue(text, "user")
            else:
                break
        await update_processing_status(bot, event, "success")    
    except Exception as e:
        logger.error(f"处理消息时出错: {type(e).__name__}: {str(e)}")
        logger.exception("详细错误信息")
        await chat.send(f"Σ(°△°|||)︴出错了喵！: {type(e).__name__.replace(gcm.api.chat_ai_url, "http://xxxxxxx")}: {str(e).replace(gcm.api.chat_ai_url, "http://xxxxxxx")}")
    finally:
        try:
            await session_manager.release(session_id)
        except Exception as e:
            logger.error(f"释放 session {session_id} 失败: {e}")
        logger.debug(f"已释放 session {session_id}")
        if "main_chat_ai" in locals():
            del main_chat_ai