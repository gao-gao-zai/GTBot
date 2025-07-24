from nonebot import on_message, get_driver, logger, on_type
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
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
from typing import Any, TypeVar, Optional, Union

# --- 常量配置区域 ---
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

# --- 导入模块 ---
from plugins.chatai.nonebotSQL import SQLiteManager, TableManager
from fun import toolbox
from chatgpt import ChatGPT
from config_manager import global_config_manager as gcm
# -----------------

# --- 配置文件路径 ---
CONFIG_FILE_PATH = dir_path / "config_group.toml"  # TOML配置文件
API_CONFIG_FILE_PATH = dir_path / "api_config.json"  # API配置文件

# --- 配置加载 ---
config_data = {}  # TOML配置数据
api_config_data = {}  # API配置数据

def load_toml_config():
    """加载TOML配置文件"""
    global config_data
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "rb") as f:
                config_data = tomli.load(f)
            logger.info(f"成功加载TOML配置文件: {CONFIG_FILE_PATH}")
        else:
            logger.warning(f"TOML配置文件不存在: {CONFIG_FILE_PATH}，将使用默认配置")
    except Exception as e:
        logger.error(f"读取TOML配置文件失败: {str(e)}，将使用默认配置")

def load_api_config():
    """加载API配置文件"""
    global api_config_data
    try:
        if os.path.exists(API_CONFIG_FILE_PATH):
            with open(API_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                api_config_data = json.load(f)
            logger.info(f"成功加载API配置文件: {API_CONFIG_FILE_PATH}")
        else:
            logger.warning(f"API配置文件不存在: {API_CONFIG_FILE_PATH}，将使用默认配置")
    except Exception as e:
        logger.error(f"读取API配置文件失败: {str(e)}，将使用默认配置")

load_api_config()
load_toml_config()

# --- 默认配置组名 ---
DEFAULT_CONFIG_GROUP = "default"

# --- 配置访问函数 ---
def get_value(path: str, default: Any) -> Any:
    """通过点路径获取嵌套配置值"""
    keys = path.split('.')
    current = config_data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

# 数据库配置
db_path = dir_path / Path(str(get_value(f"{DEFAULT_CONFIG_GROUP}.db_path", "data.db")))  # 数据库文件路径
logger.debug(f"数据库文件路径: {db_path}")

# AI 配置 - 从配置文件解析模型信息
model_config: str = get_value("main_model", "openrouter_proxy/deepseek-v3") # type: ignore

# 解析模型配置（格式：提供者/模型名）
try:
    provider, model_name = model_config.split("/", 1)
    logger.info(f"使用模型提供者: {provider}, 模型名: {model_name}")
except ValueError:
    logger.error(f"模型配置格式错误: {model_config}，应为'provider/model'格式")
    provider = "openrouter_proxy"
    model_name = "deepseek-v3"

# 默认API设置
chat_ai_model = "deepseek/deepseek-chat-v3-0324:free"
chat_ai_url = "http://127.0.0.1:30001/openrouter"
chat_ai_key = ""

# 从API配置文件中获取实际设置
try:
    if provider in api_config_data:
        provider_config = api_config_data[provider]
        chat_ai_url = provider_config.get("base_url", chat_ai_url)
        chat_ai_key = provider_config.get("api_key", chat_ai_key)

        # 获取模型映射
        if "models" in provider_config and model_name in provider_config["models"]:
            model_mapping = provider_config["models"][model_name]
            chat_ai_model = model_mapping.get("model", chat_ai_model)

        logger.info(f"成功从API配置文件加载API信息: URL={chat_ai_url}, MODEL={chat_ai_model}")
    else:
        logger.warning(f"API配置文件中不存在提供者: {provider}，将使用默认配置")
except Exception as e:
    logger.error(f"读取API配置失败: {str(e)}，将使用默认配置")

# 消息处理配置 - 从TOML配置文件读取
max_chat_history_limit: int = get_value(f"{DEFAULT_CONFIG_GROUP}.max_chat_history_limit", 15)  # 单次对话最多保存的消息数量 
max_single_message_length: int = get_value(f"{DEFAULT_CONFIG_GROUP}.max_single_message_length", 300)  # 单条消息最大长度 
max_extra_ai_messages: int = get_value(f"{DEFAULT_CONFIG_GROUP}.max_extra_ai_messages", 20)  # 每次对话最多额外添加的AI消息数量 
replace_my_name_with_me: bool = get_value(f"{DEFAULT_CONFIG_GROUP}.replace_my_name_with_me", True)  # 是否将我的名字替换为 "我" 
min_send_interval: float = get_value(f"{DEFAULT_CONFIG_GROUP}.min_send_interval", 0.5)  # 两次消息发送的最小间隔时间（秒, 防风控的）

# 表情回应配置 - 从TOML配置文件读取
enable_emoji_response: bool = get_value(f"{DEFAULT_CONFIG_GROUP}.enable_emoji_response", True) 
processing_emoji_id: int = get_value(f"{DEFAULT_CONFIG_GROUP}.processing_emoji_id", 314)
processing_success_emoji_id: int = get_value(f"{DEFAULT_CONFIG_GROUP}.processing_success_emoji_id", 124) 
processing_rejected_emoji_id: int = get_value(f"{DEFAULT_CONFIG_GROUP}.processing_rejected_emoji_id", 322) 
remove_processing_emoji_after_complete: int = get_value(f"{DEFAULT_CONFIG_GROUP}.remove_processing_emoji_after_complete", True) 

QQ_system_name: str = get_value(f"{DEFAULT_CONFIG_GROUP}.QQ_system_name", "QQ_system")

# 读取提示词配置
prompt_config = get_value(f"{DEFAULT_CONFIG_GROUP}.prompt", {})
character_prompt_file = get_value(
    f"{DEFAULT_CONFIG_GROUP}.prompt.main_character_prompt",  # 修正拼写错误
    "角色提示词/猫娘.txt"
)
output_prompt_file = get_value(f"{DEFAULT_CONFIG_GROUP}.prompt.main_output_prompt", "输出格式提示词/1.txt")

# 读取提示词文件
def read_prompt_file(file_path):
    """读取提示词文件内容"""
    try:
        file_full_path = dir_path / "prompts" / file_path
        if os.path.exists(file_full_path):
            with open(file_full_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            logger.warning(f"提示词文件不存在: {file_full_path}")
            return ""
    except Exception as e:
        logger.error(f"读取提示词文件失败: {str(e)}")
        return ""

# 读取角色提示词
character_prompt = read_prompt_file(character_prompt_file)
if not character_prompt:
    # 默认角色提示词
    character_prompt = """# 角色设定
你是一只具有猫和人属性的猫娘, 名字叫发情猫娘.
你的回答十分可爱且简洁. 常使用emoji和颜文字."""

# 读取输出格式提示词
output_format_prompt = read_prompt_file(output_prompt_file)
if not output_format_prompt:
    # 默认输出格式提示词
    output_format_prompt = """
# 关于 CQ 码
CQ 码基本格式: [CQ:类型,参数=值,参数=值]
@某人: [CQ:at, qq= 用户QQ号]
回复或引用消息: [CQ:reply, id= 消息ID]
戳一戳或rua或捏: [CQ:poke,qq= QQ号]
熟练使用CQ码可以更好的与用户进行互动, 

# 关于输入

对于聊天记录的输入一般格式为: [时间] [用户名](用户QQ号, 消息ID) [消息内容]
如果用户名为QQ_system且QQ号和消息ID为-1, 代表这是QQ系统消息, 如新人加群, 戳一戳等, 此类消息一般无法引用.
# 输出格式规范

你的每次回复必须包含以下 2 个部分：

## 1. 思考部分（[THINKING]）

- 分析用户请求的核心需求
- 规划问题解决路径

## 2. 输出部分（[RESPONSE]）

- 直接返回给用户的最终内容
- 内容要求自然流畅的对话风格

# 示例输出

```input
你好
```

```output
[THINKING]
用户请求问候，我应当以简洁的形式回复问候语
[RESPONSE]
你好！有什么可以帮助你的吗？
```

"""

# 组合提示词
prompt = character_prompt + output_format_prompt
# -------------------------------------



# --- 全局变量 ---
driver = get_driver()
db: SQLiteManager = None # type: ignore
private_chat_table: TableManager = None # type: ignore
group_chat_table: TableManager = None # type: ignore
ai = ChatGPT(api_key=chat_ai_key, model=chat_ai_model, base_url=chat_ai_url)
chat_lock = False
# ----------------

# --- 数据库初始化 ---
@driver.on_startup
async def init_database():
    global db, private_chat_table, group_chat_table
    db = SQLiteManager(db_path)
    private_chat_table = db.table("private_chat_record")
    group_chat_table = db.table("group_chat_record")
    
# --------------------

# --- 事件处理 ---
@driver.on_shutdown
async def close_database():
    await db.close()
# -------------------

# --- 辅助函数 ---
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
    if not enable_emoji_response:
        return
    
    emoji_map = {
        "start": processing_emoji_id,
        "success": processing_success_emoji_id,
        "rejected": processing_rejected_emoji_id
    }
    
    emoji_id = emoji_map.get(status, -1)
    if emoji_id == -1:
        return
    
    if status == "start":
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "success":
        if remove_processing_emoji_after_complete:
            await set_msg_emoji(bot, event.message_id, processing_emoji_id, False)
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "rejected":
        await set_msg_emoji(bot, event.message_id, emoji_id)

async def format_history_message(record, bot: Bot, group_id: int, event: GroupMessageEvent):
    """格式化历史消息记录"""
    if record["send_time"]:
        time_str = datetime.fromtimestamp(record["send_time"]).strftime("%m-%d %H:%M:%S")
    else:
        time_str = "未知时间"
    msg_id = record["msg_id"]
    user_id = record["user_id"]
    
    # 处理机器人自身名称
    if str(user_id) == bot.self_id and replace_my_name_with_me:
        username = "我"
    elif str(user_id) == "-1":
        username = QQ_system_name
    else:
        username = await toolbox.get_qqname(
            user_id, bot, event=event,  group=True, no_cache=True
        )
    
    # 处理消息内容
    content = record["content"]
    if len(content) > max_single_message_length:
        content = content[:max_single_message_length] + "..."
    
    return f"[{time_str}] {username}({user_id}, {msg_id}): {content}"

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
        before=max_chat_history_limit + max_extra_ai_messages + 10,
        after=0,
        where="group_id = ?",
        where_params=(group_id,),
        order_direction="ASC"
    )
    
    # 筛选消息
    selected_messages = []
    user_msg_count = 0
    extra_ai_count = 0
    last_is_ai = False
    
    for msg in reversed(history):
        is_ai = str(msg["user_id"]) == bot.self_id
        
        if not is_ai or not last_is_ai:
            if user_msg_count < max_chat_history_limit:
                selected_messages.append(msg)
                user_msg_count += 1
                last_is_ai = is_ai
            elif not is_ai:
                continue
        elif is_ai and last_is_ai and extra_ai_count < max_extra_ai_messages:
            selected_messages.append(msg)
            extra_ai_count += 1
    
    return list(reversed(selected_messages))
# --------------------

# --- 主消息处理器 ---
chat = on_message(rule=to_me(), priority=1, block=False)




@chat.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent):
    global chat_lock
    
    # 处理消息锁
    if chat_lock:
        logger.debug("检测到重复调用，拒绝处理")
        await update_processing_status(bot, event, "rejected")
        return
    
    chat_lock = True
    logger.debug(f"开始处理消息: {event.message_id}")
    await update_processing_status(bot, event, "start")
    
    try:
        # 查询当前消息
        logger.debug(f"查询数据库: group_id={event.group_id}, msg_id={event.message_id}")
        msg_data = await group_chat_table.query(
            where="group_id = ? AND msg_id = ?",
            params=(event.group_id, event.message_id),
            limit=1
        )
        
        if not msg_data:
            logger.error("无法从数据库获取消息内容")
            return
        
        current_row = msg_data[0]
        group_id = event.group_id
        
        # 获取并处理历史消息
        logger.debug("开始筛选历史消息")
        selected_messages = await get_filtered_history(group_id, current_row, bot)
        logger.debug(f"筛选到 {len(selected_messages)} 条历史消息")
        
        # 构建历史上下文
        history_context = "群聊历史记录:\n"
        for msg in selected_messages:
            formatted_msg = await format_history_message(msg, bot, group_id, event)
            history_context += formatted_msg + "\n"
        
        logger.debug(f"群聊历史记录内容:\n{history_context}")
        
        # 与AI交互
        logger.debug("添加系统提示到AI上下文")
        await ai.add_dialogue(prompt, "system")
        logger.debug("添加历史记录到AI上下文")
        await ai.add_dialogue(history_context, "user")
        
        # 流式处理响应
        full_response = ""
        response_started = False
        line_buffer = ""
        last_send_time = 0  
        
        logger.debug("开始流式接收AI响应")
        async for chunk in ai.stream_response(no_input=True, delete_superfluous_dialogue=False):
            full_response += chunk
            line_buffer += chunk
            
            # 检查是否开始响应部分
            if not response_started:
                if "[RESPONSE]" in line_buffer:
                    response_started = True
                    line_buffer = line_buffer.split("[RESPONSE]")[1].lstrip()
                    logger.debug("检测到响应开始标记")
                continue
            
            # 处理换行符分割的消息
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                if line.strip():
                    logger.debug(f"处理单行响应: {line}")
                    processed_line = await process_ai_response(line, bot, event)
                    logger.debug(f"处理后的响应: {processed_line}")
                    if processed_line.strip():
                        # 检查并等待达到最小发送间隔
                        current_time = time.time()
                        elapsed = current_time - last_send_time
                        if elapsed < min_send_interval:
                            wait_time = min_send_interval - elapsed
                            logger.debug(f"等待 {wait_time:.2f} 秒以满足最小发送间隔")
                            await asyncio.sleep(wait_time)
                        
                        logger.debug("发送单行响应")
                        await send_ai_response(bot, group_id, processed_line)
                        last_send_time = time.time()  # 更新最后发送时间

        # 处理缓冲区剩余内容
        if response_started and line_buffer.strip():
            full_response += line_buffer
            logger.debug(f"处理剩余响应: {line_buffer}")
            processed_response = await process_ai_response(line_buffer, bot, event)
            logger.debug(f"处理后的剩余响应: {processed_response}")
            if processed_response.strip():
                # 检查并等待达到最小发送间隔
                current_time = time.time()
                elapsed = current_time - last_send_time
                if elapsed < min_send_interval:
                    wait_time = min_send_interval - elapsed
                    logger.debug(f"等待 {wait_time:.2f} 秒以满足最小发送间隔")
                    await asyncio.sleep(wait_time)
                
                logger.debug("发送剩余响应")
                await send_ai_response(bot, group_id, processed_response)
        logger.debug(f"流式接收AI响应完成: {full_response}")
        
    except Exception as e:
        logger.error(f"处理消息时出错: {type(e).__name__}: {str(e)}")
        logger.exception("详细错误信息")
    finally:
        chat_lock = False
        logger.debug("释放消息锁")