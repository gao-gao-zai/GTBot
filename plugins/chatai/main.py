import random
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
from collections import OrderedDict


# 要添加对撤回消息的适配，在数据库中添加一列标记是否撤回

# --- 常量配置区域 ---
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))




# --- 导入模块 ---
from SQLiteManager import chat_record_db, image_description_cache_db
from fun import toolbox, parse_cq_codes, file_to_sha256
from chatgpt import ChatGPT
from config_manager import global_config_manager as gcm



# -----------------
# --- 全局变量 ---
driver = get_driver()
chat_lock = False
# ----------------

# --- 数据库初始化 ---
@driver.on_startup
async def init_database():
    global private_chat_table, group_chat_table, gcm, image_description_cache_table
    private_chat_table = chat_record_db.table("private_chat_record")
    group_chat_table = chat_record_db.table("group_chat_record")
    image_description_cache_table = image_description_cache_db.table("image_cache")
    
# --------------------


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
    if not gcm.emoji_response.enable:
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
        if gcm.emoji_response.remove_after_complete:
            await set_msg_emoji(bot, event.message_id, gcm.emoji_response.processing_emoji_id, False)
        await set_msg_emoji(bot, event.message_id, emoji_id)
    elif status == "rejected":
        await set_msg_emoji(bot, event.message_id, emoji_id)

async def get_formatted_history_messages(records: list[dict], bot: Bot, event: GroupMessageEvent, group_id: int) -> str:
    """获取格式化的历史消息记录（并发获取用户名）"""
    async def get_username(user_id: int):
        return (await toolbox.get_qqname(user_id, bot, event)), user_id
    
    task_list = []
    user_name_map = {
        gcm.message_handling.QQ_system_message_id: gcm.message_handling.QQ_system_name,
    }
    if gcm.message_handling.replace_my_name_with_me:
        user_name_map[int(bot.self_id)] = "我"
    
    # 为不在映射中的用户创建获取用户名的任务
    for i in records:
        if i["user_id"] not in user_name_map:
            task_list.append(asyncio.create_task(get_username(i["user_id"])))
    
    # 并发获取用户名
    user_name_results = await asyncio.gather(*task_list)
    
    # 将获取到的用户名添加到映射中
    for result in user_name_results:
        username, user_id = result
        user_name_map[user_id] = username
    
    text = "群聊历史消息:\n"
    for i in records:
        # 从映射中获取用户名
        user_name = user_name_map.get(i["user_id"], f"未知用户")
        
        if i["send_time"]:
            send_time = datetime.fromtimestamp(i["send_time"]).strftime("%m-%d %H:%M:%S")
        else:
            send_time = "unknown"
        text += f"[{send_time}] {user_name}({i['user_id']}, {"消息已撤回" if i["is_recalled"] else i['msg_id']}): {i['content']}\n"
    return text



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

async def get_image_description(bot: Bot, event:GroupMessageEvent, history_messages: str) -> str:
    """获取图片描述（支持并发处理）"""
    # 解析消息中的CQ码
    cq_list = parse_cq_codes(history_messages)
    logger.debug(f"消息列表: {cq_list}")
    
    # 筛选有效图片
    img_list = []
    for i in cq_list:
        if i["CQ"] == "image":
            if "file_size" in i:
                file_size = int(i["file_size"])
            elif "size" in i:
                file_size = int(i["size"])
            else:
                continue # 没有文件大小信息，跳过
            max_size = gcm.image_recognition.max_image_size_mb * 1024 * 1024
            if file_size <= max_size and i["file"] not in img_list:
                img_list.append(i["file"])
    
    logger.debug(f"图片列表: {img_list}")
    if not img_list:
        return ""
    
    # 获取图片路径和哈希值（并发执行）
    semaphore = asyncio.Semaphore(gcm.image_recognition.max_concurrent_tasks)
    img_data = []  # 存储(file_name, file_path, file_hash)
    
    async def process_image(file_name):
        async with semaphore:
            logger.debug(f"处理图片: {file_name}")
            file_path = (await bot.get_image(file=file_name))["file"]
            file_hash = await file_to_sha256(file_path)
            return file_name, file_path, file_hash
    
    tasks = [process_image(img) for img in img_list]
    results = await asyncio.gather(*tasks)
    
    # 使用OrderedDict保持最新图片的顺序
    uncached_images = OrderedDict()
    description_dict = {}
    
    # 检查缓存并分类处理
    for file_name, file_path, file_hash in results:
        row = await image_description_cache_table.query(
            where="hash = ?", 
            params=(file_hash,)
        )
        if row:
            description_dict[file_name] = row[0]["description"]
            logger.debug(f"使用缓存: {file_name}")
        else:
            # 只保留需要API处理的图片
            uncached_images[file_name] = (file_path, file_hash)

    # 如果有未缓存的图片，则更新处理状态
    if uncached_images:
        await update_processing_status(bot,event,"recognize_picture")

    
    # 处理API调用（按顺序限制数量）
    max_api_calls = gcm.image_recognition.max_api_calls_per_recognition
    if len(uncached_images) > max_api_calls:
        # 保留最新的max_api_calls个图片（从后往前取）
        keys = list(uncached_images.keys())[-max_api_calls:]
        uncached_images = OrderedDict((k, uncached_images[k]) for k in keys)
    
    # 并发处理API调用
    api_semaphore = asyncio.Semaphore(gcm.image_recognition.max_concurrent_tasks)


    
    async def call_image_api(file_name, file_path, file_hash):
        global x
        async with api_semaphore:
            
            logger.debug(f"调用API: {file_name}")
            # 为每个任务创建独立的ChatGPT实例，避免上下文干扰
            ai_instance = ChatGPT(
                api_key=gcm.image_recognition.chat_ai_key,
                model=gcm.image_recognition.chat_ai_model,
                base_url=gcm.image_recognition.chat_ai_url
            )
            
            try:
                await ai_instance.add_local_file(
                    file_path, 
                    gcm.image_recognition.prompt
                )
                response = await ai_instance.get_response(
                    no_input=True,
                    add_to_context=False,
                    delete_superfluous_dialogue=False
                )
                return file_name, response, file_hash
            finally:
                # 确保资源被正确释放
                del ai_instance
    
    api_tasks = [
        call_image_api(file_name, *data)
        for file_name, data in uncached_images.items()
    ]
    
    api_results = await asyncio.gather(*api_tasks)
    
    # 处理API结果并更新缓存
    for file_name, description, file_hash in api_results:
        description_dict[file_name] = description
        await image_description_cache_table.insert(
            hash=file_hash,
            description=description
        )
    
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

async def get_group_user_metadata(bot: Bot, user_id: int, group_id: int, no_cache = True, lang: str = "zh") -> dict:
    """
    获取聊群用户信息
    """
    u = await bot.get_group_member_info(group_id=group_id, user_id=user_id, no_cache=no_cache)
    u2 = await bot.get_stranger_info(user_id=user_id, no_cache=no_cache)
    if lang == "zh":
        if u["role"] == "member":
            u["role"] = "成员"
        elif u["role"] == "admin":
            u["role"] = "管理员"
        elif u["role"] == "owner":
            u["role"] = "群主"
        生日 = f"{u2["birthday_year"]}年{u2['birthday_month']}月{u2['birthday_day']}日"
        return {
            "QQ号": u["user_id"],
            "昵称": u["nickname"],
            "群昵称": u["card"],
            "角色": u["role"],
            "头衔": u["title"],
            "群等级": u["level"],
            "加群时间": datetime.fromtimestamp(u["join_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "最后发言时间": datetime.fromtimestamp(u["last_sent_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "性别": u["sex"],
            "年龄": u["age"],
            "生日": 生日,
            "城市": u2["city"],
            "国家": u2["country"],
        }
    else:
        birthday = f"{u2['birthday_year']}-{u2['birthday_month']}-{u2['birthday_day']}"
        return {
            "QQ Number": u["user_id"],
            "Nickname": u["nickname"],
            "Group Nickname": u["card"],
            "Role": u["role"],
            "Title": u["title"],
            "Group Level": u["level"],
            "Join Time": datetime.fromtimestamp(u["join_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Last Message Time": datetime.fromtimestamp(u["last_sent_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Gender": u["sex"],
            "Age": u["age"],
            "Birthday": birthday,
            "City": u2["city"],
            "Country": u2["country"],
        }

async def get_group_user_metadatas_from_history(bot: Bot, group_id: int, history: list[dict], remove_itself: bool = True, lang: str = "zh") -> str:
    user_ids = set()
    # 收集消息中的用户ID
    for msg in history:
        user_ids.add(msg["user_id"])
    
    # 解析CQ码中的@用户
    msg = "".join([m["content"] for m in history])
    for c in parse_cq_codes(msg):
        if c["CQ"] == "at":
            user_ids.add(int(c["qq"]))
    
    # 准备过滤条件
    self_id = int(bot.self_id)
    sys_id = int(gcm.message_handling.QQ_system_message_id)
    
    # 过滤不需要的用户ID
    targets = [
        uid for uid in user_ids
        if (not remove_itself or uid != self_id) and uid != sys_id
    ]
    
    # 并发获取所有用户信息
    tasks = [
        get_group_user_metadata(bot, uid, group_id, lang=lang)
        for uid in targets
    ]
    results = await asyncio.gather(*tasks)
    
    # 构建用户信息字典
    user_datas = {uid: data for uid, data in zip(targets, results)}
    
    return "用户信息:\n" + json.dumps(user_datas, ensure_ascii=False)
# --------------------

# --- 主消息处理器 ---
chat = on_message(rule=to_me(), priority=100, block=False)




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
        logger.debug(f"查询结果: {msg_data}")
        
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
        history_context = await get_formatted_history_messages(
            selected_messages, bot, event, group_id
        )
        images_description = ""
        if gcm.image_recognition.enable:
            images_description = await get_image_description(bot, event, history_context)
        group_metadata = ""
        user_metadata = ""
        if gcm.message_handling.inject_group_metadata:
            group_metadata = await get_group_metadata(bot, group_id)
            group_metadata = json.dumps(group_metadata, ensure_ascii=False)
            group_metadata = f"群聊信息:\n{group_metadata}"
            logger.debug(f"群聊信息: {group_metadata}")
        if gcm.message_handling.inject_user_metadata:
            user_metadata = await get_group_user_metadatas_from_history(bot, group_id, selected_messages)
            user_metadata = f"你自己的QQ号为:{bot.self_id}\n" + user_metadata
            logger.debug(f"用户信息: {user_metadata}")
        


        
        main_chat_ai = ChatGPT(
            api_key=gcm.main_ai.chat_ai_key,
            model=gcm.main_ai.chat_ai_model,
            base_url=gcm.main_ai.chat_ai_url
        )
        # 与AI交互
        logger.debug("添加系统提示到AI上下文")
        await main_chat_ai.add_dialogue(gcm.prompt.full_prompt, "system")
        if group_metadata:
            logger.debug("添加群聊元数据到AI上下文")
            await main_chat_ai.add_dialogue(group_metadata, "user")
        if user_metadata:
            logger.debug("添加用户元数据到AI上下文")
            await main_chat_ai.add_dialogue(user_metadata, "user")
        if images_description:
            logger.debug("添加图片描述到AI上下文")
            images_description = "历史消息中部分图片的描述:\n" + images_description
            await main_chat_ai.add_dialogue(images_description, "user")
        logger.debug("添加历史记录到AI上下文")
        await main_chat_ai.add_dialogue(history_context, "user")
        logger.debug(f"系统提示词: {gcm.prompt.full_prompt}")
        logger.debug(f"图片描述: {images_description}")
        logger.debug(f"历史记录: {history_context}")

        if gcm.message_handling.enable_streaming:
            # 流式处理响应
            full_response = ""
            response_started = False
            line_buffer = ""
            last_send_time = 0  
            
            logger.debug("开始流式接收AI响应")
            async for chunk in main_chat_ai.stream_response(no_input=True, delete_superfluous_dialogue=False):
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
                            if elapsed < gcm.message_handling.min_send_interval:
                                random_time = random.uniform(0, gcm.message_handling.send_interval_random_range)
                                wait_time = gcm.message_handling.min_send_interval + random_time - elapsed
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
                    if elapsed < gcm.message_handling.min_send_interval:
                        wait_time = gcm.message_handling.min_send_interval - elapsed
                        logger.debug(f"等待 {wait_time:.2f} 秒以满足最小发送间隔")
                        await asyncio.sleep(wait_time)
                    
                    logger.debug("发送剩余响应")
                    await send_ai_response(bot, group_id, processed_response)
            await update_processing_status(bot, event, "success")
            logger.debug(f"流式接收AI响应完成: {full_response}")
        else:
            # 非流式处理响应
            logger.debug("开始非流式接收AI响应")
            response = await main_chat_ai.get_response(no_input=True, delete_superfluous_dialogue=False)
            logger.debug(f"非流式接收AI响应完成: {response}")
            processed_response = await process_ai_response(response, bot, event)
            logger.debug(f"处理后的响应: {processed_response}")
            await send_ai_response(bot, group_id, processed_response)
    except Exception as e:
        logger.error(f"处理消息时出错: {type(e).__name__}: {str(e)}")
        logger.exception("详细错误信息")
        await chat.send(f"Σ(°△°|||)︴出错了喵！: {type(e).__name__}: {str(e)}")
    finally:
        chat_lock = False
        del main_chat_ai
        logger.debug("释放消息锁")