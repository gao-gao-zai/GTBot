from nonebot import on_message, on_notice, on_command, logger
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message, GroupMessageEvent, NoticeEvent, MessageSegment
from nonebot.adapters.onebot.v11 import GroupMessageEvent, PrivateMessageEvent, GroupIncreaseNoticeEvent, GroupDecreaseNoticeEvent, PokeNotifyEvent, GroupRecallNoticeEvent
from nonebot.adapters.onebot.utils import rich_escape, rich_unescape
from nonebot import get_driver
import sqlite3
from pathlib import Path
import sys

DirPath = Path(__file__).parent
sys.path.append(str(DirPath))

from SQLiteManager import chat_record_db, group_message_manager
from fun import toolbox

driver = get_driver()



@driver.on_startup
async def _():
    global chat_record_db, private_chat_table, group_chat_table, gcm
    from config_manager import global_config_manager as gcm
    private_chat_table = chat_record_db.table("private_chat_record")
    group_chat_table = chat_record_db.table("group_chat_record")

@driver.on_shutdown
async def close_database():
    await chat_record_db.close()

# --- 消息记录部分 ---
async def record_group_chat(msg_id, group_id, user_id, content, send_time):
    await group_chat_table.insert(msg_id=msg_id, group_id=group_id, user_id=user_id, content=content, send_time=send_time)

record_group_chat_ = on_message(priority=10, block=False)
record_private_chat_ = on_message(priority=10, block=False)

@record_group_chat_.handle()
async def _(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    user_id = event.user_id
    msg_id = event.message_id
    logger.debug(f"收到群聊消息: {str(event.original_message)}")
    content = toolbox.message_to_text(event.original_message, event)
    logger.debug(f"转义消息: {content}")
    await group_chat_table.insert(msg_id=msg_id, group_id=group_id, user_id=user_id, content=content, send_time=event.time)

# --- 新增事件监听部分 ---
# 1. 新成员入群
welcome_handler = on_notice(priority=10, block=False)

@welcome_handler.handle()
async def handle_welcome(bot: Bot, event: GroupIncreaseNoticeEvent):
    group_id = event.group_id
    user_id = event.user_id
    user_name = await toolbox.get_qqname(qq=user_id, bot=bot, event=event)
    await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=group_id, user_id=-1, content=f"成员{user_id}({user_id})加入了群聊", send_time=event.time)
    logger.info(f"新成员 {user_id} 加入群 {group_id}")

# 2. 群成员退群
leave_handler = on_notice(priority=10, block=False)

@leave_handler.handle()
async def handle_leave(bot:Bot, event: GroupDecreaseNoticeEvent):
    group_id = event.group_id
    user_id = event.user_id
    operator_id = event.operator_id
    user_name = await toolbox.get_qqname(qq=user_id, bot=bot, event=event)
    
    if user_id == operator_id:
        await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=group_id, user_id=-1, content=f"成员{user_name}({user_id})退出了群聊", send_time=event.time)
    else:
        operator_name = await toolbox.get_qqname(qq=operator_id, bot=bot, event=event)
        await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=group_id, user_id=-1, content=f"管理员{operator_name}({operator_id})将成员{user_id}({user_id})移出了群聊", send_time=event.time)
    
    logger.info(f"成员 {user_id} 退出群 {group_id}")

# 3. 戳一戳
poke_handler = on_notice(priority=10, block=False)

@poke_handler.handle()
async def handle_poke(bot:Bot, event: PokeNotifyEvent):
    if not hasattr(event, "group_id"):
        return
    
    sender_id = event.user_id
    target_id = event.target_id
    sender_name = await toolbox.get_qqname(qq=sender_id, bot=bot, event=event)
    target_name = await toolbox.get_qqname(qq=target_id, bot=bot, event=event)
    if target_id == event.self_id:
        await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=event.group_id, user_id=-1, content=f"成员{sender_name}({sender_id})戳了我", send_time=event.time)
    else:
        await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=event.group_id, user_id=-1, content=f"成员{sender_name}({sender_id})戳了成员{target_name}({target_id})", send_time=event.time)
        

# 4. 消息撤回
recall_handler = on_notice(priority=10, block=False)

@recall_handler.handle()
async def handle_recall(bot: Bot, event: GroupRecallNoticeEvent):
    group_id = event.group_id
    user_id = event.user_id
    operator_id = event.operator_id
    recalled_msg_id = event.message_id

    user_name = await toolbox.get_qqname(qq=user_id, bot=bot, event=event)

    await group_message_manager.update_message(old_msg_id=recalled_msg_id, is_recalled=True)
    
    # if user_id == operator_id:
    #     await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=group_id, user_id=-1, content=f"成员{user_name}({user_id})撤回了一条消息({recalled_msg_id})", send_time=event.time)
    # else:
    #     operator_name = await toolbox.get_qqname(qq=operator_id, bot=bot, event=event)
    #     await group_chat_table.insert(msg_id=gcm.message_handling.QQ_system_message_id, group_id=group_id, user_id=-1, content=f"管理员{operator_name}({operator_id})撤回了成员{user_name}({user_id})的消息({recalled_msg_id})", send_time=event.time)

    if user_id == operator_id:
        await group_message_manager.add_message(group_id=group_id, msg_id=gcm.message_handling.QQ_system_message_id, user_id=-1, content=f"成员{user_name}({user_id})撤回了一条消息({recalled_msg_id})", send_time=event.time)
    else:
        operator_name = await toolbox.get_qqname(qq=operator_id, bot=bot, event=event)
        await group_message_manager.add_message(group_id=group_id, msg_id=gcm.message_handling.QQ_system_message_id, user_id=-1, content=f"管理员{operator_name}({operator_id})撤回了成员{user_name}({user_id})的消息({recalled_msg_id})", send_time=event.time)
    
    logger.info(f"群 {group_id} 消息被撤回")