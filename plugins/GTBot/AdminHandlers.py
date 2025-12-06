"""管理员处理器模块。

提供管理员专属功能，包括：
- 查看用户/群聊画像
- 自动同意管理员邀请加群请求
- 管理员退群指令
"""

from nonebot import logger, on_message, on_request
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, GroupRequestEvent
from nonebot.params import Depends

from .ConfigManager import total_config
from .MassageManager import GroupMessageManager, get_message_manager
from .UserProfileManager import ProfileManager, get_profile_manager


def is_admin(user_id: int) -> bool:
    """检查用户是否为管理员。
    
    Args:
        user_id: 用户QQ号。
        
    Returns:
        是否为管理员。
    """
    admin_user_ids = total_config.processed_configuration.config.admin_user_ids
    return user_id in admin_user_ids


# ============================================================================
# 管理员处理器 - 查看用户和群聊画像
# ============================================================================

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
    if not is_admin(event.user_id):
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
    if not is_admin(event.user_id):
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


# ============================================================================
# 管理员邀请加群自动同意处理器
# ============================================================================

AdminGroupInviteRequest = on_request(priority=1, block=True)


@AdminGroupInviteRequest.handle()
async def handle_admin_group_invite(event: GroupRequestEvent, bot: Bot):
    """自动同意管理员发起的邀请加群请求。
    
    当管理员邀请机器人加入群聊时，自动同意邀请请求。
    
    Args:
        event: 群组请求事件。
        bot: 机器人实例。
    """
    # 只处理邀请类型的请求
    if event.sub_type != "invite":
        return
    
    # 检查邀请者是否为管理员
    if not is_admin(event.user_id):
        logger.info(f"非管理员 {event.user_id} 邀请机器人加入群 {event.group_id}，忽略")
        return
    
    try:
        # 同意加群邀请
        await bot.set_group_add_request(
            flag=event.flag,
            sub_type=event.sub_type,
            approve=True
        )
        logger.info(f"✅ 已自动同意管理员 {event.user_id} 的加群邀请（群号: {event.group_id}）")
    except Exception as e:
        logger.error(f"❌ 处理加群邀请失败: {str(e)}")


# ============================================================================
# 管理员退群指令处理器
# ============================================================================

AdminLeaveGroup = on_message(priority=4, block=False)


@AdminLeaveGroup.handle()
async def handle_admin_leave_group(
    event: GroupMessageEvent,
    bot: Bot,
):
    """处理管理员退群指令。
    
    命令格式: 
    - 退出群聊 <群号>  - 退出指定群聊
    - 退出本群         - 退出当前群聊
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
    """
    # 仅允许管理员使用
    if not is_admin(event.user_id):
        return
    
    # 检查消息格式
    msg_text = event.get_plaintext().strip()
    
    # 处理"退出本群"命令
    if msg_text == "退出本群":
        target_group_id = event.group_id
        try:
            await AdminLeaveGroup.send(f"⚠️ 即将退出当前群聊（{target_group_id}）...")
            await bot.set_group_leave(group_id=target_group_id)
            logger.info(f"✅ 管理员 {event.user_id} 指令机器人退出群聊 {target_group_id}")
        except Exception as e:
            logger.error(f"❌ 退出群聊 {target_group_id} 失败: {str(e)}")
            await AdminLeaveGroup.send(f"❌ 退出失败: {str(e)}")
        return
    
    # 处理"退出群聊 <群号>"命令
    if not msg_text.startswith("退出群聊"):
        return
    
    # 提取群号
    parts = msg_text.split()
    if len(parts) < 2:
        await AdminLeaveGroup.send("❌ 命令格式错误。用法:\n- 退出群聊 <群号>\n- 退出本群")
        return
    
    try:
        target_group_id = int(parts[1])
    except ValueError:
        await AdminLeaveGroup.send(f"❌ 无效的群号: {parts[1]}")
        return
    
    try:
        logger.info(f"管理员 {event.user_id} 请求退出群聊 {target_group_id}")
        await bot.set_group_leave(group_id=target_group_id)
        await AdminLeaveGroup.send(f"✅ 已退出群聊 {target_group_id}")
        logger.info(f"✅ 管理员 {event.user_id} 指令机器人退出群聊 {target_group_id}")
    except Exception as e:
        logger.error(f"❌ 退出群聊 {target_group_id} 失败: {str(e)}")
        await AdminLeaveGroup.send(f"❌ 退出失败: {str(e)}")
