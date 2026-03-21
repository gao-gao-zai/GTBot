"""管理员处理器模块。

提供管理员专属功能，包括：
- 查看用户/群聊画像
- 自动同意管理员邀请加群请求
- 管理员退群指令
"""

from nonebot import logger, on_command, on_request
from nonebot.adapters.onebot.v11.bot import Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, GroupRequestEvent, MessageEvent
from nonebot.params import Depends, CommandArg
from nonebot.adapters.onebot.v11.message import Message

from .MassageManager import GroupMessageManager, get_message_manager
from .PermissionManager import PermissionError, PermissionRole, get_permission_manager
from .UserProfileManager import ProfileManager, get_profile_manager


async def _ensure_admin(user_id: int) -> None:
    permission_manager = get_permission_manager()
    await permission_manager.require_role(user_id, PermissionRole.ADMIN)


async def _ensure_owner(user_id: int) -> None:
    permission_manager = get_permission_manager()
    await permission_manager.require_role(user_id, PermissionRole.OWNER)


# ============================================================================
# 管理员处理器 - 查看用户和群聊画像
# ============================================================================

# 查看用户画像处理器
AdminQueryUserProfile = on_command("查看用户画像", priority=4, block=True)


@AdminQueryUserProfile.handle()
async def handle_query_user_profile(
    event: GroupMessageEvent,
    bot: Bot,
    args: Message = CommandArg(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    profile_manager: ProfileManager = Depends(get_profile_manager)
):
    """处理管理员查看用户画像的请求。
    
    命令格式: /查看用户画像 <用户QQ号>
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
        args: 命令参数。
        msg_mg: 消息管理器。
        profile_manager: 画像管理器。
    """
    # 仅允许管理员使用
    try:
        await _ensure_admin(event.user_id)
    except PermissionError:
        return
    
    # 提取用户QQ号
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await AdminQueryUserProfile.finish("❌ 命令格式错误。用法: /查看用户画像 <用户QQ号>")
    
    try:
        target_user_id = int(arg_text)
    except ValueError:
        await AdminQueryUserProfile.finish(f"❌ 无效的用户QQ号: {arg_text}")
    
    try:
        logger.info(f"管理员 {event.user_id} 查询用户 {target_user_id} 的画像")
        
        # 获取用户画像
        user_profile = await profile_manager.user.get_user_descriptions_with_index(target_user_id)
        
        if not user_profile:
            await AdminQueryUserProfile.finish(f"ℹ️ 用户 {target_user_id} 尚未设置画像描述。")
        
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
        await AdminQueryUserProfile.finish(f"❌ 查询失败: {str(e)}")


# 查看群聊画像处理器
AdminQueryGroupProfile = on_command("查看群聊画像", priority=4, block=True)


@AdminQueryGroupProfile.handle()
async def handle_query_group_profile(
    event: GroupMessageEvent,
    bot: Bot,
    args: Message = CommandArg(),
    msg_mg: GroupMessageManager = Depends(get_message_manager),
    profile_manager: ProfileManager = Depends(get_profile_manager)
):
    """处理管理员查看群聊画像的请求。
    
    命令格式: /查看群聊画像 [群号]
    若不指定群号，则查看当前群聊的画像。
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
        args: 命令参数。
        msg_mg: 消息管理器。
        profile_manager: 画像管理器。
    """
    # 仅允许管理员使用
    try:
        await _ensure_admin(event.user_id)
    except PermissionError:
        return
    
    # 提取群号（可选，默认为当前群）
    target_group_id = event.group_id
    arg_text = args.extract_plain_text().strip()
    
    if arg_text:
        try:
            target_group_id = int(arg_text)
        except ValueError:
            await AdminQueryGroupProfile.finish(f"❌ 无效的群号: {arg_text}")
    
    try:
        logger.info(f"管理员 {event.user_id} 查询群聊 {target_group_id} 的画像")
        
        # 获取群聊画像
        group_profile = await profile_manager.group.get_group_descriptions_with_index(target_group_id)
        
        if not group_profile:
            await AdminQueryGroupProfile.finish(f"ℹ️ 群聊 {target_group_id} 尚未设置画像描述。")
        
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
        await AdminQueryGroupProfile.finish(f"❌ 查询失败: {str(e)}")


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
    permission_manager = get_permission_manager()
    if not await permission_manager.has_role(event.user_id, PermissionRole.ADMIN):
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

AdminLeaveGroup = on_command("退出群聊", priority=4, block=True)
AdminLeaveCurrentGroup = on_command("退出本群", priority=4, block=True)
PermissionInfoCommand = on_command("查看权限", priority=4, block=True)
AdminListCommand = on_command("查看管理员列表", priority=4, block=True)
PromoteAdminCommand = on_command("提拔管理员", priority=4, block=True)
DemoteAdminCommand = on_command("降级管理员", priority=4, block=True)


@AdminLeaveCurrentGroup.handle()
async def handle_admin_leave_current_group(
    event: GroupMessageEvent,
    bot: Bot,
):
    """处理管理员退出当前群聊指令。
    
    命令格式: /退出本群
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
    """
    # 仅允许管理员使用
    try:
        await _ensure_admin(event.user_id)
    except PermissionError:
        return
    
    target_group_id = event.group_id
    try:
        await AdminLeaveCurrentGroup.send(f"⚠️ 即将退出当前群聊（{target_group_id}）...")
        await bot.set_group_leave(group_id=target_group_id)
        logger.info(f"✅ 管理员 {event.user_id} 指令机器人退出群聊 {target_group_id}")
    except Exception as e:
        logger.error(f"❌ 退出群聊 {target_group_id} 失败: {str(e)}")
        await AdminLeaveCurrentGroup.finish(f"❌ 退出失败: {str(e)}")


@AdminLeaveGroup.handle()
async def handle_admin_leave_group(
    event: GroupMessageEvent,
    bot: Bot,
    args: Message = CommandArg(),
):
    """处理管理员退群指令。
    
    命令格式: /退出群聊 <群号>
    
    Args:
        event: 群消息事件。
        bot: 机器人实例。
        args: 命令参数。
    """
    # 仅允许管理员使用
    try:
        await _ensure_admin(event.user_id)
    except PermissionError:
        return
    
    # 提取群号
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await AdminLeaveGroup.finish("❌ 命令格式错误。用法:\n- /退出群聊 <群号>\n- /退出本群")
    
    try:
        target_group_id = int(arg_text)
    except ValueError:
        await AdminLeaveGroup.finish(f"❌ 无效的群号: {arg_text}")
    
    try:
        logger.info(f"管理员 {event.user_id} 请求退出群聊 {target_group_id}")
        await bot.set_group_leave(group_id=target_group_id)
        await AdminLeaveGroup.send(f"✅ 已退出群聊 {target_group_id}")
        logger.info(f"✅ 管理员 {event.user_id} 指令机器人退出群聊 {target_group_id}")
    except Exception as e:
        logger.error(f"❌ 退出群聊 {target_group_id} 失败: {str(e)}")
        await AdminLeaveGroup.finish(f"❌ 退出失败: {str(e)}")


@PermissionInfoCommand.handle()
async def handle_permission_info(
    event: MessageEvent,
    args: Message = CommandArg(),
):
    permission_manager = get_permission_manager()
    arg_text = args.extract_plain_text().strip()
    target_user_id = event.user_id

    if arg_text:
        try:
            await _ensure_admin(event.user_id)
            target_user_id = int(arg_text)
        except PermissionError as exc:
            await PermissionInfoCommand.finish(str(exc))
        except ValueError:
            await PermissionInfoCommand.finish(f"无效的 QQ 号: {arg_text}")

    role_text = await permission_manager.describe_user_role(target_user_id)
    await PermissionInfoCommand.finish(f"用户 {target_user_id} 当前权限为：{role_text}")


@AdminListCommand.handle()
async def handle_admin_list(event: MessageEvent):
    try:
        await _ensure_admin(event.user_id)
    except PermissionError as exc:
        await AdminListCommand.finish(str(exc))

    permission_manager = get_permission_manager()
    admin_ids = await permission_manager.list_admin_ids()
    owner_ids = sorted(permission_manager.owner_user_ids)
    lines = [
        "GTBot 权限列表：",
        f"所有者: {', '.join(str(user_id) for user_id in owner_ids) if owner_ids else '(空)'}",
        f"管理员: {', '.join(str(user_id) for user_id in admin_ids) if admin_ids else '(空)'}",
    ]
    await AdminListCommand.finish("\n".join(lines))


@PromoteAdminCommand.handle()
async def handle_promote_admin(
    event: MessageEvent,
    args: Message = CommandArg(),
):
    try:
        await _ensure_owner(event.user_id)
    except PermissionError as exc:
        await PromoteAdminCommand.finish(str(exc))

    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await PromoteAdminCommand.finish("用法: /提拔管理员 <QQ号>")

    try:
        target_user_id = int(arg_text)
    except ValueError:
        await PromoteAdminCommand.finish(f"无效的 QQ 号: {arg_text}")

    permission_manager = get_permission_manager()
    try:
        created = await permission_manager.add_admin(target_user_id, event.user_id)
    except ValueError as exc:
        await PromoteAdminCommand.finish(str(exc))

    if created:
        await PromoteAdminCommand.finish(f"已将 {target_user_id} 提拔为管理员。")
    await PromoteAdminCommand.finish(f"{target_user_id} 已经是管理员。")


@DemoteAdminCommand.handle()
async def handle_demote_admin(
    event: MessageEvent,
    args: Message = CommandArg(),
):
    try:
        await _ensure_owner(event.user_id)
    except PermissionError as exc:
        await DemoteAdminCommand.finish(str(exc))

    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await DemoteAdminCommand.finish("用法: /降级管理员 <QQ号>")

    try:
        target_user_id = int(arg_text)
    except ValueError:
        await DemoteAdminCommand.finish(f"无效的 QQ 号: {arg_text}")

    permission_manager = get_permission_manager()
    try:
        removed = await permission_manager.remove_admin(target_user_id, event.user_id)
    except ValueError as exc:
        await DemoteAdminCommand.finish(str(exc))

    if removed:
        await DemoteAdminCommand.finish(f"已将 {target_user_id} 从管理员降级为普通用户。")
    await DemoteAdminCommand.finish(f"{target_user_id} 当前不是管理员。")
