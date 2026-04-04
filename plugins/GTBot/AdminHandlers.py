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

from .services.help import HelpArgumentSpec, HelpCommandSpec, register_help
from .services.message import GroupMessageManager, get_message_manager
from .services.permission import PermissionError, PermissionRole, get_permission_manager
from .services.profile import ProfileManager, get_profile_manager


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


def _register_admin_help_items() -> None:
    """注册管理员相关核心命令的帮助信息。"""
    register_help(
        HelpCommandSpec(
            name="查看用户画像",
            category="画像管理",
            summary="查看指定用户的画像描述。",
            description="管理员可查询指定 QQ 用户当前保存的全部画像描述及其序号。",
            arguments=(
                HelpArgumentSpec(
                    name="<用户QQ号>",
                    description="要查询画像的目标用户 QQ 号。",
                    value_hint="整数 QQ 号",
                    example="123456789",
                ),
            ),
            examples=(
                "/查看用户画像 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊管理员命令",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="查看群聊画像",
            category="画像管理",
            summary="查看当前群或指定群的画像描述。",
            description="管理员可查看当前群聊画像；也可以额外传入群号查询其他群的画像信息。",
            arguments=(
                HelpArgumentSpec(
                    name="[群号]",
                    description="可选的目标群号；留空时默认查看当前群。",
                    required=False,
                    value_hint="整数群号",
                    example="987654321",
                ),
            ),
            examples=(
                "/查看群聊画像",
                "/查看群聊画像 987654321",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊管理员命令",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name="退出本群",
            category="群聊管理",
            summary="让机器人退出当前群聊。",
            description="管理员在群聊中直接执行该命令，机器人会尝试退出当前群。",
            examples=(
                "/退出本群",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊管理员命令",
            sort_key=30,
        )
    )
    register_help(
        HelpCommandSpec(
            name="退出群聊",
            category="群聊管理",
            summary="让机器人退出指定群聊。",
            description="管理员可通过指定群号，要求机器人退出对应群聊。",
            arguments=(
                HelpArgumentSpec(
                    name="<群号>",
                    description="要退出的目标群号。",
                    value_hint="整数群号",
                    example="987654321",
                ),
            ),
            examples=(
                "/退出群聊 987654321",
            ),
            required_role=PermissionRole.ADMIN,
            audience="管理员后台命令",
            sort_key=40,
        )
    )
    register_help(
        HelpCommandSpec(
            name="查看权限",
            category="权限管理",
            summary="查看自己或指定用户的权限等级。",
            description="留空时查看自己的权限；管理员可附带 QQ 号查看目标用户的权限等级。",
            arguments=(
                HelpArgumentSpec(
                    name="[QQ号]",
                    description="可选的目标用户 QQ 号。",
                    required=False,
                    value_hint="整数 QQ 号",
                    example="123456789",
                ),
            ),
            examples=(
                "/查看权限",
                "/查看权限 123456789",
            ),
            required_role=PermissionRole.USER,
            audience="群聊和私聊",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="查看管理员列表",
            category="权限管理",
            summary="查看当前所有者和管理员列表。",
            description="管理员可查看当前 GTBot 所有者与管理员账号列表。",
            examples=(
                "/查看管理员列表",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name="提拔管理员",
            category="权限管理",
            summary="将指定用户设为管理员。",
            description="仅所有者可用，用于把目标 QQ 号提升为 GTBot 管理员。",
            arguments=(
                HelpArgumentSpec(
                    name="<QQ号>",
                    description="要提拔为管理员的目标用户 QQ 号。",
                    value_hint="整数 QQ 号",
                    example="123456789",
                ),
            ),
            examples=(
                "/提拔管理员 123456789",
            ),
            required_role=PermissionRole.OWNER,
            audience="所有者命令",
            sort_key=30,
        )
    )
    register_help(
        HelpCommandSpec(
            name="降级管理员",
            category="权限管理",
            summary="移除指定用户的管理员权限。",
            description="仅所有者可用，用于将目标管理员降级为普通用户。",
            arguments=(
                HelpArgumentSpec(
                    name="<QQ号>",
                    description="要降级的目标用户 QQ 号。",
                    value_hint="整数 QQ 号",
                    example="123456789",
                ),
            ),
            examples=(
                "/降级管理员 123456789",
            ),
            required_role=PermissionRole.OWNER,
            audience="所有者命令",
            sort_key=40,
        )
    )


_register_admin_help_items()


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
