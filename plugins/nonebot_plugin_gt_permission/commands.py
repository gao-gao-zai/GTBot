from __future__ import annotations

from typing import Any

from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from .config import get_permission_config
from .manager import PermissionError, PermissionRole, get_permission_manager

PermissionInfoCommand = None
AdminListCommand = None
PromoteAdminCommand = None
DemoteAdminCommand = None


def _get_permission_info_command() -> Any:
    """返回已注册的“查看权限”命令 matcher。

    权限插件允许在 NoneBot 未初始化时被安全导入，因此模块加载早期
    `PermissionInfoCommand` 可能仍为 `None`。该函数统一负责在真正使用
    matcher 前做校验，既避免运行期误用，也帮助静态类型检查收敛可选值。

    Returns:
        已注册完成的“查看权限”命令 matcher。

    Raises:
        RuntimeError: 当命令尚未注册就尝试使用时抛出。
    """

    if PermissionInfoCommand is None:
        raise RuntimeError("查看权限命令尚未注册。")
    return PermissionInfoCommand


def _get_admin_list_command() -> Any:
    """返回已注册的“查看管理员列表”命令 matcher。

    Returns:
        已注册完成的“查看管理员列表”命令 matcher。

    Raises:
        RuntimeError: 当命令尚未注册就尝试使用时抛出。
    """

    if AdminListCommand is None:
        raise RuntimeError("查看管理员列表命令尚未注册。")
    return AdminListCommand


def _get_promote_admin_command() -> Any:
    """返回已注册的“提拔管理员”命令 matcher。

    Returns:
        已注册完成的“提拔管理员”命令 matcher。

    Raises:
        RuntimeError: 当命令尚未注册就尝试使用时抛出。
    """

    if PromoteAdminCommand is None:
        raise RuntimeError("提拔管理员命令尚未注册。")
    return PromoteAdminCommand


def _get_demote_admin_command() -> Any:
    """返回已注册的“降级管理员”命令 matcher。

    Returns:
        已注册完成的“降级管理员”命令 matcher。

    Raises:
        RuntimeError: 当命令尚未注册就尝试使用时抛出。
    """

    if DemoteAdminCommand is None:
        raise RuntimeError("降级管理员命令尚未注册。")
    return DemoteAdminCommand


async def _ensure_admin(user_id: int) -> None:
    """确保当前用户具备管理员权限。

    该校验同时接受管理员和所有者，因为所有者天然拥有管理员以上的权限。
    函数只负责显式权限门禁，不负责将异常转换为消息文本，便于命令处理器保持统一风格。

    Args:
        user_id: 需要校验的 QQ 号。

    Raises:
        PermissionError: 当用户不是管理员或所有者时抛出。
    """

    await get_permission_manager().require_role(int(user_id), PermissionRole.ADMIN)


async def _ensure_owner(user_id: int) -> None:
    """确保当前用户具备所有者权限。

    所有者身份来自插件配置，是最高权限且不会被管理员命令动态修改。

    Args:
        user_id: 需要校验的 QQ 号。

    Raises:
        PermissionError: 当用户不是所有者时抛出。
    """

    await get_permission_manager().require_role(int(user_id), PermissionRole.OWNER)


def _parse_user_id(raw_text: str) -> int:
    """将命令参数解析为用户 QQ 号。

    该函数集中处理空字符串和非整数输入，避免各命令重复编写相同的解析逻辑。
    返回值始终为 `int`，方便后续直接参与权限判断和数据库操作。

    Args:
        raw_text: 原始命令参数文本。

    Returns:
        解析出的整数 QQ 号。

    Raises:
        ValueError: 当参数为空或不是合法整数时抛出。
    """

    text = str(raw_text).strip()
    if not text:
        raise ValueError("请输入目标用户的 QQ 号。")
    return int(text)


def register_permission_commands() -> None:
    """按配置注册权限插件内置命令。

    该函数只负责一次性创建 matcher 并挂接处理器。
    当 NoneBot 尚未初始化或配置关闭内置命令时，会直接返回，使插件在测试和静态导入场景中仍可安全加载。
    """

    global PermissionInfoCommand, AdminListCommand, PromoteAdminCommand, DemoteAdminCommand
    if not get_permission_config().enable_commands:
        return
    if PermissionInfoCommand is not None:
        return
    try:
        get_driver()
    except ValueError:
        return

    PermissionInfoCommand = on_command("查看权限", priority=4, block=True)
    AdminListCommand = on_command("查看管理员列表", priority=4, block=True)
    PromoteAdminCommand = on_command("提拔管理员", priority=4, block=True)
    DemoteAdminCommand = on_command("降级管理员", priority=4, block=True)

    permission_info_command = _get_permission_info_command()
    admin_list_command = _get_admin_list_command()
    promote_admin_command = _get_promote_admin_command()
    demote_admin_command = _get_demote_admin_command()

    @permission_info_command.handle()
    async def handle_permission_info(event: MessageEvent, args: Message = CommandArg()) -> None:
        """处理查看自己或指定用户权限的命令。

        留空参数时查看自己的权限；带参数时，仅管理员和所有者可查询其他用户。
        这样既保留了普通用户自查的能力，也避免用户枚举他人权限。

        Args:
            event: 触发命令的消息事件。
            args: 命令参数，允许为空。

        Raises:
            PermissionError: 当普通用户尝试查询他人权限时抛出。
            ValueError: 当传入的 QQ 号不是合法整数时抛出。
        """

        permission_manager = get_permission_manager()
        arg_text = args.extract_plain_text().strip()
        target_user_id = int(event.user_id)
        if arg_text:
            target_user_id = _parse_user_id(arg_text)
            if target_user_id != int(event.user_id):
                await _ensure_admin(int(event.user_id))
        role_text = await permission_manager.describe_user_role(target_user_id)
        await permission_info_command.finish(f"用户 {target_user_id} 当前权限为：{role_text}")

    @admin_list_command.handle()
    async def handle_admin_list(event: MessageEvent) -> None:
        """处理查看所有者和管理员列表的命令。

        该命令会同时展示静态配置中的 owner 列表和 sqlite 中维护的动态管理员列表，
        以便运维排查时快速确认当前权限人员构成。

        Args:
            event: 触发命令的消息事件。

        Raises:
            PermissionError: 当调用者不具备管理员权限时抛出。
        """

        await _ensure_admin(int(event.user_id))
        permission_manager = get_permission_manager()
        admin_ids = await permission_manager.list_admin_ids()
        owner_ids = sorted(permission_manager.owner_user_ids)
        lines = [
            "权限列表：",
            f"所有者: {', '.join(str(user_id) for user_id in owner_ids) if owner_ids else '(空)'}",
            f"管理员: {', '.join(str(user_id) for user_id in admin_ids) if admin_ids else '(空)'}",
        ]
        await admin_list_command.finish("\n".join(lines))

    @promote_admin_command.handle()
    async def handle_promote_admin(event: MessageEvent, args: Message = CommandArg()) -> None:
        """处理提拔管理员命令。

        只有所有者可执行该命令。目标用户已经是管理员时不会重复写库，
        目标用户是所有者时会直接由权限管理器拒绝。

        Args:
            event: 触发命令的消息事件。
            args: 目标用户 QQ 号。

        Raises:
            PermissionError: 当调用者不是所有者时抛出。
            ValueError: 当目标参数不合法或目标用户已经是所有者时抛出。
        """

        await _ensure_owner(int(event.user_id))
        target_user_id = _parse_user_id(args.extract_plain_text().strip())
        created = await get_permission_manager().add_admin(target_user_id, int(event.user_id))
        if created:
            await promote_admin_command.finish(f"已将用户 {target_user_id} 提拔为管理员。")
        await promote_admin_command.finish(f"用户 {target_user_id} 已经是管理员。")

    @demote_admin_command.handle()
    async def handle_demote_admin(event: MessageEvent, args: Message = CommandArg()) -> None:
        """处理降级管理员命令。

        只有所有者可执行该命令。目标用户不是管理员时不会报错，
        而是返回一条幂等提示，方便重复执行自动化脚本时保持稳定。

        Args:
            event: 触发命令的消息事件。
            args: 目标用户 QQ 号。

        Raises:
            PermissionError: 当调用者不是所有者时抛出。
            ValueError: 当目标参数不合法或目标用户是所有者时抛出。
        """

        await _ensure_owner(int(event.user_id))
        target_user_id = _parse_user_id(args.extract_plain_text().strip())
        removed = await get_permission_manager().remove_admin(target_user_id, int(event.user_id))
        if removed:
            await demote_admin_command.finish(f"已移除用户 {target_user_id} 的管理员权限。")
        await demote_admin_command.finish(f"用户 {target_user_id} 当前不是管理员。")
