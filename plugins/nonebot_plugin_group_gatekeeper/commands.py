from __future__ import annotations

from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent
from nonebot.params import CommandArg

from local_plugins.nonebot_plugin_gt_help import (
    HelpArgumentSpec,
    HelpCommandSpec,
    get_help_registry,
    register_help,
)
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, require_admin

from .store import store

StatusCommand = None
EnableCommand = None
DisableCommand = None
SetMinLevelCommand = None
SetRequestExpireCommand = None
SetVerifyTimeoutCommand = None
SetCodeLengthCommand = None
SetRejectOnLowLevelCommand = None
SetPromptCommand = None
ResetPromptCommand = None

PROMPT_SETTING_LABELS: dict[str, tuple[str, str]] = {
    "欢迎": ("welcome_template", "入群欢迎提示"),
    "提醒": ("failure_template", "验证码错误/无关消息提醒"),
    "成功": ("success_template", "验证码成功提示"),
    "超时": ("timeout_template", "验证码超时提示"),
    "拒绝": ("reject_reason", "等级不足拒绝理由"),
}


def _parse_positive_int(raw_text: str, field_name: str) -> int:
    """将命令参数解析为正整数。

    Args:
        raw_text: 原始命令参数文本。
        field_name: 参数名，用于构造错误提示。

    Returns:
        解析得到的正整数值。

    Raises:
        ValueError: 当输入为空、不是整数或小于 1 时抛出。
    """

    text = str(raw_text).strip()
    if not text:
        raise ValueError(f"请提供{field_name}")
    value = int(text)
    if value < 1:
        raise ValueError(f"{field_name}必须大于 0")
    return value


def _parse_group_id(raw_text: str, event: MessageEvent) -> int:
    """从参数或当前群上下文中解析目标群号。

    Args:
        raw_text: 命令参数文本。
        event: 当前消息事件。

    Returns:
        解析得到的目标群号。

    Raises:
        ValueError: 当无法从参数或上下文中得到群号时抛出。
    """

    text = str(raw_text).strip()
    if text:
        return _parse_positive_int(text, "目标群号")
    if isinstance(event, GroupMessageEvent):
        return int(event.group_id)
    raise ValueError("请提供目标群号，或在群聊内直接执行该命令")


async def _ensure_admin_role(event: MessageEvent) -> None:
    """确保当前命令调用者具备管理员权限。

    Args:
        event: 触发命令的消息事件。

    Raises:
        PermissionError: 当调用者不具备管理员权限时抛出。
    """

    await require_admin(int(event.user_id))


async def _finish(command, message: str) -> None:  # noqa: ANN001
    """统一结束命令并返回消息。

    Args:
        command: 当前命令的 matcher。
        message: 需要返回的文本。
    """

    await command.finish(message)


async def finish_with_error(command, exc: Exception) -> None:  # noqa: ANN001
    """将权限或参数异常统一转成命令回复。

    Args:
        command: 当前命令的 matcher。
        exc: 需要展示给用户的异常对象。
    """

    await command.finish(str(exc))


def _parse_prompt_args(raw_text: str) -> tuple[str, str]:
    """解析文案设置命令中的类型和值。

    Args:
        raw_text: 命令参数原文。

    Returns:
        由配置键名和文案内容组成的二元组。

    Raises:
        ValueError: 当文案类型缺失、未知或未提供内容时抛出。
    """

    text = str(raw_text).strip()
    if not text:
        raise ValueError("请提供文案类型和新内容，例如：/设置入群守卫提示 欢迎 你的文案")
    parts = text.split(maxsplit=1)
    prompt_type = parts[0].strip()
    if prompt_type not in PROMPT_SETTING_LABELS:
        raise ValueError("文案类型只支持：欢迎、提醒、成功、超时、拒绝")
    if len(parts) < 2 or not parts[1].strip():
        raise ValueError("请提供新的文案内容")
    setting_key, _ = PROMPT_SETTING_LABELS[prompt_type]
    return setting_key, parts[1].strip()


def _parse_prompt_type(raw_text: str) -> tuple[str, str]:
    """解析文案重置命令中的文案类型。

    Args:
        raw_text: 命令参数原文。

    Returns:
        由配置键名和文案类型显示名组成的二元组。

    Raises:
        ValueError: 当文案类型缺失或未知时抛出。
    """

    prompt_type = str(raw_text).strip()
    if prompt_type not in PROMPT_SETTING_LABELS:
        raise ValueError("文案类型只支持：欢迎、提醒、成功、超时、拒绝")
    setting_key, display_name = PROMPT_SETTING_LABELS[prompt_type]
    return setting_key, display_name


def _parse_toggle_value(raw_text: str) -> bool:
    """将开关命令参数解析为布尔值。

    Args:
        raw_text: 原始命令参数文本。

    Returns:
        解析后的布尔值。

    Raises:
        ValueError: 当参数为空或不是约定的开关值时抛出。
    """

    normalized = str(raw_text).strip().lower()
    if normalized in {"开", "开启", "启用", "on", "true", "1", "yes"}:
        return True
    if normalized in {"关", "关闭", "禁用", "off", "false", "0", "no"}:
        return False
    raise ValueError("请使用 开 或 关")


def register_commands() -> None:
    """注册插件管理命令。"""

    global StatusCommand, EnableCommand, DisableCommand
    global SetMinLevelCommand, SetRequestExpireCommand, SetVerifyTimeoutCommand, SetCodeLengthCommand
    global SetRejectOnLowLevelCommand, SetPromptCommand, ResetPromptCommand

    if StatusCommand is not None:
        return
    try:
        get_driver()
    except ValueError:
        return

    StatusCommand = on_command("入群守卫状态", priority=4, block=True)
    EnableCommand = on_command("开启入群守卫", priority=4, block=True)
    DisableCommand = on_command("关闭入群守卫", priority=4, block=True)
    SetMinLevelCommand = on_command("设置入群等级", priority=4, block=True)
    SetRequestExpireCommand = on_command("设置申请处理时限", priority=4, block=True)
    SetVerifyTimeoutCommand = on_command("设置验证码时限", priority=4, block=True)
    SetCodeLengthCommand = on_command("设置验证码长度", priority=4, block=True)
    SetRejectOnLowLevelCommand = on_command("设置等级不足自动拒绝", priority=4, block=True)
    SetPromptCommand = on_command("设置入群守卫提示", priority=4, block=True)
    ResetPromptCommand = on_command("重置入群守卫提示", priority=4, block=True)

    @StatusCommand.handle()
    async def handle_status(event: MessageEvent) -> None:
        """查看当前入群守卫配置概览。

        Args:
            event: 触发命令的消息事件。
        """

        try:
            await _ensure_admin_role(event)
            config = await store.get_config()
            groups_text = ", ".join(str(group_id) for group_id in sorted(config.target_groups)) or "(空)"
            await _finish(
                StatusCommand,
                "\n".join(
                    [
                        "入群守卫状态",
                        f"数据库: {store.database_path}",
                        f"目标群: {groups_text}",
                        f"最低等级: {config.min_level}",
                        f"申请处理时限: {config.request_expire_seconds} 秒",
                        f"验证码时限: {config.verify_timeout_seconds} 秒",
                        f"验证码长度: {config.code_length}",
                        f"等级不足自动拒绝: {'是' if config.reject_on_low_level else '否'}",
                        f"超时自动踢出: {'是' if config.kick_on_timeout else '否'}",
                    ]
                ),
            )
        except (PermissionError, ValueError) as exc:
            await finish_with_error(StatusCommand, exc)

    @EnableCommand.handle()
    async def handle_enable(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """开启某个群的入群守卫功能。

        Args:
            event: 触发命令的消息事件。
            args: 可选的群号参数。
        """

        try:
            await _ensure_admin_role(event)
            group_id = _parse_group_id(args.extract_plain_text(), event)
            await store.enable_group(group_id)
            await _finish(EnableCommand, f"已开启群 {group_id} 的入群守卫。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(EnableCommand, exc)

    @DisableCommand.handle()
    async def handle_disable(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """关闭某个群的入群守卫功能。

        Args:
            event: 触发命令的消息事件。
            args: 可选的群号参数。
        """

        try:
            await _ensure_admin_role(event)
            group_id = _parse_group_id(args.extract_plain_text(), event)
            disabled = await store.disable_group(group_id)
            if disabled:
                await _finish(DisableCommand, f"已关闭群 {group_id} 的入群守卫。")
            await _finish(DisableCommand, f"群 {group_id} 当前未启用入群守卫。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(DisableCommand, exc)

    @SetMinLevelCommand.handle()
    async def handle_set_min_level(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新最低 QQ 等级门槛。

        Args:
            event: 触发命令的消息事件。
            args: 等级参数。
        """

        try:
            await _ensure_admin_role(event)
            min_level = _parse_positive_int(args.extract_plain_text(), "最低等级")
            await store.set_setting("min_level", str(min_level))
            await _finish(SetMinLevelCommand, f"已将最低等级设置为 {min_level}。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetMinLevelCommand, exc)

    @SetRequestExpireCommand.handle()
    async def handle_set_request_expire(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新申请处理时限。

        Args:
            event: 触发命令的消息事件。
            args: 秒数参数。
        """

        try:
            await _ensure_admin_role(event)
            seconds = _parse_positive_int(args.extract_plain_text(), "申请处理时限")
            await store.set_setting("request_expire_seconds", str(seconds))
            await _finish(SetRequestExpireCommand, f"已将申请处理时限设置为 {seconds} 秒。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetRequestExpireCommand, exc)

    @SetVerifyTimeoutCommand.handle()
    async def handle_set_verify_timeout(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新验证码时限。

        Args:
            event: 触发命令的消息事件。
            args: 秒数参数。
        """

        try:
            await _ensure_admin_role(event)
            seconds = _parse_positive_int(args.extract_plain_text(), "验证码时限")
            await store.set_setting("verify_timeout_seconds", str(seconds))
            await _finish(SetVerifyTimeoutCommand, f"已将验证码时限设置为 {seconds} 秒。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetVerifyTimeoutCommand, exc)

    @SetCodeLengthCommand.handle()
    async def handle_set_code_length(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新动态验证码长度。

        Args:
            event: 触发命令的消息事件。
            args: 长度参数。
        """

        try:
            await _ensure_admin_role(event)
            code_length = _parse_positive_int(args.extract_plain_text(), "验证码长度")
            await store.set_setting("code_length", str(code_length))
            await _finish(SetCodeLengthCommand, f"已将验证码长度设置为 {code_length}。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetCodeLengthCommand, exc)

    @SetRejectOnLowLevelCommand.handle()
    async def handle_set_reject_on_low_level(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新等级不足时是否自动拒绝申请。

        Args:
            event: 触发命令的消息事件。
            args: 开关参数，只支持开或关。
        """

        try:
            await _ensure_admin_role(event)
            enabled = _parse_toggle_value(args.extract_plain_text())
            await store.set_setting("reject_on_low_level", "1" if enabled else "0")
            await _finish(
                SetRejectOnLowLevelCommand,
                f"已将等级不足自动拒绝设置为{'开启' if enabled else '关闭'}。",
            )
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetRejectOnLowLevelCommand, exc)

    @SetPromptCommand.handle()
    async def handle_set_prompt(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """更新指定类型的提示文案。

        Args:
            event: 触发命令的消息事件。
            args: 文案类型和内容参数。
        """

        try:
            await _ensure_admin_role(event)
            setting_key, prompt_text = _parse_prompt_args(args.extract_plain_text())
            await store.set_setting(setting_key, prompt_text)
            await _finish(SetPromptCommand, "已更新入群守卫提示文案。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(SetPromptCommand, exc)

    @ResetPromptCommand.handle()
    async def handle_reset_prompt(event: MessageEvent, args=CommandArg()) -> None:  # noqa: ANN001
        """将指定类型的提示文案恢复为默认值。

        Args:
            event: 触发命令的消息事件。
            args: 需要重置的文案类型。
        """

        try:
            await _ensure_admin_role(event)
            setting_key, display_name = _parse_prompt_type(args.extract_plain_text())
            await store.reset_setting(setting_key)
            await _finish(ResetPromptCommand, f"已将{display_name}恢复为默认文案。")
        except (PermissionError, ValueError) as exc:
            await finish_with_error(ResetPromptCommand, exc)


def register_help_items() -> None:
    """向帮助系统注册本插件命令说明。"""

    registry = get_help_registry()
    if registry.find_any_spec("入群守卫状态") is not None:
        return

    register_help(
        HelpCommandSpec(
            name="入群守卫状态",
            category="群管理",
            summary="查看当前入群守卫的目标群和核心配置。",
            description="管理员可查看当前数据库路径、目标群列表、等级门槛和各类时限配置。",
            examples=("/入群守卫状态",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=100,
        )
    )
    register_help(
        HelpCommandSpec(
            name="开启入群守卫",
            category="群管理",
            summary="为指定群开启入群等级和验证码校验。",
            description="群聊中可省略群号，默认作用于当前群；私聊中需显式提供群号。",
            arguments=(
                HelpArgumentSpec(
                    name="[群号]",
                    description="可选的目标群号；在群聊中留空时默认使用当前群。",
                    required=False,
                    value_hint="整数群号",
                    example="123456789",
                ),
            ),
            examples=("/开启入群守卫", "/开启入群守卫 123456789"),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=110,
        )
    )
    register_help(
        HelpCommandSpec(
            name="关闭入群守卫",
            category="群管理",
            summary="关闭指定群的入群等级和验证码校验。",
            description="关闭后会同时撤销该群当前尚未完成的验证码超时任务。",
            arguments=(
                HelpArgumentSpec(
                    name="[群号]",
                    description="可选的目标群号；在群聊中留空时默认使用当前群。",
                    required=False,
                    value_hint="整数群号",
                    example="123456789",
                ),
            ),
            examples=("/关闭入群守卫", "/关闭入群守卫 123456789"),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=120,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置入群等级",
            category="群管理",
            summary="设置自动通过申请所需的最低 QQ 等级。",
            description="低于该等级的加群申请会按当前策略自动拒绝或忽略。",
            arguments=(
                HelpArgumentSpec(
                    name="<等级>",
                    description="最低 QQ 等级门槛。",
                    value_hint="正整数",
                    example="16",
                ),
            ),
            examples=("/设置入群等级 16",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=130,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置申请处理时限",
            category="群管理",
            summary="设置申请事件允许自动处理的最大秒数。",
            description="超过这个时长的旧申请将被直接忽略，不再自动审批。",
            arguments=(
                HelpArgumentSpec(
                    name="<秒数>",
                    description="申请有效处理窗口，单位为秒。",
                    value_hint="正整数",
                    example="120",
                ),
            ),
            examples=("/设置申请处理时限 120",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=140,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置验证码时限",
            category="群管理",
            summary="设置新成员完成验证码的最长期限。",
            description="成员在该时限内未发送正确验证码时，会按配置自动踢出。",
            arguments=(
                HelpArgumentSpec(
                    name="<秒数>",
                    description="验证码有效时间，单位为秒。",
                    value_hint="正整数",
                    example="300",
                ),
            ),
            examples=("/设置验证码时限 300",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=150,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置验证码长度",
            category="群管理",
            summary="设置动态验证码的位数。",
            description="验证码仅由数字组成，长度越长越不容易被误碰撞。",
            arguments=(
                HelpArgumentSpec(
                    name="<长度>",
                    description="验证码位数。",
                    value_hint="正整数",
                    example="4",
                ),
            ),
            examples=("/设置验证码长度 4",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=160,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置等级不足自动拒绝",
            category="群管理",
            summary="设置等级不足时是否直接拒绝加群申请。",
            description="开启后会直接拒绝等级不足的申请；关闭后会仅忽略，不自动同意也不自动拒绝。",
            arguments=(
                HelpArgumentSpec(
                    name="<开关>",
                    description="只支持 开 或 关。",
                    value_hint="开|关",
                    example="开",
                ),
            ),
            examples=("/设置等级不足自动拒绝 开", "/设置等级不足自动拒绝 关"),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=165,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置入群守卫提示",
            category="群管理",
            summary="修改入群守卫的提示文案。",
            description=(
                "支持修改欢迎、提醒、成功、超时、拒绝五类文案。"
                "可用占位符包括 {user_at}、{timeout_seconds}、{code}。"
            ),
            arguments=(
                HelpArgumentSpec(
                    name="<类型>",
                    description="文案类型，只支持 欢迎、提醒、成功、超时、拒绝。",
                    value_hint="固定类型名",
                    example="欢迎",
                ),
                HelpArgumentSpec(
                    name="<内容>",
                    description="新的提示文案内容。",
                    value_hint="文本",
                    example="{user_at} 请在 {timeout_seconds} 秒内直接发送图片中的验证码。",
                ),
            ),
            examples=(
                "/设置入群守卫提示 欢迎 {user_at} 请在 {timeout_seconds} 秒内直接发送图片中的验证码。",
                "/设置入群守卫提示 成功 {user_at} 验证码正确，欢迎入群。",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=170,
        )
    )
    register_help(
        HelpCommandSpec(
            name="重置入群守卫提示",
            category="群管理",
            summary="将指定类型的提示文案恢复为默认值。",
            description="支持重置欢迎、提醒、成功、超时、拒绝五类文案。",
            arguments=(
                HelpArgumentSpec(
                    name="<类型>",
                    description="文案类型，只支持 欢迎、提醒、成功、超时、拒绝。",
                    value_hint="固定类型名",
                    example="欢迎",
                ),
            ),
            examples=("/重置入群守卫提示 欢迎",),
            required_role=PermissionRole.ADMIN,
            audience="群管理",
            sort_key=180,
        )
    )


async def safe_register() -> None:
    """确保数据库就绪并完成命令与帮助注册。"""

    await store.ensure_ready()
    register_commands()
    register_help_items()


__all__ = [
    "PermissionError",
    "finish_with_error",
    "safe_register",
]
