from __future__ import annotations

from nonebot import on_command
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from .services.trigger.auto import get_group_auto_trigger_manager
from local_plugins.nonebot_plugin_gt_help import HelpArgumentSpec, HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, get_permission_manager


GroupAutoTriggerInfoCommand = on_command("查看群聊自动触发", priority=4, block=True)
GroupAutoTriggerAddEntryCommand = on_command("添加群聊自动触发名单", priority=4, block=True)
GroupAutoTriggerRemoveEntryCommand = on_command("移除群聊自动触发名单", priority=4, block=True)
GroupAutoTriggerProbabilityCommand = on_command("设置群聊自动触发概率", priority=4, block=True)
GroupAutoTriggerCooldownCommand = on_command("设置群聊自动触发冷却", priority=4, block=True)


def _register_group_auto_trigger_help_items() -> None:
    """注册群聊自动触发相关的帮助项。"""

    register_help(
        HelpCommandSpec(
            name="查看群聊自动触发",
            category="群聊自动触发",
            summary="查看全局或指定群的自动触发配置。",
            description="管理员可查看自动触发白名单、默认概率、默认冷却以及指定群的生效配置。",
            arguments=(
                HelpArgumentSpec(
                    name="[当前|群号]",
                    description="留空时查看全局概览；填写当前或群号时查看指定群。",
                    required=False,
                    value_hint="当前 / 群号",
                    example="当前",
                ),
            ),
            examples=(
                "/查看群聊自动触发",
                "/查看群聊自动触发 当前",
                "/查看群聊自动触发 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊自动触发管理命令",
            sort_key=110,
        )
    )
    register_help(
        HelpCommandSpec(
            name="添加群聊自动触发名单",
            category="群聊自动触发",
            summary="将群加入自动触发白名单。",
            description="默认所有群关闭自动触发；只有加入白名单的群才会参与定时扫描与自动触发。",
            arguments=(
                HelpArgumentSpec(
                    name="<当前|群号>",
                    description="要加入自动触发白名单的群。",
                    value_hint="当前 / 群号",
                    example="当前",
                ),
            ),
            examples=(
                "/添加群聊自动触发名单 当前",
                "/添加群聊自动触发名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊自动触发管理命令",
            sort_key=120,
        )
    )
    register_help(
        HelpCommandSpec(
            name="移除群聊自动触发名单",
            category="群聊自动触发",
            summary="将群从自动触发白名单中移除。",
            description="被移除的群将不再参与定时扫描与自动触发。",
            arguments=(
                HelpArgumentSpec(
                    name="<当前|群号>",
                    description="要移除的目标群。",
                    value_hint="当前 / 群号",
                    example="123456789",
                ),
            ),
            examples=(
                "/移除群聊自动触发名单 当前",
                "/移除群聊自动触发名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊自动触发管理命令",
            sort_key=130,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置群聊自动触发概率",
            category="群聊自动触发",
            summary="设置默认值或指定群的自动触发概率。",
            description="概率按群生效；若某群未单独设置，则回退到默认值。默认概率初始为 100%。",
            arguments=(
                HelpArgumentSpec(
                    name="<默认|当前|群号>",
                    description="目标配置槽位。",
                    value_hint="默认 / 当前 / 群号",
                    example="默认",
                ),
                HelpArgumentSpec(
                    name="<概率>",
                    description="0 到 100 之间的数字，可带百分号。",
                    value_hint="0-100",
                    example="25%",
                ),
            ),
            examples=(
                "/设置群聊自动触发概率 默认 100",
                "/设置群聊自动触发概率 当前 35",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊自动触发管理命令",
            sort_key=140,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置群聊自动触发冷却",
            category="群聊自动触发",
            summary="设置默认值或指定群的自动触发冷却时间。",
            description="冷却按群生效；若某群未单独设置，则回退到默认值。默认冷却初始为 3600 秒。",
            arguments=(
                HelpArgumentSpec(
                    name="<默认|当前|群号>",
                    description="目标配置槽位。",
                    value_hint="默认 / 当前 / 群号",
                    example="当前",
                ),
                HelpArgumentSpec(
                    name="<秒数>",
                    description="冷却时间，单位秒，可为 0。",
                    value_hint="秒",
                    example="1800",
                ),
            ),
            examples=(
                "/设置群聊自动触发冷却 默认 3600",
                "/设置群聊自动触发冷却 当前 600",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊自动触发管理命令",
            sort_key=150,
        )
    )


_register_group_auto_trigger_help_items()


async def _ensure_admin(user_id: int) -> None:
    """确保当前用户具备管理员权限。"""

    permission_manager = get_permission_manager()
    await permission_manager.require_role(int(user_id), PermissionRole.ADMIN)


def _normalize_group_target_token(
    token: str,
    *,
    event: MessageEvent,
    allow_default: bool,
) -> int | None:
    """把命令中的“默认/当前/群号”参数解析为目标群。"""

    normalized = str(token).strip().lower()
    if allow_default and normalized in {"默认", "default"}:
        return None
    if normalized in {"当前", "本群", "current"}:
        if not isinstance(event, GroupMessageEvent):
            raise ValueError("“当前”只支持在群聊中使用")
        return int(event.group_id)
    try:
        group_id = int(str(token).strip())
    except ValueError as exc:
        raise ValueError(f"无效的群号: {token}") from exc
    if group_id <= 0:
        raise ValueError(f"无效的群号: {token}")
    return group_id


def _target_label(group_id: int | None) -> str:
    """将默认值或群号格式化为展示文本。"""

    return "默认值" if group_id is None else f"群 {int(group_id)}"


@GroupAutoTriggerInfoCommand.handle()
async def handle_group_auto_trigger_info(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """查看群聊自动触发的全局概览或指定群详情。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    manager = get_group_auto_trigger_manager()
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await GroupAutoTriggerInfoCommand.finish(await manager.describe_overview())

    try:
        group_id = _normalize_group_target_token(arg_text, event=event, allow_default=False)
    except ValueError as exc:
        await GroupAutoTriggerInfoCommand.finish(
            f"{exc}\n用法: /查看群聊自动触发 [当前|群号]"
        )

    assert group_id is not None
    await GroupAutoTriggerInfoCommand.finish(await manager.describe_group(group_id))


@GroupAutoTriggerAddEntryCommand.handle()
async def handle_add_group_auto_trigger_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """将群加入自动触发白名单。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await GroupAutoTriggerAddEntryCommand.finish(
            "用法: /添加群聊自动触发名单 <当前|群号>"
        )

    try:
        group_id = _normalize_group_target_token(arg_text, event=event, allow_default=False)
    except ValueError as exc:
        await GroupAutoTriggerAddEntryCommand.finish(str(exc))

    assert group_id is not None
    manager = get_group_auto_trigger_manager()
    created = await manager.add_entry(group_id=group_id, operator_user_id=int(event.user_id))
    if not created:
        await GroupAutoTriggerAddEntryCommand.finish(f"{_target_label(group_id)} 已在自动触发白名单中")

    await GroupAutoTriggerAddEntryCommand.finish(
        f"已将{_target_label(group_id)}加入群聊自动触发白名单"
    )


@GroupAutoTriggerRemoveEntryCommand.handle()
async def handle_remove_group_auto_trigger_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """将群从自动触发白名单中移除。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await GroupAutoTriggerRemoveEntryCommand.finish(
            "用法: /移除群聊自动触发名单 <当前|群号>"
        )

    try:
        group_id = _normalize_group_target_token(arg_text, event=event, allow_default=False)
    except ValueError as exc:
        await GroupAutoTriggerRemoveEntryCommand.finish(str(exc))

    assert group_id is not None
    manager = get_group_auto_trigger_manager()
    removed = await manager.remove_entry(group_id=group_id, operator_user_id=int(event.user_id))
    if not removed:
        await GroupAutoTriggerRemoveEntryCommand.finish(f"{_target_label(group_id)} 不在自动触发白名单中")

    await GroupAutoTriggerRemoveEntryCommand.finish(
        f"已将{_target_label(group_id)}从群聊自动触发白名单中移除"
    )


@GroupAutoTriggerProbabilityCommand.handle()
async def handle_set_group_auto_trigger_probability(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """设置默认值或指定群的自动触发概率。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await GroupAutoTriggerProbabilityCommand.finish(
            "用法: /设置群聊自动触发概率 <默认|当前|群号> <概率>"
        )

    try:
        group_id = _normalize_group_target_token(parts[0], event=event, allow_default=True)
        manager = get_group_auto_trigger_manager()
        probability = await manager.set_probability(
            group_id=group_id,
            probability=parts[1],
            operator_user_id=int(event.user_id),
        )
    except ValueError as exc:
        await GroupAutoTriggerProbabilityCommand.finish(str(exc))

    await GroupAutoTriggerProbabilityCommand.finish(
        f"已将{_target_label(group_id)}的群聊自动触发概率设置为 {probability:.2f}%"
    )


@GroupAutoTriggerCooldownCommand.handle()
async def handle_set_group_auto_trigger_cooldown(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """设置默认值或指定群的自动触发冷却时间。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await GroupAutoTriggerCooldownCommand.finish(
            "用法: /设置群聊自动触发冷却 <默认|当前|群号> <秒数>"
        )

    try:
        group_id = _normalize_group_target_token(parts[0], event=event, allow_default=True)
        manager = get_group_auto_trigger_manager()
        cooldown_seconds = await manager.set_cooldown_seconds(
            group_id=group_id,
            cooldown_seconds=parts[1],
            operator_user_id=int(event.user_id),
        )
    except ValueError as exc:
        await GroupAutoTriggerCooldownCommand.finish(str(exc))

    await GroupAutoTriggerCooldownCommand.finish(
        f"已将{_target_label(group_id)}的群聊自动触发冷却设置为 {cooldown_seconds:.2f} 秒"
    )
