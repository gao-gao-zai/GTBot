from __future__ import annotations

from nonebot import on_command
from nonebot.adapters.onebot.v11.event import GroupMessageEvent, MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from .services.trigger.keyword import (
    GroupKeywordTriggerListType,
    GroupKeywordTriggerMode,
    get_group_keyword_trigger_manager,
)
from local_plugins.nonebot_plugin_gt_help import HelpArgumentSpec, HelpCommandSpec, register_help
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, get_permission_manager


GroupKeywordTriggerInfoCommand = on_command("查看群聊关键词触发", priority=4, block=True)
GroupKeywordTriggerModeCommand = on_command("设置群聊关键词触发模式", priority=4, block=True)
GroupKeywordTriggerAddEntryCommand = on_command("添加群聊关键词触发名单", priority=4, block=True)
GroupKeywordTriggerRemoveEntryCommand = on_command("移除群聊关键词触发名单", priority=4, block=True)
GroupKeywordTriggerProbabilityCommand = on_command("设置群聊关键词触发概率", priority=4, block=True)
GroupKeywordTriggerKeywordInfoCommand = on_command("查看群聊关键词", priority=4, block=True)
GroupKeywordTriggerAddKeywordCommand = on_command("添加群聊关键词", priority=4, block=True)
GroupKeywordTriggerRemoveKeywordCommand = on_command("移除群聊关键词", priority=4, block=True)


def _register_group_keyword_trigger_help_items() -> None:
    """注册群聊关键词触发相关的帮助项。"""

    register_help(
        HelpCommandSpec(
            name="查看群聊关键词触发",
            category="群聊关键词触发",
            summary="查看全局或指定群的关键词触发配置。",
            description="管理员可查看群聊关键词触发的启用模式、黑白名单、默认概率以及指定群的生效配置。",
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
                "/查看群聊关键词触发",
                "/查看群聊关键词触发 当前",
                "/查看群聊关键词触发 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置群聊关键词触发模式",
            category="群聊关键词触发",
            summary="设置群聊关键词触发的全局模式。",
            description="可将群聊关键词触发设为关闭、黑名单模式或白名单模式，决定哪些群允许通过关键词触发聊天。",
            arguments=(
                HelpArgumentSpec(
                    name="<关闭|黑名单|白名单>",
                    description="要设置的全局模式。",
                    value_hint="关闭 / 黑名单 / 白名单",
                    example="白名单",
                ),
            ),
            examples=(
                "/设置群聊关键词触发模式 关闭",
                "/设置群聊关键词触发模式 白名单",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name="添加群聊关键词触发名单",
            category="群聊关键词触发",
            summary="向黑名单或白名单中添加群号。",
            description="管理员可将某个群加入群聊关键词触发黑名单或白名单，用于控制关键词触发适用范围。",
            arguments=(
                HelpArgumentSpec(
                    name="<黑名单|白名单>",
                    description="目标名单类型。",
                    value_hint="黑名单 / 白名单",
                    example="白名单",
                ),
                HelpArgumentSpec(
                    name="<当前|群号>",
                    description="要加入名单的群，支持在群内直接使用“当前”。",
                    value_hint="当前 / 群号",
                    example="当前",
                ),
            ),
            examples=(
                "/添加群聊关键词触发名单 白名单 当前",
                "/添加群聊关键词触发名单 黑名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=30,
        )
    )
    register_help(
        HelpCommandSpec(
            name="移除群聊关键词触发名单",
            category="群聊关键词触发",
            summary="从黑名单或白名单中移除群号。",
            description="管理员可将某个群从群聊关键词触发黑名单或白名单中移除。",
            arguments=(
                HelpArgumentSpec(
                    name="<黑名单|白名单>",
                    description="目标名单类型。",
                    value_hint="黑名单 / 白名单",
                    example="黑名单",
                ),
                HelpArgumentSpec(
                    name="<当前|群号>",
                    description="要移除的群，支持在群内直接使用“当前”。",
                    value_hint="当前 / 群号",
                    example="123456789",
                ),
            ),
            examples=(
                "/移除群聊关键词触发名单 白名单 当前",
                "/移除群聊关键词触发名单 黑名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=40,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置群聊关键词触发概率",
            category="群聊关键词触发",
            summary="设置默认值或某个群的关键词触发概率。",
            description="概率按群生效；若某群未单独设置，则回退到默认值。默认概率初始为 100%。",
            arguments=(
                HelpArgumentSpec(
                    name="<默认|当前|群号>",
                    description="目标配置槽位，默认表示全局默认值。",
                    value_hint="默认 / 当前 / 群号",
                    example="默认",
                ),
                HelpArgumentSpec(
                    name="<概率>",
                    description="0 到 100 之间的数字，可带百分号。",
                    value_hint="0-100",
                    example="35",
                ),
            ),
            examples=(
                "/设置群聊关键词触发概率 默认 100",
                "/设置群聊关键词触发概率 当前 25%",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=50,
        )
    )
    register_help(
        HelpCommandSpec(
            name="查看群聊关键词",
            category="群聊关键词触发",
            summary="查看默认值或指定群的关键词列表。",
            description="支持查看默认关键词、群专属关键词，以及在群级查看时同时展示最终生效关键词。",
            arguments=(
                HelpArgumentSpec(
                    name="[默认|当前|群号]",
                    description="留空时查看默认关键词；在群内可用“当前”查看本群。",
                    required=False,
                    value_hint="默认 / 当前 / 群号",
                    example="当前",
                ),
            ),
            examples=(
                "/查看群聊关键词",
                "/查看群聊关键词 当前",
                "/查看群聊关键词 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=60,
        )
    )
    register_help(
        HelpCommandSpec(
            name="添加群聊关键词",
            category="群聊关键词触发",
            summary="向默认值或指定群添加关键词。",
            description="默认关键词会对所有群生效；群专属关键词会与默认关键词合并后作为该群最终可用关键词。",
            arguments=(
                HelpArgumentSpec(
                    name="<默认|当前|群号>",
                    description="目标配置槽位，支持默认值或某个群。",
                    value_hint="默认 / 当前 / 群号",
                    example="当前",
                ),
                HelpArgumentSpec(
                    name="<关键词>",
                    description="要添加的关键词，支持包含空格。",
                    value_hint="文本",
                    example="晚上好",
                ),
            ),
            examples=(
                "/添加群聊关键词 默认 你好",
                "/添加群聊关键词 当前 晚上好",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=70,
        )
    )
    register_help(
        HelpCommandSpec(
            name="移除群聊关键词",
            category="群聊关键词触发",
            summary="从默认值或指定群移除关键词。",
            description="可从默认关键词池或群专属关键词池中移除某个关键词。",
            arguments=(
                HelpArgumentSpec(
                    name="<默认|当前|群号>",
                    description="目标配置槽位，支持默认值或某个群。",
                    value_hint="默认 / 当前 / 群号",
                    example="默认",
                ),
                HelpArgumentSpec(
                    name="<关键词>",
                    description="要移除的关键词，支持包含空格。",
                    value_hint="文本",
                    example="你好",
                ),
            ),
            examples=(
                "/移除群聊关键词 默认 你好",
                "/移除群聊关键词 当前 晚上好",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊关键词触发管理命令",
            sort_key=80,
        )
    )


_register_group_keyword_trigger_help_items()


async def _ensure_admin(user_id: int) -> None:
    """确保当前命令调用者具备管理员权限。"""

    permission_manager = get_permission_manager()
    await permission_manager.require_role(user_id, PermissionRole.ADMIN)


def _normalize_mode_token(token: str) -> GroupKeywordTriggerMode:
    """把命令中的模式参数标准化为枚举值。"""

    normalized = str(token).strip().lower()
    if normalized in {"关闭", "关", "off"}:
        return GroupKeywordTriggerMode.OFF
    if normalized in {"黑名单", "黑", "black", "blacklist"}:
        return GroupKeywordTriggerMode.BLACKLIST
    if normalized in {"白名单", "白", "white", "whitelist"}:
        return GroupKeywordTriggerMode.WHITELIST
    raise ValueError(f"不支持的关键词触发模式: {token}")


def _normalize_list_type_token(token: str) -> GroupKeywordTriggerListType:
    """把命令中的名单类型参数标准化为枚举值。"""

    normalized = str(token).strip().lower()
    if normalized in {"黑名单", "黑", "black", "blacklist"}:
        return GroupKeywordTriggerListType.BLACKLIST
    if normalized in {"白名单", "白", "white", "whitelist"}:
        return GroupKeywordTriggerListType.WHITELIST
    raise ValueError(f"不支持的名单类型: {token}")


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


def _mode_label(mode: GroupKeywordTriggerMode) -> str:
    """返回模式的人类可读标签。"""

    labels = {
        GroupKeywordTriggerMode.OFF: "关闭",
        GroupKeywordTriggerMode.BLACKLIST: "黑名单",
        GroupKeywordTriggerMode.WHITELIST: "白名单",
    }
    return labels[mode]


def _list_type_label(list_type: GroupKeywordTriggerListType) -> str:
    """返回名单类型的人类可读标签。"""

    return "黑名单" if list_type == GroupKeywordTriggerListType.BLACKLIST else "白名单"


def _target_label(group_id: int | None) -> str:
    """将默认值或群号格式化为展示文本。"""

    return "默认值" if group_id is None else f"群 {int(group_id)}"


@GroupKeywordTriggerInfoCommand.handle()
async def handle_group_keyword_trigger_info(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """查看群聊关键词触发的全局概览或指定群详情。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    manager = get_group_keyword_trigger_manager()
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await GroupKeywordTriggerInfoCommand.finish(await manager.describe_overview())

    try:
        group_id = _normalize_group_target_token(arg_text, event=event, allow_default=False)
    except ValueError as exc:
        await GroupKeywordTriggerInfoCommand.finish(
            f"{exc}\n用法: /查看群聊关键词触发 [当前|群号]"
        )

    assert group_id is not None
    await GroupKeywordTriggerInfoCommand.finish(await manager.describe_group(group_id))


@GroupKeywordTriggerModeCommand.handle()
async def handle_set_group_keyword_trigger_mode(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """设置群聊关键词触发的全局模式。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await GroupKeywordTriggerModeCommand.finish(
            "用法: /设置群聊关键词触发模式 <关闭|黑名单|白名单>"
        )

    try:
        mode = _normalize_mode_token(arg_text)
    except ValueError as exc:
        await GroupKeywordTriggerModeCommand.finish(str(exc))

    manager = get_group_keyword_trigger_manager()
    await manager.set_mode(mode, int(event.user_id))
    await GroupKeywordTriggerModeCommand.finish(
        f"已将群聊关键词触发模式设置为 {_mode_label(mode)}"
    )


@GroupKeywordTriggerAddEntryCommand.handle()
async def handle_add_group_keyword_trigger_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """向群聊关键词触发黑白名单中添加群号。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await GroupKeywordTriggerAddEntryCommand.finish(
            "用法: /添加群聊关键词触发名单 <黑名单|白名单> <当前|群号>"
        )

    try:
        list_type = _normalize_list_type_token(parts[0])
        group_id = _normalize_group_target_token(parts[1], event=event, allow_default=False)
    except ValueError as exc:
        await GroupKeywordTriggerAddEntryCommand.finish(str(exc))

    assert group_id is not None
    manager = get_group_keyword_trigger_manager()
    created = await manager.add_entry(list_type, group_id, int(event.user_id))
    if not created:
        await GroupKeywordTriggerAddEntryCommand.finish(
            f"{_target_label(group_id)} 已存在于{_list_type_label(list_type)}"
        )

    await GroupKeywordTriggerAddEntryCommand.finish(
        f"已将{_target_label(group_id)}加入群聊关键词触发{_list_type_label(list_type)}"
    )


@GroupKeywordTriggerRemoveEntryCommand.handle()
async def handle_remove_group_keyword_trigger_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """从群聊关键词触发黑白名单中移除群号。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await GroupKeywordTriggerRemoveEntryCommand.finish(
            "用法: /移除群聊关键词触发名单 <黑名单|白名单> <当前|群号>"
        )

    try:
        list_type = _normalize_list_type_token(parts[0])
        group_id = _normalize_group_target_token(parts[1], event=event, allow_default=False)
    except ValueError as exc:
        await GroupKeywordTriggerRemoveEntryCommand.finish(str(exc))

    assert group_id is not None
    manager = get_group_keyword_trigger_manager()
    removed = await manager.remove_entry(list_type, group_id, int(event.user_id))
    if not removed:
        await GroupKeywordTriggerRemoveEntryCommand.finish(
            f"{_target_label(group_id)} 不在{_list_type_label(list_type)}中"
        )

    await GroupKeywordTriggerRemoveEntryCommand.finish(
        f"已将{_target_label(group_id)}从群聊关键词触发{_list_type_label(list_type)}中移除"
    )


@GroupKeywordTriggerProbabilityCommand.handle()
async def handle_set_group_keyword_trigger_probability(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """设置默认值或指定群的关键词触发概率。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await GroupKeywordTriggerProbabilityCommand.finish(
            "用法: /设置群聊关键词触发概率 <默认|当前|群号> <概率>"
        )

    try:
        group_id = _normalize_group_target_token(parts[0], event=event, allow_default=True)
        manager = get_group_keyword_trigger_manager()
        probability = await manager.set_probability(
            group_id=group_id,
            probability=parts[1],
            operator_user_id=int(event.user_id),
        )
    except ValueError as exc:
        await GroupKeywordTriggerProbabilityCommand.finish(str(exc))

    await GroupKeywordTriggerProbabilityCommand.finish(
        f"已将{_target_label(group_id)}的群聊关键词触发概率设置为 {probability:.2f}%"
    )


@GroupKeywordTriggerKeywordInfoCommand.handle()
async def handle_group_keyword_trigger_keyword_info(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """查看默认值或指定群的关键词配置。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    arg_text = args.extract_plain_text().strip()
    manager = get_group_keyword_trigger_manager()

    if not arg_text:
        default_keywords = await manager.list_keywords(None)
        await GroupKeywordTriggerKeywordInfoCommand.finish(
            f"默认关键词: {', '.join(default_keywords) if default_keywords else '(空)'}"
        )

    try:
        group_id = _normalize_group_target_token(arg_text, event=event, allow_default=True)
    except ValueError as exc:
        await GroupKeywordTriggerKeywordInfoCommand.finish(
            f"{exc}\n用法: /查看群聊关键词 [默认|当前|群号]"
        )

    if group_id is None:
        default_keywords = await manager.list_keywords(None)
        await GroupKeywordTriggerKeywordInfoCommand.finish(
            f"默认关键词: {', '.join(default_keywords) if default_keywords else '(空)'}"
        )

    group_keywords = await manager.list_keywords(group_id)
    effective_keywords = await manager.get_effective_keywords(group_id)
    lines = [
        f"{_target_label(group_id)}单独关键词: {', '.join(group_keywords) if group_keywords else '(空)'}",
        f"{_target_label(group_id)}生效关键词: {', '.join(effective_keywords) if effective_keywords else '(空)'}",
    ]
    await GroupKeywordTriggerKeywordInfoCommand.finish("\n".join(lines))


@GroupKeywordTriggerAddKeywordCommand.handle()
async def handle_add_group_keyword_trigger_keyword(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """向默认值或指定群添加关键词。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split(maxsplit=1)
    if len(parts) != 2:
        await GroupKeywordTriggerAddKeywordCommand.finish(
            "用法: /添加群聊关键词 <默认|当前|群号> <关键词>"
        )

    try:
        group_id = _normalize_group_target_token(parts[0], event=event, allow_default=True)
        manager = get_group_keyword_trigger_manager()
        created = await manager.add_keyword(
            group_id=group_id,
            keyword=parts[1],
            operator_user_id=int(event.user_id),
        )
    except ValueError as exc:
        await GroupKeywordTriggerAddKeywordCommand.finish(str(exc))

    if not created:
        await GroupKeywordTriggerAddKeywordCommand.finish(
            f"{_target_label(group_id)} 已存在关键词 {parts[1]!r}"
        )

    await GroupKeywordTriggerAddKeywordCommand.finish(
        f"已向{_target_label(group_id)}添加关键词 {parts[1]!r}"
    )


@GroupKeywordTriggerRemoveKeywordCommand.handle()
async def handle_remove_group_keyword_trigger_keyword(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """从默认值或指定群移除关键词。"""

    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError:
        return

    parts = args.extract_plain_text().strip().split(maxsplit=1)
    if len(parts) != 2:
        await GroupKeywordTriggerRemoveKeywordCommand.finish(
            "用法: /移除群聊关键词 <默认|当前|群号> <关键词>"
        )

    try:
        group_id = _normalize_group_target_token(parts[0], event=event, allow_default=True)
        manager = get_group_keyword_trigger_manager()
        removed = await manager.remove_keyword(
            group_id=group_id,
            keyword=parts[1],
            operator_user_id=int(event.user_id),
        )
    except ValueError as exc:
        await GroupKeywordTriggerRemoveKeywordCommand.finish(str(exc))

    if not removed:
        await GroupKeywordTriggerRemoveKeywordCommand.finish(
            f"{_target_label(group_id)} 不存在关键词 {parts[1]!r}"
        )

    await GroupKeywordTriggerRemoveKeywordCommand.finish(
        f"已从{_target_label(group_id)}移除关键词 {parts[1]!r}"
    )
