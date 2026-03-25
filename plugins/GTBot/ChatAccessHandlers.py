from __future__ import annotations

from nonebot import on_command
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from .ChatAccessManager import (
    ChatAccessListType,
    ChatAccessMode,
    ChatAccessScope,
    get_chat_access_manager,
)
from .HelpRegistry import HelpArgumentSpec, HelpCommandSpec, register_help
from .PermissionManager import PermissionError, PermissionRole, get_permission_manager

ChatAccessInfoCommand = on_command("查看会话权限", priority=4, block=True)
ChatAccessModeCommand = on_command("设置会话权限模式", priority=4, block=True)
ChatAccessAddCommand = on_command("添加会话名单", priority=4, block=True)
ChatAccessRemoveCommand = on_command("移除会话名单", priority=4, block=True)


def _register_chat_access_help_items() -> None:
    """注册会话权限相关核心命令的帮助信息。"""
    register_help(
        HelpCommandSpec(
            name="查看会话权限",
            category="会话权限",
            summary="查看群聊和私聊的会话权限配置。",
            description="管理员可查看群聊、私聊当前使用的准入模式，以及黑白名单内容。",
            arguments=(
                HelpArgumentSpec(
                    name="[群聊|私聊]",
                    description="可选的会话范围；留空时同时展示群聊和私聊配置。",
                    required=False,
                    value_hint="群聊 或 私聊",
                    example="群聊",
                ),
            ),
            examples=(
                "/查看会话权限",
                "/查看会话权限 群聊",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=10,
        )
    )
    register_help(
        HelpCommandSpec(
            name="设置会话权限模式",
            category="会话权限",
            summary="设置群聊或私聊的准入模式。",
            description="管理员可将群聊或私聊的会话权限模式设置为关闭、黑名单或白名单。",
            arguments=(
                HelpArgumentSpec(
                    name="<群聊|私聊>",
                    description="要修改的会话范围。",
                    value_hint="群聊 或 私聊",
                    example="群聊",
                ),
                HelpArgumentSpec(
                    name="<关闭|黑名单|白名单>",
                    description="目标准入模式。",
                    value_hint="关闭/黑名单/白名单",
                    example="白名单",
                ),
            ),
            examples=(
                "/设置会话权限模式 群聊 白名单",
                "/设置会话权限模式 私聊 关闭",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name="添加会话名单",
            category="会话权限",
            summary="向会话黑名单或白名单中添加目标。",
            description="管理员可把群号或私聊用户号加入指定范围的黑名单或白名单。",
            arguments=(
                HelpArgumentSpec(
                    name="<群聊|私聊>",
                    description="名单所属的会话范围。",
                    value_hint="群聊 或 私聊",
                    example="私聊",
                ),
                HelpArgumentSpec(
                    name="<黑名单|白名单>",
                    description="目标名单类型。",
                    value_hint="黑名单 或 白名单",
                    example="白名单",
                ),
                HelpArgumentSpec(
                    name="<目标ID>",
                    description="群号或私聊用户 QQ 号。",
                    value_hint="整数 ID",
                    example="123456789",
                ),
            ),
            examples=(
                "/添加会话名单 群聊 白名单 987654321",
                "/添加会话名单 私聊 黑名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=30,
        )
    )
    register_help(
        HelpCommandSpec(
            name="移除会话名单",
            category="会话权限",
            summary="从会话黑名单或白名单中移除目标。",
            description="管理员可把群号或私聊用户号从指定范围的黑名单或白名单中移除。",
            arguments=(
                HelpArgumentSpec(
                    name="<群聊|私聊>",
                    description="名单所属的会话范围。",
                    value_hint="群聊 或 私聊",
                    example="群聊",
                ),
                HelpArgumentSpec(
                    name="<黑名单|白名单>",
                    description="目标名单类型。",
                    value_hint="黑名单 或 白名单",
                    example="黑名单",
                ),
                HelpArgumentSpec(
                    name="<目标ID>",
                    description="要移除的群号或私聊用户 QQ 号。",
                    value_hint="整数 ID",
                    example="987654321",
                ),
            ),
            examples=(
                "/移除会话名单 群聊 黑名单 987654321",
                "/移除会话名单 私聊 白名单 123456789",
            ),
            required_role=PermissionRole.ADMIN,
            audience="群聊和私聊管理员命令",
            sort_key=40,
        )
    )


_register_chat_access_help_items()


async def _ensure_admin(user_id: int) -> None:
    """确保当前用户具有管理员权限。

    Args:
        user_id: 发起命令的用户 QQ 号。

    Raises:
        PermissionError: 当前用户不是管理员或所有者。
    """
    permission_manager = get_permission_manager()
    await permission_manager.require_role(user_id, PermissionRole.ADMIN)


def _normalize_scope_token(token: str) -> ChatAccessScope:
    """将命令里的中文/英文范围参数标准化为枚举值。

    Args:
        token: 命令参数里的范围描述。

    Returns:
        ChatAccessScope: 归一化后的会话范围枚举。

    Raises:
        ValueError: 参数不是支持的群聊或私聊标识。
    """
    normalized = str(token).strip().lower()
    if normalized in {"群聊", "群", "group", "g"}:
        return ChatAccessScope.GROUP
    if normalized in {"私聊", "私", "private", "p"}:
        return ChatAccessScope.PRIVATE
    raise ValueError(f"不支持的会话范围: {token}")


def _normalize_mode_token(token: str) -> ChatAccessMode:
    """将命令里的模式参数标准化为枚举值。

    Args:
        token: 命令参数里的模式描述。

    Returns:
        ChatAccessMode: 归一化后的准入模式枚举。

    Raises:
        ValueError: 参数不是支持的模式标识。
    """
    normalized = str(token).strip().lower()
    if normalized in {"关闭", "关", "off"}:
        return ChatAccessMode.OFF
    if normalized in {"黑名单", "黑", "black", "blacklist"}:
        return ChatAccessMode.BLACKLIST
    if normalized in {"白名单", "白", "white", "whitelist"}:
        return ChatAccessMode.WHITELIST
    raise ValueError(f"不支持的权限模式: {token}")


def _normalize_list_type_token(token: str) -> ChatAccessListType:
    """将命令里的名单类型参数标准化为枚举值。

    Args:
        token: 命令参数里的名单类型描述。

    Returns:
        ChatAccessListType: 归一化后的名单类型枚举。

    Raises:
        ValueError: 参数不是支持的黑白名单标识。
    """
    normalized = str(token).strip().lower()
    if normalized in {"黑名单", "黑", "black", "blacklist"}:
        return ChatAccessListType.BLACKLIST
    if normalized in {"白名单", "白", "white", "whitelist"}:
        return ChatAccessListType.WHITELIST
    raise ValueError(f"不支持的名单类型: {token}")


def _scope_label(scope: ChatAccessScope) -> str:
    """返回会话范围的人类可读名称。

    Args:
        scope: 会话范围枚举。

    Returns:
        str: 中文显示名称。
    """
    return "群聊" if scope == ChatAccessScope.GROUP else "私聊"


def _list_type_label(list_type: ChatAccessListType) -> str:
    """返回名单类型的人类可读名称。

    Args:
        list_type: 名单类型枚举。

    Returns:
        str: 中文显示名称。
    """
    return "黑名单" if list_type == ChatAccessListType.BLACKLIST else "白名单"


def _mode_label(mode: ChatAccessMode) -> str:
    """返回准入模式的人类可读名称。

    Args:
        mode: 准入模式枚举。

    Returns:
        str: 中文显示名称。
    """
    labels = {
        ChatAccessMode.OFF: "关闭",
        ChatAccessMode.BLACKLIST: "黑名单",
        ChatAccessMode.WHITELIST: "白名单",
    }
    return labels[mode]


@ChatAccessInfoCommand.handle()
async def handle_chat_access_info(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """展示群聊或私聊当前的准入配置。

    Args:
        event: 触发命令的消息事件。
        args: 命令参数，支持留空或指定 `群聊/私聊`。
    """
    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError as exc:
        await ChatAccessInfoCommand.finish(str(exc))

    arg_text = args.extract_plain_text().strip()
    manager = get_chat_access_manager()

    if not arg_text:
        group_text = await manager.describe_scope(ChatAccessScope.GROUP)
        private_text = await manager.describe_scope(ChatAccessScope.PRIVATE)
        await ChatAccessInfoCommand.finish(f"{group_text}\n\n{private_text}")

    try:
        scope = _normalize_scope_token(arg_text)
    except ValueError as exc:
        await ChatAccessInfoCommand.finish(
            f"{exc}\n用法: /查看会话权限 [群聊|私聊]"
        )

    await ChatAccessInfoCommand.finish(await manager.describe_scope(scope))


@ChatAccessModeCommand.handle()
async def handle_set_chat_access_mode(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """设置群聊或私聊的准入模式。

    Args:
        event: 触发命令的消息事件。
        args: 命令参数，格式为 `<群聊|私聊> <关闭|黑名单|白名单>`。
    """
    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError as exc:
        await ChatAccessModeCommand.finish(str(exc))

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 2:
        await ChatAccessModeCommand.finish("用法: /设置会话权限模式 <群聊|私聊> <关闭|黑名单|白名单>")

    try:
        scope = _normalize_scope_token(parts[0])
        mode = _normalize_mode_token(parts[1])
    except ValueError as exc:
        await ChatAccessModeCommand.finish(str(exc))

    manager = get_chat_access_manager()
    await manager.set_mode(scope, mode, int(event.user_id))
    await ChatAccessModeCommand.finish(
        f"已将{_scope_label(scope)}权限模式设置为 {_mode_label(mode)}"
    )


@ChatAccessAddCommand.handle()
async def handle_add_chat_access_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """向群聊或私聊黑白名单添加目标 ID。

    Args:
        event: 触发命令的消息事件。
        args: 命令参数，格式为 `<群聊|私聊> <黑名单|白名单> <目标ID>`。
    """
    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError as exc:
        await ChatAccessAddCommand.finish(str(exc))

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 3:
        await ChatAccessAddCommand.finish("用法: /添加会话名单 <群聊|私聊> <黑名单|白名单> <目标ID>")

    try:
        scope = _normalize_scope_token(parts[0])
        list_type = _normalize_list_type_token(parts[1])
        target_id = int(parts[2])
    except ValueError as exc:
        await ChatAccessAddCommand.finish(str(exc))

    manager = get_chat_access_manager()
    created = await manager.add_entry(scope, list_type, target_id, int(event.user_id))
    if not created:
        await ChatAccessAddCommand.finish(
            f"{_scope_label(scope)} {_list_type_label(list_type)}中已存在 {target_id}"
        )

    await ChatAccessAddCommand.finish(
        f"已将 {target_id} 加入{_scope_label(scope)} {_list_type_label(list_type)}"
    )


@ChatAccessRemoveCommand.handle()
async def handle_remove_chat_access_entry(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """从群聊或私聊黑白名单中移除目标 ID。

    Args:
        event: 触发命令的消息事件。
        args: 命令参数，格式为 `<群聊|私聊> <黑名单|白名单> <目标ID>`。
    """
    try:
        await _ensure_admin(int(event.user_id))
    except PermissionError as exc:
        await ChatAccessRemoveCommand.finish(str(exc))

    parts = args.extract_plain_text().strip().split()
    if len(parts) != 3:
        await ChatAccessRemoveCommand.finish("用法: /移除会话名单 <群聊|私聊> <黑名单|白名单> <目标ID>")

    try:
        scope = _normalize_scope_token(parts[0])
        list_type = _normalize_list_type_token(parts[1])
        target_id = int(parts[2])
    except ValueError as exc:
        await ChatAccessRemoveCommand.finish(str(exc))

    manager = get_chat_access_manager()
    removed = await manager.remove_entry(scope, list_type, target_id, int(event.user_id))
    if not removed:
        await ChatAccessRemoveCommand.finish(
            f"{_scope_label(scope)} {_list_type_label(list_type)}中不存在 {target_id}"
        )

    await ChatAccessRemoveCommand.finish(
        f"已将 {target_id} 从{_scope_label(scope)} {_list_type_label(list_type)}移除"
    )
