from __future__ import annotations

from nonebot import on_command
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from .services.help import (
    HelpArgumentSpec,
    HelpCommandSpec,
    get_help_registry,
    register_help,
    render_help_categories,
    render_help_category,
    render_help_detail,
)
from .services.permission import PermissionRole


HelpCommand = on_command("帮助", aliases={"菜单", "help"}, priority=4, block=True)

register_help(
    HelpCommandSpec(
        name="帮助",
        aliases=("菜单", "help"),
        category="帮助系统",
        summary="查看命令集合、集合下命令或单条命令详情。",
        description="留空时优先显示你当前可见的命令集合；传入集合名可查看该集合下命令；传入命令名可直接查看完整详情。",
        arguments=(
            HelpArgumentSpec(
                name="[命令名]",
                description="可选的目标命令名或别名。",
                required=False,
                value_hint="命令名",
                example="查看权限",
            ),
        ),
        examples=(
            "/帮助",
            "/帮助 权限管理",
            "/帮助 查看权限",
        ),
        required_role=PermissionRole.USER,
        audience="群聊和私聊",
        sort_key=0,
    )
)


async def _send_chunks(event: MessageEvent, chunks: list[str]) -> None:
    """按顺序发送帮助文本分片。

    Args:
        event: 当前触发帮助命令的消息事件。
        chunks: 已按长度切分好的帮助文本片段。
    """
    if not chunks:
        await HelpCommand.finish("当前没有可展示的帮助内容。")

    for chunk in chunks[:-1]:
        await HelpCommand.send(chunk)
    await HelpCommand.finish(chunks[-1])


@HelpCommand.handle()
async def handle_help_command(
    event: MessageEvent,
    args: Message = CommandArg(),
) -> None:
    """处理 GTBot 统一帮助命令。

    空参数时优先返回当前用户可见的命令集合；传入集合名时，
    返回该集合下的命令和参数格式；传入命令名或别名时，
    返回该命令的完整说明、参数和示例。

    Args:
        event: 触发帮助命令的消息事件。
        args: 命令参数，允许为空或携带一个目标命令名。
    """
    registry = get_help_registry()
    arg_text = args.extract_plain_text().strip()
    user_id = int(event.user_id)

    if not arg_text:
        visible_categories = await registry.get_visible_categories(user_id)
        await _send_chunks(event, render_help_categories(visible_categories))

    matched_category = await registry.find_visible_category(arg_text, user_id)
    if matched_category is not None:
        category, specs = matched_category
        await _send_chunks(event, render_help_category(category, specs))

    spec = await registry.find_visible_spec(arg_text, user_id)
    if spec is not None:
        await _send_chunks(event, render_help_detail(spec))

    category_suggestions = await registry.suggest_visible_categories(arg_text, user_id)
    if category_suggestions:
        suggestion_text = "、".join(category_suggestions)
        await HelpCommand.finish(
            f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n你可以试试这些命令集合：{suggestion_text}"
        )

    suggestions = await registry.suggest_visible_commands(arg_text, user_id)
    if suggestions:
        suggestion_text = "、".join(f"/{name}" for name in suggestions)
        await HelpCommand.finish(
            f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n你可以试试这些命令：{suggestion_text}"
        )

    await HelpCommand.finish(
        f"未找到命令集合或命令“{arg_text}”，或你无权查看它。\n发送 /帮助 查看当前可用命令集合。"
    )
