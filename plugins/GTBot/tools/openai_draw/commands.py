from __future__ import annotations
from typing import Any

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent
from nonebot.adapters.onebot.v11.message import Message
from nonebot.params import CommandArg

from plugins.GTBot.services.help import HelpArgumentSpec, HelpCommandSpec, register_help
from plugins.GTBot.services.cache import get_user_cache_manager
from plugins.GTBot.services.message import get_message_manager
from local_plugins.nonebot_plugin_gt_permission import PermissionError, PermissionRole, require_admin
from .config import get_openai_draw_plugin_config
from .manager import OpenAIDrawJobSpec, get_openai_draw_queue_manager
from .tool import _resolve_input_image


def _help_role_from_rule(rule: str) -> PermissionRole:
    """将命令权限规则映射为帮助系统使用的角色等级。

    Args:
        rule: 配置中的命令权限规则。

    Returns:
        帮助系统可消费的最低权限等级。
    """

    return PermissionRole.USER if rule == "all" else PermissionRole.ADMIN


async def _ensure_permission(rule: str, user_id: int) -> None:
    """按配置校验命令调用权限。

    Args:
        rule: 配置中的权限规则。
        user_id: 当前用户 QQ 号。

    Raises:
        PermissionError: 当用户不满足权限要求时抛出。
    """

    if rule == "all":
        return
    await require_admin(int(user_id))


cfg = get_openai_draw_plugin_config()
DrawCommand = on_command(cfg.command_prefix, priority=4, block=True)
EditCommand = on_command(f"{cfg.command_prefix}编辑", aliases={"改图"}, priority=4, block=True)
DrawTasksCommand = on_command(f"{cfg.command_prefix}任务", aliases={"draw_tasks"}, priority=4, block=True)


def _register_help_items() -> None:
    """注册绘图插件命令到帮助系统。"""

    current_cfg = get_openai_draw_plugin_config()
    register_help(
        HelpCommandSpec(
            name=current_cfg.command_prefix,
            category="绘图服务",
            summary="手动启动一条 OpenAI 文生图异步任务。",
            description="命令会立即返回任务是否启动成功。实际出图在后台异步执行，不会阻塞当前会话；图片完成后会再通过消息链路发送。",
            arguments=(
                HelpArgumentSpec(
                    name="<提示词>",
                    description="要生成图片的文本提示词。",
                    value_hint="自然语言描述",
                    example="一只戴围巾的橘猫，站在雨夜霓虹街头",
                ),
            ),
            examples=(f"/{current_cfg.command_prefix} 一只戴围巾的橘猫，站在雨夜霓虹街头",),
            required_role=_help_role_from_rule(current_cfg.permissions.submit),
            audience="群聊和私聊",
            sort_key=20,
        )
    )
    register_help(
        HelpCommandSpec(
            name=f"{current_cfg.command_prefix}编辑",
            aliases=("改图",),
            category="绘图服务",
            summary="基于显式图片参数启动一条 OpenAI 编辑图异步任务。",
            description="需通过一个或多个 `--image` 提供原图，并可选通过 `--mask` 提供遮罩图。命令只返回任务是否启动成功；实际改图在后台异步执行。",
            arguments=(
                HelpArgumentSpec(
                    name="--image",
                    description="必填原图，可重复传入多次；支持本地路径、URL 或 OneBot 图片引用名。",
                    value_hint="图片引用",
                    example="C:/images/source.png",
                ),
                HelpArgumentSpec(
                    name="--mask",
                    description="可选遮罩图，可为本地路径、URL 或 OneBot 图片引用名。",
                    value_hint="图片引用",
                    example="C:/images/mask.png",
                ),
                HelpArgumentSpec(
                    name="<提示词>",
                    description="描述希望如何修改图片。",
                    value_hint="自然语言描述",
                    example="保留主体不变，把背景改成雪山日落",
                ),
            ),
            examples=(f"/{current_cfg.command_prefix}编辑 --image C:/images/source.png --image C:/images/style.png --mask C:/images/mask.png 保留主体不变，把背景改成雪山日落",),
            required_role=_help_role_from_rule(current_cfg.permissions.submit),
            audience="群聊和私聊",
            sort_key=21,
        )
    )
    register_help(
        HelpCommandSpec(
            name=f"{current_cfg.command_prefix}任务",
            aliases=("draw_tasks",),
            category="绘图服务",
            summary="查看当前绘图队列中的运行中和排队任务。",
            description="展示当前后台绘图队列的运行数、排队数以及最近任务摘要，便于管理员或用户确认任务状态。",
            examples=(f"/{current_cfg.command_prefix}任务",),
            required_role=_help_role_from_rule(current_cfg.permissions.query),
            audience="群聊和私聊",
            sort_key=22,
        )
    )


_register_help_items()


def _parse_edit_command_args(text: str) -> tuple[list[str], str | None, str]:
    """解析编辑图命令中的图片参数和提示词。

    命令格式约定为 `--image <原图1> [--image <原图2> ...] [--mask <遮罩图>] <提示词>`。
    该解析器允许重复传入 `--image`，以便显式构造多图编辑请求。

    Args:
        text: 命令参数原始文本。

    Returns:
        原图引用列表、遮罩图引用和提示词。

    Raises:
        ValueError: 当缺少必填图片参数、参数值不完整或提示词为空时抛出。
    """

    tokens = str(text or "").split()
    images: list[str] = []
    mask: str | None = None
    prompt_tokens: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--image":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("--image 缺少图片引用")
            images.append(tokens[idx])
        elif token == "--mask":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("--mask 缺少图片引用")
            mask = tokens[idx]
        else:
            prompt_tokens.append(token)
        idx += 1

    prompt = " ".join(prompt_tokens).strip()
    if not images:
        raise ValueError("缺少必填参数 --image")
    if not prompt:
        raise ValueError("缺少编辑提示词")
    return images, mask, prompt


@DrawCommand.handle()
async def handle_draw_command(bot: Bot, event: MessageEvent, args: Message = CommandArg()) -> None:
    """处理手动绘图命令。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前消息事件。
        args: 命令后的参数消息。
    """

    current_cfg = get_openai_draw_plugin_config()
    try:
        await _ensure_permission(current_cfg.permissions.submit, int(event.user_id))
    except PermissionError:
        await DrawCommand.finish("你没有提交绘图任务的权限。")

    prompt = args.extract_plain_text().strip()
    if not prompt:
        await DrawCommand.finish(f"用法: /{current_cfg.command_prefix} <提示词>")

    chat_type = "private" if getattr(event, "group_id", None) is None else "group"
    requester_user_id = int(event.user_id)
    group_id = getattr(event, "group_id", None)
    group_id_int = int(group_id) if group_id is not None else None
    session_id = f"private:{requester_user_id}" if chat_type == "private" else f"group:{group_id_int}"

    spec = OpenAIDrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=prompt,
        size=current_cfg.default_size,
        quality=current_cfg.default_quality,
        background=current_cfg.default_background,
        output_format=str(current_cfg.default_output_format),
        group_id=group_id_int,
        requester_user_id=requester_user_id,
        target_user_id=requester_user_id,
        bot=bot,
        message_manager=await get_message_manager(),
        cache=await get_user_cache_manager(),
    )

    manager = get_openai_draw_queue_manager()
    try:
        state = await manager.submit(spec)
        snapshot = await manager.snapshot()
    except RuntimeError as exc:
        await DrawCommand.finish(str(exc))

    await DrawCommand.finish(
        f"已启动异步绘图任务 job={state.job_id} running={snapshot['running_count']} queued={snapshot['queued_count']} 图片将由后台任务完成后另行发送。"
    )


@EditCommand.handle()
async def handle_edit_command(bot: Bot, event: MessageEvent, args: Message = CommandArg()) -> None:
    """处理手动编辑图命令。

    当前命令不再从消息内容中自动提图，而是要求调用方显式提供一个或多个
    `--image` 参数，以及可选的 `--mask` 参数，使命令行为与 Agent tool 保持一致。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前消息事件。
        args: 命令后的参数消息。
    """

    current_cfg = get_openai_draw_plugin_config()
    try:
        await _ensure_permission(current_cfg.permissions.submit, int(event.user_id))
    except PermissionError:
        await EditCommand.finish("你没有提交改图任务的权限。")

    args_text = args.extract_plain_text().strip()
    try:
        images, mask, prompt = _parse_edit_command_args(args_text)
    except ValueError as exc:
        await EditCommand.finish(
            f"{exc!s}。用法: /{current_cfg.command_prefix}编辑 --image <图片> [--image <图片> ...] [--mask <图片>] <提示词>"
        )

    chat_type = "private" if getattr(event, "group_id", None) is None else "group"
    requester_user_id = int(event.user_id)
    group_id = getattr(event, "group_id", None)
    group_id_int = int(group_id) if group_id is not None else None
    session_id = f"private:{requester_user_id}" if chat_type == "private" else f"group:{group_id_int}"

    try:
        input_images: list[Any] = []
        for index, image in enumerate(images, start=1):
            input_images.append(
                await _resolve_input_image(
                    bot=bot,
                    image=image,
                    max_size_bytes=int(current_cfg.max_input_image_bytes),
                    parameter_name=f"images[{index}]",
                )
            )
        if len(input_images) > int(current_cfg.max_input_image_count):
            raise ValueError(f"images 数量不能超过 {int(current_cfg.max_input_image_count)} 张")
        mask_image = None
        if mask is not None and str(mask).strip():
            mask_image = await _resolve_input_image(
                bot=bot,
                image=mask,
                max_size_bytes=int(current_cfg.max_input_image_bytes),
                parameter_name="mask",
            )
    except ValueError as exc:
        await EditCommand.finish(str(exc))

    spec = OpenAIDrawJobSpec(
        chat_type=chat_type,
        session_id=session_id,
        prompt=prompt,
        size=current_cfg.default_size,
        quality=current_cfg.default_quality,
        background=current_cfg.default_background,
        input_fidelity=current_cfg.default_input_fidelity,
        output_format=str(current_cfg.default_output_format),
        mode="edit",
        input_images=tuple(input_images),
        mask_image=mask_image,
        group_id=group_id_int,
        requester_user_id=requester_user_id,
        target_user_id=requester_user_id,
        bot=bot,
        message_manager=await get_message_manager(),
        cache=await get_user_cache_manager(),
    )

    manager = get_openai_draw_queue_manager()
    try:
        state = await manager.submit(spec)
        snapshot = await manager.snapshot()
    except RuntimeError as exc:
        await EditCommand.finish(str(exc))

    await EditCommand.finish(
        f"已启动异步改图任务 job={state.job_id} running={snapshot['running_count']} queued={snapshot['queued_count']} 图片将由后台任务完成后另行发送。"
    )


@DrawTasksCommand.handle()
async def handle_draw_tasks_command(event: MessageEvent) -> None:
    """处理绘图任务列表命令。

    Args:
        event: 当前消息事件。
    """

    current_cfg = get_openai_draw_plugin_config()
    try:
        await _ensure_permission(current_cfg.permissions.query, int(event.user_id))
    except PermissionError:
        await DrawTasksCommand.finish("你没有查看绘图任务的权限。")

    snapshot = await get_openai_draw_queue_manager().snapshot()
    running = snapshot.get("running") or []
    queued = snapshot.get("queued") or []
    queue_max = int(snapshot.get("queue_max") or 0)
    lines = [f"running={int(snapshot.get('running_count') or 0)} queued={int(snapshot.get('queued_count') or 0)}/{queue_max}"]

    def _format(item: dict) -> str:
        prompt_preview = str(item.get("prompt_preview") or "")
        return (
            f"- {item.get('job_id', '')} {item.get('status', '')} "
            f"size={item.get('size', '')} quality={item.get('quality', '')} "
            f"target={item.get('target_user_id', '')} prompt={prompt_preview}"
        )

    if running:
        lines.append("[running]")
        for item in running[:10]:
            if isinstance(item, dict):
                lines.append(_format(item))

    if queued:
        lines.append("[queued]")
        for item in queued[:10]:
            if isinstance(item, dict):
                lines.append(_format(item))

    await DrawTasksCommand.finish("\n".join(lines))
