from __future__ import annotations

from time import time

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.event import MessageEvent

from local_plugins.nonebot_plugin_gt_permission import PermissionRole
from plugins.GTBot.services.help import HelpCommandSpec, register_help
from plugins.GTBot.services.shared import fun as Fun

from .config import get_friend_management_plugin_config
from .like_gate import get_today_like_count_for_bot_from_user, should_bypass_zanwo_gate
from .usage_limits import calculate_like_send_times, get_friend_management_like_limit_manager

LikeCommand = on_command("点赞", priority=4, block=True)
ZanWoCommand = on_command("赞我", priority=4, block=True)


def _register_help_items() -> None:
    """注册点赞相关命令到帮助系统。

    这里会同时注册 `/点赞` 与 `/赞我` 两条命令。前者始终允许直接请求机器人给自己点赞，
    后者则会额外读取配置中的门槛与豁免规则，用于实现“先给机器人点赞，再换取机器人回赞”
    的玩法。
    """

    register_help(
        HelpCommandSpec(
            name="点赞",
            category="好友管理",
            summary="让机器人给命令发起人点赞并直接点满。",
            description="命令执行后会调用 OneBot `send_like` 接口，对当前命令发起人优先发送 10 次点赞；若当天剩余额度不足，则只发送剩余额度允许的数量。",
            examples=("/点赞",),
            required_role=PermissionRole.USER,
            audience="群聊和私聊",
            sort_key=90,
        )
    )
    register_help(
        HelpCommandSpec(
            name="赞我",
            category="好友管理",
            summary="按门槛规则申请让机器人给命令发起人点赞。",
            description="命令执行前会先检查你今天是否已经给机器人点过足够数量的赞；命中管理员豁免或额外豁免名单时可直接跳过该门槛。",
            examples=("/赞我",),
            required_role=PermissionRole.USER,
            audience="群聊和私聊",
            sort_key=91,
        )
    )


_register_help_items()


@LikeCommand.handle()
async def handle_like_command(bot: Bot, event: MessageEvent) -> None:
    """处理手动点赞命令。

    该命令始终将命令发送者作为点赞目标，避免从自由文本中解析目标用户导致误判。
    执行成功后只返回简短提示，不向上层透出底层接口的原始返回体。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前命令事件，用于读取发起人的 QQ 号。
    """

    cfg = get_friend_management_plugin_config()
    target_user_id = int(event.user_id)
    now_ts = time()
    send_times = calculate_like_send_times(cfg=cfg, user_id=target_user_id, now_ts=now_ts)
    if send_times <= 0:
        limit = int(cfg.max_likes_per_user_per_day)
        await LikeCommand.finish(f"你今天已经被点满赞了 ({limit}/{limit})")

    try:
        await Fun.send_like(bot, target_user_id, times=send_times)
    except Exception as exc:  # noqa: BLE001
        await LikeCommand.finish(f"点赞失败: {exc!s}")

    get_friend_management_like_limit_manager().record_like(
        cfg=cfg,
        user_id=target_user_id,
        count=send_times,
        now_ts=now_ts,
    )
    if send_times < 10:
        await LikeCommand.finish(f"已给你点赞 {send_times} 次，今天额度已用尽 ({target_user_id})")
        return
    await LikeCommand.finish(f"已给你点满赞 ({target_user_id})")


@ZanWoCommand.handle()
async def handle_zanwo_command(bot: Bot, event: MessageEvent) -> None:
    """处理带门槛的 `赞我` 命令。

    该命令与普通 `/点赞` 的发送逻辑一致，但会在真正点赞前额外检查调用者今天是否已经
    给机器人点过足够数量的赞。命中管理员豁免或显式豁免名单时，会直接跳过门槛检查。

    Args:
        bot: 当前 OneBot Bot 实例。
        event: 当前命令事件，用于读取命令发起人的 QQ 号。
    """

    cfg = get_friend_management_plugin_config()
    target_user_id = int(event.user_id)
    required_likes = int(cfg.require_likes_before_zanwo)

    if required_likes > 0 and not await should_bypass_zanwo_gate(cfg=cfg, user_id=target_user_id):
        today_like_count = await get_today_like_count_for_bot_from_user(bot=bot, user_id=target_user_id)
        if today_like_count < required_likes:
            await ZanWoCommand.finish(
                f"你今天给机器人点的赞还不够，需要至少 {required_likes} 个，当前只有 {today_like_count} 个。"
            )
            return

    now_ts = time()
    send_times = calculate_like_send_times(cfg=cfg, user_id=target_user_id, now_ts=now_ts)
    if send_times <= 0:
        limit = int(cfg.max_likes_per_user_per_day)
        await ZanWoCommand.finish(f"你今天已经被点满赞了 ({limit}/{limit})")

    try:
        await Fun.send_like(bot, target_user_id, times=send_times)
    except Exception as exc:  # noqa: BLE001
        await ZanWoCommand.finish(f"点赞失败: {exc!s}")

    get_friend_management_like_limit_manager().record_like(
        cfg=cfg,
        user_id=target_user_id,
        count=send_times,
        now_ts=now_ts,
    )
    if send_times < 10:
        await ZanWoCommand.finish(f"已给你点赞 {send_times} 次，今天额度已用尽 ({target_user_id})")
        return
    await ZanWoCommand.finish(f"已给你点满赞 ({target_user_id})")
