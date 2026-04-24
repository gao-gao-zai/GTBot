from __future__ import annotations

import asyncio
import time

from nonebot import get_driver, logger, on_type
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupIncreaseNoticeEvent,
    GroupMessageEvent,
    Message,
    GroupRequestEvent,
    MessageSegment,
)
from nonebot.plugin import PluginMetadata
from nonebot.rule import Rule

from .commands import safe_register
from .store import (
    PendingVerification,
    build_code,
    cancel_task,
    delete_file,
    extract_level,
    log_database_path,
    render_captcha_image,
    store,
)

driver = get_driver()

__plugin_meta__ = PluginMetadata(
    name="群入群验证码守卫",
    description="按 QQ 等级自动审核指定群加群申请，并要求新成员在时限内输入动态验证码。",
    usage=(
        "管理员命令：\n"
        "/入群守卫状态\n"
        "/开启入群守卫 [群号]\n"
        "/关闭入群守卫 [群号]\n"
        "/设置入群等级 <等级>\n"
        "/设置申请处理时限 <秒数>\n"
        "/设置验证码时限 <秒数>\n"
        "/设置验证码长度 <长度>"
    ),
    type="application",
    supported_adapters={"~onebot.v11"},
)

async def _is_target_group(group_id: int) -> bool:
    """判断指定群当前是否启用了入群守卫。

    Args:
        group_id: 当前事件所属群号。

    Returns:
        当该群存在于 SQLite 持久化的目标群列表中时返回 `True`。
    """

    return int(group_id) in (await store.get_config()).target_groups


async def _should_handle_group_request(event: GroupRequestEvent) -> bool:
    """在匹配阶段预过滤需要处理的入群申请事件。

    仅放行已开启入群守卫的目标群中的主动入群申请，避免无关的邀请事件或其他群的申请
    进入处理函数后再被动返回，从而减少普通请求事件带来的日志噪声和调度开销。

    Args:
        event: 待判断的 OneBot v11 群请求事件。

    Returns:
        当事件属于目标群的 `add` 入群申请时返回 `True`，否则返回 `False`。
    """

    return event.sub_type == "add" and await _is_target_group(event.group_id)


async def _should_handle_group_increase(event: GroupIncreaseNoticeEvent) -> bool:
    """在匹配阶段预过滤需要继续处理的群成员增加事件。

    这里只判断目标群命中情况，把“是否由本插件自动批准”这类依赖运行时授权状态的检查
    留给处理函数完成，以便继续保留完整的业务日志。

    Args:
        event: 待判断的群成员增加通知事件。

    Returns:
        当事件属于已启用入群守卫的目标群时返回 `True`，否则返回 `False`。
    """

    return await _is_target_group(event.group_id)


async def _should_handle_group_message(event: GroupMessageEvent) -> bool:
    """在匹配阶段预过滤待验证成员发送的群消息。

    验证码插件只关心目标群内、且当前仍处于待验证状态的成员消息。将这部分判断前置到
    Rule 层后，普通群消息将不再进入 matcher，从而显著减少无意义的事件处理日志。

    Args:
        event: 待判断的群消息事件。

    Returns:
        当消息来自目标群内的待验证成员时返回 `True`，否则返回 `False`。
    """

    if not await _is_target_group(event.group_id):
        return False
    pending = await store.get_pending(event.group_id, event.user_id)
    return pending is not None


group_request_matcher = on_type(
    GroupRequestEvent,
    rule=Rule(_should_handle_group_request),
    priority=5,
    block=False,
)
group_increase_matcher = on_type(
    GroupIncreaseNoticeEvent,
    rule=Rule(_should_handle_group_increase),
    priority=5,
    block=False,
)
group_message_matcher = on_type(
    GroupMessageEvent,
    rule=Rule(_should_handle_group_message),
    priority=20,
    block=False,
)


async def _kick_member(bot: Bot, group_id: int, user_id: int) -> None:
    """调用通用 OneBot API 将超时未验证成员移出群聊。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        group_id: 目标群号。
        user_id: 待移出的成员 QQ 号。
    """

    await bot.call_api(
        "set_group_kick",
        group_id=group_id,
        user_id=user_id,
        reject_add_request=False,
    )


async def _handle_verification_timeout(bot: Bot, group_id: int, user_id: int, expected_code: str) -> None:
    """在成员验证码超时后执行最终清理和踢人逻辑。

    该函数会再次核对当前状态和验证码，以避免旧的超时任务在配置变更或状态替换后误伤成员。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        group_id: 群号。
        user_id: 成员 QQ 号。
        expected_code: 创建任务时绑定的验证码。
    """

    pending = await store.get_pending(group_id, user_id)
    if pending is None or pending.code != expected_code:
        return

    removed = await store.pop_pending(group_id, user_id)
    if removed is None or removed.code != expected_code:
        return

    config = await store.get_config()
    try:
        await bot.send_group_msg(
            group_id=group_id,
            message=config.timeout_template.format(
                user_at=MessageSegment.at(user_id),
                timeout_seconds=config.verify_timeout_seconds,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("发送超时提示失败: group_id={} user_id={} error={}", group_id, user_id, exc)
    finally:
        delete_file(removed.image_path)

    if not config.kick_on_timeout:
        return

    try:
        await _kick_member(bot, group_id, user_id)
        logger.info("成员验证超时已移出群聊: group_id={} user_id={}", group_id, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("成员验证超时踢出失败: group_id={} user_id={} error={}", group_id, user_id, exc)


async def _timeout_worker(
    *,
    bot: Bot,
    group_id: int,
    user_id: int,
    code: str,
    timeout_seconds: int,
) -> None:
    """等待指定时长后检查验证码是否超时。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        group_id: 群号。
        user_id: 成员 QQ 号。
        code: 本轮验证绑定的验证码。
        timeout_seconds: 最大等待秒数。
    """

    try:
        await asyncio.sleep(timeout_seconds)
        await _handle_verification_timeout(bot, group_id, user_id, code)
    except asyncio.CancelledError:
        raise


async def _schedule_verification(bot: Bot, group_id: int, user_id: int) -> PendingVerification:
    """为新成员创建新的验证码状态和超时任务。

    若该成员此前存在未完成的验证状态，会先取消旧任务，再覆盖为当前这一轮验证。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        group_id: 群号。
        user_id: 成员 QQ 号。

    Returns:
        新写入的待验证状态。
    """

    config = await store.get_config()
    timeout_seconds = max(1, int(config.verify_timeout_seconds))
    code = build_code(max(1, int(config.code_length)))
    pending = PendingVerification(
        group_id=group_id,
        user_id=user_id,
        code=code,
        deadline_ts=time.time() + timeout_seconds,
        joined_at_ts=time.time(),
        image_path=render_captcha_image(code, group_id, user_id),
    )
    pending.timeout_task = asyncio.create_task(
        _timeout_worker(
            bot=bot,
            group_id=group_id,
            user_id=user_id,
            code=code,
            timeout_seconds=timeout_seconds,
        )
    )
    previous = await store.upsert_pending(pending)
    if previous is not None:
        await cancel_task(previous.timeout_task)
    return pending


@group_request_matcher.handle()
async def handle_group_request(bot: Bot, event: GroupRequestEvent) -> None:
    """处理受保护目标群的加群申请。

    仅处理主动加群申请事件，并要求申请事件仍位于允许处理窗口内。
    QQ 等级通过 LLOneBot 的 `get_stranger_info` 查询，未达到门槛时按配置决定是否自动拒绝。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        event: 加群申请事件对象。
    """

    logger.info(
        "收到群请求事件: group_id={} user_id={} sub_type={} flag={} comment={}",
        event.group_id,
        event.user_id,
        event.sub_type,
        event.flag,
        event.comment,
    )

    if event.sub_type != "add":
        logger.debug(
            "当前群请求不是入群申请，已跳过: group_id={} user_id={} sub_type={}",
            event.group_id,
            event.user_id,
            event.sub_type,
        )
        return
    if not await _is_target_group(event.group_id):
        logger.debug(
            "当前群未开启入群守卫，已跳过申请处理: group_id={} user_id={}",
            event.group_id,
            event.user_id,
        )
        return

    config = await store.get_config()
    logger.info(
        "开始处理入群申请: group_id={} user_id={} min_level={} request_expire_seconds={} reject_on_low_level={}",
        event.group_id,
        event.user_id,
        config.min_level,
        config.request_expire_seconds,
        config.reject_on_low_level,
    )
    age = int(time.time()) - int(event.time)
    logger.info(
        "入群申请时效检查: group_id={} user_id={} age={}s max_age={}s",
        event.group_id,
        event.user_id,
        age,
        config.request_expire_seconds,
    )
    if age > int(config.request_expire_seconds):
        logger.info(
            "加群申请已过处理时效，忽略本次请求: group_id={} user_id={} age={}s max_age={}s",
            event.group_id,
            event.user_id,
            age,
            config.request_expire_seconds,
        )
        return

    try:
        logger.info(
            "开始查询申请人陌生人资料: group_id={} user_id={} no_cache={}",
            event.group_id,
            event.user_id,
            True,
        )
        stranger_info = await bot.get_stranger_info(user_id=event.user_id, no_cache=True)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "查询申请人陌生人信息失败，忽略本次请求: group_id={} user_id={} error={}",
            event.group_id,
            event.user_id,
            exc,
        )
        return

    logger.info(
        "申请人陌生人资料查询成功: group_id={} user_id={} payload={}",
        event.group_id,
        event.user_id,
        stranger_info,
    )
    level = extract_level(stranger_info)
    if level is None:
        logger.warning(
            "未能从陌生人信息中解析等级，忽略本次请求: group_id={} user_id={} payload={}",
            event.group_id,
            event.user_id,
            stranger_info,
        )
        return

    logger.info(
        "申请人等级解析成功: group_id={} user_id={} level={} required={}",
        event.group_id,
        event.user_id,
        level,
        config.min_level,
    )
    if level < int(config.min_level):
        if config.reject_on_low_level:
            logger.info(
                "申请人等级不足，准备自动拒绝申请: group_id={} user_id={} level={} required={} reason={}",
                event.group_id,
                event.user_id,
                level,
                config.min_level,
                config.reject_reason,
            )
            await event.reject(bot, reason=config.reject_reason)
            logger.info(
                "申请人等级不足，已拒绝入群申请: group_id={} user_id={} level={} required={}",
                event.group_id,
                event.user_id,
                level,
                config.min_level,
            )
        else:
            logger.info(
                "申请人等级不足，但当前配置为仅忽略不拒绝: group_id={} user_id={} level={} required={}",
                event.group_id,
                event.user_id,
                level,
                config.min_level,
            )
        return

    logger.info(
        "申请人等级达标，准备自动同意申请: group_id={} user_id={} level={}",
        event.group_id,
        event.user_id,
        level,
    )
    await store.mark_approved_join(event.group_id, event.user_id)
    try:
        await event.approve(bot)
    except Exception:
        await store.revoke_approved_join(event.group_id, event.user_id)
        raise
    logger.info(
        "申请人等级达标，已自动通过申请: group_id={} user_id={} level={}",
        event.group_id,
        event.user_id,
        level,
    )


@group_increase_matcher.handle()
async def handle_group_increase(bot: Bot, event: GroupIncreaseNoticeEvent) -> None:
    """在新成员加入受保护群后启动验证码验证流程。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        event: 群成员增加事件对象。
    """

    logger.info(
        "收到群成员增加事件: group_id={} user_id={} operator_id={} sub_type={}",
        event.group_id,
        event.user_id,
        getattr(event, "operator_id", None),
        getattr(event, "sub_type", None),
    )

    if not await _is_target_group(event.group_id):
        logger.debug(
            "当前群未开启入群守卫，跳过入群后验证流程: group_id={} user_id={}",
            event.group_id,
            event.user_id,
        )
        return
    if event.user_id == int(bot.self_id):
        logger.debug("入群成员是机器人自身，跳过验证流程: group_id={} user_id={}", event.group_id, event.user_id)
        return
    config = await store.get_config()
    approved = await store.consume_approved_join(
        event.group_id,
        event.user_id,
        ttl_seconds=config.request_expire_seconds,
    )
    logger.info(
        "检查成员是否由本插件自动批准: group_id={} user_id={} approved={} ttl_seconds={}",
        event.group_id,
        event.user_id,
        approved,
        config.request_expire_seconds,
    )
    if not approved:
        logger.debug(
            "新成员并非由本插件自动批准，跳过验证码流程: group_id={} user_id={}",
            event.group_id,
            event.user_id,
        )
        return

    pending = await _schedule_verification(bot, event.group_id, event.user_id)
    message = config.welcome_template.format(
        user_at=MessageSegment.at(event.user_id),
        timeout_seconds=config.verify_timeout_seconds,
        code=pending.code,
    )
    image_path = pending.image_path
    if image_path is None:
        logger.error(
            "验证码图片路径缺失，无法发送验证消息: group_id={} user_id={}",
            event.group_id,
            event.user_id,
        )
        return
    outgoing_message = Message(message)
    outgoing_message.append(MessageSegment.image(image_path))
    await bot.send_group_msg(group_id=event.group_id, message=outgoing_message)
    logger.info(
        "已为新成员创建验证码验证流程: group_id={} user_id={} timeout={}s",
        event.group_id,
        event.user_id,
        config.verify_timeout_seconds,
    )


@group_message_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    """校验待验证成员在群内发送的验证码。

    Args:
        bot: 当前在线的 OneBot v11 Bot 实例。
        event: 群消息事件对象。
    """

    if not await _is_target_group(event.group_id):
        return

    pending = await store.get_pending(event.group_id, event.user_id)
    if pending is None:
        return

    plain_text = event.get_plaintext().strip()

    config = await store.get_config()
    if plain_text != pending.code:
        remaining_seconds = max(1, int(pending.deadline_ts - time.time()))
        await bot.send_group_msg(
            group_id=event.group_id,
            message=config.failure_template.format(
                user_at=MessageSegment.at(event.user_id),
                timeout_seconds=remaining_seconds,
                code=pending.code,
            ),
        )
        return

    removed = await store.pop_pending(event.group_id, event.user_id)
    if removed is None:
        return
    await cancel_task(removed.timeout_task)
    try:
        await bot.send_group_msg(
            group_id=event.group_id,
            message=config.success_template.format(
                user_at=MessageSegment.at(event.user_id),
                timeout_seconds=config.verify_timeout_seconds,
                code=removed.code,
            ),
        )
        logger.info("成员验证码验证通过: group_id={} user_id={}", event.group_id, event.user_id)
    finally:
        delete_file(removed.image_path)


@driver.on_startup
async def _startup_group_gatekeeper() -> None:
    """在 Bot 启动时初始化数据库、命令和帮助项。"""

    await log_database_path()
    await safe_register()


@driver.on_shutdown
async def _shutdown_group_gatekeeper() -> None:
    """在 Bot 关闭时取消全部验证码超时任务。"""

    await store.cancel_pending_tasks(await store.clear_pending())
    await store.clear_approved_joins()

__all__ = [
    "__plugin_meta__",
]
