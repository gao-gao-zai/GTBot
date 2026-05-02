from __future__ import annotations

from random import uniform
from time import time
from typing import Any

from nonebot.adapters.onebot.v11.message import Message

from ...ConfigManager import total_config
from ...model import QueuedMessageItem


def get_current_send_timing_config() -> Any:
    """返回当前生效的聊天发送节奏配置。

    发送节奏配置归属于当前聊天配置组，因此这里按调用时刻从全局配置中心读取，
    避免模块导入时缓存旧对象，导致切组后仍使用过期参数。

    Returns:
        Any: 当前配置组中的 `chat_model.send_timing` 运行时配置对象。
    """

    return total_config.processed_configuration.current_config_group.chat_model.send_timing


def calculate_message_delay_seconds(
    message: Message,
    *,
    send_timing: Any,
) -> float:
    """根据单条消息内容计算拟人化发送等待时间。

    该函数只负责把一条已规范化消息映射为“发完这一条后应等待多久”，
    不负责决定消息是否入队，也不负责实际等待。随机扰动采用对称抖动，
    结果会被钳制在 `[base_interval_seconds, max_interval_seconds]` 范围内。

    Args:
        message: 已完成 CQ 解析和规范化的 OneBot 消息对象。
        send_timing: 当前运行时的发送节奏配置对象。

    Returns:
        float: 当前消息对应的发送等待秒数。
    """

    base_interval_seconds = max(0.0, float(getattr(send_timing, "base_interval_seconds", 0.0) or 0.0))
    per_char_seconds = max(0.0, float(getattr(send_timing, "per_char_seconds", 0.0) or 0.0))
    jitter_seconds = max(0.0, float(getattr(send_timing, "jitter_seconds", 0.0) or 0.0))
    max_interval_seconds = max(
        base_interval_seconds,
        float(getattr(send_timing, "max_interval_seconds", base_interval_seconds) or base_interval_seconds),
    )
    non_text_equivalent_chars = getattr(send_timing, "non_text_equivalent_chars", {}) or {}

    text_chars = 0
    non_text_equivalent_total = 0
    for segment in message:
        if segment.type == "text":
            text_chars += len(str(segment.data.get("text", "")))
            continue
        segment_key = str(segment.type or "").strip()
        non_text_equivalent_total += int(non_text_equivalent_chars.get(segment_key, 0) or 0)

    raw_delay = (
        base_interval_seconds
        + float(text_chars + non_text_equivalent_total) * per_char_seconds
        + uniform(-jitter_seconds, jitter_seconds)
    )
    return min(max_interval_seconds, max(base_interval_seconds, raw_delay))


def build_queued_message_items(
    messages: list[Message],
    *,
    interval_override: float | None,
    force_wait: bool = False,
) -> list[QueuedMessageItem]:
    """为一组已准备好的消息构造队列项。

    若调用方显式给出 `interval_override`，所有消息都使用该固定延迟；
    否则按当前发送节奏配置为每条消息独立计算延迟。是否强制从入队时刻起
    等待这么久，则由 `force_wait` 决定，默认关闭。

    Args:
        messages: 已完成消息规范化的待发送消息列表。
        interval_override: 调用方显式指定的固定等待时间；为 `None` 时使用自动计算。
        force_wait: 是否要求队列从入队时刻起至少等待指定延迟。

    Returns:
        list[QueuedMessageItem]: 可直接放入消息队列的数据结构。
    """

    enqueued_at = time()
    if interval_override is not None:
        normalized_interval = max(0.0, float(interval_override))
        return [
            QueuedMessageItem(
                message=message,
                delay_seconds=normalized_interval,
                force_wait=bool(force_wait),
                enqueued_at=enqueued_at,
            )
            for message in messages
        ]

    send_timing = get_current_send_timing_config()
    return [
        QueuedMessageItem(
            message=message,
            delay_seconds=calculate_message_delay_seconds(
                message,
                send_timing=send_timing,
            ),
            force_wait=bool(force_wait),
            enqueued_at=enqueued_at,
        )
        for message in messages
    ]
