from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from nonebot import logger
from nonebot.adapters.onebot.v11.message import Message, MessageSegment


def serialize_message_segments(
    message: Message | MessageSegment | Iterable[MessageSegment] | None,
) -> str | None:
    """将 OneBot v11 消息对象序列化为原样消息段 JSON。

    该函数会尽量保留适配器原始语义：每个消息段按 `type` 与 `data`
    两个字段输出，`data` 直接来自 `dict(segment.data)`，不会裁剪 `None`
    值、不会尝试把字符串数字转回整数，也不会把消息先转成 CQ 文本再二次解析。

    Args:
        message: 待序列化的消息对象，可以是完整 `Message`、单个
            `MessageSegment`、可迭代的消息段集合，或 `None`。

    Returns:
        `str | None`: 顶层为数组的 JSON 字符串；当输入为空、不可迭代或
        序列化失败时返回 `None`。
    """
    if message is None:
        return None

    if isinstance(message, MessageSegment):
        segments = [message]
    else:
        try:
            segments = list(message)
        except TypeError:
            return None

    if not segments:
        return None

    try:
        payload = [
            {
                "type": segment.type,
                "data": dict(segment.data),
            }
            for segment in segments
        ]
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"序列化消息段失败，已跳过结构化存储: {exc!s}")
        return None
