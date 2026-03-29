from __future__ import annotations

import json
from collections.abc import Iterable

from nonebot import logger
from nonebot.adapters.onebot.v11.message import Message, MessageSegment


def serialize_message_segments(
    message: Message | MessageSegment | Iterable[MessageSegment] | None,
) -> str | None:
    """将 OneBot v11 消息对象序列化为原样消息段 JSON。

    Args:
        message: 待序列化的消息对象，可以是完整 `Message`、单个 `MessageSegment`、
            可迭代的消息段集合，或 `None`。

    Returns:
        str | None: 顶层为数组的 JSON 字符串；当输入为空、不可迭代或序列化失败时返回 `None`。
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


def deserialize_message_segments(payload: str | None) -> Message | None:
    """将序列化后的消息段 JSON 还原为 OneBot `Message` 对象。

    Args:
        payload: 由 `serialize_message_segments` 生成的 JSON 字符串；为空时返回 `None`。

    Returns:
        Message | None: 成功还原时返回原样消息对象；格式非法或反序列化失败时返回 `None`。
    """

    normalized = str(payload or "").strip()
    if not normalized:
        return None

    try:
        raw_segments = json.loads(normalized)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"反序列化消息段失败，已跳过结构化恢复: {exc!s}")
        return None

    if not isinstance(raw_segments, list):
        return None

    message = Message()
    try:
        for item in raw_segments:
            if not isinstance(item, dict):
                return None
            segment_type = str(item.get("type") or "").strip()
            data = item.get("data") or {}
            if not segment_type or not isinstance(data, dict):
                return None
            message.append(MessageSegment(type=segment_type, data=data))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"构建 Message 对象失败，已回退为 None: {exc!s}")
        return None

    return message if len(message) > 0 else None
