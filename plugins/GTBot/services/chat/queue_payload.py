from __future__ import annotations

from collections.abc import Sequence

from nonebot.adapters.onebot.v11.message import Message, MessageSegment

from ..shared import fun as Fun
from ...Logger import logger
from ...constants import SUPPORTED_CQ_CODES

QueueMessageContent = str | Message | MessageSegment


async def prepare_queue_message(
    content: QueueMessageContent,
    *,
    scope: str,
) -> Message:
    """将入队前的消息内容规范化为可直接发送的 `Message`。

    这个辅助函数负责原先散落在队列里的两类输入预处理：
    1. 去掉误混入的聊天记录前缀。
    2. 将文本里的受支持 CQ 码解析为 `MessageSegment`。

    队列本体只负责顺序发送和回写数据库，不再承担输入清洗职责。

    Args:
        content: 调用方准备入队的消息内容，可以是纯文本、单个消息段或完整消息对象。
        scope: 日志范围描述，用于在发现聊天记录前缀时输出可定位的告警。

    Returns:
        Message: 可以直接交给 OneBot 发送的消息对象。
    """
    if isinstance(content, Message):
        return content

    if isinstance(content, MessageSegment):
        return Message(content)

    raw_content = str(content)
    normalized_content, hit = Fun.strip_chat_log_prefix_with_hit(raw_content)
    if hit:
        logger.warning(
            "检测到待入队消息存在聊天记录前缀并已清洗（%s）: %r -> %r",
            scope,
            raw_content[:120],
            normalized_content[:120],
        )

    return await Fun.text_to_message(
        normalized_content,
        whitelist=SUPPORTED_CQ_CODES,
    )


async def prepare_queue_messages(
    messages: Sequence[QueueMessageContent],
    *,
    scope: str,
) -> list[Message]:
    """批量规范化待入队消息。

    Args:
        messages: 调用方准备交给消息队列的一组消息内容。
        scope: 日志范围描述，会传递给单条消息的规范化逻辑。

    Returns:
        list[Message]: 已完成前缀清洗和 CQ 解析的消息列表。
    """
    prepared: list[Message] = []
    for content in messages:
        prepared.append(await prepare_queue_message(content, scope=scope))
    return prepared
