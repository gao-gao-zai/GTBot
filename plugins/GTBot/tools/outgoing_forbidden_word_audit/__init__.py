from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment

_FORBIDDEN_TERMS = (
    ("逆向", "逆向"),
    ("注册机", "注册机"),
    ("js算法分析", "js算法分析"),
    ("ddos", "ddos"),
    ("cc", "cc"),
)
_PATCH_FLAG = "_gtbot_outgoing_forbidden_word_audit_patched"


def _is_send_api(api: str) -> bool:
    """判断当前 OneBot API 名称是否属于出站发送动作。

    这里只覆盖真正会把内容发给外部会话的 `send_*` 系列接口，避免把查询、
    删除或其他管理操作也纳入违禁词扫描，导致无意义日志噪声。

    Args:
        api: 本次 `call_api` 调用的 OneBot 动作名称。

    Returns:
        当动作名称以 `send_` 开头时返回 `True`，否则返回 `False`。
    """

    return str(api or "").startswith("send_")


def _iter_text_fragments(payload: Any) -> Iterable[str]:
    """递归提取待发送负载中的全部文本片段。

    该函数不会改写任何发送数据，只负责把字符串、消息段、消息列表以及
    合并转发节点中的嵌套字段尽量展开，便于统一做违禁词命中检查。

    Args:
        payload: 即将发送给 OneBot 的任意消息负载。

    Yields:
        str: 从负载中提取出的文本片段。
    """

    if payload is None:
        return

    if isinstance(payload, str):
        if payload:
            yield payload
        return

    if isinstance(payload, Message):
        for segment in payload:
            yield from _iter_text_fragments(segment)
        return

    if isinstance(payload, MessageSegment):
        yield from _iter_text_fragments(getattr(payload, "data", {}))
        return

    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_text_fragments(value)
        return

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            yield from _iter_text_fragments(item)
        return

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python")
        except TypeError:
            dumped = model_dump()
        yield from _iter_text_fragments(dumped)


def _contains_ascii_term(text: str, term: str) -> bool:
    """按 ASCII 单词边界判断英文违禁词是否命中。

    `ddos` 与 `cc` 这类纯英文缩写容易误命中更长单词，因此这里要求其左右两侧
    不是英文字母或数字；中文词仍按普通子串匹配处理。

    Args:
        text: 待检查文本，调用方会先传入小写版本。
        term: 待匹配的英文违禁词，同样使用小写。

    Returns:
        当文本中存在独立的英文违禁词时返回 `True`，否则返回 `False`。
    """

    start = 0
    term_length = len(term)
    while True:
        index = text.find(term, start)
        if index < 0:
            return False
        left = text[index - 1] if index > 0 else ""
        right_index = index + term_length
        right = text[right_index] if right_index < len(text) else ""
        if not left.isalnum() and not right.isalnum():
            return True
        start = index + 1


def _find_matched_terms(fragments: Sequence[str]) -> list[str]:
    """在文本片段列表中查找所有命中的违禁词。

    中文词按原样子串匹配，英文缩写使用不区分大小写的边界匹配。返回结果会按配置
    顺序去重，方便日志里稳定展示。

    Args:
        fragments: 已从待发送负载中展开得到的文本片段序列。

    Returns:
        list[str]: 本次命中的违禁词列表；未命中时返回空列表。
    """

    matched: list[str] = []
    for display, needle in _FORBIDDEN_TERMS:
        normalized_needle = needle.lower()
        for fragment in fragments:
            normalized_fragment = fragment.lower()
            if needle.isascii():
                if _contains_ascii_term(normalized_fragment, normalized_needle):
                    matched.append(display)
                    break
            elif normalized_needle in normalized_fragment:
                matched.append(display)
                break
    return matched


def _build_preview(fragments: Sequence[str], *, limit: int = 200) -> str:
    """构造用于告警日志的发送内容预览。

    预览只用于帮助定位问题，因此会把多个片段拼接成一行并截断，避免消息体过大
    时把整段发送内容直接刷进日志。

    Args:
        fragments: 已提取的文本片段列表。
        limit: 预览允许保留的最大字符数。

    Returns:
        str: 适合写入告警日志的一段简短预览文本。
    """

    preview = " | ".join(fragment.strip() for fragment in fragments if str(fragment).strip())
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}...(truncated)"


def _warn_if_contains_forbidden_terms(api: str, payload: Any) -> None:
    """检查待发送负载是否命中违禁词，并在命中时输出告警日志。

    该函数只记录日志，不会阻断发送，也不会修改待发送内容。若提取不到任何文本，
    则直接静默返回。

    Args:
        api: 本次调用的 OneBot 发送动作名称。
        payload: 待发送的完整数据负载。
    """

    fragments = list(_iter_text_fragments(payload))
    if not fragments:
        return

    matched_terms = _find_matched_terms(fragments)
    if not matched_terms:
        return

    logger.warning(
        "GTBot outgoing forbidden-word audit hit: api=%s matched=%s preview=%r",
        api,
        ",".join(matched_terms),
        _build_preview(fragments),
    )


def _patch_bot_call_api() -> None:
    """为 OneBot `Bot.call_api` 安装一次性发送审计补丁。

    补丁只在首次加载插件时生效，并通过实例方法包装统一覆盖 `send_msg`、
    `send_group_msg`、`send_private_msg` 以及合并转发等发送接口。重复加载插件时
    会直接复用已有补丁，避免多层嵌套包装造成重复告警。
    """

    if getattr(Bot.call_api, _PATCH_FLAG, False):
        return

    original_call_api = Bot.call_api

    async def audited_call_api(self: Bot, api: str, **data: Any) -> Any:
        """在执行底层发送动作前检查负载中的违禁词并保持原有返回语义。

        Args:
            api: 当前调用的 OneBot API 名称。
            **data: 传给底层 API 的原始参数。

        Returns:
            Any: 原始 `call_api` 的返回结果。
        """

        if _is_send_api(api):
            _warn_if_contains_forbidden_terms(api, data)
        return await original_call_api(self, api, **data)

    setattr(audited_call_api, _PATCH_FLAG, True)
    Bot.call_api = audited_call_api


def register(registry) -> None:  # noqa: ANN001
    """注册出站违禁词审计插件，并确保发送补丁已经安装。

    该插件不向 Agent 暴露工具，也不修改对话上下文；注册动作本身只是复用 GTBot
    插件加载机制，在启动阶段完成一次性的发送链路审计补丁安装。

    Args:
        registry: GTBot 提供的插件注册器。当前实现不会直接向其注册条目。

    Returns:
        None: 无返回值。
    """

    del registry
    _patch_bot_call_api()
    return None
