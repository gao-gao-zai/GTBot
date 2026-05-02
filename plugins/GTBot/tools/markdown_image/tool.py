from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

from langchain.tools import ToolRuntime, tool
from nonebot.adapters.onebot.v11.message import Message, MessageSegment

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.services.chat.context import GroupChatContext
from plugins.GTBot.services.file_registry import register_local_file

from .config import PLUGIN_DIR, get_markdown_image_plugin_config
from .renderer import MarkdownRenderResult, render_markdown_to_image

_MARKDOWN_IMAGE_FILE_REF_TTL_SEC = 3 * 24 * 60 * 60
_MAX_MARKDOWN_LENGTH = 20_000
_SendMessagesCallable = Callable[[list[Message]], Awaitable[Any]]


def _get_markdown_image_dir() -> Path:
    """返回 Markdown 图片插件的数据目录并确保其存在。

    该目录仅用于保存当前插件自己生成的图片文件，不与 GT 文件映射层共享落盘
    责任。这样可以保持“插件自行管理物理文件，GTFile 只负责映射”的边界清晰。

    Returns:
        当前插件的图片输出目录绝对路径。
    """

    data_dir = cast(Path, total_config.get_data_dir_path())
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / "markdown_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_markdown_text(markdown_text: str) -> str:
    """清洗并校验待渲染的 Markdown 文本。

    工具面向聊天场景使用，空字符串或超长文本几乎都属于模型误调用或提示词退化。
    这里在进入浏览器渲染前尽早拦截，避免把明显异常输入转成无意义的大图或导致
    截图阶段耗时过长。

    Args:
        markdown_text: Agent 传入的原始 Markdown 文本。

    Returns:
        去掉首尾空白后的 Markdown 文本。

    Raises:
        ValueError: 当输入为空或超过当前长度上限时抛出。
    """

    normalized = str(markdown_text or "").strip()
    if not normalized:
        raise ValueError("markdown_text 不能为空")
    if len(normalized) > int(_MAX_MARKDOWN_LENGTH):
        raise ValueError(f"markdown_text 不能超过 {int(_MAX_MARKDOWN_LENGTH)} 个字符")
    return normalized


def _build_result_text(*, file_id: str, render_result: MarkdownRenderResult) -> str:
    """构造工具成功后的简短返回文本。

    Args:
        file_id: 已注册进 GT 文件系统的稳定文件引用。
        render_result: 当前图片渲染结果。

    Returns:
        供 Agent 后续继续引用的成功摘要文本。
    """

    return (
        f"Markdown 图片已发送，GT文件={file_id}，"
        f"尺寸={int(render_result.width)}x{int(render_result.height)}"
    )


async def _send_rendered_image(
    *,
    runtime: ToolRuntime[GroupChatContext],
    image_path: Path,
) -> None:
    """发送已渲染图片，并兼容自动触发等缺少 `event` 的场景。

    自动触发与部分主动触发链路下，运行时上下文通常只有 `bot` 与 `transport`，
    不一定保留原始 `event`。这里优先走 GTBot 已经注入的 transport，把图片作为
    正常消息发出；只有 transport 不可用时，才回退到传统 `bot.send(event=...)`
    发送方式。

    Args:
        runtime: 当前工具运行时。
        image_path: 待发送图片的本地路径。

    Raises:
        ValueError: 当既没有可用 transport，也没有可回退发送所需的 `bot/event`
            时抛出。
    """

    context = runtime.context
    image_message = Message(MessageSegment.image(file=image_path.as_posix()))

    transport = getattr(context, "transport", None)
    send_messages = getattr(transport, "send_messages", None)
    if callable(send_messages):
        typed_send_messages = cast(_SendMessagesCallable, send_messages)
        await typed_send_messages([image_message])
        return

    bot = getattr(context, "bot", None)
    event = getattr(context, "event", None)
    if bot is None or event is None:
        raise ValueError("runtime.context 缺少可用 transport，且 bot/event 不完整，无法发送 Markdown 图片")
    await bot.send(
        event=event,
        message=image_message,
    )


@tool("send_markdown_image")
async def send_markdown_image(
    markdown_text: str,
    runtime: ToolRuntime[GroupChatContext],
) -> str:
    """把 Markdown 渲染成图片并发送到当前会话。

    该工具主要用于“把公式本体或代码本体单独发成图片”。当回答里同时包含
    解释文字和公式/代码时，优先做下面这种拆分：
    - 解释、结论、口语化说明：继续正常直接发消息。
    - 数学公式、代码块、Mermaid 图、复杂表格：单独整理成 Markdown 后调用本工具。

    也就是说，不要把整段回答连同大段解释一起塞进图片；更推荐“先正常发解释，
    再单独发公式图片/代码图片”，或者“先发一句提示，再发图片”。这样更符合聊天
    场景，也更容易阅读。

    反过来，如果只是简短口语回复、普通问答、闲聊、单句说明或几行以内的简单文字，
    则不应滥用本工具，仍应直接按正常消息发送。

    工具会在本地渲染 PNG、将文件注册为 GTFile，并立刻通过当前会话发送图片；
    优先复用 GTBot 的 transport，因此在自动触发等没有原始 `event` 的场景下也
    仍然可以正常工作。调用成功后，用户端能直接看到成图，而 Agent 也能拿到
    可复用的 `gfid:`。

    Args:
        markdown_text: 需要渲染为图片的 Markdown 文本。更适合传入“公式本体、
            代码本体、表格本体或 Mermaid fenced code”这类需要单独展示的内容，
            而不是把大段解释文字和公式混在一起整体截图。

    Returns:
        一段包含发送结果与 GT 文件引用的摘要文本。

    Raises:
        ValueError: 当 Markdown 文本为空，或当前运行时既缺少可用 transport，
            又缺少可回退发送所需的 `bot/event` 时抛出。
        RuntimeError: 当图片渲染失败，或 Chromium 运行环境不可用时抛出。
    """

    normalized_markdown = _normalize_markdown_text(markdown_text)
    plugin_cfg = get_markdown_image_plugin_config()
    render_cfg = plugin_cfg.render
    context = runtime.context

    render_result = await render_markdown_to_image(
        normalized_markdown,
        output_dir=_get_markdown_image_dir(),
        theme_base_dir=PLUGIN_DIR,
        width=render_cfg.width,
        auto_width=bool(render_cfg.auto_width),
        min_width=int(render_cfg.min_width),
        max_width=int(render_cfg.max_width),
        padding=int(render_cfg.padding),
        scale=float(render_cfg.scale),
        theme=str(render_cfg.theme),
        code_theme=str(render_cfg.code_theme),
        custom_css=str(render_cfg.custom_css or "").strip() or None,
    )
    file_id = register_local_file(
        render_result.image_path,
        kind="markdown_image",
        source_type="markdown_image_render",
        session_id=str(getattr(context, "session_id", "") or "").strip() or None,
        group_id=getattr(context, "group_id", None),
        user_id=getattr(context, "user_id", None),
        mime_type="image/png",
        original_name=Path(render_result.image_path).name,
        extra={
            "render_width": int(render_result.width),
            "render_height": int(render_result.height),
            "theme": str(render_cfg.theme),
            "code_theme": str(render_cfg.code_theme),
        },
        expires_at=float(time.time()) + float(_MARKDOWN_IMAGE_FILE_REF_TTL_SEC),
    )
    await _send_rendered_image(
        runtime=runtime,
        image_path=render_result.image_path,
    )
    return _build_result_text(file_id=file_id, render_result=render_result)
