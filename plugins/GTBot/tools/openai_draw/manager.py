from __future__ import annotations

import asyncio
import base64
import binascii
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast
from urllib.parse import urlparse

import httpx
from nonebot import logger

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.model import MessageTask
from plugins.GTBot.services.chat.group_queue import group_message_queue_manager
from plugins.GTBot.services.chat.private_queue import PrivateMessageTask, private_message_queue_manager
from plugins.GTBot.services.chat.queue_payload import prepare_queue_messages
from plugins.GTBot.services.file_registry import register_local_file

from .client import OpenAIDrawClient, OpenAIDrawClientError
from .config import get_openai_draw_plugin_config
from .usage_limits import get_openai_draw_usage_limit_manager

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from plugins.GTBot.services import cache as CacheManager
    from plugins.GTBot.services.message import GroupMessageManager

    BotT: TypeAlias = Bot
    GroupMessageManagerT: TypeAlias = GroupMessageManager
    UserCacheManagerT: TypeAlias = CacheManager.UserCacheManager
else:
    BotT: TypeAlias = Any
    GroupMessageManagerT: TypeAlias = Any
    UserCacheManagerT: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class OpenAIInputImage:
    """描述绘图或编辑图任务中携带的一张输入图片。

    Attributes:
        file_name: 上传到上游接口时使用的文件名。
        image_bytes: 图片原始二进制内容。
    """

    file_name: str
    image_bytes: bytes


@dataclass(frozen=True, slots=True)
class OpenAIDrawJobSpec:
    """描述一次待执行的绘图任务请求。

    Attributes:
        chat_type: 当前会话类型，通常为 `group` 或 `private`。
        session_id: 会话标识，用于日志和调试。
        prompt: 文生图提示词。
        size: 图片尺寸。
        quality: 图片质量参数。
        background: 背景参数。
        output_format: 图片输出格式。
        mode: 当前任务模式，支持 `generate` 和 `edit`。
        input_images: 编辑图时上传给上游接口的输入图片列表。
        group_id: 群聊场景下的群号；私聊时为空。
        requester_user_id: 发起任务的用户。
        target_user_id: 接收绘图结果的用户。
        bot: 当前机器人实例。
        message_manager: GTBot 的消息管理器。
        cache: GTBot 的缓存管理器。
    """

    chat_type: str
    session_id: str
    prompt: str
    size: str
    quality: str
    background: str
    output_format: str
    group_id: int | None
    requester_user_id: int
    target_user_id: int
    bot: BotT
    message_manager: GroupMessageManagerT
    cache: UserCacheManagerT
    mode: str = "generate"
    input_images: tuple[OpenAIInputImage, ...] = ()


@dataclass(slots=True)
class OpenAIDrawJobState:
    """保存绘图任务的运行状态与产物路径。

    Attributes:
        job_id: 插件生成的本地任务 ID。
        spec: 提交时的任务请求参数。
        created_at: 任务进入队列的时间戳。
        status: 当前状态，包含 `queued`、`running`、`succeeded`、`failed`。
        error: 失败时的摘要错误信息。
        result_image_path: 生成完成后落盘的本地图片路径。
        revised_prompt: 服务端返回的修订提示词。
    """

    job_id: str
    spec: OpenAIDrawJobSpec
    created_at: float
    status: str = "queued"
    error: str | None = None
    result_image_path: str | None = None
    result_file_id: str | None = None
    revised_prompt: str | None = None


class OpenAIDrawQueueManager:
    """管理 OpenAI 绘图任务的排队、执行与结果通知。

    该管理器使用进程内异步队列串联任务提交与后台消费，避免一次对话阻塞到出图完成。
    结果统一复用 GTBot 现有的群聊/私聊消息队列，保证与宿主的发消息节奏一致。
    """

    def __init__(self) -> None:
        """初始化任务队列与运行时状态容器。"""

        cfg = get_openai_draw_plugin_config()
        self._queue: asyncio.Queue[OpenAIDrawJobState] = asyncio.Queue(maxsize=int(cfg.max_queue_size))
        self._workers_started = False
        self._lock = asyncio.Lock()
        self._running: dict[str, OpenAIDrawJobState] = {}
        self._queued_order: list[str] = []
        self._queued: dict[str, OpenAIDrawJobState] = {}
        self._usage_limits = get_openai_draw_usage_limit_manager()

    async def start_workers(self) -> None:
        """按当前配置启动后台消费者。

        重复调用是安全的；只有首次调用会真正创建 worker。
        """

        if self._workers_started:
            return
        self._workers_started = True

        cfg = get_openai_draw_plugin_config()
        for worker_idx in range(max(1, int(cfg.worker_concurrency))):
            asyncio.create_task(self._worker_loop(worker_idx=worker_idx))

    async def submit(self, spec: OpenAIDrawJobSpec) -> OpenAIDrawJobState:
        """提交一条新的绘图任务到后台队列。

        Args:
            spec: 当前任务的完整请求参数。

        Returns:
            刚创建的任务状态对象。

        Raises:
            RuntimeError: 当队列已满时抛出。
        """

        await self.start_workers()

        created_at = float(time.time())
        job_id = self._new_job_id(created_at)
        state = OpenAIDrawJobState(job_id=job_id, spec=spec, created_at=created_at)
        cfg = get_openai_draw_plugin_config()

        async with self._lock:
            if self._queue.full():
                raise RuntimeError("绘图队列已满，请稍后再试")
            self._usage_limits.ensure_can_submit(
                cfg=cfg,
                user_id=int(spec.requester_user_id),
                now_ts=created_at,
            )
            self._queued_order.append(job_id)
            self._queued[job_id] = state
            self._queue.put_nowait(state)
            self._usage_limits.record_submission(
                cfg=cfg,
                user_id=int(spec.requester_user_id),
                now_ts=created_at,
            )

        return state

    async def snapshot(self) -> dict[str, Any]:
        """返回当前队列与运行中任务的摘要视图。

        Returns:
            包含排队数、运行数和任务摘要的字典。
        """

        async with self._lock:
            running = list(self._running.values())
            queued = [self._queued[jid] for jid in self._queued_order if jid in self._queued]
            queued_count = len(self._queued_order)

        return {
            "running": [self._state_to_brief(item) for item in running],
            "queued": [self._state_to_brief(item) for item in queued],
            "running_count": len(running),
            "queued_count": queued_count,
            "queue_max": int(self._queue.maxsize),
        }

    def _new_job_id(self, created_at: float) -> str:
        """生成本地任务 ID。

        Args:
            created_at: 任务创建时间戳。

        Returns:
            适合作为日志和用户提示的短任务 ID。
        """

        return f"openai_draw_{int(created_at * 1000)}"

    def _state_to_brief(self, state: OpenAIDrawJobState) -> dict[str, Any]:
        """将内部状态对象压缩为队列展示摘要。

        Args:
            state: 当前任务状态。

        Returns:
            仅包含用户关心字段的摘要字典。
        """

        prompt_preview = state.spec.prompt.strip().replace("\n", " ")[:80]
        return {
            "job_id": state.job_id,
            "status": state.status,
            "created_at": state.created_at,
            "elapsed_sec": max(0.0, float(time.time()) - float(state.created_at)),
            "chat_type": state.spec.chat_type,
            "session_id": state.spec.session_id,
            "group_id": state.spec.group_id,
            "requester_user_id": state.spec.requester_user_id,
            "target_user_id": state.spec.target_user_id,
            "size": state.spec.size,
            "quality": state.spec.quality,
            "background": state.spec.background,
            "mode": state.spec.mode,
            "prompt_preview": prompt_preview,
            "result_file_id": state.result_file_id,
        }

    async def _worker_loop(self, *, worker_idx: int) -> None:
        """持续消费后台队列中的绘图任务。

        Args:
            worker_idx: 当前 worker 的序号，仅用于调试定位。
        """

        _ = worker_idx
        while True:
            state = await self._queue.get()
            try:
                await self._run_one(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("OpenAIDraw worker 运行异常: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()

    async def _run_one(self, state: OpenAIDrawJobState) -> None:
        """执行单个绘图任务并发送完成通知。

        Args:
            state: 当前任务状态对象。
        """

        async with self._lock:
            if state.job_id in self._queued_order:
                self._queued_order.remove(state.job_id)
            self._queued.pop(state.job_id, None)
            self._running[state.job_id] = state
            state.status = "running"

        try:
            await self._execute_openai_job(state)
            state.status = "succeeded" if state.result_image_path else "failed"
            if not state.result_image_path and not state.error:
                state.error = "未知错误：未获取到图片"
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc!s}"
        finally:
            async with self._lock:
                self._running.pop(state.job_id, None)

        try:
            await self._notify(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAIDraw 通知发送失败: %s", exc)

    async def _execute_openai_job(self, state: OpenAIDrawJobState) -> None:
        """调用 OpenAI Images API 并将结果保存到本地。

        Args:
            state: 当前任务状态对象。

        Raises:
            OpenAIDrawClientError: 当上游返回错误或结果不可用时抛出。
        """

        cfg = get_openai_draw_plugin_config()
        client = OpenAIDrawClient(cfg)
        if state.spec.mode == "edit":
            response = await client.edit_image(
                prompt=state.spec.prompt,
                images=[(item.file_name, item.image_bytes) for item in state.spec.input_images],
                size=state.spec.size,
                quality=state.spec.quality,
                background=state.spec.background,
                output_format=state.spec.output_format,
            )
        else:
            response = await client.generate_image(
                prompt=state.spec.prompt,
                size=state.spec.size,
                quality=state.spec.quality,
                background=state.spec.background,
                output_format=state.spec.output_format,
            )

        first = response.data[0]
        state.revised_prompt = first.revised_prompt

        if first.b64_json:
            image_bytes = self._decode_image_bytes(first.b64_json)
            saved = self._save_image_bytes(
                job_id=state.job_id,
                image_bytes=image_bytes,
                output_format=state.spec.output_format,
                source_name=None,
            )
            state.result_image_path = str(saved)
            state.result_file_id = register_local_file(
                saved,
                kind="draw_result",
                source_type="openai_draw",
                session_id=state.spec.session_id,
                group_id=state.spec.group_id,
                user_id=state.spec.target_user_id,
                original_name=saved.name,
                extra={"job_id": state.job_id, "mode": state.spec.mode},
            )
            return

        if first.url:
            image_bytes = await self._download_image_bytes(first.url, timeout_sec=float(cfg.timeout_sec))
            saved = self._save_image_bytes(
                job_id=state.job_id,
                image_bytes=image_bytes,
                output_format=state.spec.output_format,
                source_name=Path(urlparse(first.url).path).name,
            )
            state.result_image_path = str(saved)
            state.result_file_id = register_local_file(
                saved,
                kind="draw_result",
                source_type="openai_draw",
                session_id=state.spec.session_id,
                group_id=state.spec.group_id,
                user_id=state.spec.target_user_id,
                original_name=saved.name,
                extra={"job_id": state.job_id, "mode": state.spec.mode},
            )
            return

        raise OpenAIDrawClientError("绘图接口未返回可用图片内容")

    def _decode_image_bytes(self, payload: str) -> bytes:
        """将 Base64 图片内容解码为二进制字节。

        Args:
            payload: 接口返回的 Base64 字符串。

        Returns:
            图片原始字节内容。

        Raises:
            OpenAIDrawClientError: 当 Base64 内容非法时抛出。
        """

        try:
            return base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise OpenAIDrawClientError("绘图接口返回了非法的 Base64 图片数据") from exc

    async def _download_image_bytes(self, url: str, *, timeout_sec: float) -> bytes:
        """下载远程图片内容。

        Args:
            url: 图片下载地址。
            timeout_sec: 下载超时时间。

        Returns:
            下载到的图片字节。

        Raises:
            OpenAIDrawClientError: 当下载失败时抛出。
        """

        try:
            async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenAIDrawClientError(f"下载绘图结果失败: {exc!s}") from exc
        return cast(bytes, response.content)

    def _save_image_bytes(
        self,
        *,
        job_id: str,
        image_bytes: bytes,
        output_format: str,
        source_name: str | None,
    ) -> Path:
        """将图片字节保存到本地数据目录。

        Args:
            job_id: 当前任务 ID。
            image_bytes: 待保存的图片字节。
            output_format: 目标输出格式。
            source_name: 上游提供的原始文件名，可用于保留扩展名。

        Returns:
            最终保存后的文件路径。
        """

        cfg = get_openai_draw_plugin_config()
        out_dir = cfg.download_dir_path
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(str(source_name or "")).suffix.strip() if source_name else ""
        if not suffix:
            suffix = "." + str(output_format or "png").strip().lower().lstrip(".")

        safe_name = f"{job_id}{suffix}"
        target = out_dir / safe_name
        if target.exists():
            target = out_dir / f"{job_id}_{int(time.time())}{suffix}"

        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(image_bytes)
        tmp.replace(target)
        return target

    async def _notify(self, state: OpenAIDrawJobState) -> None:
        """根据会话类型分发通知。

        Args:
            state: 当前任务状态对象。
        """

        if state.spec.chat_type == "private":
            await self._notify_private(state)
            return
        await self._notify_group(state)

    async def _notify_group(self, state: OpenAIDrawJobState) -> None:
        """向群聊发送绘图完成或失败通知。

        Args:
            state: 当前任务状态对象。
        """

        group_id = int(state.spec.group_id or 0)
        target_user_id = int(state.spec.target_user_id)
        if group_id <= 0:
            return

        if state.status == "succeeded" and state.result_image_path:
            mode_label = "改图完成" if state.spec.mode == "edit" else "绘图完成"
            summary = (
                f"[{mode_label}] job={state.job_id} size={state.spec.size} "
                f"quality={state.spec.quality} background={state.spec.background}"
            )
            messages = [
                f"[CQ:at,qq={target_user_id}] {summary} file_id={state.result_file_id}",
                f"[CQ:image,file={state.result_image_path}]",
            ]
        else:
            err = (state.error or "未知错误").strip().replace("\n", " ")[:300]
            messages = [
                f"[CQ:at,qq={target_user_id}] [{'改图失败' if state.spec.mode == 'edit' else '绘图失败'}] job={state.job_id} error={err}",
            ]

        prepared = await prepare_queue_messages(messages, scope=f"群组 {group_id}")
        task = MessageTask(messages=prepared, group_id=group_id, interval=0.2)
        await group_message_queue_manager.enqueue(
            task,
            bot=state.spec.bot,
            message_manager=state.spec.message_manager,
            cache=state.spec.cache,
        )

    async def _notify_private(self, state: OpenAIDrawJobState) -> None:
        """向私聊发送绘图完成或失败通知。

        Args:
            state: 当前任务状态对象。
        """

        target_user_id = int(state.spec.target_user_id)
        if target_user_id <= 0:
            return

        if state.status == "succeeded" and state.result_image_path:
            mode_label = "改图完成" if state.spec.mode == "edit" else "绘图完成"
            messages = [
                (
                    f"[{mode_label}] job={state.job_id} size={state.spec.size} "
                    f"quality={state.spec.quality} background={state.spec.background} file_id={state.result_file_id}"
                ),
                f"[CQ:image,file={state.result_image_path}]",
            ]
        else:
            err = (state.error or "未知错误").strip().replace("\n", " ")[:300]
            messages = [f"[{'改图失败' if state.spec.mode == 'edit' else '绘图失败'}] job={state.job_id} error={err}"]

        prepared = await prepare_queue_messages(messages, scope=f"session private:{target_user_id}")
        task = PrivateMessageTask(
            messages=prepared,
            user_id=target_user_id,
            interval=0.2,
            session_id=f"private:{target_user_id}",
        )
        await private_message_queue_manager.enqueue(
            task,
            bot=state.spec.bot,
            message_manager=state.spec.message_manager,
            cache=state.spec.cache,
        )


_openai_draw_queue_manager: OpenAIDrawQueueManager | None = None


def get_openai_draw_queue_manager() -> OpenAIDrawQueueManager:
    """返回全局共享的绘图任务队列管理器。

    Returns:
        单例模式的队列管理器实例。
    """

    global _openai_draw_queue_manager
    if _openai_draw_queue_manager is None:
        _openai_draw_queue_manager = OpenAIDrawQueueManager()
    return _openai_draw_queue_manager
