from __future__ import annotations

import asyncio
from pathlib import Path
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from nonebot import logger

from plugins.GTBot.ConfigManager import total_config
from plugins.GTBot.services.chat.group_queue import group_message_queue_manager
from plugins.GTBot.services.chat.private_queue import PrivateMessageTask, private_message_queue_manager
from plugins.GTBot.services.chat.queue_payload import prepare_queue_messages
from plugins.GTBot.model import MessageTask

from .config import get_comfyui_draw_plugin_config

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from plugins.GTBot.services import cache as CacheManager
    from plugins.GTBot.services.message import GroupMessageManager

    BotT = Bot
    GroupMessageManagerT = GroupMessageManager
    UserCacheManagerT = CacheManager.UserCacheManager
else:
    BotT = Any
    GroupMessageManagerT = Any
    UserCacheManagerT = Any


@dataclass(frozen=True, slots=True)
class DrawJobSpec:
    chat_type: str
    session_id: str
    prompt: str
    width: int
    height: int
    seed: int
    group_id: int | None
    requester_user_id: int
    target_user_id: int
    bot: BotT
    message_manager: GroupMessageManagerT
    cache: UserCacheManagerT


@dataclass(slots=True)
class DrawJobState:
    job_id: str
    spec: DrawJobSpec
    created_at: float
    status: str = "queued"
    comfy_job_id: str | None = None
    error: str | None = None
    result_image_path: str | None = None


class DrawQueueManager:
    def __init__(self) -> None:
        cfg = get_comfyui_draw_plugin_config()
        self._queue: asyncio.Queue[DrawJobState] = asyncio.Queue(maxsize=int(cfg.max_queue_size))
        self._workers_started = False

        self._lock = asyncio.Lock()
        self._running: dict[str, DrawJobState] = {}
        self._queued_order: list[str] = []
        self._queued: dict[str, DrawJobState] = {}

    async def start_workers(self) -> None:
        if self._workers_started:
            return
        self._workers_started = True

        cfg = get_comfyui_draw_plugin_config()
        n = int(cfg.worker_concurrency)
        for i in range(max(1, n)):
            asyncio.create_task(self._worker_loop(worker_idx=i))

    async def submit(self, spec: DrawJobSpec) -> DrawJobState:
        await self.start_workers()

        created_at = float(time.time())
        job_id = self._new_job_id(created_at)
        state = DrawJobState(job_id=job_id, spec=spec, created_at=created_at)

        async with self._lock:
            if self._queue.full():
                raise RuntimeError("画图队列已满，请稍后再试")
            self._queued_order.append(job_id)
            self._queued[job_id] = state

        await self._queue.put(state)
        return state

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            running = list(self._running.values())
            queued = [self._queued[jid] for jid in self._queued_order if jid in self._queued]
            queued_count = len(self._queued_order)

        return {
            "running": [self._state_to_brief(x) for x in running],
            "queued": [self._state_to_brief(x) for x in queued],
            "running_count": len(running),
            "queued_count": queued_count,
            "queue_max": int(self._queue.maxsize),
        }

    def _new_job_id(self, ts: float) -> str:
        rnd = random.randint(0, 9999)
        return f"draw_{int(ts)}_{rnd:04d}"

    def _state_to_brief(self, s: DrawJobState) -> dict[str, Any]:
        prompt_preview = (s.spec.prompt or "").strip().replace("\n", " ")[:80]
        return {
            "job_id": s.job_id,
            "status": s.status,
            "created_at": s.created_at,
            "elapsed_sec": max(0.0, float(time.time()) - float(s.created_at)),
            "chat_type": s.spec.chat_type,
            "session_id": s.spec.session_id,
            "group_id": s.spec.group_id,
            "requester_user_id": s.spec.requester_user_id,
            "target_user_id": s.spec.target_user_id,
            "width": s.spec.width,
            "height": s.spec.height,
            "seed": s.spec.seed,
            "prompt_preview": prompt_preview,
        }

    async def _worker_loop(self, *, worker_idx: int) -> None:
        while True:
            state = await self._queue.get()
            try:
                await self._run_one(state=state, worker_idx=worker_idx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ComfyUIDraw worker 运行异常: %s: %s",
                    type(exc).__name__,
                    str(exc),
                    exc_info=True,
                )
            finally:
                self._queue.task_done()

    async def _run_one(self, *, state: DrawJobState, worker_idx: int) -> None:
        _ = int(worker_idx)
        async with self._lock:
            if state.job_id in self._queued_order:
                self._queued_order.remove(state.job_id)
            self._queued.pop(state.job_id, None)
            self._running[state.job_id] = state
            state.status = "running"

        try:
            await self._execute_comfyui_job(state)
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
            logger.warning(f"ComfyUIDraw 通知发送失败: {type(exc).__name__}: {exc!s}")

    async def _execute_comfyui_job(self, state: DrawJobState) -> None:
        cfg = get_comfyui_draw_plugin_config()
        base = str(cfg.base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("ComfyUIDraw 配置缺少 base_url")

        timeout = float(cfg.timeout_sec)
        poll_interval = float(cfg.poll_interval_sec)
        max_wait = float(cfg.max_wait_sec)

        submit_url = f"{base}/v1/jobs"

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.post(
                submit_url,
                json={
                    "prompt": state.spec.prompt,
                    "seed": int(state.spec.seed),
                    "width": int(state.spec.width),
                    "height": int(state.spec.height),
                },
            )
            resp.raise_for_status()
            data = resp.json()

            qd: Any = data

            comfy_job_id = data.get("job_id")
            if not isinstance(comfy_job_id, str) or not comfy_job_id.strip():
                raise RuntimeError("ComfyUI API 返回缺少 job_id")

            state.comfy_job_id = comfy_job_id

            status = data.get("status")
            start = float(time.time())

            query_url = f"{base}/v1/jobs/{comfy_job_id}"
            while status in ("queued", "running"):
                if float(time.time()) - start > max_wait:
                    raise TimeoutError("等待 ComfyUI 出图超时")

                await asyncio.sleep(poll_interval)
                q = await client.get(query_url)
                q.raise_for_status()
                qd = q.json()
                status = qd.get("status")

            if status != "succeeded":
                detail = qd if isinstance(qd, dict) else {"status": status}
                state.error = str(detail)
                return

            images = qd.get("images")
            if not isinstance(images, list) or not images:
                state.error = "ComfyUI API succeeded 但 images 为空"
                return

            first = images[0]
            if not isinstance(first, str) or not first.strip():
                state.error = "ComfyUI API images[0] 非字符串"
                return

            url = f"{base}{first}"
            dl = await client.get(url)
            dl.raise_for_status()

            file_name = Path(first).name or f"{state.job_id}.png"
            saved = self._save_image_bytes(job_id=state.job_id, file_name=file_name, image_bytes=dl.content)
            state.result_image_path = str(saved)

    def _save_image_bytes(self, *, job_id: str, file_name: str, image_bytes: bytes) -> Path:
        data_dir = total_config.get_data_dir_path()
        out_dir = data_dir / "comfyui_draw" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_name = str(file_name or "").strip() or f"{job_id}.png"
        target = out_dir / safe_name
        if target.exists():
            target = out_dir / f"{job_id}_{safe_name}"

        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(image_bytes)
        tmp.replace(target)
        return target

    async def _notify(self, state: DrawJobState) -> None:
        if state.spec.chat_type == "private":
            await self._notify_private(state)
            return
        await self._notify_group(state)

    async def _notify_group(self, state: DrawJobState) -> None:
        """向群聊会话发送绘图完成或失败通知。

        Args:
            state: 当前绘图任务的运行状态。
        """
        group_id = int(state.spec.group_id or 0)
        target_user_id = int(state.spec.target_user_id)
        if group_id <= 0:
            return

        if state.status == "succeeded" and state.result_image_path:
            image_cq = f"[CQ:image,file={state.result_image_path}]"
            text = (
                f"[绘图完成] job={state.job_id} "
                f"seed={state.spec.seed} size={state.spec.width}x{state.spec.height}"
            )
            messages = [f"[CQ:at,qq={target_user_id}] {text}", image_cq]
        else:
            err = (state.error or "未知错误").strip()
            err = err.replace("\n", " ")[:300]
            messages = [
                f"[CQ:at,qq={target_user_id}] [绘图失败] job={state.job_id} error={err}",
            ]

        prepared_messages = await prepare_queue_messages(
            messages,
            scope=f"群组 {group_id}",
        )
        task = MessageTask(messages=prepared_messages, group_id=group_id, interval=0.2)
        await group_message_queue_manager.enqueue(
            task,
            bot=state.spec.bot,
            message_manager=state.spec.message_manager,
            cache=state.spec.cache,
        )

    async def _notify_private(self, state: DrawJobState) -> None:
        """向私聊会话发送绘图完成或失败通知。

        Args:
            state: 当前绘图任务的运行状态。
        """
        target_user_id = int(state.spec.target_user_id)
        if target_user_id <= 0:
            return

        if state.status == "succeeded" and state.result_image_path:
            messages = [
                (
                    f"[绘图完成] job={state.job_id} "
                    f"seed={state.spec.seed} size={state.spec.width}x{state.spec.height}"
                ),
                f"[CQ:image,file={state.result_image_path}]",
            ]
        else:
            err = (state.error or "未知错误").strip()
            err = err.replace("\n", " ")[:300]
            messages = [f"[绘图失败] job={state.job_id} error={err}"]

        prepared_messages = await prepare_queue_messages(
            messages,
            scope=f"session private:{target_user_id}",
        )
        task = PrivateMessageTask(
            messages=prepared_messages,
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


_draw_queue_manager: DrawQueueManager | None = None


def get_draw_queue_manager() -> DrawQueueManager:
    global _draw_queue_manager
    if _draw_queue_manager is None:
        _draw_queue_manager = DrawQueueManager()
    return _draw_queue_manager
