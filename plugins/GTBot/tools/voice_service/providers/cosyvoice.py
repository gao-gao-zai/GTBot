from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from ..audio_utils import VoiceServiceError, make_cache_path, normalize_remote_url
from ..builtin_voices import ALIYUN_COSYVOICE_BUILTIN_VOICES
from ..config import VoiceServicePluginConfig
from ..models import AudioResult, CloneResult, ReplyVoiceMessage, SessionContext, TranscriptResult, VoiceItem
from ..state import SessionVoiceState, get_voice_state_store
from .base import BaseVoiceProvider

try:
    import dashscope
    from dashscope import Files
    from dashscope.audio.tts_v2 import VoiceEnrollmentService
except Exception:  # noqa: BLE001
    dashscope = None
    Files = None
    VoiceEnrollmentService = None


AUDIO_URL_RE = re.compile(r"https?://[^\s<>\"]+")
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".amr", ".wma", ".pcm", ".silk"}


class AliyunCosyVoiceProvider(BaseVoiceProvider):
    def __init__(self, cfg: VoiceServicePluginConfig) -> None:
        self.cfg = cfg

    def _require_api_key(self) -> str:
        api_key = self.cfg.aliyun.api_key.strip()
        if not api_key:
            raise VoiceServiceError("阿里云 API Key 未配置")
        return api_key

    def _ws_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_api_key()}"}

    def _require_dashscope_sdk(self) -> None:
        if dashscope is None or Files is None or VoiceEnrollmentService is None:
            raise VoiceServiceError("CosyVoice 文件上传与克隆依赖 dashscope SDK，请先安装 dashscope")

    @staticmethod
    def _sanitize_prefix(value: str) -> str:
        text = re.sub(r"[^0-9a-z]+", "", value.strip().lower())
        if not text:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
            text = f"v{digest}"
        return text[:10]

    @staticmethod
    def _looks_like_audio_url(value: str) -> bool:
        try:
            suffix = Path(httpx.URL(value).path).suffix.lower()
        except Exception:
            return False
        return suffix in AUDIO_SUFFIXES

    @staticmethod
    def _extract_text_segments(raw_message: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        message = raw_message.get("message")
        if isinstance(message, list):
            for segment in message:
                if not isinstance(segment, dict):
                    continue
                if segment.get("type") != "text":
                    continue
                data = segment.get("data")
                if isinstance(data, dict):
                    text = data.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
        raw = raw_message.get("raw_message")
        if isinstance(raw, str) and raw.strip():
            texts.append(raw.strip())
        return texts

    @staticmethod
    def _extract_ws_error(event: Any) -> str:
        if not isinstance(event, dict):
            return f"阿里云 CosyVoice 实时合成失败: {event!r}"
        header = event.get("header")
        payload = event.get("payload")
        candidates: list[str] = []
        for container in (payload, header, event):
            if not isinstance(container, dict):
                continue
            for key in ("message", "error_message", "errorMsg", "msg", "code", "error_code", "request_id"):
                value = container.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    candidates.append(f"{key}={text}")
        if candidates:
            return "阿里云 CosyVoice 实时合成失败: " + ", ".join(dict.fromkeys(candidates))
        compact = json.dumps(event, ensure_ascii=False)
        if len(compact) > 400:
            compact = compact[:400] + "..."
        return f"阿里云 CosyVoice 实时合成失败: {compact}"

    async def _validate_audio_url(self, url: str) -> str | None:
        normalized = normalize_remote_url(url)
        if not normalized:
            return None
        if self._looks_like_audio_url(normalized):
            return normalized

        async with httpx.AsyncClient(timeout=self.cfg.aliyun.timeout_sec, follow_redirects=True) as client:
            try:
                response = await client.head(normalized)
                content_type = response.headers.get("content-type", "").lower()
                if response.status_code < 400 and content_type.startswith("audio/"):
                    return normalized
            except Exception:
                pass
            try:
                response = await client.get(normalized, headers={"Range": "bytes=0-0"})
                content_type = response.headers.get("content-type", "").lower()
                if response.status_code < 400 and content_type.startswith("audio/"):
                    return normalized
            except Exception:
                return None
        return None

    async def _upload_clone_file(self, path: Path) -> str:
        if not path.exists():
            raise VoiceServiceError(f"CosyVoice 克隆上传文件不存在: {path}")
        self._require_dashscope_sdk()
        api_key = self._require_api_key()

        def _run_upload() -> str:
            assert dashscope is not None
            assert Files is not None
            dashscope.api_key = api_key
            upload_resp = Files.upload(file_path=str(path), purpose="voice_clone")
            uploaded = (getattr(upload_resp, "output", None) or {}).get("uploaded_files") or []
            if not uploaded:
                raise VoiceServiceError("CosyVoice 文件上传失败：未返回 uploaded_files")
            file_id = uploaded[0].get("file_id")
            if not file_id:
                raise VoiceServiceError("CosyVoice 文件上传失败：未返回 file_id")
            file_info = Files.get(file_id)
            url = (getattr(file_info, "output", None) or {}).get("url")
            if not isinstance(url, str) or not url.strip():
                raise VoiceServiceError("CosyVoice 文件上传成功，但未返回可用 URL")
            return url.strip()

        return await asyncio.to_thread(_run_upload)

    async def _resolve_clone_audio_url(self, reply_voice: ReplyVoiceMessage) -> str:
        if reply_voice.record_url:
            validated = await self._validate_audio_url(reply_voice.record_url)
            if validated:
                return validated

        for text in self._extract_text_segments(reply_voice.raw_message):
            for candidate in AUDIO_URL_RE.findall(text):
                validated = await self._validate_audio_url(candidate)
                if validated:
                    return validated

        upload_source = reply_voice.normalized_wav_path or reply_voice.source_path
        if not upload_source:
            raise VoiceServiceError("CosyVoice 克隆既没有可用音频链接，也没有可上传的本地音频文件")
        return await self._upload_clone_file(Path(upload_source))

    async def _create_voice_via_sdk(self, *, preferred_name: str, target_model: str, audio_url: str) -> str:
        self._require_dashscope_sdk()
        api_key = self._require_api_key()
        prefix = self._sanitize_prefix(preferred_name)

        def _run_create() -> str:
            assert dashscope is not None
            assert VoiceEnrollmentService is not None
            dashscope.api_key = api_key
            service = VoiceEnrollmentService()
            clone_task_id = service.create_voice(
                target_model=target_model,
                prefix=prefix,
                url=audio_url,
                language_hints=["zh"],
            )
            for _ in range(60):
                info = service.query_voice(clone_task_id)
                status = str(info.get("status") or "").upper()
                if status == "OK":
                    voice_id = str(info.get("voice_id") or "").strip()
                    if not voice_id:
                        raise VoiceServiceError("CosyVoice 克隆成功但未返回 voice_id")
                    return voice_id
                if status in {"UNDEPLOYED", "FAILED"}:
                    raise VoiceServiceError(f"CosyVoice 克隆失败，状态: {status}")
                time.sleep(10)
            raise VoiceServiceError("CosyVoice 克隆超时，请稍后重试")

        return await asyncio.to_thread(_run_create)

    async def list_voices(self, session: SessionContext, state: SessionVoiceState) -> list[VoiceItem]:
        builtin = [
            item.model_copy(deep=True)
            for item in ALIYUN_COSYVOICE_BUILTIN_VOICES
            if not item.target_model or item.target_model == self.cfg.aliyun.cosyvoice.tts_builtin_model
        ]
        builtin.extend(await self._list_custom_voices(state))
        return builtin

    async def _list_custom_voices(self, state: SessionVoiceState) -> list[VoiceItem]:
        if dashscope is None or VoiceEnrollmentService is None:
            return []

        api_key = self._require_api_key()
        alias_map = await get_voice_state_store().get_cosyvoice_custom_voice_aliases()

        def _run_list() -> list[VoiceItem]:
            assert dashscope is not None
            assert VoiceEnrollmentService is not None
            dashscope.api_key = api_key
            service = VoiceEnrollmentService()
            result = service.list_voices(prefix="")
            if isinstance(result, list):
                voice_list = result
            elif isinstance(result, dict):
                voice_list = result.get("voices") or result.get("voice_list") or []
            else:
                voice_list = []

            items: list[VoiceItem] = []
            if not isinstance(voice_list, list):
                return items

            for item in voice_list:
                if not isinstance(item, dict):
                    continue
                voice_id = str(item.get("voice_id") or item.get("voice") or "").strip()
                if not voice_id:
                    continue
                display_name = str(
                    item.get("name")
                    or item.get("display_name")
                    or item.get("title")
                    or item.get("prefix")
                    or ""
                ).strip()
                alias = alias_map.get(voice_id)
                if alias is not None and alias.voice_name:
                    display_name = alias.voice_name
                if state.cosyvoice.current_voice_id == voice_id and state.cosyvoice.current_voice_name:
                    display_name = state.cosyvoice.current_voice_name
                if not display_name:
                    display_name = voice_id
                target_model = str(item.get("target_model") or "").strip() or (alias.target_model if alias else None)
                items.append(
                    VoiceItem(
                        provider="aliyun_cosyvoice",
                        name=display_name,
                        display_name=display_name,
                        voice_type="custom",
                        voice_id=voice_id,
                        target_model=target_model,
                        created_at=str(item.get("gmt_create") or item.get("created_at") or "") or None,
                    )
                )
            return items

        return await asyncio.to_thread(_run_list)

    async def synthesize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        text: str,
        *,
        style_text: str | None = None,
        rate: float | None = None,
        pitch: float | None = None,
        volume: int | None = None,
    ) -> AudioResult:
        cosy_cfg = self.cfg.aliyun.cosyvoice
        voice_name = state.cosyvoice.current_voice_id or state.cosyvoice.current_voice_name
        voice_type = state.cosyvoice.current_voice_type
        if not voice_name or voice_type is None:
            raise VoiceServiceError("阿里云 CosyVoice 模式尚未设置当前音色")

        if voice_type == "custom":
            model = state.cosyvoice.current_target_model or cosy_cfg.tts_custom_model
        else:
            model = cosy_cfg.tts_builtin_model

        if voice_type == "custom" and state.cosyvoice.current_target_model and model != state.cosyvoice.current_target_model:
            raise VoiceServiceError("当前 CosyVoice 自定义音色与配置中的合成模型不一致，请调整配置")

        audio_bytes = await self._synthesize_ws(
            model=model,
            voice=voice_name,
            text=text,
            style_text=style_text,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        suffix = ".wav" if cosy_cfg.response_format == "wav" else f".{cosy_cfg.response_format}"
        audio_path = make_cache_path(self.cfg, suffix)
        await asyncio.to_thread(audio_path.write_bytes, audio_bytes)
        return AudioResult(
            provider="aliyun_cosyvoice",
            delivery="onebot_record",
            voice_name=voice_name,
            audio_path=str(audio_path),
            mime_type=f"audio/{cosy_cfg.response_format}",
        )

    async def _synthesize_ws(
        self,
        *,
        model: str,
        voice: str,
        text: str,
        style_text: str | None = None,
        rate: float | None = None,
        pitch: float | None = None,
        volume: int | None = None,
    ) -> bytes:
        cosy_cfg = self.cfg.aliyun.cosyvoice
        url = self.cfg.aliyun_cosyvoice_ws_base
        task_id = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        collected: list[bytes] = []
        started = False

        try:
            async with websockets.connect(
                url,
                additional_headers=self._ws_headers(),
                open_timeout=self.cfg.aliyun.timeout_sec,
                max_size=None,
            ) as websocket:
                parameters: dict[str, Any] = {
                    "text_type": "PlainText",
                    "voice": voice,
                    "format": cosy_cfg.response_format,
                    "sample_rate": cosy_cfg.sample_rate,
                }
                if rate is not None:
                    parameters["rate"] = rate
                if pitch is not None:
                    parameters["pitch"] = pitch
                if volume is not None:
                    parameters["volume"] = volume

                payload: dict[str, Any] = {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": model,
                    "parameters": parameters,
                    "input": {},
                }
                if isinstance(style_text, str) and style_text.strip():
                    payload["instruction"] = style_text.strip()

                await websocket.send(
                    json.dumps(
                        {
                            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                            "payload": payload,
                        }
                    )
                )

                while True:
                    raw = await websocket.recv()
                    if isinstance(raw, bytes):
                        collected.append(raw)
                        continue

                    event = json.loads(raw)
                    header = event.get("header") if isinstance(event, dict) else {}
                    event_name = str((header or {}).get("event") or "").strip()

                    if event_name == "task-started" and not started:
                        started = True
                        await websocket.send(
                            json.dumps(
                                {
                                    "header": {"action": "continue-task", "task_id": task_id, "request_id": request_id},
                                    "payload": {"input": {"text": text}},
                                }
                            )
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "header": {"action": "finish-task", "task_id": task_id, "request_id": request_id},
                                    "payload": {"input": {}},
                                }
                            )
                        )
                    elif event_name == "task-finished":
                        break
                    elif event_name in {"task-failed", "result-error"}:
                        raise VoiceServiceError(self._extract_ws_error(event))
        except ConnectionClosed as exc:
            raise VoiceServiceError(
                f"阿里云 CosyVoice WebSocket 已关闭: code={exc.code} reason={exc.reason or 'unknown'}"
            ) from exc

        if not collected:
            raise VoiceServiceError("阿里云 CosyVoice 实时合成未返回音频")
        return b"".join(collected)

    async def recognize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
    ) -> TranscriptResult:
        raise VoiceServiceError("阿里云 CosyVoice 模式不支持语音识别，请切换到阿里云 Qwen 或 QQ 模式")

    async def clone_voice(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
        preferred_name: str,
        *,
        target_model: str | None = None,
    ) -> CloneResult:
        resolved_target_model = (target_model or self.cfg.aliyun.cosyvoice.tts_custom_model).strip()
        if not resolved_target_model:
            raise VoiceServiceError("阿里云 CosyVoice 克隆目标模型不能为空")

        audio_url = await self._resolve_clone_audio_url(reply_voice)
        voice_id = await self._create_voice_via_sdk(
            preferred_name=preferred_name,
            target_model=resolved_target_model,
            audio_url=audio_url,
        )
        return CloneResult(
            provider="aliyun_cosyvoice",
            voice_name=preferred_name,
            voice_id=voice_id,
            target_model=resolved_target_model,
        )
