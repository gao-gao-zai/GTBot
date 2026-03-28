from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
import websockets

from ..audio_utils import (
    VoiceServiceError,
    build_data_url,
    download_file,
    make_cache_path,
    suffix_from_url,
    transcode_wav_to_pcm_s16le,
)
from ..builtin_voices import ALIYUN_QWEN_BUILTIN_VOICES
from ..config import VoiceServicePluginConfig
from ..models import AudioResult, CloneResult, ReplyVoiceMessage, SessionContext, TranscriptResult, VoiceItem
from ..state import SessionVoiceState, get_voice_state_store
from .base import BaseVoiceProvider


class AliyunVoiceProvider(BaseVoiceProvider):
    def __init__(self, cfg: VoiceServicePluginConfig) -> None:
        self.cfg = cfg

    def _require_api_key(self) -> str:
        api_key = self.cfg.aliyun.api_key.strip()
        if not api_key:
            raise VoiceServiceError("阿里云 API Key 未配置")
        return api_key

    def _http_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }

    def _ws_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_api_key()}"}

    async def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.cfg.aliyun.timeout_sec, follow_redirects=True) as client:
            response = await client.post(url, json=payload, headers=headers or self._http_headers())
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text.strip()
                if detail:
                    raise VoiceServiceError(f"阿里云接口请求失败: HTTP {response.status_code} {detail}") from exc
                raise VoiceServiceError(f"阿里云接口请求失败: HTTP {response.status_code}") from exc
            parsed = response.json()
        if not isinstance(parsed, dict):
            raise VoiceServiceError(f"阿里云接口返回格式异常: {parsed!r}")
        return parsed

    @staticmethod
    def _sanitize_preferred_name(value: str) -> str:
        text = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
            text = f"voice_{digest}"
        if len(text) > 16:
            text = text[:16]
        return text

    async def list_voices(self, session: SessionContext, state: SessionVoiceState) -> list[VoiceItem]:
        voices = [item.model_copy(deep=True) for item in ALIYUN_QWEN_BUILTIN_VOICES]
        voices.extend(await self._list_custom_voices(state))
        return voices

    async def _list_custom_voices(self, state: SessionVoiceState) -> list[VoiceItem]:
        alias_map = await get_voice_state_store().get_qwen_custom_voice_aliases()
        url = f"{self.cfg.aliyun_http_base}/services/audio/tts/customization"
        payload = {
            "model": self.cfg.aliyun.qwen.voice_clone_model,
            "input": {"action": "list", "page_size": 100, "page_index": 0},
        }
        result = await self._post_json(url=url, payload=payload)
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        voice_list = output.get("voice_list") if isinstance(output, dict) else None
        items: list[VoiceItem] = []
        if not isinstance(voice_list, list):
            return items
        for item in voice_list:
            if not isinstance(item, dict):
                continue
            voice_id = str(item.get("voice") or "").strip()
            if not voice_id:
                continue
            preferred_name = str(
                item.get("preferred_name")
                or item.get("name")
                or item.get("display_name")
                or item.get("title")
                or ""
            ).strip()
            alias = alias_map.get(voice_id)
            voice_name = (alias.voice_name if alias is not None else "") or preferred_name or voice_id
            if state.qwen.current_voice_id == voice_id and state.qwen.current_voice_name:
                voice_name = state.qwen.current_voice_name
            items.append(
                VoiceItem(
                    provider="aliyun_qwen",
                    name=voice_name,
                    display_name=voice_name,
                    voice_type="custom",
                    voice_id=voice_id,
                    target_model=(str(item.get("target_model") or "").strip() or (alias.target_model if alias else None)),
                    created_at=str(item.get("gmt_create") or "") or None,
                )
            )
        return items

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
        if style_text is not None or rate is not None or pitch is not None or volume is not None:
            raise VoiceServiceError("阿里云 Qwen 模式当前不支持直接设置语速、语调或音量参数")
        voice_name = state.qwen.current_voice_id or state.qwen.current_voice_name
        voice_type = state.qwen.current_voice_type
        qwen_cfg = self.cfg.aliyun.qwen

        if not voice_name or voice_type is None:
            raise VoiceServiceError("阿里云 Qwen 模式尚未设置当前音色")

        if voice_type == "custom":
            if state.qwen.current_target_model and state.qwen.current_target_model != qwen_cfg.tts_custom_model:
                raise VoiceServiceError("当前克隆音色的 target_model 与配置中的 tts_custom_model 不一致，请先调整配置")
            model = qwen_cfg.tts_custom_model
        else:
            model = qwen_cfg.tts_builtin_model

        if "realtime" in model:
            audio_bytes = await self._synthesize_realtime(model=model, voice=voice_name, text=text)
            suffix = ".wav" if qwen_cfg.response_format == "wav" else f".{qwen_cfg.response_format}"
            audio_path = make_cache_path(self.cfg, suffix)
            await asyncio.to_thread(audio_path.write_bytes, audio_bytes)
            return AudioResult(
                provider="aliyun_qwen",
                delivery="onebot_record",
                voice_name=voice_name,
                audio_path=str(audio_path),
                mime_type=f"audio/{qwen_cfg.response_format}",
            )

        audio_path = await self._synthesize_http(model=model, voice=voice_name, text=text)
        return AudioResult(
            provider="aliyun_qwen",
            delivery="onebot_record",
            voice_name=voice_name,
            audio_path=str(audio_path),
            mime_type=f"audio/{qwen_cfg.response_format}",
        )

    async def _synthesize_http(self, *, model: str, voice: str, text: str) -> Path:
        qwen_cfg = self.cfg.aliyun.qwen
        url = f"{self.cfg.aliyun_http_base}/services/aigc/multimodal-generation/generation"
        payload = {
            "model": model,
            "input": {
                "text": text,
                "voice": voice,
                "language_type": qwen_cfg.language_type,
            },
            "parameters": {
                "sample_rate": qwen_cfg.sample_rate,
                "response_format": qwen_cfg.response_format,
            },
        }
        result = await self._post_json(url=url, payload=payload)
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        audio = output.get("audio") if isinstance(output, dict) else {}
        if isinstance(audio, dict):
            if isinstance(audio.get("data"), str) and audio["data"]:
                audio_bytes = base64.b64decode(audio["data"])
                suffix = ".wav" if qwen_cfg.response_format == "wav" else f".{qwen_cfg.response_format}"
                target = make_cache_path(self.cfg, suffix)
                await asyncio.to_thread(target.write_bytes, audio_bytes)
                return target
            if isinstance(audio.get("url"), str) and audio["url"]:
                target = make_cache_path(self.cfg, suffix_from_url(audio["url"], ".wav"))
                await download_file(audio["url"], target, timeout_sec=self.cfg.aliyun.timeout_sec)
                return target
        raise VoiceServiceError("阿里云 Qwen 语音合成未返回音频数据")

    async def _synthesize_realtime(self, *, model: str, voice: str, text: str) -> bytes:
        qwen_cfg = self.cfg.aliyun.qwen
        url = f"{self.cfg.aliyun_ws_base}?model={model}"
        collected: list[bytes] = []
        done = False

        async with websockets.connect(
            url,
            additional_headers=self._ws_headers(),
            open_timeout=self.cfg.aliyun.timeout_sec,
            max_size=None,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "type": "session.update",
                        "session": {
                            "voice": voice,
                            "mode": "commit",
                            "language_type": qwen_cfg.language_type,
                            "response_format": qwen_cfg.response_format,
                            "sample_rate": qwen_cfg.sample_rate,
                        },
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "type": "input_text_buffer.append",
                        "text": text,
                    }
                )
            )
            await websocket.send(
                json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": "input_text_buffer.commit"})
            )

            while True:
                raw = await websocket.recv()
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "response.audio.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        collected.append(base64.b64decode(delta))
                elif event_type == "error":
                    error = event.get("error") or {}
                    raise VoiceServiceError(str(error.get("message") or "阿里云 Qwen 实时语音合成失败"))
                elif event_type == "response.done":
                    status = ((event.get("response") or {}).get("status")) or ""
                    if status != "completed":
                        raise VoiceServiceError(f"阿里云 Qwen 实时语音合成未完成: {status}")
                    done = True
                    await websocket.send(
                        json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"})
                    )
                elif event_type == "session.finished":
                    break

        if not collected or not done:
            raise VoiceServiceError("阿里云 Qwen 实时语音合成未返回音频")
        return b"".join(collected)

    async def recognize(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
    ) -> TranscriptResult:
        model = self.cfg.aliyun.asr_model
        if "realtime" not in model:
            raise VoiceServiceError("当前仅实现阿里云 Qwen 实时语音识别模型")
        return await self._recognize_realtime(model=model, reply_voice=reply_voice)

    async def _recognize_realtime(self, *, model: str, reply_voice: ReplyVoiceMessage) -> TranscriptResult:
        wav_path = reply_voice.normalized_path()
        if wav_path is None:
            raise VoiceServiceError("语音样本未准备完成")

        pcm_path = make_cache_path(self.cfg, ".pcm")
        await transcode_wav_to_pcm_s16le(
            wav_path,
            pcm_path,
            ffmpeg_path=self.cfg.audio.ffmpeg_path,
            sample_rate=self.cfg.audio.target_sample_rate,
            channels=self.cfg.audio.target_channels,
        )
        pcm_bytes = await asyncio.to_thread(pcm_path.read_bytes)

        final_text = ""
        final_language = None
        final_emotion = None
        preview_text = ""

        url = f"{self.cfg.aliyun_ws_base}?model={model}"
        async with websockets.connect(
            url,
            additional_headers=self._ws_headers(),
            open_timeout=self.cfg.aliyun.timeout_sec,
            max_size=None,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "type": "session.update",
                        "session": {
                            "input_audio_format": "pcm",
                            "sample_rate": self.cfg.audio.target_sample_rate,
                            "turn_detection": None,
                        },
                    }
                )
            )

            chunk_size = 32768
            for start in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[start : start + chunk_size]
                await websocket.send(
                    json.dumps(
                        {
                            "event_id": f"event_{uuid.uuid4().hex}",
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )

            await websocket.send(
                json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": "input_audio_buffer.commit"})
            )

            while True:
                raw = await websocket.recv()
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "conversation.item.input_audio_transcription.text":
                    preview_text = f"{event.get('text') or ''}{event.get('stash') or ''}".strip()
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    final_text = str(event.get("transcript") or "").strip()
                    final_language = event.get("language")
                    final_emotion = event.get("emotion")
                    await websocket.send(
                        json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": "session.finish"})
                    )
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    error = event.get("error") or {}
                    raise VoiceServiceError(str(error.get("message") or "阿里云 Qwen 语音识别失败"))
                elif event_type == "error":
                    error = event.get("error") or {}
                    raise VoiceServiceError(str(error.get("message") or "阿里云 Qwen 语音识别失败"))
                elif event_type == "session.finished":
                    break

        text = final_text or preview_text
        if not text:
            raise VoiceServiceError("阿里云 Qwen 语音识别未返回文本")
        return TranscriptResult(provider="aliyun_qwen", text=text, language=final_language, emotion=final_emotion)

    async def clone_voice(
        self,
        session: SessionContext,
        state: SessionVoiceState,
        reply_voice: ReplyVoiceMessage,
        preferred_name: str,
        *,
        target_model: str | None = None,
    ) -> CloneResult:
        wav_path = reply_voice.normalized_path()
        if wav_path is None:
            raise VoiceServiceError("语音样本未准备完成")

        resolved_target_model = (target_model or self.cfg.aliyun.qwen.tts_custom_model).strip()
        if not resolved_target_model:
            raise VoiceServiceError("阿里云 Qwen 克隆目标模型不能为空")

        url = f"{self.cfg.aliyun_http_base}/services/audio/tts/customization"
        payload = {
            "model": self.cfg.aliyun.qwen.voice_clone_model,
            "input": {
                "action": "create",
                "target_model": resolved_target_model,
                "preferred_name": self._sanitize_preferred_name(preferred_name),
                "audio": {"data": build_data_url(wav_path)},
                "language": "zh",
            },
        }
        result = await self._post_json(url=url, payload=payload)
        raw_output = result.get("output")
        output = raw_output if isinstance(raw_output, dict) else {}
        voice = str(output.get("voice") or "").strip()
        if not voice:
            raise VoiceServiceError("阿里云 Qwen 声音复刻未返回音色名称")
        return CloneResult(
            provider="aliyun_qwen",
            voice_name=preferred_name,
            voice_id=voice,
            target_model=resolved_target_model,
        )
