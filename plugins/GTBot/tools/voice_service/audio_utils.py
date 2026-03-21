from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import VoiceServicePluginConfig
from .models import ReplyVoiceMessage

try:
    from nonebot import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


class VoiceServiceError(RuntimeError):
    pass


def make_cache_path(cfg: VoiceServicePluginConfig, suffix: str) -> Path:
    cfg.cache_dir_path.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir_path / f"{uuid.uuid4().hex}{suffix}"


def suffix_from_url(url: str, default: str = ".bin") -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    return suffix or default


def normalize_remote_url(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith(("http://", "https://")):
        return text
    return None


async def cleanup_expired_cache(cfg: VoiceServicePluginConfig) -> None:
    ttl = max(60, int(cfg.storage.temp_ttl_sec))
    now = __import__("time").time()
    cache_dir = cfg.cache_dir_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    for child in cache_dir.iterdir():
        try:
            if not child.is_file():
                continue
            age = now - child.stat().st_mtime
            if age > ttl:
                child.unlink(missing_ok=True)
        except Exception:
            continue


async def download_file(url: str, dest: Path, *, timeout_sec: float) -> Path:
    normalized_url = normalize_remote_url(url)
    if not normalized_url:
        raise VoiceServiceError(f"无效的下载地址: {url}")

    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
        response = await client.get(normalized_url)
        response.raise_for_status()
        await asyncio.to_thread(dest.write_bytes, response.content)
    return dest


async def run_ffmpeg(ffmpeg_path: str, args: list[str]) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise VoiceServiceError(
            f"未找到 ffmpeg，可执行文件配置为: {ffmpeg_path}。请安装 ffmpeg 或在 config.json 中填写正确路径"
        ) from exc
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="ignore")
        raise VoiceServiceError(f"ffmpeg failed: {detail.strip()}")


async def normalize_to_wav(
    src: Path,
    dest: Path,
    *,
    ffmpeg_path: str,
    sample_rate: int,
    channels: int,
) -> Path:
    await run_ffmpeg(
        ffmpeg_path,
        [
            "-y",
            "-i",
            str(src),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(dest),
        ],
    )
    return dest


async def transcode_wav_to_pcm_s16le(
    src: Path,
    dest: Path,
    *,
    ffmpeg_path: str,
    sample_rate: int,
    channels: int,
) -> Path:
    await run_ffmpeg(
        ffmpeg_path,
        [
            "-y",
            "-i",
            str(src),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            str(dest),
        ],
    )
    return dest


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "audio/wav"


def build_data_url(path: Path) -> str:
    mime = guess_mime_type(path)
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def pcm_to_wav_bytes(pcm_bytes: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _extract_reply_message_id_from_event(event: Any) -> int | None:
    reply = getattr(event, "reply", None)
    if reply is not None:
        message_id = getattr(reply, "message_id", None)
        if isinstance(message_id, int):
            return message_id

    message = getattr(event, "message", None)
    if message is None:
        return None

    try:
        for segment in message:
            if getattr(segment, "type", "") == "reply":
                raw_id = getattr(segment, "data", {}).get("id")
                if raw_id is None:
                    continue
                return int(raw_id)
    except Exception:
        return None
    return None


def _message_segments(raw_message: dict[str, Any]) -> list[dict[str, Any]]:
    message = raw_message.get("message")
    if isinstance(message, list):
        segments: list[dict[str, Any]] = []
        for segment in message:
            if isinstance(segment, dict):
                segments.append(segment)
            elif hasattr(segment, "type") and hasattr(segment, "data"):
                segments.append({"type": segment.type, "data": dict(segment.data)})
        return segments

    if message is not None and hasattr(message, "__iter__") and not isinstance(message, (str, bytes, dict)):
        segments = []
        for segment in message:
            if hasattr(segment, "type") and hasattr(segment, "data"):
                segments.append({"type": segment.type, "data": dict(segment.data)})
        if segments:
            return segments

    raw = raw_message.get("raw_message")
    if isinstance(raw, str) and "[CQ:record" in raw:
        start = raw.find("[CQ:record")
        end = raw.find("]", start)
        if start >= 0 and end > start:
            body = raw[start + 4 : end]
            parts = body.split(",")
            data: dict[str, str] = {}
            for chunk in parts[1:]:
                if "=" not in chunk:
                    continue
                key, value = chunk.split("=", 1)
                data[key] = value
            return [{"type": "record", "data": data}]
    return []


def _extract_record_segment(raw_message: dict[str, Any]) -> dict[str, Any] | None:
    for segment in _message_segments(raw_message):
        if segment.get("type") == "record":
            return segment
    return None


def _extract_file_segment(raw_message: dict[str, Any]) -> dict[str, Any] | None:
    for segment in _message_segments(raw_message):
        if segment.get("type") != "file":
            continue
        return segment
    return None


def _looks_like_audio_filename(value: str | None) -> bool:
    if not value:
        return False
    suffix = Path(value).suffix.lower()
    return suffix in {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".amr",
        ".wma",
        ".pcm",
        ".silk",
    }


def _pick_first_string(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _try_fetch_remote_file_via_api(
    bot: Any,
    data: dict[str, Any],
) -> tuple[str | None, str | None]:
    file_id = _pick_first_string(data, ["file_id", "fid", "id"])
    busid = data.get("busid")
    if not file_id:
        return None, None

    candidates = [
        ("get_file", {"file_id": file_id}),
        ("get_file", {"fid": file_id}),
        ("get_file_url", {"file_id": file_id}),
        ("get_file_url", {"fid": file_id}),
    ]
    if busid is not None:
        candidates.extend(
            [
                ("get_file", {"file_id": file_id, "busid": busid}),
                ("get_file", {"fid": file_id, "busid": busid}),
                ("get_file_url", {"file_id": file_id, "busid": busid}),
                ("get_file_url", {"fid": file_id, "busid": busid}),
            ]
        )

    for action, params in candidates:
        try:
            resp = await bot.call_api(action, **params)
        except Exception:
            continue

        if isinstance(resp, str) and resp.startswith(("http://", "https://")):
            return resp, file_id
        if not isinstance(resp, dict):
            continue

        url = _pick_first_string(resp, ["url", "download_url", "file_url"])
        path = _pick_first_string(resp, ["file", "path", "temp_file"])
        if url:
            return url, file_id
        if path:
            return path, file_id

    return None, file_id


async def _resolve_source_file(
    bot: Any,
    media_segment: dict[str, Any],
    cfg: VoiceServicePluginConfig,
) -> tuple[Path, str | None, str | None]:
    data = media_segment.get("data") or {}
    record_file = _pick_first_string(data, ["file", "path", "temp_file", "name"])
    record_url = normalize_remote_url(_pick_first_string(data, ["url", "file_url", "download_url", "src"]))

    if not record_url:
        fetched_url_or_path, fetched_file_id = await _try_fetch_remote_file_via_api(bot, data)
        if fetched_url_or_path:
            normalized = normalize_remote_url(fetched_url_or_path)
            if normalized:
                record_url = normalized
            else:
                record_file = fetched_url_or_path
        elif fetched_file_id and not record_file:
            record_file = fetched_file_id

    if record_url:
        source_path = make_cache_path(cfg, suffix_from_url(record_url, ".audio"))
        await download_file(record_url, source_path, timeout_sec=cfg.aliyun.timeout_sec)
        return source_path, record_file, record_url

    if record_file and record_file.startswith(("http://", "https://")):
        source_path = make_cache_path(cfg, suffix_from_url(record_file, ".audio"))
        await download_file(record_file, source_path, timeout_sec=cfg.aliyun.timeout_sec)
        return source_path, record_file, record_file

    if record_file:
        if media_segment.get("type") == "record":
            try:
                resp = await bot.call_api("get_record", file=record_file, out_format="wav")
                file_path = resp.get("file") if isinstance(resp, dict) else None
                if isinstance(file_path, str) and Path(file_path).exists():
                    return Path(file_path), record_file, None
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"voice_service get_record failed: {exc!s}")

        candidate = Path(record_file)
        if candidate.exists():
            return candidate, record_file, None

    available_keys = ", ".join(sorted(str(k) for k in data.keys()))
    raise VoiceServiceError(
        f"无法解析被回复语音消息的文件地址。可见字段: {available_keys or '(空)'}"
    )


def _read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


async def resolve_reply_voice(
    bot: Any,
    event: Any,
    cfg: VoiceServicePluginConfig,
) -> ReplyVoiceMessage:
    reply_message_id = _extract_reply_message_id_from_event(event)
    if reply_message_id is None:
        raise VoiceServiceError("请先回复一条语音消息")

    return await resolve_message_voice(bot, int(reply_message_id), cfg)


async def resolve_message_voice(
    bot: Any,
    message_id: int,
    cfg: VoiceServicePluginConfig,
) -> ReplyVoiceMessage:
    reply_message_id = int(message_id)
    raw_message = await bot.get_msg(message_id=reply_message_id)
    if not isinstance(raw_message, dict):
        raise VoiceServiceError("无法获取被回复的消息内容")

    media_segment = _extract_record_segment(raw_message)
    if media_segment is None:
        file_segment = _extract_file_segment(raw_message)
        if file_segment is not None:
            data = file_segment.get("data") or {}
            name = str(data.get("name") or data.get("file") or "")
            url = str(data.get("url") or data.get("file_url") or "")
            if _looks_like_audio_filename(name) or _looks_like_audio_filename(url):
                media_segment = file_segment

    if media_segment is None:
        raise VoiceServiceError("被回复消息中没有语音段或音频文件")

    source_path, record_file, record_url = await _resolve_source_file(bot, media_segment, cfg)

    if source_path.stat().st_size > cfg.audio.max_sample_size_mb * 1024 * 1024:
        raise VoiceServiceError("语音样本超过大小限制")

    normalized_path = make_cache_path(cfg, ".wav")
    await normalize_to_wav(
        source_path,
        normalized_path,
        ffmpeg_path=cfg.audio.ffmpeg_path,
        sample_rate=cfg.audio.target_sample_rate,
        channels=cfg.audio.target_channels,
    )

    duration_sec = _read_wav_duration(normalized_path)
    if duration_sec > cfg.audio.max_sample_duration_sec:
        raise VoiceServiceError("语音样本超过时长限制")

    return ReplyVoiceMessage(
        reply_message_id=reply_message_id,
        raw_message=raw_message,
        record_file=record_file,
        record_url=record_url,
        source_path=str(source_path),
        normalized_wav_path=str(normalized_path),
        duration_sec=duration_sec,
        file_size_bytes=normalized_path.stat().st_size,
    )


def parse_json_lines_from_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        payload_parts = [line[5:].strip() for line in lines if line.startswith("data:")]
        if not payload_parts:
            continue
        payload = "\n".join(payload_parts)
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events
