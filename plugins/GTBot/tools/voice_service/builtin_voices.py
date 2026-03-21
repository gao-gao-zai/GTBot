from __future__ import annotations

from .models import VoiceItem


ALIYUN_QWEN_BUILTIN_VOICES: list[VoiceItem] = [
    VoiceItem(
        provider="aliyun_qwen",
        name="Cherry",
        display_name="Cherry",
        voice_type="builtin",
        description="Qwen 系统音色。",
    ),
    VoiceItem(
        provider="aliyun_qwen",
        name="Serena",
        display_name="Serena",
        voice_type="builtin",
        description="Qwen 系统音色。",
    ),
    VoiceItem(
        provider="aliyun_qwen",
        name="Ethan",
        display_name="Ethan",
        voice_type="builtin",
        description="Qwen 系统音色。",
    ),
    VoiceItem(
        provider="aliyun_qwen",
        name="Chelsie",
        display_name="Chelsie",
        voice_type="builtin",
        description="Qwen 系统音色。",
    ),
]


ALIYUN_COSYVOICE_BUILTIN_VOICES: list[VoiceItem] = [
    VoiceItem(
        provider="aliyun_cosyvoice",
        name="longanyang",
        display_name="longanyang",
        voice_type="builtin",
        target_model="cosyvoice-v3-flash",
        description="CosyVoice v3-flash 示例系统音色。",
    ),
    VoiceItem(
        provider="aliyun_cosyvoice",
        name="longxiaochun_v2",
        display_name="longxiaochun_v2",
        voice_type="builtin",
        target_model="cosyvoice-v2",
        description="CosyVoice v2 示例系统音色。",
    ),
]
