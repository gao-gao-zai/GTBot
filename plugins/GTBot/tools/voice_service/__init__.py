from __future__ import annotations

from .config import get_voice_service_plugin_config


def register(registry) -> None:  # noqa: ANN001
    get_voice_service_plugin_config()
    from .tool import voice_recognize_tool, voice_synthesize_tool

    registry.add_tool(voice_recognize_tool)
    registry.add_tool(voice_synthesize_tool)
    return None


try:
    from nonebot import get_driver  # type: ignore
except Exception:  # noqa: BLE001
    get_driver = None


if get_driver is not None:
    from . import commands  # noqa: F401
