from __future__ import annotations

from .config import get_voice_service_plugin_config


def register(registry) -> None:  # noqa: ANN001
    get_voice_service_plugin_config()
    from .tool import prewarm_voice_service_context, voice_recognize_tool, voice_synthesize_tool

    enabled = lambda ctx: getattr(ctx, "trigger_mode", None) != "group_auto"
    registry.add_pre_agent_processor(
        prewarm_voice_service_context,
        enabled=enabled,
        wait_until_complete=False,
    )
    registry.add_tool(voice_recognize_tool, enabled=enabled)
    registry.add_tool(voice_synthesize_tool, enabled=enabled)
    return None


try:
    from nonebot import get_driver  # type: ignore

    get_driver()
except Exception:  # noqa: BLE001
    get_driver = None


if get_driver is not None:
    from . import commands  # noqa: F401
