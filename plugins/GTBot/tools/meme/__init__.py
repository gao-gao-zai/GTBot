from __future__ import annotations

import importlib


def register(registry) -> None:  # noqa: ANN001
    cfg_mod = importlib.import_module(__name__ + ".config")
    getattr(cfg_mod, "get_meme_plugin_config")()
    mod = importlib.import_module(__name__ + ".tool")
    registry.add_tool(getattr(mod, "save_meme"))
    registry.add_pre_agent_message_appender(getattr(mod, "append_meme_context_message"))
