from __future__ import annotations

import importlib


def register(registry) -> None:  # noqa: ANN001
    cfg_mod = importlib.import_module(__name__ + ".config")
    getattr(cfg_mod, "get_meme_plugin_config")()
    mod = importlib.import_module(__name__ + ".tool")
    registry.add_tool(getattr(mod, "save_meme"))
    registry.add_agent_middleware(getattr(mod, "MemeContextMiddleware")())
