from __future__ import annotations

import importlib

def register(registry) -> None:  # noqa: ANN001
    mod = importlib.import_module(__name__ + ".tool")
    registry.add_tool(getattr(mod, "vlm_describe_image"))
    registry.add_agent_middleware(getattr(mod, "VLMImageCQTitleMiddleware")())
