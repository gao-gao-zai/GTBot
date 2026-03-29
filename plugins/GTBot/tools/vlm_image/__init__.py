from __future__ import annotations

import importlib

def register(registry) -> None:  # noqa: ANN001
    mod = importlib.import_module(__name__ + ".tool")
    registry.add_tool(getattr(mod, "vlm_describe_image"))
    registry.add_pre_agent_processor(
        getattr(mod, "prewarm_vlm_image_cq_titles"),
        wait_until_complete=False,
    )
    registry.add_pre_agent_message_injector(getattr(mod, "inject_vlm_image_cq_titles"))
