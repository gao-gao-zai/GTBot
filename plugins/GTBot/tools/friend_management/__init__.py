from __future__ import annotations

from .config import get_friend_management_plugin_config
from .tool import delete_friend_tool


def register(registry) -> None:  # noqa: ANN001
    get_friend_management_plugin_config()
    registry.add_tool(delete_friend_tool)

