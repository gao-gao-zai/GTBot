from __future__ import annotations

from .services.trigger import scan as _scan
from .services.trigger.scan import (
    GROUP_AUTO_TRIGGER_SCAN_INTERVAL_SECONDS,
    run_group_auto_trigger_scan,
)

random = _scan.random

__all__ = [
    "GROUP_AUTO_TRIGGER_SCAN_INTERVAL_SECONDS",
    "run_group_auto_trigger_scan",
    "random",
]
