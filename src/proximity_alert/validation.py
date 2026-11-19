"""Parameter validation shared by the CLI and the setup window (standard library only)."""

from __future__ import annotations

import math
from typing import List

from proximity_alert.config import Config


def config_errors(config: Config) -> List[str]:
    """Returns human-readable problems with the numeric parameters (empty if valid).

    Args:
        config: Configuration to check.

    Returns:
        One message per invalid parameter.
    """
    errors: List[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    def finite(value: float) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    check(config.imgsz > 0, "imgsz must be a positive integer")
    check(finite(config.conf) and 0.0 <= config.conf <= 1.0, "conf must be between 0 and 1")
    check(finite(config.time_window) and config.time_window > 0, "time window must be greater than 0")
    check(finite(config.min_ratio) and 0.0 <= config.min_ratio <= 1.0, "min ratio must be between 0 and 100 %")
    check(finite(config.depth_threshold) and config.depth_threshold >= 0, "depth threshold must be >= 0")
    check(finite(config.height_factor) and config.height_factor > 0, "height factor must be greater than 0")
    check(finite(config.rider_overlap) and 0.0 <= config.rider_overlap <= 1.0, "rider overlap must be between 0 and 1")
    check(finite(config.rider_memory) and config.rider_memory >= 0, "rider memory must be >= 0")
    check(config.depth_patch >= 1, "depth patch must be at least 1 pixel")
    check(finite(config.pip_scale) and 0.0 < config.pip_scale < 1.0, "pip scale must be between 0 and 1 (exclusive)")
    return errors
