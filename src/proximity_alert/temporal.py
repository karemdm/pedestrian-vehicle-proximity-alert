"""Time-based state per track id (standard library only).

Timestamps are video time in seconds and are always passed in by the caller,
so the classes are deterministic and easy to test.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Hashable, Iterable, Optional, Tuple


_EPS = 1e-9
"""Tolerance for float timestamps (``frame_idx / fps``) on window boundaries."""


class ProximityHistory:
    """Sliding window of "close / not close" observations per track id.

    The window is inclusive: a sample at ``t`` is kept while ``now - t <= window_sec``,
    so at a constant frame rate it holds ``round(window_sec * fps) + 1`` samples.

    Args:
        window_sec: Window length; samples older than ``now - window_sec`` are dropped.
    """

    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self._samples: Dict[Hashable, Deque[Tuple[float, bool]]] = {}

    def update(self, track_id: Hashable, timestamp: float, is_close: bool) -> None:
        """Adds an observation and drops samples outside the window.

        Args:
            track_id: Track identifier.
            timestamp: Current video time in seconds.
            is_close: Whether the person is close to a vehicle in this frame.
        """
        samples = self._samples.setdefault(track_id, deque())
        samples.append((timestamp, is_close))
        while samples and timestamp - samples[0][0] > self.window_sec + _EPS:
            samples.popleft()

    def ratio(self, track_id: Hashable) -> float:
        """Fraction of close observations in the window (``0.0`` if unknown)."""
        samples = self._samples.get(track_id)
        if not samples:
            return 0.0
        return sum(1 for _, close in samples if close) / len(samples)

    def count(self, track_id: Hashable) -> int:
        """Number of samples currently in the window for ``track_id``."""
        return len(self._samples.get(track_id, ()))

    def prune(self, active_ids: Iterable[Hashable]) -> None:
        """Forgets every track id not in ``active_ids`` (lost or rider tracks)."""
        keep = set(active_ids)
        for track_id in [t for t in self._samples if t not in keep]:
            del self._samples[track_id]

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._samples

    def __len__(self) -> int:
        return len(self._samples)


def is_alert(ratio: float, currently_close: bool, min_ratio: float) -> bool:
    """Alert rule: enough close frames in the window, or close right now."""
    return ratio >= min_ratio or currently_close


class RiderMemory:
    """Keeps the rider status of a track for a while after the last overlap.

    Args:
        memory_sec: How long the status persists after the last rider detection.
    """

    def __init__(self, memory_sec: float) -> None:
        self.memory_sec = memory_sec
        self._last_seen: Dict[Hashable, float] = {}

    def mark(self, track_id: Hashable, timestamp: float) -> None:
        """Records that ``track_id`` was detected as a rider at ``timestamp``."""
        self._last_seen[track_id] = timestamp

    def last_seen(self, track_id: Hashable) -> Optional[float]:
        """Time of the last rider detection, or ``None`` if never seen."""
        return self._last_seen.get(track_id)

    def is_rider(self, track_id: Hashable, timestamp: float) -> bool:
        """Whether ``track_id`` was a rider within the last ``memory_sec`` seconds."""
        last = self._last_seen.get(track_id)
        return last is not None and timestamp - last <= self.memory_sec + _EPS

    def cleanup(self, timestamp: float) -> None:
        """Drops entries older than twice the memory duration."""
        limit = 2.0 * self.memory_sec
        for track_id in [t for t, last in self._last_seen.items() if timestamp - last > limit]:
            del self._last_seen[track_id]

    def __len__(self) -> int:
        return len(self._last_seen)
