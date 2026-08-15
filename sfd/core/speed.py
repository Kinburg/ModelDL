"""Throughput measurement and the stall watchdog.

A read timeout only fires when *nothing* arrives. The failure mode that actually costs
people hours is different: bytes keep trickling in at a few KB/s, so every socket looks
healthy and the download never finishes. That needs a throughput floor, not a timeout.
"""

from __future__ import annotations

import time
from collections import deque


class SpeedTracker:
    """Rolling-window byte rate.

    `rate()` returns None until the window covers at least `min_span` seconds, so a
    connection is never judged on its first moments — TLS handshake and TTFB would
    otherwise look like a stall.
    """

    __slots__ = ("_window", "_samples", "_min_span", "_total", "_started")

    def __init__(self, window: float = 15.0, min_span: float = 8.0) -> None:
        self._window = window
        self._min_span = min_span
        self._samples: deque[tuple[float, int]] = deque()
        self._total = 0
        self._started = time.monotonic()

    def add(self, nbytes: int) -> None:
        now = time.monotonic()
        self._samples.append((now, nbytes))
        self._total += nbytes
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def rate(self) -> float | None:
        """Bytes per second over the window, or None if the sample is too young to judge."""
        now = time.monotonic()
        self._prune(now)
        span = now - self._started
        if span < self._min_span:
            return None
        # Measure against the window we actually retain, not against total elapsed time.
        effective = min(span, self._window)
        if effective <= 0:
            return None
        return sum(n for _, n in self._samples) / effective

    def is_stalled(self, floor: float) -> bool:
        r = self.rate()
        return r is not None and r < floor

    @property
    def total(self) -> int:
        return self._total


class SmoothedSpeed:
    """Exponentially smoothed aggregate speed, for display and ETA."""

    __slots__ = ("_alpha", "_value", "_last_bytes", "_last_time")

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._value: float = 0.0
        self._last_bytes: int | None = None
        self._last_time: float | None = None

    def update(self, total_bytes: int) -> float:
        now = time.monotonic()
        if self._last_bytes is None or self._last_time is None:
            self._last_bytes, self._last_time = total_bytes, now
            return self._value
        dt = now - self._last_time
        if dt <= 0:
            return self._value
        instant = max(0, total_bytes - self._last_bytes) / dt
        self._value = instant if self._value == 0 else self._alpha * instant + (1 - self._alpha) * self._value
        self._last_bytes, self._last_time = total_bytes, now
        return self._value

    @property
    def value(self) -> float:
        return self._value
