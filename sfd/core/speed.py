"""Throughput: measuring it, watching for stalls, and holding it down on purpose.

A read timeout only fires when *nothing* arrives. The failure mode that actually costs
people hours is different: bytes keep trickling in at a few KB/s, so every socket looks
healthy and the download never finishes. That needs a throughput floor, not a timeout.

The ceiling at the bottom of this file is the opposite problem: a 40 GB model fetched over
sixteen connections will take the whole line, and the machine is usually being used for
something else at the same time.
"""

from __future__ import annotations

import asyncio
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


class RateLimiter:
    """A ceiling on how fast bytes are taken off the network, shared by everything running.

    One bucket for every connection of every download, because the thing being protected is
    the link: sixteen connections each politely capped at "1 MB/s" is not the limit anyone
    meant to set. `rate` may be changed while downloads are in flight — the setting is meant
    to be reachable mid-download, which is when you actually notice you want it.
    """

    __slots__ = ("rate", "_allowance", "_last", "_lock")

    def __init__(self, rate: float = 0.0) -> None:
        self.rate = rate  # bytes per second; 0 means no ceiling at all
        self._allowance = 0.0
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, nbytes: int) -> None:
        """Block until `nbytes` fit under the ceiling, then count them against it."""
        if self.rate <= 0 or nbytes <= 0:
            return
        async with self._lock:
            while True:
                rate = self.rate
                if rate <= 0:  # lifted while we were waiting
                    return
                now = time.monotonic()
                # The bucket holds a second's worth, or one read — whichever is larger, or a
                # read bigger than the budget could never be paid for and would hang here.
                self._allowance = min(
                    max(rate, nbytes), self._allowance + (now - self._last) * rate
                )
                self._last = now
                if self._allowance >= nbytes:
                    self._allowance -= nbytes
                    return
                await asyncio.sleep((nbytes - self._allowance) / rate)


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
