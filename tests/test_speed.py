"""The watchdog: the point is catching a trickle, not just catching silence."""

from __future__ import annotations

import time

from sfd.core.speed import SmoothedSpeed, SpeedTracker


def test_no_verdict_before_the_window_has_data():
    tracker = SpeedTracker(window=10.0, min_span=5.0)
    tracker.add(1)
    # A connection must never be judged on its first moments — TLS setup and time-to-first-
    # byte would look identical to a stall.
    assert tracker.rate() is None
    assert not tracker.is_stalled(floor=1024)


def test_a_slow_trickle_is_reported_as_stalled():
    """The failure mode no socket timeout will ever catch."""
    tracker = SpeedTracker(window=10.0, min_span=0.0)
    tracker._started = time.monotonic() - 12  # pretend the connection has been alive a while
    for _ in range(10):
        tracker.add(300)  # ~3 KB total across a 10s window

    rate = tracker.rate()
    assert rate is not None and rate < 1024
    assert tracker.is_stalled(floor=64 * 1024)
    assert not tracker.is_stalled(floor=100)


def test_healthy_throughput_is_not_flagged():
    tracker = SpeedTracker(window=10.0, min_span=0.0)
    tracker._started = time.monotonic() - 12
    for _ in range(100):
        tracker.add(1024 * 1024)

    assert not tracker.is_stalled(floor=64 * 1024)


def test_old_samples_leave_the_window():
    tracker = SpeedTracker(window=1.0, min_span=0.0)
    tracker._started = time.monotonic() - 5
    tracker.add(10_000_000)
    time.sleep(1.1)
    tracker.add(1)

    # The burst has aged out; only the recent trickle counts.
    assert tracker.is_stalled(floor=64 * 1024)
    assert tracker.total == 10_000_001  # cumulative total is unaffected by pruning


def test_smoothed_speed_reacts_to_a_collapse():
    smoothed = SmoothedSpeed(alpha=0.5)
    smoothed.update(0)
    time.sleep(0.05)
    smoothed.update(5_000_000)
    fast = smoothed.value
    assert fast > 0

    for _ in range(8):
        time.sleep(0.02)
        smoothed.update(5_000_000)  # no new bytes at all
    assert smoothed.value < fast / 4
