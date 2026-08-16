"""The speed ceiling.

Rates are asserted with generous slack: what matters is that a limit holds the throughput
down to roughly what was asked for, not that a token bucket is accurate to the millisecond
on a machine that is also running the rest of the suite.
"""

from __future__ import annotations

import asyncio
import time

from sfd.core.speed import RateLimiter


async def _drain(limiter: RateLimiter, reads: int, size: int) -> float:
    start = time.monotonic()
    for _ in range(reads):
        await limiter.take(size)
    return time.monotonic() - start


async def test_no_ceiling_costs_nothing():
    assert await _drain(RateLimiter(0), 200, 64 * 1024) < 0.1


async def test_a_ceiling_holds_the_rate_down():
    limiter = RateLimiter(1_000_000)
    moved = 20 * 25_000  # half a second's worth at the ceiling
    elapsed = await _drain(limiter, 20, 25_000)

    assert moved / elapsed <= 1_000_000 * 1.5


async def test_the_ceiling_is_shared_rather_than_per_connection():
    """Sixteen connections each politely capped at the limit is not the limit anyone set."""
    limiter = RateLimiter(1_000_000)
    start = time.monotonic()
    await asyncio.gather(*(_drain(limiter, 10, 25_000) for _ in range(4)))
    elapsed = time.monotonic() - start

    assert 1_000_000 / elapsed <= 1_000_000 * 1.5


async def test_a_read_larger_than_a_seconds_worth_is_paid_for_in_proportion():
    """The bucket holds a second's worth. A bigger read has to be able to grow it, or the
    balance never reaches the asking price and the download hangs on the first block."""
    limiter = RateLimiter(1_000_000)
    start = time.monotonic()
    await asyncio.wait_for(limiter.take(2_000_000), timeout=10.0)

    assert 1.0 <= time.monotonic() - start <= 5.0


async def test_lifting_the_ceiling_mid_wait_releases_the_waiter():
    """The setting is reachable while downloads are running, which is when it gets used."""
    limiter = RateLimiter(1000)
    await limiter.take(1000)  # empty the bucket

    waiter = asyncio.create_task(limiter.take(1000))
    await asyncio.sleep(0.05)
    limiter.rate = 0
    await asyncio.wait_for(waiter, timeout=1.0)


async def test_the_stall_floor_drops_under_a_ceiling():
    """Otherwise the watchdog shoots the connections we are deliberately holding back."""
    from sfd.core.transfer import Transfer, TransferOptions
    from sfd.core.types import FileIdentity

    identity = FileIdentity(provider="direct", ref={"url": "https://example.com/m"})
    options = TransferOptions(connections=8, min_speed=64 * 1024)

    unlimited = Transfer(None, identity, "m.bin", options)
    assert unlimited._stall_floor() == 64 * 1024

    limited = Transfer(None, identity, "m.bin", options, limiter=RateLimiter(64 * 1024))
    assert limited._stall_floor() == 64 * 1024 / 8 / 2
