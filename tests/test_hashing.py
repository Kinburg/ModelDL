"""The catch-up hasher must produce exactly the digest of a plain sequential read."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from sfd.core.hashing import PrefixHasher

DATA = os.urandom(500_000)


async def test_matches_a_plain_digest_when_fed_gradually(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    prefix = 0
    hasher = PrefixHasher(path, lambda: prefix)
    hasher.start()

    # Reveal the file in uneven steps, the way out-of-order chunks complete.
    for step in (10_000, 137, 250_000, 1, 99_862, 140_000):
        prefix += step
        await asyncio.sleep(0.01)

    assert prefix == len(DATA)
    digest = await hasher.finish(len(DATA))
    assert digest == hashlib.sha256(DATA).hexdigest()


async def test_finishes_even_if_it_never_got_to_run(tmp_path: Path):
    """A tiny file can complete before the background task takes a single turn."""
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    hasher = PrefixHasher(path, lambda: 0)
    hasher.start()
    digest = await hasher.finish(len(DATA))
    assert digest == hashlib.sha256(DATA).hexdigest()


async def test_aclose_stops_the_chaser_without_raising(tmp_path: Path):
    """A failed transfer must not leave a task reading a file that is about to vanish."""
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    prefix = 0
    hasher = PrefixHasher(path, lambda: prefix)
    hasher.start()
    prefix = len(DATA)
    await asyncio.sleep(0.01)

    await hasher.aclose()
    await hasher.aclose()  # idempotent

    # With the chaser stopped, removing the file must not surface anything.
    path.unlink()
    await asyncio.sleep(0.3)


async def test_aclose_waits_for_the_read_it_already_dispatched(tmp_path: Path):
    """Cancelling the task does not stop the thread it handed the read to.

    On Windows an open handle blocks the file from being moved or deleted, so a `stop()`
    that returns while a read is still in flight turns into a sharing violation somewhere
    else entirely.
    """
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    hasher = PrefixHasher(path, lambda: len(DATA))
    hasher.start()
    await asyncio.sleep(0)          # let it dispatch a read
    await hasher.aclose()

    # If a handle were still open this would raise PermissionError on Windows.
    path.unlink()
    assert not path.exists()


async def test_aclose_after_finish_is_a_noop(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    hasher = PrefixHasher(path, lambda: len(DATA))
    hasher.start()
    digest = await hasher.finish(len(DATA))
    await hasher.aclose()
    assert digest == hashlib.sha256(DATA).hexdigest()


async def test_resume_style_start_with_a_large_prefix(tmp_path: Path):
    """Resumed downloads begin with most of the file already present."""
    path = tmp_path / "f.bin"
    path.write_bytes(DATA)

    prefix = 400_000
    hasher = PrefixHasher(path, lambda: prefix)
    hasher.start()
    await asyncio.sleep(0.05)
    assert hasher.position == 400_000  # caught up while "downloading" continued

    prefix = len(DATA)
    digest = await hasher.finish(len(DATA))
    assert digest == hashlib.sha256(DATA).hexdigest()
