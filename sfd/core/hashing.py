"""Incremental SHA256 over the contiguous completed prefix.

With several connections writing out of order you cannot hash bytes as they arrive — SHA256
is sequential and the pieces show up shuffled. The usual workaround is a full re-read of the
finished file, which costs minutes on a 20 GB download.

Instead a background task chases the contiguous prefix: whenever the run of completed bytes
from offset 0 grows, it reads just the new stretch back from disk and feeds the digest. Those
bytes were written moments ago and are almost always still in the page cache, so the read is
nearly free, and by the time the last chunk lands the digest is essentially done.

This also makes resume uniform: a resumed download simply starts with a large prefix already
available and the chaser catches up on it while new bytes are still arriving.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
from pathlib import Path
from typing import Callable

READ_BLOCK = 4 * 1024 * 1024
IDLE_SLEEP = 0.25


class PrefixHasher:
    """Chases `prefix_fn()` and digests everything below it, exactly once, in order."""

    def __init__(self, path: Path, prefix_fn: Callable[[], int]) -> None:
        self._path = path
        self._prefix_fn = prefix_fn
        self._digest = hashlib.sha256()
        self._position = 0
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Held while a read is in flight. Cancelling the task does not stop the worker
        # thread it dispatched, so without this the file can still be open after `aclose()`
        # returns — and on Windows an open handle blocks the file from being moved or
        # deleted, which surfaces much later as a mysterious sharing violation.
        self._reading = threading.Lock()

    @property
    def position(self) -> int:
        """How many bytes have been folded into the digest."""
        return self._position

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="prefix-hasher")

    async def finish(self, expected_total: int) -> str:
        """Drain the remaining prefix and return the hex digest.

        Call only once the transfer is complete — `expected_total` is the full file size and
        this will keep reading until the digest covers it.
        """
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        await self._advance(expected_total)
        if self._position != expected_total:
            raise RuntimeError(
                f"hasher covered {self._position} of {expected_total} bytes"
            )
        return self._digest.hexdigest()

    async def aclose(self) -> None:
        """Stop chasing. Idempotent, and safe to call after `finish()` or on a failure path.

        Without this the chaser survives an aborted transfer and keeps reading a file that
        is about to be renamed or removed.
        """
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Wait out any read the cancelled task had already handed to a thread, so the file
        # really is closed by the time this returns.
        await asyncio.to_thread(self._reading.acquire)
        self._reading.release()

    async def _run(self) -> None:
        while not self._stop.is_set():
            advanced = await self._advance(self._prefix_fn())
            if not advanced:
                # Nothing new to chew on; yield rather than spin.
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=IDLE_SLEEP)
                except TimeoutError:
                    pass

    async def _advance(self, target: int) -> int:
        """Digest bytes in [self._position, target). Returns how many were consumed."""
        if target <= self._position:
            return 0
        consumed = await asyncio.to_thread(self._read_and_digest, target)
        self._position += consumed
        return consumed

    def _read_and_digest(self, target: int) -> int:
        """Runs in a worker thread — must not touch anything the event loop mutates."""
        consumed = 0
        start = self._position
        remaining = target - start
        with self._reading, open(self._path, "rb") as fh:
            fh.seek(start)
            while remaining > 0:
                block = fh.read(min(READ_BLOCK, remaining))
                if not block:
                    # File is shorter than the caller claimed. Stop where the data does;
                    # the digest stays consistent with what we actually read.
                    break
                self._digest.update(block)
                consumed += len(block)
                remaining -= len(block)
        return consumed
