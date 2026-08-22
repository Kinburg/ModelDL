"""The transfer engine.

Knows nothing about HuggingFace or Civitai. It is handed a `Provider` that can mint a fresh
URL on demand, and its whole job is to get the bytes onto disk without ever corrupting what
is already there.

The rules that make that true:

  * A signed URL is a disposable. It is re-minted before it expires, and again on any 403.
  * A resume request that comes back 200 instead of 206 is refused outright, without writing
    a byte. This is the single most common way downloaders corrupt or restart files.
  * Every connection is watched for throughput, not just for silence, so a download that
    trickles at 3 KB/s gets cut and restarted rather than running until morning.
  * Nothing is renamed into place until the digest matches.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import random
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .. import __version__
from ..providers.base import HTML_CONTENT_TYPES, Provider
from .chunks import ChunkMap, Chunk, pick_chunk_size
from .diskinfo import recommend_connections
from .errors import (
    AccessDenied,
    AuthRequired,
    ChecksumMismatch,
    NotBinaryContent,
    RangeIgnored,
    RangeNotHonored,
    RemoteChanged,
    Retryable,
    SignatureExpired,
    Stalled,
    TransferFailed,
    TransportError,
)
from .hashing import PrefixHasher
from .prealloc import allocate
from .speed import RateLimiter, SmoothedSpeed, SpeedTracker
from .state import PartState
from .types import DiskKind, FileIdentity, ProgressSnapshot, ResolvedTarget

ProgressCallback = Callable[[ProgressSnapshot], None | Awaitable[None]]

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)

# How long an idle worker waits before looking for stealable work again.
IDLE_POLL = 0.25


class TransferOptions:
    __slots__ = (
        "connections", "chunk_size", "min_speed", "stall_window", "stall_min_span",
        "connect_timeout", "read_timeout", "max_attempts", "verify_hash",
        "respect_disk_kind", "user_agent", "progress_interval",
        "backoff_base", "backoff_max", "disk_kind", "verify_existing",
    )

    def __init__(
        self,
        # Measured against HuggingFace's CDN, extra connections buy consistency rather than
        # peak speed: a single connection can draw a bad edge and crawl at a tenth of the
        # achievable rate, while sixteen average that lottery out. This is also what
        # hf-xet uses by default (HF_XET_NUM_CONCURRENT_RANGE_GETS).
        connections: int = 16,
        chunk_size: int | None = None,
        min_speed: float = 64 * 1024,
        stall_window: float = 15.0,
        stall_min_span: float = 8.0,
        connect_timeout: float = 15.0,
        read_timeout: float = 30.0,
        max_attempts: int = 20,
        verify_hash: bool = True,
        respect_disk_kind: bool = True,
        user_agent: str = f"model-dl/{__version__}",
        progress_interval: float = 0.5,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        disk_kind: DiskKind | None = None,
        verify_existing: bool = True,
    ) -> None:
        self.connections = connections
        self.chunk_size = chunk_size
        self.min_speed = min_speed
        self.stall_window = stall_window
        self.stall_min_span = stall_min_span
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_attempts = max_attempts
        self.verify_hash = verify_hash
        self.respect_disk_kind = respect_disk_kind
        self.user_agent = user_agent
        self.progress_interval = progress_interval
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        # Set from settings when the user knows better than Windows does.
        self.disk_kind = disk_kind
        # Hash a file that is already in place before deciding to skip it. Turning this off
        # makes an already-complete library re-scan instant, at the cost of trusting size.
        self.verify_existing = verify_existing


class Transfer:
    """One file, start to finish."""

    def __init__(
        self,
        provider: Provider,
        identity: FileIdentity,
        dest: Path | str,
        options: TransferOptions | None = None,
        on_progress: ProgressCallback | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._provider = provider
        self._identity = identity
        self._dest_arg = Path(dest)
        self._opts = options or TransferOptions()
        self._on_progress = on_progress
        # Shared with every other transfer running, so the ceiling is on the link rather
        # than on each file. Absent means no ceiling.
        self._limiter = limiter

        self._target: ResolvedTarget | None = None
        self._resolved_at = 0.0
        self._resolve_lock = asyncio.Lock()
        self._map: ChunkMap | None = None
        self._state: PartState | None = None
        self._active = 0
        self._speed = SmoothedSpeed()
        self._transferred = 0

    @property
    def bytes_transferred(self) -> int:
        """Bytes actually pulled over the network by this run.

        Not the file size and not the progress delta: a resumed transfer moves only the
        remainder, and a file already on disk moves nothing at all. Callers reporting a
        rate need this number, or they end up crediting themselves with work they skipped.
        """
        return self._transferred

    # --- public -----------------------------------------------------------

    async def run(self) -> Path:
        limits = httpx.Limits(max_connections=self._opts.connections + 4)
        timeout = httpx.Timeout(
            connect=self._opts.connect_timeout,
            read=self._opts.read_timeout,
            write=self._opts.read_timeout,
            pool=self._opts.connect_timeout,
        )
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=False,
            headers={"User-Agent": self._opts.user_agent},
        ) as client:
            # Metadata first. probe() is the strict check, so a missing token, a gated
            # repo or a login page surfaces here with a usable message rather than as
            # twenty mysterious chunk failures later on.
            info = await self._provider.probe(self._identity, client)
            await self._resolve(client)

            dest = self._dest_arg / info.filename if self._dest_arg.is_dir() else self._dest_arg
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")

            state = PartState.load(part)

            if dest.exists() and state is None:
                verdict = await self._existing_matches(dest, info)
                if verdict:
                    return dest

            if not info.size or not info.accept_ranges:
                # Unknown length or no range support: one stream, no resume. Rare for the
                # services we care about, but a plain HTTP file server can land here.
                if state is not None and state.chunk_map.completed_prefix() > 0:
                    # A CDN edge that has temporarily stopped honouring Range must not cost
                    # us a partial download. Restarting from zero is precisely the behaviour
                    # this project exists to prevent, so refuse and let the caller retry
                    # later against a node that behaves.
                    raise RangeIgnored(
                        "the server stopped honouring byte ranges while a partial download "
                        f"of {state.chunk_map.completed_prefix()} bytes exists — refusing to "
                        "restart from zero"
                    )
                return await self._run_unsized(client, dest, part, info)

            chunk_size = self._opts.chunk_size or pick_chunk_size(info.size)
            if state is not None:
                try:
                    state.check_resumable(self._identity, info, part)
                    self._map = state.chunk_map
                except RemoteChanged:
                    state.discard()
                    part.unlink(missing_ok=True)
                    state = None
            if state is None:
                self._map = ChunkMap(info.size, chunk_size)
                state = PartState.create(part, self._identity, info, self._map)
            self._state = state

            self._allocate(part, info.size)

            connections = self._opts.connections
            if self._opts.respect_disk_kind:
                connections = recommend_connections(
                    part, connections, info.size, self._opts.disk_kind
                )
            # Deliberately not capped at the number of remaining chunks. A worker with
            # nothing to claim is not wasted — it waits, then splits somebody's chunk and
            # takes half. That is exactly what keeps the tail from collapsing onto one
            # unlucky connection.
            connections = max(1, connections)

            hasher: PrefixHasher | None = None
            if self._opts.verify_hash:
                hasher = PrefixHasher(part, self._map.completed_prefix)
                hasher.start()

            # Everything below runs under a guard that shuts the hasher down on any exit.
            # A background task that outlives a failed transfer keeps reading a file that
            # is about to be renamed or deleted, and in a long-lived process one leaks per
            # failure.
            try:
                reporter = asyncio.create_task(self._report_loop(info.size), name="progress")
                try:
                    workers = [
                        asyncio.create_task(self._worker(client, part), name=f"chunk-{i}")
                        for i in range(connections)
                    ]
                    try:
                        await self._gather_workers(workers)
                    finally:
                        state.flush(force=True)
                finally:
                    reporter.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reporter

                if not self._map.complete:
                    raise TransferFailed("workers finished with chunks still missing")

                if hasher is not None:
                    digest = await hasher.finish(info.size)
                    if info.sha256 and digest != info.sha256:
                        corrupt = part.with_name(part.name + ".corrupt")
                        part.replace(corrupt)
                        state.discard()
                        raise ChecksumMismatch(
                            f"expected {info.sha256}, got {digest}; kept the data at {corrupt}"
                        )
            finally:
                if hasher is not None:
                    await hasher.aclose()

            os.replace(part, dest)
            state.discard()
            return dest

    async def _existing_matches(self, dest: Path, info) -> bool:
        """Is the file already sitting there the one we were asked for?

        Re-fetching 18 GB that is already on disk is the single most annoying thing a
        download manager can do. Hashing to find out costs a read of the file, which is
        one or two orders of magnitude cheaper than the transfer it avoids.

        A file of a different size is a different file, and gets replaced. Size alone is
        weak evidence, so it is only accepted when no hash was advertised or when the
        caller has turned hashing off.
        """
        actual = dest.stat().st_size
        if info.size is not None and actual != info.size:
            return False
        if not (info.sha256 and self._opts.verify_hash and self._opts.verify_existing):
            return info.size is not None and actual == info.size

        digest = await asyncio.to_thread(hash_file, dest)
        return digest == info.sha256

    # --- resolution -------------------------------------------------------

    async def _resolve(
        self, client: httpx.AsyncClient, stale_url: str | None = None
    ) -> ResolvedTarget:
        """Return a usable target, minting a new one only when the current one is spent.

        `stale_url` is the URL that just got rejected. Identifying the dead target by URL
        rather than by a timestamp matters: when six connections all hit 403 on the same
        expired signature they arrive here one after another, and only the first should pay
        for a round trip — the rest must see that the URL already changed and take the new
        one. A time-based guard would instead make the stragglers reuse the dead URL.
        """
        async with self._resolve_lock:
            current = self._target
            if current is not None:
                if stale_url is not None:
                    if current.url != stale_url:
                        return current  # somebody already refreshed it
                elif not current.is_stale():
                    return current
            self._target = await self._provider.resolve(self._identity, client)
            self._resolved_at = time.monotonic()
            return self._target

    # --- workers ----------------------------------------------------------

    async def _gather_workers(self, workers: list[asyncio.Task[None]]) -> None:
        """Run workers; on the first terminal failure cancel the rest and re-raise it."""
        done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
        error: BaseException | None = None
        for task in done:
            exc = task.exception()
            if exc is not None:
                error = exc
                break
        if error is not None:
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            raise error
        for task in pending:
            await task

    async def _worker(self, client: httpx.AsyncClient, part: Path) -> None:
        assert self._map is not None
        # A dedicated handle per worker: each gets its own file position, so concurrent
        # writers need no locking. `buffering=0` matters — the hasher reads this file
        # through a separate handle and would not see data parked in a userspace buffer.
        with open(part, "r+b", buffering=0) as fh:
            while True:
                chunk = self._map.claim()
                if chunk is None:
                    if self._map.complete:
                        return
                    # Nothing to claim or steal right now — the remaining work is too small
                    # to split. Stay alive rather than exit: a chunk may yet grow stealable
                    # if its owner stalls, and an exited worker cannot come back.
                    await asyncio.sleep(IDLE_POLL)
                    continue
                try:
                    await self._run_chunk(client, chunk, fh)
                finally:
                    self._map.release(chunk)

    async def _run_chunk(self, client: httpx.AsyncClient, chunk: Chunk, fh) -> None:
        attempts = 0
        last: Exception | None = None
        while not chunk.complete:
            before = chunk.done
            target = await self._resolve(client)
            try:
                await self._fetch(client, target, chunk, fh)
            except Retryable as exc:
                last = exc
                attempts += 1
                if isinstance(exc, (SignatureExpired, RangeIgnored)):
                    # Mint a replacement now, naming the URL that died so concurrent
                    # workers hitting the same wall do not each trigger a round trip.
                    # An expired signature has to be replaced; a URL whose edge stopped
                    # honouring Range does not, but retrying the same one is how you get
                    # the same edge again, and a re-resolve is the only lever we have on
                    # which node answers.
                    await self._resolve(client, stale_url=target.url)
            else:
                # No exception but no progress either (server closed early). Still counts
                # against the budget, otherwise this loops forever.
                attempts = 0 if chunk.done > before else attempts + 1

            if chunk.complete:
                return
            if attempts >= self._opts.max_attempts:
                raise TransferFailed(
                    f"chunk {chunk.index} failed after {attempts} attempts: {last}"
                ) from last
            if attempts:
                await asyncio.sleep(
                    _backoff(attempts, self._opts.backoff_base, self._opts.backoff_max)
                )

    async def _fetch(
        self, client: httpx.AsyncClient, target: ResolvedTarget, chunk: Chunk, fh
    ) -> None:
        assert self._map is not None and self._state is not None
        start = chunk.offset
        if start >= chunk.end:
            return

        headers = {
            **target.headers,
            "Range": f"bytes={start}-{chunk.end - 1}",
            # Content-coding would make the byte arithmetic meaningless.
            "Accept-Encoding": "identity",
        }
        tracker = SpeedTracker(self._opts.stall_window, self._opts.stall_min_span)
        self._active += 1
        try:
            async with client.stream(
                "GET", target.url, headers=headers, follow_redirects=False
            ) as resp:
                self._validate(resp, start, chunk.end - 1)
                fh.seek(start)
                async for data in resp.aiter_bytes():
                    if not data:
                        continue
                    room = chunk.end - chunk.offset
                    if room <= 0:
                        break
                    if len(data) > room:
                        data = data[:room]
                    if self._limiter is not None:
                        # Held before the write, so waiting here stops reading the socket
                        # and the pause is pushed back to the sender rather than buffered.
                        await self._limiter.take(len(data))
                    _write_all(fh, data)
                    # Only now, with the bytes handed to the OS, may `done` advance — the
                    # hasher trusts everything below it.
                    chunk.done += len(data)
                    self._transferred += len(data)
                    self._state.touch()
                    self._state.flush()
                    tracker.add(len(data))
                    if tracker.is_stalled(self._stall_floor()):
                        raise Stalled(
                            f"chunk {chunk.index}: {tracker.rate():.0f} B/s over the last "
                            f"{self._opts.stall_window:.0f}s"
                        )
        except httpx.HTTPError as exc:
            raise TransportError(f"chunk {chunk.index}: {exc}") from exc
        finally:
            self._active -= 1

    def _stall_floor(self) -> float:
        """The throughput below which a connection counts as dead rather than slow.

        A speed ceiling has to move this floor, or the watchdog starts shooting the very
        connections we are deliberately holding back: the cap is shared out between them,
        so each one is *meant* to be slow. Half of its share leaves the check doing its real
        job — a connection delivering nothing still reads as nothing.
        """
        floor = self._opts.min_speed
        if self._limiter is not None and self._limiter.rate > 0:
            share = self._limiter.rate / max(1, self._opts.connections)
            floor = min(floor, share / 2)
        return floor

    # --- response validation ---------------------------------------------

    def _validate(self, resp: httpx.Response, start: int, end: int) -> None:
        status = resp.status_code
        size = self._map.size if self._map else None

        if status in (301, 302, 303, 307, 308):
            # The provider is supposed to hand us a fully resolved target. Following
            # redirects here would bypass the cross-host Authorization stripping.
            raise RangeNotHonored(
                f"provider returned a URL that still redirects ({status})"
            )
        if status == 416:
            # Our own byte arithmetic is out of step with the server. Retrying cannot help.
            raise RangeNotHonored("server rejected the byte range as unsatisfiable")
        if status >= 500:
            raise TransportError(f"server error {status}")
        if status >= 400:
            # Any 4xx here means the signed URL is spent. Permission problems do not
            # surface at this point — resolve() probes before a transfer starts, and the
            # re-resolve this triggers will raise AuthRequired or AccessDenied if access
            # was genuinely revoked. Guessing between "expired" and "denied" from the
            # status code alone is what turns a routine expiry into a failed download.
            raise SignatureExpired(f"signed URL rejected ({status})")

        if status == 200:
            # The server ignored Range and is about to send the whole file. Appending it
            # would corrupt the part file; truncating and restarting would throw away good
            # bytes. Refuse, and let the retry loop get a fresh URL.
            if start != 0 or (size is not None and end != size - 1):
                raise RangeIgnored(
                    f"asked for bytes {start}-{end} but the server sent the whole file"
                )
        elif status == 206:
            header = resp.headers.get("content-range", "")
            match = _CONTENT_RANGE.search(header)
            if not match:
                raise RangeIgnored(f"206 with unparsable Content-Range: {header!r}")
            got_start, _got_end, total = int(match.group(1)), match.group(2), match.group(3)
            if got_start != start:
                raise RangeIgnored(
                    f"asked to resume at {start} but the server started at {got_start}"
                )
            if size is not None and total != "*" and int(total) != size:
                raise RemoteChanged(f"file size changed: {size} -> {total}")
        else:
            raise TransportError(f"unexpected status {status}")

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type in HTML_CONTENT_TYPES:
            raise NotBinaryContent(
                "the server returned an HTML page instead of file data — "
                "this is usually a login or consent page"
            )

    # --- single-stream fallback ------------------------------------------

    async def _run_unsized(
        self, client: httpx.AsyncClient, dest: Path, part: Path, info
    ) -> Path:
        """No known size or no range support: one pass, no resume, still watched for stalls."""
        target = await self._resolve(client)
        tracker = SpeedTracker(self._opts.stall_window, self._opts.stall_min_span)
        written = 0
        with open(part, "wb", buffering=0) as fh:
            async with client.stream(
                "GET", target.url, headers={**target.headers, "Accept-Encoding": "identity"},
                follow_redirects=False,
            ) as resp:
                self._validate_unsized(resp)
                async for data in resp.aiter_bytes():
                    if self._limiter is not None:
                        await self._limiter.take(len(data))
                    _write_all(fh, data)
                    written += len(data)
                    self._transferred += len(data)
                    tracker.add(len(data))
                    if tracker.is_stalled(self._stall_floor()):
                        raise Stalled(f"{tracker.rate():.0f} B/s — giving up on this connection")
                    self._emit(ProgressSnapshot(
                        downloaded=written, total=info.size,
                        speed=self._speed.update(written), connections=1,
                        hashed=0, eta=None,
                    ))
        os.replace(part, dest)
        return dest

    def _validate_unsized(self, resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise AuthRequired("the service requires a token for this file")
        if resp.status_code in (403, 410):
            raise AccessDenied(f"server refused the download ({resp.status_code})")
        if resp.status_code >= 400:
            raise TransportError(f"unexpected status {resp.status_code}")
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type in HTML_CONTENT_TYPES:
            raise NotBinaryContent("the server returned an HTML page instead of file data")

    # --- progress ---------------------------------------------------------

    async def _report_loop(self, total: int) -> None:
        while True:
            await asyncio.sleep(self._opts.progress_interval)
            if self._map is None:
                continue
            downloaded = self._map.downloaded
            speed = self._speed.update(downloaded)
            remaining = total - downloaded
            self._emit(ProgressSnapshot(
                downloaded=downloaded,
                total=total,
                speed=speed,
                connections=self._active,
                hashed=self._map.completed_prefix(),
                eta=remaining / speed if speed > 0 else None,
            ))

    def _emit(self, snapshot: ProgressSnapshot) -> None:
        if self._on_progress is None:
            return
        result = self._on_progress(snapshot)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)  # noqa: RUF006 — fire and forget by design

    # --- disk -------------------------------------------------------------

    @staticmethod
    def _allocate(part: Path, size: int) -> None:
        """Size the .part file up front when that is cheap; see sfd.core.prealloc."""
        allocate(part, size)


def hash_file(path: Path) -> str:
    """Plain sequential SHA256 of a finished file, for skip checks and other engines."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_all(fh, data: bytes) -> None:
    """Raw handles may accept a short write; loop until the buffer is drained."""
    view = memoryview(data)
    while view:
        written = fh.write(view)
        if not written:
            raise TransportError("write returned 0 bytes")
        view = view[written:]


def _backoff(attempt: int, base: float, cap: float) -> float:
    """Exponential with jitter — the jitter keeps parallel connections from re-colliding."""
    return min(cap, base * (2 ** min(attempt, 6))) * (0.7 + random.random() * 0.6)
