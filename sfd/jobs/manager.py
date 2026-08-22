"""The queue runner.

Owns the lifecycle of every download: expanding a pasted link into files, working out where
each belongs, running transfers a few at a time, and surviving being closed mid-download.

Pausing is a cancellation, not a suspension. The transfer's own state file already makes a
partial `.part` resumable from any point, so dropping the connections and coming back later
costs nothing but the bytes in flight — and it frees the socket instead of holding it open
across a lunch break.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from ..core.diskinfo import free_bytes
from ..core.errors import (
    AccessDenied,
    AuthRequired,
    ChecksumMismatch,
    NotBinaryContent,
    NotEnoughSpace,
    RangeNotHonored,
    RemoteChanged,
    SfdError,
)
from ..core.speed import RateLimiter
from ..core.state import PartState
from ..core.transfer import Transfer, TransferOptions, hash_file
from ..engines.hf_hub import HfHubEngine, HfHubOptions
from ..core.types import FileIdentity, ProgressSnapshot
from ..library import relocate, sidecar
from ..library.categories import ALIASES
from ..library.classify import Verdict, classify
from ..library.inspect import sniff_remote
from ..library.layout import Layout, adopt, flat
from ..providers.base import Provider
from ..providers.civitai import DEFAULT_HOST as CIVITAI_DEFAULT_HOST
from ..providers.civitai import CivitaiProvider
from ..providers.direct import DirectProvider
from ..providers.huggingface import HuggingFaceProvider
from ..providers.registry import Item, expand, source_url
from ..settings import Settings
from . import db
from .db import Database, Task

PROGRESS_INTERVAL = 0.4
# How often progress is written to the queue. Far slower than the event stream, because its
# only job is to survive a page reload or a restart — but it must happen, or a reloaded page
# shows 0% for a download that is eight gigabytes in, which reads exactly like the restart
# this project exists to prevent.
PERSIST_INTERVAL = 5.0
# Slack kept between the queue and a completely full volume. Filling the last byte of a
# system disk breaks more than this download.
SPACE_HEADROOM = 64 * 1024**2

# How long to wait before each automatic attempt, and how many there are. Wide gaps on
# purpose: what these recover from is an outage — a router rebooting, a CDN edge having a
# bad ten minutes — and hammering a service that just refused us is how a temporary failure
# becomes a permanent one.
RETRY_DELAYS = (30.0, 120.0, 600.0)
RETRY_POLL = 5.0

# Failures no amount of waiting fixes. Everything else — a reset connection, an expired
# signature, a chunk that ran out of its own retries — is worth another go later.
NO_RETRY = (
    AccessDenied, AuthRequired, ChecksumMismatch, NotBinaryContent,
    NotEnoughSpace, RangeNotHonored, RemoteChanged,
)


def _gb(size: float) -> str:
    return f"{size / 1024**3:.1f} GB"


class Manager:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.db = database
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._running: dict[int, asyncio.Task[None]] = {}
        # Keyed by slot, so a worker keeps its identity across a resize and the ones past
        # the limit know they are the ones to go.
        self._workers: dict[int, asyncio.Task[None]] = {}
        # Moves in flight, by task. A cross-drive move is a copy on a worker thread, and a
        # thread cannot be cancelled — the event is the only way to ask it to stop, and it
        # lives here so any request, from any tab, can do the asking.
        self._moves: dict[int, threading.Event] = {}
        self._retries: asyncio.Task[None] | None = None
        self._wanted = 0
        # One ceiling for everything running, adjustable while it runs.
        self.limiter = RateLimiter(max(0.0, settings.max_speed_kb * 1024))
        self._wake = asyncio.Event()
        self._stopping = False
        self._last_emit: dict[int, float] = {}

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        self._stopping = False
        self.reconcile()
        self.resize()
        self._retries = asyncio.create_task(self._retry_loop(), name="queue-retries")

    def apply_settings(self) -> None:
        """Take up the settings that are allowed to change while the queue is running."""
        self.limiter.rate = max(0.0, self.settings.max_speed_kb * 1024)
        self.resize()

    def resize(self) -> int:
        """Match the worker pool to the setting.

        Called again whenever the setting is saved, because a number that only takes effect
        after a restart reads as a broken control — especially next to `connections`, which
        is picked up by the very next download.

        A worker past the new limit retires when its current download finishes; it is not
        cancelled. The setting says how many run at once, not that one nine tenths of the
        way through should be dropped.
        """
        self._wanted = max(1, int(self.settings.concurrent_downloads))
        for slot, worker in list(self._workers.items()):
            if worker.done():
                del self._workers[slot]
        for slot in range(self._wanted):
            if slot not in self._workers:
                self._workers[slot] = asyncio.create_task(
                    self._worker(slot), name=f"queue-worker-{slot}"
                )
        # Wakes the idle ones so a shrink takes effect now rather than on their next poll.
        self._wake.set()
        return self._wanted

    @property
    def workers(self) -> int:
        """Workers still alive. A retiring one is counted until it actually returns."""
        return sum(1 for worker in self._workers.values() if not worker.done())

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        for task in list(self._running.values()):
            task.cancel()
        for worker in self._workers.values():
            worker.cancel()
        if self._retries is not None:
            self._retries.cancel()
        for task in [*self._running.values(), *self._workers.values(), self._retries]:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._workers.clear()
        self._running.clear()
        self._retries = None

    def reconcile(self) -> None:
        """Re-read progress from the files themselves.

        The queue's counter is only ever a cache; each `.part` carries its own state file
        and that is the thing that is actually true. After a crash — or after any gap
        between the last persisted number and the process dying — this is what stops a
        half-finished 24 GB download from being displayed as untouched.
        """
        for task in self.db.list():
            if task.state in (db.DONE, db.FAILED) or not task.dest:
                continue

            destination = Path(task.dest)
            if destination.exists() and task.size and destination.stat().st_size == task.size:
                # Finished during the gap: the transfer renamed it into place.
                self.db.update(task.id, state=db.DONE, downloaded=task.size)
                continue

            state = PartState.load(destination.with_name(destination.name + ".part"))
            if state is None:
                continue
            actual = state.chunk_map.downloaded
            if actual != task.downloaded:
                self.db.update(task.id, downloaded=actual)

    # --- events -----------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def emit(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A browser tab that stopped reading must not stall the downloads.
                self._subscribers.discard(queue)

    def _emit_task(self, task_id: int) -> None:
        task = self.db.get(task_id)
        if task is not None:
            self.emit({"type": "task", "task": task.to_json()})

    # --- adding work ------------------------------------------------------

    async def add(self, source: str) -> list[Task]:
        """Expand a pasted link and queue everything it names."""
        layout = self.layout()
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
        created: list[Task] = []

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resolution = await expand(
                source,
                client,
                hf_token=self.settings.effective_hf_token,
                civitai_token=self.settings.effective_civitai_token,
            )
            # Reserved for the whole expansion at once, so the files of one link keep the
            # order the provider listed them in whichever end of the queue they join.
            positions = self.db.reserve_positions(
                len(resolution.items), self.settings.queue_position
            )
            for item, position in zip(resolution.items, positions):
                verdict = None
                if self.settings.library_root:
                    verdict = await self._classify(resolution.provider, item, client)
                    destination = layout.destination(verdict, item.filename)
                else:
                    destination = layout.root / item.filename if item.filename else layout.root

                # An uncertain placement outranks auto-start: it is a question, not a queue
                # position, and answering it is what releases the task.
                if verdict and verdict.needs_confirmation:
                    state = db.BLOCKED
                else:
                    state = db.PENDING if self.settings.auto_start else db.PAUSED

                task = self.db.add(
                    state=state,
                    source=source,
                    label=resolution.label,
                    provider=resolution.provider.name,
                    identity={
                        "provider": item.identity.provider,
                        "ref": item.identity.ref,
                    },
                    filename=item.filename,
                    size=item.size,
                    sha256=item.sha256,
                    dest=destination,
                    category=verdict.category.value if verdict else None,
                    confidence=verdict.confidence if verdict else None,
                    reason=verdict.reason if verdict else None,
                    disagreement=(
                        verdict.disagreement.value if verdict and verdict.disagreement else None
                    ),
                    base_model=verdict.base_model if verdict else None,
                    meta=item.meta,
                    position=position,
                )
                if task is not None:
                    created.append(task)
                    self.emit({"type": "task", "task": task.to_json()})

        self._wake.set()
        return created

    async def _classify(
        self, provider: Provider, item: Item, client: httpx.AsyncClient
    ) -> Verdict:
        if item.size is None or not item.filename:
            try:
                info = await provider.probe(item.identity, client)
                item.size = item.size or info.size
                item.sha256 = item.sha256 or info.sha256
                item.filename = item.filename or info.filename
            except SfdError:
                pass
        header = await sniff_remote(provider, item.identity, client)
        return classify(item.filename, item.meta, header)

    # --- commands ---------------------------------------------------------

    def pause(self, task_id: int) -> None:
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
        # Pausing also calls off a booked retry: the answer to "start again in ten minutes"
        # is no longer yes once someone has said stop.
        self.db.update(task_id, state=db.PAUSED, retry_at=None)
        self._emit_task(task_id)

    def resume(self, task_id: int) -> None:
        self.db.update(task_id, state=db.PENDING, error=None, retry_at=None)
        self._emit_task(task_id)
        self._wake.set()

    def retry(self, task_id: int) -> None:
        """Asked for by hand, which also forgives the attempts spent so far.

        Otherwise a task that used its three tries at 3am gets exactly one more forever,
        even after the thing that broke it has been fixed.
        """
        self.db.update(task_id, attempts=0)
        self.resume(task_id)

    def confirm(
        self, task_id: int, category: str | None = None, folder: Path | None = None
    ) -> None:
        """Accept a placement the classifier was not sure about, optionally correcting it.

        A folder outranks a category, and is taken exactly as given: nothing is appended to
        it, not even base-model grouping. The picker shows the whole directory, so what was
        chosen is what the file gets — a path quietly extended underneath the person who
        typed it is how you end up hunting for a model that downloaded successfully.
        """
        task = self.db.get(task_id)
        if task is None:
            return

        updates: dict[str, Any] = {"state": db.PENDING, "error": None}
        if folder is not None:
            # A folder that names a kind says what the file is as well as where it goes; one
            # that does not — `sams`, `insightface`, whatever a custom node brought — leaves
            # the classifier's guess standing rather than inventing a better-sounding one.
            kind = ALIASES.get(folder.name.lower())
            updates["category"] = kind.value if kind else task.category
            updates["confidence"] = "high"
            updates["reason"] = f"filed by hand into {folder.name}"
            updates["disagreement"] = None
            updates["dest"] = str(folder / task.filename)
        elif category and category != task.category:
            verdict = Verdict(
                _category(category), "high", "chosen by hand", base_model=task.base_model
            )
            updates["category"] = category
            updates["confidence"] = "high"
            updates["reason"] = "chosen by hand"
            updates["disagreement"] = None
            updates["dest"] = str(self.layout().destination(verdict, task.filename))

        self.db.update(task_id, **updates)
        self._emit_task(task_id)
        self._wake.set()

    async def move(self, task_id: int, folder: Path) -> relocate.Move:
        """Put a finished download in a different folder, sidecars and all.

        The correction for a classifier that guessed wrong. It runs on a thread because a
        library spread across two drives turns this into a copy of the whole file, and the
        queue must keep running while that happens.
        """
        task = self.db.get(task_id)
        if task is None:
            raise LookupError("no such task")
        if task.state != db.DONE or not task.dest:
            raise ValueError("only a finished download can be moved")

        if task_id in self._moves:
            raise ValueError("this file is already being moved")

        stop = threading.Event()
        self._moves[task_id] = stop
        try:
            result = await asyncio.to_thread(
                relocate.move,
                Path(task.dest),
                folder,
                sidecar_dir=Path(self.settings.sidecar_dir) if self.settings.sidecar_dir else None,
                library_root=Path(self.settings.library_root) if self.settings.library_root else None,
                progress=self._move_progress(task_id),
                stop=stop,
            )
        finally:
            # Whatever happened, nothing is copying any more, and every page watching needs
            # to hear that as much as it heard the progress.
            self._moves.pop(task_id, None)
            self.emit({"type": "moved", "id": task_id})

        if result.unchanged:
            return result

        # The card explains where the file went and why. Leaving the old guess on it after
        # a person has overruled it by hand is how the explanation stops being true.
        kind = ALIASES.get(folder.name.lower())
        self.db.update(
            task_id,
            dest=str(result.path),
            category=kind.value if kind is not None else task.category,
            confidence="high",
            reason=f"moved by hand into {folder.name}",
            disagreement=None,
        )
        self._emit_task(task_id)
        return result

    def stop_move(self, task_id: int) -> bool:
        """Ask an in-flight move to give up. True if there was one to ask."""
        stop = self._moves.get(task_id)
        if stop is None:
            return False
        stop.set()
        return True

    def _move_progress(self, task_id: int) -> relocate.Progress:
        """Report a cross-drive move, which is a copy and therefore has a duration.

        Called from the worker thread doing the copying, so the event cannot be put on the
        subscriber queues directly — those belong to the loop. Throttled to the same rhythm
        as download progress: a 4 MB block off an NVMe drive arrives faster than anyone can
        read, and the page redraws for every one of them.
        """
        loop = asyncio.get_running_loop()
        last = 0.0

        def report(copied: int, total: int) -> None:
            nonlocal last
            now = time.monotonic()
            if copied < total and now - last < PROGRESS_INTERVAL:
                return
            last = now
            loop.call_soon_threadsafe(
                self.emit,
                {"type": "moving", "id": task_id, "copied": copied, "total": total},
            )

        return report

    def cancel(self, task_id: int) -> None:
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
        self.db.delete(task_id)
        self.emit({"type": "removed", "id": task_id})

    def start_all(self) -> int:
        """Release everything that is merely waiting for permission to begin.

        Blocked tasks are left alone on purpose. They are waiting on a decision about where
        the file belongs, and starting them wholesale would file models by a guess the
        classifier already said it was not confident in.
        """
        released = 0
        for task in self.db.list([db.PAUSED]):
            self.db.update(task.id, state=db.PENDING, error=None)
            released += 1
        if released:
            self.emit({"type": "reload"})
            self._wake.set()
        return released

    def clear_finished(self) -> int:
        removed = self.db.clear([db.DONE])
        self.emit({"type": "reload"})
        return removed

    # --- the worker loop --------------------------------------------------

    async def _worker(self, slot: int = 0) -> None:
        while not self._stopping and slot < self._wanted:
            task = self.db.claim_next()
            if task is None:
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=5.0)
                continue

            self._emit_task(task.id)
            runner = asyncio.create_task(self._run(task), name=f"download-{task.id}")
            self._running[task.id] = runner
            try:
                await runner
            except asyncio.CancelledError:
                # Paused or cancelled from the UI; the .part file keeps the progress.
                if self.db.get(task.id) is not None:
                    current = self.db.get(task.id)
                    if current and current.state == db.RUNNING:
                        self.db.update(task.id, state=db.PAUSED)
                        self._emit_task(task.id)
            except Exception as exc:  # noqa: BLE001 - a failed download must not kill the worker
                self._fail(task, exc)
            finally:
                self._running.pop(task.id, None)
                self._last_emit.pop(task.id, None)

    async def _run(self, task: Task) -> None:
        kind = task.identity.get("provider", task.provider)
        identity = FileIdentity(provider=kind, ref=task.identity.get("ref", {}))
        destination = Path(task.dest)

        try:
            self._require_space(task, destination)
        except SfdError as exc:
            self._fail(task, exc)
            return

        use_hub = kind == "huggingface" and self.settings.hf_engine == "hf_hub"
        if use_hub:
            try:
                await self._run_hf_hub(task, identity, destination)
                return
            except SfdError as exc:
                if not self.settings.hf_fallback:
                    self._fail(task, exc)
                    return
                self.emit({
                    "type": "note", "id": task.id,
                    "message": f"hf_hub engine failed ({exc}); trying the native transfer",
                })

        try:
            await self._run_native(task, identity, destination)
        except SfdError as exc:
            # The official client speaks protocols we do not, so it occasionally succeeds
            # where the native path is refused. One extra attempt, only when it might help.
            if kind == "huggingface" and self.settings.hf_fallback and not use_hub:
                self.emit({
                    "type": "note", "id": task.id,
                    "message": f"native transfer failed ({exc}); trying huggingface_hub",
                })
                try:
                    await self._run_hf_hub(task, identity, destination)
                    return
                except SfdError as fallback_error:
                    self._fail(task, fallback_error)
                    return
            self._fail(task, exc)

    def _require_space(self, task: Task, destination: Path) -> None:
        """Refuse a file the volume cannot hold, before a byte of it is fetched.

        Said at the start, this is one line to read and act on. Said by the file system
        forty gigabytes in, it is an OSError from a chunk writer at four in the morning,
        with the whole queue behind it failing the same way.
        """
        remaining = (task.size or 0) - task.downloaded
        if remaining <= 0:
            return
        free = free_bytes(destination.parent)
        if free is None or free >= remaining + SPACE_HEADROOM:
            return
        raise NotEnoughSpace(
            f"{_gb(remaining)} still to fetch, {_gb(free)} free on "
            f"{destination.drive or destination.parent}"
        )

    def _fail(self, task: Task, exc: Exception) -> None:
        """Record a failure, and book another attempt when one might help.

        The wait grows because what this recovers from is an outage, and the queue it was
        written for is one left running overnight. What it must not do is spin: a wrong
        token or a full disk is answered by a person, not by asking again in thirty seconds.
        """
        attempts = task.attempts + 1
        eligible = (
            self.settings.auto_retry
            and not isinstance(exc, NO_RETRY)
            and attempts <= len(RETRY_DELAYS)
        )
        retry_at = time.time() + RETRY_DELAYS[attempts - 1] if eligible else None
        self.db.update(
            task.id, state=db.FAILED, error=str(exc), attempts=attempts, retry_at=retry_at,
        )
        self._emit_task(task.id)

    async def _retry_loop(self) -> None:
        """Put failed tasks back in the queue once their wait is up."""
        while not self._stopping:
            for task in self.db.due_retries(time.time()):
                self.db.update(task.id, state=db.PENDING, error=None, retry_at=None)
                self.emit({
                    "type": "note", "id": task.id,
                    "message": f"retrying (attempt {task.attempts + 1})",
                })
                self._emit_task(task.id)
                self._wake.set()
            await asyncio.sleep(RETRY_POLL)

    async def _run_hf_hub(
        self, task: Task, identity: FileIdentity, destination: Path
    ) -> None:
        ref = task.identity.get("ref", {})
        engine = HfHubEngine(
            self.settings.effective_hf_token,
            HfHubOptions(**self.settings.hf_hub_options),
        )
        result = await engine.download_file(
            str(ref["repo_id"]),
            str(ref["path"]),
            destination,
            repo_type=str(ref.get("repo_type") or "model"),
            revision=str(ref.get("revision") or "main"),
            total=task.size,
            on_progress=self._progress_reporter(task, {"downloaded": task.downloaded,
                                                       "persisted": 0.0}),
        )

        path = result.paths[0]
        # The other engine does not do our verification, and dropping the guarantee when the
        # engine changes would make it worthless. Hash what landed.
        if task.sha256 and self.settings.verify_hash:
            digest = await asyncio.to_thread(hash_file, path)
            if digest != task.sha256:
                corrupt = path.with_name(path.name + ".corrupt")
                path.replace(corrupt)
                raise ChecksumMismatch(
                    f"expected {task.sha256}, got {digest}; kept the data at {corrupt}"
                )

        self.db.update(
            task.id, state=db.DONE, downloaded=path.stat().st_size, dest=str(path),
            error=None, finished_at=time.time(), transferred=result.transferred,
        )
        if self.settings.write_sidecars:
            await self._write_sidecar(path, task, identity)
        # Deliberately not inside the branch above. The preview beside the model is read by
        # the model managers, and someone who turned our JSON records off did not thereby
        # ask their model manager to stop showing pictures.
        if self.settings.fetch_previews and self.settings.write_compat_files:
            await self._write_preview(path, task)
        self._emit_task(task.id)

    def _progress_reporter(self, task: Task, progress: dict) -> ProgressCallback:
        def on_progress(snapshot: ProgressSnapshot) -> None:
            progress["downloaded"] = snapshot.downloaded
            now = time.monotonic()
            if now - self._last_emit.get(task.id, 0.0) >= PROGRESS_INTERVAL:
                self._last_emit[task.id] = now
                self.emit({
                    "type": "progress", "id": task.id,
                    "downloaded": snapshot.downloaded, "total": snapshot.total,
                    "speed": snapshot.speed, "connections": snapshot.connections,
                    "eta": snapshot.eta,
                })
            if now - progress["persisted"] >= PERSIST_INTERVAL:
                progress["persisted"] = now
                self.db.update(task.id, downloaded=snapshot.downloaded)

        return on_progress

    async def _run_native(
        self, task: Task, identity: FileIdentity, destination: Path
    ) -> None:
        provider = self._provider_for(identity.provider, task.meta)

        progress = {"downloaded": task.downloaded, "persisted": 0.0}
        on_progress = self._progress_reporter(task, progress)

        transfer = Transfer(
            provider, identity, destination, self._options(), on_progress, self.limiter
        )
        try:
            path = await transfer.run()
        except SfdError:
            # Recorded, but re-raised: the caller decides whether the other engine is worth
            # a try before this counts as a failure.
            self.db.update(task.id, downloaded=progress["downloaded"])
            raise
        except asyncio.CancelledError:
            # Paused from the UI. Record how far it got, so the row does not snap back to
            # zero while the .part file quietly holds the bytes.
            self.db.update(task.id, downloaded=progress["downloaded"])
            raise

        size = path.stat().st_size
        self.db.update(
            task.id,
            state=db.DONE,
            downloaded=size,
            dest=str(path),
            error=None,
            finished_at=time.time(),
            # Bytes this run actually pulled: zero for a file that was already present,
            # and only the remainder for a resumed one.
            transferred=transfer.bytes_transferred,
        )
        if self.settings.write_sidecars:
            await self._write_sidecar(path, task, identity)
        if self.settings.fetch_previews and self.settings.write_compat_files:
            await self._write_preview(path, task)
        self._emit_task(task.id)

    async def _write_sidecar(self, path: Path, task: Task, identity: FileIdentity) -> None:
        verdict = Verdict(
            _category(task.category or "other"),
            task.confidence or "low",
            task.reason or "",
            base_model=task.base_model,
            disagreement=_category(task.disagreement) if task.disagreement else None,
        )
        record = sidecar.Record(
            filename=path.name,
            provider=task.provider,
            source_url=source_url(identity, task.meta.get("host")),
            sha256=task.sha256,
            size=task.size,
            meta=task.meta,
        )
        sidecar.write(
            path,
            verdict,
            record,
            sidecar_dir=Path(self.settings.sidecar_dir) if self.settings.sidecar_dir else None,
            library_root=Path(self.settings.library_root) if self.settings.library_root else None,
            compat=self.settings.write_compat_files,
            triggers=self.settings.write_trigger_txt,
        )

    async def _write_preview(self, path: Path, task: Task) -> None:
        """Put `<model>.preview.png` beside the file, at full size.

        Not the cached thumbnail the page draws: that one is 320 pixels wide because it is
        going into a 56-pixel row, and a model manager showing it as a card would render a
        blurred stamp.
        """
        if not task.meta.get("preview_url"):
            return
        timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await sidecar.fetch_preview(path, task.meta, client)

    # --- configuration ----------------------------------------------------

    def layout(self) -> Layout:
        if not self.settings.library_root:
            return flat(Path(self.settings.download_dir))
        layout = adopt(Path(self.settings.library_root), self.settings.profile)
        layout.group_by_base_model = self.settings.group_by_base_model
        for key, path in (self.settings.layout_overrides or {}).items():
            with contextlib.suppress(ValueError):
                layout.paths[_category(key)] = Path(path)
        return layout

    def _options(self) -> TransferOptions:
        return TransferOptions(
            connections=self.settings.connections,
            min_speed=self.settings.min_speed_kb * 1024,
            verify_hash=self.settings.verify_hash,
            verify_existing=self.settings.verify_existing,
            disk_kind=self.settings.effective_disk_kind,
        )

    def _provider_for(self, name: str, meta: dict[str, Any] | None = None) -> Provider:
        if name == "huggingface":
            return HuggingFaceProvider(self.settings.effective_hf_token)
        if name == "civitai":
            # The mirror the link came from, remembered per task so a queue can hold
            # downloads from both domains at once.
            return CivitaiProvider(
                self.settings.effective_civitai_token,
                host=(meta or {}).get("host") or CIVITAI_DEFAULT_HOST,
            )
        return DirectProvider()


def _category(value: str):
    from ..library.categories import Category

    return Category(value)


ProgressCallback = Callable[[ProgressSnapshot], None]
