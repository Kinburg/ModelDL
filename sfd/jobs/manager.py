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
import secrets
import threading
import time
from dataclasses import dataclass
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
from ..library import erase, previews, relocate, sidecar
from ..library.categories import ALIASES, Category
from ..library.classify import Verdict, classify
from ..library.files import is_model_file, shard_of, without_variant
from ..library.inspect import sniff_remote
from ..library.layout import Layout, adopt, flat
from ..providers.base import Provider
from ..providers.civitai import DEFAULT_HOST as CIVITAI_DEFAULT_HOST
from ..providers.civitai import CivitaiProvider
from ..providers.direct import DirectProvider
from ..providers.huggingface import HuggingFaceProvider
from ..providers.registry import Item, Resolution, expand, source_url
from ..settings import Settings
from . import db
from .db import Database, Task
from .library import Library

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

# How long a resolved link waits for the answer to "where does it go?". Long enough to think
# about it; not so long that a dialog left open overnight queues yesterday's idea.
RESOLVED_FOR = 1800.0
# Files of one link read at once to classify them: a repository of twenty quantisations is
# twenty header reads, and in a row they would keep the dialog waiting half a minute.
CLASSIFY_AT_ONCE = 4
# Never ticked for anyone: nobody downloads a repository for its `.gitattributes` or for
# the pictures in its model card.
UNWANTED_SUFFIXES = (".gitattributes", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm")

# Failures no amount of waiting fixes. Everything else — a reset connection, an expired
# signature, a chunk that ran out of its own retries — is worth another go later.
NO_RETRY = (
    AccessDenied, AuthRequired, ChecksumMismatch, NotBinaryContent,
    NotEnoughSpace, RangeNotHonored, RemoteChanged,
)


def _gb(size: float) -> str:
    return f"{size / 1024**3:.1f} GB"


@dataclass(slots=True)
class Resolved:
    """A link expanded and classified, waiting for the page to say which files go where."""

    source: str
    resolution: Resolution
    verdicts: list[Verdict]
    at: float
    # The model a "download again" puts back, so the new file lands in its history.
    restores: int | None = None


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
        # Moves of library models, by model — the same stop mechanism as `_moves`, for the
        # moves that start from the library rather than from a download's card.
        self._model_moves: dict[int, threading.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self.library = Library(settings, database, self.emit_threadsafe)
        self._background: set[asyncio.Task[Any]] = set()
        # Links resolved for the page to place, by token — see `resolve_links`.
        self._resolved: dict[str, Resolved] = {}
        # What a search for a missing model found: (model, answers, when), by token.
        self._found: dict[str, tuple[int, list[dict[str, Any]], float]] = {}

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        self._stopping = False
        self._loop = asyncio.get_running_loop()
        self.reconcile()
        self.resize()
        self._retries = asyncio.create_task(self._retry_loop(), name="queue-retries")
        self.library.start()
        # The first walk of the library happens behind the page rather than in front of
        # it: the queue is usable at once, and the tree fills in a moment later.
        self.spawn(self.library.refresh(), "library-first-walk")

    def spawn(self, coroutine, name: str) -> asyncio.Task[Any]:
        """Run something in the background, and keep hold of it until it is done — a task
        nothing refers to can be collected halfway through."""
        task = asyncio.get_running_loop().create_task(coroutine, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def apply_settings(self) -> None:
        """Take up the settings that are allowed to change while the queue is running."""
        self.limiter.rate = max(0.0, self.settings.max_speed_kb * 1024)
        self.resize()
        # The library's folders may be different ones now.
        with contextlib.suppress(RuntimeError):
            self.spawn(self.library.refresh(), "library-after-settings")

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
        for event in self._model_moves.values():
            event.set()
        for event in self._moves.values():
            event.set()
        await self.library.stop()
        for task in list(self._background):
            task.cancel()
        for task in list(self._background):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

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

    def emit_threadsafe(self, event: dict[str, Any]) -> None:
        """`emit` for code running on a worker thread — the library's walk, a hash.

        The subscriber queues belong to the event loop, and putting into one from another
        thread is a race; the loop is asked to do it instead.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            self.emit(event)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.emit(event)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.emit, event)

    def _emit_task(self, task_id: int) -> None:
        task = self.db.get(task_id)
        if task is not None:
            self.emit({"type": "task", "task": task.to_json()})

    # --- adding work ------------------------------------------------------

    async def add(self, source: str) -> list[Task]:
        """Expand a pasted link and queue everything it names."""
        return (await self.add_links(source))["created"]

    async def add_links(self, source: str) -> dict[str, Any]:
        """Expand a pasted link and queue everything it names that is not already here.

        "Already here" means in the library, on disk: a file downloaded before and still
        where it landed is not fetched again. One that was deleted, or went missing, is —
        that is what pasting its link again is for.
        """
        layout = self.layout()
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
        created: list[Task] = []
        skipped: list[Any] = []

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
                have = self._already_have(item)
                if have is not None:
                    skipped.append(have)
                    continue
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

                task = self._add_task(source, resolution, item, verdict, destination, state, position)
                if task is not None:
                    created.append(task)

        self._wake.set()
        return {"created": created, "skipped": skipped}

    def _add_task(
        self,
        source: str,
        resolution: Resolution,
        item: Item,
        verdict: Verdict | None,
        destination: Path,
        state: str,
        position: float,
        **extra: Any,
    ) -> Task | None:
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
            **extra,
        )
        if task is not None:
            self.emit({"type": "task", "task": task.to_json()})
        return task

    # --- asking where a download goes -------------------------------------

    async def resolve_links(self, source: str) -> dict[str, Any]:
        """Expand a pasted link and say what it names and where it could go — queueing
        nothing. The answer is kept under a token until the page says which files go where.

        Everything that takes the network happens here, before the question is put: the
        files a link expands into, and what each of them is, read from its header. What is
        left to answer is the person's alone.
        """
        timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resolution = await expand(
                source,
                client,
                hf_token=self.settings.effective_hf_token,
                civitai_token=self.settings.effective_civitai_token,
            )
            if not resolution.items:
                raise ValueError("the link names no files")
            verdicts = await self._classify_all(resolution, client)
        return self._hold(Resolved(source, resolution, verdicts, time.time()))

    def _hold(self, resolved: Resolved, hint: tuple[Path, str] | None = None) -> dict[str, Any]:
        now = time.time()
        self._resolved = {k: v for k, v in self._resolved.items() if now - v.at < RESOLVED_FOR}
        token = secrets.token_urlsafe(16)
        self._resolved[token] = resolved
        return self._describe(token, resolved, hint)

    async def _classify_all(self, resolution: Resolution, client: httpx.AsyncClient) -> list[Verdict]:
        """What each file of a link is.

        Only model files are read — a `config.json` is what its name says — and of those,
        one of each family: twenty quantisations of one model are one kind of file, and
        reading twenty headers to learn that once kept the question waiting for seconds.
        The smallest of a family is the one read, its header being the quickest to reach.
        """
        gate = asyncio.Semaphore(CLASSIFY_AT_ONCE)
        items = resolution.items
        families: dict[tuple[str, str], list[int]] = {}
        verdicts: list[Verdict | None] = [None] * len(items)
        for index, item in enumerate(items):
            if item.filename and not is_model_file(item.filename, item.size):
                verdicts[index] = classify(item.filename, item.meta, None)
            else:
                families.setdefault(_family(item.filename), []).append(index)

        async def one(members: list[int]) -> None:
            first = min(members, key=lambda i: items[i].size or 0)
            async with gate:
                verdict = await self._classify(resolution.provider, items[first], client)
            for index in members:
                verdicts[index] = verdict if index == first else Verdict(
                    verdict.category, verdict.confidence, verdict.reason,
                    base_model=verdict.base_model, disagreement=verdict.disagreement,
                )

        await asyncio.gather(*(one(members) for members in families.values()))
        return [v for v in verdicts if v is not None]

    def _describe(
        self, token: str, resolved: Resolved, hint: tuple[Path, str] | None = None
    ) -> dict[str, Any]:
        """What the page shows while it asks: the files, which of them are worth having,
        and where they could go — one ranking for each kind of file among them."""
        resolution = resolved.resolution
        items = resolution.items
        have = [self._already_have(item) for item in items]
        checked, structure = _preselect(resolution, have)
        layout = self.layout()
        rankings: dict[str, list[dict[str, Any]]] = {}
        described = []
        main = next((i for i, it in enumerate(items) if checked[i] and _model(it)), None)
        if main is None:
            main = next((i for i, it in enumerate(items) if _model(it)), 0)
        for index, (item, verdict) in enumerate(zip(items, resolved.verdicts)):
            # A config goes wherever the model it belongs to goes.
            source = verdict if _model(item) else resolved.verdicts[main]
            key = f"{source.category.value}|{source.base_model or ''}"
            if key not in rankings:
                rankings[key] = self.library.rank_folders(
                    layout, source.category, source.base_model,
                    item.filename if _model(item) else items[main].filename,
                    meta=item.meta if _model(item) else items[main].meta,
                    confidence=source.confidence, hint=hint,
                )
            model = have[index]
            described.append({
                "index": index,
                "filename": item.filename,
                "relative": item.relative,
                "size": item.size,
                "model_file": _model(item),
                "primary": item.primary,
                "category": verdict.category.value,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
                "base_model": verdict.base_model,
                "model_name": item.meta.get("model_name"),
                "version_name": item.meta.get("version_name"),
                "checked": checked[index],
                "have": None if model is None else {
                    **dict(zip(("root", "relative"), self.library.placement(model.path))),
                    "id": model.id, "path": model.path,
                },
                "ranking": key,
            })
        first = items[main]
        return {
            "token": token,
            "source": resolved.source,
            "label": resolution.label,
            "provider": resolution.provider.name,
            "host": first.meta.get("host") or (
                "huggingface.co" if resolution.provider.name == "huggingface" else ""
            ),
            "items": described,
            "main": main,
            "rankings": rankings,
            "folder_name": resolution.folder_name if len(items) > 1 else None,
            "structure": structure,
            "previews": len(previews.entries(first.meta)),
            "nsfw": bool(first.meta.get("nsfw")),
            "roots": self.library.roots_json(),
        }

    def resolve_again(self, model_id: int | None = None, task_id: int | None = None) -> dict[str, Any]:
        """A file the library once had, as a resolved link: what its download kept, or what
        identifying it found — for the page to ask where it goes back to, the folder it was
        in first. Nothing is fetched to know it; it was all kept."""
        model = self.db.get_model(model_id) if model_id is not None else None
        task = None
        if model_id is not None:
            if model is None:
                raise LookupError("no such model")
            task = next((t for t in self.db.tasks_for_model(model.id) if t.identity.get("ref")), None)
        elif task_id is not None:
            task = self.db.get(task_id)
            if task is None or task.state != db.DONE:
                raise LookupError("no such download")
            model = self.db.get_model(task.model_id) if task.model_id else None
        if model is not None and model.state == db.PRESENT and Path(model.path).is_file():
            raise ValueError(f"it is still in your library at {model.path}")

        if task is not None:
            identity = FileIdentity(
                provider=str(task.identity.get("provider") or task.provider),
                ref=dict(task.identity.get("ref") or {}),
            )
            item = Item(identity=identity, filename=model.filename if model else task.filename,
                        size=task.size, sha256=task.sha256, meta=task.meta)
            verdict = Verdict(_category(task.category) if task.category in Category._value2member_map_
                              else Category.OTHER, task.confidence or "low", task.reason or "",
                              base_model=task.base_model)
            source, label = task.source, task.label
            was = Path(model.path).parent if model else Path(task.dest).parent
        elif model is not None and model.identity.get("provider"):
            identity = FileIdentity(provider=str(model.identity["provider"]),
                                    ref=dict(model.identity.get("ref") or {}))
            item = Item(identity=identity, filename=model.filename, size=model.size,
                        sha256=model.sha256, meta=model.meta)
            verdict = Verdict(_category(model.category) if model.category in Category._value2member_map_
                              else Category.OTHER, model.confidence or "low", model.reason or "",
                              base_model=model.base_model)
            source = source_url(identity, model.meta.get("host")) or ""
            label = model.title or model.filename
            was = Path(model.path).parent
        else:
            raise ValueError(
                "nothing says where this file came from — look for it online, or find the file"
            )
        resolution = Resolution(
            provider=self._provider_for(identity.provider, item.meta), items=[item], label=label
        )
        resolved = Resolved(source, resolution, [verdict], time.time(),
                            restores=model.id if model is not None else None)
        return self._hold(resolved, hint=(was, "where it was"))

    async def find_online(self, model_id: int) -> dict[str, Any]:
        """Where a model could be downloaded from, looked for on the services. What each
        answer would download is kept here, under a token; the page is shown what it is and
        says which one, by its place in the list."""
        answer = await self.library.find_online(model_id)
        now = time.time()
        self._found = {k: v for k, v in self._found.items() if now - v[2] < RESOLVED_FOR}
        token = secrets.token_urlsafe(16)
        self._found[token] = (model_id, answer["found"], now)
        shown = [{k: v for k, v in hit.items() if not k.startswith("_")} for hit in answer["found"]]
        return {**answer, "found": shown, "token": token}

    def resolve_found(self, token: str, index: int) -> dict[str, Any]:
        """One of the answers `find_online` gave, as a link resolved — for the page to ask
        where it goes, the folder the model was in first."""
        held = self._found.get(token)
        if held is None or not 0 <= index < len(held[1]):
            raise LookupError("that search has expired — look again")
        model_id, found, _at = held
        model = self.db.get_model(model_id)
        if model is None:
            raise LookupError("no such model")
        hit = found[index]
        identity = FileIdentity(provider=hit["_identity"]["provider"], ref=dict(hit["_identity"]["ref"]))
        item = Item(identity=identity, filename=model.filename, size=hit.get("size") or model.size,
                    sha256=model.sha256 if hit.get("proven") else None, meta=dict(hit["_meta"]))
        verdict = Verdict(_category(model.category) if model.category in Category._value2member_map_
                          else Category.OTHER, model.confidence or "low", model.reason or "",
                          base_model=model.base_model)
        resolution = Resolution(provider=self._provider_for(identity.provider, item.meta),
                                items=[item], label=hit.get("title") or model.filename)
        resolved = Resolved(hit.get("page") or "", resolution, [verdict], time.time(), restores=model.id)
        return self._hold(resolved, hint=(Path(model.path).parent, "where it was"))

    def redownload_models(self, model_ids: list[int]) -> dict[str, Any]:
        """Queue several missing models again, each back into the folder it was in — the
        answer to "where?" given once for all of them, by asking to put them back."""
        created: list[Task] = []
        failed: list[dict[str, Any]] = []
        for model_id in model_ids:
            try:
                answer = self.resolve_again(model_id=model_id)
                model = self.db.get_model(model_id)
                assert model is not None
                result = self.queue_resolved(answer["token"], [0], Path(model.path).parent)
            except (LookupError, ValueError) as exc:
                failed.append({"id": model_id, "error": str(exc)})
                continue
            if result["created"]:
                created.extend(result["created"])
            else:
                failed.append({"id": model_id, "error": "already downloading"})
        return {"created": created, "failed": failed}

    def resolved_meta(self, token: str) -> dict[str, Any]:
        """The service's description of the main file of a resolved link — its pictures."""
        resolved = self._resolved.get(token)
        if resolved is None:
            raise LookupError("that link has expired — paste it again")
        items = resolved.resolution.items
        main = next((it for it in items if _model(it)), items[0])
        return main.meta

    def drop_resolved(self, token: str) -> None:
        self._resolved.pop(token, None)

    def queue_resolved(
        self, token: str, picks: list[int], folder: Path, keep_structure: bool = False
    ) -> dict[str, Any]:
        """Queue the files the page picked, into the folder it chose.

        The folder is taken exactly as given, like a placement confirmed by hand: nothing is
        added under it, not even a base-model subfolder — except the repository's own
        folders, when asked to keep them. A folder that names a kind says what the files are
        as well as where they go.
        """
        resolved = self._resolved.get(token)
        if resolved is None:
            raise LookupError("that link has expired — paste it again")
        items = resolved.resolution.items
        chosen = sorted({i for i in picks if 0 <= i < len(items)})
        if not chosen:
            raise ValueError("no file was picked")
        container = resolved.resolution.folder_name if keep_structure else None
        kind = ALIASES.get(folder.name.lower())
        state = db.PENDING if self.settings.auto_start else db.PAUSED
        # A model put back keeps what was written about it, whichever folder it lands in.
        back = self.db.get_model(resolved.restores) if resolved.restores else None
        note = back.note if back is not None else None
        positions = self.db.reserve_positions(len(chosen), self.settings.queue_position)
        created: list[Task] = []
        queued: list[str] = []
        for index, position in zip(chosen, positions):
            item = items[index]
            verdict = resolved.verdicts[index]
            if container and item.relative:
                destination = folder / container / Path(*item.relative.split("/"))
            else:
                destination = folder / item.filename
            filed = Verdict(
                kind or verdict.category,
                "high" if kind is not None else verdict.confidence,
                verdict.reason if kind in (None, verdict.category)
                else f"filed by hand into {folder.name}",
                base_model=verdict.base_model,
            )
            task = self._add_task(
                resolved.source, resolved.resolution, item, filed, destination, state, position,
                model_id=resolved.restores, note=note,
            )
            if task is None:
                queued.append(item.filename)
            else:
                created.append(task)
        self._resolved.pop(token, None)
        self._wake.set()
        return {"created": created, "queued": queued}

    def _already_have(self, item: Item):
        """The model a finished download of this very file left, if it is still on disk."""
        identity = {"provider": item.identity.provider, "ref": item.identity.ref}
        for task in self.db.finished_with_identity(identity):
            if task.model_id is None:
                continue
            model = self.db.get_model(task.model_id)
            if model is not None and model.state == db.PRESENT and Path(model.path).is_file():
                return model
        return None

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
        if not Path(task.dest).is_file():
            raise FileNotFoundError(f"{task.dest} is not there any more")

        # The file is the library's model now, and the model is what moves: the library
        # carries the change over to every download that produced it — this card included —
        # so the explanation on it stops naming a guess a person has overruled by hand.
        model = self.library.model_for_task(task)
        stop = threading.Event()
        self._moves[task_id] = stop
        try:
            result = await asyncio.to_thread(
                self.library.move, model.id, folder, self._move_progress(task_id), stop
            )
        finally:
            # Whatever happened, nothing is copying any more, and every page watching needs
            # to hear that as much as it heard the progress.
            self._moves.pop(task_id, None)
            self.emit({"type": "moved", "id": task_id})

        self._emit_task(task_id)
        return result

    async def move_model(self, model_id: int, folder: Path) -> relocate.Move:
        """Move a model of the library, which may not be anybody's download."""
        if model_id in self._model_moves:
            raise ValueError("this model is already being moved")
        stop = threading.Event()
        self._model_moves[model_id] = stop
        try:
            return await asyncio.to_thread(
                self.library.move, model_id, folder, self._move_progress(model_id, "model_id"), stop
            )
        finally:
            self._model_moves.pop(model_id, None)
            self.emit({"type": "moved", "model_id": model_id})

    async def separate_model(self, model_id: int) -> dict[str, Any]:
        """Give one name of a shared file a copy of its own: a copy of the whole file on the
        same drive, with its progress and a Stop, as a move across drives has."""
        if model_id in self._model_moves:
            raise ValueError("this model is being moved or copied already")
        stop = threading.Event()
        self._model_moves[model_id] = stop
        try:
            return await asyncio.to_thread(
                self.library.separate, model_id,
                self._move_progress(model_id, "model_id", verb="Copying"), stop,
            )
        finally:
            self._model_moves.pop(model_id, None)
            self.emit({"type": "moved", "model_id": model_id})

    def stop_model_move(self, model_id: int) -> bool:
        stop = self._model_moves.get(model_id)
        if stop is None:
            return False
        stop.set()
        return True

    def stop_move(self, task_id: int) -> bool:
        """Ask an in-flight move to give up. True if there was one to ask."""
        stop = self._moves.get(task_id)
        if stop is None:
            return False
        stop.set()
        return True

    def _move_progress(
        self, task_id: int, field: str = "id", verb: str | None = None
    ) -> relocate.Progress:
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
            event = {"type": "moving", field: task_id, "copied": copied, "total": total}
            if verb:
                event["verb"] = verb
            loop.call_soon_threadsafe(self.emit, event)

        return report

    async def rename(self, task_id: int, name: str) -> relocate.Move:
        """Give a finished download a different filename, sidecars and all.

        The other half of the correction `move` makes. A file lands under whatever the
        service happened to call it, and `pytorch_lora_weights.safetensors` — HuggingFace's
        own default — is four files that all have to change together, in a folder where
        several downloads may already be carrying that exact name.

        On a thread like a move, though for a different reason: a rename within a folder is
        instant, and it is the four `stat` calls that find the companions which have no
        business happening on the event loop while a download is running.
        """
        task = self.db.get(task_id)
        if task is None:
            raise LookupError("no such task")
        if task.state != db.DONE or not task.dest:
            raise ValueError("only a finished download can be renamed")
        if task_id in self._moves:
            raise ValueError("this file is being moved right now")
        if not Path(task.dest).is_file():
            raise FileNotFoundError(f"{task.dest} is not there any more")

        # Through the library, which renames the model and then every download of it: the
        # history has to say what the file is called now, and `original_filename` keeps
        # what it arrived as.
        model = self.library.model_for_task(task)
        result = await asyncio.to_thread(self.library.rename, model.id, name)
        self._emit_task(task_id)
        return result

    async def set_note(self, task_id: int, note: str) -> tuple[str | None, bool]:
        """Write your own note about a download, or clear it.

        Returns what the note now says and whether a record had to be written to hold it.

        The note goes on the disk first and into the queue second, and that order is the
        whole design: this row is deleted by `Clear finished` and the record is not, so the
        record is where the note actually lives and the column is a copy for the card to
        draw and the filter box to search. The same relationship `downloaded` has with the
        `.part.json` beside the file.

        Until the file lands there is no record to be the note's home, and refusing one
        until then would ask for it at the one moment it is hardest to write: what you know
        about a model is in your head while you are queueing it, not an hour later when the
        bytes stop. So a note on a download that has not finished waits in the row, and the
        write that lands the file is the write that puts it in the record. It is the row's
        for that stretch, which means `Clear finished` and `Remove` take it — but nothing
        else exists yet for it to be taken from.
        """
        task = self.db.get(task_id)
        if task is None:
            raise LookupError("no such task")
        if task_id in self._moves:
            raise ValueError("this file is being moved right now")

        if task.state != db.DONE or not task.dest:
            written = note.strip() or None
            self.db.update(task_id, note=written)
            self._emit_task(task_id)
            return written, False

        if not Path(task.dest).is_file():
            raise FileNotFoundError(f"{task.dest} is not there any more")
        # The model is where the note lives; the library writes it into the record — and
        # writes the record first when sidecars were turned off, since everything else in a
        # record can be fetched again and this is the one field that cannot — then copies it
        # to every download of the model, this one included.
        model = self.library.model_for_task(task)
        written, created = await asyncio.to_thread(self.library.set_note, model.id, note)
        self._emit_task(task_id)
        return written, created

    async def delete_files(self, task_id: int) -> erase.Erased:
        """Delete what this download put on disk.

        Permanent, and therefore only ever reached from a question already answered — see
        the endpoint. A finished download stays in the history, marked deleted: what arrived
        and when is still true after the file has gone, and it is what "download it again"
        needs. One that never finished was never history, and its row goes with the files.

        A download still running is paused first and waited for. Pulling a `.part` out from
        under a transfer that is still writing to it is the one way this could damage
        something other than what it was pointed at.
        """
        task = self.db.get(task_id)
        if task is None:
            raise LookupError("no such task")
        if task_id in self._moves:
            raise ValueError("this file is being moved right now")
        if not task.dest:
            raise ValueError("this download never got as far as a file")

        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await running

        if task.state == db.DONE:
            model = self.library.model_for_task(task)
            result = await asyncio.to_thread(self.library.delete_files, model.id)
            self._emit_task(task_id)
            return result

        result = await asyncio.to_thread(
            erase.erase, Path(task.dest), **self.library.places()
        )
        self.db.delete(task_id)
        self.emit({"type": "removed", "id": task_id})
        return result

    def cancel(self, task_id: int) -> None:
        """Take a download off the list.

        One that finished goes into the history rather than away: the list is for what is
        happening, the history for what happened. One that did not is cancelled, and its
        `.part` stays on disk for the cleanup view to find.
        """
        task = self.db.get(task_id)
        if task is not None and task.state == db.DONE:
            self.db.update(task_id, archived=True)
            self._emit_task(task_id)
            return
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
        self.db.delete(task_id)
        self.emit({"type": "removed", "id": task_id})

    def remove_from_history(self, task_id: int) -> None:
        """Drop a finished download from the history. The model, if any, is untouched."""
        task = self.db.get(task_id)
        if task is None:
            raise LookupError("no such download")
        if task.state != db.DONE:
            raise ValueError("only a finished download is history")
        self.db.delete(task_id)
        self.emit({"type": "removed", "id": task_id})

    def redownload(self, task_id: int) -> Task:
        """Queue a finished download again — the "Get again" of a model that was deleted
        or went missing. Everything needed was kept: which file, from where, and where it
        was when it was last seen, which is where it goes back to."""
        old = self.db.get(task_id)
        if old is None:
            raise LookupError("no such download")
        if old.state != db.DONE:
            raise ValueError("only a finished download can be fetched again")
        model = self.db.get_model(old.model_id) if old.model_id else None
        if model is not None and model.state == db.PRESENT and Path(model.path).is_file():
            raise ValueError(f"it is still in your library at {model.path}")
        dest = model.path if model is not None else old.dest
        task = self.db.add(
            # The model it puts back, when the library still remembers it.
            model_id=model.id if model is not None else None,
            state=db.PENDING if self.settings.auto_start else db.PAUSED,
            source=old.source,
            label=old.label,
            provider=old.provider,
            identity=old.identity,
            filename=Path(dest).name,
            size=old.size,
            sha256=old.sha256,
            dest=dest,
            category=old.category,
            confidence=old.confidence,
            reason=old.reason,
            base_model=old.base_model,
            meta=old.meta,
            note=old.note,
            original_filename=old.original_filename,
            position_mode=self.settings.queue_position,
        )
        if task is None:
            raise ValueError("this file is already in the queue")
        self.emit({"type": "task", "task": task.to_json()})
        self._wake.set()
        return task

    def pause_all(self) -> int:
        paused = 0
        for task in self.db.list([db.PENDING, db.RUNNING]):
            self.pause(task.id)
            paused += 1
        return paused

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
        """Take every finished download off the list. Into the history, not away."""
        archived = self.db.archive_finished()
        self.emit({"type": "reload"})
        return archived

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
        await self._write_sidecar(path, task, identity)
        # Deliberately not tied to the records setting the write above answers to. The
        # preview beside the model is read by the model managers, and someone who turned our
        # JSON records off did not thereby ask their model manager to stop showing pictures.
        if self.settings.fetch_previews and self.settings.write_compat_files:
            await self._write_preview(path, task)
        self._landed(task, path)

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
        await self._write_sidecar(path, task, identity)
        if self.settings.fetch_previews and self.settings.write_compat_files:
            await self._write_preview(path, task)
        self._landed(task, path)

    def _landed(self, task: Task, path: Path) -> None:
        """A download is done: into the library with it, then tell the page."""
        try:
            self.library.adopt_download(task, path)
            self.library.inspect_soon()
        except Exception as exc:  # noqa: BLE001 - the file is safely on disk either way
            self.emit({"type": "toast", "level": "error",
                       "message": f"{path.name} downloaded, but the library could not take it in: {exc}"})
        self._emit_task(task.id)

    def _record_for(
        self, path: Path, task: Task, identity: FileIdentity
    ) -> tuple[Verdict, sidecar.Record]:
        """The record a task would write, without writing it.

        Split out because a note needs one too: a file downloaded with sidecars off has
        nowhere to keep a note, and what it needs then is the record it would have had, not
        a stub holding one field.
        """
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
            # Whatever was written about this download before it had a record to be written
            # in. Empty for a file annotated the usual way, which annotates the record
            # itself a moment later.
            note=task.note,
        )
        return verdict, record

    async def _write_sidecar(self, path: Path, task: Task, identity: FileIdentity) -> None:
        """The record beside a finished file — and, with records turned off, the one a note
        needs anyway.

        Records off is a choice about clutter, not a choice to lose the one field no service
        can supply and nothing can fetch again: a download annotated while it was still
        running would otherwise land and drop the note on the floor. Only the record, though
        — the compatibility files and the trigger `.txt` are a separate choice, and a note
        is not the moment to overrule it. The same reasoning as annotating a file that was
        downloaded with them off.
        """
        # The row rather than the snapshot this download started from: a note written while
        # the bytes were moving is in the queue and nowhere else, and this is the write that
        # gives it a home.
        task = self.db.get(task.id) or task
        # A missing model downloaded again, into another folder: its record comes along
        # first, so that what is written now adds to it — the note included — rather than
        # starting a record of its own beside the old one.
        back = self.db.get_model(task.model_id) if task.model_id else None
        if back is not None and back.state == db.MISSING:
            self.library.carry_record(back, path)
        records = self.settings.write_sidecars
        if not records and not task.note:
            return

        verdict, record = self._record_for(path, task, identity)
        if not record.note:
            # The same file fetched again — after it was deleted, or went missing — lands on
            # a model the library already knows, and what was written about it then is still
            # true. A pasted link knows nothing of that note; the record and the library do.
            record.note = self._known_note(path)
        sidecar.write(
            path,
            verdict,
            record,
            compat=records and self.settings.write_compat_files,
            triggers=records and self.settings.write_trigger_txt,
            **self.library.places(),
        )

    def _known_note(self, path: Path) -> str | None:
        places = self.library.places()
        found = sidecar.find_record(path, **places)
        if found is not None:
            data = sidecar.read_record(found) or {}
            if data.get("note"):
                return str(data["note"])
        model = self.db.model_at(path)
        return model.note if model is not None else None

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
    return Category(value)


def _family(filename: str) -> tuple[str, str]:
    """Files that are one model in different sizes: the name without its variant, and the
    container, since a GGUF and a safetensors of one name are not read the same way."""
    stem, _, suffix = (filename or "").lower().rpartition(".")
    if not stem:
        stem, suffix = suffix, ""
    shard = shard_of(filename or "")
    if shard is not None:
        stem = shard[0].lower().rpartition(".")[0]
    return suffix, without_variant(stem)


def _model(item: Item) -> bool:
    """Whether a file of a link is a model, rather than something that comes with one. A
    direct link names no file until it is fetched, and is taken to be one."""
    return not item.filename or is_model_file(item.filename, item.size)


def _doc(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".md") or lowered.split(".")[0] in ("readme", "license", "notice")


def _preselect(resolution: Resolution, have: list[Any]) -> tuple[list[bool], bool]:
    """Which files of a link are ticked when the question is put, and whether they keep the
    repository's folders.

    One file: that file. A Civitai version: the file its own download button gives. A
    repository holding one model — its weights, in shards or not — the weights; and when
    configs come with them, all of it, as the repository lays it out, since a transformers
    model is a folder that only works whole. A repository of several models — a list of
    quantisations, a pack of files for different nodes — nothing: which of them is wanted
    is the question being asked. Nothing already in the library is ticked.
    """
    items = resolution.items
    fresh = [have[i] is None for i in range(len(items))]
    if len(items) == 1:
        return [fresh[0]], False
    if resolution.provider.name == "civitai":
        if any(item.primary for item in items):
            return [item.primary and fresh[i] for i, item in enumerate(items)], False
        first = next((i for i, item in enumerate(items) if _model(item) and fresh[i]), None)
        return [i == first for i in range(len(items))], False
    models = [item for item in items if _model(item)]
    sets = {(shard_of(item.filename) or (item.filename,))[0].lower() for item in models}
    if len(sets) != 1:
        return [False] * len(items), False
    configs = [
        item for item in items
        if not _model(item) and not _doc(item.filename)
        and not item.filename.lower().endswith(UNWANTED_SUFFIXES)
    ]
    if configs:
        checked = [
            fresh[i] and (_model(item) or item in configs or _doc(item.filename))
            for i, item in enumerate(items)
        ]
        return checked, True
    nested = any("/" in (item.relative or "") for item in models)
    return [fresh[i] and _model(item) for i, item in enumerate(items)], nested


ProgressCallback = Callable[[ProgressSnapshot], None]
