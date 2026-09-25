"""The library: every model the app knows about, whether or not it downloaded it.

The queue answers "what is downloading"; this answers "what do I have, and where is it".
It is kept as a table, and the table is a cache of the disk: rebuilt by walking the
library's folders, which is cheap enough to do every time the window comes back to the
front. The walk is where the three situations the queue could never express are noticed:

  A model that is gone.      Deleted or moved outside the app. It stays in the library,
                             grey, where it was last seen — until it is found again,
                             pointed at, or forgotten.
  A model that moved.        The same name and the same size turning up somewhere else,
                             while exactly one model of that name and size went missing,
                             is the same file dragged in Explorer. It is relinked on its
                             own; what it left behind in its old folder is offered back.
  A model from elsewhere.    Downloaded by a browser, copied off another machine, put there
                             by another tool. Everything that can be read off the file and
                             its neighbours is, and a hash lookup on Civitai can fill in
                             the rest when someone asks for it.

What cannot be rebuilt by a walk is the note, and it is not kept here: it lives in the
model's `.json` record exactly as it always has, and this table carries a copy.

Nothing in this module talks to the network unless a person pressed something, with one
exception that Settings can switch off: the check for newer versions made once each time the
app starts. Identifying a model is only ever a button.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus

import httpx

from ..core.diskinfo import free_bytes
from ..core.errors import SfdError
from ..core.types import FileIdentity
from ..library import details, erase, fingerprint, folders, links, relocate, sidecar, versions
from ..library import lookup as online
from ..library import scan as scanning
from ..library.categories import ALIASES, Category
from ..library.classify import Verdict
from ..library.files import is_model_file, shard_of
from ..providers.registry import source_url
from ..settings import Settings
from . import db
from .db import Database, Model, Task, path_key

HASH_BLOCK = 8 * 1024 * 1024
PROGRESS_INTERVAL = 0.4
# Above this many changes one "reload" is cheaper for the page than a message per model.
BULK = 150
# How long a *Locate…* answer that needs confirming is kept for the confirmation.
PENDING_FOR = 600.0
CIVITAI_API = "https://{host}/api/v1"
UPDATE_CONCURRENCY = 4
# What the Hub's error codes mean for a file that was there once: taken down, or moved out of
# reach — which, for someone who downloaded it, comes to the same thing.
HUB_GONE = {
    "EntryNotFound": "no longer in its repository on HuggingFace",
    "RepoNotFound": "its repository is gone from HuggingFace, or private now",
    "RevisionNotFound": "the branch it came from is gone from its repository",
}
# Below this, a "copy" is a tokenizer or a config that happens to share a size with another,
# and nobody is short of disk over it.
DUPLICATE_MIN = 1024 * 1024
# Above this, identifying a model asks the services for anything like it before reading the
# whole file: below it, reading the file is quicker than asking.
QUICK_ABOVE = 1024 ** 3
# Slack kept between a copy and a completely full volume, as the queue keeps it.
SPACE_HEADROOM = 64 * 1024**2

Emit = Callable[[dict[str, Any]], None]


class Busy(ValueError):
    """Another operation already has hold of this model."""


class Library:
    def __init__(self, settings: Settings, database: Database, emit: Emit) -> None:
        self.settings = settings
        self.db = database
        self.emit = emit
        self._sync_lock = threading.Lock()
        self._busy: set[int] = set()
        self._busy_lock = threading.Lock()
        self.last_scan: scanning.Scan | None = None
        self.synced_at: float | None = None
        self._sync_future: asyncio.Future[dict[str, Any] | None] | None = None
        self._sync_again = False
        self._inspecting = False
        self._inspect_again = False
        # Hashing is a queue of its own: one file at a time, because two reads of two big
        # files on one disk are slower than the same two reads one after the other.
        self._jobs: list[dict[str, Any]] = []
        self.current_job: dict[str, Any] | None = None
        self._job_wake: asyncio.Event | None = None
        self._job_task: asyncio.Task[None] | None = None
        self._job_stop = threading.Event()
        self._pending: dict[str, tuple[int, str, float]] = {}
        self._cleanup: dict[int, dict[str, Any]] = {}
        self._client: httpx.AsyncClient | None = None
        # What the Hub said about its repositories, for a while: identifying a library asks
        # after the same ones again and again.
        self._hub_cache: dict[str, tuple[float, Any]] = {}
        # The check for newer versions under way, if one is — how far it has got — and
        # whether it has been asked to stop. One at a time: the one at start and a Check now
        # pressed during it would otherwise ask every service everything twice.
        self._update_run: dict[str, Any] | None = None
        self._update_stop = False
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Called on the event loop, once the server is up."""
        self._closing = False
        self._loop = asyncio.get_running_loop()
        self._job_wake = asyncio.Event()
        self._job_task = asyncio.create_task(self._job_loop(), name="library-jobs")

    async def stop(self) -> None:
        self._closing = True
        self._job_stop.set()
        if self._job_wake is not None:
            self._job_wake.set()
        if self._job_task is not None:
            self._job_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._job_task
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=15.0),
                follow_redirects=True,
                headers={"User-Agent": "model-dl"},
            )
        return self._client

    # --- where things are ---------------------------------------------------

    @property
    def roots(self) -> list[Path]:
        return self.settings.roots

    def places(self) -> dict[str, Any]:
        """The three things every sidecar calculation needs, as keyword arguments."""
        return {
            "sidecar_dir": self.settings.sidecar_path,
            "library_root": self.settings.library_path,
            "roots": tuple(self.roots),
        }

    def placement(self, path: str) -> tuple[int | None, str]:
        """Which folder of the library a path is in, and the folder under it.

        (None, "") for a path outside every one of them — a model moved to another drive,
        or one whose folder was taken off the list.
        """
        key = path_key(path)
        best: tuple[int, int] | None = None
        for index, root in enumerate(self.roots):
            base = path_key(root).rstrip("\\/")
            if key.startswith(base + os.sep) and (best is None or len(base) > best[1]):
                best = (index, len(base))
        if best is None:
            return None, ""
        root = self.roots[best[0]]
        try:
            folder = Path(path).parent.relative_to(root)
        except ValueError:
            relative = key[best[1] + 1:]
            folder = Path(relative).parent
        text = folder.as_posix()
        return best[0], "" if text == "." else text

    def summary(self, model: Model) -> dict[str, Any]:
        data = model.to_json()
        data["root"], data["relative"] = self.placement(model.path)
        data["busy"] = model.id in self._busy
        # Whether there is somewhere to download it from again, should it go.
        data["source"] = bool((model.identity or {}).get("provider"))
        return data

    def roots_json(self) -> list[dict[str, Any]]:
        found = []
        missing = set(self.last_scan.missing_roots) if self.last_scan else set()
        for index, root in enumerate(self.roots):
            exists = root.is_dir()
            found.append({
                "index": index,
                "path": str(root),
                "name": root.name or str(root),
                "primary": index == 0,
                # Without a library root, the first folder is where downloads land anyway.
                "downloads": index == 0 and not self.settings.library_root,
                "exists": exists and index not in missing,
                "free": free_bytes(root) if exists else None,
                "key": path_key(root),
            })
        return found

    def folders_json(self) -> list[dict[str, Any]]:
        """Every folder the tree shows: the ones on disk, and the ones a missing model was
        last seen in, which may be gone themselves."""
        listed: dict[tuple[int, str], dict[str, Any]] = {}
        if self.last_scan is not None:
            for root, relative, count in self.last_scan.folders:
                listed[(root, relative)] = {"root": root, "relative": relative, "exists": True}
        for model in self.db.list_models():
            root, relative = self.placement(model.path)
            if root is None or not relative:
                continue
            parts = relative.split("/")
            for depth in range(1, len(parts) + 1):
                folder = "/".join(parts[:depth])
                if (root, folder) not in listed:
                    exists = (self.roots[root] / folder).is_dir()
                    listed[(root, folder)] = {"root": root, "relative": folder, "exists": exists}
        return sorted(listed.values(), key=lambda f: (f["root"], f["relative"].lower()))

    def snapshot(self) -> dict[str, Any]:
        return {
            "roots": self.roots_json(),
            "folders": self.folders_json(),
            "models": [self.summary(m) for m in self.db.list_models()],
            "synced_at": self.synced_at,
            "sep": os.sep,
            "hidden": list(self.settings.exclude_dirs),
            "job": self.job_state(),
            "update_check": self.update_state(),
        }

    # --- keeping up with the disk -------------------------------------------

    async def refresh(self) -> dict[str, Any] | None:
        """Walk the library now, and answer once the disk has been read.

        A walk already under way is not started twice. Whoever asks during one is answered
        by the walk after it — the one that ran knowing about their request — since what
        they may have just done, the running walk has perhaps already passed.
        """
        running = self._sync_future
        if running is not None and not running.done():
            self._sync_again = True
            return await asyncio.shield(running)
        future: asyncio.Future[dict[str, Any] | None] = asyncio.get_running_loop().create_future()
        self._sync_future = future
        report: dict[str, Any] | None = None
        try:
            while True:
                self._sync_again = False
                report = await asyncio.to_thread(self.sync)
                if not self._sync_again:
                    break
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                # Nobody may be waiting for it; that is not a second error worth a warning.
                future.exception()
            raise
        else:
            future.set_result(report)
        self.inspect_soon()
        return report

    def inspect_soon(self) -> None:
        """Read the headers of whatever is new, in the background.

        Safe to call from a worker thread: the reading is always started from the loop,
        since that is where the flag guarding against two readings at once lives.
        """
        loop = self._loop
        if loop is None or self._closing:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not loop:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.inspect_soon)
            return
        if self._inspecting:
            self._inspect_again = True
            return
        self._inspecting = True

        async def run() -> None:
            try:
                while True:
                    self._inspect_again = False
                    await asyncio.to_thread(self.inspect_pending)
                    if not self._inspect_again:
                        break
            finally:
                self._inspecting = False

        loop.create_task(run(), name="library-inspect")

    def sync(self) -> dict[str, Any]:
        with self._sync_lock:
            return self._sync()

    def _sync(self) -> dict[str, Any]:
        roots = self.roots
        found = scanning.walk(roots, self.settings.exclude_dirs)
        self.last_scan = found
        now = time.time()
        previous = self.synced_at or now
        places = self.places()
        hidden = [path_key(p) for p in self.settings.exclude_dirs if str(p).strip()]
        root_keys = [path_key(r).rstrip("\\/") for r in roots]
        # A folder that is not there right now — a drive unplugged, a share not yet mounted
        # — says nothing about what is in it. Its models are left exactly as they were
        # rather than marked missing, relinked elsewhere, or dropped as gone.
        away = [root_keys[i] for i in found.missing_roots if i < len(root_keys)]

        def under_away(key: str) -> bool:
            return any(key.startswith(base + os.sep) for base in away)

        changed: set[int] = set()
        removed: set[int] = set()
        relinked: list[dict[str, Any]] = []
        linked_tasks = False
        models = {m.key: m for m in self.db.list_models()}
        missing_now: list[Model] = []

        # 1. What was already known: still there, gone, or changed.
        for key, model in list(models.items()):
            if self.is_busy(model.id) or under_away(key):
                continue
            entry = found.files.get(key) or _stat_outside(model)
            if entry is not None:
                outside = _outside(key, root_keys, hidden)
                if outside and model.origin == db.FOUND and not self._worth_keeping(model):
                    # Found in a folder that is no longer part of the library, and nothing
                    # of anyone's is attached to it. It was only ever a listing.
                    self.db.delete_model(model.id)
                    removed.add(model.id)
                    del models[key]
                    continue
                updates = _present_updates(model, entry)
                if updates:
                    self.db.update_model(model.id, **updates)
                    changed.add(model.id)
            else:
                if model.state != db.MISSING:
                    self.db.update_model(
                        model.id, state=db.MISSING, missing_since=now, last_seen=previous
                    )
                    changed.add(model.id)
                refreshed = self.db.get_model(model.id)
                if refreshed is not None:
                    missing_now.append(refreshed)

        # 2. Finished downloads the library has not met yet: every download from before the
        # library existed, and any whose file was already there when it landed.
        for task in self.db.list([db.DONE]):
            if task.model_id is not None or task.fate or not task.dest:
                continue
            dest = Path(os.path.abspath(task.dest))
            key = path_key(dest)
            model = models.get(key)
            if model is None and under_away(key):
                continue
            if model is None:
                model = self._model_from_task(task, dest, found.files.get(key), now)
                if model is None:
                    continue
                models[key] = model
                if model.state == db.MISSING:
                    missing_now.append(model)
            elif model.origin == db.FOUND:
                self.db.update_model(model.id, **_download_facts(task, model))
            self.db.update(task.id, model_id=model.id)
            changed.add(model.id)
            linked_tasks = True

        # 3. Files nobody has seen before — unless one of them is a model that went missing,
        # turning up under the same name and size somewhere else.
        by_name: dict[tuple[str, int], list[Model]] = {}
        for model in missing_now:
            if model.size is not None and not self.is_busy(model.id):
                by_name.setdefault((model.filename.lower(), model.size), []).append(model)
        fresh = [entry for key, entry in found.files.items() if key not in models]
        fresh_names: dict[tuple[str, int], int] = {}
        for entry in fresh:
            name_size = (entry.path.name.lower(), entry.size)
            fresh_names[name_size] = fresh_names.get(name_size, 0) + 1
        for entry in fresh:
            name_size = (entry.path.name.lower(), entry.size)
            lost = by_name.get(name_size, [])
            if len(lost) == 1 and fresh_names[name_size] == 1 and not self.is_busy(lost[0].id):
                model = lost[0]
                self._relocate_record(model, entry.path, places)
                self._settle(model, entry.path, entry.size, entry.mtime, entry.parts, places)
                relinked.append({"id": model.id, "from": model.path, "to": str(entry.path)})
                by_name.pop(name_size, None)
                models[path_key(entry.path)] = model
                changed.add(model.id)
                continue
            added = self.db.add_model(
                path=str(entry.path),
                filename=entry.path.name,
                size=entry.size,
                mtime=entry.mtime,
                parts=[str(p) for p in entry.parts],
                state=db.PRESENT,
                origin=db.FOUND,
                last_seen=now,
            )
            if added is not None:
                models[added.key] = added
                changed.add(added.id)

        # 4. A found model that disappeared and carries nothing of anyone's goes quietly:
        # keeping every file that ever passed through a folder, grey, forever, is how a
        # library fills up with rubbish.
        for model in missing_now:
            current = self.db.get_model(model.id)
            if (
                current is not None
                and current.state == db.MISSING
                and current.origin == db.FOUND
                and not self._worth_keeping(current)
            ):
                self.db.delete_model(current.id)
                removed.add(current.id)
                changed.discard(current.id)

        # 5. Notes, from where they actually live — and which names are one file, read fresh
        # every time: making a hard link changes no date on the file, so nothing above would
        # have noticed one.
        for model in self.db.list_models():
            if model.state != db.PRESENT or self.is_busy(model.id) or under_away(model.key):
                continue
            record = sidecar.find_record(Path(model.path), **places)
            note = model.note
            if record is not None:
                data = sidecar.read_record(record)
                if data is not None and "note" in data:
                    note = data.get("note") or None
            where = str(record) if record is not None else None
            file_id, count = links.identity(model.path) or (None, None)
            if (where, note, file_id, count) != (model.record, model.note, model.file_id, model.links):
                self.db.update_model(model.id, record=where, note=note, file_id=file_id, links=count)
                changed.add(model.id)

        self.synced_at = now
        self._announce(changed, removed)
        self.emit({
            "type": "folders",
            "roots": self.roots_json(),
            "folders": self.folders_json(),
            "synced_at": now,
        })
        if linked_tasks:
            self.emit({"type": "reload"})
        if relinked:
            count = len(relinked)
            self.emit({
                "type": "toast",
                "level": "info",
                "message": f"Found {count} model{'s' if count > 1 else ''} moved outside "
                           f"ModelDL and linked {'them' if count > 1 else 'it'} to the new "
                           f"location",
                "models": [r["id"] for r in relinked],
            })
        return {
            "at": now,
            "elapsed": found.elapsed,
            "models": len(models) - len(removed),
            "changed": len(changed),
            "removed": len(removed),
            "relinked": relinked,
            "missing_roots": found.missing_roots,
        }

    def remap_records(self, old_library_root: Path | None, old_roots: Iterable[Path]) -> int:
        """Move collected records to where the folders as they are now expect them.

        Making another folder the main one changes which folder is mirrored at the top of
        `sidecar_dir` and which under `@roots/`. Left where they were, the old main folder's
        records would sit exactly where the new main folder's models look for theirs. Each
        model's record goes from its old place to its new one; a place still taken is tried
        again once the record in it has moved on, so two folders can trade places.
        """
        sidecar_dir = self.settings.sidecar_path
        if sidecar_dir is None:
            return 0
        old_roots = tuple(old_roots)
        places = self.places()
        pending = []
        for model in self.db.list_models():
            path = Path(model.path)
            old = sidecar.record_path(path, sidecar_dir, old_library_root, old_roots)
            new = sidecar.record_path(path, **places)
            if path_key(old) != path_key(new):
                pending.append((model, old, new))
        moved = 0
        while pending:
            waiting = []
            for model, old, new in pending:
                if not old.is_file():
                    continue
                if new.exists():
                    waiting.append((model, old, new))
                    continue
                try:
                    new.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(old, new)
                except OSError:
                    continue
                self.db.update_model(model.id, record=str(new))
                moved += 1
            if len(waiting) == len(pending):
                break
            pending = waiting
        return moved

    def _worth_keeping(self, model: Model) -> bool:
        """Whether a model carries something that a walk could not recreate."""
        return bool(
            model.origin == db.DOWNLOADED
            or model.note
            or model.identified
            or (model.record and Path(model.record).is_file())
        )

    def _model_from_task(
        self, task: Task, dest: Path, entry: scanning.Found | None, now: float
    ) -> Model | None:
        if entry is None and dest.is_file():
            with contextlib.suppress(OSError):
                stat = dest.stat()
                entry = scanning.Found(dest, stat.st_size, stat.st_mtime)
        values = _download_facts(task, None)
        values["last_seen"] = now
        if entry is not None:
            values.update(size=entry.size, mtime=entry.mtime, state=db.PRESENT,
                          parts=[str(p) for p in entry.parts])
        else:
            values.update(size=task.size, state=db.MISSING, missing_since=now,
                          last_seen=task.finished_at)
        return self.db.add_model(path=str(dest), filename=dest.name, **values)

    def carry_record(self, model: Model, new: Path) -> None:
        """Take a model's collected record to where a file of it has just arrived."""
        self._relocate_record(model, new, self.places())

    def _relocate_record(self, model: Model, new: Path, places: dict[str, Any]) -> None:
        """Take a model's record along to its new place, when the record is one of ours.

        Only a record collected in `sidecar_dir`: that directory is this app's, and a record
        left mirroring a folder the model is no longer in would be found by nothing. Files in
        the library's own folders are the person's to move, and are offered, not moved.
        """
        sidecar_dir = places.get("sidecar_dir")
        old = Path(model.path)
        record = sidecar.find_record(old, **places)
        if sidecar_dir is None or record is None:
            return
        try:
            record.relative_to(sidecar_dir)
        except ValueError:
            return
        target = sidecar.record_path(new, **places)
        if target.exists() or path_key(target) == path_key(record):
            return
        with contextlib.suppress(OSError):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(record, target)
            if new.name != old.name:
                sidecar.retitle(target, new.name)

    def _settle(
        self,
        model: Model,
        path: Path,
        size: int,
        mtime: float,
        parts: Iterable[Path],
        places: dict[str, Any],
    ) -> None:
        """Point a model at where it is now, and note what it left where it was."""
        old = Path(model.path)
        moved = path_key(old) != path_key(path)
        left = [
            {"from": str(a), "to": str(b)}
            for a, b in relocate.strays(old, path, **places)
        ] if moved else []
        record = sidecar.find_record(path, **places)
        self.db.update_model(
            model.id,
            path=str(path),
            filename=path.name,
            size=size,
            mtime=mtime,
            parts=[str(p) for p in parts],
            state=db.PRESENT,
            missing_since=None,
            left_behind=left,
            record=str(record) if record is not None else model.record,
            sniffed_mtime=None if model.sniffed_mtime != mtime else model.sniffed_mtime,
            # Another file, as far as anyone can tell until it is read. The fingerprint is
            # what decides what may be deleted as a copy, so it is taken again rather than
            # trusted across a move.
            sampled_mtime=None if moved else model.sampled_mtime,
        )
        self.db.retarget_tasks(model.id, dest=str(path), filename=path.name)

    def _announce(self, changed: Iterable[int], removed: Iterable[int]) -> None:
        changed, removed = set(changed), set(removed)
        if len(changed) + len(removed) > BULK:
            self.emit({"type": "library"})
            return
        if changed:
            summaries = [self.summary(m) for m in map(self.db.get_model, sorted(changed)) if m]
            if summaries:
                self.emit({"type": "models", "models": summaries})
        for model_id in removed:
            self.emit({"type": "model_removed", "id": model_id})

    def announce(self, model_id: int, tasks: bool = True) -> None:
        """Tell every open page about one model, and about the downloads that made it."""
        model = self.db.get_model(model_id)
        if model is None:
            self.emit({"type": "model_removed", "id": model_id})
        else:
            self.emit({"type": "models", "models": [self.summary(model)]})
        if tasks:
            for task in self.db.tasks_for_model(model_id):
                self.emit({"type": "task", "task": task.to_json()})

    # --- reading the files --------------------------------------------------

    def inspect_pending(self) -> int:
        """Read the header and the neighbours of every model whose file changed since the
        last reading, and take its fingerprint. Once per version of a file: the answer does
        not change until the file does."""
        batch: list[int] = []
        last = time.monotonic()
        done = 0
        for model in self.db.list_models():
            if self._closing:
                break
            if model.state != db.PRESENT:
                continue
            header_due = not _same_time(model.sniffed_mtime, model.mtime)
            # A split model is a set of files, and nobody keeps a set twice by accident.
            sample_due = not model.parts and not _same_time(model.sampled_mtime, model.mtime)
            if not (header_due or sample_due) or self.is_busy(model.id):
                continue
            updates: dict[str, Any] | None = {}
            if header_due:
                try:
                    updates = self._inspect(model)
                except Exception as exc:  # noqa: BLE001 - one odd file must not stop the rest
                    # Someone else's sidecar, in whatever shape it is in, or a file that went
                    # away halfway through reading it. Said on the model, and not read again
                    # until the file changes.
                    updates = {"header": {"format": None, "error": f"could not be read: {exc}"},
                               "sniffed_mtime": model.mtime}
                if updates is None:
                    continue
            if sample_due:
                updates.update(self._sample(model))
            self.db.update_model(model.id, **updates)
            done += 1
            # A fingerprint changes nothing the page shows; a header does.
            if header_due:
                batch.append(model.id)
            if batch and (len(batch) >= 25 or time.monotonic() - last > 0.7):
                self._announce(batch, ())
                batch, last = [], time.monotonic()
        if batch:
            self._announce(batch, ())
        return done

    def _sample(self, model: Model) -> dict[str, Any]:
        """The fingerprint and the AutoV1 hash of the file as it is now. A file that cannot
        be read gets neither, and is not read again until it changes."""
        try:
            taken = fingerprint.sample(Path(model.path))
        except OSError:
            return {"fingerprint": None, "autov1": None, "sampled_mtime": model.mtime}
        return {"fingerprint": taken.fingerprint, "autov1": taken.autov1,
                "sampled_mtime": model.mtime}

    def _inspect(self, model: Model) -> dict[str, Any] | None:
        path = Path(model.path)
        if not path.is_file():
            return None
        header, sniffed = details.inspect(path)
        extras, local = details.neighbours(path, model.size)
        verified = (model.lookup or {}).get("verified")
        if verified and verified.get("mtime") != model.mtime:
            lookup = dict(model.lookup)
            lookup.pop("verified", None)
        else:
            lookup = None
        updates: dict[str, Any] = {"header": header, "extras": extras, "sniffed_mtime": model.mtime}
        if lookup is not None:
            updates["lookup"] = lookup

        meta = model.meta
        if model.origin == db.FOUND and not model.meta.get("version_id") and local:
            meta = local
            updates["meta"] = local
            updates["provider"] = "civitai"
            if local.get("version_id") and local.get("file_id"):
                updates["identity"] = {
                    "provider": "civitai",
                    "ref": {"version_id": local["version_id"], "file_id": local["file_id"]},
                }
        if model.origin == db.FOUND:
            verdict = details.judge(path, sniffed, header, meta, extras, self.roots)
            updates.update(
                category=verdict.category.value,
                confidence=verdict.confidence,
                reason=verdict.reason,
                base_model=verdict.base_model,
            )
        else:
            kind, _folder, _below = details.folder_kind(path, self.roots)
            own = Category(model.category) if model.category in Category._value2member_map_ else None
            if kind is not None and own is not None and kind is not own \
                    and not details.interchangeable(own, kind):
                header["folder_says"] = kind.value
        return updates

    # --- a download landing -------------------------------------------------

    def adopt_download(self, task: Task, path: Path) -> Model | None:
        """Put a file that has just finished downloading into the library."""
        task = self.db.get(task.id) or task
        try:
            stat = path.stat()
        except OSError:
            return None
        values = _download_facts(task, None)
        record = sidecar.find_record(path, **self.places())
        values.update(
            size=stat.st_size,
            mtime=stat.st_mtime,
            state=db.PRESENT,
            missing_since=None,
            sniffed_mtime=None,
            sampled_mtime=None,
            record=str(record) if record is not None else None,
            last_seen=time.time(),
        )
        existing = self.db.model_at(path)
        # A model that went missing, downloaded again — wherever it was put this time, it is
        # that model back, with its history and its note, not a second model beside it.
        back = self.db.get_model(task.model_id) if task.model_id else None
        if back is not None and back.state == db.MISSING and (existing is None or existing.id != back.id):
            with contextlib.suppress(Busy):
                with self.working_on(back.id):
                    places = self.places()
                    if existing is not None:
                        # A walk met the new file before the download could say whose it is.
                        self._absorb(back, existing)
                        self.emit({"type": "model_removed", "id": existing.id})
                    else:
                        self._relocate_record(back, path, places)
                        self.db.retarget_tasks(back.id, dest=str(path), filename=path.name)
                    existing = self.db.get_model(back.id)
        if existing is not None:
            if not task.sha256:
                values.pop("sha256", None)
                values.pop("hash_source", None)
            if not values.get("note"):
                values["note"] = existing.note
            # Whatever the last check said was about the file that was here before this
            # one; the next check speaks for this one.
            values["updates"] = {}
            self.db.update_model(existing.id, path=str(path), filename=path.name, **values)
            model_id = existing.id
        else:
            added = self.db.add_model(path=str(path), filename=path.name, **values)
            if added is None:
                return None
            model_id = added.id
        self.db.update(task.id, model_id=model_id)
        self.announce(model_id, tasks=False)
        return self.db.get_model(model_id)

    def model_for_task(self, task: Task) -> Model:
        """The model a finished download produced, meeting it first if need be."""
        if task.model_id is not None:
            model = self.db.get_model(task.model_id)
            if model is not None:
                return model
        dest = Path(os.path.abspath(task.dest))
        model = self.db.model_at(dest)
        if model is None:
            model = self._model_from_task(task, dest, None, time.time())
            if model is None:
                raise LookupError("this download has no file to speak of")
        elif model.origin == db.FOUND:
            self.db.update_model(model.id, **_download_facts(task, model))
            model = self.db.get_model(model.id) or model
        self.db.update(task.id, model_id=model.id)
        return model

    # --- operations on a model ----------------------------------------------

    def get(self, model_id: int) -> Model:
        model = self.db.get_model(model_id)
        if model is None:
            raise LookupError("no such model")
        return model

    @contextlib.contextmanager
    def working_on(self, *model_ids: int):
        with self._busy_lock:
            taken = [m for m in model_ids if m in self._busy]
            if taken:
                raise Busy("this model is busy with something else right now")
            self._busy.update(model_ids)
        try:
            yield
        finally:
            with self._busy_lock:
                self._busy.difference_update(model_ids)

    def is_busy(self, model_id: int) -> bool:
        with self._busy_lock:
            return model_id in self._busy

    def _present(self, model: Model) -> Path:
        path = Path(model.path)
        if model.state != db.PRESENT or not path.is_file():
            raise FileNotFoundError(f"{path} is not there any more")
        return path

    def _claimed(self, model_id: int, target: Path) -> None:
        """Refuse a destination the library knows as another model's — before a byte moves.

        Usually that is a model that went missing from there: its row still holds the path,
        and a file moved onto it would be taken for that model coming back, carrying its
        hash, its history and its note. The person has to say which it is.
        """
        other = self.db.model_at(target)
        if other is not None and other.id != model_id:
            if other.state == db.MISSING:
                raise FileExistsError(
                    f"the library remembers a missing model at {target} — find its file or "
                    f"forget it first"
                )
            raise FileExistsError(f"{target.parent} already holds a {target.name}")

    def rename(self, model_id: int, name: str) -> relocate.Move:
        # Read again once it is ours: a walk or another request may have changed it between
        # the page asking and this line running.
        with self.working_on(model_id):
            model = self.get(model_id)
            path = self._present(model)
            if model.parts:
                raise ValueError(
                    "a model split into several files cannot be renamed here — every part "
                    "and its index would have to change together"
                )
            self._claimed(model_id, path.with_name(relocate.intended_name(path, name)))
            places = self.places()
            result = relocate.rename(path, name, **places)
            if not result.unchanged:
                record = sidecar.find_record(result.path, **places)
                self.db.update_model(
                    model_id,
                    path=str(result.path),
                    filename=result.path.name,
                    record=str(record) if record is not None else None,
                )
                self.db.retarget_tasks(model_id, dest=str(result.path), filename=result.path.name)
        self.announce(model_id)
        return result

    def move(
        self,
        model_id: int,
        folder: Path,
        progress: relocate.Progress | None = None,
        stop: threading.Event | None = None,
    ) -> relocate.Move:
        with self.working_on(model_id):
            model = self.get(model_id)
            path = self._present(model)
            for piece in [Path(p) for p in model.parts] or [path]:
                self._claimed(model_id, folder / piece.name)
            places = self.places()
            if model.parts:
                result = self._move_parts(model, folder, progress, stop, places)
            else:
                result = relocate.move(path, folder, progress=progress, stop=stop, **places)
            if not result.unchanged:
                kind = ALIASES.get(folder.name.lower())
                record = sidecar.find_record(result.path, **places)
                reason = f"moved by hand into {folder.name}"
                updates: dict[str, Any] = {
                    "path": str(result.path),
                    "filename": result.path.name,
                    "record": str(record) if record is not None else None,
                    "left_behind": [],
                    "confidence": "high",
                    "reason": reason,
                }
                if kind is not None:
                    updates["category"] = kind.value
                if model.parts:
                    updates["parts"] = [str(folder / Path(p).name) for p in model.parts]
                self.db.update_model(model_id, **updates)
                self.db.retarget_tasks(
                    model_id,
                    dest=str(result.path),
                    filename=result.path.name,
                    category=kind.value if kind is not None else model.category,
                    confidence="high",
                    reason=reason,
                    disagreement=None,
                )
        self.announce(model_id)
        return result

    def _move_parts(
        self,
        model: Model,
        folder: Path,
        progress: relocate.Progress | None,
        stop: threading.Event | None,
        places: dict[str, Any],
    ) -> relocate.Move:
        """Move every file of a split model, and its index, as one — or none of them.

        Half a split model in each of two folders is not a model in either, so a part that
        will not go, or a stop partway, puts back the parts that already went.
        """
        parts = [Path(p) for p in model.parts] or [Path(model.path)]
        index = _shard_index(parts[0])
        pieces = parts + ([index] if index is not None else [])
        first = folder / parts[0].name
        if path_key(first) == path_key(parts[0]):
            return relocate.Move(path=parts[0], unchanged=True)
        for piece in pieces:
            if (folder / piece.name).exists():
                raise FileExistsError(f"{folder} already holds a {piece.name}")
        folder.mkdir(parents=True, exist_ok=True)
        result = relocate.Move(path=first)
        done: list[tuple[Path, Path]] = []
        try:
            for position, piece in enumerate(pieces):
                # The first part is the one the record is named after, so it is the one that
                # takes the record with it.
                extra = places if position == 0 else {}
                moved = relocate.move(piece, folder, progress=progress, stop=stop, **extra)
                done.append((moved.path, piece.parent))
                result.companions.extend(moved.companions)
                result.failed.extend(moved.failed)
        except BaseException:
            for moved_to, came_from in reversed(done):
                with contextlib.suppress(OSError):
                    relocate.move(moved_to, came_from, **(places if moved_to.name == first.name else {}))
            raise
        return result

    def files(self, model_id: int) -> list[Path]:
        """Everything a delete of this model would take."""
        model = self.get(model_id)
        places = self.places()
        if model.parts:
            parts = [Path(p) for p in model.parts]
            found = [p for p in parts if p.is_file()]
            index = _shard_index(parts[0])
            if index is not None:
                found.append(index)
            return found + relocate.companions(parts[0], **places)
        path = Path(model.path)
        listed = erase.belongings(path, **places)
        for pair in model.left_behind:
            stray = Path(pair.get("from", ""))
            if stray.is_file() and stray not in listed:
                listed.append(stray)
        return listed

    def delete_files(self, model_id: int) -> erase.Erased:
        with self.working_on(model_id):
            model = self.get(model_id)
            places = self.places()
            if model.parts:
                parts = [Path(p) for p in model.parts]
                index = _shard_index(parts[0])
                result = erase.remove(parts)
                rest = erase.remove(
                    ([index] if index is not None else []) + relocate.companions(parts[0], **places)
                )
                result.deleted.extend(rest.deleted)
                result.failed.extend(rest.failed)
            else:
                result = erase.erase(Path(model.path), **places)
                strays = erase.remove(
                    Path(pair["from"]) for pair in model.left_behind if pair.get("from")
                )
                result.deleted.extend(strays.deleted)
                result.failed.extend(strays.failed)
            self.db.delete_model(model_id)
            self._orphan_tasks(model_id, db.DELETED)
        self.emit({"type": "model_removed", "id": model_id})
        return result

    def _orphan_tasks(self, model_id: int, fate: str) -> None:
        for task in self.db.tasks_for_model(model_id):
            self.db.update(task.id, model_id=None, fate=fate, archived=True)
            refreshed = self.db.get(task.id)
            if refreshed is not None:
                self.emit({"type": "task", "task": refreshed.to_json()})

    def leftovers(self, model_id: int) -> list[Path]:
        """What a missing model has left on disk: its record, and whatever was named after
        it where it used to be. Offered for deletion when it is forgotten."""
        model = self.get(model_id)
        places = self.places()
        path = Path(model.path)
        found: list[Path] = []
        record = sidecar.find_record(path, **places)
        if record is not None:
            found.append(record)
        if model.record and Path(model.record).is_file() and Path(model.record) not in found:
            found.append(Path(model.record))
        # Not a file another model of the same name before the dot can claim: those are
        # nobody's to delete on behalf of a model that is not even there.
        for other in relocate.named_companions(path):
            if other not in found:
                found.append(other)
        found.extend(erase.leftovers(path))
        return found

    def forget(self, model_id: int, cleanup: bool = False) -> erase.Erased:
        with self.working_on(model_id):
            model = self.get(model_id)
            # The file may have come back since the last walk: a model that is on the disk
            # is not forgotten, and certainly not with its sidecars deleted.
            if Path(model.path).is_file():
                raise ValueError(
                    "this model is on the disk — read the folders again, or delete its files"
                )
            result = erase.remove(self.leftovers(model_id)) if cleanup else erase.Erased()
            self.db.delete_model(model_id)
            self._orphan_tasks(model_id, db.FORGOTTEN)
        self.emit({"type": "model_removed", "id": model_id})
        return result

    def candidates(self, model: Model) -> list[Model]:
        """Models on disk that could be this missing one: its name and its size, or the
        same hash under any name."""
        if model.state != db.MISSING:
            return []
        found = []
        for other in self.db.list_models():
            if other.id == model.id or other.state != db.PRESENT:
                continue
            same_hash = model.sha256 and other.sha256 and model.sha256 == other.sha256
            same_file = (
                other.filename.lower() == model.filename.lower()
                and model.size is not None and other.size == model.size
            )
            if same_hash or same_file:
                found.append(other)
        return found

    def relink(self, model_id: int, candidate_id: int) -> Model:
        """Make a missing model the one found on disk, keeping everything it carried."""
        with self.working_on(model_id, candidate_id):
            model = self.get(model_id)
            other = self.get(candidate_id)
            if model.state != db.MISSING:
                raise ValueError("this model is not missing")
            if other.state != db.PRESENT or not Path(other.path).is_file():
                raise FileNotFoundError(f"{other.path} is not there any more")
            self._absorb(model, other)
        self.emit({"type": "model_removed", "id": candidate_id})
        self.announce(model_id)
        return self.get(model_id)

    def _absorb(self, model: Model, other: Model) -> None:
        places = self.places()
        new = Path(other.path)
        self._relocate_record(model, new, places)
        note = model.note
        if other.note and other.note != note:
            note = f"{note}\n\n{other.note}" if note else other.note
        # What a download knows about its file — where it came from, the hash it was
        # checked against — is kept, whichever of the two rows it was on.
        carried: dict[str, Any] = {}
        if other.origin == db.DOWNLOADED and model.origin != db.DOWNLOADED:
            carried = {
                "origin": db.DOWNLOADED, "provider": other.provider, "identity": other.identity,
                "meta": other.meta, "sha256": other.sha256, "hash_source": other.hash_source,
                "hashed_mtime": other.hashed_mtime, "category": other.category,
                "confidence": other.confidence, "reason": other.reason,
                "base_model": other.base_model, "lookup": other.lookup, "updates": other.updates,
            }
        self.db.delete_model(other.id)
        for task in self.db.tasks_for_model(other.id):
            self.db.update(task.id, model_id=model.id)
        self._settle(model, new, other.size or 0, other.mtime or 0.0,
                     [Path(p) for p in other.parts], places)
        self.db.update_model(
            model.id,
            header=other.header,
            extras=other.extras,
            sniffed_mtime=other.sniffed_mtime,
            fingerprint=other.fingerprint,
            autov1=other.autov1,
            sampled_mtime=other.sampled_mtime,
            note=note,
            **carried,
        )
        if note != model.note:
            record = sidecar.find_record(new, **places)
            if record is not None:
                with contextlib.suppress(OSError, ValueError):
                    sidecar.annotate(record, note or "")

    def link_to(self, model_id: int, chosen: Path, confirmed: bool = False) -> dict[str, Any]:
        """Point a missing model at a file a person picked in the system's own dialog."""
        model = self.get(model_id)
        if model.state != db.MISSING:
            raise ValueError("this model is not missing")
        if not chosen.is_file():
            raise FileNotFoundError(f"{chosen} is not a file")
        if not is_model_file(chosen.name):
            raise ValueError(f"{chosen.name} is not a model file")
        size = chosen.stat().st_size
        if model.size is not None and size != model.size and not confirmed:
            token = secrets.token_urlsafe(16)
            self._pending[token] = (model_id, str(chosen), time.time() + PENDING_FOR)
            return {
                "confirm": {
                    "token": token,
                    "path": str(chosen),
                    "expected": model.size,
                    "actual": size,
                }
            }
        existing = self.db.model_at(chosen)
        if existing is not None and existing.id != model_id:
            self.relink(model_id, existing.id)
            return {"ok": True, "path": str(chosen)}
        with self.working_on(model_id):
            model = self.get(model_id)
            if model.state != db.MISSING:
                raise ValueError("this model is not missing")
            places = self.places()
            self._relocate_record(model, chosen, places)
            stat = chosen.stat()
            self._settle(model, chosen, stat.st_size, stat.st_mtime, [], places)
            if model.hash_source == "computed" and model.size != stat.st_size:
                self.db.update_model(model_id, sha256=None, hash_source=None, hashed_mtime=None)
        self.announce(model_id)
        self.inspect_soon()
        return {"ok": True, "path": str(chosen)}

    def confirm_link(self, model_id: int, token: str) -> dict[str, Any]:
        self._pending = {k: v for k, v in self._pending.items() if v[2] > time.time()}
        pending = self._pending.pop(token, None)
        if pending is None or pending[0] != model_id:
            raise LookupError("that choice has expired — pick the file again")
        return self.link_to(model_id, Path(pending[1]), confirmed=True)

    def bring_back(self, model_id: int) -> relocate.Move:
        """Move what a model left in its old folder to where it belongs beside it now."""
        with self.working_on(model_id):
            model = self.get(model_id)
            path = self._present(model)
            pairs = [
                (Path(p["from"]), Path(p["to"]))
                for p in model.left_behind if p.get("from") and p.get("to")
            ]
            result = relocate.bring(path, pairs)
            remaining = [
                {"from": str(a), "to": str(b)}
                for a, b in pairs if a.is_file() and not b.exists()
            ]
            record = sidecar.find_record(path, **self.places())
            self.db.update_model(
                model_id,
                left_behind=remaining,
                record=str(record) if record is not None else model.record,
            )
        self.announce(model_id, tasks=False)
        return result

    def set_note(self, model_id: int, note: str) -> tuple[str | None, bool]:
        """Write your own note, into the record, writing the record first if need be.

        A missing model keeps the note in the library — there is no file to put a record
        beside — and in its record too, if the record is still where it was.
        """
        text = note.strip() or None
        with self.working_on(model_id):
            model = self.get(model_id)
            places = self.places()
            path = Path(model.path)
            if model.state != db.PRESENT or not path.is_file():
                record = sidecar.find_record(path, **places)
                if record is not None:
                    text = sidecar.annotate(record, note)
                self.db.update_model(model_id, note=text)
                self.db.retarget_tasks(model_id, note=text)
                created = False
            else:
                record = sidecar.find_record(path, **places)
                created = False
                if record is None:
                    if text is None:
                        self.db.update_model(model_id, note=None)
                        self.db.retarget_tasks(model_id, note=None)
                        self.announce(model_id)
                        return None, False
                    verdict, built = self.record_for(model)
                    record = sidecar.write(path, verdict, built, compat=False, triggers=False, **places)
                    created = True
                text = sidecar.annotate(record, note)
                self.db.update_model(model_id, note=text, record=str(record))
                self.db.retarget_tasks(model_id, note=text)
        self.announce(model_id)
        return text, created

    def record_for(self, model: Model) -> tuple[Verdict, sidecar.Record]:
        """The record this model would have, had it been downloaded here."""
        try:
            category = Category(model.category) if model.category else Category.OTHER
        except ValueError:
            category = Category.OTHER
        verdict = Verdict(category, model.confidence or "low", model.reason or "",
                          base_model=model.base_model)
        url = None
        if model.identity.get("provider"):
            identity = FileIdentity(
                provider=str(model.identity.get("provider")),
                ref=dict(model.identity.get("ref") or {}),
            )
            url = source_url(identity, model.meta.get("host"))
        record = sidecar.Record(
            filename=Path(model.path).name,
            provider=model.provider,
            source_url=url,
            sha256=model.sha256,
            size=model.size,
            meta=model.meta,
            note=model.note,
        )
        return verdict, record

    # --- details of one model -----------------------------------------------

    def details(self, model_id: int) -> dict[str, Any]:
        model = self.get(model_id)
        data = self.summary(model)
        places = self.places()
        path = Path(model.path)
        record_data = None
        record = sidecar.find_record(path, **places)
        if record is not None:
            record_data = sidecar.read_record(record)
        meta = model.meta
        data.update({
            "header": model.header,
            "extras": model.extras,
            "reason": model.reason,
            "lookup_info": model.lookup,
            "updates": model.updates,
            "record": record_data,
            "record_path": str(record) if record is not None else None,
            "left_behind_files": model.left_behind,
            "parts_list": model.parts,
            "meta": {
                key: meta.get(key)
                for key in ("model_name", "version_name", "model_id", "version_id", "repo_id",
                            "commit", "path", "model_type", "precision", "quantisation",
                            "format", "host", "file_type", "base_model")
                if meta.get(key) not in (None, "")
            },
            "page": _page_url(meta),
            "history": [t.to_json() for t in self.db.tasks_for_model(model_id)],
            "duplicates": [
                self.summary(other) for other in (
                    self.db.models_with_hash(model.sha256) if model.sha256 else []
                ) if other.id != model_id
            ],
            "candidates": [self.summary(c) for c in self.candidates(model)],
            "exists": path.is_file(),
            "names": self._names(model),
        })
        return data

    def _names(self, model: Model) -> list[dict[str, Any]]:
        """Every name of this model's file, when it has more than one — the library's own
        and the ones it does not list — for the inspector to show where else it is."""
        path = Path(model.path)
        got = links.identity(path) if model.state == db.PRESENT else None
        if got is None or got[1] < 2:
            return []
        known = {m.key: m.id for m in self.db.list_models()}
        return [
            {"path": str(name), "model_id": known.get(path_key(name)),
             "this": path_key(name) == path_key(path)}
            for name in links.names(path)
        ]

    def previews(self, model: Model) -> list[dict[str, Any]]:
        from ..library import previews as remote

        listed = remote.entries(model.meta)
        if listed:
            return [
                {
                    "index": index,
                    "type": entry.get("type") or "image",
                    "nsfw": bool(entry.get("nsfw")),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "meta": entry.get("meta") or {},
                    "url": entry.get("url"),
                    "local": False,
                }
                for index, entry in enumerate(listed)
            ]
        if model.extras.get("image") or model.header.get("thumbnail"):
            return [{"index": 0, "type": "image", "nsfw": False, "meta": {}, "url": None,
                     "local": True}]
        return []

    # --- hashing, identifying, verifying ------------------------------------

    def job_state(self) -> dict[str, Any]:
        return {
            "current": self.current_job,
            "queued": [dict(j) for j in self._jobs],
        }

    def enqueue(self, kind: str, model_ids: Iterable[int]) -> int:
        """Line models up to be hashed — to identify them, to verify them, or only to know
        their hash. Returns how many were added; one already waiting is not added twice."""
        waiting = {(j["kind"], j["model_id"]) for j in self._jobs}
        if self.current_job:
            waiting.add((self.current_job["kind"], self.current_job["model_id"]))
        added = 0
        for model_id in model_ids:
            model = self.db.get_model(model_id)
            if model is None or model.state != db.PRESENT or model.parts:
                continue
            if (kind, model_id) in waiting:
                continue
            self._jobs.append({"kind": kind, "model_id": model_id, "name": model.filename,
                               "size": model.size})
            waiting.add((kind, model_id))
            added += 1
        if added and self._job_wake is not None:
            self._job_wake.set()
        self._emit_jobs()
        return added

    def stop_jobs(self) -> int:
        dropped = len(self._jobs)
        self._jobs.clear()
        self._job_stop.set()
        self._emit_jobs()
        return dropped

    def _emit_jobs(self, **progress: Any) -> None:
        state = self.job_state()
        if progress and state["current"] is not None:
            state["current"] = {**state["current"], **progress}
        self.emit({"type": "jobs", **state})

    async def _job_loop(self) -> None:
        assert self._job_wake is not None
        while not self._closing:
            await self._job_wake.wait()
            self._job_wake.clear()
            while self._jobs and not self._closing:
                job = self._jobs.pop(0)
                self.current_job = job
                self._job_stop.clear()
                self._emit_jobs(done=0)
                try:
                    await self._run_job(job)
                except relocate.Cancelled:
                    self.emit({"type": "toast", "level": "info",
                               "message": f"Stopped hashing {job['name']}"})
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the queue
                    self.emit({"type": "toast", "level": "error",
                               "message": f"{job['name']}: {exc}"})
                finally:
                    self.current_job = None
                    self._emit_jobs()

    async def _run_job(self, job: dict[str, Any]) -> None:
        model = self.db.get_model(job["model_id"])
        if model is None or model.state != db.PRESENT:
            return
        path = Path(model.path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not there any more")
        kind = job["kind"]

        fresh = (
            model.sha256 and model.hash_source == "computed"
            and model.hashed_mtime is not None and model.mtime is not None
            and abs(model.hashed_mtime - model.mtime) < 1e-3
        )
        # A download's own hash was checked against the bytes as they landed; looking the
        # model up by it needs no second read of the whole file.
        if kind == "identify" and model.sha256 and model.hash_source == "download":
            await self._identify(model.id, model.sha256)
            return
        hub: list[online.HubFile] | None = None
        if kind == "identify" and not fresh and (model.size or 0) >= QUICK_ABOVE:
            # The quick look first: a big file is read whole only when one of the services
            # has something it could be. Most text encoders and VAEs are on neither — or on
            # the Hub, which is found by name and proven by the hash read next.
            try:
                seen, hub = await self._quick_look(model)
            except (RuntimeError, OSError):
                seen, hub = True, None
            if not seen:
                self._not_found(model, quick=True)
                return
        if fresh and kind != "verify":
            digest = model.sha256
        else:
            with self.working_on(model.id):
                digest = await asyncio.to_thread(self._hash, path, model.size or 0)

        model = self.get(model.id)
        if model.hash_source == "download" and model.sha256:
            # The hash the service advertised is what the file should be. Kept, and the
            # one read off the disk now is compared against it rather than replacing it.
            lookup = dict(model.lookup)
            lookup["verified"] = {"at": time.time(), "ok": digest == model.sha256,
                                  "sha256": digest, "mtime": model.mtime}
            self.db.update_model(model.id, lookup=lookup)
        else:
            lookup = dict(model.lookup)
            if kind == "verify":
                lookup["verified"] = {"at": time.time(), "ok": None, "sha256": digest,
                                      "mtime": model.mtime}
            self.db.update_model(model.id, sha256=digest, hash_source="computed",
                                 hashed_mtime=model.mtime, lookup=lookup)
        self.announce(model.id, tasks=False)

        if kind == "verify":
            model = self.get(model.id)
            ok = (model.lookup.get("verified") or {}).get("ok")
            message = (
                f"{model.filename} matches the hash it was downloaded with" if ok
                else f"{model.filename} does not match the hash it was downloaded with"
                if ok is False else f"{model.filename} hashed: {digest[:12]}…"
            )
            self.emit({"type": "toast", "level": "error" if ok is False else "info",
                       "message": message, "models": [model.id]})
        elif kind == "identify":
            await self._identify(model.id, digest, hub)

    async def _quick_look(self, model: Model) -> tuple[bool, list[online.HubFile] | None]:
        """Whether Civitai or the Hub has anything this file could be, without reading it:
        Civitai by the AutoV1 hash, the Hub by the name and the exact size."""
        autov1 = model.autov1 if _same_time(model.sampled_mtime, model.mtime) else None
        if autov1 is None:
            taken = await asyncio.to_thread(self._sample, model)
            self.db.update_model(model.id, **taken)
            autov1 = taken.get("autov1")
        if autov1:
            host = str(model.meta.get("host") or "civitai.com")
            version = await online.civitai_by_hash(
                self._http(), autov1, host, self.settings.effective_civitai_token
            )
            if version is not None and online.civitai_file(version, autov1=autov1, size=model.size):
                return True, None
        else:
            # Nothing to ask Civitai with: only the whole file's hash can say.
            return True, None
        hub = await online.find_on_hub(
            self._http(), model.filename, model.size, None,
            self.settings.effective_hf_token, self._hub_cache,
        )
        return bool(hub), hub

    def _not_found(self, model: Model, quick: bool = False) -> None:
        current = self.get(model.id)
        record = dict(current.lookup)
        record.update(result="not_found", at=time.time(), message=None,
                      searched=["civitai", "huggingface"], quick=quick)
        self.db.update_model(model.id, lookup=record)
        self.announce(model.id, tasks=False)
        self.emit({"type": "toast", "level": "info", "models": [model.id],
                   "message": f"{model.filename} is not on Civitai or HuggingFace"
                              + (" — checked without reading the whole file" if quick else "")})

    def _hash(self, path: Path, total: int) -> str:
        digest = hashlib.sha256()
        done = 0
        last = 0.0
        loop_emit = self._emit_jobs
        with open(path, "rb") as handle:
            while True:
                if self._job_stop.is_set():
                    raise relocate.Cancelled("stopped")
                block = handle.read(HASH_BLOCK)
                if not block:
                    break
                digest.update(block)
                done += len(block)
                now = time.monotonic()
                if now - last >= PROGRESS_INTERVAL:
                    last = now
                    loop_emit(done=done, total=total)
        return digest.hexdigest()

    def _lookup_failed(self, model_id: int, message: str) -> None:
        current = self.get(model_id)
        record = dict(current.lookup)
        record.update(result="error", at=time.time(), message=message)
        self.db.update_model(model_id, lookup=record)
        self.announce(model_id, tasks=False)

    async def _identify(
        self, model_id: int, digest: str, hub: list[online.HubFile] | None = None
    ) -> None:
        """Ask Civitai by the file's hash; failing that, the Hub by its name, the answer
        proven by the same hash. `hub` is what a quick look already found there."""
        model = self.get(model_id)
        host = str(model.meta.get("host") or "civitai.com")
        try:
            version = await online.civitai_by_hash(
                self._http(), digest, host, self.settings.effective_civitai_token
            )
        except RuntimeError as exc:
            self._lookup_failed(model_id, str(exc))
            raise
        if version is None:
            try:
                if hub is None:
                    hub = await online.find_on_hub(
                        self._http(), model.filename, model.size, digest,
                        self.settings.effective_hf_token, self._hub_cache,
                    )
                else:
                    hub = [h for h in hub if h.sha256 == digest.lower()]
            except RuntimeError as exc:
                self._lookup_failed(model_id, f"not on Civitai; {exc}")
                raise
            if hub:
                await self._found_on_hub(model_id, hub[0])
            else:
                self._not_found(model)
            return

        record = dict(model.lookup)
        path = Path(model.path)
        meta = _meta_for_hash(version, digest, path, model.size)
        meta["host"] = host
        if model.origin == db.DOWNLOADED or (model.provider and model.provider != "civitai"):
            # Downloaded from somewhere else: where it came from is a fact about this file,
            # and Civitai having the same bytes does not change it. Said, not rewritten.
            record.update(
                result="found", at=time.time(), message=None,
                version_id=meta.get("version_id"), model_id=meta.get("model_id"),
                model_name=meta.get("model_name"), version_name=meta.get("version_name"),
                page=_page_url(meta),
            )
            self.db.update_model(model_id, lookup=record)
            self.announce(model_id, tasks=False)
            self.emit({"type": "toast", "level": "info", "models": [model_id],
                       "message": f"{model.filename} is also on Civitai, as "
                                  f"{meta.get('model_name') or 'a model'}"})
            return
        header, sniffed = await asyncio.to_thread(details.inspect, path)
        updates: dict[str, Any] = {
            "meta": meta,
            "provider": "civitai",
            "lookup": {**record, "result": "found", "source": "civitai", "at": time.time(),
                       "message": None, "version_id": meta.get("version_id"),
                       "model_id": meta.get("model_id")},
        }
        if meta.get("version_id") and meta.get("file_id"):
            updates["identity"] = {
                "provider": "civitai",
                "ref": {"version_id": meta["version_id"], "file_id": meta["file_id"]},
            }
        if model.origin == db.FOUND:
            verdict = details.judge(path, sniffed, header, meta, model.extras, self.roots)
            updates.update(category=verdict.category.value, confidence=verdict.confidence,
                           reason=verdict.reason, base_model=verdict.base_model, header=header)
        self.db.update_model(model_id, **updates)
        await self._write_identified(model_id, version)
        self.announce(model_id, tasks=False)
        name = meta.get("model_name") or model.filename
        self.emit({"type": "toast", "level": "info", "models": [model_id],
                   "message": f"{model.filename} is {name}"
                              + (f" / {meta['version_name']}" if meta.get("version_name") else "")})

    async def find_online(self, model_id: int) -> dict[str, Any]:
        """Where a model could be downloaded from, when nothing kept says: Civitai by the
        hash kept for it — the whole file's, which is proof, or the AutoV1, which with the
        size is a strong hint — and the Hub by its name and size, proven by the hash when
        the hash is known. The file itself is not needed: a missing model is the usual case.

        Each answer carries what a download of it needs, under keys starting with `_` that
        the page is never shown.
        """
        model = self.get(model_id)
        host = str(model.meta.get("host") or "civitai.com")
        token = self.settings.effective_civitai_token
        found: list[dict[str, Any]] = []
        problems: list[str] = []
        for digest, proof in ((model.sha256, True), (model.autov1, False)):
            if not digest:
                continue
            try:
                version = await online.civitai_by_hash(self._http(), digest, host, token)
            except RuntimeError as exc:
                problems.append(str(exc))
                break
            entry = version and online.civitai_file(
                version, sha256=digest if proof else None, autov1=None if proof else digest,
                size=model.size,
            )
            if not entry:
                continue
            meta = _meta_for_hash(version, str(entry.get("hashes", {}).get("SHA256") or digest).lower(),
                                  Path(model.path), model.size)
            meta["host"] = host
            found.append({
                "source": "civitai", "host": host, "proven": proof,
                "title": " / ".join(str(p) for p in (meta.get("model_name"), meta.get("version_name")) if p)
                         or "a model on Civitai",
                "detail": str(entry.get("name") or ""),
                "size": int(float(entry.get("sizeKB") or 0) * 1024) or None,
                "page": _page_url(meta),
                "_identity": {"provider": "civitai",
                              "ref": {"version_id": meta.get("version_id"), "file_id": entry.get("id")}},
                "_meta": meta,
            })
            break
        try:
            hub = await online.find_on_hub(
                self._http(), model.filename, model.size, model.sha256,
                self.settings.effective_hf_token, self._hub_cache,
            )
        except RuntimeError as exc:
            problems.append(str(exc))
            hub = []
        for hit in hub[:6]:
            found.append({
                "source": "huggingface", "host": "huggingface.co",
                "proven": bool(model.sha256 and hit.sha256 == model.sha256),
                "same_name": hit.same_name,
                "title": hit.repo_id, "detail": hit.path, "size": hit.size, "page": hit.page,
                "downloads": hit.downloads,
                "_identity": {"provider": "huggingface", "ref": {
                    "repo_id": hit.repo_id, "repo_type": "model", "revision": "main", "path": hit.path}},
                "_meta": online.hub_meta(hit),
            })
        # Proof first; then the service it came from, since that is where it is likeliest to
        # stay; then the most downloaded, which is the original more often than its mirrors.
        found.sort(key=lambda hit: (
            not hit["proven"], hit["source"] != (model.provider or "civitai"), -(hit.get("downloads") or 0),
        ))
        stem = Path(model.filename).stem
        return {
            "found": found,
            "problems": problems,
            "searched": {"sha256": bool(model.sha256), "autov1": bool(model.autov1)},
            # Where a person can look for themselves, when the services' own lookups are not
            # enough — a file renamed on the way here is found by nothing but a person.
            "search": {
                "civitai": f"https://{host}/search/models?query={_query(stem)}",
                "huggingface": f"{online.HUB}/models?search={_query(stem)}",
            },
        }

    async def _found_on_hub(self, model_id: int, hit: online.HubFile) -> None:
        """The Hub has this very file — the hash says so. A model found on disk takes its
        description from there: the repository, the path in it, its licence and base model,
        and a way to download it again should it go."""
        model = self.get(model_id)
        meta = online.hub_meta(hit)
        record = dict(model.lookup)
        record.update(result="found", source="huggingface", at=time.time(), message=None,
                      repo_id=hit.repo_id, path=hit.path, page=hit.page,
                      model_name=hit.repo_id, version_name=None)
        if model.origin == db.DOWNLOADED or (model.provider and model.provider != "huggingface"):
            self.db.update_model(model_id, lookup=record)
            self.announce(model_id, tasks=False)
            self.emit({"type": "toast", "level": "info", "models": [model_id],
                       "message": f"{model.filename} is also on HuggingFace, in {hit.repo_id}"})
            return
        path = Path(model.path)
        header, sniffed = await asyncio.to_thread(details.inspect, path)
        verdict = details.judge(path, sniffed, header, meta, model.extras, self.roots)
        self.db.update_model(
            model_id,
            meta=meta,
            provider="huggingface",
            identity={"provider": "huggingface", "ref": {
                "repo_id": hit.repo_id, "repo_type": "model", "revision": "main", "path": hit.path,
            }},
            lookup=record,
            category=verdict.category.value, confidence=verdict.confidence,
            reason=verdict.reason, base_model=verdict.base_model, header=header,
        )
        await self._write_identified(model_id, None)
        self.announce(model_id, tasks=False)
        self.emit({"type": "toast", "level": "info", "models": [model_id],
                   "message": f"{model.filename} is {hit.repo_id} / {hit.path}"})

    async def _write_identified(self, model_id: int, version: dict[str, Any] | None) -> None:
        """Give an identified model the files a download of it would have left, following
        the same settings — and never over what is already there, since it may be another
        tool's, or somebody's own. From the Hub that is the record alone: the compatibility
        files are Civitai's own formats, with nothing of the Hub's to put in them."""
        model = self.get(model_id)
        path = Path(model.path)
        places = self.places()
        existing = sidecar.find_record(path, **places)
        readable = existing is None or sidecar.read_record(existing) is not None
        # A record that is there and cannot be read is more likely one somebody edited by
        # hand than one that is broken; it is left alone, as `annotate` leaves it alone.
        if self.settings.write_sidecars and readable:
            verdict, record = self.record_for(model)
            if existing is not None:
                data = sidecar.read_record(existing) or {}
                record.note = data.get("note") or model.note
            civitai = version is not None
            with contextlib.suppress(OSError):
                written = await asyncio.to_thread(
                    sidecar.write, path, verdict, record,
                    version, places["sidecar_dir"], places["library_root"],
                    civitai and self.settings.write_compat_files,
                    civitai and self.settings.write_trigger_txt,
                    places["roots"], True,
                )
                self.db.update_model(model_id, record=str(written))
        if (
            self.settings.fetch_previews
            and self.settings.write_compat_files
            and details.local_image(path) is None
        ):
            with contextlib.suppress(Exception):
                await sidecar.fetch_preview(path, model.meta, self._http())
        self.db.update_model(model_id, sniffed_mtime=None)
        self.inspect_soon()

    # --- newer versions -----------------------------------------------------

    def checkable_ids(self) -> list[int]:
        """Every model on disk whose service can be asked about a newer version."""
        return [m.id for m in self.db.list_models() if m.state == db.PRESENT and m.checkable]

    def update_state(self) -> dict[str, Any] | None:
        """How far the check under way has got, for a page that opens in the middle of it."""
        run = self._update_run
        return None if run is None else {**run, "stopping": self._update_stop}

    def _emit_update_run(self) -> None:
        self.emit({"type": "update_check", **(self.update_state() or {"running": False})})

    def stop_update_check(self) -> bool:
        """Ask the check under way to stop: what is being asked right now is answered, and
        nothing more is asked. Everything learnt so far is kept."""
        if self._update_run is None:
            return False
        self._update_stop = True
        self._emit_update_run()
        return True

    async def check_updates(
        self, model_ids: Iterable[int], *, startup: bool = False
    ) -> dict[str, Any]:
        """Ask each model's service whether there is something newer than what is here.

        What counts as newer is the rule in `library/versions.py`: on Civitai, a version for
        the same base model whose name carries a higher number than any version of the
        model here. On the Hub a file keeps its name when it changes, so the question there
        is whether the same path on the same branch now has a different hash.

        Answered with how many were checked; how many updates there are among them — a
        version counted once, however many files of the library it updates — and how many
        of those the last check had not seen; how many are gone from their site, and how
        many could not be asked about this time.
        """
        if self._update_run is not None:
            raise Busy("newer versions are being checked already — see the status bar")
        models = [
            m for m in map(self.db.get_model, model_ids)
            if m is not None and m.state == db.PRESENT and m.checkable
        ]
        run = {"running": True, "done": 0, "total": len(models), "found": 0, "startup": startup}
        self._update_run = run
        self._update_stop = False
        self._emit_update_run()
        have = self._versions_here()
        pages: dict[int, asyncio.Future[Any]] = {}
        semaphore = asyncio.Semaphore(UPDATE_CONCURRENCY)
        found: set[str] = set()
        new: set[str] = set()
        counts = {"checked": 0, "gone": 0, "failed": 0}

        async def one(model: Model) -> None:
            async with semaphore:
                if self._update_stop:
                    return
                before = versions.normalise(model.updates)
                try:
                    record = await self._check_one(model, pages, have)
                except Exception as exc:  # noqa: BLE001 - reported per model
                    counts["failed"] += 1
                    failed = {"at": time.time(), "error": str(exc) or type(exc).__name__}
                    # A check that could not be made this time — the network down, a
                    # service busy — says nothing new: what the last one found stands.
                    if before is not None and before.get("status") != versions.FAILED:
                        record = {**before, "failed": failed}
                    else:
                        record = {"error": failed["error"], "failed": failed}
                        if before and before.get("skipped"):
                            record["skipped"] = before["skipped"]
                        record["status"] = versions.status(record)
                else:
                    counts["checked"] += 1
                    record["checked_at"] = time.time()
                    if before and before.get("skipped"):
                        record["skipped"] = before["skipped"]
                    record["status"] = versions.status(record)
                    if record["status"] == versions.GONE:
                        counts["gone"] += 1
                if record["status"] == versions.UPDATE:
                    found.add(record["group"])
                    if (
                        before is None or before.get("status") != versions.UPDATE
                        or versions.target(before) != versions.target(record)
                    ):
                        new.add(record["group"])
                self.db.update_model(model.id, updates=record)
                run["done"] += 1
                run["found"] = len(found)
                self.announce(model.id, tasks=False)
                self._emit_update_run()

        result: dict[str, Any] = {}
        try:
            await asyncio.gather(*(one(m) for m in models))
        finally:
            result = {
                **counts, "updates": len(found), "new": len(new), "stopped": self._update_stop,
            }
            self._update_run = None
            self._update_stop = False
            self.emit({"type": "update_check", "running": False, "startup": startup,
                       "result": result})
        return result

    def skip_updates(self, model_ids: Iterable[int], skip: bool = True) -> int:
        """Skip the update each of these has — it stops being counted until a version higher
        than it comes out — or count it again. Nothing is asked of any service."""
        changed = 0
        for model_id in model_ids:
            model = self.db.get_model(model_id)
            record = versions.normalise(model.updates) if model is not None else None
            if record is None:
                continue
            updated = versions.skip(record) if skip else versions.unskip(record)
            if updated is None:
                continue
            self.db.update_model(model_id, updates=updated)
            self.announce(model_id, tasks=False)
            changed += 1
        return changed

    def _versions_here(self) -> dict[int, set[int]]:
        """Every Civitai version the library has on disk, by the model it is a version of:
        none of them is anything to download, and the highest is the one to beat."""
        here: dict[int, set[int]] = {}
        for model in self.db.list_models():
            meta = model.meta
            if model.state != db.PRESENT or model.provider != "civitai":
                continue
            with contextlib.suppress(TypeError, ValueError):
                here.setdefault(int(meta["model_id"]), set()).add(int(meta["version_id"]))
        return here

    async def _civitai_page(
        self, host: str, model_id: int, pages: dict[int, asyncio.Future[Any]]
    ) -> dict[str, Any] | None:
        """A model's page on Civitai — every version of it — fetched once per check however
        many files of it are here. None when the model is not there any more."""
        if model_id not in pages:
            pages[model_id] = asyncio.ensure_future(self._fetch_civitai_page(host, model_id))
        return await asyncio.shield(pages[model_id])

    async def _fetch_civitai_page(self, host: str, model_id: int) -> dict[str, Any] | None:
        token = self.settings.effective_civitai_token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = await self._http().get(
            f"{CIVITAI_API.format(host=host)}/models/{model_id}", headers=headers
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def _owned(self, host: str, version_id: Any, file_id: Any) -> bool | None:
        """Whether the account of the Civitai key has bought a version that is sold: one
        request, to its download link, and only for a version that is sold."""
        from ..providers.civitai import CivitaiProvider

        provider = CivitaiProvider(self.settings.effective_civitai_token, host)
        return await provider.owned(int(version_id), file_id, self._http())

    def _hub_http(self) -> httpx.AsyncClient:
        """A client for asking the Hub about one file. It follows no redirect itself: the
        provider does, so that it can tell the Hub's own answer from the CDN's."""
        return httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=False)

    async def _check_one(
        self, model: Model, pages: dict[int, asyncio.Future[Any]], have: dict[int, set[int]]
    ) -> dict[str, Any]:
        """What the service says about this one file now: the record a check leaves, but for
        when it was made and what was skipped, which are the caller's."""
        meta = model.meta
        if model.provider == "civitai":
            host = str(meta.get("host") or "civitai.com")
            model_id = int(meta["model_id"])
            page = f"https://{host}/models/{model_id}"
            record: dict[str, Any] = {"family": f"civitai:{model_id}", "page": page}
            data = await self._civitai_page(host, model_id, pages)
            if data is None:
                return {**record, "gone": True, "error": "the model is no longer on Civitai"}
            listed = data.get("modelVersions") or []
            mine = int(meta["version_id"])
            standing = versions.assess(listed, mine, have.get(model_id, set()))
            if standing.gone:
                return {**record, "gone": True, "error": standing.gone}
            if standing.pick is not None:
                pick = standing.pick
                own = next((v for v in listed if v.get("id") == mine), {})
                ref = (model.identity or {}).get("ref") or {}
                here = versions.find_file(
                    own.get("files"), ref.get("file_id") or meta.get("file_id"), model.sha256
                )
                file = versions.choose_file(pick.get("files"), here)
                paid = versions.access(pick)
                if paid is not None:
                    # Still counted: the choice to buy it is someone else's. Whether it has
                    # been bought is what decides whether Download all fetches it.
                    paid["owned"] = await self._owned(host, pick.get("id"), (file or {}).get("id"))
                record.update(
                    update={
                        "id": pick.get("id"),
                        "name": pick.get("name"),
                        "base_model": pick.get("baseModel"),
                        "published_at": pick.get("publishedAt") or pick.get("createdAt"),
                        "number": list(versions.number(pick.get("name")) or ()),
                        "file": file,
                        "page": f"{page}?modelVersionId={pick.get('id')}",
                        "access": paid,
                    },
                    count=standing.count,
                    group=f"civitai:{model_id}:{pick.get('id')}",
                )
            if standing.others:
                record["others"] = [
                    {"id": v.get("id"), "name": v.get("name"), "base_model": v.get("baseModel"),
                     "access": versions.access(v)}
                    for v in standing.others[: versions.OTHERS_KEPT]
                ]
                record["others_count"] = len(standing.others)
            return record

        from ..providers.huggingface import HuggingFaceProvider

        ref = (model.identity or {}).get("ref") or {}
        revision = str(ref.get("revision") or "main")
        # A download pinned to a commit is compared with the branch it would have followed.
        if len(revision) == 40:
            revision = "main"
        page = f"https://huggingface.co/{ref['repo_id']}/blob/{revision}/{ref['path']}"
        record = {"family": f"huggingface:{ref['repo_id']}:{ref['path']}", "page": page}
        provider = HuggingFaceProvider(self.settings.effective_hf_token)
        identity = FileIdentity(provider="huggingface", ref={
            "repo_id": ref["repo_id"], "repo_type": ref.get("repo_type") or "model",
            "revision": revision, "path": ref["path"],
        })
        try:
            async with self._hub_http() as hub:
                info = await provider.probe(identity, hub)
        except SfdError as exc:
            if exc.code in HUB_GONE:
                return {**record, "gone": True, "error": HUB_GONE[exc.code]}
            raise
        if not info.sha256 or not model.sha256:
            return {**record, "error": "the Hub does not say what this file hashes to"}
        if info.sha256.lower() != model.sha256.lower():
            record.update(
                update={
                    "sha256": info.sha256.lower(),
                    "commit": (info.meta or {}).get("commit"),
                    "name": f"a newer commit on {revision}",
                    "page": page,
                },
                group=record["family"],
            )
        return record

    # --- duplicates and rubbish ---------------------------------------------

    def duplicates(self) -> dict[str, Any]:
        """The same file kept in more than one place.

        A size shared by several files is only a reason to look closer: files of one
        architecture at one precision come out the same number of bytes. Their fingerprints
        tell different files apart for the price of a few small reads — taken in the
        background already, and here for any file that changed since. What is left is either
        certain, because the hashes of the whole files agree, or very likely, until the
        files are hashed to be sure.
        """
        present = [
            m for m in self.db.list_models()
            if m.state == db.PRESENT and not m.parts and (m.size or 0) >= DUPLICATE_MIN
        ]
        by_size: dict[int, list[Model]] = {}
        for model in present:
            by_size.setdefault(model.size or 0, []).append(model)
        shared = {size: group for size, group in by_size.items() if len(group) > 1}

        buckets: dict[tuple[int, str], list[Model]] = {}
        for size, group in shared.items():
            for model in group:
                mark = self._fresh_fingerprint(model)
                if mark is not None:
                    buckets.setdefault((size, mark), []).append(model)

        # Which file each name is, read now: a link made a minute ago changed no date.
        ident = {m.id: links.identity(m.path) for m in present}
        groups: list[dict[str, Any]] = []
        for (size, mark), members in buckets.items():
            if len(members) < 2:
                continue
            for status, copies in _by_hash(members):
                files = _physical(copies, ident)
                # Several names of one file on disk — hard links — take the room of one.
                if len(files) > 1:
                    groups.append(self._duplicate_group(size, mark, status, copies, len(files), ident))
        groups.sort(key=lambda g: (-g["wasted"], g["models"][0]["filename"].lower()))
        sizes = {g["size"] for g in groups}

        # Files already under several names: nothing to gain, and a way back to copies.
        by_file: dict[str, list[Model]] = {}
        for model in present:
            got = ident.get(model.id)
            if got is not None and got[1] > 1:
                by_file.setdefault(got[0], []).append(model)
        linked = []
        for file_id, names in by_file.items():
            count = ident[names[0].id][1]
            linked.append({
                "key": f"link:{file_id}",
                "size": names[0].size or 0,
                "names": count,
                # Names the library does not list: in a folder that is not part of it.
                "outside": max(0, count - len(names)),
                "saved": (names[0].size or 0) * (count - 1),
                "models": [self._linked_summary(m, ident) for m in names],
            })
        linked.sort(key=lambda g: (-g["saved"], g["models"][0]["filename"].lower()))
        return {
            "groups": groups,
            "linked": linked,
            "wasted": sum(g["wasted"] for g in groups if g["status"] == "same"),
            "likely": sum(g["wasted"] for g in groups if g["status"] == "likely"),
            "saved": sum(g["saved"] for g in linked),
            # How many sizes were shared, and how many of those the fingerprints showed to
            # be different models — what the old guess by size alone would have listed.
            "shared_sizes": len(shared),
            "told_apart": len([size for size in shared if size not in sizes]),
        }

    def _linked_summary(self, model: Model, ident: dict[int, tuple[str, int] | None]) -> dict[str, Any]:
        got = ident.get(model.id)
        return {
            **self.summary(model),
            "file_id": got[0] if got else None,
            # The volume: a hard link reaches only as far as it.
            "volume": got[0].split(":", 1)[0] if got else None,
        }

    def _fresh_fingerprint(self, model: Model) -> str | None:
        """The fingerprint of the file as it is now, taking it first if it is out of date."""
        if model.fingerprint and _same_time(model.sampled_mtime, model.mtime):
            return model.fingerprint
        if self.is_busy(model.id):
            return None
        updates = self._sample(model)
        self.db.update_model(model.id, **updates)
        return updates.get("fingerprint")

    def _duplicate_group(
        self, size: int, mark: str, status: str, copies: list[Model], files: int,
        ident: dict[int, tuple[str, int] | None],
    ) -> dict[str, Any]:
        keep, why = self._keeper(copies)
        unhashed = [m for m in copies if not m.sha256]
        return {
            # Stable across looks, so the page can remember which copy was chosen to keep.
            "key": f"{size}:{mark}" + (f":{copies[0].sha256[:16]}" if status == "same" else ""),
            "status": status,
            "size": size,
            "copies": files,
            "wasted": size * (files - 1),
            "keep": keep.id,
            "keep_why": why,
            "caution": self._caution(copies),
            "to_hash": [m.id for m in unhashed],
            "to_read": sum(m.size or 0 for m in unhashed),
            "models": [self._linked_summary(m, ident) for m in copies],
        }

    def _caution(self, copies: list[Model]) -> str | None:
        """Why deleting a copy might break something, when the folders suggest it.

        A folder named for a kind — `vae`, `loras` — is read whole by its loader, whatever is
        under it. Any other folder is some node's own, and the insightface packs are the
        usual case: the same `.onnx` in `insightface/models/antelopev2`, `.../buffalo_l` and
        `simswap/models/buffalo_l`, because each node reads its own folder and each pack is
        loaded as a set.
        """
        loaders = sorted({self._loader_folder(m) for m in copies}, key=str.lower)
        if len(loaders) > 1:
            return (
                f"Kept in folders different nodes read ({', '.join(loaders)}) — each may need "
                f"its own copy."
            )
        parents = sorted({Path(m.path).parent.name for m in copies}, key=str.lower)
        if len(parents) > 1 and ALIASES.get(loaders[0].lower()) is None:
            return (
                f"Kept in different folders of {loaders[0]} ({', '.join(parents)}) — a node may "
                f"load each folder as a set."
            )
        return None

    def _keeper(self, copies: list[Model]) -> tuple[Model, str]:
        """Which copy to keep when nobody has said: the one the library knows most about,
        in the folder downloads are filed into."""

        def weigh(model: Model) -> tuple[int, list[str]]:
            points, reasons = 0, []
            root, _relative = self.placement(model.path)
            if root == 0 and len(self.roots) > 1:
                points += 8
                reasons.append("in the main library folder")
            if model.origin == db.DOWNLOADED:
                points += 4
                reasons.append("downloaded here")
            if model.note:
                points += 2
                reasons.append("has your note")
            elif model.identified:
                points += 2
                reasons.append("identified")
            loader = self._loader_folder(model)
            kind = ALIASES.get(loader.lower())
            if kind is not None and model.category == kind.value:
                points += 1
                reasons.append(f"in the {loader} folder")
            return points, reasons

        ranked = sorted(copies, key=lambda m: (-weigh(m)[0], m.first_seen, m.id))
        reasons = weigh(ranked[0])[1]
        return ranked[0], ", ".join(reasons[:2]) if reasons else "the first one found"

    # --- one file under several names -----------------------------------------

    def link_copies(self, keep_id: int, other_ids: Iterable[int]) -> dict[str, Any]:
        """Make the other copies of a file names of the one kept.

        Every path and every name keeps working, and the room of each copy is freed. Only
        for copies whose whole-file hashes prove them the same — the rule deleting a copy
        follows too, since the copy's own bytes go either way — and only on the drive the
        kept one is on, which is as far as a hard link reaches. A file that changed since the
        library last read it is left alone: the hash it carries may no longer be its own.
        """
        others = [i for i in dict.fromkeys(other_ids) if i != keep_id]
        results: list[dict[str, Any]] = []
        with self.working_on(keep_id, *others):
            keep = self.get(keep_id)
            keep_path = self._present(keep)
            if not keep.sha256:
                raise ValueError("the copy to keep has no hash yet — confirm by hash first")
            if not _unchanged(keep, keep_path):
                raise ValueError(f"{keep.filename} changed since the library read it — look again")
            kept = links.identity(keep_path)
            for other_id in others:
                other = self.db.get_model(other_id)
                try:
                    if other is None:
                        raise LookupError("no such model")
                    path = self._present(other)
                    if other.sha256 != keep.sha256:
                        raise ValueError("not proven the same file — confirm by hash first")
                    if not _unchanged(other, path):
                        raise ValueError("changed since the library read it — look again")
                    now = links.identity(path)
                    if kept is not None and now is not None and now[0] == kept[0]:
                        results.append({"id": other_id, "ok": True, "freed": 0, "unchanged": True})
                        continue
                    if not links.same_volume(keep_path, path):
                        raise ValueError("on another drive, where a hard link cannot reach")
                    # Its own room comes back only if this was the copy's last name.
                    freed = (other.size or 0) if now is None or now[1] == 1 else 0
                    links.link_over(keep_path, path)
                except (LookupError, ValueError, OSError) as exc:
                    results.append({"id": other_id, "ok": False, "error": _refusal(exc)})
                    continue
                # The same bytes as the kept file, so its hash and its pieces hold, for the
                # dates of the file it is now a name of.
                self.db.update_model(
                    other_id,
                    mtime=keep.mtime,
                    hashed_mtime=keep.mtime if other.hash_source == "computed" else other.hashed_mtime,
                    fingerprint=keep.fingerprint, autov1=keep.autov1,
                    sampled_mtime=keep.sampled_mtime, sniffed_mtime=None,
                )
                results.append({"id": other_id, "ok": True, "freed": freed})
            self._recount([keep_id, *others])
        for model_id in [keep_id, *others]:
            self.announce(model_id, tasks=False)
        self.inspect_soon()
        return {"results": results, "freed": sum(r.get("freed") or 0 for r in results)}

    def separate(
        self,
        model_id: int,
        progress: relocate.Progress | None = None,
        stop: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Give one name of a shared file a copy of its own again — the way back from linking.

        It takes the file's room again, so the room is asked about first. The copy keeps the
        file's dates, which keeps its hash and its pieces valid: the bytes are the same.
        """
        with self.working_on(model_id):
            model = self.get(model_id)
            path = self._present(model)
            before = links.identity(path)
            if before is None or before[1] < 2:
                return {"ok": True, "unchanged": True, "size": 0}
            size = path.stat().st_size
            free = free_bytes(path.parent)
            if free is not None and free < size + SPACE_HEADROOM:
                raise ValueError(
                    f"a copy of its own needs {_gib(size)} on the drive, and {_gib(free)} is free"
                )
            siblings = [
                m.id for m in self.db.list_models()
                if m.id != model_id and m.state == db.PRESENT and m.size == model.size
                and (links.identity(m.path) or ("",))[0] == before[0]
            ]
            links.separate(path, progress, stop)
            self._recount([model_id, *siblings])
        for other_id in [model_id, *siblings]:
            self.announce(other_id, tasks=False)
        return {"ok": True, "size": size}

    def other_names(self, model_id: int) -> dict[str, Any]:
        """The other names of this model's file, for the question before deleting it: while
        one is left, deleting this one frees nothing."""
        model = self.get(model_id)
        path = Path(model.path)
        got = links.identity(path) if model.state == db.PRESENT else None
        if got is None or got[1] < 2:
            return {"count": 0, "paths": []}
        paths = [str(p) for p in links.names(path) if path_key(p) != path_key(path)]
        return {"count": got[1] - 1, "paths": paths}

    def _recount(self, model_ids: Iterable[int]) -> None:
        for model_id in model_ids:
            model = self.db.get_model(model_id)
            if model is None or model.state != db.PRESENT:
                continue
            file_id, count = links.identity(model.path) or (None, None)
            self.db.update_model(model_id, file_id=file_id, links=count)

    def _loader_folder(self, model: Model) -> str:
        """The folder under a library folder that a model sits in — the one a loader reads:
        `insightface` for `insightface/models/buffalo_l/w600k_r50.onnx`."""
        root, relative = self.placement(model.path)
        if root is None:
            return Path(model.path).parent.name
        return relative.split("/")[0] if relative else self.roots[root].name

    def cleanup_scan(self) -> dict[str, Any]:
        """What is on disk that belongs to nothing: fragments of downloads nobody is coming
        back for, and files named after models that are not there any more."""
        places = self.places()
        roots = self.roots
        hidden = {path_key(p) for p in self.settings.exclude_dirs if str(p).strip()}
        scan = scanning.walk(roots, self.settings.exclude_dirs)
        active = {
            path_key(t.dest) for t in self.db.list()
            if t.state != db.DONE and t.dest
        }
        known = {m.key for m in self.db.list_models()}
        items: list[dict[str, Any]] = []

        suffixes = (".part.corrupt", ".part.json", ".part", ".moving",
                    links.LINKING, links.SEPARATING)
        groups: dict[str, list[Path]] = {}
        for fragment in scan.fragments:
            name = fragment.name
            for suffix in suffixes:
                if name.lower().endswith(suffix):
                    base = fragment.with_name(name[: -len(suffix)])
                    break
            else:
                continue
            if path_key(base) in active:
                continue
            groups.setdefault(path_key(base), []).append(fragment)
        for base_key, files in groups.items():
            names = {f.name.lower() for f in files}
            kind = (
                "an interrupted move" if any(n.endswith(".moving") for n in names)
                else "an interrupted copy" if any(n.endswith(links.SEPARATING) for n in names)
                # A second name of the model's own file, never swapped in: deleting it frees
                # nothing, and nothing is lost.
                else "an interrupted link" if any(n.endswith(links.LINKING) for n in names)
                else "a download that failed its checksum" if any(n.endswith(".part.corrupt") for n in names)
                else "an unfinished download"
            )
            first = files[0]
            base_name = first.name
            for suffix in suffixes:
                if base_name.lower().endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            items.append(_cleanup_item("fragment", base_name, first.parent, files, kind))

        # Names that can only belong to a model: a lone `.txt` or `.png` could be anything,
        # and is only swept up when one of these says whose it was.
        definite = (".civitai.info", ".preview.png", ".preview.jpg", ".preview.jpeg",
                    ".preview.webp", ".metadata.json", ".cm-info.json")
        model_suffixes = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".onnx")
        for directory in _visible_dirs(roots, hidden):
            try:
                names = os.listdir(directory)
            except OSError:
                continue
            lowered = {n.lower(): n for n in names}
            stems = {Path(n).stem.lower() for n in names if is_model_file(n)}
            found: dict[str, tuple[str, list[Path]]] = {}
            for name in names:
                low = name.lower()
                stem = None
                for suffix in definite:
                    if low.endswith(suffix):
                        stem = name[: -len(suffix)]
                        break
                if stem is None and low.endswith(".json"):
                    inner = name[:-5]
                    if is_model_file(inner) and inner.lower() not in lowered:
                        stem = Path(inner).stem
                if stem is None or stem.lower() in stems:
                    continue
                # A model the library still remembers — missing, not gone — keeps what it
                # left; forgetting it is where that is offered for deletion.
                if any(path_key(directory / (stem + s)) in known for s in model_suffixes):
                    continue
                found.setdefault(stem.lower(), (stem, []))[1].append(directory / name)
            for low_stem, (stem, files) in found.items():
                for suffix in (".txt", ".png", ".jpg", ".jpeg", ".webp", ".json", ".yaml"):
                    extra = lowered.get(low_stem + suffix)
                    if extra and directory / extra not in files:
                        files.append(directory / extra)
                items.append(_cleanup_item("orphan", stem, directory, files,
                                           "files named after a model that is not there"))

        sidecar_dir = places["sidecar_dir"]
        if sidecar_dir is not None and sidecar_dir.is_dir():
            library_root = places["library_root"]
            # A record whose model is in the library under another folder has lost track of
            # it rather than outlived it, and what it holds is still that model's.
            present = {m.filename.lower() for m in self.db.list_models() if m.state == db.PRESENT}
            for record in sidecar_dir.rglob("*.json"):
                # Only a record whose model can be named is judged at all: one collected flat,
                # one for a file outside the library, one mirroring a folder no longer on
                # the list — none of those says where its model should be, so none can be
                # said to have outlived it.
                model_path = sidecar.owner_of(record, sidecar_dir, library_root, roots)
                if model_path is None:
                    continue
                if not is_model_file(model_path.name):
                    continue
                if model_path.exists() or path_key(model_path) in known:
                    continue
                if model_path.name.lower() in present:
                    continue
                items.append(_cleanup_item("record", model_path.name, record.parent, [record],
                                           "a record for a model that is not there"))

        self._cleanup = {}
        for index, item in enumerate(items):
            item["id"] = index
            self._cleanup[index] = item
        # Items are numbered by this scan. A second window looking again renumbers them, and
        # a delete sent with the old numbers would take whatever the new ones are — so a
        # delete names the scan it was shown, and a stale one is refused.
        self._cleanup_token = secrets.token_hex(8)
        return {
            "items": items,
            "total": sum(i["size"] for i in items),
            "token": self._cleanup_token,
        }

    def cleanup_delete(self, ids: Iterable[int], token: str | None = None) -> erase.Erased:
        if token is None or token != getattr(self, "_cleanup_token", None):
            raise ValueError("the list changed since it was shown — look again")
        chosen = [self._cleanup[i] for i in ids if i in self._cleanup]
        paths = [Path(f["path"]) for item in chosen for f in item["files"]]
        result = erase.remove(paths)
        for item in chosen:
            self._cleanup.pop(item["id"], None)
        return result

    # --- folders --------------------------------------------------------------

    def new_folder(self, root: int, parent: str, name: str) -> Path:
        base = self._root(root)
        wanted = name.strip().rstrip(". ")
        if not wanted or wanted in {".", ".."} or relocate.FORBIDDEN.intersection(wanted) \
                or any(ord(c) < 32 for c in wanted):
            raise ValueError(f"{name!r} is not a folder name")
        relative = f"{parent.strip('/')}/{wanted}" if parent.strip("/") else wanted
        target = folders.resolve_inside(base, relative)
        if target is None:
            raise ValueError(f"{relative} is not inside {base}")
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        target.mkdir(parents=True)
        return target

    def _root(self, index: int) -> Path:
        roots = self.roots
        if not 0 <= index < len(roots):
            raise LookupError("no such library folder")
        return roots[index]

    def resolve_folder(self, root: int, relative: str) -> Path | None:
        base = self._root(root)
        if not relative.strip().strip("/"):
            return base.resolve() if base.exists() else base
        return folders.resolve_inside(base, relative)

    def move_targets(self, model_id: int, layout) -> dict[str, Any]:
        """Folders a model could be moved to, the likeliest first, across every folder of
        the library. The one it is in now is not among them."""
        model = self.get(model_id)
        return {
            "roots": self.roots_json(),
            "folders": self.rank_folders(
                layout, _category(model.category), model.base_model, model.filename,
                meta=model.meta, exclude=Path(model.path).parent, skip_model=model.id,
            ),
        }

    def rank_folders(
        self,
        layout,
        category: Category | None,
        base_model: str | None,
        filename: str | None,
        *,
        meta: dict[str, Any] | None = None,
        confidence: str | None = None,
        hint: tuple[Path, str] | None = None,
        exclude: Path | None = None,
        skip_model: int | None = None,
        limit: int = folders.LIMIT,
    ) -> list[dict[str, Any]]:
        """Every folder of the library as an answer to "where does this go?", best first.

        The layout answers from what the folders are called. The library can answer from
        what is in them, which is the better answer in a library that is actually used:
        where the other Krea 2 LoRAs are is where a Krea 2 LoRA goes, whatever that folder
        is called and whichever library folder it is in. Nearer still is where the version
        already here sits, and where the file itself was. Every row says why it is where it
        is in the list.
        """
        roots = self.roots
        if not roots:
            return []
        meta = meta or {}
        wanted = category.value if category is not None else None
        kinds = folders.kind_name(category, 2)
        rows: dict[tuple[int, str], dict[str, Any]] = {}

        def row(root: int | None, relative: str, exists: bool | None = None) -> dict[str, Any] | None:
            # The top of a library folder is never where a model goes.
            if root is None or not relative:
                return None
            key = (root, relative)
            if key not in rows:
                own = ALIASES.get(Path(relative).name.lower())
                rows[key] = {
                    "root": root, "relative": relative,
                    "category": own.value if own is not None else None,
                    "models": 0, "score": 0, "reasons": [],
                    "exists": (roots[root] / relative).is_dir() if exists is None else exists,
                }
            return rows[key]

        def say(entry: dict[str, Any] | None, points: int, reason: str) -> None:
            if entry is not None:
                entry["score"] += points
                entry["reasons"].append((points, reason))

        def folder_of(path: Path) -> tuple[int | None, str]:
            return self.placement(str(path / "_"))

        # What is on disk: every folder the walk saw, and how many models are at or below it.
        if self.last_scan is not None:
            for root, relative, count in self.last_scan.folders:
                entry = row(root, relative, True)
                if entry is not None:
                    entry["models"] = count
        else:
            # Before the first walk has finished — a moment after starting — the main folder
            # as the layout sees it is all there is to go on.
            for offered in folders.offer(layout, category, base_model, filename):
                entry = row(0, offered.relative, offered.exists)
                if entry is not None:
                    entry["models"] = offered.models

        # What the library knows is in them.
        same: dict[tuple[int, str], int] = {}
        below: dict[tuple[int, str], int] = {}
        related: set[tuple[int, str]] = set()
        version_of = meta.get("model_id") if meta.get("version_id") else None
        repo = meta.get("repo_id")
        for model in self.db.list_models():
            if model.state != db.PRESENT or model.id == skip_model:
                continue
            root, relative = self.placement(model.path)
            if root is None or not relative:
                continue
            if wanted is not None and model.category == wanted:
                parts = relative.split("/")
                for depth in range(1, len(parts) + 1):
                    key = (root, "/".join(parts[:depth]))
                    below[key] = below.get(key, 0) + 1
                if folders.same_base(model.base_model, base_model):
                    same[(root, relative)] = same.get((root, relative), 0) + 1
            if (version_of and str(model.meta.get("model_id")) == str(version_of)) or (
                repo and model.meta.get("repo_id") == repo
            ):
                related.add((root, relative))

        # Where the file itself was, and where the version already here is, outrank anything
        # the folders' contents say: the person put them there.
        if hint is not None:
            say(row(*folder_of(hint[0])), 300, hint[1])
        for key in related:
            say(row(*key), 250, "the version you have is here" if version_of
                else "from the same repository")
        for key, count in same.items():
            say(row(*key), 80 + min(count, 20), f"{count} {base_model} {folders.kind_name(category, count)} here")

        if category is not None:
            target = layout.directory_for(Verdict(category, "low", "", base_model=base_model))
            grouped = target != layout.paths.get(category, target)
            say(row(*folder_of(target)), 70,
                f"where {kinds} go" + (f", grouped by {base_model}" if grouped and base_model else ""))
            for name in layout.ambiguities.get(category, []):
                say(row(0, name), 25, f"also holds {kinds}")

        # Where the last download of this kind went: the habit, not the rule.
        for task in sorted(self.db.list([db.DONE]), key=lambda t: t.finished_at or 0, reverse=True):
            if wanted is None or task.category != wanted or not task.dest:
                continue
            if base_model and not folders.same_base(task.base_model, base_model):
                continue
            root, relative = self.placement(task.dest)
            what = f"{base_model} {folders.kind_name(category)}" if base_model else folders.kind_name(category)
            say(row(root, relative), 40, f"your last {what} went here")
            break

        # A word of the filename naming a folder is the signal left when the classifier is
        # unsure — `mystery_sam_model.pt` and the library's own `sams` — and a weaker one
        # when it is sure.
        named = 60 if confidence == "low" or category in (None, Category.OTHER) else 35
        for (root, relative), entry in rows.items():
            name = Path(relative).name
            parent = ALIASES.get(Path(relative).parent.name.lower()) if "/" in relative else None
            count = below.get((root, relative), 0)
            if category is not None and entry["category"] == wanted:
                say(entry, 30 + min(count, 10), f"holds {count} {folders.kind_name(category, count)}"
                    if count else f"the folder for {kinds}")
            elif count:
                say(entry, 10 + min(count, 10) // 2, f"holds {count} {folders.kind_name(category, count)}")
            # An existing folder for the base model, even with nothing in it yet. The one the
            # layout would make is the layout's answer already, and is not counted twice.
            if base_model and parent is category and category is not None and entry["exists"] \
                    and folders.same_base(name, base_model) and (root, relative) not in same:
                say(entry, 50, f"named after {base_model}")
            # Inside another kind's folder, a matching word is a coincidence: `krea2` under
            # text_encoders says nothing about where a Krea 2 LoRA goes.
            context = next((ALIASES[p.lower()] for p in relative.split("/") if p.lower() in ALIASES), None)
            if folders.named_in(name, filename) and (context is None or context is category):
                say(entry, named, "named in the file name")

        # Homes the layout would create for a kind that has none yet: last, but there — the
        # file may be the first of its kind in this library.
        for kind, path in layout.paths.items():
            if not path.is_dir():
                entry = row(*folder_of(path), False)
                if entry is not None and not entry["reasons"]:
                    say(entry, -20, f"the usual home for {folders.kind_name(kind, 2)} — not created yet")

        if exclude is not None:
            rows.pop(folder_of(exclude), None)
        ranked = sorted(
            rows.values(),
            key=lambda r: (-r["score"], -r["models"], r["root"], r["relative"].lower()),
        )
        answer = []
        for entry in ranked[:limit]:
            reasons = [text for _points, text in sorted(entry["reasons"], key=lambda p: -p[0])]
            if not reasons:
                count = entry["models"]
                reasons = [f"holds {count} model{'' if count == 1 else 's'}" if count
                           else "already in the library"]
            answer.append({
                "root": entry["root"], "relative": entry["relative"],
                "category": entry["category"], "models": entry["models"],
                "exists": entry["exists"], "score": entry["score"],
                "reason": " · ".join(dict.fromkeys(reasons[:2])),
            })
        return answer


# --- helpers ---------------------------------------------------------------------


def _category(value: str | None) -> Category | None:
    return Category(value) if value in Category._value2member_map_ else None


def _unchanged(model: Model, path: Path) -> bool:
    """Whether the file is still the one the library last read: its size and its date."""
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size == model.size and _same_time(stat.st_mtime, model.mtime)


def _refusal(exc: BaseException) -> str:
    # On Windows a file a program holds open cannot be replaced, and says so as a denial.
    if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32):
        return "in use by another program — close it (ComfyUI, say) and try again"
    return str(exc).strip("'\"")


def _gib(size: float) -> str:
    return f"{size / 1024**3:.1f} GB"


def _query(text: str) -> str:
    return quote_plus(text)


def _same_time(a: float | None, b: float | None) -> bool:
    """Whether a reading taken of a file at one mtime still stands for it at another."""
    return a is not None and b is not None and abs(a - b) < 1e-3


def _by_hash(members: list[Model]) -> list[tuple[str, list[Model]]]:
    """Files with one fingerprint, as groups that are certain or only likely.

    Pieces that agree are not proof, and a hash that is known is: two files whose whole
    hashes differ are two files, however alike their pieces looked. While any of them is
    unhashed, they stay together as a likely group — hashing it is what sorts them out.
    """
    if any(not m.sha256 for m in members):
        return [("likely", members)]
    hashes: dict[str, list[Model]] = {}
    for model in members:
        hashes.setdefault(str(model.sha256), []).append(model)
    return [("same", group) for group in hashes.values() if len(group) > 1]


def _physical(models: list[Model], ident: dict[int, tuple[str, int] | None]) -> set[str]:
    """The files on disk behind these paths. A hard link is one file under two names: it
    takes no more room than one, and deleting a name frees nothing."""
    found: set[str] = set()
    for model in models:
        got = ident.get(model.id)
        # A file system that numbers no files: every path counts as a file of its own.
        found.add(got[0] if got is not None else f"path:{model.id}")
    return found


def _stat_outside(model: Model) -> scanning.Found | None:
    """A model the walk did not pass, but which is still where it was — outside every
    folder of the library, or in one of the hidden ones."""
    path = Path(model.path)
    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    parts = [Path(p) for p in model.parts]
    size = stat.st_size
    if parts:
        with contextlib.suppress(OSError):
            size = sum(p.stat().st_size for p in parts if p.is_file())
    return scanning.Found(path, size, stat.st_mtime, [p for p in parts if p.is_file()])


def _outside(key: str, root_keys: list[str], hidden: list[str]) -> bool:
    if any(key.startswith(h.rstrip("\\/") + os.sep) for h in hidden):
        return True
    return not any(key.startswith(r + os.sep) for r in root_keys)


def _present_updates(model: Model, entry: scanning.Found) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if model.state != db.PRESENT:
        updates.update(state=db.PRESENT, missing_since=None)
    if model.size != entry.size:
        updates["size"] = entry.size
    if model.mtime is None or abs(model.mtime - entry.mtime) > 1e-3:
        updates["mtime"] = entry.mtime
    parts = [str(p) for p in entry.parts]
    if parts != model.parts:
        updates["parts"] = parts
    if str(entry.path) != model.path:
        updates["path"] = str(entry.path)
        updates["filename"] = entry.path.name
    if ("size" in updates or "mtime" in updates) and model.hash_source == "computed":
        updates.update(sha256=None, hash_source=None, hashed_mtime=None)
    return updates


def _download_facts(task: Task, model: Model | None) -> dict[str, Any]:
    """What a finished download knows about the file that no walk could tell."""
    facts: dict[str, Any] = {
        "origin": db.DOWNLOADED,
        "provider": task.provider or None,
        "identity": task.identity,
        "meta": task.meta,
        "category": task.category,
        "confidence": task.confidence,
        "reason": task.reason,
        "base_model": task.base_model,
    }
    if task.sha256:
        facts["sha256"] = task.sha256
        facts["hash_source"] = "download"
    if task.note:
        facts["note"] = task.note
    elif model is not None:
        facts["note"] = model.note
    return facts


def _meta_for_hash(version: dict[str, Any], digest: str, path: Path, size: int | None) -> dict[str, Any]:
    """The meta of the one file in a Civitai version whose hash is this file's."""
    from ..providers.civitai import _describe_files

    try:
        files = _describe_files(version)
    except Exception:  # noqa: BLE001
        files = []
    for entry in files:
        if entry.sha256 and entry.sha256.lower() == digest.lower():
            meta = dict(entry.meta)
            meta["file_id"] = entry.file_id
            return meta
    return details._civitai_meta(version, path, size)


def _page_url(meta: dict[str, Any]) -> str | None:
    if meta.get("model_id") and meta.get("version_id"):
        host = meta.get("host") or "civitai.com"
        return f"https://{host}/models/{meta['model_id']}?modelVersionId={meta['version_id']}"
    if meta.get("repo_id"):
        return f"https://huggingface.co/{meta['repo_id']}"
    return None


def _shard_index(first: Path) -> Path | None:
    """The index file that tells a loader how a split model's parts fit together."""
    shard = shard_of(first.name)
    if shard is None:
        return None
    index = first.with_name(shard[0] + ".index.json")
    return index if index.is_file() else None


def _cleanup_item(kind: str, name: str, folder: Path, files: list[Path], why: str) -> dict[str, Any]:
    listed = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        listed.append({"path": str(path), "name": path.name, "size": size})
    return {
        "kind": kind,
        "name": name,
        "folder": str(folder),
        "files": listed,
        "size": sum(f["size"] for f in listed),
        "why": why,
    }


def _visible_dirs(roots: list[Path], hidden: set[str]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for directory, subdirs, _files in os.walk(root):
            subdirs[:] = [
                d for d in subdirs
                if not d.startswith(".") and d not in scanning.SKIPPED_DIRS
                and path_key(Path(directory) / d) not in hidden
            ]
            yield Path(directory)
