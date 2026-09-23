"""The queue's storage.

SQLite, because the queue has to survive closing the app, a crash and a reboot — a download
that has to be restarted from the beginning because the manager forgot about it is the
problem this project exists to avoid.

The per-file resume state still lives in `<file>.part.json` next to the data, deliberately
duplicated. This database is the queue; that sidecar is the file's own record of itself, and
a `.part` stays resumable even if this database is deleted.

Calls are synchronous. They are sub-millisecond against a local file, and a queue of a few
hundred rows does not justify the complexity of an async driver.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from ..library import previews, sidecar

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    state         TEXT NOT NULL,
    source        TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    provider      TEXT NOT NULL,
    identity      TEXT NOT NULL,
    filename      TEXT NOT NULL DEFAULT '',
    size          INTEGER,
    sha256        TEXT,
    dest          TEXT NOT NULL DEFAULT '',
    downloaded    INTEGER NOT NULL DEFAULT 0,
    category      TEXT,
    confidence    TEXT,
    reason        TEXT,
    disagreement  TEXT,
    base_model    TEXT,
    meta          TEXT NOT NULL DEFAULT '{}',
    error         TEXT,
    -- Manual ordering, so the queue can be dragged into the order you actually want.
    position      REAL,
    -- Wall-clock facts about the run. `transferred` is bytes this machine actually pulled,
    -- which is not the file size: a resumed download moves only the remainder, and average
    -- speed computed from size would flatter it.
    started_at    REAL,
    finished_at   REAL,
    transferred   INTEGER NOT NULL DEFAULT 0,
    -- Automatic retries. `attempts` counts the ones already spent, `retry_at` is when the
    -- next one is due; both are cleared when a person presses Retry themselves.
    attempts      INTEGER NOT NULL DEFAULT 0,
    retry_at      REAL,
    -- What you wrote about the file yourself. A cache of the `note` in the `.json` record,
    -- which is where it actually lives: the row can be taken out of the history and the
    -- record is not, and a note that survives only until the list is tidied is no note.
    note          TEXT,
    -- The model in the library this download produced. A finished task is the history of
    -- a model, not the model: the model can be renamed, moved or deleted, and the history
    -- has to keep saying what arrived and when.
    model_id      INTEGER,
    -- Taken off the Downloads list by `Clear finished`, still in the history.
    archived      INTEGER NOT NULL DEFAULT 0,
    -- The name the file arrived under, before any rename. The name a service gave a file
    -- is what anyone searching that service for it again will type.
    original_filename TEXT,
    -- What became of a finished download whose model is no longer in the library:
    -- 'deleted' when its files were deleted here, 'forgotten' when it went missing and was
    -- taken out of the library.
    fate          TEXT
);

-- The library: every model file the app knows about, downloaded here or found on disk.
-- A cache of the disk, rebuilt by scanning; what cannot be rebuilt (the note) lives in the
-- model's `.json` record, and this table only carries a copy of it.
CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- The path as the file system compares it, so `Loras/A.safetensors` and
    -- `loras/a.safetensors` are one model on Windows rather than two.
    key           TEXT NOT NULL,
    path          TEXT NOT NULL,
    filename      TEXT NOT NULL,
    size          INTEGER,
    mtime         REAL,
    state         TEXT NOT NULL,
    origin        TEXT NOT NULL,
    provider      TEXT,
    identity      TEXT NOT NULL DEFAULT '{}',
    meta          TEXT NOT NULL DEFAULT '{}',
    sha256        TEXT,
    -- Where the hash came from: 'download' (what the service advertised and the transfer
    -- verified) or 'computed' (read off the file here, valid for `hashed_mtime`).
    hash_source   TEXT,
    hashed_mtime  REAL,
    category      TEXT,
    confidence    TEXT,
    reason        TEXT,
    base_model    TEXT,
    -- What the file's own header says, read once per version of the file.
    header        TEXT NOT NULL DEFAULT '{}',
    sniffed_mtime REAL,
    -- What other tools left beside it: a `.civitai.info`, an A1111 description, a picture.
    extras        TEXT NOT NULL DEFAULT '{}',
    note          TEXT,
    -- Where its `.json` record was last found, which after a move made in Explorer is not
    -- necessarily where it would be written today.
    record        TEXT,
    -- Every file of a model split into shards, `path` being the first of them.
    parts         TEXT NOT NULL DEFAULT '[]',
    -- Files named after the model that stayed behind when it was moved outside the app.
    left_behind   TEXT NOT NULL DEFAULT '[]',
    -- The last attempt to identify it on Civitai, and the last check for a newer version.
    lookup        TEXT NOT NULL DEFAULT '{}',
    updates       TEXT NOT NULL DEFAULT '{}',
    first_seen    REAL NOT NULL,
    last_seen     REAL,
    missing_since REAL,
    updated_at    REAL NOT NULL
);
"""

# Indexes are created separately, after migration. `CREATE TABLE IF NOT EXISTS` is a no-op
# against an existing table, so an index on a column that only the migration adds would try
# to build before that column exists — and take the whole queue down on upgrade.
INDEXES = """
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS tasks_position ON tasks(position);
CREATE INDEX IF NOT EXISTS tasks_model ON tasks(model_id);
-- One unfinished row per remote file: adding the same thing twice should adopt the task
-- already under way rather than race another download onto the same path. Finished rows
-- are history, and a file can have been downloaded more than once — deleted in March,
-- fetched again in May — so they are left out of the rule.
CREATE UNIQUE INDEX IF NOT EXISTS tasks_identity_open ON tasks(identity) WHERE state != 'done';
CREATE UNIQUE INDEX IF NOT EXISTS models_key ON models(key);
CREATE INDEX IF NOT EXISTS models_sha256 ON models(sha256);
"""

# The states a task moves through. `blocked` means classification was not confident enough
# to file the model on its own; it waits for a person rather than guessing.
PENDING = "pending"
BLOCKED = "blocked"
RUNNING = "running"
PAUSED = "paused"
DONE = "done"
FAILED = "failed"

ACTIVE_STATES = (PENDING, RUNNING)

# Which end of the queue a newly added task joins — `queue_position` in the settings.
TOP = "top"
BOTTOM = "bottom"

# Whether a model in the library is where the app last saw it.
PRESENT = "present"
MISSING = "missing"

# How a model came to be in the library.
DOWNLOADED = "downloaded"
FOUND = "found"

# What became of a finished download whose model left the library.
DELETED = "deleted"
FORGOTTEN = "forgotten"


@dataclass(slots=True)
class Task:
    id: int
    created_at: float
    updated_at: float
    state: str
    source: str
    label: str
    provider: str
    identity: dict[str, Any]
    filename: str
    size: int | None
    sha256: str | None
    dest: str
    downloaded: int
    category: str | None
    confidence: str | None
    reason: str | None
    disagreement: str | None
    base_model: str | None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    position: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    transferred: int = 0
    attempts: int = 0
    retry_at: float | None = None
    note: str | None = None
    model_id: int | None = None
    archived: bool = False
    original_filename: str | None = None
    fate: str | None = None

    @property
    def duration(self) -> float | None:
        """How long the last run actually took, once it finished."""
        if self.started_at and self.finished_at and self.finished_at > self.started_at:
            return self.finished_at - self.started_at
        return None

    @property
    def average_speed(self) -> float | None:
        """Bytes per second over the run, measured on bytes really fetched.

        A skipped or resumed file would otherwise report a nonsense rate: dividing the full
        size by the seconds spent verifying an existing copy once produced "364 MB/s".
        """
        duration = self.duration
        if duration and self.transferred:
            return self.transferred / duration
        return None

    def to_json(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "state": self.state,
            "source": self.source,
            "label": self.label,
            "provider": self.provider,
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
            "dest": self.dest,
            "downloaded": self.downloaded,
            "category": self.category,
            "confidence": self.confidence,
            "reason": self.reason,
            "disagreement": self.disagreement,
            "base_model": self.base_model,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "position": self.position,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "transferred": self.transferred,
            "attempts": self.attempts,
            "retry_at": self.retry_at,
            "note": self.note,
            "duration": self.duration,
            "average_speed": self.average_speed,
            # Split and tidied the same way the `.txt` beside the model is, because the page
            # offers them for copying and the two must not disagree. Civitai often returns
            # them already comma-joined inside one string, which as one chip on a card reads
            # as a trigger word with commas in it.
            "trigger_words": sidecar.normalise_triggers(self.meta.get("trained_words")),
            # Which service this came off, short enough for the header of a card. The
            # provider name alone is in `provider`; this is the domain, which is the part
            # a person recognises.
            "origin": _origin(self.provider, self.identity, self.meta),
            # A count, not the pictures and not their prompts: this payload is sent again
            # on every state change of every task, and a queue of two hundred models would
            # be carrying two hundred prompt collections through it. The page asks for the
            # details of the one it is showing.
            "previews": len(previews.entries(self.meta)),
            "nsfw": bool(self.meta.get("nsfw")),
            "model_id": self.model_id,
            "archived": self.archived,
            "original_filename": self.original_filename or self.filename,
            "fate": self.fate,
            "model_name": self.meta.get("model_name"),
            "version_name": self.meta.get("version_name"),
        }
        if self.size:
            data["fraction"] = min(1.0, self.downloaded / self.size)
        return data


@dataclass(slots=True)
class Model:
    """One model file in the library, wherever it came from."""

    id: int
    key: str
    path: str
    filename: str
    size: int | None
    mtime: float | None
    state: str
    origin: str
    provider: str | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    hash_source: str | None = None
    hashed_mtime: float | None = None
    category: str | None = None
    confidence: str | None = None
    reason: str | None = None
    base_model: str | None = None
    header: dict[str, Any] = field(default_factory=dict)
    sniffed_mtime: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    record: str | None = None
    parts: list[str] = field(default_factory=list)
    left_behind: list[str] = field(default_factory=list)
    lookup: dict[str, Any] = field(default_factory=dict)
    updates: dict[str, Any] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float | None = None
    missing_since: float | None = None
    updated_at: float = 0.0

    @property
    def identified(self) -> bool:
        """Whether a service has said what this is: downloaded from one, or looked up."""
        return bool(
            self.meta.get("version_id") or self.meta.get("repo_id") or self.meta.get("model_name")
        )

    @property
    def title(self) -> str | None:
        """The model's own name, as opposed to its file's."""
        return (
            self.meta.get("model_name")
            or self.extras.get("model_name")
            or self.header.get("title")
            or None
        )

    @property
    def trigger_words(self) -> list[str]:
        """What wakes it up, from the most trustworthy source that says.

        The service first — that is what the uploader published. Then what other tools
        wrote beside it (A1111's activation text), then what the trainer put in the header.
        """
        for source in (
            self.meta.get("trained_words"),
            self.extras.get("trained_words"),
            self.extras.get("activation_text"),
            self.header.get("trigger_phrase"),
        ):
            words = sidecar.normalise_triggers(source)
            if words:
                return words
        return []

    @property
    def preview_count(self) -> int:
        remote = len(previews.entries(self.meta))
        if remote:
            return remote
        return 1 if (self.extras.get("image") or self.header.get("thumbnail")) else 0

    def to_json(self) -> dict[str, Any]:
        """What the page lists. Deliberately light: this goes out for every model in the
        library, and the prompts, the header and the record are fetched for the one that
        is being looked at."""
        precision = self.header.get("precision") or self.meta.get("precision")
        return {
            "id": self.id,
            "path": self.path,
            "folder": str(Path(self.path).parent),
            "filename": self.filename,
            "size": self.size,
            "mtime": self.mtime,
            "state": self.state,
            "origin": self.origin,
            "provider": self.provider,
            "host": _origin(self.provider or "", self.identity, self.meta)
            if self.provider else "",
            "identified": self.identified,
            "category": self.category,
            "confidence": self.confidence,
            "base_model": self.base_model,
            "title": self.title,
            "version_name": self.meta.get("version_name") or self.extras.get("version_name"),
            "note": self.note,
            "trigger_words": self.trigger_words,
            "previews": self.preview_count,
            "nsfw": bool(self.meta.get("nsfw") or self.extras.get("nsfw")),
            "sha256": self.sha256,
            "hash_source": self.hash_source,
            "format": self.header.get("format"),
            "precision": precision,
            "parts": len(self.parts),
            "left_behind": len(self.left_behind),
            "has_record": bool(self.record),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "missing_since": self.missing_since,
            "lookup": self.lookup.get("result"),
            "update": self.updates if self.updates.get("available") else None,
        }


# Columns of `models` stored as JSON, and what an empty one reads back as.
_MODEL_JSON = {
    "identity": dict, "meta": dict, "header": dict, "extras": dict,
    "parts": list, "left_behind": list, "lookup": dict, "updates": dict,
}


def path_key(path: Path | str) -> str:
    """How the file system compares two spellings of a path."""
    return os.path.normcase(os.path.abspath(str(path)))


class Database:
    def __init__(self, path: Path | str = "queue.db") -> None:
        self._lock = threading.Lock()
        self._path = str(path)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._keep_a_copy_before_the_library()
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.executescript(INDEXES)
            # A task marked running when the process died is not running now.
            self._conn.execute(
                "UPDATE tasks SET state = ? WHERE state = ?", (PENDING, RUNNING)
            )
            self._conn.commit()

    def _keep_a_copy_before_the_library(self) -> None:
        """Copy a queue from before the library aside, once, before changing it.

        The change cannot be undone by an older build: it drops the rule that a file is
        queued once ever, and once a file has been downloaded twice that rule can no longer
        be put back — an older build opening this database would fail to start. The copy
        is what that build can be pointed at instead.
        """
        old = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'tasks_identity'"
        ).fetchone()
        if old is None or self._path in ("", ":memory:"):
            return
        target = Path(self._path + ".before-library.bak")
        if target.exists():
            return
        with contextlib.suppress(sqlite3.Error, OSError):
            copy = sqlite3.connect(str(target))
            try:
                self._conn.backup(copy)
            finally:
                copy.close()

    def _migrate(self) -> None:
        """Add columns a database from an older build is missing.

        Dropping and recreating would throw away a queue, which is the one thing this table
        exists to protect.
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(tasks)")}
        additions = {
            "position": "REAL",
            "started_at": "REAL",
            "finished_at": "REAL",
            "transferred": "INTEGER NOT NULL DEFAULT 0",
            "attempts": "INTEGER NOT NULL DEFAULT 0",
            "retry_at": "REAL",
            "note": "TEXT",
            "model_id": "INTEGER",
            "archived": "INTEGER NOT NULL DEFAULT 0",
            "original_filename": "TEXT",
            "fate": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")

        # Backfill, or the queue misbehaves in a way that looks arbitrary: with every
        # existing row at NULL, MAX(position) is NULL, the next task is handed position 1.0,
        # and it sorts ahead of everything already waiting.
        self._conn.execute("UPDATE tasks SET position = id WHERE position IS NULL")
        # The name a file had when this build first sees it is the best record there is of
        # the name it arrived under; a rename made before now left no other trace.
        self._conn.execute(
            "UPDATE tasks SET original_filename = filename WHERE original_filename IS NULL"
        )
        # The old rule — one row per remote file, ever — made a finished download block the
        # same file from being fetched again once it was deleted. It is replaced by one that
        # only covers the rows still under way.
        self._conn.execute("DROP INDEX IF EXISTS tasks_identity")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes -----------------------------------------------------------

    def add(self, **values: Any) -> Task | None:
        """Insert a task. Returns None when this exact file is already queued."""
        now = time.time()
        mode = values.pop("position_mode", BOTTOM)
        position = values.pop("position", None)
        if position is None:
            position = self.reserve_positions(1, mode)[0]

        payload = {
            "created_at": now,
            "updated_at": now,
            "state": values.pop("state", PENDING),
            "source": values.pop("source", ""),
            "label": values.pop("label", ""),
            "provider": values.pop("provider", ""),
            "identity": json.dumps(values.pop("identity", {}), sort_keys=True),
            "filename": values.pop("filename", ""),
            "size": values.pop("size", None),
            "sha256": values.pop("sha256", None),
            "dest": str(values.pop("dest", "")),
            "downloaded": 0,
            "category": values.pop("category", None),
            "confidence": values.pop("confidence", None),
            "reason": values.pop("reason", None),
            "disagreement": values.pop("disagreement", None),
            "base_model": values.pop("base_model", None),
            "meta": json.dumps(values.pop("meta", {}), ensure_ascii=False),
            "error": None,
            # Fractional positions let a row be dropped between two others without
            # renumbering the whole queue.
            "position": position,
            "note": values.pop("note", None),
            "model_id": values.pop("model_id", None),
        }
        payload["original_filename"] = values.pop("original_filename", None) or payload["filename"]
        columns = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        with self._lock:
            try:
                cursor = self._conn.execute(
                    f"INSERT INTO tasks ({columns}) VALUES ({marks})",
                    tuple(payload.values()),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                return None
            task_id = cursor.lastrowid
        return self.get(task_id)

    def reserve_positions(self, count: int, mode: str = BOTTOM) -> list[float]:
        """Hand out `count` consecutive slots at one end of the queue, in order.

        A batch has to be reserved in one go rather than a slot at a time. Adding to the top
        means taking the slot above whatever is first right now, so one link expanding into
        five files would put each one above the last and land the whole batch reversed.
        """
        if count <= 0:
            return []
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(position) AS lo, MAX(position) AS hi FROM tasks"
            ).fetchone()
        lowest = row["lo"] if row and row["lo"] is not None else 0.0
        highest = row["hi"] if row and row["hi"] is not None else 0.0
        first = lowest - count if mode == TOP else highest + 1.0
        return [first + offset for offset in range(count)]

    def reorder(self, ids: list[int]) -> None:
        """Apply an explicit order, as dragged in the UI."""
        with self._lock:
            for index, task_id in enumerate(ids):
                self._conn.execute(
                    "UPDATE tasks SET position = ? WHERE id = ?", (float(index), task_id)
                )
            self._conn.commit()

    def update(self, task_id: int, **values: Any) -> None:
        if not values:
            return
        for key in ("identity", "meta"):
            if key in values and not isinstance(values[key], str):
                values[key] = json.dumps(values[key], ensure_ascii=False)
        if "dest" in values:
            values["dest"] = str(values["dest"])
        values["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            self._conn.execute(
                f"UPDATE tasks SET {assignments} WHERE id = ?",
                (*values.values(), task_id),
            )
            self._conn.commit()

    def delete(self, task_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self._conn.commit()

    def clear(self, states: Iterable[str]) -> int:
        states = tuple(states)
        if not states:
            return 0
        marks = ", ".join("?" for _ in states)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM tasks WHERE state IN ({marks})", states
            )
            self._conn.commit()
            return cursor.rowcount

    def archive_finished(self) -> int:
        """Take every finished download off the Downloads list. The history keeps them."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE tasks SET archived = 1, updated_at = ? "
                "WHERE state = ? AND archived = 0",
                (time.time(), DONE),
            )
            self._conn.commit()
            return cursor.rowcount

    def tasks_for_model(self, model_id: int) -> list[Task]:
        """Every download that produced this model, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE model_id = ? "
                "ORDER BY COALESCE(finished_at, created_at) DESC, id DESC",
                (model_id,),
            ).fetchall()
        return [_to_task(row) for row in rows]

    def finished_with_identity(self, identity: dict[str, Any]) -> list[Task]:
        """Finished downloads of this remote file — the history of one link."""
        encoded = json.dumps(identity, sort_keys=True)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE identity = ? AND state = ? ORDER BY id DESC",
                (encoded, DONE),
            ).fetchall()
        return [_to_task(row) for row in rows]

    # --- the library ------------------------------------------------------

    def add_model(self, **values: Any) -> Model | None:
        """Insert a model. None when a model at that path is already known."""
        now = time.time()
        path = str(values.pop("path"))
        payload: dict[str, Any] = {
            "key": path_key(path),
            "path": path,
            "filename": values.pop("filename", None) or Path(path).name,
            "state": values.pop("state", PRESENT),
            "origin": values.pop("origin", FOUND),
            "first_seen": values.pop("first_seen", now),
            "updated_at": now,
        }
        payload.update(values)
        payload = _encode_model(payload)
        columns = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        with self._lock:
            try:
                cursor = self._conn.execute(
                    f"INSERT INTO models ({columns}) VALUES ({marks})", tuple(payload.values())
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                return None
            model_id = cursor.lastrowid
        return self.get_model(model_id)

    def update_model(self, model_id: int, **values: Any) -> None:
        if not values:
            return
        if "path" in values:
            values["path"] = str(values["path"])
            values["key"] = path_key(values["path"])
        values["updated_at"] = time.time()
        values = _encode_model(values)
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            self._conn.execute(
                f"UPDATE models SET {assignments} WHERE id = ?", (*values.values(), model_id)
            )
            self._conn.commit()

    def delete_model(self, model_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            self._conn.commit()

    def get_model(self, model_id: int) -> Model | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return _to_model(row) if row else None

    def model_at(self, path: Path | str) -> Model | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM models WHERE key = ?", (path_key(path),)
            ).fetchone()
        return _to_model(row) if row else None

    def list_models(self) -> list[Model]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM models ORDER BY id").fetchall()
        return [_to_model(row) for row in rows]

    def models_with_hash(self, sha256: str) -> list[Model]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM models WHERE sha256 = ? ORDER BY id", (sha256.lower(),)
            ).fetchall()
        return [_to_model(row) for row in rows]

    def retarget_tasks(self, model_id: int, **values: Any) -> None:
        """Carry a change to a model over to every download that produced it."""
        if not values:
            return
        if "dest" in values:
            values["dest"] = str(values["dest"])
        values["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            self._conn.execute(
                f"UPDATE tasks SET {assignments} WHERE model_id = ?", (*values.values(), model_id)
            )
            self._conn.commit()

    # --- reads ------------------------------------------------------------

    def get(self, task_id: int) -> Task | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _to_task(row) if row else None

    def list(self, states: Iterable[str] | None = None) -> list[Task]:
        query = "SELECT * FROM tasks"
        params: tuple[Any, ...] = ()
        if states:
            states = tuple(states)
            query += f" WHERE state IN ({', '.join('?' for _ in states)})"
            params = states
        query += " ORDER BY COALESCE(position, id), id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_to_task(row) for row in rows]

    def due_retries(self, now: float) -> list[Task]:
        """Failed tasks whose next automatic attempt has come round."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE state = ? AND retry_at IS NOT NULL "
                "AND retry_at <= ? ORDER BY COALESCE(position, id), id",
                (FAILED, now),
            ).fetchall()
        return [_to_task(row) for row in rows]

    def claim_next(self) -> Task | None:
        """Take the oldest pending task and mark it running, atomically."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE state = ? "
                "ORDER BY COALESCE(position, id), id LIMIT 1",
                (PENDING,),
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            self._conn.execute(
                "UPDATE tasks SET state = ?, updated_at = ?, started_at = ?, "
                "finished_at = NULL, error = NULL WHERE id = ?",
                (RUNNING, now, now, row["id"]),
            )
            self._conn.commit()
        task = _to_task(row)
        task.state = RUNNING
        return task


def _origin(provider: str, identity: dict[str, Any], meta: dict[str, Any]) -> str:
    """Where a file came from, in the few characters a card header has room for.

    The service's own domain rather than our provider name: `civitai.red` is a mirror of
    Civitai, and calling it "civitai" would hide the one thing that explains why this
    download talks to a different host than the row above it. A direct link has no service
    to name, so it is named by its host — which is what anyone reading the queue is looking
    for anyway.

    Deliberately not a URL. The identity holds no link by design, and this is a label, not
    somewhere to click: the page link belongs to the record, which has one already.
    """
    if provider == "civitai":
        return str(meta.get("host") or "civitai.com")
    if provider == "huggingface":
        return "huggingface.co"
    url = (identity.get("ref") or {}).get("url")
    if isinstance(url, str) and url:
        # Userinfo and port are noise here, and a `user:pass@` left in would put a
        # credential on screen next to the filename.
        host = urlsplit(url).netloc.rsplit("@", 1)[-1].split(":")[0]
        if host:
            return host.removeprefix("www.")
    return provider


def _to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        state=row["state"],
        source=row["source"],
        label=row["label"],
        provider=row["provider"],
        identity=json.loads(row["identity"] or "{}"),
        filename=row["filename"],
        size=row["size"],
        sha256=row["sha256"],
        dest=row["dest"],
        downloaded=row["downloaded"],
        category=row["category"],
        confidence=row["confidence"],
        reason=row["reason"],
        disagreement=row["disagreement"],
        base_model=row["base_model"],
        meta=json.loads(row["meta"] or "{}"),
        error=row["error"],
        position=row["position"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        transferred=row["transferred"] or 0,
        attempts=row["attempts"] or 0,
        retry_at=row["retry_at"],
        note=row["note"],
        model_id=row["model_id"],
        archived=bool(row["archived"]),
        original_filename=row["original_filename"],
        fate=row["fate"],
    )


def _encode_model(values: dict[str, Any]) -> dict[str, Any]:
    encoded = dict(values)
    for key in _MODEL_JSON:
        if key in encoded and not isinstance(encoded[key], str):
            encoded[key] = json.dumps(encoded[key] or _MODEL_JSON[key](), ensure_ascii=False)
    if encoded.get("sha256"):
        encoded["sha256"] = str(encoded["sha256"]).lower()
    return encoded


def _to_model(row: sqlite3.Row) -> Model:
    decoded: dict[str, Any] = {}
    for key, kind in _MODEL_JSON.items():
        try:
            value = json.loads(row[key] or "null")
        except (TypeError, ValueError):
            value = None
        decoded[key] = value if isinstance(value, kind) else kind()
    return Model(
        id=row["id"],
        key=row["key"],
        path=row["path"],
        filename=row["filename"],
        size=row["size"],
        mtime=row["mtime"],
        state=row["state"],
        origin=row["origin"],
        provider=row["provider"],
        sha256=row["sha256"],
        hash_source=row["hash_source"],
        hashed_mtime=row["hashed_mtime"],
        category=row["category"],
        confidence=row["confidence"],
        reason=row["reason"],
        base_model=row["base_model"],
        sniffed_mtime=row["sniffed_mtime"],
        note=row["note"],
        record=row["record"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        missing_since=row["missing_since"],
        updated_at=row["updated_at"],
        **decoded,
    )
