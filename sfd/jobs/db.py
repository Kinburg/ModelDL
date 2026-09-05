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

import json
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
    retry_at      REAL
);
"""

# Indexes are created separately, after migration. `CREATE TABLE IF NOT EXISTS` is a no-op
# against an existing table, so an index on a column that only the migration adds would try
# to build before that column exists — and take the whole queue down on upgrade.
INDEXES = """
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS tasks_position ON tasks(position);
-- One row per remote file: adding the same thing twice should adopt the existing task
-- rather than race another download onto the same path.
CREATE UNIQUE INDEX IF NOT EXISTS tasks_identity ON tasks(identity);
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
        }
        if self.size:
            data["fraction"] = min(1.0, self.downloaded / self.size)
        return data


class Database:
    def __init__(self, path: Path | str = "queue.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.executescript(INDEXES)
            # A task marked running when the process died is not running now.
            self._conn.execute(
                "UPDATE tasks SET state = ? WHERE state = ?", (PENDING, RUNNING)
            )
            self._conn.commit()

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
        }
        for column, definition in additions.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")

        # Backfill, or the queue misbehaves in a way that looks arbitrary: with every
        # existing row at NULL, MAX(position) is NULL, the next task is handed position 1.0,
        # and it sorts ahead of everything already waiting.
        self._conn.execute("UPDATE tasks SET position = id WHERE position IS NULL")
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
        }
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
    )
