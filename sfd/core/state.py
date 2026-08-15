"""Sidecar state for an in-flight download.

State lives in `<target>.part.json`, right next to `<target>.part`. Keeping it beside the
data rather than only in a central database means a `.part` file stays resumable even if
it is moved to another machine or the queue database is lost.

Writes are atomic (temp file + os.replace) and debounced, so a crash mid-download leaves
either the previous consistent state or the new one — never a half-written file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunks import ChunkMap
from .errors import RemoteChanged
from .types import FileIdentity, RemoteFileInfo

FORMAT_VERSION = 2
FLUSH_INTERVAL = 1.0


@dataclass(slots=True)
class PartState:
    path: Path                  # the .part.json path
    identity_key: str
    size: int
    etag: str | None
    sha256: str | None
    chunk_map: ChunkMap
    created_at: float
    _last_flush: float = 0.0
    _dirty: bool = False

    # --- construction -----------------------------------------------------

    @classmethod
    def sidecar_path(cls, part_path: Path) -> Path:
        return part_path.with_name(part_path.name + ".json")

    @classmethod
    def create(
        cls,
        part_path: Path,
        identity: FileIdentity,
        info: RemoteFileInfo,
        chunk_map: ChunkMap,
    ) -> PartState:
        return cls(
            path=cls.sidecar_path(part_path),
            identity_key=identity.key(),
            size=info.size or 0,
            etag=info.etag,
            sha256=info.sha256,
            chunk_map=chunk_map,
            created_at=time.time(),
            _dirty=True,
        )

    @classmethod
    def load(cls, part_path: Path) -> PartState | None:
        sidecar = cls.sidecar_path(part_path)
        if not sidecar.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(sidecar.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("version") != FORMAT_VERSION:
            return None
        try:
            chunk_map = ChunkMap.from_dict(data["chunks"])
        except (KeyError, ValueError):
            return None
        return cls(
            path=sidecar,
            identity_key=data.get("identity", ""),
            size=int(data.get("size", 0)),
            etag=data.get("etag"),
            sha256=data.get("sha256"),
            chunk_map=chunk_map,
            created_at=float(data.get("created_at", time.time())),
        )

    # --- validation -------------------------------------------------------

    def check_resumable(self, identity: FileIdentity, info: RemoteFileInfo, part_path: Path) -> None:
        """Raise RemoteChanged if these bytes cannot be trusted as a prefix of `info`.

        Better to restart a 20 GB download than to hand back a file that is a splice of two
        different model revisions and fails its hash check an hour later.
        """
        if self.identity_key != identity.key():
            raise RemoteChanged("state file belongs to a different source")
        if info.size is not None and self.size != info.size:
            raise RemoteChanged(f"size changed: {self.size} -> {info.size}")
        if info.etag and self.etag and info.etag != self.etag:
            raise RemoteChanged(f"etag changed: {self.etag} -> {info.etag}")

        # The state file claims bytes that the .part file cannot physically hold.
        actual = part_path.stat().st_size if part_path.exists() else 0
        if actual < self.chunk_map.completed_prefix():
            raise RemoteChanged("part file is shorter than the recorded progress")

    # --- persistence ------------------------------------------------------

    def touch(self) -> None:
        self._dirty = True

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and (not self._dirty or now - self._last_flush < FLUSH_INTERVAL):
            return
        payload = {
            "version": FORMAT_VERSION,
            "identity": self.identity_key,
            "size": self.size,
            "etag": self.etag,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "updated_at": now,
            "chunks": self.chunk_map.to_dict(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), "utf-8")
        os.replace(tmp, self.path)
        self._last_flush = now
        self._dirty = False

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
