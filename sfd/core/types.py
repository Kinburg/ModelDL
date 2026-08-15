"""Core value types.

The central idea of this package lives here: `FileIdentity` (what we want) is a separate
thing from `ResolvedTarget` (where we can currently get it). The identity is stable and
gets persisted; the resolved URL is a disposable that we re-mint on every reconnect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiskKind(str, Enum):
    SSD = "ssd"
    HDD = "hdd"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Canonical, provider-independent reference to one remote file.

    Never contains a signed URL. `ref` holds whatever the provider needs to mint a fresh
    one: (repo, revision, path) for HuggingFace, (versionId, fileId) for Civitai.
    """

    provider: str
    ref: dict[str, Any]

    def key(self) -> str:
        parts = [self.provider] + [f"{k}={self.ref[k]}" for k in sorted(self.ref)]
        return "|".join(parts)


@dataclass(slots=True)
class RemoteFileInfo:
    """What the provider knows about the file before any bytes move."""

    filename: str
    size: int | None = None
    sha256: str | None = None
    etag: str | None = None
    accept_ranges: bool = True
    content_type: str | None = None
    # Free-form provider metadata (Civitai model type, baseModel, trainedWords, HF tags).
    # Used later for library placement; the transfer layer ignores it.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedTarget:
    """A short-lived, ready-to-fetch URL.

    `headers` are the headers to send *to this URL specifically*. For signed CDN targets
    this is normally empty — the credentials are baked into the query string, and sending
    an Authorization header alongside them makes S3-compatible storage reject the request
    outright.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    expires_at: float | None = None
    info: RemoteFileInfo | None = None

    def is_stale(self, margin: float = 60.0) -> bool:
        """True when the signature is close enough to expiry that a long read would outlive it."""
        if self.expires_at is None:
            return False
        return time.time() + margin >= self.expires_at


@dataclass(slots=True)
class ProgressSnapshot:
    downloaded: int
    total: int | None
    speed: float          # bytes/sec, smoothed
    connections: int
    hashed: int           # bytes fed to the hasher so far
    eta: float | None     # seconds

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return self.downloaded / self.total
