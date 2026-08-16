"""Target disk classification.

Parallel range downloads are a clear win on SSD/NVMe and can be a net loss on a spinning
disk, where N out-of-order writers turn a sequential stream into a seek storm. HuggingFace
hit the same wall and exposed HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY for it.

So: look at what the destination actually sits on and pick the connection count from that,
with a manual override for cases we get wrong (network shares, virtual disks, RAID).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .types import DiskKind

# Get-PhysicalDisk reports SpindleSpeed as this when it does not know.
_SPINDLE_UNKNOWN = 4294967295

_QUERY = (
    "Get-Partition | Where-Object DriveLetter | ForEach-Object { "
    "$d = Get-PhysicalDisk -DeviceNumber $_.DiskNumber -ErrorAction SilentlyContinue; "
    "[pscustomobject]@{ Letter = [string]$_.DriveLetter; "
    "MediaType = [string]$d.MediaType; "
    "Spindle = [string]$d.SpindleSpeed; "
    "BusType = [string]$d.BusType } } | ConvertTo-Json -Compress"
)


@lru_cache(maxsize=1)
def _drive_table() -> dict[str, DiskKind]:
    """Map drive letter -> DiskKind. Cached; the answer does not change while we run."""
    if sys.platform != "win32":
        return {}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _QUERY],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    if isinstance(rows, dict):  # ConvertTo-Json emits a bare object for a single row
        rows = [rows]

    table: dict[str, DiskKind] = {}
    for row in rows:
        letter = (row.get("Letter") or "").strip().upper()
        if not letter:
            continue
        table[letter] = _classify(
            media=(row.get("MediaType") or "").strip().lower(),
            spindle=(row.get("Spindle") or "").strip(),
            bus=(row.get("BusType") or "").strip().lower(),
        )
    return table


def _classify(media: str, spindle: str, bus: str) -> DiskKind:
    if media == "ssd":
        return DiskKind.SSD
    if media == "hdd":
        return DiskKind.HDD

    # Some drives report a real spindle speed even when MediaType is "Unspecified".
    try:
        rpm = int(spindle)
    except (TypeError, ValueError):
        rpm = 0
    if rpm and rpm != _SPINDLE_UNKNOWN:
        return DiskKind.HDD

    if bus == "nvme":
        return DiskKind.SSD

    # Everything else is genuinely ambiguous and must stay that way. Plenty of SATA disks —
    # spinning and solid state alike — report MediaType "Unspecified" with SpindleSpeed 0,
    # so there is nothing left to tell them apart. Calling those HDD would throttle healthy
    # SSDs; calling them SSD would let a mechanical disk get hammered by parallel writers.
    # UNKNOWN takes the middle road, and the user can pin it explicitly in settings.
    return DiskKind.UNKNOWN


def detect_disk_kind(path: Path | str) -> DiskKind:
    """Classify the physical media backing `path` (which need not exist yet)."""
    try:
        drive = Path(path).resolve().drive  # "F:" for local paths, r"\\srv\share" for UNC
    except OSError:
        return DiskKind.UNKNOWN
    if not drive or not drive.endswith(":"):
        return DiskKind.UNKNOWN  # UNC / mapped network path — do not guess
    return _drive_table().get(drive[0].upper(), DiskKind.UNKNOWN)


def free_bytes(path: Path | str) -> int | None:
    """Free space on the volume `path` would land on, or None when it cannot be told.

    The directory does not have to exist yet — a library root is created on the first
    download — so this walks up to the nearest parent that does.
    """
    candidate = Path(path).resolve()
    while True:
        try:
            return shutil.disk_usage(candidate).free
        except OSError:
            if candidate.parent == candidate:
                return None
            candidate = candidate.parent


def recommend_connections(
    path: Path | str,
    requested: int,
    size: int | None = None,
    kind: DiskKind | None = None,
) -> int:
    """Clamp the requested connection count to what the destination can usefully absorb.

    Pass `kind` to override detection — the honest answer for a lot of SATA hardware is
    that Windows does not know, and the user does.
    """
    n = max(1, requested)

    if kind is None:
        kind = detect_disk_kind(path)
    if kind is DiskKind.HDD:
        # Two streams still hide latency; beyond that the head thrashes.
        n = min(n, 2)
    elif kind is DiskKind.UNKNOWN:
        # Might be mechanical, so do not go wide — but not so narrow that a bad CDN edge
        # can halve the download either. Pinning the disk type in settings beats this guess.
        n = min(n, 6)

    # No point opening eight connections for a file that fits in one chunk.
    if size:
        from .chunks import pick_chunk_size

        n = min(n, max(1, size // pick_chunk_size(size)))
    return max(1, n)
