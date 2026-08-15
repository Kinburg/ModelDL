"""The child process that runs HuggingFace's own client.

Deliberately a separate process rather than a library call, for three reasons that all come
from the same place — `huggingface_hub` reads its configuration once, at import:

  * **Settings only apply per-process.** Half of what tunes the client is environment
    variables, and the docs are explicit that they are read at import time and later changes
    are ignored. A long-lived server cannot switch Xet on or off, or turn on sequential
    writes for a mechanical disk, without a fresh interpreter.

  * **A fresh connection pool every time.** A process that has been up for hours holds
    keep-alive sockets that a NAT may already have dropped; the download then sits at
    "connected" and moves nothing, which is the failure people describe as the library
    hanging on the second repository.

  * **It can be killed.** A wedged thread inside somebody else's library cannot be stopped
    from the outside. A process can.

The job arrives as JSON on stdin, progress goes out as JSON lines on stdout, and the token
comes through the environment so it never appears in a process listing. tqdm keeps its own
output on stderr, where it is ignored.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

EMIT_INTERVAL = 0.3

_lock = threading.Lock()
_moved = 0
_last_emit = 0.0


def emit(**payload) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _make_progress_class():
    """A tqdm subclass that also reports bytes on stdout.

    Subclassing the real tqdm rather than imitating it: `snapshot_download` drives these
    objects in ways beyond `update()`, and a partial imitation breaks in the middle of a
    download rather than at the start.
    """
    from tqdm.auto import tqdm as base

    class JsonProgress(base):
        def update(self, n=1):
            global _moved, _last_emit
            # Only byte meters count. The outer "Fetching N files" bar is measured in files
            # and would otherwise be added to the byte total.
            if getattr(self, "unit", "") == "B" and n:
                now = time.monotonic()
                with _lock:
                    _moved += n
                    due = now - _last_emit >= EMIT_INTERVAL
                    if due:
                        _last_emit = now
                        total = _moved
                if due:
                    emit(e="progress", bytes=total)
            return super().update(n)

    return JsonProgress


def run(job: dict) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    progress_class = _make_progress_class()
    common = {
        "repo_id": job["repo_id"],
        "repo_type": job.get("repo_type", "model"),
        "revision": job.get("revision") or "main",
        "tqdm_class": progress_class,
        # Never the default cache. It builds a blobs+snapshots tree out of symlinks, which
        # on Windows needs developer mode or an elevated process; without them the library
        # silently degrades to duplicating every file.
        "local_dir": job["local_dir"],
    }

    try:
        if job.get("filename"):
            path = hf_hub_download(filename=job["filename"], **common)
            paths = [path]
            target = job.get("target")
            if target:
                # A single file is staged under the repo's own directory layout, then moved
                # to where the library wants it.
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, target)
                paths = [target]
                _cleanup_staging(Path(job["local_dir"]), Path(target))
        else:
            snapshot_download(
                allow_patterns=job.get("allow_patterns") or None,
                ignore_patterns=job.get("ignore_patterns") or None,
                max_workers=int(job.get("max_workers") or 8),
                **common,
            )
            paths = [job["local_dir"]]
    except HfHubHTTPError as exc:
        emit(e="error", message=f"{type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - the message is what the caller needs
        emit(e="error", message=f"{type(exc).__name__}: {exc}")
        return 1

    with _lock:
        moved = _moved
    emit(e="done", paths=paths, bytes=moved)
    return 0


def _cleanup_staging(staging: Path, target: Path) -> None:
    """Remove the staging tree, but never anything the download was aimed at."""
    if staging == target.parent or staging in target.parents:
        return
    shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
    except ValueError as exc:
        emit(e="error", message=f"bad job: {exc}")
        return 2
    os.makedirs(job["local_dir"], exist_ok=True)
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())
