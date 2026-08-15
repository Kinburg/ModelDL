"""Driving HuggingFace's own client from the outside.

The parent half of the subprocess engine. Its job is to hand over a description of the work,
turn the child's JSON lines back into the same `ProgressSnapshot` the native transfer emits,
and be able to stop it.

What this engine is *for*: whole repositories. A sharded model is thirty files with an index,
and `snapshot_download` fetches them correctly — including Xet's chunk deduplication, which
pays off when a repo is re-fetched after an update. It is also the fallback when the native
path is refused for reasons we cannot fix from here.

What it is not for: placing a single file exactly where the library layout wants it. The
native transfer does that better, with resume we control and a hash we verify ourselves.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.errors import SfdError, TransferFailed
from ..core.speed import SmoothedSpeed
from ..core.types import ProgressSnapshot

ProgressCallback = Callable[[ProgressSnapshot], None]


@dataclass(slots=True)
class HfHubOptions:
    """Everything here is an environment variable, which is exactly why a subprocess.

    `huggingface_hub` reads these once at import, so a running server cannot change its mind
    about any of them without a new interpreter.
    """

    disable_xet: bool = False
    # Xet writes byte ranges in parallel by direct addressing, which is right for NVMe and
    # a seek storm on a mechanical disk.
    sequential_writes: bool = False
    high_performance: bool = False
    download_timeout: int = 30
    max_workers: int = 8
    extra_env: dict[str, str] = field(default_factory=dict)

    def environment(self, token: str | None) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "HF_HUB_DOWNLOAD_TIMEOUT": str(self.download_timeout),
                # Python must not buffer the child's stdout, or progress arrives in bursts
                # at the end instead of as it happens.
                "PYTHONUNBUFFERED": "1",
            }
        )
        if self.disable_xet:
            env["HF_HUB_DISABLE_XET"] = "1"
        if self.sequential_writes:
            env["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
        if self.high_performance:
            env["HF_XET_HIGH_PERFORMANCE"] = "1"
        # Through the environment, not the command line, which is readable by every process
        # on the machine.
        if token:
            env["HF_TOKEN"] = token
        else:
            env.pop("HF_TOKEN", None)
        env.update(self.extra_env)
        return env


@dataclass(slots=True)
class HfHubResult:
    paths: list[Path]
    transferred: int


class HfHubEngine:
    name = "hf_hub"

    def __init__(self, token: str | None = None, options: HfHubOptions | None = None) -> None:
        self._token = token
        self._options = options or HfHubOptions()
        self._process: asyncio.subprocess.Process | None = None
        # Tests point this at a stub that speaks the same protocol, so the contract between
        # the two halves can be checked without the network.
        self._worker_override: Path | None = None

    async def download_repo(
        self,
        repo_id: str,
        destination: Path,
        *,
        repo_type: str = "model",
        revision: str = "main",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        total: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> HfHubResult:
        return await self._run(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "local_dir": str(destination),
                "allow_patterns": allow_patterns,
                "ignore_patterns": ignore_patterns,
                "max_workers": self._options.max_workers,
            },
            total,
            on_progress,
        )

    async def download_file(
        self,
        repo_id: str,
        path_in_repo: str,
        destination: Path,
        *,
        repo_type: str = "model",
        revision: str = "main",
        total: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> HfHubResult:
        # The client insists on reproducing the repo's directory layout, so the file is
        # staged beside its destination and moved into place afterwards.
        staging = destination.parent / f".hf-staging-{destination.name}"
        return await self._run(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "filename": path_in_repo,
                "local_dir": str(staging),
                "target": str(destination),
            },
            total,
            on_progress,
        )

    async def cancel(self) -> None:
        """Stop the child. Its partial `.incomplete` files are what let it resume later."""
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()

    # --- internals --------------------------------------------------------

    async def _run(
        self, job: dict[str, Any], total: int | None, on_progress: ProgressCallback | None
    ) -> HfHubResult:
        worker = self._worker_override or Path(__file__).with_name("hf_worker.py")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,   # tqdm draws its own bars there
            env=self._options.environment(self._token),
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        self._process = process

        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(job).encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        speed = SmoothedSpeed()
        moved = 0
        paths: list[Path] = []
        error: str | None = None

        try:
            async for raw in process.stdout:
                try:
                    event = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue     # anything not ours is not worth failing over
                kind = event.get("e")
                if kind == "progress":
                    moved = int(event.get("bytes") or 0)
                    if on_progress is not None:
                        on_progress(_snapshot(moved, total, speed))
                elif kind == "done":
                    moved = int(event.get("bytes") or moved)
                    paths = [Path(p) for p in event.get("paths") or []]
                elif kind == "error":
                    error = str(event.get("message"))
        except asyncio.CancelledError:
            await self.cancel()
            raise
        finally:
            await process.wait()
            self._process = None

        if error:
            raise TransferFailed(f"huggingface_hub: {error}")
        if process.returncode != 0:
            raise TransferFailed(
                f"huggingface_hub worker exited with {process.returncode}"
            )
        if not paths:
            raise TransferFailed("huggingface_hub reported no files")
        return HfHubResult(paths=paths, transferred=moved)


def _snapshot(moved: int, total: int | None, speed: SmoothedSpeed) -> ProgressSnapshot:
    rate = speed.update(moved)
    remaining = (total - moved) if total else None
    return ProgressSnapshot(
        downloaded=moved,
        total=total,
        speed=rate,
        # The child owns the sockets; a per-connection count is not ours to report.
        connections=0,
        hashed=0,
        eta=(remaining / rate) if (remaining and rate > 0) else None,
    )


__all__ = ["HfHubEngine", "HfHubOptions", "HfHubResult", "SfdError"]
