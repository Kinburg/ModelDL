"""Local web interface.

A single page served on localhost. Commands go over plain REST; progress comes back over
Server-Sent Events, which is one-directional and reconnects on its own — everything a
progress feed needs and nothing it does not.

The server binds to 127.0.0.1 and has no authentication, because it holds the user's tokens
and must not be reachable from anywhere else on the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.diskinfo import free_bytes
from ..jobs import db
from ..jobs.db import Database
from ..jobs.manager import Manager
from ..library import sidecar
from ..library.categories import Category
from ..library.layout import adopt
from ..settings import Settings


def _get_static_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "sfd" / "web" / "static"
        if bundled.exists():
            return bundled
        alt = Path(sys._MEIPASS) / "static"
        if alt.exists():
            return alt
    return Path(__file__).parent / "static"


STATIC = _get_static_dir()
HEARTBEAT = 20.0

# Names this server answers to. Anything else arriving at the loopback socket got here under
# a hostname that resolves to 127.0.0.1 but is not ours — see `_local_only` below.
LOCAL_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_name(header: str) -> str:
    """The name from a Host header, without the port. `[::1]:7788` is a bracketed address."""
    host = header.strip().lower()
    if host.startswith("["):
        return host[1 : host.index("]")] if "]" in host else ""
    return host.split(":", 1)[0]


class AddRequest(BaseModel):
    source: str


class ConfirmRequest(BaseModel):
    category: str | None = None


class ReorderRequest(BaseModel):
    ids: list[int]


class PickFolderRequest(BaseModel):
    initial: str = ""


class SettingsPatch(BaseModel):
    """What the settings form is allowed to say.

    Every field optional, because the page sends a patch. The bounds are the ones the form
    already shows — repeated here because the form is not the only thing that can post, and
    a value like `connections: "several"` accepted here would not fail here: it would fail
    much later, inside a transfer, as something that reads like a bug in the downloader.

    Unknown keys are dropped rather than refused, which is what keeps `_path` from being
    redirected through this endpoint.
    """

    library_root: str | None = None
    profile: Literal["comfyui", "a1111"] | None = None
    group_by_base_model: bool | None = None
    download_dir: str | None = None

    hf_token: str | None = None
    civitai_token: str | None = None

    connections: int | None = Field(None, ge=1, le=64)
    concurrent_downloads: int | None = Field(None, ge=1, le=8)
    auto_start: bool | None = None
    auto_retry: bool | None = None
    queue_position: Literal[db.TOP, db.BOTTOM] | None = None
    min_speed_kb: float | None = Field(None, ge=0)
    max_speed_kb: float | None = Field(None, ge=0)
    verify_hash: bool | None = None
    verify_existing: bool | None = None
    disk_kind: Literal["", "ssd", "hdd"] | None = None

    hf_engine: Literal["native", "hf_hub"] | None = None
    hf_fallback: bool | None = None
    hf_disable_xet: bool | None = None
    hf_xet_high_performance: bool | None = None
    hf_xet_sequential_writes: bool | None = None

    write_sidecars: bool | None = None
    fetch_previews: bool | None = None
    sidecar_dir: str | None = None
    write_compat_files: bool | None = None
    write_trigger_txt: bool | None = None

    layout_overrides: dict[str, str] | None = None


def create_app(settings: Settings, database: Database) -> FastAPI:
    manager = Manager(settings, database)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()
            database.close()

    app = FastAPI(
        title="ModelDL", docs_url=None, redoc_url=None, lifespan=lifespan
    )

    @app.middleware("http")
    async def _local_only(request: Request, call_next):
        """Answer only to our own names.

        Binding to the loopback interface keeps the rest of the network out, but not the
        browser: any page can point a hostname it owns at 127.0.0.1 and then talk to this
        API as same-origin — DNS rebinding — and there is no authentication here to stop it
        from rewriting the library path or queueing downloads. The Host header is the part
        that attack cannot fake, because the browser sends the name it was told to visit.
        """
        if _host_name(request.headers.get("host", "")) not in LOCAL_NAMES:
            return JSONResponse({"detail": "not served under that name"}, status_code=403)
        return await call_next(request)

    # --- page -------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # --- tasks ------------------------------------------------------------

    @app.get("/api/tasks")
    async def list_tasks() -> dict[str, Any]:
        return {"tasks": [t.to_json() for t in database.list()]}

    @app.post("/api/tasks")
    async def add_task(request: AddRequest) -> dict[str, Any]:
        source = request.source.strip()
        if not source:
            raise HTTPException(400, "nothing to add")
        try:
            created = await manager.add(source)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part here
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        if not created:
            raise HTTPException(409, "already in the queue")
        return {"tasks": [t.to_json() for t in created]}

    @app.post("/api/tasks/{task_id}/pause")
    async def pause(task_id: int) -> dict[str, bool]:
        manager.pause(task_id)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/resume")
    async def resume(task_id: int) -> dict[str, bool]:
        manager.resume(task_id)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/retry")
    async def retry(task_id: int) -> dict[str, bool]:
        manager.retry(task_id)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/confirm")
    async def confirm(task_id: int, request: ConfirmRequest) -> dict[str, bool]:
        if request.category and request.category not in Category._value2member_map_:
            raise HTTPException(400, f"unknown category {request.category}")
        manager.confirm(task_id, request.category)
        return {"ok": True}

    @app.get("/api/tasks/{task_id}/record")
    async def record(task_id: int) -> dict[str, Any]:
        """The JSON record for a finished download.

        Collecting records into their own directory makes the library tidy and the records
        hard to find; this puts them back within reach of the thing they describe. The path
        is derived from the task, never from the request, so there is nothing here to point
        at an arbitrary file.
        """
        task = database.get(task_id)
        if task is None or not task.dest:
            raise HTTPException(404, "no such task")

        destination = Path(task.dest)
        directory = Path(settings.sidecar_dir) if settings.sidecar_dir else None
        root = Path(settings.library_root) if settings.library_root else None
        data = sidecar.read(destination, directory, root)
        if data is None:
            raise HTTPException(404, "no record was written for this file")
        return {
            "record": data,
            "path": str(sidecar.record_path(destination, directory, root)),
        }

    @app.post("/api/tasks/{task_id}/reveal")
    async def reveal(task_id: int) -> dict[str, bool]:
        """Show a downloaded file in Explorer or Finder.

        The path is taken from the task rather than from the request, for the same reason
        the record endpoint does it: this one hands a string to the shell, and a server with
        no authentication must not take that string from whoever asked.
        """
        task = database.get(task_id)
        if task is None or not task.dest:
            raise HTTPException(404, "no such task")

        from ..desktop import open_system_path
        return {"ok": open_system_path(task.dest)}

    @app.delete("/api/tasks/{task_id}")
    async def cancel(task_id: int) -> dict[str, bool]:
        manager.cancel(task_id)
        return {"ok": True}

    @app.post("/api/tasks/start-all")
    async def start_all() -> dict[str, int]:
        return {"released": manager.start_all()}

    @app.post("/api/tasks/clear")
    async def clear() -> dict[str, int]:
        return {"removed": manager.clear_finished()}

    @app.post("/api/tasks/reorder")
    async def reorder(request: ReorderRequest) -> dict[str, bool]:
        """Apply the order the queue was dragged into.

        Only affects what is picked up next; a task already running keeps running.
        """
        database.reorder(request.ids)
        manager.emit({"type": "reload"})
        return {"ok": True}

    @app.get("/api/space")
    async def space() -> dict[str, Any]:
        """What the queue still has to fetch, against what the disk has left.

        A queue is assembled long before it runs, so the answer is worth having while there
        is still a chance to drop something from it — rather than at four in the morning,
        one task at a time, as each one refuses to start.
        """
        pending = [
            t for t in database.list()
            if t.state in (db.PENDING, db.RUNNING, db.PAUSED, db.BLOCKED)
        ]
        needed = sum(max(0, (t.size or 0) - t.downloaded) for t in pending)
        root = Path(settings.library_root or settings.download_dir)
        return {
            "needed": needed,
            "free": free_bytes(root),
            "unknown": sum(1 for t in pending if not t.size),
            "path": str(root),
        }

    # --- settings ---------------------------------------------------------

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {
            "settings": settings.redacted(),
            "categories": [c.value for c in Category],
            "error": settings.error,
        }

    @app.put("/api/settings")
    async def put_settings(patch: SettingsPatch) -> dict[str, Any]:
        # Only what was actually sent: the form posts a subset, and filling the rest in from
        # the model's defaults would quietly reset every field it does not show.
        settings.apply(patch.model_dump(exclude_unset=True))
        settings.save()
        # "Files at once" and the speed ceiling are live controls, not ones that wait for a
        # restart — you reach for them precisely while something is downloading.
        manager.apply_settings()
        return {"settings": settings.redacted()}

    @app.get("/api/layout")
    async def get_layout() -> dict[str, Any]:
        """What an adopted tree would look like, so the mapping can be checked."""
        if not settings.library_root:
            return {"root": None, "paths": {}, "ambiguities": {}}
        layout = adopt(Path(settings.library_root), settings.profile)
        data = layout.to_dict()
        data["exists"] = {
            category: Path(path).is_dir() for category, path in data["paths"].items()
        }
        return data

    # --- desktop utils ----------------------------------------------------

    @app.post("/api/utils/pick-folder")
    def pick_folder(request: PickFolderRequest) -> dict[str, Any]:
        """Open a native folder picker dialog.

        Synchronous on purpose. The dialog blocks until it is answered, and on the event
        loop that would freeze every running download for as long as the window sits open;
        FastAPI runs a `def` handler in a worker thread instead.
        """
        from ..desktop import pick_system_folder
        return {"path": pick_system_folder(request.initial)}

    # --- progress ---------------------------------------------------------

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        queue = manager.subscribe()

        async def stream():
            try:
                yield _sse({"type": "hello"})
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT)
                    except TimeoutError:
                        # Keeps proxies and the browser from deciding the stream is dead.
                        yield ": ping\n\n"
                        continue
                    yield _sse(event)
            finally:
                manager.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.manager = manager
    return app


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


__all__ = ["create_app", "db", "contextlib"]
