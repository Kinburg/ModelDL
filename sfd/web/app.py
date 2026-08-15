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
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..jobs import db
from ..jobs.db import Database
from ..jobs.manager import Manager
from ..library import sidecar
from ..library.categories import Category
from ..library.layout import adopt
from ..settings import Settings

STATIC = Path(__file__).parent / "static"
HEARTBEAT = 20.0


class AddRequest(BaseModel):
    source: str


class ConfirmRequest(BaseModel):
    category: str | None = None


class ReorderRequest(BaseModel):
    ids: list[int]


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

    # --- page -------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

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

    # --- settings ---------------------------------------------------------

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {
            "settings": settings.redacted(),
            "categories": [c.value for c in Category],
            "error": settings.error,
        }

    @app.put("/api/settings")
    async def put_settings(patch: dict[str, Any]) -> dict[str, Any]:
        settings.apply(patch)
        settings.save()
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
