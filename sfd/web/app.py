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

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.diskinfo import free_bytes
from ..jobs import db
from ..jobs.db import Database
from ..jobs.manager import Manager
from ..library import folders, previews, relocate, sidecar
from ..library.categories import ALIASES, Category
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

# A preview never changes under its URL — the CDN path contains the image's own id — so the
# browser is told not to ask again. Without this the queue would re-request every thumbnail
# on every redraw, and the list redraws whenever any task changes state.
IMAGE_CACHE = {"Cache-Control": "private, max-age=604800"}

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
    # A folder relative to the library root, as offered by /folders or typed by hand. It does
    # not have to exist yet — the transfer creates it when the file lands.
    folder: str | None = None
    # Make this the home of the folder's kind from now on, not just for this file.
    remember: bool = False


class MoveRequest(BaseModel):
    # A folder relative to the library root. Unlike ConfirmRequest's, this one takes effect
    # immediately rather than when a download lands, so an empty value has nothing to mean.
    folder: str
    remember: bool = False


class BrowseMoveRequest(BaseModel):
    # No folder field, and that is the point — see the endpoint.
    remember: bool = False


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
    blur_nsfw: bool | None = None
    preview_dir: str | None = None
    sidecar_dir: str | None = None
    write_compat_files: bool | None = None
    write_trigger_txt: bool | None = None

    layout_overrides: dict[str, str] | None = None


def create_app(settings: Settings, database: Database) -> FastAPI:
    manager = Manager(settings, database)

    # One client for every preview this process ever fetches. Building one per request
    # would open a fresh TLS session for each thumbnail in a screenful of them.
    images = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=15.0, pool=10.0),
        follow_redirects=True,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()
            await images.aclose()
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

    @app.get("/api/tasks/{task_id}/folders")
    async def offer_folders(task_id: int) -> dict[str, Any]:
        """Where this file could go, best guesses first.

        The list is the library as it really is, not the canonical category names: the folder
        a model actually belongs in is often one we deliberately refuse to claim
        automatically, and until it is on the list the question cannot be answered correctly.
        """
        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        if not settings.library_root:
            return {"root": "", "folders": [], "category": None}

        category = Category(task.category) if task.category in Category._value2member_map_ else None
        offered = folders.offer(manager.layout(), category, task.base_model, task.filename)
        return {
            "root": settings.library_root,
            "category": category.value if category else None,
            "base_model": task.base_model,
            "folders": [f.to_json() for f in offered],
        }

    @app.post("/api/tasks/{task_id}/confirm")
    async def confirm(task_id: int, request: ConfirmRequest) -> dict[str, bool]:
        if request.category and request.category not in Category._value2member_map_:
            raise HTTPException(400, f"unknown category {request.category}")

        chosen: Path | None = None
        if request.folder:
            chosen = folders.resolve_inside(Path(settings.library_root), request.folder)
            if chosen is None:
                raise HTTPException(400, f"{request.folder} is not inside the library root")

        manager.confirm(task_id, request.category, chosen)

        # "Always put this kind here" is a different statement from "put this file here", so
        # it is only taken when the folder itself names a kind: remembering `checkpoints/Krea
        # 2` would send every future checkpoint into one base model's folder.
        if request.remember and chosen is not None:
            kind = ALIASES.get(chosen.name.lower())
            if kind is not None:
                settings.layout_overrides[kind.value] = str(chosen)
                settings.save()
        return {"ok": True}

    async def _run_move(task_id: int, folder: Path) -> relocate.Move | None:
        """Run a move, turning every way it can end into the right status code.

        None is the one outcome that is not a failure: somebody pressed stop, and the file
        is still exactly where it was.
        """
        try:
            return await manager.move(task_id, folder)
        except relocate.Cancelled:
            return None
        except LookupError:
            raise HTTPException(404, "no such task") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except OSError as exc:
            raise HTTPException(500, f"could not move the file: {exc}") from None

    @app.post("/api/tasks/{task_id}/move/stop")
    async def stop_move(task_id: int) -> dict[str, bool]:
        """Give up on a move that is still copying.

        Only a move across drives can be stopped, and only because that one is a copy that
        takes minutes. Within a drive the move is a rename that is over before this request
        could be sent, which is why an answer of false is not an error: it means there was
        nothing left to stop.
        """
        return {"ok": manager.stop_move(task_id)}

    @app.post("/api/tasks/{task_id}/move")
    async def move(task_id: int, request: MoveRequest) -> dict[str, Any]:
        """Move a finished download into another folder of the library.

        Like reveal, the file being moved is the task's own — the request says which folder,
        never which file. And like confirm, the folder is refused unless it resolves inside
        the library root: this endpoint moves data on a server with no authentication, so
        the one string it accepts from the browser is confined before it becomes a path.
        """
        if not settings.library_root:
            raise HTTPException(400, "no library root is set, so there is nowhere to move to")

        chosen = folders.resolve_inside(Path(settings.library_root), request.folder)
        if chosen is None:
            raise HTTPException(400, f"{request.folder} is not inside the library root")

        result = await _run_move(task_id, chosen)
        if result is None:
            return {"ok": False, "stopped": True}

        remembered = None
        if request.remember:
            # Same rule as confirm: only a folder that names a kind can stand for that kind.
            kind = ALIASES.get(chosen.name.lower())
            if kind is not None:
                settings.layout_overrides[kind.value] = str(chosen)
                settings.save()
                remembered = kind.value
        return {
            "ok": True,
            "unchanged": result.unchanged,
            "dest": str(result.path),
            # What the mapping now says, or null when the folder named no kind and the
            # checkbox therefore did nothing. The page has no business working that out.
            "remembered": remembered,
            "moved": len(result.companions),
            # Named rather than counted: "two sidecars stayed behind" is not something a
            # person can act on without knowing which ones.
            "failed": [{"path": str(p), "reason": reason} for p, reason in result.failed],
        }

    @app.post("/api/tasks/{task_id}/move-anywhere")
    async def move_anywhere(task_id: int, request: BrowseMoveRequest) -> dict[str, Any]:
        """Move a finished download to a folder chosen in the system's own dialog.

        The sibling of /move, and the reason it is a second endpoint rather than a flag: it
        accepts a destination anywhere on the machine, including another drive, which is
        precisely what /move refuses. What makes that safe is that the request never names
        one. The path comes from a modal dialog the operating system put in front of whoever
        is at the keyboard, so an attacker who can reach this unauthenticated API — a page
        that resolved its own hostname to 127.0.0.1, say — can start a folder picker and
        nothing else. It cannot answer it, and it cannot say where the file should land.

        The dialog blocks until it is answered, so it runs on a thread; the queue keeps
        going, and so does the download whose file this is not.
        """
        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        if task.state != db.DONE or not task.dest:
            raise HTTPException(409, "only a finished download can be moved")

        from ..desktop import pick_system_folder

        current = str(Path(task.dest).parent)
        chosen = await asyncio.to_thread(pick_system_folder, current)
        if not chosen:
            return {"ok": False, "cancelled": True}

        result = await _run_move(task_id, Path(chosen))
        if result is None:
            return {"ok": False, "stopped": True}

        remembered = None
        if request.remember:
            # An absolute path outside the library is a legitimate mapping — a drive that
            # holds nothing but checkpoints is the usual reason anyone moves one there.
            kind = ALIASES.get(Path(chosen).name.lower())
            if kind is not None:
                settings.layout_overrides[kind.value] = chosen
                settings.save()
                remembered = kind.value
        return {
            "ok": True,
            "unchanged": result.unchanged,
            "dest": str(result.path),
            "folder": chosen,
            "remembered": remembered,
            "moved": len(result.companions),
            "failed": [{"path": str(p), "reason": reason} for p, reason in result.failed],
        }

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

    @app.get("/api/tasks/{task_id}/previews")
    async def list_previews(task_id: int) -> dict[str, Any]:
        """What pictures this download has, and what made them.

        The URLs are handed back too. They are what the page needs to offer "open the
        original", and they are already public — this is the model's own page, not a
        signed download link.
        """
        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        return {
            "blur_nsfw": settings.blur_nsfw,
            "previews": [
                {
                    "index": index,
                    "type": entry.get("type") or "image",
                    "nsfw": bool(entry.get("nsfw")),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "meta": entry.get("meta") or {},
                    "url": entry.get("url"),
                }
                for index, entry in enumerate(previews.entries(task.meta))
            ],
        }

    async def _serve_preview(task_id: int, index: int, width: int | None) -> FileResponse:
        """Fetch a sample image once, then serve it from disk forever.

        Which image is decided here, from the task, and never by the caller: the request
        says *which of this download's previews*, an index into a list the service gave us,
        and cannot say *this URL* or *this file*. Everything else in this server that
        touches a path takes the same line, and an unauthenticated local server that
        fetched arbitrary URLs on request would be a proxy into the machine it runs on.
        """
        if not settings.fetch_previews:
            raise HTTPException(404, "previews are turned off")

        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        entries = previews.entries(task.meta)
        if not 0 <= index < len(entries):
            raise HTTPException(404, "no such preview")

        entry = entries[index]
        # A video asked for small is asked for as a still: a queue row wants a picture, not
        # eight seconds of playback it never plays.
        still = entry.get("type") == "video" and width is not None
        url = previews.variant_url(str(entry["url"]), width, still=still)

        directory = Path(settings.preview_dir or "previews")
        path = previews.cache_path(directory, url)
        if not path.exists():
            fetched = await previews.fetch(url, directory, images)
            if fetched is None:
                raise HTTPException(502, "the preview could not be fetched")
            path = fetched
        return FileResponse(path, media_type=previews.media_type(path), headers=IMAGE_CACHE)

    @app.get("/api/tasks/{task_id}/preview")
    async def preview(
        task_id: int, w: int | None = Query(None, ge=32, le=2048)
    ) -> FileResponse:
        return await _serve_preview(task_id, 0, w)

    @app.get("/api/tasks/{task_id}/preview/{index}")
    async def preview_at(
        task_id: int, index: int, w: int | None = Query(None, ge=32, le=2048)
    ) -> FileResponse:
        return await _serve_preview(task_id, index, w)

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
        """What an adopted tree would look like, so the mapping can be checked.

        Taken from the manager rather than adopted afresh, so that what is shown includes the
        corrections made by hand — a mapping display that disagreed with where files actually
        go would be worse than not offering one.
        """
        if not settings.library_root:
            return {"root": None, "paths": {}, "ambiguities": {}}
        data = manager.layout().to_dict()
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
