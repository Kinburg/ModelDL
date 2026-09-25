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
import mimetypes
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.diskinfo import free_bytes
from ..jobs import db
from ..jobs.db import Database
from ..jobs.library import Busy
from ..jobs.manager import Manager
from ..library import details, erase, folders, previews, relocate, sidecar, workflows
from ..library.categories import ALIASES, Category
from ..library.layout import adopt
from ..settings import Settings

# Windows keeps its own idea of what a `.js` file is in the registry, and on a fair number of
# machines some installer has set it to `text/plain`. Python's `mimetypes` reads that, the
# browser refuses to run a module served as plain text, and the page comes up blank. Said
# here, once, it no longer depends on the machine.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


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


SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _from_this_page(request: Request) -> bool:
    """Whether a request that changes something came from a page this server served.

    A browser names the origin of every such request, and fetch metadata says outright when
    it is another site. A request carrying neither is not from a browser page at all — a
    script, a test — and those were never the danger: they could send any header they like.
    """
    site = (request.headers.get("sec-fetch-site") or "").lower()
    if site and site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin is None:
        return True
    if origin == "null":
        return False
    return _host_name(urlsplit(origin).netloc) in LOCAL_NAMES


class AddRequest(BaseModel):
    source: str


class QueueRequest(BaseModel):
    # Which files of the resolved link, by their place in it — never a name or a URL.
    files: list[int] = Field(default_factory=list, max_length=5000)
    # Which folder of the library, and a folder inside it, confined to it like every other
    # folder the page names. It does not have to exist: the transfer makes it.
    root: int = 0
    folder: str = ""
    # Keep the repository's own folders under the one chosen.
    keep_structure: bool = False
    remember: bool = False


class QueueAnywhereRequest(BaseModel):
    files: list[int] = Field(default_factory=list, max_length=5000)
    keep_structure: bool = False
    remember: bool = False


class ConfirmRequest(BaseModel):
    category: str | None = None
    # A folder of the library — which one, by its place in the list, and a folder inside
    # it — as offered by /folders or typed by hand. It does not have to exist yet: the
    # transfer creates it when the file lands.
    root: int = 0
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


class RenameRequest(BaseModel):
    # A filename, and only ever a filename. What that means — and what happens to anything
    # that is not one — is `relocate.intended_name`.
    name: str


class NoteRequest(BaseModel):
    # Long enough for a paragraph about what a model actually turned out to be like; short
    # enough that the record stays a record. An empty string is how a note is removed.
    note: str = Field("", max_length=4000)


class ReorderRequest(BaseModel):
    ids: list[int]


class PickFolderRequest(BaseModel):
    initial: str = ""


class WorkflowNameRequest(BaseModel):
    # A name in the workflows folder, as a save reported it — never a path.
    name: str = Field(..., max_length=260)


class ModelMoveRequest(BaseModel):
    # Which folder of the library, by its place in the list, and a folder inside it. The
    # pair is confined exactly as the queue's single folder is: `resolve_inside` the root.
    root: int = 0
    folder: str = ""
    remember: bool = False


class BatchMoveRequest(ModelMoveRequest):
    ids: list[int] = Field(default_factory=list, max_length=500)


class IdsRequest(BaseModel):
    ids: list[int] = Field(default_factory=list, max_length=5000)


class BatchAnywhereRequest(IdsRequest):
    remember: bool = False


class SkipRequest(IdsRequest):
    # False counts a skipped update again.
    skip: bool = True


class LinkRequest(BaseModel):
    # The copy kept, and the copies that become other names of it — all models of the
    # library, never paths.
    keep: int
    ids: list[int] = Field(default_factory=list, max_length=500)


class CleanupDeleteRequest(IdsRequest):
    # Which scan the numbers are from: a second window looking again renumbers them.
    token: str = ""


class ForgetRequest(BaseModel):
    cleanup: bool = False


class BatchForgetRequest(IdsRequest):
    cleanup: bool = False


class RelinkRequest(BaseModel):
    # Another model of the library, never a path: the file it points at was found by the
    # walk, not named by the page.
    candidate: int


class ConfirmLocateRequest(BaseModel):
    token: str


class NewFolderRequest(BaseModel):
    root: int = 0
    parent: str = ""
    name: str = Field(..., max_length=200)


class FolderRequest(BaseModel):
    root: int = 0
    relative: str = ""


class UnhideRequest(BaseModel):
    index: int


class UiPatch(BaseModel):
    # Whatever the page wants to remember about how it was left. Bounded, because it is
    # written into the settings file on every change.
    prefs: dict[str, Any] = Field(default_factory=dict)


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
    extra_roots: list[str] | None = Field(None, max_length=64)
    exclude_dirs: list[str] | None = Field(None, max_length=256)
    profile: Literal["comfyui", "a1111"] | None = None
    group_by_base_model: bool | None = None
    smart_placement: bool | None = None
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

    write_sidecars: bool | None = None
    fetch_previews: bool | None = None
    blur_nsfw: bool | None = None
    workflow_dir: str | None = None
    preview_dir: str | None = None
    sidecar_dir: str | None = None
    write_compat_files: bool | None = None
    write_trigger_txt: bool | None = None
    check_updates_on_start: bool | None = None

    layout_overrides: dict[str, str] | None = None
    ui: dict[str, Any] | None = None


# How much the page may keep in `ui`. Pane widths and a list of open folders fit in a
# fraction of this; the limit is there so that nothing can turn the settings file into a
# dumping ground.
UI_LIMIT = 64 * 1024


class NoCacheStatic(StaticFiles):
    """The page's own files, revalidated on every load.

    The page is served by the process that is also the app, and it changes when the app
    is updated. A WebView that kept yesterday's script against today's server would show a
    page that calls endpoints which no longer answer the way it expects. An ETag check is
    one small request per file and never a stale page.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


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
        # The Host check stops a page that renamed itself into this server. It does not stop
        # a page on any other site from sending a request straight to 127.0.0.1: a POST with
        # no body, or with one of no declared type, needs no permission from us to be sent,
        # and would be carried out — a move, a delete, a folder taken off the list. A browser
        # says where a request comes from, and only the app's own page is let through.
        if request.method not in SAFE_METHODS and not _from_this_page(request):
            return JSONResponse({"detail": "not from this app's own page"}, status_code=403)
        return await call_next(request)

    # --- page -------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", NoCacheStatic(directory=STATIC), name="static")

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
            added = await manager.add_links(source)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part here
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        created, skipped = added["created"], added["skipped"]
        if not created:
            if skipped:
                where = skipped[0].path
                more = f" (and {len(skipped) - 1} more)" if len(skipped) > 1 else ""
                raise HTTPException(409, f"already in your library: {where}{more}")
            raise HTTPException(409, "already in the queue")
        return {
            "tasks": [t.to_json() for t in created],
            "skipped": [{"id": m.id, "path": m.path} for m in skipped],
        }

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
        offered = await asyncio.to_thread(
            manager.library.rank_folders, manager.layout(), category, task.base_model,
            task.filename, meta=task.meta, confidence=task.confidence,
        )
        return {
            "root": settings.library_root,
            "category": category.value if category else None,
            "base_model": task.base_model,
            "folders": offered,
        }

    @app.post("/api/tasks/{task_id}/confirm")
    async def confirm(task_id: int, request: ConfirmRequest) -> dict[str, bool]:
        if request.category and request.category not in Category._value2member_map_:
            raise HTTPException(400, f"unknown category {request.category}")

        chosen: Path | None = None
        if request.folder:
            try:
                chosen = manager.library.resolve_folder(request.root, request.folder)
            except LookupError:
                raise HTTPException(400, "no such library folder") from None
            if chosen is None or not request.folder.strip("/\\ "):
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

    @app.post("/api/tasks/{task_id}/rename")
    async def rename(task_id: int, request: RenameRequest) -> dict[str, Any]:
        """Give a finished download a different filename, sidecars and all.

        The one endpoint here that takes a name from the browser and puts it on the disk,
        which is why `intended_name` is strict about what a name is: everything else on this
        server derives its paths from the task. A separator, a `..`, a drive letter — none of
        them are a filename, so all of them are refused as one rather than quietly becoming
        a move to somewhere nobody asked for. The folder is never part of the question.
        """
        try:
            result = await manager.rename(task_id, request.name)
        except LookupError:
            raise HTTPException(404, "no such task") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except OSError as exc:
            raise HTTPException(500, f"could not rename the file: {exc}") from None

        return {
            "ok": True,
            "unchanged": result.unchanged,
            "dest": str(result.path),
            "filename": result.path.name,
            "renamed": len(result.companions),
            "failed": [{"path": str(p), "reason": reason} for p, reason in result.failed],
        }

    @app.post("/api/tasks/{task_id}/note")
    async def note(task_id: int, request: NoteRequest) -> dict[str, Any]:
        """Write your own note about a download, or clear it.

        The note is the one thing in a record that no service can supply and nothing can
        fetch again — which is why an emptied box removes it rather than storing a blank,
        and why a file downloaded with sidecars turned off gets a record written to hold it
        rather than being told there is nowhere to put it.

        Taken before the file lands, too, and for the same reason: what is worth writing
        down is known while the download is being queued. It waits in the row until there
        is a record for it, so `record_written` is False then — nothing was put on disk.
        """
        try:
            written, record_written = await manager.set_note(task_id, request.note)
        except LookupError:
            raise HTTPException(404, "no such task") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except OSError as exc:
            raise HTTPException(500, f"could not write the note: {exc}") from None

        return {"ok": True, "note": written, "record_written": record_written}

    @app.get("/api/tasks/{task_id}/files")
    async def files(task_id: int) -> dict[str, Any]:
        """Everything on disk that belongs to this download, with sizes.

        Asked for before a delete, and the reason the delete can be a single click: what is
        about to go permanently is a list a person can read, not a number they have to take
        on trust. A finished model brings four sidecars; an abandoned one brings a `.part`
        that is most of the file's size and the whole reason to be deleting anything.
        """
        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        if not task.dest:
            return {"files": [], "total": 0}

        found = await asyncio.to_thread(
            erase.belongings, Path(task.dest), **manager.library.places()
        )
        listed = _listing(found)
        return {
            "files": listed,
            "total": sum(f["size"] or 0 for f in listed),
            # The model itself is always first when it is there at all, so the page can say
            # "and 4 others" without working out which one is the model.
            "model": str(task.dest) if listed and listed[0]["path"] == task.dest else None,
        }

    @app.delete("/api/tasks/{task_id}/files")
    async def delete_files(task_id: int) -> dict[str, Any]:
        """Delete this download's files from the disk, and the task with them.

        Permanent — there is no recycle bin behind this — so the page asks first, with the
        list from /files in front of whoever is answering. Which files is decided here from
        the task, exactly as reveal and move decide it: the request names a task and can
        never name a path.
        """
        try:
            result = await manager.delete_files(task_id)
        except LookupError:
            raise HTTPException(404, "no such task") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except OSError as exc:
            # The model would not go, so nothing else was touched — see `erase`. On Windows
            # this is almost always a loader holding the weights open.
            raise HTTPException(500, f"could not delete the file: {exc}") from None

        return {
            "ok": True,
            "deleted": [str(p) for p in result.deleted],
            "missing": result.missing,
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
        found = sidecar.find_record(destination, **manager.library.places())
        data = sidecar.read_record(found) if found is not None else None
        if data is None:
            raise HTTPException(404, "no record was written for this file")
        return {"record": data, "path": str(found)}

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
        return await _remote_preview(previews.entries(task.meta), index, width)

    async def _remote_preview(
        entries: list[dict[str, Any]], index: int, width: int | None
    ) -> FileResponse:
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

    @app.post("/api/tasks/pause-all")
    async def pause_all() -> dict[str, int]:
        return {"paused": manager.pause_all()}

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

    # --- the library ------------------------------------------------------
    #
    # Every model the app knows about, by id. Like the task endpoints above, nothing here
    # takes a path from the page: a model is named by its id and the path comes from the
    # library; a folder is named by which folder of the library and a place inside it, and
    # is confined to it; anywhere else is chosen in the system's own dialog.

    library = manager.library

    def _library_folder(root: int, folder: str) -> Path:
        try:
            target = library.resolve_folder(root, folder)
        except LookupError:
            raise HTTPException(400, "no such library folder") from None
        if target is None:
            raise HTTPException(400, f"{folder} is not inside that library folder")
        return target

    def _remember(chosen: Path, asked: bool) -> str | None:
        """Make a folder the home of its kind, when asked and when its name is a kind."""
        if not asked:
            return None
        kind = ALIASES.get(chosen.name.lower())
        if kind is None:
            return None
        settings.layout_overrides[kind.value] = str(chosen)
        settings.save()
        return kind.value

    def _model(model_id: int):
        model = database.get_model(model_id)
        if model is None:
            raise HTTPException(404, "no such model")
        return model

    # --- asking where a download goes ---------------------------------------
    #
    # A link is resolved first and queued second, with the question in between: which of
    # its files, and where. The page names the files by their place in the resolved link
    # and the folder the way it names every folder — a library folder and a place inside
    # it — or leaves the choice to the system's own dialog.

    @app.post("/api/resolve")
    async def resolve(request: AddRequest) -> dict[str, Any]:
        source = request.source.strip()
        if not source:
            raise HTTPException(400, "nothing to add")
        try:
            return await manager.resolve_links(source)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part here
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc

    def _queued(result: dict[str, Any], chosen: Path, remember: bool) -> dict[str, Any]:
        return {
            "tasks": [t.to_json() for t in result["created"]],
            # Already downloading: the same file cannot be queued twice at once.
            "queued": result["queued"],
            "folder": str(chosen),
            "remembered": _remember(chosen, remember) if result["created"] else None,
        }

    @app.post("/api/resolve/{token}/queue")
    async def queue_resolved(token: str, request: QueueRequest) -> dict[str, Any]:
        chosen = _library_folder(request.root, request.folder)
        if not request.folder.strip("/\\ "):
            raise HTTPException(400, "a model goes into a folder of the library, not on top of it")
        with _answering():
            result = manager.queue_resolved(token, request.files, chosen, request.keep_structure)
        return _queued(result, chosen, request.remember)

    @app.post("/api/resolve/{token}/queue-anywhere")
    async def queue_resolved_anywhere(token: str, request: QueueAnywhereRequest) -> dict[str, Any]:
        """Queue into a folder chosen in the system's own dialog — another drive, say. The
        request cannot name the folder, only ask for the dialog, as with moving anywhere."""
        from ..desktop import pick_system_folder

        with _answering():
            manager.resolved_meta(token)
        chosen = await asyncio.to_thread(pick_system_folder, settings.library_root or "")
        if not chosen:
            return {"ok": False, "cancelled": True}
        with _answering():
            result = manager.queue_resolved(token, request.files, Path(chosen), request.keep_structure)
        return _queued(result, Path(chosen), request.remember)

    @app.post("/api/models/{model_id}/again")
    async def model_again(model_id: int) -> dict[str, Any]:
        """A missing model's file, as a link resolved from what its download kept, for the
        page to ask where it goes back to. Queued by /api/resolve/{token}/queue."""
        with _answering():
            return manager.resolve_again(model_id=model_id)

    @app.post("/api/history/{task_id}/again")
    async def history_again(task_id: int) -> dict[str, Any]:
        with _answering():
            return manager.resolve_again(task_id=task_id)

    @app.post("/api/models/{model_id}/find-online")
    async def find_online(model_id: int) -> dict[str, Any]:
        """Look for a model on Civitai and the Hub, by what the library kept of it: its
        hashes, its name, its size. Only ever on request — this asks two services about a
        file of yours."""
        with _answering():
            return await manager.find_online(model_id)

    @app.post("/api/found/{token}/{index}")
    async def resolve_found(token: str, index: int) -> dict[str, Any]:
        with _answering():
            return manager.resolve_found(token, index)

    @app.post("/api/models/redownload")
    async def redownload_models(request: IdsRequest) -> dict[str, Any]:
        """Download several missing models again, each into the folder it was in."""
        result = manager.redownload_models(request.ids)
        return {"tasks": [t.to_json() for t in result["created"]], "failed": result["failed"]}

    @app.delete("/api/resolve/{token}")
    async def drop_resolved(token: str) -> dict[str, bool]:
        manager.drop_resolved(token)
        return {"ok": True}

    @app.get("/api/resolve/{token}/preview")
    async def resolved_preview(
        token: str, w: int | None = Query(None, ge=32, le=2048)
    ) -> FileResponse:
        """The first sample of what is about to be downloaded — through here, and from the
        list the service gave, like every other preview."""
        if not settings.fetch_previews:
            raise HTTPException(404, "previews are turned off")
        with _answering():
            meta = manager.resolved_meta(token)
        return await _remote_preview(previews.entries(meta), 0, w)

    @app.get("/api/library")
    async def get_library() -> dict[str, Any]:
        return await asyncio.to_thread(library.snapshot)

    @app.post("/api/library/rescan")
    async def rescan() -> dict[str, Any]:
        """Walk the library's folders again. Asked for whenever the window comes back to
        the front, which is when files are likeliest to have been moved behind its back."""
        report = await library.refresh()
        return {"ok": True, "report": report}

    @app.get("/api/models/{model_id}")
    async def model_details(model_id: int) -> dict[str, Any]:
        with _answering():
            return await asyncio.to_thread(library.details, model_id)

    @app.post("/api/models/{model_id}/rename")
    async def rename_model(model_id: int, request: RenameRequest) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(library.rename, model_id, request.name)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except OSError as exc:
            raise HTTPException(500, f"could not rename the file: {exc}") from None
        return {
            "ok": True,
            "unchanged": result.unchanged,
            "dest": str(result.path),
            "filename": result.path.name,
            "renamed": len(result.companions),
            "failed": _failures(result.failed),
        }

    @app.post("/api/models/{model_id}/move")
    async def move_model(model_id: int, request: ModelMoveRequest) -> dict[str, Any]:
        chosen = _library_folder(request.root, request.folder)
        with _answering():
            try:
                result = await manager.move_model(model_id, chosen)
            except relocate.Cancelled:
                return {"ok": False, "stopped": True}
        return _moved(result, _remember(chosen, request.remember))

    @app.post("/api/models/move")
    async def move_models(request: BatchMoveRequest) -> dict[str, Any]:
        """Move several models into one folder: a drag of a selection onto the tree."""
        chosen = _library_folder(request.root, request.folder)
        return await _move_all(request.ids, chosen, request.remember)

    async def _move_all(ids: list[int], chosen: Path, remember: bool) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for model_id in ids:
            try:
                result = await manager.move_model(model_id, chosen)
            except relocate.Cancelled:
                results.append({"id": model_id, "ok": False, "stopped": True})
                break
            except (LookupError, ValueError, OSError) as exc:
                results.append({"id": model_id, "ok": False, "error": str(exc)})
                continue
            results.append({"id": model_id, **_moved(result)})
        moved_any = any(r.get("ok") and not r.get("unchanged") for r in results)
        return {
            "results": results,
            "remembered": _remember(chosen, remember) if moved_any else None,
            "folder": str(chosen),
        }

    @app.post("/api/models/{model_id}/move-anywhere")
    async def move_model_anywhere(model_id: int, request: BrowseMoveRequest) -> dict[str, Any]:
        """The system's own folder dialog, for a place outside the library's folders."""
        model = _model(model_id)
        from ..desktop import pick_system_folder

        chosen = await asyncio.to_thread(pick_system_folder, str(Path(model.path).parent))
        if not chosen:
            return {"ok": False, "cancelled": True}
        with _answering():
            try:
                result = await manager.move_model(model_id, Path(chosen))
            except relocate.Cancelled:
                return {"ok": False, "stopped": True}
        return {**_moved(result, _remember(Path(chosen), request.remember)), "folder": chosen}

    @app.post("/api/models/move-anywhere")
    async def move_models_anywhere(request: BatchAnywhereRequest) -> dict[str, Any]:
        if not request.ids:
            raise HTTPException(400, "nothing to move")
        first = _model(request.ids[0])
        from ..desktop import pick_system_folder

        chosen = await asyncio.to_thread(pick_system_folder, str(Path(first.path).parent))
        if not chosen:
            return {"ok": False, "cancelled": True}
        return await _move_all(request.ids, Path(chosen), request.remember)

    @app.post("/api/models/{model_id}/move/stop")
    async def stop_model_move(model_id: int) -> dict[str, bool]:
        return {"ok": manager.stop_model_move(model_id)}

    @app.get("/api/models/{model_id}/folders")
    async def model_folders(model_id: int) -> dict[str, Any]:
        with _answering():
            return await asyncio.to_thread(library.move_targets, model_id, manager.layout())

    @app.post("/api/models/{model_id}/note")
    async def model_note(model_id: int, request: NoteRequest) -> dict[str, Any]:
        with _answering():
            written, created = await asyncio.to_thread(library.set_note, model_id, request.note)
        return {"ok": True, "note": written, "record_written": created}

    @app.get("/api/models/{model_id}/files")
    async def model_files(model_id: int) -> dict[str, Any]:
        with _answering():
            found = await asyncio.to_thread(library.files, model_id)
            # The file's other names: while one is left, deleting this name frees nothing.
            others = await asyncio.to_thread(library.other_names, model_id)
        listed = _listing(found)
        return {"files": listed, "total": sum(f["size"] or 0 for f in listed), "others": others}

    @app.post("/api/models/files")
    async def models_files(request: IdsRequest) -> dict[str, Any]:
        """What deleting a whole selection would take, model by model."""
        groups = []
        for model_id in request.ids:
            with contextlib.suppress(LookupError, OSError):
                listed = _listing(await asyncio.to_thread(library.files, model_id))
                others = await asyncio.to_thread(library.other_names, model_id)
                groups.append({"id": model_id, "files": listed, "others": others,
                               "total": sum(f["size"] or 0 for f in listed)})
        return {"groups": groups, "total": sum(g["total"] for g in groups)}

    @app.post("/api/duplicates/link")
    async def link_copies(request: LinkRequest) -> dict[str, Any]:
        """Make copies proven identical other names of the one kept: every path keeps
        working, and the room of each copy is freed. The way back is /separate."""
        with _answering():
            return await asyncio.to_thread(library.link_copies, request.keep, request.ids)

    @app.post("/api/models/{model_id}/separate")
    async def separate_model(model_id: int) -> dict[str, Any]:
        """Give this name of a shared file a copy of its own again. Stopped like a move."""
        with _answering():
            try:
                return await manager.separate_model(model_id)
            except relocate.Cancelled:
                return {"ok": False, "stopped": True}

    @app.post("/api/models/separate")
    async def separate_models(request: IdsRequest) -> dict[str, Any]:
        results = []
        for model_id in request.ids:
            try:
                result = await manager.separate_model(model_id)
            except relocate.Cancelled:
                results.append({"id": model_id, "ok": False, "stopped": True})
                break
            except (LookupError, ValueError, OSError) as exc:
                results.append({"id": model_id, "ok": False, "error": str(exc)})
                continue
            results.append({"id": model_id, **result})
        return {"results": results}

    @app.delete("/api/models/{model_id}/files")
    async def delete_model_files(model_id: int) -> dict[str, Any]:
        """Permanent, like the queue's: asked first, with the list in front of whoever is
        answering. Which files is decided here, from the model."""
        if model_id in manager._model_moves:
            raise HTTPException(409, "this model is being moved right now")
        with _answering():
            result = await asyncio.to_thread(library.delete_files, model_id)
        return _erased(result)

    @app.post("/api/models/delete")
    async def delete_models(request: IdsRequest) -> dict[str, Any]:
        results = []
        for model_id in request.ids:
            if model_id in manager._model_moves:
                results.append({"id": model_id, "ok": False, "error": "being moved"})
                continue
            try:
                result = await asyncio.to_thread(library.delete_files, model_id)
            except (LookupError, ValueError, OSError) as exc:
                results.append({"id": model_id, "ok": False, "error": str(exc)})
                continue
            results.append({"id": model_id, **_erased(result)})
        return {"results": results}

    @app.get("/api/models/{model_id}/leftovers")
    async def model_leftovers(model_id: int) -> dict[str, Any]:
        with _answering():
            found = await asyncio.to_thread(library.leftovers, model_id)
        listed = _listing(found)
        return {"files": listed, "total": sum(f["size"] or 0 for f in listed)}

    @app.post("/api/models/{model_id}/forget")
    async def forget_model(model_id: int, request: ForgetRequest) -> dict[str, Any]:
        with _answering():
            result = await asyncio.to_thread(library.forget, model_id, request.cleanup)
        return _erased(result)

    @app.post("/api/models/forget")
    async def forget_models(request: BatchForgetRequest) -> dict[str, Any]:
        results = []
        for model_id in request.ids:
            try:
                result = await asyncio.to_thread(library.forget, model_id, request.cleanup)
            except (LookupError, ValueError, OSError) as exc:
                results.append({"id": model_id, "ok": False, "error": str(exc)})
                continue
            results.append({"id": model_id, **_erased(result)})
        return {"results": results}

    @app.post("/api/models/{model_id}/relink")
    async def relink_model(model_id: int, request: RelinkRequest) -> dict[str, Any]:
        with _answering():
            model = await asyncio.to_thread(library.relink, model_id, request.candidate)
        return {"ok": True, "path": model.path}

    @app.post("/api/models/{model_id}/locate")
    async def locate_model(model_id: int) -> dict[str, Any]:
        """Point a missing model at its file, chosen in the system's own dialog.

        The same rule as moving anywhere: the request says which model, the operating system
        asks the person at the keyboard which file, and the page never names one. When the
        file is not the size the model was, the answer is held here under a token and the
        page is asked to confirm it — still without ever sending the path back.
        """
        model = _model(model_id)
        if model.state != db.MISSING:
            raise HTTPException(409, "this model is not missing")
        from ..desktop import pick_system_file

        chosen = await asyncio.to_thread(pick_system_file, str(Path(model.path).parent))
        if not chosen:
            return {"ok": False, "cancelled": True}
        with _answering():
            return await asyncio.to_thread(library.link_to, model_id, Path(chosen))

    @app.post("/api/models/{model_id}/locate/confirm")
    async def confirm_locate(model_id: int, request: ConfirmLocateRequest) -> dict[str, Any]:
        with _answering():
            return await asyncio.to_thread(library.confirm_link, model_id, request.token)

    @app.post("/api/models/{model_id}/bring-back")
    async def bring_back(model_id: int) -> dict[str, Any]:
        with _answering():
            result = await asyncio.to_thread(library.bring_back, model_id)
        return _moved(result)

    @app.post("/api/models/identify")
    async def identify(request: IdsRequest) -> dict[str, Any]:
        """Hash each model and ask Civitai what it is. Only ever on request: hashing reads
        the whole file, and a terabyte of models is hours of a disk's time."""
        return {"queued": library.enqueue("identify", request.ids), "jobs": library.job_state()}

    @app.post("/api/models/verify")
    async def verify(request: IdsRequest) -> dict[str, Any]:
        return {"queued": library.enqueue("verify", request.ids), "jobs": library.job_state()}

    @app.post("/api/models/hash")
    async def hash_models(request: IdsRequest) -> dict[str, Any]:
        return {"queued": library.enqueue("hash", request.ids), "jobs": library.job_state()}

    @app.get("/api/jobs")
    async def jobs() -> dict[str, Any]:
        return library.job_state()

    @app.post("/api/jobs/stop")
    async def stop_jobs() -> dict[str, int]:
        return {"dropped": library.stop_jobs()}

    # --- newer versions -----------------------------------------------------
    #
    # A check runs one at a time and says how far it has got over the event stream; these
    # answer when it is done, with what it found.

    @app.post("/api/models/check-updates")
    async def check_updates(request: IdsRequest) -> dict[str, Any]:
        with _answering():
            return await library.check_updates(request.ids)

    @app.post("/api/updates/check")
    async def check_all_updates() -> dict[str, Any]:
        """Every model that came from Civitai or the Hub — what the app asks on its own when
        it starts, asked for now."""
        with _answering():
            return await library.check_updates(library.checkable_ids())

    @app.post("/api/updates/stop")
    async def stop_update_check() -> dict[str, bool]:
        return {"stopping": library.stop_update_check()}

    @app.post("/api/models/skip-update")
    async def skip_update(request: SkipRequest) -> dict[str, int]:
        return {"changed": library.skip_updates(request.ids, request.skip)}

    @app.post("/api/updates/resolve")
    async def resolve_update(request: IdsRequest) -> dict[str, Any]:
        """A newer version as a link resolved, for the page to ask where it goes. Queued by
        /api/resolve/{token}/queue, like any other."""
        try:
            return await manager.resolve_update(request.ids)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part here
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/updates/download")
    async def download_updates(request: IdsRequest) -> dict[str, Any]:
        """The newer version of each, into the folder of the version it updates."""
        result = await manager.download_updates(request.ids)
        return {"tasks": [t.to_json() for t in result["created"]], "failed": result["failed"]}

    @app.get("/api/models/{model_id}/previews")
    async def model_previews(model_id: int) -> dict[str, Any]:
        model = _model(model_id)
        return {"blur_nsfw": settings.blur_nsfw, "previews": library.previews(model)}

    @app.get("/api/models/{model_id}/preview/{index}")
    async def model_preview(
        model_id: int, index: int, w: int | None = Query(None, ge=32, le=2048)
    ) -> Response:
        model = _model(model_id)
        entries = previews.entries(model.meta)
        if entries:
            if not settings.fetch_previews:
                raise HTTPException(404, "previews are turned off")
            return await _remote_preview(entries, index, w)
        if index != 0:
            raise HTTPException(404, "no such preview")
        # A picture beside the model, or the one inside its header. The path is the model's
        # own, looked up here; the request only says which model.
        path = Path(model.path)
        image = await asyncio.to_thread(details.local_image, path)
        if image is not None:
            return FileResponse(image, media_type=previews.media_type(image),
                                headers={"Cache-Control": "no-cache"})
        inside = await asyncio.to_thread(details.thumbnail, path)
        if inside is not None:
            return Response(content=inside[0], media_type=inside[1],
                            headers={"Cache-Control": "no-cache"})
        raise HTTPException(404, "no picture for this model")

    # --- the workflow inside a sample -------------------------------------------
    #
    # Named the way the pictures are: which model or download, and which of its samples by
    # its place in the list the service gave. The picture read is the one that list names;
    # a workflow is saved into the one folder the settings name, under a name made here.

    finder = workflows.Finder(images)

    def _workflow_folder() -> Path | None:
        chosen = (settings.workflow_dir or "").strip()
        if chosen:
            return Path(chosen)
        return workflows.comfy_folder(settings.roots)

    def _model_samples(model) -> tuple[list[dict[str, Any]], Path | None]:
        """A model's samples as the viewer lists them — the service's, or else the one
        picture it has of its own — and the model's path, for reading that one."""
        entries = previews.entries(model.meta)
        if entries:
            return entries, None
        if model.extras.get("image") or model.header.get("thumbnail"):
            return [{"type": "image", "local": True}], Path(model.path)
        return [], None

    def _task_samples(task_id: int):
        task = database.get(task_id)
        if task is None:
            raise HTTPException(404, "no such task")
        return task, previews.entries(task.meta)

    def _local_workflow(path: Path) -> workflows.Workflow | None:
        image = details.local_image(path)
        if image is not None:
            return workflows.read_file(image)
        inside = details.thumbnail(path)
        return workflows.read_bytes(inside[0]) if inside is not None else None

    def _looked_inside(entry: dict[str, Any]) -> bool:
        """Only still pictures are read, and a remote one only while previews may be
        fetched at all: turned off, no picture is ever requested."""
        if entry.get("type") == "video":
            return False
        return bool(entry.get("local")) or settings.fetch_previews

    async def _look(entry: dict[str, Any], local: Path | None) -> workflows.Workflow | None:
        if entry.get("local"):
            return await asyncio.to_thread(_local_workflow, local) if local else None
        return await finder.find(str(entry["url"]), Path(settings.preview_dir or "previews"))

    async def _workflow_states(
        entries: list[dict[str, Any]], local: Path | None
    ) -> dict[str, Any]:
        """What each sample carries, all of them at once, for the viewer to mark."""

        async def state_of(entry: dict[str, Any]) -> dict[str, Any]:
            if not _looked_inside(entry):
                return {"kind": "skipped"}
            try:
                found = await _look(entry, local)
            except workflows.Unreadable as exc:
                return {"kind": "error", "error": str(exc)}
            if found is None:
                return {"kind": "none"}
            return {"kind": found.kind, "nodes": found.nodes}

        states = await asyncio.gather(*(state_of(entry) for entry in entries))
        return {"workflows": [{"index": i, **state} for i, state in enumerate(states)]}

    async def _workflow_at(
        entries: list[dict[str, Any]], index: int, local: Path | None
    ) -> workflows.Workflow:
        if not 0 <= index < len(entries):
            raise HTTPException(404, "no such sample")
        if not _looked_inside(entries[index]):
            raise HTTPException(404, "this sample is not looked inside")
        try:
            found = await _look(entries[index], local)
        except workflows.Unreadable as exc:
            raise HTTPException(502, f"the picture could not be read: {exc}") from None
        if found is None:
            raise HTTPException(404, "this picture carries no workflow")
        return found

    async def _save_workflow(found: workflows.Workflow, owner: str, index: int) -> dict[str, Any]:
        folder = _workflow_folder()
        if folder is None:
            # Not a failure: the page asks where, once, and tries again.
            return {"ok": False, "needs_folder": True}
        name = workflows.file_name(owner, index, found.kind)
        try:
            path, existed = await asyncio.to_thread(workflows.save, folder, name, found.text)
        except OSError as exc:
            raise HTTPException(500, f"the workflow could not be saved in {folder}: {exc}") from None
        return {"ok": True, "name": path.name, "path": str(path), "folder": str(folder),
                "existed": existed, "kind": found.kind}

    def _workflow_json(found: workflows.Workflow) -> dict[str, Any]:
        return {"kind": found.kind, "nodes": found.nodes, "text": found.text}

    @app.get("/api/models/{model_id}/workflows")
    async def model_workflows(model_id: int) -> dict[str, Any]:
        return await _workflow_states(*_model_samples(_model(model_id)))

    @app.get("/api/models/{model_id}/workflows/{index}")
    async def model_workflow(model_id: int, index: int) -> dict[str, Any]:
        entries, local = _model_samples(_model(model_id))
        return _workflow_json(await _workflow_at(entries, index, local))

    @app.post("/api/models/{model_id}/workflows/{index}/save")
    async def save_model_workflow(model_id: int, index: int) -> dict[str, Any]:
        model = _model(model_id)
        entries, local = _model_samples(model)
        found = await _workflow_at(entries, index, local)
        return await _save_workflow(found, model.filename, index)

    @app.get("/api/tasks/{task_id}/workflows")
    async def task_workflows(task_id: int) -> dict[str, Any]:
        _, entries = _task_samples(task_id)
        return await _workflow_states(entries, None)

    @app.get("/api/tasks/{task_id}/workflows/{index}")
    async def task_workflow(task_id: int, index: int) -> dict[str, Any]:
        _, entries = _task_samples(task_id)
        return _workflow_json(await _workflow_at(entries, index, None))

    @app.post("/api/tasks/{task_id}/workflows/{index}/save")
    async def save_task_workflow(task_id: int, index: int) -> dict[str, Any]:
        task, entries = _task_samples(task_id)
        found = await _workflow_at(entries, index, None)
        return await _save_workflow(found, task.filename, index)

    @app.post("/api/workflows/reveal")
    async def reveal_workflow(request: WorkflowNameRequest) -> dict[str, bool]:
        """Show a saved workflow in Explorer.

        By the name a save reported, and only inside the workflows folder: this hands a path
        to the shell, so the page can say which saved workflow, never where.
        """
        folder = _workflow_folder()
        name = request.name
        if folder is None or Path(name).name != name or not name.lower().endswith(".json"):
            raise HTTPException(404, "no such workflow")
        target = folder / name
        if not target.is_file():
            raise HTTPException(404, "no such workflow")
        from ..desktop import open_system_path

        return {"ok": open_system_path(str(target))}

    @app.post("/api/models/{model_id}/reveal")
    async def reveal_model(model_id: int) -> dict[str, bool]:
        model = _model(model_id)
        from ..desktop import open_system_path

        return {"ok": open_system_path(model.path)}

    @app.get("/api/duplicates")
    async def duplicates() -> dict[str, Any]:
        return await asyncio.to_thread(library.duplicates)

    @app.get("/api/cleanup")
    async def cleanup() -> dict[str, Any]:
        return await asyncio.to_thread(library.cleanup_scan)

    @app.post("/api/cleanup/delete")
    async def cleanup_delete(request: CleanupDeleteRequest) -> dict[str, Any]:
        """Delete items of the last cleanup scan, by their number in it. The files are the
        ones that scan found; the page cannot add one, and a scan it did not see cannot be
        the one its numbers are read against."""
        with _answering():
            result = await asyncio.to_thread(library.cleanup_delete, request.ids, request.token)
        manager.spawn(library.refresh(), "library-after-cleanup")
        return _erased(result)

    # --- the history ------------------------------------------------------

    @app.delete("/api/history/{task_id}")
    async def remove_from_history(task_id: int) -> dict[str, bool]:
        with _answering():
            manager.remove_from_history(task_id)
        return {"ok": True}

    @app.post("/api/history/{task_id}/redownload")
    async def redownload(task_id: int) -> dict[str, Any]:
        with _answering():
            task = manager.redownload(task_id)
        return {"ok": True, "task": task.to_json()}

    # --- the library's folders --------------------------------------------

    def _folders_now() -> tuple[Path | None, tuple[Path, ...]]:
        return settings.library_path, tuple(settings.roots)

    async def _folders_changed(before: tuple[Path | None, tuple[Path, ...]]) -> None:
        """After the library's folders changed: records first, then a walk.

        Which folder is the main one decides where each collected record belongs, so the
        records are moved before anything reads them under the new arrangement.
        """
        old_root, old_roots = before
        new_root, new_roots = _folders_now()
        same_root = (old_root is None) == (new_root is None) and (
            old_root is None or db.path_key(old_root) == db.path_key(new_root)
        )
        same_roots = [db.path_key(r) for r in old_roots] == [db.path_key(r) for r in new_roots]
        if not (same_root and same_roots):
            await asyncio.to_thread(library.remap_records, old_root, old_roots)
        await library.refresh()

    @app.post("/api/roots/add")
    async def add_root() -> dict[str, Any]:
        """Add a folder to the library, chosen in the system's own dialog."""
        from ..desktop import pick_system_folder

        chosen = await asyncio.to_thread(pick_system_folder, settings.library_root or "")
        if not chosen:
            return {"ok": False, "cancelled": True}
        path = Path(chosen)
        if not path.is_dir():
            raise HTTPException(400, f"{chosen} is not a folder")
        known = {db.path_key(r) for r in settings.roots}
        if db.path_key(path) in known:
            return {"ok": True, "unchanged": True, "roots": library.roots_json()}
        before = _folders_now()
        if not settings.library_root:
            # With no library at all, the first folder added becomes it: downloads have
            # been landing in a flat folder only because there was nowhere better.
            settings.library_root = str(path)
        else:
            settings.extra_roots.append(str(path))
        settings.save()
        await _folders_changed(before)
        return {"ok": True, "roots": library.roots_json()}

    @app.post("/api/roots/{index}/remove")
    async def remove_root(index: int) -> dict[str, Any]:
        """Stop reading a folder into the library. Nothing on disk is touched."""
        roots = settings.roots
        if not 0 <= index < len(roots):
            raise HTTPException(404, "no such library folder")
        target = db.path_key(roots[index])
        before = _folders_now()
        if index == 0:
            if not settings.library_root:
                raise HTTPException(
                    409, "this is the downloads folder — it is where files go until a "
                         "library folder is set, so it cannot be taken off the list"
                )
            settings.library_root = settings.extra_roots.pop(0) if settings.extra_roots else ""
        else:
            settings.extra_roots = [
                r for r in settings.extra_roots if db.path_key(r) != target
            ]
        settings.save()
        await _folders_changed(before)
        return {"ok": True, "roots": library.roots_json()}

    @app.post("/api/roots/{index}/primary")
    async def make_primary(index: int) -> dict[str, Any]:
        """Make a folder the one downloads are filed into."""
        roots = settings.roots
        if not 0 <= index < len(roots):
            raise HTTPException(404, "no such library folder")
        if index == 0:
            return {"ok": True, "unchanged": True, "roots": library.roots_json()}
        chosen = roots[index]
        before = _folders_now()
        previous = settings.library_root
        settings.extra_roots = [
            r for r in settings.extra_roots if db.path_key(r) != db.path_key(chosen)
        ]
        if previous:
            settings.extra_roots.insert(0, previous)
        settings.library_root = str(chosen)
        settings.save()
        await _folders_changed(before)
        return {"ok": True, "roots": library.roots_json()}

    @app.post("/api/folders")
    async def new_folder(request: NewFolderRequest) -> dict[str, Any]:
        with _answering():
            made = await asyncio.to_thread(
                library.new_folder, request.root, request.parent, request.name
            )
        await library.refresh()
        return {"ok": True, "path": str(made)}

    @app.post("/api/folders/reveal")
    async def reveal_folder(request: FolderRequest) -> dict[str, bool]:
        target = _library_folder(request.root, request.relative)
        from ..desktop import open_system_path

        return {"ok": open_system_path(str(target))}

    @app.post("/api/folders/hide")
    async def hide_folder(request: FolderRequest) -> dict[str, Any]:
        """Leave a folder out of the library — a code checkout, a node's test data."""
        if not request.relative.strip("/"):
            raise HTTPException(400, "a whole library folder is taken off the list, not hidden")
        target = _library_folder(request.root, request.relative)
        if db.path_key(target) not in {db.path_key(p) for p in settings.exclude_dirs}:
            settings.exclude_dirs.append(str(target))
            settings.save()
        await library.refresh()
        return {"ok": True, "hidden": settings.exclude_dirs}

    @app.post("/api/folders/unhide")
    async def unhide_folder(request: UnhideRequest) -> dict[str, Any]:
        if not 0 <= request.index < len(settings.exclude_dirs):
            raise HTTPException(404, "no such hidden folder")
        settings.exclude_dirs.pop(request.index)
        settings.save()
        await library.refresh()
        return {"ok": True, "hidden": settings.exclude_dirs}

    @app.put("/api/ui")
    async def put_ui(patch: UiPatch) -> dict[str, bool]:
        """Remember how the window was left. A null value forgets a key."""
        merged = dict(settings.ui or {})
        for key, value in patch.prefs.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[str(key)[:64]] = value
        if len(json.dumps(merged, ensure_ascii=False)) > UI_LIMIT:
            raise HTTPException(413, "too much to remember")
        settings.ui = merged
        settings.save()
        return {"ok": True}

    # --- settings ---------------------------------------------------------

    def _settings_json() -> dict[str, Any]:
        data = settings.redacted()
        # What an empty "Save workflows to" means right now, for the form to show.
        found = workflows.comfy_folder(settings.roots)
        data["workflow_dir_found"] = str(found) if found else ""
        # The folder the app's own files are in: the one relative paths are resolved
        # against, which the launcher makes the data folder. Beside the exe, as a rule — in
        # the user's application data when that could not be written.
        data["data_dir"] = str(Path.cwd())
        return data

    @app.post("/api/data-dir/reveal")
    async def reveal_data_dir() -> dict[str, bool]:
        from ..desktop import open_system_path

        return {"ok": open_system_path(str(Path.cwd()))}

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {
            "settings": _settings_json(),
            "categories": [c.value for c in Category],
            "error": settings.error,
        }

    @app.put("/api/settings")
    async def put_settings(patch: SettingsPatch) -> dict[str, Any]:
        # Only what was actually sent: the form posts a subset, and filling the rest in from
        # the model's defaults would quietly reset every field it does not show.
        before = _folders_now()
        settings.apply(patch.model_dump(exclude_unset=True))
        settings.save()
        old_root, old_roots = before
        if [db.path_key(r) for r in old_roots] != [db.path_key(r) for r in settings.roots] \
                or (old_root is None) != (settings.library_path is None):
            await asyncio.to_thread(library.remap_records, old_root, old_roots)
        # "Files at once" and the speed ceiling are live controls, not ones that wait for a
        # restart — you reach for them precisely while something is downloading.
        manager.apply_settings()
        return {"settings": _settings_json()}

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


@contextlib.contextmanager
def _answering():
    """Turn the ways a library operation can refuse into the status codes that say so."""
    try:
        yield
    except HTTPException:
        raise
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, f"the library already has a model at that path ({exc})") from None
    except LookupError as exc:
        raise HTTPException(404, str(exc).strip("'\"")) from None
    except Busy as exc:
        raise HTTPException(409, str(exc)) from None
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from None
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    except OSError as exc:
        raise HTTPException(500, str(exc)) from None


def _listing(paths) -> list[dict[str, Any]]:
    listed = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        listed.append({"path": str(path), "name": path.name, "size": size})
    return listed


def _failures(failed) -> list[dict[str, str]]:
    # Named rather than counted: "two sidecars stayed behind" is not something a person can
    # act on without knowing which ones.
    return [{"path": str(p), "reason": reason} for p, reason in failed]


def _moved(result, remembered: str | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "unchanged": result.unchanged,
        "dest": str(result.path),
        "moved": len(result.companions),
        "failed": _failures(result.failed),
        "remembered": remembered,
    }


def _erased(result) -> dict[str, Any]:
    return {
        "ok": True,
        "deleted": [str(p) for p in result.deleted],
        "missing": result.missing,
        "failed": _failures(result.failed),
    }


__all__ = ["create_app", "db", "contextlib"]
