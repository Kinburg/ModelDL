"""The ComfyUI workflow a sample picture was made with, when the picture still carries it.

ComfyUI writes the graph it ran into every picture it saves. A PNG gets two text chunks:
`workflow`, the graph as the editor draws it — what Ctrl+V on its canvas opens — and
`prompt`, the same run in the format of its API. WebP, and the JPEG savers that copied the
idea, put the same pair into EXIF as `workflow:{...}` and `prompt:{...}`. Civitai serves an
upload untouched at its `original=true` address, so the graph a sample's author published is
still inside the file — also for the many samples whose page on the site shows no generation
data at all, because the author hid it or the site did not read it.

Only the start of a picture is read. PNG and JPEG keep their metadata ahead of the pixels, so
the read stops where the image data begins: tens of kilobytes instead of the several
megabytes a full-size sample weighs. WebP is the exception — its EXIF comes after the image —
and is read whole, unless its header already says there is none. Whatever the answer, found
or not, it is kept beside the preview cache and that picture is never read again.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..providers.base import sanitize_filename
from . import previews

# What a picture can carry: the graph the editor opens, or only the API's format of the run.
GRAPH = "graph"
API = "api"

# The two keys ComfyUI writes, and the only ones read. The rest of a picture's text —
# A1111's `parameters`, a camera's maker notes — belongs to somebody else.
KEYS = ("workflow", "prompt")

# A workflow is tens of kilobytes; a big one with notes in it, a few hundred. A text chunk
# that would unpack into more than this is not something anybody is going to paste.
MAX_TEXT = 16 * 1024 * 1024
# How much of a remote picture is read before giving up — the ceiling a preview has.
MAX_BYTES = previews.MAX_BYTES
# A picture beside the model is on disk anyway, so it is read whole.
MAX_LOCAL = 64 * 1024 * 1024

CACHE_SUFFIX = ".workflow.json"
CACHE_VERSION = 1

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT = (b"tEXt", b"zTXt", b"iTXt")


@dataclass(frozen=True, slots=True)
class Workflow:
    kind: str       # GRAPH or API
    text: str       # the JSON, ready to paste or to save
    nodes: int


class Unreadable(Exception):
    """The picture could not be had this time. A failure, not an answer: never remembered."""


# --- looking inside a picture ------------------------------------------------------


def scan(
    data: bytes | bytearray, *, complete: bool = True, whole: bool = False
) -> tuple[bool, Workflow | None]:
    """Look for a workflow in the start of a picture: (settled, found).

    Unsettled means the answer may still be further on — the bytes stopped inside a chunk,
    or before the pixels began — and only happens while `complete` is false, that is while
    more of the file could still be read. `whole` also looks past a PNG's pixels, which is
    worth it only for a file that is on disk anyway.
    """
    head = bytes(data[:12])
    if head[:8] == PNG_SIGNATURE:
        texts, settled = _png(data, whole)
    elif head[:3] == b"\xff\xd8\xff":
        texts, settled = _jpeg(data)
    elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        texts, settled = _webp(data)
    elif len(head) < 12 and not complete:
        return False, None
    else:
        return True, None

    found = _pick(texts)
    if settled or complete or (found is not None and found.kind == GRAPH):
        return True, found
    return False, None


def _png(data: bytes | bytearray, whole: bool) -> tuple[dict[str, str], bool]:
    texts: dict[str, str] = {}
    pos = 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        if kind == b"IEND" or (kind == b"IDAT" and not whole):
            return texts, True
        end = pos + 12 + length
        if kind in _PNG_TEXT:
            if end > len(data):
                return texts, False
            entry = _png_text(kind, bytes(data[pos + 8 : pos + 8 + length]))
            if entry is not None:
                texts.setdefault(*entry)
                if "workflow" in texts:
                    return texts, True
        pos = end
    return texts, False


def _png_text(kind: bytes, body: bytes) -> tuple[str, str] | None:
    keyword, sep, rest = body.partition(b"\0")
    key = keyword.decode("latin-1")
    if not sep or key not in KEYS:
        return None
    if kind == b"tEXt":
        raw: bytes | None = rest
    elif kind == b"zTXt":
        raw = _inflate(rest[1:]) if rest[:1] == b"\0" else None
    else:
        # iTXt: a compression flag and method, then a language tag and a translated
        # keyword, each ended by a zero byte, and only then the text.
        if len(rest) < 2:
            return None
        compressed, method, tail = rest[0], rest[1], rest[2:]
        _, first, tail = tail.partition(b"\0")
        _, second, tail = tail.partition(b"\0")
        if not (first and second):
            return None
        raw = (_inflate(tail) if method == 0 else None) if compressed else tail
    return (key, _text(raw)) if raw is not None else None


def _inflate(data: bytes) -> bytes | None:
    """Unpack a compressed text chunk, refusing one that would unpack into something huge."""
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(data, MAX_TEXT)
    except zlib.error:
        return None
    return None if inflater.unconsumed_tail else out


def _jpeg(data: bytes | bytearray) -> tuple[dict[str, str], bool]:
    texts: dict[str, str] = {}
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            return texts, True           # not a marker: whatever follows is not metadata
        marker = data[pos + 1]
        if marker == 0xFF:               # padding before a marker
            pos += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):       # the pixels begin, or the file ends
            return texts, True
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        if length < 2:
            return texts, True
        end = pos + 2 + length
        if marker == 0xE1:
            if end > len(data):
                return texts, False
            segment = bytes(data[pos + 4 : end])
            if segment[:6] == b"Exif\0\0":
                for key, value in _exif(segment[6:]).items():
                    texts.setdefault(key, value)
                if "workflow" in texts:
                    return texts, True
        pos = end
    return texts, False


def _webp(data: bytes | bytearray) -> tuple[dict[str, str], bool]:
    pos = 12
    while pos + 8 <= len(data):
        fourcc = bytes(data[pos : pos + 4])
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        end = pos + 8 + size
        if pos == 12:
            # A simple WebP is one image chunk and nothing else, and an extended one says
            # in its first chunk whether EXIF follows — so a file with none is settled here
            # rather than read to its end looking for it.
            if fourcc != b"VP8X":
                return {}, True
            if pos + 9 > len(data):
                return {}, False
            if not data[pos + 8] & 0x08:
                return {}, True
        if fourcc == b"EXIF":
            if end > len(data):
                return {}, False
            payload = bytes(data[pos + 8 : end])
            if payload[:6] == b"Exif\0\0":
                payload = payload[6:]
            return _exif(payload), True
        pos = end + (size & 1)
    return {}, False


def _exif(tiff: bytes) -> dict[str, str]:
    """ComfyUI's two keys from a TIFF block, as `workflow:{...}` in its first directory.

    That is where ComfyUI writes them for WebP — on tags counting down from 0x010F, the
    camera's make, since EXIF has no field meant for this — and the only place its page
    looks, so a picture that kept them anywhere else could not be opened there either.
    """
    order = {b"II": "<", b"MM": ">"}.get(tiff[:2])
    if order is None or len(tiff) < 8:
        return {}
    magic, first = struct.unpack(order + "HI", tiff[2:8])
    if magic != 42 or first + 2 > len(tiff):
        return {}
    (count,) = struct.unpack(order + "H", tiff[first : first + 2])
    found: dict[str, str] = {}
    for number in range(min(count, 1024)):
        at = first + 2 + 12 * number
        if at + 12 > len(tiff):
            break
        _tag, kind, size = struct.unpack(order + "HHI", tiff[at : at + 8])
        if kind not in (2, 7):           # ASCII, or UNDEFINED bytes
            continue
        if size <= 4:
            raw = tiff[at + 8 : at + 8 + size]
        else:
            (offset,) = struct.unpack(order + "I", tiff[at + 8 : at + 12])
            raw = tiff[offset : offset + size]
        key, sep, value = _text(raw.rstrip(b"\0")).partition(":")
        if sep and key in KEYS:
            found.setdefault(key, value)
    return found


def _text(raw: bytes) -> str:
    """Text the way ComfyUI's page decodes it: UTF-8, which plain JSON already is.

    The PNG standard says Latin-1 for `tEXt`; a chunk that is not valid UTF-8 is read that
    way rather than dropped.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# --- what was found ------------------------------------------------------------------


def _pick(texts: dict[str, str]) -> Workflow | None:
    """The graph when there is one; the API's format only when that is all there is."""
    return _graph(texts.get("workflow")) or _api(texts.get("prompt"))


def _graph(text: str | None) -> Workflow | None:
    data, odd = _load(text)
    if not isinstance(data, dict):
        return None
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return None
    # ComfyUI's paste handler opens a graph only when `version`, `nodes` and `extra` are all
    # there, and parses it strictly: an older save without `extra`, or a NaN some node
    # wrote, would paste as nothing at all, without a word.
    changed = odd
    if not _truthy(data.get("version")):
        data["version"] = 0.4
        changed = True
    if not _truthy(data.get("extra")):
        data["extra"] = {}
        changed = True
    body = json.dumps(data, ensure_ascii=False) if changed else str(text).strip()
    return Workflow(GRAPH, body, len(nodes))


def _api(text: str | None) -> Workflow | None:
    data, _ = _load(text)
    if not isinstance(data, dict) or not data:
        return None
    # The test ComfyUI applies before it opens a file as an API prompt. Kept as written, NaN
    # and all: ComfyUI reads files leniently, and this format only ever goes into a file.
    if not all(
        isinstance(node, dict)
        and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict)
        for node in data.values()
    ):
        return None
    return Workflow(API, str(text).strip(), len(data))


def _load(text: str | None) -> tuple[Any, bool]:
    """Parse JSON, noting whether it held NaN or Infinity — Python writes those, and the
    JavaScript that reads a paste refuses them. They come back as null."""
    if not text:
        return None, False
    odd: list[str] = []
    try:
        data = json.loads(text, parse_constant=lambda name: odd.append(name))
    except (ValueError, RecursionError):
        return None, False
    return data, bool(odd)


def _truthy(value: Any) -> bool:
    """JavaScript's idea of it, since the check being met is written in JavaScript: an
    empty object counts, zero and the empty string do not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and not math.isnan(value)
    if isinstance(value, str):
        return value != ""
    return value is not None


# --- reading one ---------------------------------------------------------------------


async def read(url: str, client: httpx.AsyncClient) -> Workflow | None:
    """What the picture at `url` carries, reading no more of it than that takes.

    The same host allowlist as the previews: the URL came in an API response, which is
    remote data. A picture that is gone is an answer — there is nothing in it any more; a
    refusal, a server error or a page served in place of the picture is not.
    """
    if not previews.allowed(url):
        return None
    buffer = bytearray()
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code in (404, 410):
                return None
            if response.status_code != 200:
                raise Unreadable(f"the service answered {response.status_code}")
            kind = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            if kind.startswith("text/"):
                raise Unreadable("the service sent a page instead of the picture")
            if kind.startswith(("video/", "audio/")):
                return None
            async for chunk in response.aiter_bytes():
                buffer += chunk
                settled, found = scan(buffer, complete=False)
                if settled:
                    return found
                if len(buffer) >= MAX_BYTES:
                    return None
    except httpx.HTTPError as exc:
        raise Unreadable(str(exc) or type(exc).__name__) from exc
    return scan(buffer, complete=True)[1]


def read_file(path: Path) -> Workflow | None:
    """What a picture on disk carries — the one named after a model, for instance."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_LOCAL)
    except OSError:
        return None
    return read_bytes(data)


def read_bytes(data: bytes) -> Workflow | None:
    return scan(data, complete=True, whole=True)[1]


# --- remembering the answer ------------------------------------------------------------


def cache_path(directory: Path | str, url: str) -> Path:
    """Keyed by URL like the previews, under a suffix of its own beside them."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return Path(directory) / f"{digest}{CACHE_SUFFIX}"


def cached(directory: Path | str, url: str) -> tuple[bool, Workflow | None]:
    """(known, found): whether this picture was read before, and what it carried."""
    try:
        data = json.loads(cache_path(directory, url).read_text("utf-8"))
    except (OSError, ValueError):
        return False, None
    if not isinstance(data, dict) or data.get("v") != CACHE_VERSION:
        return False, None
    kind = data.get("kind")
    if kind is None:
        return True, None
    if kind in (GRAPH, API) and isinstance(data.get("text"), str):
        return True, Workflow(kind, data["text"], int(data.get("nodes") or 0))
    return False, None


def remember(directory: Path | str, url: str, found: Workflow | None) -> None:
    """Keep the answer. Best effort: a cache that cannot be written only costs a re-read."""
    payload: dict[str, Any] = {"v": CACHE_VERSION, "url": url, "kind": found.kind if found else None}
    if found is not None:
        payload.update(nodes=found.nodes, text=found.text)
    target = cache_path(directory, url)
    tmp = target.with_name(target.name + ".tmp")
    with contextlib.suppress(OSError):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        os.replace(tmp, target)


class Finder:
    """Looks inside remote pictures, each one once.

    The viewer asks about all of a model's samples as it opens, and two windows — or the
    same viewer closed and opened again — can ask about the same pictures at the same time;
    a second question waits for the read already running instead of starting its own. A few
    pictures are read at a time, not a screenful.
    """

    def __init__(self, client: httpx.AsyncClient, at_once: int = 4) -> None:
        self._client = client
        self._gate = asyncio.Semaphore(at_once)
        self._reading: dict[str, asyncio.Future[Workflow | None]] = {}

    async def find(self, url: str, directory: Path) -> Workflow | None:
        url = previews.original_url(url)
        known, found = await asyncio.to_thread(cached, directory, url)
        if known:
            return found
        key = str(cache_path(directory, url))
        task = self._reading.get(key)
        if task is None:
            task = asyncio.ensure_future(self._read(url, directory))
            self._reading[key] = task
            task.add_done_callback(lambda done, key=key: self._settle(key, done))
        # Shielded: a window closed mid-read abandons its question, not everybody's answer.
        return await asyncio.shield(task)

    def _settle(self, key: str, done: asyncio.Future[Workflow | None]) -> None:
        self._reading.pop(key, None)
        if not done.cancelled():
            done.exception()    # looked at, so a failure nobody waited for is not reported lost

    async def _read(self, url: str, directory: Path) -> Workflow | None:
        async with self._gate:
            # Asked again inside the gate: a read that finished while this one waited for
            # its turn has already answered it.
            known, found = await asyncio.to_thread(cached, directory, url)
            if known:
                return found
            found = await read(url, self._client)
        await asyncio.to_thread(remember, directory, url, found)
        return found


# --- where a saved one goes -------------------------------------------------------------


def comfy_folder(roots: Iterable[Path]) -> Path | None:
    """ComfyUI's own workflows folder, when a library folder is the `models` of an install.

    `<ComfyUI>/models` sits beside `<ComfyUI>/user/default/workflows` in the portable build,
    a git checkout and the desktop app alike, and what is saved there appears in ComfyUI's
    Workflows panel. A shared folder read through `extra_model_paths.yaml` has no such
    neighbour and is passed over.
    """
    for root in roots:
        root = Path(root)
        if root.name.lower() != "models":
            continue
        user = root.parent / "user" / "default"
        if user.is_dir():
            return user / "workflows"
    return None


def file_name(model_name: str, index: int, kind: str) -> str:
    """`<model> - sample 2.json`: named after the file the workflow came with.

    The model's own name, as ComfyUI's loaders list it, is what ties a saved workflow back
    to the LoRA or checkpoint it was published with; the number is the one the viewer
    shows. An API-format prompt says so in its name, because ComfyUI opens that kind only
    when the file is dropped onto its canvas — its Workflows panel cannot.
    """
    stem = model_name.rsplit(".", 1)[0] if "." in model_name.strip(".") else model_name
    stem = sanitize_filename(stem)[:120].rstrip(". ") or "workflow"
    return f"{stem} - sample {index + 1}{' (API)' if kind == API else ''}.json"


def save(folder: Path, name: str, text: str) -> tuple[Path, bool]:
    """Write a workflow into `folder`, never over a different file: (where, already there).

    The same sample saved twice finds its first copy and says so. A different workflow under
    a name already taken — another download of the model, a copy edited in ComfyUI since —
    gets ` (2)`, ` (3)` rather than replacing it.
    """
    folder.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    stem = name[: -len(".json")] if name.lower().endswith(".json") else name
    for number in range(1, 100):
        target = folder / (f"{stem}.json" if number == 1 else f"{stem} ({number}).json")
        try:
            handle = open(target, "xb")
        except FileExistsError:
            with contextlib.suppress(OSError):
                if target.read_bytes() == data:
                    return target, True
            continue
        try:
            with handle:
                handle.write(data)
        except OSError:
            with contextlib.suppress(OSError):
                target.unlink()
            raise
        return target, False
    raise FileExistsError(f"{folder} already holds too many workflows named {stem}")
