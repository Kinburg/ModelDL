"""The sample images a model is published with.

They are not decoration. A folder of LoRAs named `add_detail_v3.safetensors` is unreadable
by name alone, and the picture answers "which one is that" instantly; the picture's own
`meta` carries the prompt that produced it, which is the same class of thing as trigger
words — published by the service, discarded by every plain download, unrecoverable
afterwards.

Nothing here is fetched ahead of time. The page asks for a thumbnail when it draws one,
this module fetches it once and keeps it under a cache keyed by URL, and every later
request — including every other file of the same model version, which shares the images —
is answered from disk. The cache is disposable: deleting it costs one round trip per
picture and nothing else.

The URLs come from a service's API response, which is remote data, and this module hands
them to an HTTP client running on the user's machine. That is the shape of a server-side
request forgery, so the scheme and host are checked against a list rather than trusted.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

# Enough for a full-size sample or a short animated one. Thumbnails are a hundredth of it.
MAX_BYTES = 24 * 1024 * 1024
# What the queue rows ask for. Civitai resizes on its own CDN, so a row thumbnail costs
# ~40 KB rather than the 3 MB of the original.
THUMB_WIDTH = 320
CACHE_SUFFIX = ".img"

# Hosts whose images we are willing to fetch. Nothing else: a compromised or simply
# creative API response must not be able to point this at 169.254.169.254 or at a service
# listening on another port of this machine.
ALLOWED_HOSTS = ("civitai.com", "civitai.red", "huggingface.co", "hf.co")

# Civitai's CDN takes its transformations as a path segment — `/width=450/`, `/original=true/`,
# `/anim=false,width=320/` — which is what lets a thumbnail be asked for by rewriting a URL.
_TRANSFORM = re.compile(r"^[a-z]+=[^/,]*$", re.IGNORECASE)

_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"GIF8", 0, "image/gif"),
    (b"RIFF", 0, "image/webp"),          # narrowed below by the WEBP tag at offset 8
    (b"ftyp", 4, "video/mp4"),
    (b"\x1a\x45\xdf\xa3", 0, "video/webm"),
)


def entries(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """The previews recorded for a file, tolerating the shape we used to store.

    A queue assembled by an older build has only `preview_url`, one image and no metadata.
    Those tasks are still in the database and still deserve a thumbnail, so the single URL
    is presented as a list of one rather than treated as nothing.
    """
    recorded = meta.get("previews")
    if isinstance(recorded, list):
        found = [e for e in recorded if isinstance(e, dict) and e.get("url")]
        if found:
            return found
    url = meta.get("preview_url")
    return [{"url": str(url), "type": "image"}] if url else []


def allowed(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme != "https":
        return False
    host = (parts.hostname or "").lower()
    return any(host == name or host.endswith("." + name) for name in ALLOWED_HOSTS)


def variant_url(url: str, width: int | None = None, still: bool = False) -> str:
    """Ask the CDN for a smaller copy, or for a video's poster frame.

    Only Civitai's image host understands this; anything else is handed back untouched
    rather than mangled into a 404.
    """
    if width is None and not still:
        return url
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not (host.endswith("civitai.com") or host.endswith("civitai.red")):
        return url

    operations = ([f"width={width}"] if width else []) + (["anim=false"] if still else [])
    if not operations:
        return url

    segments = parts.path.split("/")
    for index, segment in enumerate(segments):
        if segment and all(_TRANSFORM.match(part) for part in segment.split(",")):
            segments[index] = ",".join(operations)
            break
    else:
        # No transformation in the path — insert one before the filename, which is where
        # theirs always sits.
        if len(segments) < 2:
            return url
        segments.insert(len(segments) - 1, ",".join(operations))
    return urlunsplit(parts._replace(path="/".join(segments)))


def cache_path(directory: Path | str, url: str) -> Path:
    """Where a fetched image lives. Keyed by URL, so the width is part of the key.

    Content-addressing by URL rather than by task means the five quantisations of one
    Civitai version share a single copy of each picture instead of five.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return Path(directory) / f"{digest}{CACHE_SUFFIX}"


async def fetch(url: str, directory: Path | str, client: httpx.AsyncClient) -> Path | None:
    """Put an image in the cache and return where it landed, or None if it cannot be had.

    Streamed and counted rather than read whole: `content` on a response that turns out to
    be a video file nobody asked for would be in memory before there was a chance to refuse
    it.
    """
    if not allowed(url):
        return None

    target = cache_path(directory, url)
    if target.exists():
        return target

    chunks: list[bytes] = []
    total = 0
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != 200:
                return None
            kind = (response.headers.get("content-type") or "").split(";")[0].strip()
            if not kind.startswith(("image/", "video/")):
                return None
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_BYTES:
                    return None
                chunks.append(chunk)
    except httpx.HTTPError:
        return None

    if not total:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, b"".join(chunks))
    return target


def media_type(path: Path) -> str:
    """What a cached file actually is.

    Never what its name says: the cache names everything `.img`, and Civitai serves JPEG
    and WebP from URLs ending in `.png` often enough that trusting the extension would have
    the browser refuse pictures that are perfectly fine.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return "application/octet-stream"
    return sniff(head)


def sniff(head: bytes) -> str:
    for magic, offset, kind in _SIGNATURES:
        if head[offset : offset + len(magic)] == magic:
            if magic == b"RIFF":
                return "image/webp" if head[8:12] == b"WEBP" else "application/octet-stream"
            return kind
    return "application/octet-stream"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
