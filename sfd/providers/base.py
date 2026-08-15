"""Provider contract, plus the HTTP machinery every provider needs.

A provider knows how to turn a stable `FileIdentity` into a fresh, ready-to-fetch URL, and
how to answer questions about the file. It knows nothing about chunks, resume or retries —
that is the transfer layer's job. The transfer layer, in turn, knows nothing about
HuggingFace or Civitai. The only thing crossing the boundary is `resolve()`, which the
transfer calls again every time a signature goes stale.

The redirect walking below is shared because getting it wrong is subtle and expensive:
dropping credentials at the right moment, never sending a Range on a probe, and collecting
metadata that is scattered across several hops.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import httpx

from ..core.errors import AccessDenied, AuthRequired, NotBinaryContent, ResolveError
from ..core.types import FileIdentity, RemoteFileInfo, ResolvedTarget

HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
MAX_HOPS = 10
REDIRECT_CODES = {301, 302, 303, 307, 308}

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_FILENAME_STAR = re.compile(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", re.IGNORECASE)
_FILENAME = re.compile(r"filename\s*=\s*\"([^\"]+)\"|filename\s*=\s*([^;]+)", re.IGNORECASE)
SHA256_RE = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)


class Provider(ABC):
    name: str

    @abstractmethod
    async def resolve(self, identity: FileIdentity, client: httpx.AsyncClient) -> ResolvedTarget:
        """Mint a fresh download URL for `identity`.

        Called once at the start of a transfer and again whenever the previous URL expires
        or is rejected. Implementations should populate `ResolvedTarget.info` when the
        answer comes for free with the same round trip.
        """

    async def probe(self, identity: FileIdentity, client: httpx.AsyncClient) -> RemoteFileInfo:
        """Metadata about the file. Defaults to whatever `resolve()` learned."""
        target = await self.resolve(identity, client)
        if target.info is None:
            raise NotImplementedError(f"{self.name} provider returned no file info")
        return target.info


# --- redirect walking -------------------------------------------------------


@dataclass(slots=True)
class Walk:
    """The outcome of following a redirect chain once."""

    canonical: str
    url: str                                  # where the chain ended
    status: int
    # Headers from every hop, later ones winning. HuggingFace puts x-linked-etag and
    # x-linked-size on its 302 and the real content-length only appears on the CDN
    # response, so neither hop alone tells the whole story.
    merged: dict[str, str] = field(default_factory=dict)
    final: dict[str, str] = field(default_factory=dict)
    auth_headers: dict[str, str] = field(default_factory=dict)
    authenticated: bool = False
    hops: int = 0                             # redirects actually followed


async def walk_redirects(
    client: httpx.AsyncClient,
    canonical: str,
    auth_headers: dict[str, str] | None = None,
    method: str = "HEAD",
    extra: dict[str, str] | None = None,
) -> Walk:
    """Follow redirects by hand, collecting headers from every hop.

    Two things this does that `follow_redirects=True` does not:

    * Drops `Authorization` when the redirect crosses to another host. Presigned S3/R2 URLs
      carry their credentials in the query string, and sending a bearer token alongside
      them makes the storage layer reject the request with "only one auth mechanism
      allowed" — this is why `curl -L -H "Authorization: ..."` fails against Civitai.

    * Keeps headers from intermediate hops. The origin's 302 is often the only place the
      real content hash and size appear.

    The body is never read; responses are closed as soon as the headers arrive.
    """
    auth_headers = dict(auth_headers or {})
    url = canonical
    headers = dict(auth_headers)
    merged: dict[str, str] = {}
    hops = 0

    for _ in range(MAX_HOPS):
        request_headers = {**headers, "Accept-Encoding": "identity", **(extra or {})}
        try:
            async with client.stream(
                method, url, headers=request_headers, follow_redirects=False
            ) as resp:
                hop = {k.lower(): v for k, v in resp.headers.items()}
                merged.update(hop)

                if resp.status_code in REDIRECT_CODES:
                    location = resp.headers.get("location")
                    if not location:
                        raise ResolveError(f"{resp.status_code} without Location header")
                    nxt = urljoin(url, location)
                    if origin_of(nxt) != origin_of(url):
                        headers.pop("Authorization", None)
                        headers.pop("authorization", None)
                    url = nxt
                    hops += 1
                    continue

                return Walk(
                    canonical=canonical,
                    url=url,
                    status=resp.status_code,
                    merged=merged,
                    final=hop,
                    auth_headers=headers,
                    authenticated=bool(auth_headers),
                    hops=hops,
                )
        except httpx.HTTPError as exc:
            raise ResolveError(f"could not resolve {canonical}: {exc}") from exc

    raise ResolveError(f"too many redirects while resolving {canonical}")


# --- interpreting a walk ----------------------------------------------------


def check_status(walk: Walk) -> None:
    if walk.status == 401:
        raise AuthRequired("the service requires a token for this file")
    if walk.status == 403:
        raise AccessDenied(
            "access denied (gated repo, unaccepted licence, or invalid token)"
            if walk.authenticated
            else "access denied — a token is probably required"
        )
    if walk.status == 404:
        raise AccessDenied("file not found")
    if walk.status >= 400:
        raise ResolveError(f"unexpected status {walk.status} while resolving")


def check_content(walk: Walk) -> None:
    """Reject an HTML body where a file was expected.

    An unauthenticated Civitai request lands on a login page served as 200 text/html, which
    otherwise gets saved as a 12 KB `.safetensors`.
    """
    content_type = (walk.final.get("content-type") or "").split(";")[0].strip().lower()
    if content_type in HTML_CONTENT_TYPES:
        raise NotBinaryContent(
            "the server returned an HTML page instead of a file — "
            "this usually means a login or consent page, so a token is required"
        )


def read_info(walk: Walk, filename: str | None = None) -> RemoteFileInfo:
    """Build file metadata from everything the walk collected."""
    merged, final = walk.merged, walk.final

    size: int | None = None
    accept_ranges = (merged.get("accept-ranges") or "").lower() == "bytes"

    # A 206 reports the range length in Content-Length; only Content-Range carries the
    # total, so it has to win when both are present.
    match = _CONTENT_RANGE.search(merged.get("content-range", ""))
    if match and match.group(3) != "*":
        size = int(match.group(3))
        accept_ranges = True
    if size is None and walk.status == 200:
        raw = final.get("content-length")
        if raw and raw.isdigit():
            size = int(raw)
    if size is None:
        raw = merged.get("x-linked-size")
        if raw and raw.isdigit():
            size = int(raw)

    etag = final.get("etag") or merged.get("etag")

    # Only `x-linked-etag`, set by the origin on its redirect, is the content's SHA256.
    #
    # The CDN's own ETag must never be used for this, however much it looks the part. On
    # Xet-backed HuggingFace repos it is a 64-hex Xet content id — the same value that
    # appears in the CDN path — and it is *not* the SHA256 of the bytes. Trusting it means
    # computing a perfectly correct digest, comparing it against an unrelated hash, and
    # quarantining a flawless 18 GB download as corrupt.
    #
    # An unrecognised ETag simply leaves sha256 unset, and the transfer skips verification
    # rather than failing it. `etag` is still kept: comparing it against itself across a
    # resume is valid regardless of what it means.
    sha256 = None
    linked = merged.get("x-linked-etag")
    if linked:
        cleaned = linked.strip().removeprefix("W/").strip('"')
        if SHA256_RE.fullmatch(cleaned):
            sha256 = cleaned.lower()

    content_type = (final.get("content-type") or "").split(";")[0].strip().lower()

    return RemoteFileInfo(
        filename=filename or filename_from(merged, walk.canonical, walk.url),
        size=size,
        sha256=sha256,
        etag=etag,
        accept_ranges=accept_ranges,
        content_type=content_type or None,
    )


# --- small helpers ----------------------------------------------------------


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def filename_from(headers: dict[str, str], canonical: str, final_url: str) -> str:
    disposition = headers.get("content-disposition", "")
    if disposition:
        star = _FILENAME_STAR.search(disposition)
        if star:
            return sanitize_filename(unquote(star.group(1).strip()))
        plain = _FILENAME.search(disposition)
        if plain:
            name = sanitize_filename((plain.group(1) or plain.group(2) or "").strip())
            if name:
                return name

    # Prefer the URL the user gave us. A CDN path is often the object's hash, which would
    # otherwise become the filename.
    for url in (canonical, final_url):
        name = sanitize_filename(unquote(urlsplit(url).path.rsplit("/", 1)[-1]))
        if name and not SHA256_RE.fullmatch(name):
            return name
    return "download.bin"


def sanitize_filename(name: str) -> str:
    """Strip path separators and characters Windows refuses in filenames."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip().strip('"')
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    return name.rstrip(". ")


def parse_signed_expiry(url: str) -> float | None:
    """Best-effort expiry (unix seconds) read out of a presigned URL's query string.

    Knowing this lets us re-sign *before* a long-running read dies, instead of discovering
    the problem as a 403 halfway through a chunk.
    """
    q = {k.lower(): v for k, v in parse_qs(urlsplit(url).query).items()}

    # SigV4 (S3, Cloudflare R2): signing timestamp plus a validity window.
    if "x-amz-date" in q and "x-amz-expires" in q:
        try:
            signed = datetime.strptime(q["x-amz-date"][0], "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
            return signed.timestamp() + int(q["x-amz-expires"][0])
        except (ValueError, IndexError):
            pass

    # CloudFront canned policy, and HuggingFace's cas-bridge links, use absolute stamps.
    for key in ("expires", "exp", "se"):
        if key in q:
            try:
                value = int(q[key][0])
            except (ValueError, IndexError):
                continue
            # Some services use milliseconds.
            return value / 1000 if value > 1e11 else float(value)

    return None
