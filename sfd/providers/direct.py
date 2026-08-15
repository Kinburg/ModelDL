"""Plain-URL provider.

The generic case: keep the *canonical* URL the user pasted and walk its redirect chain
again on every resolve, so each reconnect gets a freshly signed CDN target. All the HTTP
subtlety lives in `base` — this module is just the policy for when to probe with what.

The one rule worth restating here: **never send a Range on a probe.** HuggingFace's Xet
bridge binds the signature it hands out to the byte range of the request that triggered it,
so resolving with `Range: bytes=0-0` yields a URL that answers `403 Auth failed: invalid
range` to every real chunk request. A probe that quietly invalidates the URL it returns is
worse than no probe at all.
"""

from __future__ import annotations

import httpx

from ..core.types import FileIdentity, RemoteFileInfo, ResolvedTarget
from .base import (
    Provider,
    check_content,
    check_status,
    parse_signed_expiry,
    read_info,
    walk_redirects,
)


def make_identity(url: str, headers: dict[str, str] | None = None) -> FileIdentity:
    ref: dict[str, object] = {"url": url}
    if headers:
        ref["headers"] = dict(sorted(headers.items()))
    return FileIdentity(provider="direct", ref=ref)


class DirectProvider(Provider):
    name = "direct"

    async def resolve(self, identity: FileIdentity, client: httpx.AsyncClient) -> ResolvedTarget:
        """Mint a URL fit for ranged transfer.

        Deliberately lenient about the final status: a presigned URL's signature covers the
        HTTP method, so storage backends may answer HEAD with 403 while serving GET
        perfectly. Permission problems are diagnosed by `probe()`, which runs first.
        """
        url, auth = _unpack(identity)
        walk = await walk_redirects(client, url, auth, "HEAD")

        if walk.status in (405, 501) or (walk.status >= 400 and walk.hops == 0):
            # The origin refuses HEAD outright, so the redirect chain never even started.
            # Repeat the walk with GET — and with no Range header, so the signature we get
            # back stays valid for arbitrary ranges.
            walk = await walk_redirects(client, url, auth, "GET")

        return ResolvedTarget(
            url=walk.url,
            headers=walk.auth_headers,
            expires_at=parse_signed_expiry(walk.url),
            info=read_info(walk) if walk.status < 400 else None,
        )

    async def probe(self, identity: FileIdentity, client: httpx.AsyncClient) -> RemoteFileInfo:
        """Authoritative metadata, with strict error reporting."""
        url, auth = _unpack(identity)
        walk = await walk_redirects(client, url, auth, "HEAD")
        if walk.status < 400:
            check_content(walk)
            info = read_info(walk)
            if info.size is not None:
                return info

        # HEAD is unsupported or uninformative. Fall back to a one-byte ranged GET, which
        # every server answers — and throw the resulting URL away, because on Xet-backed
        # repos it is now bound to that single byte and useless for the actual transfer.
        walk = await walk_redirects(client, url, auth, "GET", {"Range": "bytes=0-0"})
        check_status(walk)
        check_content(walk)
        return read_info(walk)


def _unpack(identity: FileIdentity) -> tuple[str, dict[str, str]]:
    url: str = identity.ref["url"]  # type: ignore[assignment]
    headers: dict[str, str] = dict(identity.ref.get("headers") or {})  # type: ignore[arg-type]
    return url, headers
