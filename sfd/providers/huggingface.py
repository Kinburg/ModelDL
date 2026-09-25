"""HuggingFace provider.

Adds three things on top of the generic redirect walk:

* **Link parsing.** Every URL shape the Hub shows a human resolves to the same file:
  the `/blob/` page you land on from search, the `/resolve/` link the download button
  produces, a `/tree/` directory, or a bare `org/name`. Users paste whichever one they
  happen to have.

* **Stable identity.** A download is keyed by `(repo, type, revision, path)` and never by a
  URL. The token is held by the provider, not the identity, so changing it does not
  invalidate a half-finished download.

* **Errors that say what to do.** The Hub reports the difference between "this repo does not
  exist", "you have not accepted the licence" and "your token is wrong" in `x-error-code`,
  and each needs a different action from the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import httpx

from ..core.errors import AccessDenied, AuthRequired, ResolveError, SfdError
from ..core.types import FileIdentity, RemoteFileInfo, ResolvedTarget
from .base import (
    Provider,
    Walk,
    check_content,
    parse_signed_expiry,
    read_info,
    sanitize_filename,
    walk_redirects,
)

HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
ENDPOINT = "https://huggingface.co"

# URL segments that separate the repo id from a revision and path.
VERBS = {"blob", "resolve", "tree", "raw", "blame"}

_TYPE_FROM_SEGMENT = {"models": "model", "datasets": "dataset", "spaces": "space"}
_PREFIX = {"model": "", "dataset": "datasets/", "space": "spaces/"}
_API_SEGMENT = {"model": "models", "dataset": "datasets", "space": "spaces"}


@dataclass(frozen=True, slots=True)
class HfRef:
    repo_id: str
    repo_type: str = "model"
    revision: str = "main"
    path: str | None = None      # None means the repository root
    # A /tree/ link and a repo root both name a directory, not a file. Without this a
    # folder would be turned into a download task pointing at a path that serves nothing.
    is_directory: bool = False

    @property
    def is_file(self) -> bool:
        return self.path is not None and not self.is_directory

    @property
    def page_url(self) -> str:
        return f"{ENDPOINT}/{_PREFIX[self.repo_type]}{self.repo_id}"

    def download_url(self) -> str:
        if not self.is_file:
            raise ValueError("cannot build a download URL for a directory")
        return (
            f"{ENDPOINT}/{_PREFIX[self.repo_type]}{self.repo_id}"
            f"/resolve/{quote(self.revision, safe='')}/{quote(self.path, safe='/')}"
        )

    def api_url(self, suffix: str = "") -> str:
        return f"{ENDPOINT}/api/{_API_SEGMENT[self.repo_type]}/{self.repo_id}{suffix}"


@dataclass(slots=True)
class HfFile:
    path: str
    size: int | None
    sha256: str | None           # LFS/Xet oid, when the entry is not stored inline


def is_hf_url(value: str) -> bool:
    host = urlsplit(value if "://" in value else f"https://{value}").netloc.lower()
    return host.split(":")[0] in HOSTS


def parse_ref(value: str) -> HfRef:
    """Turn any Hub URL — or a bare `org/name` — into a reference.

    Raises ValueError for anything that is not recognisably a Hub location, so callers can
    fall back to another provider.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty reference")

    if "://" in text or urlsplit(f"https://{text}").netloc.split(":")[0] in HOSTS:
        parts = urlsplit(text if "://" in text else f"https://{text}")
        if parts.netloc.split(":")[0].lower() not in HOSTS:
            raise ValueError(f"not a HuggingFace URL: {value}")
        segments = [unquote(s) for s in parts.path.strip("/").split("/") if s]
    else:
        # A bare "org/name" (or a legacy single-segment repo like "gpt2").
        segments = [unquote(s) for s in text.strip("/").split("/") if s]
        if not 1 <= len(segments) <= 2:
            raise ValueError(f"not a HuggingFace reference: {value}")

    if not segments:
        raise ValueError(f"no repository in {value}")

    repo_type = "model"
    if segments[0] in _TYPE_FROM_SEGMENT:
        repo_type = _TYPE_FROM_SEGMENT[segments[0]]
        segments = segments[1:]
    if not segments:
        raise ValueError(f"no repository in {value}")

    verb_at = next((i for i, s in enumerate(segments) if s in VERBS), None)
    if verb_at is None:
        if not 1 <= len(segments) <= 2:
            raise ValueError(f"cannot tell the repository from {value}")
        return HfRef(repo_id="/".join(segments), repo_type=repo_type, is_directory=True)

    repo_segments = segments[:verb_at]
    if not 1 <= len(repo_segments) <= 2:
        raise ValueError(f"cannot tell the repository from {value}")

    is_directory = segments[verb_at] == "tree"
    rest = segments[verb_at + 1 :]
    if not rest:
        return HfRef(
            repo_id="/".join(repo_segments), repo_type=repo_type, is_directory=True
        )

    # Branch names may contain slashes: pull requests live at refs/pr/N and the automatic
    # conversions at refs/convert/<format>. Those consume three segments, not one.
    if rest[0] == "refs" and len(rest) >= 3:
        revision, path_parts = "/".join(rest[:3]), rest[3:]
    else:
        revision, path_parts = rest[0], rest[1:]

    path = "/".join(path_parts) or None
    return HfRef(
        repo_id="/".join(repo_segments),
        repo_type=repo_type,
        revision=revision,
        path=path,
        is_directory=is_directory or path is None,
    )


def make_identity(ref: HfRef) -> FileIdentity:
    """Identity for the transfer layer.

    Note what is absent: the token and the URL. A download keyed on either would restart
    from zero the moment a signature is re-minted or a token is rotated.
    """
    if not ref.is_file:
        what = "a whole repository" if ref.path is None else f"the directory {ref.path}"
        raise ValueError(f"{what} is not a single transfer — expand it into files first")
    return FileIdentity(
        provider="huggingface",
        ref={
            "repo_id": ref.repo_id,
            "repo_type": ref.repo_type,
            "revision": ref.revision,
            "path": ref.path,
        },
    )


def ref_from_identity(identity: FileIdentity) -> HfRef:
    return HfRef(
        repo_id=identity.ref["repo_id"],          # type: ignore[arg-type]
        repo_type=identity.ref["repo_type"],      # type: ignore[arg-type]
        revision=identity.ref["revision"],        # type: ignore[arg-type]
        path=identity.ref["path"],                # type: ignore[arg-type]
    )


class HuggingFaceProvider(Provider):
    name = "huggingface"

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._repo_meta: dict[str, dict[str, Any]] = {}

    @property
    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    # --- transfer interface ----------------------------------------------

    async def resolve(self, identity: FileIdentity, client: httpx.AsyncClient) -> ResolvedTarget:
        ref = ref_from_identity(identity)
        walk = await walk_redirects(client, ref.download_url(), self._auth, "HEAD")
        if walk.status >= 400 and walk.hops == 0:
            _raise_for_hf(walk, ref, authenticated=bool(self._token))
        return ResolvedTarget(
            url=walk.url,
            headers=walk.auth_headers,
            expires_at=parse_signed_expiry(walk.url),
            info=self._info(walk, ref) if walk.status < 400 else None,
        )

    async def probe(self, identity: FileIdentity, client: httpx.AsyncClient) -> RemoteFileInfo:
        ref = ref_from_identity(identity)
        walk = await walk_redirects(client, ref.download_url(), self._auth, "HEAD")
        if walk.status >= 400:
            _raise_for_hf(walk, ref, authenticated=bool(self._token))
        check_content(walk)
        return self._info(walk, ref)

    def _info(self, walk: Walk, ref: HfRef) -> RemoteFileInfo:
        # The filename comes from the repo path, not from the CDN URL — Xet serves objects
        # under their content hash, which would otherwise become the filename on disk.
        assert ref.path is not None
        info = read_info(walk, filename=sanitize_filename(ref.path.rsplit("/", 1)[-1]))
        info.meta.update(
            {
                "repo_id": ref.repo_id,
                "repo_type": ref.repo_type,
                "revision": ref.revision,
                "path": ref.path,
                # The exact commit behind a moving branch name. Worth recording: it is what
                # makes "resume this next week" meaningful.
                "commit": walk.merged.get("x-repo-commit"),
            }
        )
        return info

    # --- repository queries ----------------------------------------------

    async def repo_info(self, ref: HfRef, client: httpx.AsyncClient) -> dict[str, Any]:
        """Repository metadata, cached per revision. Feeds model-type classification."""
        key = f"{ref.repo_type}:{ref.repo_id}@{ref.revision}"
        if key in self._repo_meta:
            return self._repo_meta[key]

        url = ref.api_url(f"/revision/{quote(ref.revision, safe='')}")
        response = await self._api_get(client, url, ref)
        data = response.json()
        self._repo_meta[key] = data
        return data

    async def list_files(self, ref: HfRef, client: httpx.AsyncClient) -> list[HfFile]:
        """Every file in the repo (or under `ref.path`), for whole-repo downloads."""
        suffix = f"/tree/{quote(ref.revision, safe='')}"
        if ref.path:
            suffix += f"/{quote(ref.path, safe='/')}"
        url = ref.api_url(suffix)

        files: list[HfFile] = []
        cursor: str | None = url
        params: dict[str, str] | None = {"recursive": "true", "expand": "true"}
        while cursor:
            response = await self._api_get(client, cursor, ref, params=params)
            for entry in response.json():
                if entry.get("type") != "file":
                    continue
                lfs = entry.get("lfs") or {}
                oid = lfs.get("oid") or lfs.get("sha256")
                files.append(
                    HfFile(
                        path=entry["path"],
                        size=lfs.get("size") or entry.get("size"),
                        sha256=oid if isinstance(oid, str) and len(oid) == 64 else None,
                    )
                )
            # The tree endpoint paginates through a Link header once a repo gets large.
            cursor = _next_link(response)
            params = None
        return files

    async def _api_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        ref: HfRef,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = await client.get(
                url, headers=self._auth, params=params, follow_redirects=True
            )
        except httpx.HTTPError as exc:
            raise ResolveError(f"HuggingFace API request failed: {exc}") from exc
        if response.status_code >= 400:
            _raise_for_hf_headers(
                response.status_code,
                {k.lower(): v for k, v in response.headers.items()},
                ref,
                authenticated=bool(self._token),
            )
        return response


# --- error mapping ----------------------------------------------------------


def _raise_for_hf(walk: Walk, ref: HfRef, authenticated: bool) -> None:
    _raise_for_hf_headers(walk.status, walk.merged, ref, authenticated)


def _raise_for_hf_headers(
    status: int, headers: dict[str, str], ref: HfRef, authenticated: bool
) -> None:
    """Turn the Hub's error signalling into something a user can act on.

    `x-error-code` is the useful part: without it a private repo and a mistyped name are
    both a bare 404, and a gated repo and a bad token are both a 401.
    """
    code = (headers.get("x-error-code") or "").strip()
    detail = (headers.get("x-error-message") or "").strip()

    if code == "GatedRepo" or (status == 403 and "gated" in detail.lower()):
        if not authenticated:
            raise AuthRequired(
                f"{ref.repo_id} is gated — accept the terms at {ref.page_url}, then give "
                f"this program a token of that account: in Settings, in $HF_TOKEN, or with "
                f"hf auth login"
            )
        raise AccessDenied(
            f"{ref.repo_id} is gated — open {ref.page_url} and accept the terms with the "
            f"same account the token belongs to"
        )
    if code == "RepoNotFound":
        if authenticated:
            raise _coded(AccessDenied(
                f"{ref.repo_id} does not exist, or the token has no access to it"
            ), code)
        raise _coded(AuthRequired(
            f"{ref.repo_id} does not exist or is private — set a token if it is private"
        ), code)
    if code == "EntryNotFound":
        raise _coded(AccessDenied(f"{ref.path} is not in {ref.repo_id} at revision {ref.revision}"), code)
    if code == "RevisionNotFound":
        raise _coded(AccessDenied(f"{ref.repo_id} has no revision {ref.revision}"), code)

    if status == 401:
        raise AuthRequired(
            "HuggingFace rejected the token" if authenticated
            else f"{ref.repo_id} requires a token"
        )
    if status == 403:
        raise AccessDenied(detail or f"access to {ref.repo_id} was refused")
    if status == 404:
        raise AccessDenied(detail or f"{ref.repo_id} not found")
    raise ResolveError(detail or f"HuggingFace returned {status}")


def _coded(error: SfdError, code: str) -> SfdError:
    """The error, carrying the Hub's name for it: the check for newer versions tells a file
    taken down from a refusal by that, and not by the wording."""
    error.code = code
    return error


def _next_link(response: httpx.Response) -> str | None:
    """Pull rel="next" out of a Link header."""
    for part in (response.headers.get("link") or "").split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        if any('rel="next"' in s.replace(" ", "") for s in section[1:]):
            return section[0].strip().strip("<>")
    return None
