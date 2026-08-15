"""Civitai provider.

Civitai's shape differs from the Hub's in three ways that matter:

* **A version is not a file.** One model version routinely carries several quantisations —
  bf16, fp8, mxfp8, int8, nf4 — and the download button you clicked determines which. The
  API publishes them as `files[]`, each selectable by `fileId` or by its precision.

* **They can all share one filename.** The API returns the same `name` for every variant of
  a version, so writing two of them into the same folder silently destroys the first. Names
  have to be disambiguated on our side.

* **Metadata is public, downloads are not.** `/api/v1/model-versions/{id}` answers without
  credentials, so variants and sizes can be listed before anyone is asked for a key; the
  download endpoint returns 401 without one.

The API is also where the useful cataloguing data lives: model type (checkpoint, LoRA, VAE),
base model, and trigger words — which are what make a downloaded LoRA usable later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from ..core.errors import AccessDenied, AuthRequired, ResolveError
from ..core.types import FileIdentity, RemoteFileInfo, ResolvedTarget
from .base import (
    Provider,
    Walk,
    parse_signed_expiry,
    read_info,
    sanitize_filename,
    walk_redirects,
)

# civitai.red is a full mirror: same API, same payloads, same auth. People reach for it when
# .com is blocked where they are, so the domain they arrived with is preserved rather than
# rewritten — sending them back to .com could be sending them nowhere.
HOSTS = {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}
DEFAULT_HOST = "civitai.com"

_AIR = re.compile(
    r"^urn:air:(?P<ecosystem>[^:]*):(?P<type>[^:]*):civitai:(?P<model>\d+)(@(?P<version>\d+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CivitaiRef:
    version_id: int | None = None
    model_id: int | None = None
    file_id: int | None = None
    # Set when the input was a bare /api/download/ link. That URL means "the primary file",
    # so expanding it into every variant of the version would be a five-fold surprise.
    wants_primary: bool = False
    # Civitai's own per-variant links select with ?fp=&format=&size=&type=. Stored as a
    # tuple of pairs so the reference stays hashable and comparable.
    variant: tuple[tuple[str, str], ...] = ()
    # Which mirror the link came from. Transport detail, like the token — deliberately not
    # part of the identity, so the same file pasted from either domain is one download.
    host: str = DEFAULT_HOST

    @property
    def is_file(self) -> bool:
        return self.version_id is not None and self.file_id is not None

    @property
    def endpoint(self) -> str:
        return f"https://{self.host}"

    @property
    def page_url(self) -> str:
        if self.model_id is not None:
            url = f"{self.endpoint}/models/{self.model_id}"
            return f"{url}?modelVersionId={self.version_id}" if self.version_id else url
        return f"{self.endpoint}/model-versions/{self.version_id}"


@dataclass(slots=True)
class CivitaiFile:
    file_id: int
    filename: str                 # already disambiguated
    size: int | None
    sha256: str | None
    primary: bool
    meta: dict[str, Any] = field(default_factory=dict)


def is_civitai_url(value: str) -> bool:
    if _AIR.match(value.strip()):
        return True
    host = urlsplit(value if "://" in value else f"https://{value}").netloc.lower()
    return host.split(":")[0] in HOSTS


def parse_ref(value: str) -> CivitaiRef:
    """Accept any of the shapes a user can end up holding."""
    text = value.strip()
    if not text:
        raise ValueError("empty reference")

    air = _AIR.match(text)
    if air:
        version = air.group("version")
        return CivitaiRef(
            version_id=int(version) if version else None,
            model_id=int(air.group("model")),
        )

    parts = urlsplit(text if "://" in text else f"https://{text}")
    host = parts.netloc.split(":")[0].lower()
    if host not in HOSTS:
        raise ValueError(f"not a Civitai URL: {value}")
    host = host.removeprefix("www.")

    segments = [s for s in parts.path.strip("/").split("/") if s]
    query = {k.lower(): v for k, v in parse_qs(parts.query).items()}
    file_id = _first_int(query.get("fileid"))

    # https://civitai.com/api/download/models/<versionId>
    if segments[:3] == ["api", "download", "models"] and len(segments) >= 4:
        variant = _variant_from(query)
        return CivitaiRef(
            version_id=_as_int(segments[3]),
            file_id=file_id,
            wants_primary=file_id is None and not variant,
            variant=variant,
            host=host,
        )

    # https://civitai.com/models/<modelId>[/slug][?modelVersionId=...]
    if segments[:1] == ["models"] and len(segments) >= 2:
        return CivitaiRef(
            version_id=_first_int(query.get("modelversionid")),
            model_id=_as_int(segments[1]),
            file_id=file_id,
            host=host,
        )

    # https://civitai.com/model-versions/<versionId>
    if segments[:1] == ["model-versions"] and len(segments) >= 2:
        return CivitaiRef(version_id=_as_int(segments[1]), file_id=file_id, host=host)

    raise ValueError(f"cannot tell a model from {value}")


def make_identity(ref: CivitaiRef) -> FileIdentity:
    """Identity for the transfer layer — no token, no signed URL, so resume survives both."""
    if not ref.is_file:
        raise ValueError(
            "a model version may hold several files — expand it and pick one first"
        )
    return FileIdentity(
        provider="civitai",
        ref={"version_id": ref.version_id, "file_id": ref.file_id},
    )


def ref_from_identity(identity: FileIdentity, host: str = DEFAULT_HOST) -> CivitaiRef:
    return CivitaiRef(
        version_id=identity.ref["version_id"],   # type: ignore[arg-type]
        file_id=identity.ref["file_id"],         # type: ignore[arg-type]
        host=host,
    )


class CivitaiProvider(Provider):
    name = "civitai"

    def __init__(self, token: str | None = None, host: str = DEFAULT_HOST) -> None:
        self._token = token
        self.host = host if host in HOSTS else DEFAULT_HOST
        self._versions: dict[int, dict[str, Any]] = {}
        # Civitai serves from more than one CDN. Backblaze answers HEAD normally; Cloudflare
        # R2 hands out URLs presigned for GET and rejects HEAD on them with 403. Once we
        # have seen that, stop paying for the failed probe.
        self._head_works = True

    @property
    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    @property
    def _api(self) -> str:
        return f"https://{self.host}/api/v1"

    # --- transfer interface ----------------------------------------------

    async def resolve(self, identity: FileIdentity, client: httpx.AsyncClient) -> ResolvedTarget:
        ref = ref_from_identity(identity, self.host)
        walk = await self._walk_download(ref, client)
        return ResolvedTarget(
            url=walk.url,
            headers=walk.auth_headers,
            expires_at=parse_signed_expiry(walk.url),
            info=None,   # the API is a better source; see probe()
        )

    async def _walk_download(self, ref: CivitaiRef, client: httpx.AsyncClient) -> Walk:
        """Resolve the download URL, tolerating a CDN that refuses HEAD.

        A presigned URL's signature covers the HTTP method, so a storage backend can answer
        HEAD with 403 while serving GET perfectly — and the URL from that rejected HEAD is
        still good for ranged reads. Only a refusal from civitai.com itself, before any
        redirect, is a real access problem worth reporting.
        """
        url = f"{ref.endpoint}/api/download/models/{ref.version_id}?fileId={ref.file_id}"

        if self._head_works:
            walk = await walk_redirects(client, url, self._auth, "HEAD")
            if walk.status < 400:
                return walk
            if walk.hops == 0:
                self._raise(walk.status, walk, ref)
            self._head_works = False

        walk = await walk_redirects(client, url, self._auth, "GET")
        if walk.status >= 400:
            self._raise(walk.status, walk, ref)
        return walk

    async def probe(self, identity: FileIdentity, client: httpx.AsyncClient) -> RemoteFileInfo:
        """Metadata comes from the API, which is authoritative and needs no token.

        The download endpoint would answer 401 for an anonymous user, so asking it first
        would turn "here is what you are about to fetch" into "log in" for no reason.
        """
        ref = ref_from_identity(identity, self.host)
        assert ref.version_id is not None
        version = await self.version_info(ref.version_id, client)
        entry = self._find_file(version, ref.file_id)
        if entry is None:
            raise AccessDenied(
                f"file {ref.file_id} is not part of model version {ref.version_id}"
            )

        # Range support is the one thing the API cannot tell us. Both Civitai CDNs support
        # it, so a transport hiccup here should not block a download — but a genuine
        # permission answer must still surface, and early.
        accept_ranges = True
        try:
            accept_ranges = read_info(await self._walk_download(ref, client)).accept_ranges
        except (AuthRequired, AccessDenied):
            raise
        except ResolveError:
            pass

        return RemoteFileInfo(
            filename=entry.filename,
            size=entry.size,
            sha256=entry.sha256,
            etag=None,
            accept_ranges=accept_ranges,
            content_type=None,
            meta=entry.meta,
        )

    # --- catalogue queries -----------------------------------------------

    async def version_info(self, version_id: int, client: httpx.AsyncClient) -> dict[str, Any]:
        if version_id in self._versions:
            return self._versions[version_id]
        data = await self._get(client, f"{self._api}/model-versions/{version_id}")
        self._versions[version_id] = data
        return data

    async def resolve_version(self, ref: CivitaiRef, client: httpx.AsyncClient) -> CivitaiRef:
        """Fill in a missing version id by taking the model's newest one."""
        if ref.version_id is not None or ref.model_id is None:
            return ref
        model = await self._get(client, f"{self._api}/models/{ref.model_id}")
        versions = model.get("modelVersions") or []
        if not versions:
            raise AccessDenied(f"model {ref.model_id} has no published versions")
        return CivitaiRef(
            version_id=versions[0].get("id"), model_id=ref.model_id, file_id=ref.file_id
        )

    async def list_files(
        self, ref: CivitaiRef, client: httpx.AsyncClient
    ) -> list[CivitaiFile]:
        ref = await self.resolve_version(ref, client)
        assert ref.version_id is not None
        version = await self.version_info(ref.version_id, client)
        files = _describe_files(version)
        if ref.file_id is not None:
            return [f for f in files if f.file_id == ref.file_id]
        if ref.variant:
            matched = [f for f in files if _matches_variant(f, ref.variant)]
            if not matched:
                wanted = ", ".join(f"{k}={v}" for k, v in ref.variant)
                raise AccessDenied(f"no file in this version matches {wanted}")
            return matched
        if ref.wants_primary:
            primary = [f for f in files if f.primary]
            return primary or files[:1]
        return files

    async def _get(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        try:
            response = await client.get(url, headers=self._auth, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise ResolveError(f"Civitai API request failed: {exc}") from exc
        if response.status_code >= 400:
            self._raise(response.status_code, None, None)
        try:
            return response.json()
        except ValueError as exc:
            raise ResolveError("Civitai returned a non-JSON response") from exc

    # --- errors -----------------------------------------------------------

    def _raise(self, status: int, walk, ref: CivitaiRef | None) -> None:
        if status == 401:
            raise AuthRequired(
                "Civitai requires an API key for downloads — create one under Account "
                "settings on civitai.com and put it in settings, or set $CIVITAI_TOKEN"
                if not self._token
                else "Civitai rejected the API key"
            )
        if status == 403:
            where = f" ({ref.page_url})" if ref else ""
            raise AccessDenied(
                "Civitai refused the download — the model may be in early access, which "
                f"has to be purchased before it can be fetched{where}"
            )
        if status == 404:
            raise AccessDenied("no such model, version or file on Civitai")
        raise ResolveError(f"Civitai returned {status}")

    @staticmethod
    def _find_file(version: dict[str, Any], file_id: int | None) -> CivitaiFile | None:
        for entry in _describe_files(version):
            if entry.file_id == file_id:
                return entry
        return None


# --- file description -------------------------------------------------------


def _describe_files(version: dict[str, Any]) -> list[CivitaiFile]:
    """Turn the API's `files[]` into something safe to write to disk.

    The disambiguation is the point. Every variant of a version can carry the identical
    `name`, so five downloads would land on one path and leave one file. A suffix is only
    added where a name is actually shared, so the common single-file case is untouched.
    """
    raw = version.get("files") or []
    names: dict[str, int] = {}
    for entry in raw:
        name = (entry.get("name") or "").strip()
        names[name] = names.get(name, 0) + 1

    model = version.get("model") or {}
    # The first image on the version is what the site shows as its thumbnail.
    images = version.get("images") or []
    preview = next(
        (img.get("url") for img in images if img.get("type", "image") == "image" and img.get("url")),
        None,
    )
    described: list[CivitaiFile] = []
    for entry in raw:
        name = sanitize_filename((entry.get("name") or "model.safetensors").strip())
        metadata = entry.get("metadata") or {}
        if names.get((entry.get("name") or "").strip(), 0) > 1:
            name = _tag_filename(name, metadata)

        size_kb = entry.get("sizeKB")
        hashes = entry.get("hashes") or {}
        sha256 = hashes.get("SHA256")

        described.append(
            CivitaiFile(
                file_id=entry.get("id"),
                filename=name,
                size=int(size_kb * 1024) if size_kb else None,
                sha256=sha256.lower() if isinstance(sha256, str) else None,
                primary=bool(entry.get("primary")),
                meta={
                    "source": "civitai",
                    "version_id": version.get("id"),
                    "version_name": version.get("name"),
                    "model_id": (version.get("modelId") or model.get("id")),
                    "model_name": model.get("name"),
                    # These three decide where the file belongs on disk and how to use it.
                    "model_type": model.get("type"),
                    "base_model": version.get("baseModel"),
                    "trained_words": version.get("trainedWords") or [],
                    "nsfw": model.get("nsfw"),
                    "preview_url": preview,
                    "file_type": entry.get("type"),
                    "precision": metadata.get("fp"),
                    "format": metadata.get("format"),
                    "quantisation": metadata.get("size"),
                },
            )
        )
    return described


def _tag_filename(name: str, metadata: dict[str, Any]) -> str:
    """Insert the distinguishing detail before the extension: `model.bf16.safetensors`."""
    tag = metadata.get("fp") or metadata.get("size") or metadata.get("format")
    if not tag:
        return name
    tag = sanitize_filename(str(tag)).lower().replace(" ", "-")
    stem, dot, ext = name.rpartition(".")
    return f"{stem}.{tag}{dot}{ext}" if dot else f"{name}.{tag}"


_VARIANT_KEYS = ("fp", "format", "size", "type")


def _variant_from(query: dict[str, list[str]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, query[key][0]) for key in _VARIANT_KEYS if query.get(key) and query[key][0]
    )


def _matches_variant(entry: CivitaiFile, variant: tuple[tuple[str, str], ...]) -> bool:
    fields = {
        "fp": entry.meta.get("precision"),
        "format": entry.meta.get("format"),
        "size": entry.meta.get("quantisation"),
        "type": entry.meta.get("file_type"),
    }
    return all(
        str(fields.get(key) or "").lower() == value.lower() for key, value in variant
    )


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(values: list[str] | None) -> int | None:
    return _as_int(values[0]) if values else None
