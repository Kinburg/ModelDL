"""Turning whatever the user pasted into concrete download tasks.

One entry point, because "paste any link and it works" is most of the usability of this
tool. It accepts a Hub page URL, a direct download link, a bare `org/name`, or a plain HTTP
URL to anything else, and answers with a list of files — a repository or a folder expands
into its contents rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..core.types import FileIdentity
from . import civitai as cv
from . import huggingface as hf
from .base import Provider
from .direct import DirectProvider, make_identity as direct_identity


@dataclass(slots=True)
class Item:
    """One file to fetch."""

    identity: FileIdentity
    filename: str
    size: int | None = None
    sha256: str | None = None
    # Provider metadata for later folder placement (repo tags, model type, base model).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Resolution:
    provider: Provider
    items: list[Item]
    label: str                      # what the user asked for, for display

    @property
    def total_size(self) -> int | None:
        sizes = [i.size for i in self.items]
        return sum(s for s in sizes if s) if all(s is not None for s in sizes) else None


async def expand(
    text: str,
    client: httpx.AsyncClient,
    *,
    hf_token: str | None = None,
    civitai_token: str | None = None,
) -> Resolution:
    """Resolve user input into the set of files it names."""
    text = text.strip()
    if not text:
        raise ValueError("nothing to download")

    if cv.is_civitai_url(text):
        return await _expand_civitai(text, client, civitai_token)

    if _looks_like_hub(text):
        return await _expand_hf(text, client, hf_token)

    headers = {"Authorization": f"Bearer {civitai_token}"} if civitai_token else None
    identity = direct_identity(text, headers)
    return Resolution(
        provider=DirectProvider(),
        # Size and name are unknown until the transfer probes; leaving them unset is
        # honest, and the transfer fills them in.
        items=[Item(identity=identity, filename="")],
        label=text,
    )


def source_url(identity: FileIdentity, host: str | None = None) -> str | None:
    """Rebuild a human-usable link from a stored identity.

    Identities deliberately hold no URL — that is what makes them survive re-signing and
    token rotation — so the link has to be reconstructed when one is needed for a record.
    """
    ref = identity.ref
    if identity.provider == "huggingface":
        return hf.HfRef(
            repo_id=str(ref["repo_id"]),
            repo_type=str(ref["repo_type"]),
            revision=str(ref["revision"]),
            path=str(ref["path"]),
        ).download_url()
    if identity.provider == "civitai":
        return (
            f"https://{host or cv.DEFAULT_HOST}/api/download/models/{ref['version_id']}"
            f"?fileId={ref['file_id']}"
        )
    if identity.provider == "direct":
        return str(ref.get("url")) or None
    return None


def _looks_like_hub(text: str) -> bool:
    if hf.is_hf_url(text):
        return True
    # A bare "org/name" with no scheme and no dots in the first segment is a Hub repo id;
    # anything with a scheme or a hostname belongs to the direct provider.
    if "://" in text or "." in text.split("/")[0]:
        return False
    try:
        hf.parse_ref(text)
    except ValueError:
        return False
    return True


async def _expand_civitai(
    text: str, client: httpx.AsyncClient, token: str | None
) -> Resolution:
    ref = cv.parse_ref(text)
    provider = cv.CivitaiProvider(token, host=ref.host)
    ref = await provider.resolve_version(ref, client)
    files = await provider.list_files(ref, client)
    if not files:
        raise ValueError(f"no downloadable files in {ref.page_url}")

    items = [
        Item(
            identity=cv.make_identity(
                cv.CivitaiRef(version_id=ref.version_id, file_id=entry.file_id)
            ),
            filename=entry.filename,
            size=entry.size,
            sha256=entry.sha256,
            # The mirror travels with the task, not with the identity: the same file from
            # either domain stays one download, but keeps talking to the domain that works.
            meta={**entry.meta, "host": ref.host},
        )
        for entry in files
    ]
    first = files[0].meta
    label = " / ".join(
        str(part) for part in (first.get("model_name"), first.get("version_name")) if part
    ) or ref.page_url
    return Resolution(provider=provider, items=items, label=label)


async def _expand_hf(
    text: str, client: httpx.AsyncClient, token: str | None
) -> Resolution:
    ref = hf.parse_ref(text)
    provider = hf.HuggingFaceProvider(token)

    if ref.is_file:
        assert ref.path is not None
        return Resolution(
            provider=provider,
            items=[
                Item(
                    identity=hf.make_identity(ref),
                    filename=ref.path.rsplit("/", 1)[-1],
                    meta={"repo_id": ref.repo_id, "path": ref.path},
                )
            ],
            label=f"{ref.repo_id}/{ref.path}",
        )

    files = await provider.list_files(ref, client)
    items = [
        Item(
            identity=hf.make_identity(
                hf.HfRef(ref.repo_id, ref.repo_type, ref.revision, entry.path)
            ),
            filename=entry.path.rsplit("/", 1)[-1],
            size=entry.size,
            sha256=entry.sha256,
            meta={"repo_id": ref.repo_id, "path": entry.path},
        )
        for entry in files
    ]
    scope = f"{ref.repo_id}/{ref.path}" if ref.path else ref.repo_id
    return Resolution(provider=provider, items=items, label=scope)
