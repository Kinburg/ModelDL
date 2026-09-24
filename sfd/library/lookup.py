"""Asking the services what a file is, and where it could be downloaded from again.

Civitai answers a hash: the SHA256 of the whole file, or the AutoV1 hash A1111 used to show,
which takes sixty-four kilobytes to work out. An AutoV1 is not proof — two merges that share
their first layers share it too — but when Civitai does not know it, it does not have the
file.

HuggingFace answers no hash at all: nothing on the Hub looks a file up by its contents. What
it answers is a name, twice over — which repositories are called something like it, and
which model cards mention it, since a card often links the file's real home — and, for any
repository, the exact size and SHA256 of every file in it. So on the Hub a file is found by
its name, narrowed by its size, and proven by its hash.

Nothing here runs unless a person pressed something.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from .files import without_variant

HUB = "https://huggingface.co"
# Repositories whose files are compared, per file looked for. The right one is almost always
# among the first few: a card that links the file, or the repository named like it.
MAX_REPOS = 8
# How long an answer from the Hub is trusted. Identifying a whole library asks after the
# same repositories again and again — Comfy-Org's repackaged files are half of most — and
# the Hub limits how often it may be asked.
CACHE_FOR = 3600.0

_LINK = re.compile(r"huggingface\.co/([\w.\-]+/[\w.\-]+)/(?:blob|resolve)/")


class SlowDown(RuntimeError):
    """A service asked for fewer requests. Worth saying as it is: it passes on its own."""


@dataclass(slots=True)
class HubFile:
    """A file on the Hub that could be the one on disk."""

    repo_id: str
    path: str
    size: int
    sha256: str | None
    commit: str | None
    downloads: int
    same_name: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def page(self) -> str:
        return f"{HUB}/{self.repo_id}/blob/main/{quote(self.path, safe='/')}"


# --- Civitai -------------------------------------------------------------------------


async def civitai_by_hash(
    client: httpx.AsyncClient, digest: str, host: str = "civitai.com", token: str | None = None
) -> dict[str, Any] | None:
    """The model version holding a file with this hash — SHA256 or AutoV1 — or None."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = await client.get(f"https://{host}/api/v1/model-versions/by-hash/{digest}", headers=headers)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Civitai could not be reached: {exc}") from None
    if response.status_code == 404:
        return None
    if response.status_code == 429:
        raise SlowDown("Civitai asked for fewer requests — try again in a few minutes")
    if response.status_code != 200:
        raise RuntimeError(f"Civitai answered {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Civitai sent something that is not JSON") from None
    return data if isinstance(data, dict) else None


def civitai_file(
    version: dict[str, Any],
    sha256: str | None = None,
    autov1: str | None = None,
    size: int | None = None,
) -> dict[str, Any] | None:
    """The file of a version that is this one: by its SHA256, or by its AutoV1 and its size
    to within the kilobyte the API rounds to."""
    for entry in version.get("files") or []:
        hashes = {str(k).lower(): str(v).lower() for k, v in (entry.get("hashes") or {}).items()}
        if sha256 and hashes.get("sha256") == sha256.lower():
            return entry
        if autov1 and hashes.get("autov1") == autov1.lower():
            size_kb = entry.get("sizeKB")
            if not size or not size_kb or abs(float(size_kb) * 1024 - size) <= 1024:
                return entry
    return None


# --- the Hub -------------------------------------------------------------------------


async def find_on_hub(
    client: httpx.AsyncClient,
    filename: str,
    size: int | None,
    sha256: str | None = None,
    token: str | None = None,
    cache: dict[str, tuple[float, Any]] | None = None,
) -> list[HubFile]:
    """Files on the Hub of this size, found by this file's name, best first.

    Given a hash, only a file with that hash is an answer — anything else of the same size
    is another file. Without one, every file of the same size in a repository the name led
    to is a candidate, for the hash to settle; the one of the same name first, then the most
    downloaded, which is the original more often than any of its mirrors.
    """
    if not filename or not size:
        return []
    found: list[HubFile] = []
    for repo in (await hub_repos(client, filename, token, cache))[:MAX_REPOS]:
        info = await hub_repo(client, repo, token, cache)
        if not info:
            continue
        for sibling in info.get("siblings") or []:
            lfs = sibling.get("lfs") or {}
            if (lfs.get("size") or sibling.get("size")) != size:
                continue
            digest = str(lfs.get("sha256") or "").lower() or None
            if sha256 and digest != sha256.lower():
                continue
            path = str(sibling.get("rfilename") or "")
            found.append(HubFile(
                repo_id=repo, path=path, size=size, sha256=digest,
                commit=info.get("sha"), downloads=int(info.get("downloads") or 0),
                same_name=path.rsplit("/", 1)[-1].lower() == filename.lower(), info=info,
            ))
    found.sort(key=lambda f: (not f.same_name, -f.downloads, f.repo_id))
    return found


async def hub_repos(
    client: httpx.AsyncClient,
    filename: str,
    token: str | None = None,
    cache: dict[str, tuple[float, Any]] | None = None,
) -> list[str]:
    """Repositories a file of this name is likely to be in: those whose model cards link it
    — the link names its real home — then those whose cards mention it, then those named
    like it, the most downloaded first."""
    repos: list[str] = []

    def add(repo: Any) -> None:
        if isinstance(repo, str) and "/" in repo and repo not in repos:
            repos.append(repo)

    cards = await _get(client, f"{HUB}/api/search/full-text",
                       {"q": filename, "type": "model", "limit": 10}, token, cache)
    hits = (cards or {}).get("hits") if isinstance(cards, dict) else None
    for hit in hits or []:
        text = "".join(p.get("text", "") for p in ((hit.get("formatted") or {}).get("fileContent") or []))
        for link in _LINK.finditer(text):
            add(link.group(1))
    for hit in hits or []:
        add(hit.get("name"))
    stem = filename.rsplit(".", 1)[0]
    # `hunyuan-video-t2v-720p-Q4_K_M` is looked for as `hunyuan-video-t2v-720p` as well:
    # repositories are named after the model, not after one of its sizes.
    for query in dict.fromkeys((stem, without_variant(stem))):
        if len(query) < 3:
            continue
        named = await _get(client, f"{HUB}/api/models",
                           {"search": query, "limit": 8, "sort": "downloads", "direction": -1}, token, cache)
        for model in named if isinstance(named, list) else []:
            add(model.get("id"))
    return repos


async def hub_repo(
    client: httpx.AsyncClient,
    repo_id: str,
    token: str | None = None,
    cache: dict[str, tuple[float, Any]] | None = None,
) -> dict[str, Any] | None:
    """A repository's description, with the size and SHA256 of every file in it."""
    data = await _get(client, f"{HUB}/api/models/{repo_id}", {"blobs": "true"}, token, cache)
    return data if isinstance(data, dict) else None


def hub_meta(hit: HubFile) -> dict[str, Any]:
    """What a download of this file from the Hub would have known about it."""
    tags = [str(t) for t in hit.info.get("tags") or []]
    # `base_model:org/name`, not `base_model:finetune:org/name` — the plain one names it.
    base = next((t.split(":", 1)[1] for t in tags if t.startswith("base_model:") and t.count(":") == 1), None)
    licence = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    return {
        "source": "huggingface",
        "repo_id": hit.repo_id,
        "path": hit.path,
        "revision": "main",
        "commit": hit.commit,
        "model_name": hit.repo_id.rsplit("/", 1)[-1],
        "base_model": base.rsplit("/", 1)[-1] if base else None,
        "license": licence,
        "pipeline_tag": hit.info.get("pipeline_tag"),
        "library_name": hit.info.get("library_name"),
        "tags": tags[:40],
    }


async def _get(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    token: str | None,
    cache: dict[str, tuple[float, Any]] | None,
) -> Any:
    key = f"{url}?{urlencode(sorted(params.items()))}"
    if cache is not None:
        held = cache.get(key)
        if held is not None and time.time() - held[0] < CACHE_FOR:
            return held[1]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = await client.get(url, params=params, headers=headers)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"HuggingFace could not be reached: {exc}") from None
    if response.status_code == 429:
        raise SlowDown("HuggingFace asked for fewer requests — try again in a few minutes")
    if response.status_code in (401, 403, 404):
        value = None
    elif response.status_code >= 400:
        raise RuntimeError(f"HuggingFace answered {response.status_code}")
    else:
        try:
            value = response.json()
        except ValueError:
            value = None
    if cache is not None:
        cache[key] = (time.time(), value)
    return value
