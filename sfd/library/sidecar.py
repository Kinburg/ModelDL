"""Metadata written next to a model file.

The other half of "I will never find this again". A folder full of correctly sorted
`.safetensors` is still opaque: nothing in the file says where it came from, what it was
trained on, or — for a LoRA — which words activate it. Trigger words in particular are
published by Civitai, discarded by every plain download, and impossible to recover
afterwards short of finding the model page again.

Two files are written beside the model:

  `<model>.json`          our own record: source, hash, category and why, base model,
                          trigger words, when it arrived.
  `<model>.civitai.info`  the raw Civitai version payload, for Civitai downloads only.
                          This is the filename the existing A1111 and ComfyUI model-manager
                          extensions already look for, so they pick the model up with its
                          previews and triggers for free.

Plus `<model>.preview.png` when the service offers a preview image.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .classify import Verdict

PREVIEW_SUFFIX = ".preview.png"
MAX_PREVIEW_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class Record:
    filename: str
    source_url: str | None = None
    page_url: str | None = None
    sha256: str | None = None
    size: int | None = None
    provider: str | None = None
    meta: dict[str, Any] | None = None

    def build(self, verdict: Verdict) -> dict[str, Any]:
        meta = self.meta or {}
        return {
            "schema": 1,
            "filename": self.filename,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": {
                "provider": self.provider,
                "url": self.source_url,
                "page": self.page_url or _page_url(meta),
                "repo_id": meta.get("repo_id"),
                "commit": meta.get("commit"),
                "model_id": meta.get("model_id"),
                "version_id": meta.get("version_id"),
                "model_name": meta.get("model_name"),
                "version_name": meta.get("version_name"),
            },
            "integrity": {"sha256": self.sha256, "size": self.size},
            "classification": {
                "category": verdict.category.value,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
                # Kept deliberately: months later, "the site called this a checkpoint but
                # its contents said otherwise" is the note that explains the folder.
                "service_called_it": (
                    verdict.disagreement.value if verdict.disagreement else None
                ),
            },
            "usage": {
                "base_model": verdict.base_model or meta.get("base_model"),
                # The thing that makes a LoRA usable, and the thing always lost.
                "trigger_words": meta.get("trained_words") or [],
                "precision": meta.get("precision"),
                "nsfw": meta.get("nsfw"),
            },
        }


def write(
    path: Path,
    verdict: Verdict,
    record: Record,
    raw: dict[str, Any] | None = None,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    compat: bool = True,
    triggers: bool = True,
) -> Path:
    """Write the sidecars for a model already at `path`. Returns where the record went."""
    target = record_path(path, sidecar_dir, library_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, record.build(verdict))

    if compat and raw is not None and record.provider == "civitai":
        # Deliberately not moved: the model managers look here and nowhere else.
        _write_json(path.with_name(path.stem + ".civitai.info"), raw)

    if triggers:
        write_trigger_text(path, (record.meta or {}).get("trained_words"))
    return target


def write_trigger_text(path: Path, words: Any) -> Path | None:
    """Write `<model>.txt` holding the LoRA's activation words, and nothing else.

    This filename is a convention with a specific meaning: A1111 extensions and several
    ComfyUI loader nodes read it as *activation text* and paste its contents straight into
    the prompt. So it gets the trigger words and only the trigger words — a source URL or a
    description in here would end up inside someone's generation. Everything else about the
    model lives in the JSON record.

    Civitai often returns the words already comma-joined inside a single string, so they are
    re-split and tidied rather than written through as-is.
    """
    cleaned = normalise_triggers(words)
    if not cleaned:
        return None
    target = path.with_name(path.stem + ".txt")
    _atomic_write(target, (", ".join(cleaned) + "\n").encode("utf-8"))
    return target


def normalise_triggers(words: Any) -> list[str]:
    if isinstance(words, str):
        words = [words]
    if not isinstance(words, (list, tuple)):
        return []

    seen: dict[str, None] = {}
    for entry in words:
        for part in str(entry).split(","):
            word = part.strip()
            if word and word.lower() not in {w.lower() for w in seen}:
                seen[word] = None
    return list(seen)


def record_path(
    path: Path, sidecar_dir: Path | None = None, library_root: Path | None = None
) -> Path:
    """Where a model's `.json` record belongs.

    Beside the model by default. When collected elsewhere, the library's folder structure is
    mirrored underneath — two `model.safetensors` in different categories are different
    files, and flattening them into one directory would make one overwrite the other.
    """
    if sidecar_dir is None:
        return path.with_name(path.name + ".json")

    relative = None
    if library_root is not None:
        with contextlib.suppress(ValueError):
            relative = path.relative_to(library_root)
    if relative is None:
        relative = Path(path.name)
    return sidecar_dir / relative.with_name(relative.name + ".json")


async def fetch_preview(
    path: Path, meta: dict[str, Any], client: httpx.AsyncClient
) -> Path | None:
    """Save the service's preview image beside the model, if there is one."""
    url = meta.get("preview_url")
    if not url:
        return None
    try:
        response = await client.get(str(url), follow_redirects=True)
        if response.status_code != 200:
            return None
        if len(response.content) > MAX_PREVIEW_BYTES:
            return None
        if not (response.headers.get("content-type") or "").startswith("image/"):
            return None
    except httpx.HTTPError:
        return None

    target = path.with_name(path.stem + PREVIEW_SUFFIX)
    _atomic_write(target, response.content)
    return target


def read(
    path: Path, sidecar_dir: Path | None = None, library_root: Path | None = None
) -> dict[str, Any] | None:
    try:
        return json.loads(record_path(path, sidecar_dir, library_root).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --- helpers ----------------------------------------------------------------


def _page_url(meta: dict[str, Any]) -> str | None:
    if meta.get("model_id") and meta.get("version_id"):
        host = meta.get("host") or "civitai.com"
        return (
            f"https://{host}/models/{meta['model_id']}"
            f"?modelVersionId={meta['version_id']}"
        )
    if meta.get("repo_id"):
        return f"https://huggingface.co/{meta['repo_id']}"
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    """Never leave a half-written sidecar next to a good model file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
