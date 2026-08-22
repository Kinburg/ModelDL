"""Persistent configuration.

Lives in `settings.json` next to the project, which is gitignored — it holds tokens. Nothing
here is required: with no configuration at all the downloader still works, dropping files
into `downloads/` unsorted.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .core.types import DiskKind

DEFAULT_PATH = Path("settings.json")


@dataclass(slots=True)
class Settings:
    # Where files go. Empty means "downloads/", unsorted.
    library_root: str = ""
    profile: str = "comfyui"
    group_by_base_model: bool = True
    download_dir: str = "downloads"

    # Credentials. Environment variables win, so a shared machine need not store them.
    hf_token: str = ""
    civitai_token: str = ""

    # Transfer tuning.
    connections: int = 16
    concurrent_downloads: int = 2
    # Whether a newly added link starts downloading straight away. Turned off, links pile up
    # paused so a session of collecting them costs no bandwidth until you say go.
    auto_start: bool = True
    # Where newly added downloads are queued: "bottom" (default) or "top"
    queue_position: str = "bottom"
    # Pick a failed download back up on its own, a few times, with growing gaps. The point is
    # the queue left running overnight: a router rebooting at 3am should cost minutes, not
    # the whole night. Failures that a wait cannot fix are never retried — see manager.
    auto_retry: bool = True
    min_speed_kb: float = 64.0
    # Ceiling on the whole queue in KB/s, 0 for none. Shared across every connection of
    # every file: the thing worth protecting is the link, not each download. Applies to the
    # native transfer; the huggingface_hub engine downloads in a subprocess of its own.
    max_speed_kb: float = 0.0
    verify_hash: bool = True
    verify_existing: bool = True
    # "" means detect; set to ssd/hdd when Windows reports Unspecified and you know better.
    disk_kind: str = ""

    # "native" is our own transfer: resume we control, a hash we verify, byte-level
    # progress. "hf_hub" runs HuggingFace's client in a subprocess instead — worth choosing
    # for Xet chunk deduplication when re-fetching an updated repo.
    hf_engine: str = "native"
    # Try the other engine once when the native path fails outright. Costs one extra attempt
    # and occasionally succeeds where we cannot, since the official client speaks protocols
    # we do not.
    hf_fallback: bool = True
    hf_disable_xet: bool = False
    hf_xet_high_performance: bool = False
    # Xet writes byte ranges in parallel by direct addressing. On a mechanical disk that is
    # a seek storm; HuggingFace exposes this switch for exactly that case.
    hf_xet_sequential_writes: bool = False

    write_sidecars: bool = True
    # The sample images a model is published with: shown in the queue, and stored beside the
    # model when the compatibility files are on. Turned off, no picture is ever requested.
    fetch_previews: bool = True
    # Where fetched previews are kept. A cache, not a library: it holds thumbnails nobody
    # asked to keep, and deleting it costs one round trip per picture.
    preview_dir: str = "previews"
    # Civitai marks its own samples, and a model's pictures appearing unasked in a queue on
    # a shared screen is its own kind of problem. Covered until clicked.
    blur_nsfw: bool = True
    # Where our own `<name>.json` records go. Empty keeps them beside the model; a path
    # collects them in one place, mirroring the library's folder structure so names cannot
    # collide. The compatibility files stay put regardless — see write_compat_files.
    sidecar_dir: str = ""
    # `.civitai.info` and `.preview.png` are read by the A1111 and ComfyUI model managers,
    # which look for them next to the model and nowhere else. Moving them would break those
    # tools, so they are only ever written in place or not at all.
    write_compat_files: bool = True
    # `<model>.txt` beside a LoRA, holding its activation words. Read automatically by A1111
    # extensions and ComfyUI loader nodes, which paste the contents into the prompt — so it
    # carries the trigger words alone; the link and everything else stay in the JSON record.
    write_trigger_txt: bool = True

    # Layout overrides keyed by category value, for correcting an adopted tree.
    layout_overrides: dict[str, str] = field(default_factory=dict)

    # Where this instance was loaded from, so `save()` writes back to the same place. Kept
    # out of the serialised payload, and out of `apply()`, so a request to the settings
    # endpoint can never redirect where the file lands.
    _path: str = ""
    # Set when an existing settings file could not be read, so the UI and the console can
    # say so instead of running on silent defaults.
    _error: str = ""

    @property
    def error(self) -> str:
        return self._error

    # --- credentials ------------------------------------------------------

    @property
    def effective_hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or self.hf_token or None

    @property
    def effective_civitai_token(self) -> str | None:
        return os.environ.get("CIVITAI_TOKEN") or self.civitai_token or None

    @property
    def hf_hub_options(self) -> dict[str, Any]:
        """Xet tuning, resolved against the target disk.

        A detected mechanical disk turns on sequential writes on its own — the setting is
        for overriding that, not for having to know about it.
        """
        return {
            "disable_xet": self.hf_disable_xet,
            "high_performance": self.hf_xet_high_performance,
            "sequential_writes": (
                self.hf_xet_sequential_writes or self.effective_disk_kind is DiskKind.HDD
            ),
            "max_workers": max(1, self.connections // 2),
        }

    @property
    def effective_disk_kind(self) -> DiskKind | None:
        try:
            return DiskKind(self.disk_kind) if self.disk_kind else None
        except ValueError:
            return None

    # --- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> Settings:
        """Read the settings file, or start from defaults if there is not one yet.

        A file that exists but cannot be read is a different situation entirely, and is not
        allowed to pass quietly: silently falling back to defaults means the library root
        disappears and downloads start landing somewhere else with nothing said about it.
        The broken file is set aside rather than left in place, because the next save would
        overwrite whatever was in it.

        Read as utf-8-sig: PowerShell's `-Encoding utf8` writes a BOM, and plain utf-8
        decoding chokes on the very first byte of an otherwise perfect file.
        """
        path = Path(path)
        if not path.exists():
            return cls(_path=str(path))

        try:
            data = json.loads(path.read_text("utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("settings must be a JSON object")
        except (OSError, ValueError) as exc:
            broken = path.with_name(path.name + ".broken")
            with contextlib.suppress(OSError):
                os.replace(path, broken)
            settings = cls(_path=str(path))
            settings._error = (
                f"{path} could not be read ({exc}); it has been moved to {broken.name} "
                f"and defaults are in use — your library path is not configured"
            )
            return settings

        known = {f.name for f in fields(cls)} - {"_path", "_error"}
        return cls(_path=str(path), **{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path or self._path or DEFAULT_PATH)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(self._payload(), indent=2, ensure_ascii=False), "utf-8")
        os.replace(tmp, target)

    def _payload(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def redacted(self) -> dict[str, Any]:
        """For sending to the browser: say whether a token is set, never what it is."""
        data = self._payload()
        for key in ("hf_token", "civitai_token"):
            data[key] = ""
            data[f"{key}_set"] = bool(getattr(self, key))
        data["hf_token_from_env"] = bool(os.environ.get("HF_TOKEN"))
        data["civitai_token_from_env"] = bool(os.environ.get("CIVITAI_TOKEN"))
        return data

    def apply(self, patch: dict[str, Any]) -> None:
        """Merge a partial update, ignoring unknown keys and blank token overwrites."""
        known = {f.name for f in fields(self)} - {"_path", "_error"}
        for key, value in patch.items():
            if key not in known:
                continue
            # An empty token field in the form means "leave it alone", not "erase it".
            if key in ("hf_token", "civitai_token") and value == "":
                continue
            setattr(self, key, value)
