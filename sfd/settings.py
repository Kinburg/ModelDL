"""Persistent configuration.

Lives in `settings.json` next to the project, which is gitignored — it holds tokens. Nothing
here is required: with no configuration at all the downloader still works, dropping files
into `downloads/` unsorted.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .core.types import DiskKind

DEFAULT_PATH = Path("settings.json")


@dataclass(slots=True)
class Settings:
    # Where files go. Empty means "downloads/", unsorted.
    library_root: str = ""
    # More folders to show in the library, beside the one downloads are filed into. ComfyUI
    # reads models from several folders and drives at once, so a library that only knew the
    # one root would be a partial answer to "what do I have and where is it".
    extra_roots: list[str] = field(default_factory=list)
    # Folders inside the roots that are not part of the library: a llama.cpp checkout with
    # its vocabulary GGUFs, a custom node's own test data. Hidden, not deleted.
    exclude_dirs: list[str] = field(default_factory=list)
    profile: str = "comfyui"
    group_by_base_model: bool = True
    # Whether a new download is filed on its own, by what it turns out to be, with a question
    # only when that is uncertain. Off, every download asks where it goes before it starts —
    # the folder the library already uses for its kind first, so the answer is one keypress.
    smart_placement: bool = False
    download_dir: str = "downloads"

    # Credentials. Environment variables win, so a shared machine need not store them. With
    # neither, HuggingFace falls back to the token `hf auth login` saved.
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
    # every file: the thing worth protecting is the link, not each download.
    max_speed_kb: float = 0.0
    verify_hash: bool = True
    verify_existing: bool = True
    # "" means detect; set to ssd/hdd when Windows reports Unspecified and you know better.
    disk_kind: str = ""

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

    # How the window was left: pane widths, which folders were open, the sort order. Kept
    # here rather than in the page's own storage because the desktop window runs WebView2
    # in private mode, which forgets `localStorage` every time the app is closed.
    ui: dict[str, Any] = field(default_factory=dict)

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
        # The login last: a token set for this program is never overruled by one saved for
        # every tool on the machine.
        return os.environ.get("HF_TOKEN") or self.hf_token or hf_login()[0]

    @property
    def effective_civitai_token(self) -> str | None:
        return os.environ.get("CIVITAI_TOKEN") or self.civitai_token or None

    @property
    def effective_disk_kind(self) -> DiskKind | None:
        try:
            return DiskKind(self.disk_kind) if self.disk_kind else None
        except ValueError:
            return None

    # --- library folders --------------------------------------------------

    @property
    def roots(self) -> list[Path]:
        """Every folder the library is read from, the one downloads land in first.

        Without a library root that first folder is the plain downloads directory, since
        that is where files are going. Duplicates are dropped, whatever their spelling: a
        folder listed twice would show every model in it twice.
        """
        listed = [self.library_root or self.download_dir, *self.extra_roots]
        found: list[Path] = []
        seen: set[str] = set()
        for entry in listed:
            if not entry or not str(entry).strip():
                continue
            # Absolute, so that `downloads` means the same folder to every piece of code
            # that compares a model's path against it.
            path = Path(os.path.abspath(str(entry).strip()))
            key = os.path.normcase(str(path))
            if key not in seen:
                seen.add(key)
                found.append(path)
        return found

    @property
    def sidecar_path(self) -> Path | None:
        return Path(self.sidecar_dir) if self.sidecar_dir else None

    @property
    def library_path(self) -> Path | None:
        return Path(self.library_root) if self.library_root else None

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
        settings = cls(_path=str(path), **{k: v for k, v in data.items() if k in known})
        # A hand-edited file with a string where a list belongs would otherwise be iterated
        # character by character, and show a library of one-letter folders.
        for name in ("extra_roots", "exclude_dirs"):
            value = getattr(settings, name)
            if not isinstance(value, list):
                setattr(settings, name, [value] if isinstance(value, str) and value else [])
        if not isinstance(settings.ui, dict):
            settings.ui = {}
        return settings

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
        login, expired = hf_login()
        data["hf_token_from_login"] = bool(login)
        data["hf_token_login_expired"] = expired
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


# --- the token `hf auth login` saved -------------------------------------------


def hf_login_path() -> Path:
    """Where `hf auth login` keeps the active token, found the way huggingface_hub finds it.

    $HF_TOKEN_PATH names the file outright. Otherwise it is `token` in $HF_HOME, and that
    defaults to `huggingface` under $XDG_CACHE_HOME, or under ~/.cache.
    """
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return Path(os.path.expandvars(os.path.expanduser(explicit)))
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "huggingface",
    )
    return Path(os.path.expandvars(os.path.expanduser(home))) / "token"


def hf_login() -> tuple[str | None, bool]:
    """The token `hf auth login` left behind, and whether it has run out.

    Read on every call, so a login made while the program runs counts straight away.

    A token pasted in at the login prompt lasts until it is revoked. One from the browser
    login expires, and the `hf` tool renews it whenever it runs; nothing here does, so a
    token past the `expires_at` recorded beside it in `stored_tokens` is not used. It
    cannot open anything a token is needed for, and sent anyway it would only make the
    error misleading — a gated model the account did accept would read as terms not
    accepted. Left out, the Settings page can say that the login is what ran out.
    """
    path = hf_login_path()
    try:
        token = path.read_text("utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        return None, False
    if not token:
        return None, False

    stored = configparser.ConfigParser(interpolation=None)
    try:
        stored.read(path.with_name("stored_tokens"), encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError):
        return token, False
    for name in stored.sections():
        if stored.get(name, "hf_token", fallback=None) != token:
            continue
        try:
            expires_at = float(stored.get(name, "expires_at"))
        except (configparser.Error, ValueError):
            return token, False     # a pasted token: nothing recorded, nothing to run out
        return (None, True) if expires_at <= time.time() else (token, False)
    return token, False
