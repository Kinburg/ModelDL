"""Walking the library's folders to find out what is actually in them.

The library index is a cache of this walk, and the walk is cheap enough to repeat whenever
the window comes back to the front: a terabyte of models is a few hundred files, and a
directory listing does not read any of them. Everything that does read a file — the header,
the hash — happens elsewhere, once per version of that file.

What is deliberately left out: hidden folders (`.cache` from huggingface_hub, `.git`),
`__pycache__` and `node_modules`, and any folder the person has asked to hide. A model
folder is where people keep other things too — a whole llama.cpp checkout, with a dozen
vocabulary GGUFs that are not models anybody loads.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .categories import ALIASES
from .files import is_model_file, shard_of

SKIPPED_DIRS = frozenset({"__pycache__", "node_modules"})
# Deep enough for `models/loras/Krea 2/styles/v2`, shallow enough that a junction pointing
# back up the tree cannot keep the walk busy forever — the visited set below is the real
# guard against that, this is the backstop.
MAX_DEPTH = 16
# What an interrupted download or move leaves behind. Found here so the cleanup view can
# offer them; never treated as models.
FRAGMENT_SUFFIXES = (".part", ".part.json", ".part.corrupt", ".moving")


@dataclass(slots=True)
class Found:
    """One model on disk. A model split into shards is one of these, not several."""

    path: Path
    size: int
    mtime: float
    parts: list[Path] = field(default_factory=list)
    root: int = 0


@dataclass(slots=True)
class Scan:
    files: dict[str, Found] = field(default_factory=dict)
    # (root index, folder relative to it, whether a model is at or below it). A folder is
    # listed if it holds a model somewhere below it, or if its name says what it is for —
    # `hypernetworks` with nothing in it yet is still where hypernetworks go.
    folders: list[tuple[int, str, int]] = field(default_factory=list)
    fragments: list[Path] = field(default_factory=list)
    missing_roots: list[int] = field(default_factory=list)
    elapsed: float = 0.0


def key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def walk(roots: Iterable[Path], excluded: Iterable[Path | str] = ()) -> Scan:
    started = time.monotonic()
    roots = list(roots)
    scan = Scan()
    hidden = {key(p) for p in excluded if str(p).strip()}
    root_keys = {key(r): index for index, r in enumerate(roots)}
    seen_dirs: set[tuple[int, int]] = set()

    for index, root in enumerate(roots):
        if not root.is_dir():
            scan.missing_roots.append(index)
            continue
        counts: dict[str, int] = {}
        _walk_root(root, index, scan, hidden, root_keys, seen_dirs, counts)
        for relative, models in counts.items():
            if relative:
                scan.folders.append((index, relative, models))
    scan.elapsed = time.monotonic() - started
    return scan


def _walk_root(
    root: Path,
    index: int,
    scan: Scan,
    hidden: set[str],
    root_keys: dict[str, int],
    seen_dirs: set[tuple[int, int]],
    counts: dict[str, int],
) -> None:
    """Depth-first, by hand rather than `os.walk`: the directory entries already carry the
    sizes and times on Windows, and the walk has to refuse to enter a folder twice."""

    def visit(directory: Path, relative: str, depth: int) -> int:
        try:
            identity = os.stat(directory)
            mark = (identity.st_dev, identity.st_ino)
            if identity.st_ino and mark in seen_dirs:
                return 0
            seen_dirs.add(mark)
            entries = list(os.scandir(directory))
        except OSError:
            return 0

        models = 0
        shards: dict[str, list[tuple[int, Path, int, float]]] = {}
        for entry in entries:
            name = entry.name
            try:
                if entry.is_dir():
                    if name.startswith(".") or name in SKIPPED_DIRS or depth >= MAX_DEPTH:
                        continue
                    path = Path(entry.path)
                    folder_key = key(path)
                    if folder_key in hidden:
                        continue
                    # Another folder of the library, nested inside this one: it is walked
                    # as itself, and walking it here too would list its models twice.
                    if root_keys.get(folder_key, index) != index:
                        continue
                    below = f"{relative}/{name}" if relative else name
                    models += visit(path, below, depth + 1)
                    continue
                if not entry.is_file():
                    continue
                lowered = name.lower()
                if lowered.endswith(FRAGMENT_SUFFIXES):
                    scan.fragments.append(Path(entry.path))
                    continue
                stat = entry.stat()
                if not is_model_file(name, stat.st_size):
                    continue
                shard = shard_of(name)
                if shard is not None:
                    shards.setdefault(shard[0].lower(), []).append(
                        (shard[1], Path(entry.path), stat.st_size, stat.st_mtime)
                    )
                    continue
                path = Path(entry.path)
                scan.files.setdefault(key(path), Found(path, stat.st_size, stat.st_mtime, root=index))
                models += 1
            except OSError:
                continue

        for pieces in shards.values():
            pieces.sort()
            first = pieces[0][1]
            scan.files.setdefault(key(first), Found(
                path=first,
                size=sum(p[2] for p in pieces),
                mtime=max(p[3] for p in pieces),
                parts=[p[1] for p in pieces],
                root=index,
            ))
            models += 1

        # A folder named for a kind of model is where that kind goes, whether or not
        # anything has gone there yet. Only at the top: `loras/sdxl/vae` is not the VAEs.
        if relative and (models or ("/" not in relative and relative.lower() in ALIASES)):
            counts[relative] = models
        return models

    visit(root, "", 0)
