"""The folders a person can actually choose from, in the order worth showing them.

When classification is not confident enough to file a model on its own, the question put to
the user is "where does this go?". Answering it with the sixteen canonical categories is
answering a different question: the categories are internal names, and `categories.py`
deliberately refuses to claim the folders that custom nodes bring with them — `sams`,
`insightface`, `reactor` and the rest — precisely because a bare `.pt` gives us nothing to
tell them apart with. So the one folder the file actually belongs in was, until now, not on
the list of possible answers at all.

This module answers with the tree that is really there. Ranking matters more than
completeness: a library has dozens of folders, and the right one is almost always the
layout's own guess or a near neighbour of it, so those come first and the reason is carried
along with each row rather than left as an unexplained order.

Nothing here creates a directory. A chosen folder that does not exist yet is created by the
transfer when the file lands, which keeps a failed download from leaving an empty folder
behind — and keeps `mkdir` off an HTTP API that has no authentication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .categories import ALIASES, Category
from .classify import Verdict
from .layout import Layout

# Two levels: the top of the library, and the base-model folders inside it. Deeper trees
# exist, but by the third level the names stop being about what the file is.
DEPTH = 2
# Model-shaped files, counted to tell a folder in use from one that happens to exist.
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"}
LIMIT = 300


@dataclass(slots=True)
class Folder:
    """One row of the picker."""

    relative: str
    # The category this folder *is*, by name — not the category of whatever is inside it. A
    # base-model subfolder like `checkpoints/Krea 2` has none, which is also what stops it
    # from being remembered as the home of every future checkpoint.
    category: Category | None
    models: int
    reason: str
    exists: bool
    rank: int

    def to_json(self) -> dict[str, Any]:
        return {
            "relative": self.relative,
            "category": self.category.value if self.category else None,
            "models": self.models,
            "reason": self.reason,
            "exists": self.exists,
        }


def offer(
    layout: Layout,
    category: Category | None = None,
    base_model: str | None = None,
    filename: str | None = None,
    limit: int = LIMIT,
) -> list[Folder]:
    """Rank the folders under the library root as answers to "where does this go?".

    `category` and `base_model` are the classifier's guess — uncertain, but not worthless:
    the folder it would have used and that folder's neighbours are where the answer usually
    is, so they go to the top. The filename gets a say too, and a loud one for exactly the
    files this dialog exists for: `mystery_sam_model.safetensors` has no category we trust,
    but the library's own `sams` folder is named after a word in it.
    """
    counts = _walk(layout.root)

    proposed: Path | None = None
    if category is not None:
        proposed = layout.directory_for(Verdict(category, "low", "", base_model=base_model))

    mapped = layout.paths.get(category) if category is not None else None
    runners = set(layout.ambiguities.get(category, [])) if category is not None else set()
    group = _relative(layout.root, proposed) if proposed is not None else None

    offered: dict[str, Folder] = {}

    def remember(relative: str, rank: int, reason: str, exists: bool | None = None) -> None:
        if not relative or relative in offered:
            return
        own = ALIASES.get(Path(relative).name.lower())
        offered[relative] = Folder(
            relative=relative,
            category=own,
            models=counts.get(relative, 0),
            reason=reason,
            exists=(layout.root / relative).is_dir() if exists is None else exists,
            rank=rank,
        )

    # 0. What the layout would have done, spelled out — including the base-model grouping,
    #    so the row says the whole directory rather than implying half of it.
    if group and category is not None:
        grouped = group != _relative(layout.root, mapped) if mapped else False
        remember(
            group, 0,
            f"where {category.value} goes"
            + (f", grouped by {base_model}" if grouped and base_model else ""),
        )
    # 1. Folders that hold the same kind and lost the tie in adoption, so the tie can be
    #    broken by hand here instead.
    for name in sorted(runners):
        remember(name, 1, f"also holds {category.value if category else 'models'}")
    # 2. Everything else on disk. A folder the filename itself points at comes first: it is
    #    the signal we have when the category is the thing we are unsure about.
    words = _words(filename)
    for relative in sorted(counts, key=lambda r: (-counts[r], r.lower())):
        name = Path(relative).name
        own = ALIASES.get(name.lower())
        if _named_in(name, words):
            remember(relative, 2, "named in the file name")
        elif own is not None:
            remember(relative, 3 if own is category else 4, f"holds {own.value}")
        elif base_model and name.lower() == base_model.lower():
            remember(relative, 4, f"named after {base_model}")
        else:
            remember(relative, 5, "already in the library")
    # 3. Canonical homes that have no folder yet. Last, but present: the file may well be
    #    the first LoRA in a library that has never had one.
    for kind, path in sorted(layout.paths.items(), key=lambda item: item[0].value):
        relative = _relative(layout.root, path)
        if relative and not (layout.root / relative).is_dir():
            remember(relative, 6, f"the usual home for {kind.value} — not created yet", False)

    ranked = sorted(
        offered.values(), key=lambda f: (f.rank, -f.models, f.relative.lower())
    )
    return ranked[:limit]


def resolve_inside(root: Path, relative: str) -> Path | None:
    """A user-supplied folder, resolved under `root`, or None if it escapes.

    The picker lets a path be typed rather than chosen, which is the whole point of it — a
    folder that does not exist yet cannot be in a list of folders that do. That also makes
    this the one place where a string from the browser turns into a filesystem path, so it
    is refused unless it lands inside the library.
    """
    if not str(relative).strip() or not str(root):
        return None
    candidate = Path(str(relative).strip().replace("\\", "/"))
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        return None
    try:
        resolved = (root / candidate).resolve()
        base = root.resolve()
    except OSError:
        return None
    if resolved == base or base not in resolved.parents:
        return None
    return resolved


def _walk(root: Path) -> dict[str, int]:
    """Folders under `root` down to `DEPTH`, each with the model files at or below it.

    Counted here rather than with `layout._file_count`, which walks a whole subtree per
    folder: this runs on a click, over a library that can hold tens of thousands of files.
    The walk stops one level past what it offers — enough to tell a folder in daily use from
    one that merely exists, without reading the entire tree to decide a sort order.
    """
    counts: dict[str, int] = {}
    if not root.is_dir():
        return counts

    def scan(directory: Path, relative: str | None, depth: int) -> int:
        if depth > DEPTH + 1:
            return 0
        total = 0
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return 0
        for entry in entries:
            try:
                if entry.is_dir():
                    if entry.name.startswith(".") or entry.name == "__pycache__":
                        continue
                    offered = None
                    if depth <= DEPTH:
                        offered = f"{relative}/{entry.name}" if relative else entry.name
                    total += scan(entry, offered, depth + 1)
                elif entry.suffix.lower() in MODEL_SUFFIXES:
                    total += 1
            except OSError:
                continue
        if relative:
            counts[relative] = total
        return total

    scan(root, None, 1)
    return counts


def _words(filename: str | None) -> set[str]:
    """The words of a filename, long enough to mean something."""
    if not filename:
        return set()
    stem = Path(filename).stem.lower()
    return {word for word in re.split(r"[\W_]+", stem) if len(word) > 2}


def _named_in(folder: str, words: set[str]) -> bool:
    """Whether a folder's whole name appears in those words, give or take a plural.

    Whole name only. Matching parts of it would put a style LoRA in `style_models` on the
    strength of one word, which is exactly the confident misfiling this dialog exists to
    let a person correct.
    """
    name = folder.lower()
    return bool(words) and (
        name in words or name.rstrip("s") in {word.rstrip("s") for word in words}
    )


def _relative(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return relative.as_posix() or None
