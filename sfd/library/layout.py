"""Where a classified model goes on disk.

The layout is adopted from the tree that is already there rather than imposed. Someone with
a working ComfyUI install has forty folders and firm habits about them; a downloader that
invents its own structure alongside is worse than useless.

Adoption has to resolve synonyms, because ComfyUI accepts several names for the same thing
and installs accumulate both. When `unet` and `diffusion_models` both exist, or `clip` and
`text_encoders`, the tie is broken by which one already holds files — what the user actually
does beats what any naming convention says they should do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .categories import ALIASES, PROFILES, Category
from .classify import Verdict

# Kinds where splitting by base model earns its keep. A Flux LoRA among a thousand SD1.5
# ones is unusable; there is no comparable problem with a handful of VAEs.
SUBFOLDER_BY_BASE_MODEL = {
    Category.LORA,
    Category.EMBEDDING,
    Category.CHECKPOINT,
    Category.DIFFUSION_MODEL,
    Category.CONTROLNET,
}


@dataclass(slots=True)
class Layout:
    root: Path
    paths: dict[Category, Path]
    # Categories where several existing folders could have served, with the runners-up.
    # Surfaced in settings so the choice can be corrected rather than silently lived with.
    ambiguities: dict[Category, list[str]] = field(default_factory=dict)
    group_by_base_model: bool = True

    def directory_for(self, verdict: Verdict) -> Path:
        """The folder this verdict files into, base-model grouping included."""
        directory = self.paths.get(verdict.category, self.root / "other")
        if (
            self.group_by_base_model
            and verdict.base_model
            and verdict.category in SUBFOLDER_BY_BASE_MODEL
        ):
            group = _folder_safe(verdict.base_model)
            if group:
                directory = directory / group
        return directory

    def destination(self, verdict: Verdict, filename: str) -> Path:
        return self.directory_for(verdict) / filename

    # --- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "group_by_base_model": self.group_by_base_model,
            "paths": {c.value: str(p) for c, p in sorted(self.paths.items())},
            "ambiguities": {c.value: names for c, names in sorted(self.ambiguities.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layout:
        return cls(
            root=Path(data["root"]),
            paths={
                Category(key): Path(value)
                for key, value in (data.get("paths") or {}).items()
                if key in Category._value2member_map_
            },
            ambiguities={
                Category(key): list(value)
                for key, value in (data.get("ambiguities") or {}).items()
                if key in Category._value2member_map_
            },
            group_by_base_model=bool(data.get("group_by_base_model", True)),
        )


def adopt(root: Path, profile: str = "comfyui") -> Layout:
    """Build a layout from the folders that already exist under `root`.

    Folders we do not recognise are left alone — an install carries plenty of directories
    belonging to custom nodes, and guessing at them would only misfile things.
    """
    defaults = PROFILES.get(profile, PROFILES["comfyui"])

    candidates: dict[Category, list[Path]] = {}
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            category = ALIASES.get(entry.name.lower())
            if category is not None:
                candidates.setdefault(category, []).append(entry)

    paths: dict[Category, Path] = {}
    ambiguities: dict[Category, list[str]] = {}
    for category, found in candidates.items():
        preferred = defaults.get(category, "")
        ranked = sorted(
            found,
            key=lambda p: (
                -_file_count(p),                       # what is actually in use wins
                0 if p.name.lower() == preferred.lower() else 1,
                p.name.lower(),
            ),
        )
        paths[category] = ranked[0]
        if len(ranked) > 1:
            ambiguities[category] = [p.name for p in ranked[1:]]

    # Anything with no existing home gets the profile's default name, created on demand.
    for category in Category:
        paths.setdefault(category, root / defaults.get(category, category.value))

    return Layout(root=root, paths=paths, ambiguities=ambiguities)


def flat(root: Path) -> Layout:
    """Everything in one directory — the default until a real library is configured."""
    return Layout(
        root=root,
        paths={category: root for category in Category},
        group_by_base_model=False,
    )


def _file_count(path: Path) -> int:
    """How many model-shaped files live under a folder. Cheap and shallow on purpose."""
    try:
        return sum(
            1
            for entry in path.rglob("*")
            if entry.is_file()
            and entry.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"}
        )
    except OSError:
        return 0


def _folder_safe(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    return cleaned[:64]
