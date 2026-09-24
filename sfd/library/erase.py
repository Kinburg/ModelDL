"""Deleting a model, and everything named after it.

A model is not one file. Four more are written beside it — the record, the Civitai payload,
the trigger words, the preview — and an interrupted download leaves two or three more again:
the `.part` holding whatever arrived, the `.part.json` that makes it resumable, and the
`.part.corrupt` a failed checksum renames it to. Deleting the model in Explorer leaves every
one of those behind, named after a file that no longer exists, in a folder nobody will think
to sweep. A 40 GB `.part` from a download abandoned in March is the usual way a drive fills
up.

So the set goes together, and the order is the part worth stating:

  The model is deleted first. It is the file the whole set exists to describe, and on
  Windows it is the one that can refuse: a loader with the weights mapped into memory holds
  the handle, and the unlink fails. If it does, nothing else is touched and the caller hears
  about it — a model left on disk with its record, its preview and its triggers already
  deleted is a worse outcome than one that would not delete at all.

  Everything after it is reported rather than raised. Once the model is gone the set is gone
  whatever happens next; a sidecar that would not go is a stray file to name, not a reason
  to stop halfway through the rest.

Nothing here goes to a recycle bin. The delete is permanent, which is why nothing calls this
without asking first, and why `belongings` exists to be shown before that question.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .relocate import companions

# What an unfinished download leaves lying next to where the file was going to be. Named
# from the destination rather than found by globbing: a glob for `*.part` in a folder full
# of models would sweep up another download's file.
LEFTOVERS = (".part", ".part.json", ".part.corrupt")


@dataclass(slots=True)
class Erased:
    """What actually went, which is not always everything that was asked for."""

    deleted: list[Path] = field(default_factory=list)
    # Named, not counted: a file that stayed has to be findable by whoever reads this.
    failed: list[tuple[Path, str]] = field(default_factory=list)
    # The model was already gone — a file deleted in Explorer, or a download that never
    # landed. Not a failure: the sidecars and the `.part` are still worth clearing.
    missing: bool = False


def leftovers(path: Path) -> list[Path]:
    """The fragments of a download to `path` that never finished."""
    return [p for p in (path.with_name(path.name + s) for s in LEFTOVERS) if p.is_file()]


def belongings(
    path: Path,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    roots: Iterable[Path] = (),
) -> list[Path]:
    """Every file on disk that a delete would take, model first.

    Meant to be shown before anything is deleted. The list is the honest answer to "what
    exactly is about to go", which for a permanent delete of a set of files is a question
    nobody should have to answer from memory.
    """
    found = [path] if path.is_file() else []
    return found + companions(path, sidecar_dir, library_root, roots) + leftovers(path)


def erase(
    path: Path,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    roots: Iterable[Path] = (),
) -> Erased:
    """Delete `path` and everything named after it. Permanently.

    Raises `OSError` if the model itself will not go, having touched nothing else.
    """
    result = Erased()
    rest = companions(path, sidecar_dir, library_root, roots) + leftovers(path)

    if path.is_file():
        # Deliberately not guarded: if this raises, the sidecars are still on disk beside
        # a model that is still on disk, which is exactly the state the caller started in.
        path.unlink()
        result.deleted.append(path)
    else:
        result.missing = True

    for other in rest:
        try:
            other.unlink()
        except OSError as exc:
            result.failed.append((other, str(exc)))
        else:
            result.deleted.append(other)
    return result


def remove(paths: Iterable[Path]) -> Erased:
    """Delete a list of files that has already been shown to someone and agreed to.

    For what is left of a model that is gone — its record, a preview in a folder it was
    dragged out of — and for the fragments of downloads nobody is coming back for. A file
    that has already gone is not a failure: the point was for it not to be there.
    """
    result = Erased()
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            result.failed.append((path, str(exc)))
        else:
            result.deleted.append(path)
    return result
