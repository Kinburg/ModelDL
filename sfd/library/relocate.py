"""Moving a model that was filed in the wrong place.

Classification is a guess, and a guess that lands a 6 GB checkpoint in `loras` is not worth
correcting by hand in Explorer — not because dragging a file is hard, but because the model
is not the only thing that has to move. Four files are written alongside it, all named after
it, and a model manager that finds `model.safetensors` without `model.preview.png` shows a
blank card. Dragging one file leaves the rest behind in a folder nobody will look in again.

So the model and everything that belongs to it move together, and the awkward parts get the
attention:

  Nothing is overwritten. If the destination already holds a file of that name the move is
  refused outright, before anything has been touched — silently replacing a model with
  another of the same name is the one outcome worse than filing it wrongly in the first
  place.

  The model moves first. It is the file that matters and the only one big enough to fail
  partway, so if it cannot move, nothing else has either. The companions follow one at a
  time; one of those failing leaves the model where it was asked to go and says which ones
  did not make it, because reporting the model as stuck when it moved perfectly well would
  send someone hunting in the wrong folder.

  A cross-volume move is a copy and a delete, so this can take as long as writing the file
  did. It belongs on a thread, not on the event loop, and it reports its progress — a
  6 GB checkpoint crossing from one drive to another is minutes of an application that
  would otherwise look like it had hung.

Renaming is the same problem with the axes swapped — the folder stays and the name changes —
and it is here for the same reason: `pytorch_lora_weights.safetensors`, which is what half of
HuggingFace calls its LoRAs, is unreadable in a folder of two hundred, and renaming it in
Explorer orphans four files at a stroke.
"""

from __future__ import annotations

import errno
import os
import shutil
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .files import is_model_file
from .sidecar import PREVIEW_SUFFIX, find_record, record_path, retitle

# What other tools write beside a model, named after it. Ours first, then the conventions a
# library collects from everything else that has touched it: A1111 keeps a description in
# `<model>.json` and a checkpoint's config in `<model>.yaml`, ComfyUI's LoRA Manager writes
# `<model>.metadata.json`, Stability Matrix `<model>.cm-info.json`, and nearly every model
# manager looks for a picture under the model's own name. A move that left any of these
# behind would be the half-move this module exists to prevent, for somebody else's tool.
COMPANION_SUFFIXES = (
    ".civitai.info",
    ".txt",
    PREVIEW_SUFFIX,
    ".preview.jpg", ".preview.jpeg", ".preview.webp", ".preview.gif",
    ".preview.mp4", ".preview.webm",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm",
    ".json", ".yaml", ".yml",
    ".metadata.json", ".cm-info.json", ".sha256",
)

# Read size for the copy a cross-volume move turns into. Large enough that the progress
# callback fires a few times a second on a slow drive rather than a few thousand.
COPY_CHUNK = 4 * 1024 * 1024

# What a filename may not contain: the separators, which would make a rename into a move,
# and the characters Windows refuses outright.
FORBIDDEN = frozenset(r'\/:*?"<>|')

# Called with (bytes copied, total) while a move is copying. Never called for a move that
# is a rename, because there is nothing to report: it is instant.
Progress = Callable[[int, int], None]


class Cancelled(Exception):
    """The copy was stopped on request, before the original was touched.

    Not an error in the sense the caller needs to worry about: it is the outcome that was
    asked for, and the file is where it always was.
    """


@dataclass(slots=True)
class Move:
    """What actually happened, which is not always what was asked for."""

    path: Path
    companions: list[Path] = field(default_factory=list)
    # Companions that would not move, each with the reason. The model itself is never in
    # here: if it could not move, this object does not exist.
    failed: list[tuple[Path, str]] = field(default_factory=list)
    # True when the file was already in the folder that was asked for.
    unchanged: bool = False


def companions(
    path: Path,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    roots: Iterable[Path] = (),
) -> list[Path]:
    """The files written alongside `path` that mean nothing without it.

    The record is asked for by name rather than guessed at, because it is the one that does
    not have to be next to the model: with `sidecar_dir` set it lives in a mirror of the
    library tree, and the mirror has to be followed to the new location rather than the file
    dumped beside the model it was deliberately kept away from.
    """
    record = find_record(path, sidecar_dir, library_root, roots)
    found = [record] if record is not None else []
    for other in named_companions(path):
        if other not in found:
            found.append(other)
    return found


def named_companions(path: Path) -> list[Path]:
    """The files beside `path` that are named after it and exist — unless its name is
    shared.

    `model.safetensors` beside `model.ckpt`: whose is `model.png`? There is no telling, and
    taking it with one of them strips it from the other. Only the record, which carries the
    model's whole name, is unambiguously its own, and that is `find_record`'s to find.
    """
    if _stem_is_shared(path):
        return []
    return [other for other in named_after(path) if other.is_file()]


def _stem_is_shared(path: Path) -> bool:
    """Whether another model file in the same folder has this one's name before the dot."""
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return False
    stem = path.stem.lower()
    return any(
        other.stem.lower() == stem
        and other.name.lower() != path.name.lower()
        and is_model_file(other.name)
        for other in siblings
    )


def named_after(path: Path) -> list[Path]:
    """Every name a companion of `path` could have beside it, whether or not it exists.

    A model is never its own companion — `<stem>.json` for a model that is itself a
    `.json` would be — and the record is left to `find_record`, which knows where it lives.
    """
    own = path.name.lower()
    record = (path.name + ".json").lower()
    names = []
    for suffix in COMPANION_SUFFIXES:
        candidate = path.stem + suffix
        if candidate.lower() in (own, record):
            continue
        names.append(path.with_name(candidate))
    return names


def move(
    path: Path,
    folder: Path,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    progress: Progress | None = None,
    stop: threading.Event | None = None,
    roots: Iterable[Path] = (),
) -> Move:
    """Move `path` and its companions into `folder`.

    Blocking, and potentially for minutes: call it off the event loop. `stop` is how it is
    interrupted from the loop's side — a thread running a copy cannot be cancelled, only
    asked, so the ask has to be something the copy looks at.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there any more")

    target = folder / path.name
    if _same_path(target, path):
        return Move(path=path, unchanged=True)
    if target.exists():
        raise FileExistsError(f"{folder} already holds a {path.name}")

    # The picker offers folders that do not exist yet on purpose — the one a misfiled model
    # needs is often the one nothing has been filed into. Creating it here is the same thing
    # a download does when it lands. Nothing is vetted at this level: a caller reaches this
    # function only after confining the folder to the library (`resolve_inside`) or after a
    # person named it in the system's own dialog.
    folder.mkdir(parents=True, exist_ok=True)

    # Where each companion has to end up is worked out before anything moves, so that the
    # record's mirrored path is derived from the model's new home rather than from a model
    # that is by then no longer where the calculation assumes.
    roots = tuple(roots)
    record = find_record(path, sidecar_dir, library_root, roots)
    followers = [
        (source, _destination(source, record, target, sidecar_dir, library_root, roots))
        for source in companions(path, sidecar_dir, library_root, roots)
    ]

    # Only the model's own copy is interruptible. Past this line the file is across and
    # stopping would strand it away from its sidecars — and those are small enough that
    # there is nothing to wait for anyway.
    _transfer(path, target, progress, stop)

    result = Move(path=target)
    _follow(followers, result)
    return result


def strays(
    old: Path,
    new: Path,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    roots: Iterable[Path] = (),
) -> list[tuple[Path, Path]]:
    """What a model moved outside the app left behind, and where each piece belongs now.

    Dragged in Explorer, a model goes and its preview, its trigger words and its record
    stay in the folder it left. Each of those is paired with where it would be had the
    model been moved here by this app: the record into its place in the mirror, everything
    else beside the model under the model's current name — which after a rename is not the
    name the stray still carries.
    """
    roots = tuple(roots)
    record = find_record(old, sidecar_dir, library_root, roots)
    pairs: list[tuple[Path, Path]] = []
    if record is not None:
        pairs.append((record, record_path(new, sidecar_dir, library_root, roots)))
    for other in named_companions(old):
        pairs.append((other, new.with_name(new.stem + other.name[len(old.stem):])))
    return [(a, b) for a, b in pairs if not _same_path(a, b)]


def bring(path: Path, pairs: Iterable[tuple[Path, Path]]) -> Move:
    """Move each (stray, destination) pair, never over a file that is already there.

    The other half of `strays`: that one says what would move, this one moves it, once a
    person has said yes.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there any more")
    result = Move(path=path)
    _follow([(a, b) for a, b in pairs if a.is_file()], result)
    return result


def _follow(followers: list[tuple[Path, Path]], result: Move) -> None:
    for source, destination in followers:
        if _same_path(source, destination):
            continue
        if destination.exists():
            # Not fatal. The model is already across, and a stale sidecar at the
            # destination is a smaller problem than a half-moved set of them.
            result.failed.append((source, "a file of that name is already there"))
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _transfer(source, destination)
        except OSError as exc:
            result.failed.append((source, str(exc)))
        else:
            result.companions.append(destination)


def _same_path(a: Path, b: Path) -> bool:
    """The same file, however each was spelt — `Loras` and `loras` are one folder on
    Windows, and moving a file onto itself must not be refused as a collision."""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def intended_name(path: Path, name: str) -> str:
    """The filename a rename really means, or a refusal saying why it cannot be one.

    Two things are settled here rather than left to the filesystem to complain about later.

    A name is a name and not a path. Anything with a separator in it is a move disguised as
    a rename, and this function is reached from a server with no authentication: `../../id_rsa`
    typed into a rename box must fail as a bad name, not succeed as a relocation. The
    characters Windows refuses outright go the same way, because "the system cannot find the
    file specified" is not an answer anyone can act on.

    And the extension is kept, always. `.safetensors` is not decoration — it is what every
    loader dispatches on, and a model renamed to `.ckpt` would be a file claiming to be a
    format it is not. Typing the extension is allowed and typing it wrong is not: whatever
    is given, the model's own suffix is what ends up on the end.
    """
    wanted = name.strip().rstrip(". ")
    if not wanted:
        raise ValueError("a file needs a name")
    if wanted in {".", ".."} or FORBIDDEN.intersection(wanted) or any(
        ord(c) < 32 for c in wanted
    ):
        raise ValueError(f"{name!r} is not a filename")

    suffix = path.suffix
    if suffix and wanted.lower().endswith(suffix.lower()):
        wanted = wanted[: -len(suffix)].rstrip(". ")
        if not wanted:
            raise ValueError("a file needs more of a name than its extension")
    return wanted + suffix


def rename(
    path: Path,
    name: str,
    sidecar_dir: Path | None = None,
    library_root: Path | None = None,
    roots: Iterable[Path] = (),
) -> Move:
    """Give `path` a new name, and every file named after it the matching one.

    The correction for a filename rather than for a folder — `pytorch_lora_weights.safetensors`,
    which is what half of HuggingFace calls its LoRAs, tells you nothing in a folder of two
    hundred. It is the same operation as a move and shares its rules: nothing is overwritten,
    the model goes first, and a companion that will not go is named rather than swallowed.

    What it does not share is the drive boundary. A rename stays in the folder it started
    in, so it is always the instant kind — there is no copy here to report on or to stop.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there any more")

    target = path.with_name(intended_name(path, name))
    if target == path:
        return Move(path=path, unchanged=True)
    if target.exists():
        # Including a difference of case only, on the filesystems that cannot tell them
        # apart. Refusing is the wrong answer for `Model.safetensors` -> `model.safetensors`
        # and the right one for everything else, and the two are indistinguishable from
        # here — so the safe reading wins, as it does for a move.
        raise FileExistsError(f"{path.parent} already holds a {target.name}")

    roots = tuple(roots)
    record = find_record(path, sidecar_dir, library_root, roots)
    followers = [
        (source, _renamed(source, path, record, target, sidecar_dir, library_root, roots))
        for source in companions(path, sidecar_dir, library_root, roots)
    ]

    _transfer(path, target)

    result = Move(path=target)
    _follow(followers, result)

    # The record names the file it describes. Leaving the old name in it would make the one
    # document that explains where a model came from disagree with the model.
    retitle(record_path(target, sidecar_dir, library_root, roots), target.name)
    return result


def _transfer(
    source: Path,
    target: Path,
    progress: Progress | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Move one file, whether or not the destination is on the same drive.

    A rename first, because within a drive that is instant and atomic however large the
    file is. Only when the OS refuses it for the one reason worth handling — the target is
    on another volume, where a rename cannot exist — does this become what it really is: a
    copy of every byte followed by a delete. Any other error is the caller's to hear about
    unchanged, rather than quietly reinterpreted as a slow path.

    The copy lands under a temporary name and is renamed into place at the end, so the
    destination never holds a half-written model that looks finished. If it fails partway —
    or is stopped — the fragment is removed and the original is still there, untouched.
    """
    try:
        os.replace(source, target)
        return
    except OSError as exc:
        cross_volume = exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17
        if not cross_volume:
            raise

    partial = target.with_name(target.name + ".moving")
    try:
        copied = 0
        total = source.stat().st_size
        with open(source, "rb") as reader, open(partial, "wb") as writer:
            while True:
                if stop is not None and stop.is_set():
                    # Between blocks, which is the only place it can be: nothing is half
                    # written, and the bytes so far are in a file about to be deleted.
                    raise Cancelled(f"the move of {source.name} was stopped")
                block = reader.read(COPY_CHUNK)
                if not block:
                    break
                writer.write(block)
                copied += len(block)
                if progress is not None:
                    progress(copied, total)
        shutil.copystat(source, partial)
        os.replace(partial, target)
    except BaseException:
        # Including cancellation: a fragment named after the model is worse than no file,
        # and the original has not been touched yet, so there is nothing else to undo.
        partial.unlink(missing_ok=True)
        raise
    source.unlink()


def _destination(
    source: Path,
    record: Path | None,
    target: Path,
    sidecar_dir: Path | None,
    library_root: Path | None,
    roots: tuple[Path, ...] = (),
) -> Path:
    """Where one companion belongs once the model is at `target`.

    The record goes where a record for `target` is written today, which is not always the
    mirror of where it was found: one left beside the model before `sidecar_dir` was set
    is collected on the way past, the same as a fresh download's would be.
    """
    if record is not None and source == record:
        return record_path(target, sidecar_dir, library_root, roots)
    return target.with_name(source.name)


def _renamed(
    source: Path,
    path: Path,
    record: Path | None,
    target: Path,
    sidecar_dir: Path | None,
    library_root: Path | None,
    roots: tuple[Path, ...] = (),
) -> Path:
    """What one companion of `path` is called once the model is called `target`.

    Not the same calculation as a move's. There the name is what stays fixed and the folder
    changes; here it is the other way about, and the companions do not all take the model's
    name the same way — the record is `<model>.safetensors.json`, everything else is
    `<model>.txt`. Splitting on the stem rather than rebuilding from a guess keeps
    `.civitai.info` and `.preview.png` intact, both of which look like two extensions to
    anything that reasons in suffixes.
    """
    if record is not None and source == record:
        return record_path(target, sidecar_dir, library_root, roots)
    return source.with_name(target.stem + source.name[len(path.stem):])
