"""Where the app keeps its own files: the settings, the queue, the caches.

Run from source, that is the folder it is started in — `run.cmd` starts it in the project.
Run as the built program it is the folder the program is in, wherever it was started from:
a shortcut with another "Start in", a terminal somewhere else. A copy of that folder is a
copy of the app, settings and history included, and moving the folder moves them with it.

The exception is a folder the app may not write to — `Program Files` — where its files go to
the user's own application data instead. Files already beside the program win over that,
even there: somebody put them there, and they are the ones meant.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# What says a folder already holds this app's own files.
OWN_FILES = ("settings.json", "queue.db")


@dataclass(frozen=True, slots=True)
class Home:
    folder: Path
    settings: Path
    db: Path


def locate(
    data: str | None = None,
    settings: str | None = None,
    db: str | None = None,
    *,
    frozen: bool | None = None,
    executable: Path | None = None,
    cwd: Path | None = None,
    local_appdata: Path | None = None,
) -> Home:
    """The folder the app's files belong in, and the settings file and queue in it.

    A path given on the command line means what it says from where the command was typed,
    so those are made absolute here, before anything moves the working directory.
    """
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    cwd = Path.cwd() if cwd is None else cwd
    if data:
        folder = _absolute(data, cwd)
    elif frozen:
        folder = beside_program(Path(executable or sys.executable), local_appdata)
    else:
        folder = cwd
    return Home(
        folder=folder,
        settings=_absolute(settings, cwd) if settings else folder / "settings.json",
        db=_absolute(db, cwd) if db else folder / "queue.db",
    )


def beside_program(executable: Path, local_appdata: Path | None = None) -> Path:
    """The program's own folder, or the user's application data when it cannot be written."""
    # Not resolved: that would turn a mapped drive into its network path, and a folder
    # shown as \\server\share\… is not the one the user put the program in.
    beside = executable.absolute().parent
    if holds_own_files(beside) or writable(beside):
        return beside
    appdata = local_appdata if local_appdata is not None else _local_appdata()
    return appdata / "ModelDL" if appdata is not None else beside


def holds_own_files(folder: Path) -> bool:
    return any((folder / name).is_file() for name in OWN_FILES)


def writable(folder: Path) -> bool:
    """Whether a file can be made in the folder — asked by making one, since permissions on
    Windows say less than trying does."""
    try:
        handle, name = tempfile.mkstemp(dir=folder, prefix=".modeldl-", suffix=".probe")
    except OSError:
        return False
    os.close(handle)
    try:
        os.unlink(name)
    except OSError:
        pass
    return True


def _absolute(path: str, cwd: Path) -> Path:
    return Path(os.path.normpath(cwd / Path(path).expanduser()))


def _local_appdata() -> Path | None:
    if sys.platform == "win32":
        value = os.environ.get("LOCALAPPDATA")
        return Path(value) if value else None
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) if base else Path.home() / ".local" / "share"
