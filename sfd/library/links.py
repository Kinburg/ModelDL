"""One file under several names: hard links, made and undone.

A model kept in two folders because two nodes each read their own takes its room twice. A
hard link is the same file under a second name: every path keeps working, a loader sees an
ordinary file, and the room is taken once. There is no original and no link — every name
is the file, equally — and the data goes only when the last name does.

Two things follow that the rest of the app has to say out loud. Deleting one name frees
nothing while another is left. And a program that saves by writing a new file and renaming
it over the old name quietly gives that name a copy of its own again: nothing breaks, the
room saved is simply taken again.

A hard link cannot cross a volume, and needs a file system that has them — NTFS does; the
FAT family on memory cards and USB sticks does not.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

from .relocate import Progress, copy_bytes

# The names a link and a copy are made under before they are swapped in. Left behind only
# by a crash, and then offered by the cleanup view.
LINKING = ".linking"
SEPARATING = ".separating"


def identity(path: Path | str) -> tuple[str, int] | None:
    """Which file this name is — its volume and number, as one string — and how many names
    it has. None when the file cannot be read, or its file system numbers no files."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if not stat.st_ino:
        return None
    return f"{stat.st_dev}:{stat.st_ino}", stat.st_nlink


def same_volume(a: Path | str, b: Path | str) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def link_over(keep: Path, other: Path) -> None:
    """Make `other` another name of the file at `keep`.

    The new name is made beside `other` first and swapped in with one rename, so `other` is
    never missing — and if the swap is refused, which on Windows means a program has it
    open, `other` is left exactly as it was.
    """
    temp = other.with_name(other.name + LINKING)
    temp.unlink(missing_ok=True)
    os.link(keep, temp)
    try:
        os.replace(temp, other)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def separate(
    path: Path, progress: Progress | None = None, stop: threading.Event | None = None
) -> None:
    """Give this name a file of its own: every byte copied beside it, with its dates, and
    swapped in with a rename. Until the swap the name is still the shared file; a copy that
    is stopped or fails takes its fragment with it."""
    partial = path.with_name(path.name + SEPARATING)
    copy_bytes(path, partial, progress, stop, what=f"making {path.name} a copy of its own")
    try:
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def names(path: Path) -> list[Path]:
    """Every name the file at `path` has, on the whole of its volume — not only the ones in
    the library. Windows can list them; anywhere else only this one is known."""
    path = Path(path)
    if sys.platform != "win32":
        return [path]
    try:
        return _windows_names(path) or [path]
    except (OSError, AttributeError, ValueError):
        return [path]


def _windows_names(path: Path) -> list[Path]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstFileNameW
    find_first.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                           ctypes.c_wchar_p]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextFileNameW
    find_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_wchar_p]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int

    size = 32768
    buffer = ctypes.create_unicode_buffer(size)
    length = ctypes.c_uint32(size)
    handle = find_first(str(path), 0, ctypes.byref(length), buffer)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        return []
    # The names come back relative to the volume: `\models\vae\ae.safetensors`.
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    found: list[Path] = []
    try:
        while True:
            found.append(Path(drive + buffer.value))
            length = ctypes.c_uint32(size)
            if not find_next(handle, ctypes.byref(length), buffer):
                break
    finally:
        find_close(handle)
    return found
