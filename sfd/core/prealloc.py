"""Cheap preallocation.

Sizing the `.part` file up front lets every connection write at its own offset from the
start. The naive way to do that — `truncate(size)` — is catastrophically expensive on
Windows: NTFS zero-fills the extension, so a 18.5 GB file costs about a minute and a half
of pure disk churn before a single byte of the download arrives, and burns a full write
cycle of the file's size on the SSD for nothing.

Marking the file sparse first makes the same `truncate` instant, because the zero regions
are then recorded in metadata instead of written out. As chunks land, the sparse regions
fill in; once the download completes there are no holes left and the result is an ordinary
file.

If the sparse flag cannot be set (non-NTFS volume, unusual filesystem), we skip
preallocation entirely rather than pay the zero-fill. Chunks are handed out in ascending
order, so writers stay clustered near the current end of the file and NTFS never has a
large gap to zero on demand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FSCTL_SET_SPARSE = 0x000900C4
FILE_BEGIN = 0


def _kernel32():
    """Bind the two calls we need, or None when they are unavailable."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    ]
    k32.DeviceIoControl.restype = wintypes.BOOL
    k32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD,
    ]
    k32.SetFilePointerEx.restype = wintypes.BOOL
    k32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    k32.SetEndOfFile.restype = wintypes.BOOL
    return k32


def _handle(fileno: int):
    import msvcrt

    try:
        return msvcrt.get_osfhandle(fileno)
    except OSError:
        return None


def _mark_sparse(fileno: int) -> bool:
    import ctypes
    from ctypes import wintypes

    k32 = _kernel32()
    handle = _handle(fileno) if k32 else None
    if k32 is None or handle is None:
        return False

    returned = wintypes.DWORD()
    return bool(
        k32.DeviceIoControl(
            wintypes.HANDLE(handle), FSCTL_SET_SPARSE, None, 0, None, 0,
            ctypes.byref(returned), None,
        )
    )


def _set_end_of_file(fileno: int, size: int) -> bool:
    """Resize via the Win32 API rather than `file.truncate()`.

    Python's `truncate` goes through the CRT's `_chsize_s`, which extends a file by
    *writing* zero buffers in a loop. That defeats the sparse flag entirely — the zeros get
    materialised and the operation costs a full pass over the file's size. `SetEndOfFile`
    only moves the end-of-file marker, which is what makes the sparse flag worth setting.
    """
    import ctypes
    from ctypes import wintypes

    k32 = _kernel32()
    handle = _handle(fileno) if k32 else None
    if k32 is None or handle is None:
        return False

    win_handle = wintypes.HANDLE(handle)
    if not k32.SetFilePointerEx(win_handle, ctypes.c_longlong(size), None, FILE_BEGIN):
        return False
    return bool(k32.SetEndOfFile(win_handle))


def allocate(path: Path, size: int) -> bool:
    """Ensure `path` exists and, if it can be done cheaply, is exactly `size` bytes.

    Returns True when the file was sized up front. False means writes will extend it as
    they go, which is correct but leaves the file short until the tail chunk lands.
    """
    exists = path.exists()
    with open(path, "r+b" if exists else "w+b") as fh:
        if fh.seek(0, os.SEEK_END) == size:
            return True

        # On a POSIX filesystem truncate is already a metadata-only operation.
        if sys.platform != "win32":
            fh.truncate(size)
            return True

        if not _mark_sparse(fh.fileno()):
            return False
        return _set_end_of_file(fh.fileno(), size)


def is_sparse_supported(path: Path) -> bool:
    """Probe whether the volume backing `path` accepts the sparse flag."""
    probe = path.parent / f".{path.name}.sparse-probe"
    try:
        with open(probe, "w+b") as fh:
            return _mark_sparse(fh.fileno())
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
