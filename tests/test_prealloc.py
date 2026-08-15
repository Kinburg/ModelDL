from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from sfd.core.prealloc import allocate, is_sparse_supported


def test_creates_a_file_of_exactly_the_right_size(tmp_path: Path):
    target = tmp_path / "f.bin"
    assert allocate(target, 5_000_000)
    assert target.stat().st_size == 5_000_000


def test_is_idempotent(tmp_path: Path):
    target = tmp_path / "f.bin"
    allocate(target, 5_000_000)
    assert allocate(target, 5_000_000)
    assert target.stat().st_size == 5_000_000


def test_extends_an_existing_short_file(tmp_path: Path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"hello")
    assert allocate(target, 1_000_000)
    assert target.stat().st_size == 1_000_000
    with open(target, "rb") as fh:
        assert fh.read(5) == b"hello"


def test_arbitrary_offsets_are_writable_and_read_back(tmp_path: Path):
    """What preallocation is for: every connection writes wherever it likes."""
    target = tmp_path / "f.bin"
    allocate(target, 1_000_000)

    with open(target, "r+b", buffering=0) as fh:
        fh.seek(900_000)
        fh.write(b"tail")
    with open(target, "r+b", buffering=0) as fh:
        fh.seek(0)
        fh.write(b"head")

    data = target.read_bytes()
    assert len(data) == 1_000_000
    assert data[:4] == b"head"
    assert data[900_000:900_004] == b"tail"
    assert data[500_000:500_010] == b"\0" * 10  # the untouched middle reads as zeros


@pytest.mark.skipif(sys.platform != "win32", reason="the zero-fill trap is Windows-specific")
def test_large_allocation_does_not_zero_fill(tmp_path: Path):
    """Regression guard for a genuinely expensive mistake.

    `file.truncate()` on Windows extends by writing zeros through the CRT — about 75
    seconds and 18 GB of pointless writes for a large model file, before the download even
    starts. Marking the file sparse and moving the end-of-file marker instead costs
    milliseconds.
    """
    target = tmp_path / "big.bin"
    if not is_sparse_supported(target):
        pytest.skip("volume does not support sparse files")

    size = 8 * 1024**3
    started = time.monotonic()
    assert allocate(target, size)
    elapsed = time.monotonic() - started

    assert target.stat().st_size == size
    assert elapsed < 5.0, f"allocation took {elapsed:.1f}s — it is zero-filling again"

    os.remove(target)
