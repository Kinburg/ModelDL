"""A cheap look at a file that tells two different files apart.

Two model files of the same size are usually two different models: every LoRA trained at the
same rank for the same architecture comes out the same number of bytes, and so does every
fine-tune of a checkpoint saved at the same precision. A byte count cannot say which of them
are copies — reading a few small pieces of each can. Pieces that differ are proof that the
files differ. Pieces that agree are not proof that the files are the same: two merges that
left the text encoder untouched can agree wherever they are sampled. Only the hash of the
whole file says that, and the fingerprint is what decides which files are worth hashing.

The same read yields the AutoV1 hash too — the "model hash" A1111 used to show, taken from
sixty-four kilobytes one megabyte in. Civitai still answers to it, which lets a model be
looked up there without reading the rest of the file.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

BLOCK = 64 * 1024
# Where A1111's hash reads from, and how much of the file it reads.
AUTOV1_OFFSET = 0x100000
AUTOV1_LENGTH = 0x10000


@dataclass(frozen=True, slots=True)
class Sample:
    fingerprint: str
    # None for a file too small to reach the place the hash is taken from.
    autov1: str | None


def sample(path: Path) -> Sample:
    """Read the pieces the fingerprint and the AutoV1 hash are made of.

    About a third of a megabyte, whatever the size of the file: the start, the quarters and
    the end, and the piece AutoV1 takes. A file smaller than the pieces is read whole.
    """
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        digest = hashlib.sha256(size.to_bytes(8, "little"))
        if size <= 5 * BLOCK:
            digest.update(handle.read())
        else:
            for offset in (0, size // 4, size // 2, size * 3 // 4, size - BLOCK):
                handle.seek(offset)
                digest.update(handle.read(BLOCK))
        autov1 = None
        if size > AUTOV1_OFFSET:
            handle.seek(AUTOV1_OFFSET)
            autov1 = hashlib.sha256(handle.read(AUTOV1_LENGTH)).hexdigest()[:8]
    return Sample(fingerprint=digest.hexdigest()[:32], autov1=autov1)
