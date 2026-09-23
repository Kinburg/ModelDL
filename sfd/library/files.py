"""What counts as a model file, and how a split one is recognised.

Shared by the scan that builds the library and by everything that has to tell a model from
the files kept beside it. Kept apart from both, because each of them needs it and neither
should have to import the other to get it.
"""

from __future__ import annotations

import re
from pathlib import Path

# The containers model weights actually ship in. `.bin` is on the list because PyTorch
# checkpoints from HuggingFace still use it, and off it below a size, because so does every
# tokenizer and test fixture in a repository that happens to sit in a model folder.
MODEL_SUFFIXES = frozenset({
    ".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".onnx",
})
SMALL_BIN = 1024 * 1024

# `model-00001-of-00004.safetensors`: one model in four files. HuggingFace's convention,
# and llama.cpp's for GGUF splits.
_SHARD = re.compile(r"^(?P<base>.+?)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.[^.]+)$")


def is_model_file(name: str, size: int | None = None) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix not in MODEL_SUFFIXES:
        return False
    if suffix == ".bin" and size is not None and size < SMALL_BIN:
        return False
    return True


def shard_of(name: str) -> tuple[str, int, int] | None:
    """(set name, index, count) for a shard, None for a file that is whole on its own.

    The set name keeps the suffix — `model.safetensors` for its shards — so two sets that
    differ only in format are two sets.
    """
    match = _SHARD.match(name)
    if match is None:
        return None
    count = int(match.group("count"))
    index = int(match.group("index"))
    if count < 2 or not 1 <= index <= count:
        return None
    return match.group("base") + match.group("suffix"), index, count
