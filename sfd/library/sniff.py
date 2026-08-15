"""Reading a model's own header to find out what it is.

Both formats we care about put their metadata at byte zero, which means the question "what
kind of model is this" can be answered from the first megabyte — over a Range request,
before committing to a 20 GB transfer.

This matters because the alternatives are unreliable. Filenames lie, and file extensions lie
loudest of all: `.gguf` is widely assumed to mean a language model, but ComfyUI ships
quantised Flux and Wan diffusion models in exactly that container. `general.architecture`
inside the header settles it in a way no naming convention can.

  safetensors  8-byte little-endian header length, then that many bytes of JSON holding
               every tensor name plus an optional `__metadata__` block.
  GGUF         "GGUF" magic, version, tensor count, then a typed key/value table whose
               first entries are conventionally `general.*`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

# Enough for the vast majority of headers; a large safetensors index can exceed it, in
# which case the reader reports exactly how much more it needs.
DEFAULT_HEAD = 1024 * 1024
MAX_HEADER = 64 * 1024 * 1024

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types, from the format specification.
(
    _UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL,
    _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64,
) = range(13)

_FIXED = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4), _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}


@dataclass(slots=True)
class Sniff:
    """What the first bytes of a file say about it."""

    format: str | None = None                 # "safetensors" | "gguf" | None
    tensor_names: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    architecture: str | None = None           # GGUF general.architecture
    # Set when the header is longer than the bytes provided. The caller can fetch this
    # many bytes total and try again, instead of guessing.
    needs_bytes: int | None = None

    @property
    def understood(self) -> bool:
        return self.format is not None and self.needs_bytes is None


def sniff(head: bytes) -> Sniff:
    """Identify `head`, the opening bytes of a file. Never raises on malformed input."""
    if head[:4] == GGUF_MAGIC:
        return _sniff_gguf(head)
    if len(head) >= 8:
        result = _sniff_safetensors(head)
        if result is not None:
            return result
    return Sniff()


# --- safetensors ------------------------------------------------------------


def _sniff_safetensors(head: bytes) -> Sniff | None:
    (length,) = struct.unpack("<Q", head[:8])
    # A plausible header is the only evidence that this is safetensors at all — there is no
    # magic number. Anything absurd means the format is something else.
    if not 2 <= length <= MAX_HEADER:
        return None

    end = 8 + length
    if end > len(head):
        return Sniff(format="safetensors", needs_bytes=end)

    try:
        payload = json.loads(head[8:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    metadata = payload.get("__metadata__")
    names = [k for k in payload if k != "__metadata__"]
    return Sniff(
        format="safetensors",
        tensor_names=names,
        metadata={
            str(k): str(v) for k, v in (metadata or {}).items() if isinstance(metadata, dict)
        },
    )


# --- GGUF -------------------------------------------------------------------


def _sniff_gguf(head: bytes) -> Sniff:
    reader = _Reader(head)
    try:
        reader.skip(4)                       # magic
        version = reader.u32()
        reader.u64()                         # tensor count
        kv_count = reader.u64()
    except _Truncated as exc:
        return Sniff(format="gguf", needs_bytes=exc.needed)

    if version not in (1, 2, 3):
        # Unknown revision: say it is GGUF and stop guessing at the layout.
        return Sniff(format="gguf")

    metadata: dict[str, str] = {}
    architecture: str | None = None
    try:
        for _ in range(min(kv_count, 4096)):
            key = reader.string()
            value = reader.value()
            if key.startswith("general.") and isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
            if key == "general.architecture" and isinstance(value, str):
                architecture = value
                # Everything after this is tokeniser tables and tensor descriptions; the
                # question has already been answered.
                break
    except _Truncated as exc:
        return Sniff(
            format="gguf", metadata=metadata, architecture=architecture, needs_bytes=exc.needed
        )
    except (UnicodeDecodeError, struct.error, ValueError):
        return Sniff(format="gguf", metadata=metadata, architecture=architecture)

    return Sniff(format="gguf", metadata=metadata, architecture=architecture)


class _Truncated(Exception):
    def __init__(self, needed: int) -> None:
        super().__init__(f"need {needed} bytes")
        self.needed = needed


class _Reader:
    """Bounds-checked little-endian reader that reports how far it wanted to go."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise _Truncated(end)
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def skip(self, n: int) -> None:
        self._take(n)

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > MAX_HEADER:
            raise ValueError("implausible string length")
        return self._take(length).decode("utf-8", errors="replace")

    def value(self, kind: int | None = None):
        kind = self.u32() if kind is None else kind
        if kind in _FIXED:
            fmt, size = _FIXED[kind]
            return struct.unpack(fmt, self._take(size))[0]
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item_kind = self.u32()
            count = self.u64()
            if count > 1_000_000:
                raise ValueError("implausible array length")
            return [self.value(item_kind) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")
