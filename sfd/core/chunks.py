"""Chunk bookkeeping.

The file is cut into fixed-size chunks and each chunk tracks how many of its bytes are
already on disk. Two invariants make everything else safe:

  1. `done` never decreases. A failed attempt keeps the bytes it managed to write and the
     next attempt resumes at `start + done`. The only thing that resets progress is the
     remote file actually changing, and that wipes the whole map.

  2. Bytes below `done` are already durably written before `done` is advanced. That lets
     the hasher read the completed prefix concurrently without locking against writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

MIN_CHUNK = 8 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024
TARGET_CHUNKS = 512

# Smallest slice worth handing to a fresh connection. Below this the request overhead
# outweighs the parallelism.
MIN_STEAL = 4 * 1024 * 1024


def pick_chunk_size(size: int) -> int:
    """Aim for ~512 chunks, clamped to a sane byte range.

    Chunk count mainly affects how evenly work spreads across connections; resume
    granularity is already byte-level thanks to per-chunk `done`.
    """
    raw = max(size // TARGET_CHUNKS, 1)
    return max(MIN_CHUNK, min(MAX_CHUNK, raw))


@dataclass(slots=True)
class Chunk:
    index: int
    start: int
    end: int          # exclusive
    done: int = 0
    claimed: bool = False

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def offset(self) -> int:
        """Absolute position of the next byte to fetch."""
        return self.start + self.done

    @property
    def complete(self) -> bool:
        return self.done >= self.size


class ChunkMap:
    """Ordered set of chunks covering [0, size)."""

    def __init__(self, size: int, chunk_size: int) -> None:
        self.size = size
        self.chunk_size = chunk_size
        self.chunks: list[Chunk] = []
        for i, start in enumerate(range(0, size, chunk_size)):
            self.chunks.append(Chunk(index=i, start=start, end=min(start + chunk_size, size)))
        if not self.chunks:  # zero-byte file
            self.chunks.append(Chunk(index=0, start=0, end=0, done=0))

    # --- progress ---------------------------------------------------------

    @property
    def downloaded(self) -> int:
        return sum(c.done for c in self.chunks)

    @property
    def complete(self) -> bool:
        return all(c.complete for c in self.chunks)

    def completed_prefix(self) -> int:
        """Length of the contiguous run of bytes from offset 0.

        This is what the hasher is allowed to read. A partially filled chunk still
        contributes its `done` bytes, because they are contiguous from the chunk start —
        but it terminates the run.
        """
        total = 0
        for c in self.chunks:
            total += c.done
            if not c.complete:
                break
        return total

    # --- work distribution ------------------------------------------------

    def claim(self) -> Chunk | None:
        """Hand out work: an unclaimed chunk, or failing that, half of somebody else's.

        Single-threaded event loop, so no lock is needed — callers must not await between
        checking and claiming, and this method never awaits.
        """
        for c in self.chunks:
            if not c.complete and not c.claimed:
                c.claimed = True
                return c
        return self._steal()

    def _steal(self) -> Chunk | None:
        """Split the slowest-looking chunk so an idle connection can finish it in half.

        Without this, the tail of a download collapses onto whichever connection happens to
        be holding the last chunk — and connection quality varies by an order of magnitude,
        so that can easily be the worst one. Everybody else sits idle waiting for it.

        Splitting is safe against a live writer: the owner re-reads `end` on every block and
        stops as soon as its range shrinks past its current position. The bytes already in
        flight beyond the split point are simply dropped.
        """
        victim: Chunk | None = None
        most = 0
        for c in self.chunks:
            if c.complete or not c.claimed:
                continue
            remaining = c.end - c.offset
            if remaining > most:
                victim, most = c, remaining

        # Splitting only pays off if both halves are still worth a request of their own.
        if victim is None or most < 2 * MIN_STEAL:
            return None

        midpoint = victim.offset + most // 2
        tail = Chunk(index=len(self.chunks), start=midpoint, end=victim.end, claimed=True)
        victim.end = midpoint

        # Keep the list ordered by start offset — completed_prefix() walks it in order and
        # relies on the chunks being contiguous.
        self.chunks.insert(self.chunks.index(victim) + 1, tail)
        for i, c in enumerate(self.chunks):
            c.index = i
        return tail

    def release(self, chunk: Chunk) -> None:
        """Put a chunk back after a failed attempt; its `done` bytes are retained."""
        chunk.claimed = False

    def incomplete(self) -> Iterator[Chunk]:
        return (c for c in self.chunks if not c.complete)

    # --- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Boundaries are stored explicitly rather than derived from chunk_size: stealing
        # splits chunks, so after a steal the layout no longer follows a fixed stride.
        return {
            "size": self.size,
            "chunk_size": self.chunk_size,
            "spans": [[c.start, c.end, c.done] for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMap:
        size = int(data["size"])
        spans = data.get("spans")
        if not spans:
            raise ValueError("state file has no chunk spans")

        cm = cls.__new__(cls)
        cm.size = size
        cm.chunk_size = int(data["chunk_size"])
        cm.chunks = []

        # Rebuild strictly: the spans must tile [0, size) with no gap and no overlap.
        # Anything else means the file on disk and this map disagree about what is where,
        # and acting on that would corrupt the download.
        expected = 0
        for index, span in enumerate(spans):
            start, end, done = (int(v) for v in span)
            if start != expected or end < start or end > size:
                raise ValueError(f"chunk span {index} is not contiguous: {span}")
            # Clamp defensively: a truncated or hand-edited state file must not make us
            # believe we have more bytes than the chunk can hold.
            done = max(0, min(done, end - start))
            cm.chunks.append(Chunk(index=index, start=start, end=end, done=done))
            expected = end

        if expected != size:
            raise ValueError(f"chunk spans cover {expected} of {size} bytes")
        return cm
