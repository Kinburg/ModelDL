from __future__ import annotations

import pytest

from sfd.core.chunks import ChunkMap, pick_chunk_size


def test_covers_the_whole_file_without_gaps():
    cm = ChunkMap(size=1000, chunk_size=256)
    assert [(c.start, c.end) for c in cm.chunks] == [(0, 256), (256, 512), (512, 768), (768, 1000)]
    assert sum(c.size for c in cm.chunks) == 1000


def test_completed_prefix_stops_at_the_first_hole():
    cm = ChunkMap(size=1000, chunk_size=256)
    cm.chunks[0].done = 256
    cm.chunks[1].done = 100          # partial — contributes, then terminates the run
    cm.chunks[2].done = 256          # complete but not contiguous, must not count
    assert cm.completed_prefix() == 356
    assert cm.downloaded == 612


def test_completed_prefix_spans_everything_when_full():
    cm = ChunkMap(size=1000, chunk_size=256)
    for c in cm.chunks:
        c.done = c.size
    assert cm.completed_prefix() == 1000
    assert cm.complete


def test_claim_hands_out_each_chunk_once_until_released():
    cm = ChunkMap(size=1000, chunk_size=256)
    first = cm.claim()
    second = cm.claim()
    assert first is not None and second is not None
    assert first.index != second.index

    cm.release(first)
    again = cm.claim()
    assert again is first  # released chunks go back into circulation


def test_idle_worker_steals_half_of_the_biggest_remaining_chunk():
    """The tail problem: without this, the slowest connection holds everyone else up."""
    cm = ChunkMap(size=64 * 1024**2, chunk_size=32 * 1024**2)
    busy = cm.claim()
    other = cm.claim()
    assert busy is not None and other is not None
    other.done = other.size          # its owner is finished; only `busy` still has work
    busy.done = 4 * 1024**2

    stolen = cm.claim()
    assert stolen is not None

    remaining_before = 32 * 1024**2 - 4 * 1024**2
    midpoint = busy.offset + remaining_before // 2
    assert busy.end == midpoint, "the victim's range must shrink"
    assert stolen.start == midpoint and stolen.end == 32 * 1024**2
    # Still one contiguous tiling, in order — completed_prefix() depends on it.
    assert [(c.start, c.end) for c in cm.chunks] == sorted((c.start, c.end) for c in cm.chunks)
    assert sum(c.size for c in cm.chunks) == 64 * 1024**2


def test_stealing_stops_when_the_remainder_is_too_small_to_split():
    cm = ChunkMap(size=64 * 1024**2, chunk_size=32 * 1024**2)
    busy = cm.claim()
    other = cm.claim()
    other.done = other.size
    busy.done = busy.size - 1024      # a kilobyte left; splitting it helps nobody

    assert cm.claim() is None


def test_a_split_chunk_completes_at_its_new_boundary():
    cm = ChunkMap(size=64 * 1024**2, chunk_size=32 * 1024**2)
    busy = cm.claim()
    cm.claim().done = 32 * 1024**2
    stolen = cm.claim()
    assert stolen is not None

    # The owner keeps writing and must finish at the shrunken end, not the original one.
    busy.done = busy.size
    assert busy.complete
    stolen.done = stolen.size
    assert cm.complete
    assert cm.completed_prefix() == 64 * 1024**2


def test_claim_returns_none_when_everything_is_done():
    cm = ChunkMap(size=500, chunk_size=256)
    for c in cm.chunks:
        c.done = c.size
    assert cm.claim() is None


def test_roundtrip_preserves_progress():
    cm = ChunkMap(size=1000, chunk_size=256)
    cm.chunks[0].done = 256
    cm.chunks[2].done = 17
    restored = ChunkMap.from_dict(cm.to_dict())
    assert [c.done for c in restored.chunks] == [256, 0, 17, 0]
    assert restored.completed_prefix() == 256


def test_roundtrip_survives_a_split_layout():
    # After stealing, boundaries no longer follow a fixed stride, so they have to be
    # persisted explicitly or a resume would misplace every byte after the split.
    cm = ChunkMap(size=64 * 1024**2, chunk_size=32 * 1024**2)
    busy = cm.claim()
    cm.claim().done = 32 * 1024**2
    busy.done = 4 * 1024**2
    stolen = cm.claim()
    assert stolen is not None

    restored = ChunkMap.from_dict(cm.to_dict())
    assert [(c.start, c.end, c.done) for c in restored.chunks] == [
        (c.start, c.end, c.done) for c in cm.chunks
    ]
    assert stolen.start in {c.start for c in restored.chunks}
    assert len(restored.chunks) == 3  # the split really did survive the round trip


def test_corrupt_state_cannot_inflate_progress():
    # A hand-edited or truncated state file must never make us skip real bytes.
    cm = ChunkMap(size=1000, chunk_size=256)
    payload = cm.to_dict()
    payload["spans"] = [[0, 256, 9999], [256, 512, -5], [512, 768, 0], [768, 1000, 0]]
    restored = ChunkMap.from_dict(payload)
    assert [c.done for c in restored.chunks] == [256, 0, 0, 0]


@pytest.mark.parametrize(
    "spans",
    [
        [[0, 256, 0], [256, 512, 0]],                              # does not reach size
        [[0, 256, 0], [512, 768, 0], [768, 1000, 0]],              # gap
        [[0, 512, 0], [256, 768, 0], [768, 1000, 0]],              # overlap
        [[0, 256, 0], [256, 2000, 0]],                             # runs past the end
    ],
)
def test_spans_that_do_not_tile_the_file_are_rejected(spans):
    # A layout that disagrees with the bytes on disk would relocate data silently.
    with pytest.raises(ValueError):
        ChunkMap.from_dict({"size": 1000, "chunk_size": 256, "spans": spans})


@pytest.mark.parametrize("size", [1, 10 * 1024**2, 2 * 1024**3, 40 * 1024**3])
def test_chunk_size_stays_within_bounds(size):
    chosen = pick_chunk_size(size)
    assert 8 * 1024**2 <= chosen <= 64 * 1024**2
