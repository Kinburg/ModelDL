"""End-to-end tests against a deliberately hostile local server.

These are the tests that matter. Everything else in the package exists so that these pass:
the file that lands on disk is byte-for-byte correct, or nothing lands at all.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from sfd.core.errors import (
    ChecksumMismatch,
    NotBinaryContent,
    RangeIgnored,
    TransferFailed,
)
from sfd.core.transfer import Transfer, TransferOptions
from sfd.providers.direct import DirectProvider, make_identity
from tests.http_stub import StubServer

DATA = os.urandom(1_000_000)
CHUNK = 64 * 1024  # 16 chunks, so multi-connection paths are actually exercised


def options(**kwargs) -> TransferOptions:
    defaults = dict(
        connections=4,
        chunk_size=CHUNK,
        max_attempts=30,
        # The disk clamp is about real throughput, not correctness; pinning it keeps the
        # tests deterministic on whatever machine they run on.
        respect_disk_kind=False,
        # Stall detection needs a long observation window, and these transfers are over in
        # milliseconds. Disabled here; covered directly in test_speed.py.
        min_speed=0.0,
        # Real backoff is measured in seconds; the retry logic is identical either way.
        backoff_base=0.005,
        backoff_max=0.05,
    )
    defaults.update(kwargs)
    return TransferOptions(**defaults)


async def download(server: StubServer, dest: Path, **kwargs) -> Path:
    transfer = Transfer(
        DirectProvider(), make_identity(server.url), dest, options(**kwargs)
    )
    return await transfer.run()


# --- the happy path ---------------------------------------------------------


async def test_downloads_verifies_and_cleans_up(tmp_path: Path):
    with StubServer(DATA) as server:
        path = await download(server, tmp_path)

    assert path.name == "model.safetensors"
    assert path.read_bytes() == DATA
    # No debris left behind.
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.part.json"))


async def test_single_connection_matches_multi(tmp_path: Path):
    with StubServer(DATA) as server:
        path = await download(server, tmp_path / "one.bin", connections=1)
    assert path.read_bytes() == DATA


# --- the failure modes this project exists for ------------------------------


async def test_survives_connections_dropping_mid_body(tmp_path: Path):
    """Every response is cut off partway through; the file must still come out intact."""
    with StubServer(DATA) as server:
        server.state.fail_after = 9_000   # less than one chunk, so no response completes
        server.state.fail_times = 40
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA
    assert server.state.data_requests > 16  # it really did have to retry


async def test_range_bound_signature_is_not_poisoned_by_our_own_probe(tmp_path: Path):
    """Regression: HuggingFace's Xet bridge binds the signature to the probe's byte range.

    Resolving with `Range: bytes=0-0` yields a URL that answers every real chunk request
    with `403 Auth failed: invalid range` — the probe destroys the URL it returns, and the
    download can never start no matter how many times it retries. Probing with HEAD, which
    carries no Range at all, is what keeps this working.
    """
    with StubServer(DATA) as server:
        server.state.bind_range = True
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA


async def test_falls_back_to_a_ranged_get_when_head_is_refused(tmp_path: Path):
    with StubServer(DATA) as server:
        server.state.head_supported = False
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA


async def test_rotating_signature_is_re_resolved(tmp_path: Path):
    """The CDN invalidates the signed URL constantly. We should just keep re-minting it."""
    with StubServer(DATA) as server:
        server.state.rotate_every = 2
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA
    # One resolve at the start plus many forced refreshes.
    assert server.state.resolve_requests > 3


async def test_resume_does_not_refetch_what_is_already_on_disk(tmp_path: Path):
    with StubServer(DATA) as server:
        # First run: one connection, no retry budget, dies after ~30 KB.
        server.state.fail_after = 30_000
        server.state.fail_times = 1
        with pytest.raises(TransferFailed):
            await download(server, tmp_path, connections=1, max_attempts=1)

        part = tmp_path / "model.safetensors.part"
        assert part.exists() and (tmp_path / "model.safetensors.part.json").exists()

        served_before = server.state.bytes_served
        assert served_before >= 30_000

        # Second run picks up where the first stopped.
        server.state.fail_after = None
        server.state.fail_times = 0
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA
    resumed_bytes = server.state.bytes_served - served_before
    assert resumed_bytes < len(DATA), "resume re-downloaded the whole file"


async def test_refuses_to_write_when_the_server_ignores_range(tmp_path: Path):
    """The corruption case.

    A partial file exists, we ask to resume from byte 30000, and the server answers 200 with
    the whole file instead of 206. Appending would produce an oversized broken file;
    overwriting would throw away verified bytes. The only safe move is to write nothing.
    """
    with StubServer(DATA) as server:
        server.state.fail_after = 30_000
        server.state.fail_times = 1
        with pytest.raises(TransferFailed):
            await download(server, tmp_path, connections=1, max_attempts=1)

        part = tmp_path / "model.safetensors.part"
        good = part.read_bytes()[:30_000]
        assert good == DATA[:30_000]

        server.state.fail_after = None
        server.state.fail_times = 0
        server.state.ignore_range = True

        # No chunk ever starts here — the server has stopped advertising range support at
        # all, and run() refuses before the workers do. Nothing wraps it.
        with pytest.raises(RangeIgnored):
            await download(server, tmp_path, connections=1, max_attempts=1)

        # The bytes we already trusted are exactly as they were.
        assert part.read_bytes()[:30_000] == good


async def test_refuses_a_200_in_place_of_a_206_mid_transfer(tmp_path: Path):
    """The other half of the same defect.

    Here the server still answers the small resolve probe with a well-behaved 206, so the
    transfer starts in ranged mode — and then a chunk request comes back 200 with the whole
    body. This is the response a downloader must never write: appending it corrupts the
    file, overwriting it discards verified bytes.
    """
    with StubServer(DATA) as server:
        server.state.fail_after = 30_000
        server.state.fail_times = 1
        with pytest.raises(TransferFailed):
            await download(server, tmp_path, connections=1, max_attempts=1)

        part = tmp_path / "model.safetensors.part"
        good = part.read_bytes()[:30_000]

        # Small probes stay honest; anything chunk-sized comes back as a full-body 200.
        server.state.fail_after = None
        server.state.fail_times = 0
        server.state.ignore_range_over = 1

        with pytest.raises(TransferFailed) as failure:
            await download(server, tmp_path, connections=1, max_attempts=1)
        assert isinstance(failure.value.__cause__, RangeIgnored)

        assert part.read_bytes()[:30_000] == good
        assert part.stat().st_size == len(DATA), "the preallocated file was resized"


async def test_an_edge_that_ignores_range_once_does_not_lose_the_download(tmp_path: Path):
    """The transient case, and by far the common one.

    One CDN node answers a chunk request with the whole file instead of the range asked
    for, and the next connection lands somewhere sane. Nothing was written from the bad
    response — the check runs before the first byte — so there is nothing to recover and
    nothing at risk. Failing the transfer over it strands a download that succeeds on the
    very next attempt, which is what makes this so maddening to report: by the time anyone
    looks, the edge is behaving again.
    """
    with StubServer(DATA) as server:
        server.state.ignore_range_over = 1
        server.state.ignore_range_over_times = 1

        path = await download(server, tmp_path, connections=1)

    assert path.read_bytes() == DATA
    assert not list(tmp_path.glob("*.part"))


async def test_a_partial_survives_an_edge_that_ignores_range_and_resumes_after(tmp_path: Path):
    """The same misbehaviour with bytes already on disk — the case with something to lose.

    The retry must resume from where the partial left off, not restart, and the bytes that
    were already verified must be the same ones in the finished file.
    """
    with StubServer(DATA) as server:
        server.state.fail_after = 30_000
        server.state.fail_times = 1
        with pytest.raises(TransferFailed):
            await download(server, tmp_path, connections=1, max_attempts=1)

        part = tmp_path / "model.safetensors.part"
        good = part.read_bytes()[:30_000]
        assert good == DATA[:30_000]

        server.state.fail_after = None
        server.state.fail_times = 0
        server.state.ignore_range_over = 1
        server.state.ignore_range_over_times = 2
        served_before = server.state.ranged_bytes_served

        path = await download(server, tmp_path, connections=1)

    assert path.read_bytes() == DATA
    resumed = server.state.ranged_bytes_served - served_before
    assert resumed < len(DATA), "the retry restarted from zero instead of resuming"


async def test_a_permanently_broken_edge_still_gives_up_with_the_reason(tmp_path: Path):
    """Retrying must not turn a hopeless case into an endless one."""
    with StubServer(DATA) as server:
        server.state.ignore_range_over = 1

        with pytest.raises(TransferFailed) as failure:
            await download(server, tmp_path, connections=1, max_attempts=3)

    assert isinstance(failure.value.__cause__, RangeIgnored)
    assert "the server sent the whole file" in str(failure.value)


async def test_idle_connections_steal_work_instead_of_waiting(tmp_path: Path):
    """More connections than chunks must still produce a byte-perfect file.

    The real download finished with five of six connections idle while the last one crawled
    at 1.9 MB/s. Splitting a chunk under its owner is delicate — the owner is mid-stream
    when its range shrinks — so the thing to prove is that the result is still exact.
    """
    big = os.urandom(48 * 1024 * 1024)
    with StubServer(big) as server:
        transfer = Transfer(
            DirectProvider(),
            make_identity(server.url),
            tmp_path,
            options(connections=6, chunk_size=16 * 1024 * 1024),
        )
        path = await transfer.run()
        chunks = len(transfer._map.chunks)

    assert path.read_bytes() == big
    assert chunks > 3, "started with three chunks; idle workers should have split them"


async def test_a_file_already_in_place_is_not_fetched_again(tmp_path: Path):
    """Re-fetching gigabytes that are already on disk is the worst thing a downloader does."""
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(DATA)

    with StubServer(DATA) as server:
        path = await download(server, tmp_path)
        assert server.state.bytes_served == 0, "it downloaded a file it already had"

    assert path == dest
    assert path.read_bytes() == DATA


async def test_a_stale_file_of_the_wrong_size_is_replaced(tmp_path: Path):
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(b"an older, shorter version")

    with StubServer(DATA) as server:
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA


async def test_a_file_of_the_right_size_but_wrong_content_is_replaced(tmp_path: Path):
    """Size is weak evidence; when a hash is advertised it decides."""
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(bytes(len(DATA)))

    with StubServer(DATA) as server:
        path = await download(server, tmp_path)
        assert server.state.bytes_served > 0

    assert path.read_bytes() == DATA


async def test_size_alone_decides_when_verification_is_off(tmp_path: Path):
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(bytes(len(DATA)))

    with StubServer(DATA) as server:
        await download(server, tmp_path, verify_existing=False)
        assert server.state.bytes_served == 0

    assert dest.read_bytes() == bytes(len(DATA))  # left alone, as instructed


async def test_login_page_is_not_saved_as_a_model(tmp_path: Path):
    """Civitai answers unauthenticated requests with 200 text/html."""
    with StubServer(DATA) as server:
        server.state.serve_html = True
        with pytest.raises(NotBinaryContent):
            await download(server, tmp_path)

    assert not list(tmp_path.iterdir())


async def test_checksum_mismatch_quarantines_instead_of_publishing(tmp_path: Path):
    with StubServer(DATA) as server:
        # Origin advertises a hash that does not match what the CDN serves.
        server.state.linked_etag = hashlib.sha256(b"something else").hexdigest()
        with pytest.raises(ChecksumMismatch):
            await download(server, tmp_path)

    assert not (tmp_path / "model.safetensors").exists()
    corrupt = tmp_path / "model.safetensors.part.corrupt"
    assert corrupt.exists(), "the data should be kept for inspection, not silently deleted"
    assert not (tmp_path / "model.safetensors.part.json").exists()


async def test_hash_is_verified_on_the_happy_path(tmp_path: Path):
    """Guard against the verification silently not running at all."""
    with StubServer(DATA) as server:
        assert server.state.linked_etag == hashlib.sha256(DATA).hexdigest()
        path = await download(server, tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == server.state.linked_etag


async def test_cdn_etag_is_never_mistaken_for_a_checksum(tmp_path: Path):
    """Regression: a 64-hex ETag from the CDN is not the file's SHA256.

    On Xet-backed HuggingFace repos the CDN serves the Xet content id in the ETag — same
    shape as a SHA256, entirely different value. Treating it as a checksum turns a perfect
    18 GB download into a quarantined "corrupt" file. Only the origin's x-linked-etag counts.
    """
    with StubServer(DATA) as server:
        # The decoy is well-formed and wrong; verification must ignore it and use the
        # origin's value, which is correct.
        assert server.state.cdn_etag != hashlib.sha256(DATA).hexdigest()
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA


async def test_verification_is_skipped_rather_than_failed_without_a_known_hash(tmp_path: Path):
    """No advertised hash means no verdict — not a false one."""
    with StubServer(DATA) as server:
        server.state.linked_etag = "not-a-hash"
        path = await download(server, tmp_path)

    assert path.read_bytes() == DATA
