"""Identifying a model before downloading it.

The point of reading headers over a Range request: a 20 GB file can be classified from its
first megabyte, so the decision about where it belongs is made — and shown to the user —
before a single gigabyte moves.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..core.errors import SfdError
from ..core.types import FileIdentity
from ..providers.base import Provider
from .sniff import DEFAULT_HEAD, Sniff, sniff

# A safetensors index for a large sharded model can be a few megabytes. Beyond this the
# answer is not worth more round trips.
MAX_FETCH = 16 * 1024 * 1024


async def sniff_remote(
    provider: Provider,
    identity: FileIdentity,
    client: httpx.AsyncClient,
    budget: int = DEFAULT_HEAD,
) -> Sniff:
    """Read just enough of a remote file to identify it.

    Never raises: an unidentifiable file is a normal outcome that the caller resolves with
    other evidence, and a download must not be blocked because a probe failed.
    """
    try:
        target = await provider.resolve(identity, client)
    except SfdError:
        return Sniff()

    head = b""
    want = budget
    while True:
        try:
            response = await client.get(
                target.url,
                headers={
                    **target.headers,
                    "Range": f"bytes=0-{want - 1}",
                    "Accept-Encoding": "identity",
                },
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return sniff(head) if head else Sniff()

        if response.status_code not in (200, 206):
            return sniff(head) if head else Sniff()
        head = response.content

        result = sniff(head)
        if result.needs_bytes is None or result.needs_bytes <= len(head):
            return result
        if result.needs_bytes > MAX_FETCH:
            # Identifiable in principle, but not worth the transfer. Return what we know.
            return result
        want = result.needs_bytes


def sniff_file(path: Path, budget: int = DEFAULT_HEAD) -> Sniff:
    """Same question, asked of a file already on disk."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(budget)
            result = sniff(head)
            if result.needs_bytes and result.needs_bytes <= MAX_FETCH:
                fh.seek(0)
                result = sniff(fh.read(result.needs_bytes))
            return result
    except OSError:
        return Sniff()
