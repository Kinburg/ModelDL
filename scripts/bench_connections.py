"""Find out where the throughput ceiling actually is.

Downloads a fixed amount of data at several concurrency levels and discards it, reporting
throughput and client CPU cost for each. The shape of the resulting table says which of
three very different problems you have:

  * throughput scales with connections  -> the server caps each connection; raise the count
  * throughput is flat                  -> an aggregate cap on the account, IP or route
  * throughput flat AND cpu/sec near 1  -> the client is the bottleneck, and only separate
                                           processes will help, because one Python process
                                           cannot parse and decrypt on more than one core

    python scripts/bench_connections.py <url> [--token ...] [--mib-per-run 96]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from sfd.core.errors import SfdError  # noqa: E402
from sfd.providers.direct import DirectProvider, make_identity  # noqa: E402

LEVELS = (1, 2, 4, 8, 16)


async def fetch_range(client: httpx.AsyncClient, target, start: int, length: int) -> int:
    got = 0
    headers = {
        **target.headers,
        "Range": f"bytes={start}-{start + length - 1}",
        "Accept-Encoding": "identity",
    }
    async with client.stream(
        "GET", target.url, headers=headers, follow_redirects=False
    ) as resp:
        if resp.status_code not in (200, 206):
            raise SfdError(f"unexpected status {resp.status_code}")
        async for data in resp.aiter_bytes():
            got += len(data)          # deliberately not written anywhere
    return got


async def run_level(client, target, size: int, connections: int, budget: int, cursor: int):
    """Split `budget` bytes across `connections` parallel range requests."""
    per = budget // connections
    starts = []
    for i in range(connections):
        start = (cursor + i * per) % max(size - per, 1)
        starts.append(start)

    cpu_before = time.process_time()
    wall_before = time.monotonic()
    got = sum(await asyncio.gather(*(fetch_range(client, target, s, per) for s in starts)))
    wall = time.monotonic() - wall_before
    cpu = time.process_time() - cpu_before
    return got, wall, cpu


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("CIVITAI_TOKEN"),
        help="bearer token; defaults to $HF_TOKEN / $CIVITAI_TOKEN",
    )
    parser.add_argument("--mib-per-run", type=int, default=96)
    parser.add_argument("--levels", default=",".join(str(n) for n in LEVELS))
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="repeat each level; per-connection throughput varies enough that a single "
             "sample can be off by an order of magnitude",
    )
    args = parser.parse_args()
    levels = [int(n) for n in args.levels.split(",") if n.strip()]

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else None
    identity = make_identity(args.url, headers)
    budget = args.mib_per_run * 1024**2

    timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
    limits = httpx.Limits(max_connections=max(LEVELS) + 4)
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, follow_redirects=False
    ) as client:
        provider = DirectProvider()
        info = await provider.probe(identity, client)
        if not info.size:
            print("cannot benchmark a file of unknown size", file=sys.stderr)
            return 1
        total_mib = args.mib_per_run * len(levels) * args.repeats
        print(f"file    : {info.filename}  ({info.size / 1024**3:.2f} GiB)")
        print(f"budget  : {args.mib_per_run} MiB per run, {total_mib} MiB total\n")

        print(f"{'conn':>5}  {'run':>4}  {'MiB/s':>9}  {'Mbit/s':>9}  "
              f"{'per conn':>9}  {'cpu/sec':>8}")
        print("-" * 56)

        cursor = 0
        for connections in levels:
            samples: list[float] = []
            for run in range(1, args.repeats + 1):
                # A fresh target per run, and a different region of the file each time, so
                # no run benefits from another's edge-cache warmth.
                target = await provider.resolve(identity, client)
                got, wall, cpu = await run_level(
                    client, target, info.size, connections, budget, cursor
                )
                cursor = (cursor + budget * 3) % max(info.size - budget, 1)

                mib = got / 1024**2 / wall
                samples.append(mib)
                print(
                    f"{connections:>5}  {run:>4}  {mib:>9.1f}  {mib * 8:>9.1f}  "
                    f"{mib / connections:>9.1f}  {cpu / wall:>8.2f}"
                )
            if len(samples) > 1:
                print(f"{connections:>5}  {'best':>4}  {max(samples):>9.1f}  "
                      f"{max(samples) * 8:>9.1f}")

    print("\ncpu/sec near 1.0 means one core is saturated — that is the client's ceiling,")
    print("and only separate processes get past it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
