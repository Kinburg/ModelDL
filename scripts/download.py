"""Download anything you can paste.

    python scripts/download.py <url-or-repo> [-o DIR] [--include '*.gguf']

Accepts a Hub page URL, a direct download link, or a bare `org/name`. A repository or a
folder expands into its files; use --include / --exclude to take only part of it.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from sfd.core.diskinfo import detect_disk_kind  # noqa: E402
from sfd.core.errors import SfdError  # noqa: E402
from sfd.core.transfer import Transfer, TransferOptions  # noqa: E402
from sfd.core.types import ProgressSnapshot  # noqa: E402
from sfd.library import sidecar  # noqa: E402
from sfd.library.classify import Verdict, classify  # noqa: E402
from sfd.library.inspect import sniff_remote  # noqa: E402
from sfd.library.layout import Layout, adopt, flat  # noqa: E402
from sfd.providers.registry import Item, Resolution, expand, source_url  # noqa: E402
from sfd.settings import hf_login  # noqa: E402


def human(n: float | None) -> str:
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds > 86400 * 7:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Reporter:
    """Progress for one file. Falls back to periodic lines when output is redirected."""

    LOG_EVERY = 15.0

    def __init__(self, label: str) -> None:
        self.label = label
        self.baseline: int | None = None
        self._interactive = sys.stdout.isatty()
        self._last_log = 0.0

    def __call__(self, p: ProgressSnapshot) -> None:
        if self.baseline is None:
            self.baseline = p.downloaded
        fraction = p.fraction or 0.0
        body = (
            f"{fraction * 100:5.1f}%  {human(p.downloaded)}/{human(p.total)}  "
            f"{human(p.speed)}/s  conn {p.connections}  ETA {duration(p.eta)}"
        )
        if self._interactive:
            filled = int(fraction * 24)
            sys.stdout.write(f"\r[{'#' * filled}{'-' * (24 - filled)}] {body}   ")
            sys.stdout.flush()
            return
        now = time.monotonic()
        if now - self._last_log < self.LOG_EVERY:
            return
        self._last_log = now
        print(f"{time.strftime('%H:%M:%S')}  {self.label}  {body}", flush=True)


def select(items: list[Item], include: list[str], exclude: list[str]) -> list[Item]:
    chosen = items
    if include:
        chosen = [i for i in chosen if any(fnmatch.fnmatch(i.filename, p) for p in include)]
    if exclude:
        chosen = [i for i in chosen if not any(fnmatch.fnmatch(i.filename, p) for p in exclude)]
    return chosen


Plan = list[tuple[Item, Verdict | None, Path]]


async def make_plan(
    items: list[Item], resolution: Resolution, layout: Layout, classify_files: bool
) -> Plan:
    """Work out where each file goes before anything is downloaded.

    Classification reads each file's header over a Range request — a megabyte per file — so
    the whole plan, including the uncertain entries, is on screen before a single gigabyte
    moves.
    """
    if not classify_files:
        return [(item, None, layout.root) for item in items]

    timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
    plan: Plan = []
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for item in items:
            # A single-file link carries no size until something asks; since we are already
            # talking to the server, fill it in so the plan states real numbers.
            if item.size is None or not item.filename:
                try:
                    info = await resolution.provider.probe(item.identity, client)
                    item.size = item.size or info.size
                    item.sha256 = item.sha256 or info.sha256
                    item.filename = item.filename or info.filename
                except SfdError:
                    pass

            header = await sniff_remote(resolution.provider, item.identity, client)
            verdict = classify(item.filename, item.meta, header)
            plan.append((item, verdict, layout.destination(verdict, item.filename)))
    return plan


async def write_sidecar(
    path: Path,
    verdict: Verdict,
    item: Item,
    resolution: Resolution,
    sidecar_dir: str | None = None,
    library_root: str | None = None,
) -> None:
    record = sidecar.Record(
        filename=path.name,
        provider=resolution.provider.name,
        source_url=source_url(item.identity, item.meta.get("host")),
        sha256=item.sha256,
        size=item.size,
        meta=item.meta,
    )
    sidecar.write(
        path, verdict, record,
        sidecar_dir=Path(sidecar_dir) if sidecar_dir else None,
        library_root=Path(library_root) if library_root else None,
    )

    if item.meta.get("preview_url"):
        timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await sidecar.fetch_preview(path, item.meta, client)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="URL, or a HuggingFace repo id")
    parser.add_argument("-o", "--out", default="downloads", help="destination directory")
    parser.add_argument("-c", "--connections", type=int, default=16)
    parser.add_argument("--include", action="append", default=[], metavar="GLOB")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN") or hf_login()[0],
        help="defaults to $HF_TOKEN, then to the token hf auth login saved; prefer either "
             "of those, a command line is visible to every process on the machine",
    )
    parser.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN"))
    parser.add_argument("--min-speed", type=float, default=64.0, help="stall floor, KB/s")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--force-connections", action="store_true",
                        help="do not clamp the connection count to the disk type")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    parser.add_argument(
        "--all", action="store_true",
        help="confirm fetching every file when a link expands into more than one",
    )
    parser.add_argument(
        "--library", metavar="DIR",
        help="sort into an existing model tree (e.g. F:\\ComfyUI\\Shared) instead of "
             "dropping everything in one folder",
    )
    parser.add_argument("--profile", default="comfyui", choices=["comfyui", "a1111"])
    parser.add_argument(
        "--yes", action="store_true",
        help="accept uncertain placements instead of stopping to ask",
    )
    parser.add_argument("--no-sidecar", action="store_true",
                        help="do not write metadata next to each model")
    parser.add_argument("--sidecar-dir", metavar="DIR",
                        help="collect the .json records here instead of beside each model")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            resolution: Resolution = await expand(
                args.source, client,
                hf_token=args.hf_token, civitai_token=args.civitai_token,
            )
        except (SfdError, ValueError) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    items = select(resolution.items, args.include, args.exclude)
    if not items:
        print("nothing matched the include/exclude filters", file=sys.stderr)
        return 1

    layout = adopt(Path(args.library), args.profile) if args.library else flat(out)
    plan = await make_plan(items, resolution, layout, classify_files=bool(args.library))

    # A dry run must always say what it would do, even for a single file — silence reads
    # as "nothing matched" when it actually means "one file, all set".
    if args.dry_run or len(resolution.items) > 1 or args.library:
        total = sum(i.size for i in items if i.size) or None
        print(f"{resolution.label}: {len(items)} of {len(resolution.items)} file(s), "
              f"{human(total)}")
        for item, verdict, destination in plan:
            name = item.filename or "(name resolved at download time)"
            print(f"  {human(item.size):>10}  {name}")
            if args.library:
                mark = "?" if verdict and verdict.needs_confirmation else " "
                print(f"             {mark} -> {destination}")
                if verdict:
                    print(f"               {verdict.category.value} "
                          f"({verdict.confidence}) — {verdict.reason}")
        print()

    unsure = [(i, v) for i, v, _ in plan if v is not None and v.needs_confirmation]
    if unsure and not (args.yes or args.dry_run):
        print(
            f"{len(unsure)} file(s) could not be placed confidently. Review the '?' lines "
            f"above and re-run with --yes to accept, or drop --library to keep everything "
            f"in one folder.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        return 0

    # One Civitai link can name five quantisations of the same checkpoint — 67 GB when you
    # wanted 12. Expanding is useful; expanding silently is not.
    if len(items) > 1 and not (args.all or args.include or args.exclude):
        print(
            f"this expands to {len(items)} files. Narrow it with --include, "
            f"or pass --all to fetch everything.",
            file=sys.stderr,
        )
        return 1

    print(f"target disk: {detect_disk_kind(out).value}\n")
    options = TransferOptions(
        connections=args.connections,
        min_speed=args.min_speed * 1024,
        verify_hash=not args.no_verify,
        respect_disk_kind=not args.force_connections,
    )

    started = time.monotonic()
    moved = 0
    skipped = 0
    for index, (item, verdict, destination) in enumerate(plan, 1):
        label = item.filename or resolution.label
        prefix = f"[{index}/{len(plan)}] " if len(plan) > 1 else ""
        transfer = Transfer(
            resolution.provider, item.identity, destination, options, Reporter(label)
        )
        try:
            path = await transfer.run()
        except SfdError as exc:
            print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\ninterrupted — progress is saved, run the same command to resume")
            return 130

        if not args.no_sidecar and verdict is not None:
            await write_sidecar(
                path, verdict, item, resolution, args.sidecar_dir, args.library
            )

        if transfer.bytes_transferred == 0:
            skipped += 1
            print(f"{prefix}{path.name}: already present, verified")
            continue
        moved += transfer.bytes_transferred
        if sys.stdout.isatty():
            print()
        print(f"{prefix}{path.name}: {human(transfer.bytes_transferred)} fetched")

    elapsed = time.monotonic() - started
    print(f"\ndone: {layout.root}")
    if skipped:
        print(f"{skipped} file(s) were already present")
    print(f"{human(moved)} transferred in {duration(elapsed)} "
          f"({human(moved / max(elapsed, 0.001))}/s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
