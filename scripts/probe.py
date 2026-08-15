"""Inspect a URL without downloading it.

    python scripts/probe.py <url>

Shows what the provider can determine up front: filename, size, hash, range support, and
where the redirect chain actually lands. Useful for telling a Xet-backed repo from a plain
LFS one, and for checking whether a link needs a token before starting a 20 GB transfer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from sfd.core.errors import SfdError  # noqa: E402
from sfd.providers.registry import expand, source_url  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    # Separate flags on purpose: one shared token would be sent to whichever service the
    # link happens to name, and a HuggingFace token offered to Civitai is simply rejected.
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN"))
    args = parser.parse_args()

    timeout = httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        # Through the registry, so a Hub page URL or a bare org/name is understood here
        # exactly as it is by the downloader — a probe that disagrees with the tool it is
        # meant to explain is worse than none.
        try:
            resolution = await expand(
                args.url, client,
                hf_token=args.hf_token, civitai_token=args.civitai_token,
            )
        except (SfdError, ValueError) as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        if len(resolution.items) > 1:
            total = sum(i.size for i in resolution.items if i.size)
            print(f"{resolution.label}: {len(resolution.items)} files, {human(total)}\n")
            for item in resolution.items:
                print(f"  {human(item.size or 0):>10}  {item.filename}")
            return 0

        item = resolution.items[0]
        provider = resolution.provider
        try:
            info = await provider.probe(item.identity, client)
            target = await provider.resolve(item.identity, client)
        except SfdError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    print(f"provider      : {provider.name}")
    print(f"canonical     : {source_url(item.identity, item.meta.get('host'))}")
    print(f"filename      : {info.filename}")
    print(f"size          : {info.size:,} bytes ({human(info.size or 0)})")
    print(f"sha256        : {info.sha256 or '(not advertised — verification will be skipped)'}")
    print(f"etag          : {info.etag}")
    print(f"accept ranges : {info.accept_ranges}")
    print(f"content type  : {info.content_type}")
    print(f"authenticated : {bool(args.hf_token or args.civitai_token)}")
    print(f"cdn host      : {httpx.URL(target.url).host}")
    if target.expires_at:
        remaining = (target.expires_at - time.time()) / 60
        stamp = datetime.fromtimestamp(target.expires_at).strftime("%H:%M:%S")
        print(f"signature     : expires {stamp} (in {remaining:.0f} min)")
    else:
        print("signature     : no expiry found in the URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
