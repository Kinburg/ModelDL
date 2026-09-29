"""Asking a service's API no faster than it wants to be asked.

Civitai and the Hub both answer too many requests with 429, and neither says in advance how
many is too many — Civitai sends no rate-limit headers at all. So every request to either API
goes through the gate of its service: a few at once, their starts spaced out. A 429 or a 503
is waited out and asked again rather than taken for an answer — for as long as the service
says in `Retry-After`, or, when it does not say, for a pause that grows each time. And the
pause is the whole app's, not the one request's: while Civitai has asked for a rest, nothing
else asks it anything either, which is the difference between slowing down and knocking on
the same door with four hands.

The files themselves come from CDNs, and none of this is about them: only the services' own
hosts are paced — Civitai's `/api/`, which its download links are part of, and the Hub.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import re
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx


@dataclass(frozen=True, slots=True)
class Pace:
    name: str         # the service, as the page names it while it waits
    at_once: int      # requests under way together
    spacing: float    # seconds between the starts of two of them


# Fixed on purpose. Neither service publishes a number to aim at; what matters is not to
# burst, and to back off when told to, which is what the waits below are for.
PACES = {
    "civitai": Pace("Civitai", at_once=2, spacing=0.5),
    "huggingface": Pace("HuggingFace", at_once=4, spacing=0.25),
}
CIVITAI_HOSTS = {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}
HUB_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}

# Too many requests, and a service too busy to answer: both are over by waiting.
RETRY_ON = {429, 503}
# The waits between attempts when the service does not say how long: a limit that resets
# every minute is over by the third, and one that is not over by the fourth is not going to
# be for someone watching a progress bar.
BACKOFF = (5.0, 15.0, 45.0, 120.0)
# The longest one wait may be, whatever the service asks for.
MAX_WAIT = 120.0
# Off in the test suite, whose stub servers answer at once: pacing them only slows it down.
ENABLED = True


def service_of(url: httpx.URL) -> str | None:
    """Which service's pace a request keeps, or None for one that keeps none."""
    host = (url.host or "").lower()
    if host in CIVITAI_HOSTS and url.path.startswith("/api/"):
        return "civitai"
    if host in HUB_HOSTS:
        return "huggingface"
    return None


def asked_wait(response: httpx.Response) -> float | None:
    """How long the service asked to be left alone, if it said: `Retry-After`, in seconds or
    as a date, or the Hub's `RateLimit` with the seconds until its window resets."""
    value = (response.headers.get("retry-after") or "").strip()
    if value:
        if re.fullmatch(r"\d+(\.\d+)?", value):
            return float(value)
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    limit = response.headers.get("ratelimit") or ""
    match = re.search(r"(?:^|[;,])\s*t=(\d+)", limit)
    return float(match.group(1)) if match else None


class _Gate:
    """One service's door: how many are through it, when the next may go, and until when
    the service has asked everyone to wait."""

    def __init__(self, pace: Pace) -> None:
        self.pace = pace
        self.slots = asyncio.Semaphore(pace.at_once)
        self.lock = asyncio.Lock()
        self.next_start = 0.0
        self.until = 0.0

    async def turn(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                start = max(now, self.next_start, self.until)
                self.next_start = start + self.pace.spacing
            if start > now:
                await asyncio.sleep(start - now)
            # A pause the service asked for while this request waited its turn is waited
            # out as well: it was asked of everyone.
            if self.until <= time.monotonic():
                return

    def hold(self, seconds: float) -> None:
        self.until = max(self.until, time.monotonic() + seconds)


# One gate per service and per event loop: asyncio's locks belong to the loop that made
# them, and the test suite runs a fresh loop for every test.
_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _Gate]] = (
    weakref.WeakKeyDictionary()
)


def _gate(service: str) -> _Gate:
    gates = _gates.setdefault(asyncio.get_running_loop(), {})
    if service not in gates:
        gates[service] = _Gate(PACES[service])
    return gates[service]


# Who is told that a service asked for a pause — the page, through the queue's events — as
# (the service's name, the seconds of the pause).
_listener: Callable[[str, float], None] | None = None


def listen(callback: Callable[[str, float], None]) -> None:
    global _listener
    _listener = callback


def unlisten(callback: Callable[[str, float], None]) -> None:
    global _listener
    if _listener is callback:
        _listener = None


def _tell(name: str, seconds: float) -> None:
    listener = _listener
    if listener is not None:
        try:
            listener(name, seconds)
        except Exception:  # noqa: BLE001 - a page that cannot be told must not stop the request
            pass


async def send(
    client: httpx.AsyncClient, request: httpx.Request, *, stream: bool = False, **options: Any
) -> httpx.Response:
    """`client.send`, at the pace of the service the request goes to, with a refusal for too
    many requests waited out and asked again. A request to anywhere else is sent as it is.

    Only a GET or a HEAD is asked again: asking twice is harmless only for a question. What
    comes back after the last attempt is returned as it is, 429 and all, for the caller to
    report as it always has.
    """
    service = service_of(request.url) if ENABLED else None
    if service is None:
        return await client.send(request, stream=stream, **options)
    gate = _gate(service)
    again = request.method in ("GET", "HEAD")
    attempt = 0
    while True:
        await gate.turn()
        async with gate.slots:
            response = await client.send(request, stream=stream, **options)
        if response.status_code not in RETRY_ON or not again or attempt >= len(BACKOFF):
            return response
        asked = asked_wait(response)
        wait = min(MAX_WAIT, BACKOFF[attempt] if asked is None else asked)
        # A little more than asked, and not the same little for everyone: requests that were
        # refused together should not all come back in the same instant.
        wait *= random.uniform(1.0, 1.2)
        await response.aclose()
        gate.hold(wait)
        _tell(gate.pace.name, wait)
        attempt += 1


async def get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Any = None,
    headers: Any = None,
    follow_redirects: Any = httpx.USE_CLIENT_DEFAULT,
) -> httpx.Response:
    """`client.get`, by way of `send`."""
    request = client.build_request("GET", url, params=params, headers=headers)
    return await send(client, request, follow_redirects=follow_redirects)
