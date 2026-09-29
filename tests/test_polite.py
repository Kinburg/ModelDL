"""Asking Civitai and the Hub no faster than they want to be asked.

A 429 is waited out and asked again, for as long as the service says or for a pause that
grows; the pause holds back every request to that service, not only the one refused; and
nothing but the services' own hosts is paced — the CDNs the files come from are not.
"""

from __future__ import annotations

import asyncio
import time
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from sfd.core import polite
from sfd.providers.base import walk_redirects


@pytest.fixture(autouse=True)
def pacing(monkeypatch):
    """On, and quick: the waits of the real thing, scaled down to what a test can sit out."""
    monkeypatch.setattr(polite, "ENABLED", True)
    monkeypatch.setattr(polite, "BACKOFF", (0.02, 0.04, 0.08))
    monkeypatch.setattr(polite, "PACES", {
        "civitai": polite.Pace("Civitai", at_once=2, spacing=0.0),
        "huggingface": polite.Pace("HuggingFace", at_once=4, spacing=0.0),
    })
    told: list[tuple[str, float]] = []
    polite.listen(lambda name, seconds: told.append((name, seconds)))
    yield told
    polite._listener = None


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def refusing(times: int, headers: dict | None = None, then: int = 200):
    """A server that says 429 so many times, and then answers."""
    calls: list[float] = []

    def handler(request):
        calls.append(time.monotonic())
        if len(calls) <= times:
            return httpx.Response(429, headers=headers or {})
        return httpx.Response(then, json={"ok": True})
    return handler, calls


async def test_a_429_is_waited_out_and_asked_again(pacing):
    handler, calls = refusing(2)
    async with client(handler) as http:
        response = await polite.get(http, "https://civitai.com/api/v1/models/1")

    assert response.status_code == 200 and len(calls) == 3
    assert [name for name, _ in pacing] == ["Civitai", "Civitai"], "the page is told each pause"
    assert calls[1] - calls[0] >= 0.02 and calls[2] - calls[1] >= 0.04, "and each pause is longer"


async def test_the_wait_the_service_asks_for_is_the_one_kept(pacing):
    handler, calls = refusing(1, {"Retry-After": "0.3"})
    async with client(handler) as http:
        await polite.get(http, "https://civitai.red/api/v1/models/1")

    assert calls[1] - calls[0] >= 0.3
    assert 0.3 <= pacing[0][1] <= 0.36


async def test_it_gives_up_when_the_waits_are_spent(monkeypatch):
    handler, calls = refusing(10)
    async with client(handler) as http:
        response = await polite.get(http, "https://huggingface.co/api/models/org/repo")

    assert response.status_code == 429, "the refusal is the caller's to report"
    assert len(calls) == 1 + len(polite.BACKOFF)


async def test_a_pause_holds_back_every_request_to_that_service(monkeypatch):
    started: dict[str, float] = {}

    def handler(request):
        path = request.url.path
        if path.endswith("/first") and "first" not in started:
            started["first"] = time.monotonic()
            return httpx.Response(429, headers={"Retry-After": "0.3"})
        started.setdefault(path.rsplit("/", 1)[-1], time.monotonic())
        return httpx.Response(200)

    async with client(handler) as http:
        first = asyncio.create_task(polite.get(http, "https://civitai.com/api/v1/first"))
        await asyncio.sleep(0.05)
        await polite.get(http, "https://civitai.com/api/v1/second")
        await first

    assert started["second"] - started["first"] >= 0.3, "it waited out the pause asked of the other"


async def test_other_services_and_the_cdns_are_not_held_back(monkeypatch):
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "civitai.com":
            return httpx.Response(429, headers={"Retry-After": "5"})
        return httpx.Response(200)

    async with client(handler) as http:
        refused = asyncio.create_task(polite.get(http, "https://civitai.com/api/v1/models/1"))
        await asyncio.sleep(0.05)
        began = time.monotonic()
        await polite.get(http, "https://huggingface.co/api/models/a/b")
        await polite.get(http, "https://image.civitai.com/x/original=true/1.jpeg")
        await polite.get(http, "https://b2.civitai.com/file/model.safetensors")
        assert time.monotonic() - began < 1.0
        refused.cancel()

    assert seen[1:] == ["huggingface.co", "image.civitai.com", "b2.civitai.com"]


async def test_a_429_from_a_cdn_is_passed_on_as_it_is(pacing):
    handler, calls = refusing(1)
    async with client(handler) as http:
        response = await polite.get(http, "https://image.civitai.com/x/1.jpeg")

    assert response.status_code == 429 and len(calls) == 1 and pacing == []


async def test_only_a_question_is_asked_twice():
    handler, calls = refusing(1)
    async with client(handler) as http:
        request = http.build_request("POST", "https://civitai.com/api/v1/model-versions/by-hash", json=["ab"])
        response = await polite.send(http, request)

    assert response.status_code == 429 and len(calls) == 1


async def test_requests_are_spaced_and_only_so_many_run_at_once(monkeypatch):
    monkeypatch.setattr(polite, "PACES", {**polite.PACES, "civitai": polite.Pace("Civitai", at_once=2, spacing=0.05)})
    starts: list[float] = []
    running = 0
    most = 0

    async def handler(request):
        nonlocal running, most
        starts.append(time.monotonic())
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.12)
        running -= 1
        return httpx.Response(200)

    async with client(handler) as http:
        await asyncio.gather(*(polite.get(http, f"https://civitai.com/api/v1/models/{n}") for n in range(5)))

    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.045 and most == 2


async def test_the_redirect_walk_waits_out_a_429_on_the_services_own_hop(pacing):
    calls: list[str] = []

    def handler(request):
        calls.append(f"{request.method} {request.url.host}")
        if request.url.host == "civitai.com":
            if len([c for c in calls if "civitai.com" in c]) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(307, headers={"location": "https://b2.example/file?sig=1"})
        return httpx.Response(200, headers={"content-length": "7"})

    async with client(handler) as http:
        walk = await walk_redirects(http, "https://civitai.com/api/download/models/3", {"Authorization": "Bearer k"})

    assert walk.status == 200 and walk.url == "https://b2.example/file?sig=1"
    assert calls == ["HEAD civitai.com", "HEAD civitai.com", "HEAD b2.example"]


async def test_switched_off_it_is_only_a_request(monkeypatch, pacing):
    monkeypatch.setattr(polite, "ENABLED", False)
    handler, calls = refusing(1)
    async with client(handler) as http:
        response = await polite.get(http, "https://civitai.com/api/v1/models/1")

    assert response.status_code == 429 and len(calls) == 1


async def test_the_page_is_told_how_long_the_pause_is(tmp_path):
    from sfd.jobs.db import Database
    from sfd.jobs.manager import Manager
    from sfd.settings import Settings

    database = Database(tmp_path / "queue.db")
    manager = Manager(Settings(_path=str(tmp_path / "settings.json")), database)
    events = manager.subscribe()

    manager._slowed("Civitai", 30.0)

    event = events.get_nowait()
    assert (event["type"], event["service"], event["seconds"]) == ("slow_down", "Civitai", 30.0)
    assert 29 < event["until"] - time.time() <= 30
    database.close()


@pytest.mark.parametrize("headers, expected", [
    ({"Retry-After": "30"}, 30.0),
    ({"Retry-After": "1.5"}, 1.5),
    ({"RateLimit": '"api";r=0;t=147'}, 147.0),
    ({}, None),
    ({"Retry-After": "soon"}, None),
])
def test_how_long_the_service_asked_for(headers, expected):
    assert polite.asked_wait(httpx.Response(429, headers=headers)) == expected


def test_a_date_in_retry_after_is_counted_from_now():
    later = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)

    assert 55 <= polite.asked_wait(httpx.Response(429, headers={"Retry-After": later})) <= 60


@pytest.mark.parametrize("url, service", [
    ("https://civitai.com/api/v1/models/1", "civitai"),
    ("https://civitai.red/api/download/models/3?fileId=4", "civitai"),
    ("https://civitai.com/models/1", None),
    ("https://image.civitai.com/x/1.jpeg", None),
    ("https://huggingface.co/org/repo/resolve/main/model.safetensors", "huggingface"),
    ("https://cas-bridge.xethub.hf.co/xet-bridge/abc", None),
    ("https://example.com/model.safetensors", None),
])
def test_which_requests_keep_a_services_pace(url, service):
    assert polite.service_of(httpx.URL(url)) == service
