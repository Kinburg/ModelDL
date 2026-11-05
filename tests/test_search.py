"""Searching Civitai: what it is asked, what is left out of its answer, and what is kept.

The pages are built the way Civitai's `/api/v1/models` answers — a model with its versions
newest first, each with its files and pictures — and the cases are the ones the filters are
for: a character LoRA among the rest, a version sold for good next to one in early access,
a version that can only be used on the site, an author hidden, a version seen before.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.jobs.search import (
    Filters, Search, describe, downloadable, needs_ownership, paid_out, pick_version, plain_text,
    tag_matcher,
)
from sfd.settings import Settings
from sfd.web.app import create_app

PAGE = "https://civitai.red/api/v1/models"
BY_HASH = "https://civitai.red/api/v1/model-versions/by-hash"


def version(id: int, base: str = "Krea 2", *, sold: bool = False, early: str | None = None,
            nsfw_level: int = 1, sha: str | None = None) -> dict:
    entry = {
        "id": id, "name": f"v{id}", "baseModel": base, "publishedAt": "2026-11-03T10:00:00.000Z",
        "trainedWords": ["word"],
        "files": [{"id": id * 10, "name": f"file{id}.safetensors", "primary": True, "sizeKB": 1024.0,
                   "type": "Model", "metadata": {"fp": "bf16", "format": "SafeTensor"},
                   "hashes": {"SHA256": sha or f"{id:064x}"}}],
        "images": [{"url": f"https://image.civitai.com/x/{id}/width=450/{id}.jpeg", "type": "image",
                    "nsfwLevel": nsfw_level}],
    }
    if sold:
        entry["paidAccess"] = {"permanent": True, "endsAt": None}
    if early:
        entry["paidAccess"] = {"permanent": False, "endsAt": early}
    return entry


def model(id: int, *versions: dict, tags=("style",), author: str = "someone", nsfw: bool = False,
          kind: str = "LORA") -> dict:
    return {
        "id": id, "name": f"Model {id}", "type": kind, "nsfw": nsfw, "poi": False,
        "tags": list(tags), "creator": {"username": author},
        "description": "<p>Skin <b>texture</b> &amp; pores</p><p>Second line</p>",
        "stats": {"downloadCount": 12, "thumbsUpCount": 3},
        "modelVersions": list(versions) or [version(id * 100)],
    }


def page(*items: dict, cursor: str | None = None) -> dict:
    return {"items": list(items), "metadata": {"nextCursor": cursor}}


class Civitai:
    """A stub of the API: pages by cursor, and how each version may be used by hash."""

    def __init__(self, pages: dict[str | None, dict], usage: dict[int, str] | None = None,
                 owned: dict[int, int] | None = None) -> None:
        self.pages = pages
        self.usage = usage or {}
        self.owned = owned or {}
        self.asked: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request)
        url = str(request.url)
        if request.method == "POST" and url == BY_HASH:
            digests = json.loads(request.content)
            answer = [{"id": int(d, 16), "usageControl": self.usage.get(int(d, 16), "Download")}
                      for d in digests]
            return httpx.Response(200, json=answer)
        if request.method == "HEAD" and "/api/download/models/" in url:
            version_id = int(request.url.path.rsplit("/", 1)[1])
            return httpx.Response(self.owned.get(version_id, 403),
                                  headers={"Location": "https://cdn.example/f"})
        if url.startswith(PAGE):
            return httpx.Response(200, json=self.pages[request.url.params.get("cursor")])
        return httpx.Response(404, json={"error": "not stubbed"})

    def kinds(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.asked]


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "queue.db")


def searcher(database: Database, stub: Civitai, token: str = "", here=None) -> Search:
    search = Search(Settings(civitai_token=token), database, here or (lambda: {}))
    search._client = httpx.AsyncClient(transport=httpx.MockTransport(stub))
    return search


RED = {"site": "civitai.red", "base_models": ["Krea 2", "Krea 2 Turbo"], "types": ["LORA", "LoCon", "DoRA"]}


# --- what is asked ----------------------------------------------------------------------------


def test_filters_keep_only_what_they_understand():
    filters = Filters.read({
        "query": "  skin  ", "base_models": ["Krea 2", "", "Krea 2", 5], "types": ["LORA", "Nonsense"],
        "period": "Fortnight", "sort": "Newest", "site": "civitai.evil", "paid": "no_sold",
        "exclude_tags": ["Character*"], "seen": "hide", "hidden_authors": "maybe",
    })

    assert filters.query == "skin"
    assert filters.base_models == ("Krea 2",)
    assert filters.types == ("LORA",)
    assert filters.period == "Week"            # the default, for a period Civitai has not got
    assert filters.site == "civitai.com"       # never a host it was not meant to ask
    assert filters.exclude_tags == ("character*",)
    assert filters.paid == "no_sold"
    assert filters.seen == "hide"
    assert filters.hidden_authors == "hide"
    assert Filters.read(None) == Filters()
    assert Filters.read(filters.to_json()) == filters


def test_civitai_red_is_asked_with_nsfw_and_com_without():
    red = dict(Filters.read(RED).params())
    assert red["nsfw"] == "true"
    assert "nsfw" not in dict(Filters.read({**RED, "site": "civitai.com"}).params())
    params = Filters.read(RED).params()
    assert ("types", "LoCon") in params and ("baseModels", "Krea 2 Turbo") in params
    assert ("sort", "Newest") in params and ("period", "Week") in params


@pytest.mark.parametrize("tags, pattern, hit", [
    (["characters"], "character*", "characters"),
    (["Character,"], "character*", "Character,"),
    (["anime character"], "character*", None),     # whole tags, not words inside them
    (["anime character"], "*character*", "anime character"),
    (["style"], "character", None),
    (["celebrity"], "celebrity", "celebrity"),
    (["style"], "*", None),                         # a bare star would leave everything out
])
def test_tags_are_matched_whole_with_a_star_for_anything(tags, pattern, hit):
    assert tag_matcher([pattern])(tags) == hit


# --- what is found ------------------------------------------------------------------------------


def test_a_model_is_shown_at_its_newest_version_for_the_base_models_asked_for():
    item = model(1, version(12, "Flux.1 D"), version(11, "Krea 2"), version(10, "Krea 2"))

    assert pick_version(item, ["Krea 2"])["id"] == 11
    assert pick_version(item, [])["id"] == 12
    assert pick_version(item, ["Qwen"]) is None


def test_a_card_says_what_the_list_and_the_inspector_need():
    card = describe(model(1, version(11, nsfw_level=16), tags=["style"]), Filters.read(RED))

    assert card.version_id == 11 and card.model_id == 1
    assert card.page == "https://civitai.red/models/1?modelVersionId=11"
    assert card.nsfw is True                   # its first picture is marked adult
    assert card.description == "Skin texture & pores\nSecond line"
    assert card.files[0] == {"id": 110, "name": "file11.safetensors", "size": 1024 * 1024,
                             "type": "Model", "fp": "bf16", "format": "SafeTensor", "primary": True}
    shown = card.to_json()
    assert shown["category"] == "lora" and shown["previews"] == 1
    assert "url" not in json.dumps(shown["previews"])   # the pictures are asked for by number


def test_markup_becomes_text_and_nothing_stays_nothing():
    assert plain_text("<h2>Usage</h2><ul><li>weight 0.8</li></ul>") == "Usage\nweight 0.8"
    assert plain_text(None) == ""


def test_what_the_paid_filter_leaves_out():
    card = describe(model(1, version(11, sold=True)), Filters())
    early = describe(model(2, version(21, early="2099-01-01T00:00:00Z")), Filters())
    free = describe(model(3), Filters())

    assert [paid_out(c, "all") for c in (card, early, free)] == [False, False, False]
    assert [paid_out(c, "no_sold") for c in (card, early, free)] == [True, False, False]
    assert [paid_out(c, "free") for c in (card, early, free)] == [True, True, False]
    assert needs_ownership(card, "no_sold") and not needs_ownership(early, "no_sold")
    assert needs_ownership(early, "free") and not needs_ownership(free, "free")
    card.owned = True
    assert not paid_out(card, "free")           # bought, so it downloads like any other
    early_over = describe(model(4, version(41, early="2020-01-01T00:00:00Z")), Filters())
    assert early_over.access is None            # early access that has ended is free


def test_only_a_version_said_to_be_for_the_site_is_not_downloadable():
    assert downloadable("Download") and downloadable(None)
    assert not downloadable("Generation") and not downloadable("InternalGeneration")


# --- a search ----------------------------------------------------------------------------------------


async def test_a_search_leaves_out_what_its_filters_say_and_says_how_much(database):
    stub = Civitai({None: page(
        model(1),
        model(2, tags=["character"]),
        model(3, version(300, sold=True)),
        model(4, version(400, early="2099-01-01T00:00:00Z")),
        model(5),
    )}, usage={500: "Generation"})
    search = searcher(database, stub)

    body = await search.start(Filters.read({**RED, "exclude_tags": ["character*"], "paid": "no_sold"}))

    assert [c["model_id"] for c in body["cards"]] == [1, 4]
    assert body["left_out"] == {"tags": 1, "only on the site": 1, "paid": 1}
    assert body["scanned"] == 5 and body["done"] is True
    # One lookup by hash for all of them, and no question about buying without a key.
    assert stub.kinds() == ["GET /api/v1/models", "POST /api/v1/model-versions/by-hash"]
    asked = stub.asked[0].url.params
    assert asked.get_list("types") == ["LORA", "LoCon", "DoRA"] and asked["nsfw"] == "true"


@pytest.mark.parametrize("status, kept", [(302, True), (403, False)])
async def test_a_sold_version_is_kept_once_the_key_has_bought_it(database, status, kept):
    stub = Civitai({None: page(model(3, version(300, sold=True)))}, owned={300: status})
    search = searcher(database, stub, token="key")

    body = await search.start(Filters.read({**RED, "paid": "free"}))

    assert bool(body["cards"]) is kept
    if kept:
        assert body["cards"][0]["access"]["owned"] is True
    head = [r for r in stub.asked if r.method == "HEAD"]
    assert len(head) == 1 and head[0].headers["Authorization"] == "Bearer key"
    assert head[0].url.params["fileId"] == "3000"


async def test_showing_every_paid_version_asks_nobody_whether_it_is_bought(database):
    stub = Civitai({None: page(model(3, version(300, sold=True)))})
    body = await searcher(database, stub, token="key").start(Filters.read(RED))

    assert body["cards"][0]["access"]["permanent"] is True
    assert not [r for r in stub.asked if r.method == "HEAD"]


async def test_a_search_reads_on_while_its_filters_leave_too_little(database, monkeypatch):
    from sfd.jobs import search as module

    monkeypatch.setattr(module, "WANTED", 2)
    stub = Civitai({
        None: page(model(1, tags=["character"]), cursor="a"),
        "a": page(model(2), cursor="b"),
        "b": page(model(3), model(4), cursor="c"),
        "c": page(model(5)),
    })
    search = searcher(database, stub)

    first = await search.start(Filters.read({**RED, "exclude_tags": ["character"]}))
    assert [c["model_id"] for c in first["cards"]] == [2, 3, 4]
    assert first["done"] is False and first["scanned"] == 4

    rest = await search.more(first["token"])
    assert [c["model_id"] for c in rest["cards"]] == [5]
    assert rest["done"] is True and rest["shown"] == 4


async def test_seen_versions_are_marked_or_left_out(database):
    stub = Civitai({None: page(model(1), model(2))})
    search = searcher(database, stub)
    database.mark_seen([(100, 1)])

    shown = await search.start(Filters.read(RED))
    assert [c["seen"] for c in shown["cards"]] == [True, False]

    hidden = await search.start(Filters.read({**RED, "seen": "hide"}))
    assert [c["model_id"] for c in hidden["cards"]] == [2]
    assert hidden["left_out"] == {"seen": 1}


async def test_a_hidden_author_is_left_out_or_shown_dimmed(database):
    stub = Civitai({None: page(model(1, author="Spammer"), model(2))})
    search = searcher(database, stub)
    database.set_author_hidden("spammer")

    hidden = await search.start(Filters.read(RED))
    assert [c["model_id"] for c in hidden["cards"]] == [2]

    shown = await search.start(Filters.read({**RED, "hidden_authors": "show"}))
    assert [c["author_hidden"] for c in shown["cards"]] == [True, False]


async def test_a_version_here_is_marked_as_here(database):
    stub = Civitai({None: page(model(1), model(2))})
    search = searcher(database, stub, here=lambda: {1: {100}, 2: {1}})

    body = await search.start(Filters.read(RED))

    assert [c["here"] for c in body["cards"]] == ["this", "other"]


async def test_a_preset_counts_what_it_found_new_and_counts_down(database, monkeypatch):
    from sfd.jobs import search as module

    monkeypatch.setattr(module, "WANTED", 3)
    preset = database.add_preset("Krea 2 LoRA", Filters.read(RED).to_json())
    stub = Civitai({None: page(model(1), model(2), model(3), cursor="more"), "more": page()})
    search = searcher(database, stub)

    body = await search.start(Filters.read(RED), preset["id"])
    assert database.get_preset(preset["id"])["unseen"]["count"] == 3
    assert database.get_preset(preset["id"])["unseen"]["more"] is True

    search.mark_seen([100, 200], token=body["token"])
    assert database.get_preset(preset["id"])["unseen"]["count"] == 1
    search.mark_seen([200], seen=False, token=body["token"])
    assert database.get_preset(preset["id"])["unseen"]["count"] == 2


async def test_a_new_search_of_a_list_takes_the_place_of_its_last_one_only(database, monkeypatch):
    from sfd.jobs import search as module

    monkeypatch.setattr(module, "WANTED", 1)
    stub = Civitai({None: page(model(1), cursor="more"), "more": page(model(2))})
    search = searcher(database, stub)
    krea = database.add_preset("Krea", {})["id"]
    moss = database.add_preset("MOSS", {})["id"]

    old = await search.start(Filters.read(RED), krea)
    other = await search.start(Filters.read(RED), moss)
    free = await search.start(Filters.read(RED))
    new = await search.start(Filters.read(RED), krea)

    with pytest.raises(KeyError):
        await search.more(old["token"])
    # The other lists the page keeps still read on.
    assert [c["model_id"] for c in (await search.more(other["token"]))["cards"]] == [2]
    assert (await search.more(free["token"]))["done"] is True
    assert (await search.more(new["token"]))["done"] is True


async def test_a_search_saved_as_a_preset_goes_on_as_its_search(database, monkeypatch):
    from sfd.jobs import search as module

    monkeypatch.setattr(module, "WANTED", 2)
    stub = Civitai({None: page(model(1), model(2), cursor="more"), "more": page(model(3))})
    search = searcher(database, stub)
    free = await search.start(Filters.read(RED))
    preset = database.add_preset("Krea", Filters.read(RED).to_json())["id"]

    answer = search.adopt(free["token"], preset)

    assert answer["presets"][0]["unseen"] == {**answer["presets"][0]["unseen"], "count": 2, "more": True}
    # A free search after it is another list: the preset's goes on reading.
    await search.start(Filters.read(RED))
    assert [c["model_id"] for c in (await search.more(free["token"]))["cards"]] == [3]
    with pytest.raises(KeyError):
        search.adopt("gone", preset)


async def test_the_cards_a_kept_search_shows_are_kept(database, monkeypatch):
    from sfd.jobs import search as module

    monkeypatch.setattr(module, "CARDS_KEPT", 1)
    stub = Civitai({None: page(model(1), model(2))})
    search = searcher(database, stub)
    preset = database.add_preset("Krea", {})["id"]

    await search.start(Filters.read(RED), preset)
    assert search.card(100) and search.card(200)     # past the limit, but on screen

    search.forget(preset)
    with pytest.raises(KeyError):
        search.card(100)


async def test_a_refusal_from_civitai_is_said(database):
    def refuse(request):
        return httpx.Response(400, json={"error": {"name": "ZodError", "message": "bad period"}})

    search = Search(Settings(), database, lambda: {})
    search._client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    from sfd.jobs.search import SearchError

    with pytest.raises(SearchError, match="bad period"):
        await search.start(Filters.read(RED))


async def test_the_base_models_come_from_civitai_or_from_a_list_of_our_own(database):
    def enums(request):
        return httpx.Response(200, json={"ActiveBaseModel": ["Krea 2", "MOSS Image"]})

    search = Search(Settings(), database, lambda: {})
    search._client = httpx.AsyncClient(transport=httpx.MockTransport(enums))
    assert (await search.options())["base_models"] == ["Krea 2", "MOSS Image"]

    offline = Search(Settings(), database, lambda: {})
    offline._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    answer = await offline.options()
    assert answer["base_models_fallback"] is True and "Krea 2" in answer["base_models"]


# --- what is kept --------------------------------------------------------------------------------


def test_presets_are_kept_in_order(database):
    a = database.add_preset("A", {"site": "civitai.red"})
    b = database.add_preset("B", {})
    c = database.add_preset("C", {})

    database.reorder_presets([c["id"], a["id"]])
    assert [p["name"] for p in database.list_presets()] == ["C", "A", "B"]

    database.update_preset(a["id"], name="A2", unseen={"count": 4})
    changed = database.get_preset(a["id"])
    assert changed["name"] == "A2" and changed["filters"] == {"site": "civitai.red"}
    assert changed["unseen"] == {"count": 4}
    assert database.delete_preset(b["id"]) and not database.delete_preset(b["id"])


def test_seen_and_hidden_authors_are_kept(database):
    database.mark_seen([(1, 10), (2, 20)])
    assert database.seen_versions([1, 2, 3]) == {1, 2}
    database.mark_seen([(1, 10)], seen=False)
    assert database.seen_versions(range(1, 1200)) == {2}

    assert database.set_author_hidden("Spammer")
    assert not database.set_author_hidden("spammer")       # the same name, whatever its case
    assert database.hidden_authors() == ["Spammer"]
    assert database.set_author_hidden("SPAMMER", hidden=False)
    assert database.hidden_authors() == []


# --- through the page -------------------------------------------------------------------------------

LOCAL = {"Origin": "http://127.0.0.1:7788"}


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(download_dir=str(tmp_path / "downloads"), check_updates_on_start=False,
                        fetch_previews=False, _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    app = create_app(settings, database)
    with TestClient(app, base_url="http://127.0.0.1:7788") as test:
        test.search = app.state.manager.search  # type: ignore[attr-defined]
        yield test


def test_presets_are_made_changed_and_deleted_from_the_page(client):
    made = client.post("/api/civitai/presets", json={"name": "  Krea 2   LoRA ", "filters": {**RED, "junk": 1}},
                       headers=LOCAL).json()
    preset = made["preset"]
    assert preset["name"] == "Krea 2 LoRA"
    assert preset["filters"]["site"] == "civitai.red" and "junk" not in preset["filters"]

    renamed = client.put(f"/api/civitai/presets/{preset['id']}", json={"name": "Krea"}, headers=LOCAL).json()
    assert renamed["preset"]["name"] == "Krea" and renamed["preset"]["filters"] == preset["filters"]
    assert client.post("/api/civitai/presets", json={"name": " "}, headers=LOCAL).status_code == 400

    assert client.get("/api/civitai/state").json()["presets"][0]["name"] == "Krea"
    assert client.delete(f"/api/civitai/presets/{preset['id']}", headers=LOCAL).json() == {"presets": []}
    assert client.delete(f"/api/civitai/presets/{preset['id']}", headers=LOCAL).status_code == 404


def test_a_search_from_the_page_and_its_pictures(client):
    stub = Civitai({None: page(model(1))})
    client.search._client = httpx.AsyncClient(transport=httpx.MockTransport(stub))

    body = client.post("/api/civitai/search", json={"filters": RED}, headers=LOCAL).json()
    assert [c["version_id"] for c in body["cards"]] == [100]

    listed = client.get("/api/civitai/versions/100/previews").json()
    assert len(listed["previews"]) == 1
    # Turned off in these settings: no picture is fetched.
    assert client.get("/api/civitai/versions/100/preview/0").status_code == 404
    assert client.get("/api/civitai/versions/999/previews").status_code == 404
    assert client.post("/api/civitai/search/nope/more", headers=LOCAL).status_code == 404

    seen = client.post("/api/civitai/seen", json={"ids": [100], "token": body["token"]}, headers=LOCAL).json()
    assert seen["changed"] == 1
    hidden = client.post("/api/civitai/authors", json={"username": "someone"}, headers=LOCAL).json()
    assert hidden["hidden_authors"] == ["someone"]


def test_a_search_is_saved_as_a_preset_and_goes_with_it(client):
    stub = Civitai({None: page(model(1), cursor="more"), "more": page(model(2))})
    client.search._client = httpx.AsyncClient(transport=httpx.MockTransport(stub))
    token = client.post("/api/civitai/search", json={"filters": RED}, headers=LOCAL).json()["token"]
    preset = client.post("/api/civitai/presets", json={"name": "Krea", "filters": RED}, headers=LOCAL).json()["preset"]

    adopted = client.post(f"/api/civitai/search/{token}/adopt", json={"preset_id": preset["id"]}, headers=LOCAL)
    assert adopted.json()["presets"][0]["unseen"]["count"] == 2
    assert client.post(f"/api/civitai/search/{token}/adopt", json={"preset_id": 999}, headers=LOCAL).status_code == 404

    client.delete(f"/api/civitai/presets/{preset['id']}", headers=LOCAL)
    assert client.post(f"/api/civitai/search/{token}/more", headers=LOCAL).status_code == 404
