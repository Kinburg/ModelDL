"""Newer versions: which of a model's newer versions is an update, and the check that asks.

The cases for the rule are the ones a real library turned up — a collection whose versions
are characters, the same LoRA trained for another base model, variants published side by
side, a newer version already downloaded — next to the updates that are real.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sfd.core.types import FileIdentity
from sfd.jobs import db as states
from sfd.jobs import manager as manager_module
from sfd.jobs.db import Database
from sfd.jobs.library import Busy, Library
from sfd.jobs.manager import Manager
from sfd.library import versions
from sfd.library.categories import Category
from sfd.library.classify import Verdict
from sfd.providers.civitai import CivitaiProvider
from sfd.providers.registry import Item, Resolution
from sfd.settings import Settings
from sfd.web.app import create_app


def v(id: int, name: str, base: str = "Krea 2", files: list | None = None, published: str = "") -> dict:
    return {"id": id, "name": name, "baseModel": base, "files": files or [], "publishedAt": published}


def f(id: int, fp: str = "bf16", *, primary: bool = False, kind: str = "Model", size_kb: float = 1.0,
      sha: str | None = None) -> dict:
    entry = {"id": id, "name": f"file{id}.safetensors", "type": kind, "primary": primary,
             "sizeKB": size_kb, "metadata": {"format": "SafeTensor", "fp": fp, "size": "full"}}
    if sha:
        entry["hashes"] = {"SHA256": sha.upper()}
    return entry


# --- the version number a name carries ----------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    ("v1.0", (1, 0)),
    ("V3", (3,)),
    ("V3.0 Turbo", (3, 0)),
    ("v2 Qwen2.1", (2,)),
    ("v1.0 Qwen2.1 Asians", (1, 0)),
    ("Krea2 v1.0", (1, 0)),
    ("FLUX v1.0b", (1, 0)),
    ("v10 (Krea 2)", (10,)),
    ("version 2", (2,)),
    ("2.0", (2, 0)),
    ("Kay", None),
    ("VAE", None),
    ("4steps bf16 (2511)", None),
    ("TEnc qwen_0.6b_ace15", None),
    ("mast", None),
])
def test_the_number_is_read_only_where_it_is_a_version(name, expected):
    assert versions.number(name) == expected


def test_a_shorter_number_is_padded():
    assert versions.higher((2, 0, 1), (2,))
    assert not versions.higher((2,), (2, 0))
    assert versions.higher((10,), (9, 5))


# --- which newer version is an update ------------------------------------------------------


def test_the_characters_of_a_collection_are_not_updates_of_each_other():
    page = [v(4, "Erica"), v(3, "Kay"), v(2, "Dianda"), v(1, "Penelope")]

    standing = versions.assess(page, mine=1, have={2, 3})

    assert standing.pick is None
    assert [o["name"] for o in standing.others] == ["Erica"], "Kay and Dianda are here already"


def test_a_version_for_another_base_model_is_not_an_update():
    page = [v(3, "v2 Qwen2.1", "Qwen 2.1"), v(2, "v1.0 Qwen2.1", "Qwen 2.1"), v(1, "v1.0 Krea2")]

    standing = versions.assess(page, mine=1, have=set())

    assert standing.pick is None
    assert [o["name"] for o in standing.others] == ["v2 Qwen2.1", "v1.0 Qwen2.1"]


def test_variants_published_side_by_side_are_not_updates():
    ace = [v(3, "v1.5 XL Turbo", "ACE Audio"), v(2, "v1.5 XL SFT", "ACE Audio"), v(1, "v1.5 XL Base", "ACE Audio")]
    lightning = [v(2, "8steps bf16 (2511)", "Qwen"), v(1, "4steps bf16 (2511)", "Qwen")]

    assert versions.assess(ace, mine=1, have=set()).pick is None
    assert versions.assess(lightning, mine=1, have=set()).pick is None


def test_a_version_here_already_raises_the_number_to_beat():
    page = [v(3, "V3"), v(2, "V2"), v(1, "V1")]

    assert versions.assess(page, mine=1, have={3}).pick is None, "V3 is here: V2 is behind it"
    standing = versions.assess(page, mine=1, have={2})
    assert standing.pick["name"] == "V3" and standing.count == 1


def test_the_highest_update_is_the_one_offered():
    page = [v(4, "V3.0 Turbo"), v(3, "V2.0 Base"), v(2, "V2.0 Turbo"), v(1, "v1.0")]

    standing = versions.assess(page, mine=1, have=set())

    assert standing.pick["name"] == "V3.0 Turbo" and standing.count == 3
    assert standing.others == []


def test_an_update_and_a_sibling_on_the_same_page():
    page = [v(3, "v2 Qwen2.1", "Qwen 2.1"), v(2, "v1.0 Qwen2.1 Asians", "Qwen 2.1"),
            v(1, "v1.0 Qwen2.1", "Qwen 2.1")]

    standing = versions.assess(page, mine=1, have=set())

    assert standing.pick["name"] == "v2 Qwen2.1"
    assert [o["name"] for o in standing.others] == ["v1.0 Qwen2.1 Asians"]


def test_a_version_taken_down_is_gone():
    assert versions.assess([v(2, "v2")], mine=1, have=set()).gone == "this version is no longer published"


# --- which file of the new version ---------------------------------------------------------


def test_the_counterpart_of_the_file_here_is_chosen():
    here = f(10, "int8")
    new = [f(20, "bf16", primary=True), f(21, "int8")]

    chosen = versions.choose_file(new, here)

    assert chosen["id"] == 21 and chosen["exact"]


def test_without_a_counterpart_the_versions_main_file_is_chosen():
    new = [f(21, "fp8"), f(20, "bf16", primary=True), f(22, "bf16", kind="VAE")]

    assert versions.choose_file(new, f(10, "int8"))["id"] == 20
    assert versions.choose_file(new, None)["id"] == 20
    assert versions.choose_file(new, None)["exact"] is False


def test_the_file_here_is_found_by_its_id_or_its_hash():
    files = [f(10, sha="ab" * 32), f(11)]

    assert versions.find_file(files, 11)["id"] == 11
    assert versions.find_file(files, 99, "AB" * 32)["id"] == 10
    assert versions.find_file(files, 99) is None


# --- what a version costs ------------------------------------------------------------------------


def test_a_version_sold_for_good_or_in_early_access_says_so():
    assert versions.access({"paidAccess": None}) is None
    assert versions.access({}) is None
    assert versions.access({"paidAccess": {"permanent": True, "endsAt": None}}) == {"permanent": True, "until": None}
    later = {"paidAccess": {"permanent": False, "endsAt": "2999-10-01T18:12:00.000Z"}}
    assert versions.access(later) == {"permanent": False, "until": "2999-10-01T18:12:00.000Z"}


def test_early_access_that_has_ended_is_free():
    assert versions.access({"paidAccess": {"permanent": False, "endsAt": "2001-01-01T00:00:00.000Z"}}) is None


def civitai_provider(handler, token: str | None = "key"):
    from sfd.providers.civitai import CivitaiProvider

    return CivitaiProvider(token, "civitai.red"), httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status, owned", [(307, True), (200, True), (403, False), (401, None), (500, None)])
async def test_whether_a_version_is_bought_is_asked_of_its_download_link(status, owned):
    asked = []

    def answer(request):
        asked.append((request.method, str(request.url), request.headers.get("authorization")))
        return httpx.Response(status, headers={"location": "https://cdn.example/file"} if status == 307 else {})
    provider, client = civitai_provider(answer)

    assert await provider.owned(3, 30, client) is owned
    assert asked == [("HEAD", "https://civitai.red/api/download/models/3?fileId=30", "Bearer key")]
    await client.aclose()


async def test_without_a_key_nobody_is_asked():
    asked = []
    provider, client = civitai_provider(lambda request: asked.append(request) or httpx.Response(307), token=None)

    assert await provider.owned(3, 30, client) is None
    assert asked == []
    await client.aclose()


async def test_a_link_that_takes_no_head_is_asked_with_a_get_that_reads_nothing():
    methods = []

    def answer(request):
        methods.append(request.method)
        return httpx.Response(405) if request.method == "HEAD" else httpx.Response(302, headers={"location": "https://cdn.example/f"})
    provider, client = civitai_provider(answer)

    assert await provider.owned(3, None, client) is True
    assert methods == ["HEAD", "GET"]
    await client.aclose()


# --- skipping ----------------------------------------------------------------------------------


def test_a_skipped_version_is_counted_again_only_for_a_newer_one():
    record = {"status": "update", "update": {"id": 3, "name": "V3", "number": [3]}}

    skipped = versions.skip(record)

    assert skipped["status"] == versions.OTHER
    assert versions.status({**skipped, "update": {"id": 3, "name": "V3", "number": [3]}}) == versions.OTHER
    assert versions.status({**skipped, "update": {"id": 4, "name": "V4", "number": [4]}}) == versions.UPDATE
    assert versions.unskip(skipped)["status"] == versions.UPDATE


def test_skipping_a_file_on_the_hub_skips_that_content():
    record = {"status": "update", "update": {"sha256": "aa" * 32, "name": "a newer commit on main"}}

    skipped = versions.skip(record)

    assert skipped["status"] == versions.OTHER
    assert versions.status({**skipped, "update": {"sha256": "bb" * 32}}) == versions.UPDATE


def test_a_record_from_before_the_rule_counts_as_not_checked():
    assert versions.normalise({"available": True, "version_name": "v2", "checked_at": 1.0}) is None
    assert versions.normalise({}) is None


# --- the check -----------------------------------------------------------------------------------


def put(path: Path, data: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(
        library_root=str(root), download_dir=str(tmp_path / "downloads"),
        auto_start=False, fetch_previews=False, write_sidecars=False,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    events: list[dict] = []
    library = Library(settings, database, events.append)
    yield library, root, database, events
    database.close()


def civitai_model(database: Database, path: Path, *, model_id=1, version_id=1, file_id=10,
                  name="Style", version_name="v1", base="Krea 2", sha="ab" * 32, **meta):
    put(path)
    task = database.add(
        state="done", source=f"https://civitai.com/models/{model_id}", provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": version_id, "file_id": file_id}},
        filename=path.name, dest=str(path), size=7, sha256=sha, category="lora",
        confidence="high", reason="Civitai lists this as LORA", base_model=base,
        meta={"model_name": name, "version_name": version_name, "model_id": model_id,
              "version_id": version_id, "base_model": base, "host": "civitai.com", **meta},
    )
    return task


def civitai(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def page(model_id: int, *listed: dict) -> dict:
    return {"id": model_id, "modelVersions": list(listed)}


async def test_a_page_is_asked_for_once_however_many_files_of_it_are_here(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "noct_bf16.safetensors", version_id=1, file_id=10, sha="aa" * 32)
    civitai_model(database, root / "loras" / "noct_int8.safetensors", version_id=1, file_id=11, sha="bb" * 32)
    library.sync()
    asked = []

    def answer(request):
        asked.append(request.url.path)
        return httpx.Response(200, json=page(1,
            v(3, "V3.0 Turbo", files=[f(30, "bf16", primary=True), f(31, "int8")], published="2026-09-25"),
            v(1, "v1.0", files=[f(10, "bf16", primary=True), f(11, "int8")]),
        ))
    library._client = civitai(answer)

    result = await library.check_updates(library.checkable_ids())

    assert asked == ["/api/v1/models/1"]
    assert result == {"checked": 2, "gone": 0, "failed": 0, "updates": 1, "new": 1, "stopped": False}
    picks = {m.filename: m.updates["update"]["file"]["id"] for m in database.list_models()}
    assert picks == {"noct_bf16.safetensors": 30, "noct_int8.safetensors": 31}, "bf16 for bf16, int8 for int8"
    await library.stop()


async def test_the_check_says_how_far_it_has_got_and_when_it_is_done(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    library._client = civitai(lambda request: httpx.Response(200, json=page(1, v(1, "v1"))))

    await library.check_updates(library.checkable_ids(), startup=True)

    checks = [e for e in events if e["type"] == "update_check"]
    assert checks[0]["running"] and checks[0]["total"] == 1 and checks[0]["done"] == 0
    assert checks[-1] == {"type": "update_check", "running": False, "startup": True, "result": {
        "checked": 1, "gone": 0, "failed": 0, "updates": 0, "new": 0, "stopped": False}}
    assert database.list_models()[0].updates["status"] == versions.CURRENT
    assert library.update_state() is None
    await library.stop()


async def test_a_model_taken_down_is_gone(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    library._client = civitai(lambda request: httpx.Response(404, json={"error": "gone"}))

    result = await library.check_updates(library.checkable_ids())

    record = database.list_models()[0].updates
    assert record["status"] == versions.GONE and record["error"] == "the model is no longer on Civitai"
    assert result["gone"] == 1
    await library.stop()


async def test_an_update_already_known_is_not_new(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    library._client = civitai(lambda request: httpx.Response(200, json=page(1, v(2, "v2", files=[f(20)]), v(1, "v1"))))

    first = await library.check_updates(library.checkable_ids())
    second = await library.check_updates(library.checkable_ids())

    assert (first["updates"], first["new"]) == (1, 1)
    assert (second["updates"], second["new"]) == (1, 0), "the one at start says nothing about it again"
    await library.stop()


async def test_a_check_that_cannot_be_made_keeps_what_the_last_one_found(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    library._client = civitai(lambda request: httpx.Response(200, json=page(1, v(2, "v2", files=[f(20)]), v(1, "v1"))))
    await library.check_updates(library.checkable_ids())

    def offline(request):
        raise httpx.ConnectError("no network")
    library._client = civitai(offline)
    result = await library.check_updates(library.checkable_ids())

    record = database.list_models()[0].updates
    assert result["failed"] == 1 and result["updates"] == 1
    assert record["status"] == versions.UPDATE and record["update"]["name"] == "v2"
    assert "no network" in record["failed"]["error"]
    await library.stop()


async def test_a_skipped_version_stays_skipped_until_a_newer_one(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    (model,) = database.list_models()
    listed = [v(2, "v2", files=[f(20)]), v(1, "v1")]
    library._client = civitai(lambda request: httpx.Response(200, json=page(1, *listed)))
    await library.check_updates([model.id])

    assert library.skip_updates([model.id]) == 1
    await library.check_updates([model.id])
    assert database.get_model(model.id).updates["status"] == versions.OTHER

    listed.insert(0, v(3, "v3", files=[f(30)]))
    result = await library.check_updates([model.id])
    record = database.get_model(model.id).updates
    assert record["status"] == versions.UPDATE and record["update"]["name"] == "v3"
    assert result["new"] == 1
    await library.stop()


async def test_only_one_check_runs_at_a_time(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    release = asyncio.Event()

    async def slow(request):
        await release.wait()
        return httpx.Response(200, json=page(1, v(1, "v1")))
    library._client = civitai(slow)

    running = asyncio.create_task(library.check_updates(library.checkable_ids()))
    await asyncio.sleep(0.05)
    with pytest.raises(Busy):
        await library.check_updates(library.checkable_ids())
    assert library.update_state()["running"]
    release.set()
    await running
    await library.stop()


async def test_a_stopped_check_asks_nothing_more(setup):
    library, root, database, events = setup
    for n in range(8):
        civitai_model(database, root / "loras" / f"m{n}.safetensors", model_id=n + 1, version_id=100 + n,
                      file_id=200 + n, sha=f"{n:02x}" * 32)
    library.sync()
    asked = []

    async def answer(request):
        asked.append(request.url.path)
        library.stop_update_check()
        model_id = int(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json=page(model_id, v(100 + model_id - 1, "v1")))
    library._client = civitai(answer)

    result = await library.check_updates(library.checkable_ids())

    assert result["stopped"] and len(asked) < 8, "only what was already being asked"
    assert result["checked"] == len(asked)
    await library.stop()


async def test_a_paid_update_is_counted_and_says_whether_it_is_bought(setup):
    library, root, database, events = setup
    library.settings.civitai_token = "key"
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    asked = []

    def answer(request):
        asked.append((request.method, request.url.path, request.url.query.decode()))
        if request.method == "HEAD":
            return httpx.Response(403)
        return httpx.Response(200, json=page(1,
            v(3, "v3", files=[f(30, primary=True)]) | {"paidAccess": {"permanent": True, "endsAt": None}},
            v(2, "Erica") | {"paidAccess": {"permanent": False, "endsAt": "2999-01-01T00:00:00Z"}},
            v(1, "v1"),
        ))
    library._client = civitai(answer)

    result = await library.check_updates(library.checkable_ids())

    record = database.list_models()[0].updates
    assert result["updates"] == 1 and record["status"] == versions.UPDATE, "counted: buying it is a choice"
    assert record["update"]["access"] == {"permanent": True, "until": None, "owned": False}
    assert record["others"][0]["access"] == {"permanent": False, "until": "2999-01-01T00:00:00Z"}
    assert ("HEAD", "/api/download/models/3", "fileId=30") in asked
    assert len([a for a in asked if a[0] == "HEAD"]) == 1, "one question, for the one version that is sold"
    await library.stop()


async def test_a_free_update_asks_nothing_about_buying(setup):
    library, root, database, events = setup
    library.settings.civitai_token = "key"
    civitai_model(database, root / "loras" / "a.safetensors")
    library.sync()
    methods = []

    def answer(request):
        methods.append(request.method)
        return httpx.Response(200, json=page(1, v(2, "v2", files=[f(20)]), v(1, "v1")))
    library._client = civitai(answer)

    await library.check_updates(library.checkable_ids())

    assert methods == ["GET"]
    assert database.list_models()[0].updates["update"]["access"] is None
    await library.stop()


def hub_model(database: Database, path: Path, sha: str = "aa" * 32):
    put(path)
    database.add(
        state="done", source="https://huggingface.co/org/repo", provider="huggingface",
        identity={"provider": "huggingface", "ref": {
            "repo_id": "org/repo", "repo_type": "model", "revision": "main", "path": path.name}},
        filename=path.name, dest=str(path), size=7, sha256=sha, category="vae",
        confidence="high", reason="", meta={"repo_id": "org/repo", "path": path.name},
    )


async def test_a_file_on_the_hub_that_changed_is_an_update(setup):
    library, root, database, events = setup
    hub_model(database, root / "vae" / "ae.safetensors")
    library.sync()
    headers = {"x-linked-etag": f'"{"bb" * 32}"', "x-linked-size": "7", "x-repo-commit": "c0ffee",
               "content-type": "application/octet-stream", "content-length": "7", "accept-ranges": "bytes"}
    library._hub_http = lambda: httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers=headers)))

    await library.check_updates(library.checkable_ids())

    record = database.list_models()[0].updates
    assert record["status"] == versions.UPDATE
    assert record["update"] == {"sha256": "bb" * 32, "commit": "c0ffee", "name": "a newer commit on main",
                                "page": "https://huggingface.co/org/repo/blob/main/ae.safetensors"}
    await library.stop()


async def test_a_file_taken_out_of_its_repository_is_gone(setup):
    library, root, database, events = setup
    hub_model(database, root / "vae" / "ae.safetensors")
    library.sync()
    library._hub_http = lambda: httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(404, headers={"x-error-code": "EntryNotFound"})))

    result = await library.check_updates(library.checkable_ids())

    record = database.list_models()[0].updates
    assert record["status"] == versions.GONE and "no longer in its repository" in record["error"]
    assert result["gone"] == 1
    await library.stop()


async def test_the_page_is_told_what_the_last_check_said(setup):
    library, root, database, events = setup
    civitai_model(database, root / "loras" / "a.safetensors", version_id=7)
    library.sync()
    library._client = civitai(lambda request: httpx.Response(200, json=page(1, v(8, "v2", files=[f(20)]), v(7, "v1"))))
    await library.check_updates(library.checkable_ids())

    (summary,) = library.snapshot()["models"]

    assert summary["checkable"] and summary["version_id"] == 7
    assert summary["update"]["status"] == "update" and summary["update"]["group"] == "civitai:1:8"
    await library.stop()


# --- asking on start -------------------------------------------------------------------------


async def test_the_check_on_start_waits_for_the_window_and_runs_once(tmp_path: Path, monkeypatch):
    settings = Settings(library_root=str(tmp_path / "models"), _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    asked = []

    async def check(ids, *, startup=False):
        asked.append(startup)
        return {}
    monkeypatch.setattr(manager.library, "check_updates", check)
    monkeypatch.setattr(manager_module, "START_CHECK_DELAY", 0)

    manager.subscribe()
    await asyncio.sleep(0.05)
    assert asked == [], "the queue has not started: nothing has read the disk yet"

    manager._first_walk = asyncio.create_task(asyncio.sleep(0))
    manager.subscribe()
    manager.subscribe()
    await asyncio.sleep(0.05)
    assert asked == [True], "once a run of the app, however many pages connect"
    database.close()


async def test_the_check_on_start_can_be_turned_off(tmp_path: Path, monkeypatch):
    settings = Settings(library_root=str(tmp_path / "models"), check_updates_on_start=False,
                        _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    asked = []

    async def check(ids, *, startup=False):
        asked.append(startup)
    monkeypatch.setattr(manager.library, "check_updates", check)
    monkeypatch.setattr(manager_module, "START_CHECK_DELAY", 0)
    manager._first_walk = asyncio.create_task(asyncio.sleep(0))

    manager.subscribe()
    await asyncio.sleep(0.05)

    assert asked == []
    database.close()


# --- downloading an update -------------------------------------------------------------------------


@pytest.fixture
def placed(tmp_path: Path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(
        library_root=str(root), download_dir=str(tmp_path / "downloads"), auto_start=False,
        fetch_previews=False, write_sidecars=False, _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    expanded = []

    def item(file_id: int, filename: str, primary: bool = False) -> Item:
        return Item(
            identity=FileIdentity(provider="civitai", ref={"version_id": 3, "file_id": file_id}),
            filename=filename, size=1000, sha256=f"{file_id:02x}" * 32, primary=primary,
            meta={"model_name": "Style", "version_name": "V3", "model_id": 1, "version_id": 3},
        )

    files = {30: item(30, "style.safetensors", True), 31: item(31, "style.int8.safetensors")}

    async def expand(text, client, **kwargs):
        expanded.append(text)
        if "fileId=" in text:
            return Resolution(CivitaiProvider(), [files[int(text.rsplit("=", 1)[-1])]], "Style / V3")
        return Resolution(CivitaiProvider(), list(files.values()), "Style / V3")

    async def classify(self, provider, item, client):
        return Verdict(Category.LORA, "high", "Civitai lists this as LORA", base_model="Krea 2")

    monkeypatch.setattr("sfd.jobs.manager.expand", expand)
    monkeypatch.setattr(Manager, "_classify", classify)
    yield manager, root, database, expanded
    database.close()


def flagged(database: Database, path: Path, file_id: int, pick_file: int):
    """A model the last check found an update for: V3, and which file of it."""
    civitai_model(database, path, version_id=1, file_id=file_id, sha=f"{file_id:02x}" * 32)
    return path, pick_file


def mark_updates(manager: Manager, database: Database, picks: dict[Path, int]):
    manager.library.sync()
    for path, pick_file in picks.items():
        model = database.model_at(path)
        record = {"status": "update", "group": "civitai:1:3", "family": "civitai:1",
                  "update": {"id": 3, "name": "V3", "number": [3], "file": {"id": pick_file}}}
        database.update_model(model.id, updates=record)
    return [database.model_at(p).id for p in picks]


async def test_download_all_puts_each_new_version_beside_the_one_it_updates(placed):
    manager, root, database, expanded = placed
    bf16, _ = flagged(database, root / "loras" / "krea" / "style_v1.safetensors", 10, 30)
    int8, _ = flagged(database, root / "loras" / "int8" / "style_v1.int8.safetensors", 11, 31)
    ids = mark_updates(manager, database, {bf16: 30, int8: 31})

    result = await manager.download_updates(ids)

    dests = sorted(t.dest for t in result["created"])
    assert dests == sorted([str(root / "loras" / "krea" / "style.safetensors"),
                            str(root / "loras" / "int8" / "style.int8.safetensors")])
    assert result["failed"] == []
    assert all("fileId=" in text for text in expanded), "each file named exactly, not the version"


async def test_one_file_is_fetched_once_for_every_file_it_updates(placed):
    manager, root, database, expanded = placed
    one, _ = flagged(database, root / "loras" / "a.safetensors", 10, 30)
    two, _ = flagged(database, root / "loras" / "b.safetensors", 11, 30)
    ids = mark_updates(manager, database, {one: 30, two: 30})

    result = await manager.download_updates(ids)

    assert len(result["created"]) == 1 and len(expanded) == 1


async def test_a_new_version_named_like_the_old_one_does_not_take_its_place(placed):
    manager, root, database, expanded = placed
    old, _ = flagged(database, root / "loras" / "style.safetensors", 10, 30)
    ids = mark_updates(manager, database, {old: 30})

    (task,) = (await manager.download_updates(ids))["created"]

    assert task.dest == str(root / "loras" / "style.v3.safetensors")
    assert old.read_bytes() == b"weights"


async def test_the_add_dialog_ticks_the_counterparts_and_offers_the_old_folder_first(placed):
    manager, root, database, expanded = placed
    old, _ = flagged(database, root / "loras" / "krea" / "style_v1.int8.safetensors", 11, 31)
    ids = mark_updates(manager, database, {old: 31})

    answer = await manager.resolve_update(ids)

    assert expanded == ["https://civitai.com/models/1?modelVersionId=3"]
    assert [i["checked"] for i in answer["items"]] == [False, True], "the int8 file, not the main one"
    ranking = answer["rankings"][answer["items"][1]["ranking"]]
    assert (ranking[0]["root"], ranking[0]["relative"]) == (0, "loras/krea")


async def test_the_add_dialog_says_a_version_is_sold_and_whether_it_is_bought(placed, monkeypatch):
    manager, root, database, expanded = placed
    old, _ = flagged(database, root / "loras" / "style_v1.safetensors", 10, 30)
    ids = mark_updates(manager, database, {old: 30})
    asked = []

    async def expand(text, client, **kwargs):
        item = Item(
            identity=FileIdentity(provider="civitai", ref={"version_id": 3, "file_id": 30}),
            filename="style.safetensors", size=1000, primary=True,
            meta={"model_name": "Style", "version_name": "V3", "access": {"permanent": True, "until": None}},
        )
        return Resolution(CivitaiProvider("key"), [item], "Style / V3")

    async def owned(self, version_id, file_id, client):
        asked.append((version_id, file_id))
        return False
    monkeypatch.setattr("sfd.jobs.manager.expand", expand)
    monkeypatch.setattr(CivitaiProvider, "owned", owned)

    answer = await manager.resolve_update(ids)

    (item,) = answer["items"]
    assert item["access"] == {"permanent": True, "until": None, "owned": False}
    assert asked == [(3, 30)]


async def test_a_link_with_nothing_for_sale_asks_nothing_about_buying(placed, monkeypatch):
    manager, root, database, expanded = placed
    asked = []

    async def owned(self, version_id, file_id, client):
        asked.append(version_id)
        return True
    monkeypatch.setattr(CivitaiProvider, "owned", owned)

    answer = await manager.resolve_links("https://civitai.com/models/1?modelVersionId=3")

    assert asked == [] and all(i["access"] is None for i in answer["items"])


def test_a_file_already_at_the_destination_that_is_this_one_keeps_the_name(placed):
    manager, root, database, expanded = placed
    path = put(root / "vae" / "shared_vae.safetensors", b"x" * 1000)
    item = Item(identity=FileIdentity(provider="civitai", ref={"version_id": 3, "file_id": 40}),
                filename=path.name, size=1000, meta={"version_name": "V3"})

    assert manager._unclaimed(path, item) == path, "the transfer compares the hashes"
    other_size = Item(identity=item.identity, filename=path.name, size=5, meta={"version_name": "V3"})
    assert manager._unclaimed(path, other_size) == path.with_name("shared_vae.v3.safetensors")


def test_a_name_another_download_is_landing_on_is_not_taken(placed):
    manager, root, database, expanded = placed
    target = root / "loras" / "style.safetensors"
    database.add(state=states.PAUSED, source="x", provider="civitai",
                 identity={"provider": "civitai", "ref": {"version_id": 2, "file_id": 20}},
                 filename=target.name, dest=str(target), size=1000)
    item = Item(identity=FileIdentity(provider="civitai", ref={"version_id": 3, "file_id": 30}),
                filename=target.name, size=1000, meta={"version_name": "V3.0 Turbo"})

    assert manager._unclaimed(target, item) == target.with_name("style.v3.0-turbo.safetensors")


# --- through the page ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "downloads"),
                        check_updates_on_start=False, _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database  # type: ignore[attr-defined]
        test.settings = settings  # type: ignore[attr-defined]
        yield test


def test_the_setting_is_saved_from_the_page(client):
    body = client.put("/api/settings", json={"check_updates_on_start": True},
                      headers={"Origin": "http://127.0.0.1:7788"}).json()

    assert body["settings"]["check_updates_on_start"] is True
    assert client.settings.check_updates_on_start is True


def test_checking_a_library_with_nothing_to_ask_about(client):
    body = client.post("/api/updates/check", headers={"Origin": "http://127.0.0.1:7788"}).json()

    assert body == {"checked": 0, "gone": 0, "failed": 0, "updates": 0, "new": 0, "stopped": False}
    assert client.post("/api/updates/stop", headers={"Origin": "http://127.0.0.1:7788"}).json() == {"stopping": False}
