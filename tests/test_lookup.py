"""Finding out what a file is from the services, and where it could be downloaded again.

The Hub has no lookup by hash, so a file is found there by its name — through the model
cards that mention it and the repositories named like it — narrowed by its exact size, and
proven by the SHA256 the Hub publishes for every file. Civitai answers a hash, including
the AutoV1 that takes sixty-four kilobytes to work out, which is what lets a big file be
ruled out without reading it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from sfd.jobs import db as states
from sfd.jobs.db import Database
from sfd.jobs.library import Library
from sfd.library import lookup, sidecar
from sfd.settings import Settings

BIG = 1024 * 1024 + 70_000


def weights(seed: int = 0, size: int = BIG) -> bytes:
    block = hashlib.sha256(f"seed {seed}".encode()).digest() * 2048
    return (block * (size // len(block) + 1))[:size]


def put(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def hub(files: dict[str, dict], cards: list[dict] | None = None, named: list[str] | None = None):
    """A Hub that knows these repositories, and cards and names that lead to them."""
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(f"{request.url.host}{request.url.path}")
        path = request.url.path
        if request.url.host != "huggingface.co":
            return httpx.Response(404, json={"error": "not here"})
        if path == "/api/search/full-text":
            return httpx.Response(200, json={"hits": cards or []})
        if path == "/api/models":
            return httpx.Response(200, json=[{"id": name} for name in named or []])
        repo = path.removeprefix("/api/models/")
        if repo in files:
            return httpx.Response(200, json=files[repo])
        return httpx.Response(404, json={"error": "RepoNotFound"})

    return httpx.AsyncClient(transport=httpx.MockTransport(answer)), asked


def repo(sha: str, size: int, name: str = "split_files/text_encoders/qwen_te.safetensors", downloads: int = 10):
    return {"sha": "c0ffee", "downloads": downloads, "tags": ["license:apache-2.0", "base_model:org/Base-1"],
            "siblings": [{"rfilename": name, "size": size, "lfs": {"sha256": sha, "size": size}}]}


# --- the Hub -------------------------------------------------------------------------------


async def test_a_card_that_links_the_file_leads_to_its_repository():
    data = weights()
    digest = hashlib.sha256(data).hexdigest()
    card = {"name": "someone/quantised", "formatted": {"fileContent": [
        {"text": "made from https://huggingface.co/Comfy-Org/Repack/blob/main/split_files/text_encoders/", "type": "text"},
        {"text": "qwen_te.safetensors", "type": "highlight"}]}}
    client, asked = hub({"Comfy-Org/Repack": repo(digest, len(data))}, cards=[card])

    (found,) = await lookup.find_on_hub(client, "qwen_te.safetensors", len(data), digest)

    assert (found.repo_id, found.path) == ("Comfy-Org/Repack", "split_files/text_encoders/qwen_te.safetensors")
    assert found.page.endswith("/Comfy-Org/Repack/blob/main/split_files/text_encoders/qwen_te.safetensors")
    await client.aclose()


async def test_a_file_of_the_same_size_and_another_hash_is_another_file():
    client, _ = hub({"org/model": repo("ff" * 32, BIG)}, named=["org/model"])

    assert await lookup.find_on_hub(client, "qwen_te.safetensors", BIG, "00" * 32) == []
    assert len(await lookup.find_on_hub(client, "qwen_te.safetensors", BIG)) == 1, \
        "without a hash, the same size is a candidate for the hash to settle"
    await client.aclose()


async def test_the_original_outranks_its_mirrors():
    client, _ = hub({
        "someone/mirror": repo("aa" * 32, BIG, "qwen_te.safetensors", downloads=3),
        "Comfy-Org/Original": repo("aa" * 32, BIG, "qwen_te.safetensors", downloads=90_000),
    }, named=["someone/mirror", "Comfy-Org/Original"])

    found = await lookup.find_on_hub(client, "qwen_te.safetensors", BIG)

    assert found[0].repo_id == "Comfy-Org/Original"
    await client.aclose()


def test_a_civitai_file_is_matched_by_autov1_and_size():
    version = {"files": [
        {"id": 1, "sizeKB": BIG / 1024, "hashes": {"AutoV1": "ABCD1234", "SHA256": "11" * 32}},
    ]}

    assert lookup.civitai_file(version, autov1="abcd1234", size=BIG)["id"] == 1
    assert lookup.civitai_file(version, autov1="abcd1234", size=BIG * 2) is None, "another file of that hash"
    assert lookup.civitai_file(version, sha256="11" * 32)["id"] == 1


def test_what_a_hub_file_says_about_itself():
    hit = lookup.HubFile("org/Model-7B", "model.safetensors", 1, None, "c0ffee", 5, True,
                         {"tags": ["base_model:finetune:x/y", "base_model:org/Base-1", "license:mit"],
                          "pipeline_tag": "text-generation"})

    meta = lookup.hub_meta(hit)

    assert meta["base_model"] == "Base-1" and meta["license"] == "mit"
    assert meta["model_name"] == "Model-7B" and meta["repo_id"] == "org/Model-7B"


# --- identifying -------------------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "d"), fetch_previews=False,
                        _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    library = Library(settings, database, lambda event: None)
    yield library, root, database
    database.close()


async def identify(library: Library, model_id: int) -> None:
    library.start()
    try:
        library.enqueue("identify", [model_id])
        for _ in range(300):
            await asyncio.sleep(0.02)
            if library.current_job is None and not library._jobs:
                break
    finally:
        await library.stop()


async def test_a_file_civitai_does_not_know_is_found_on_the_hub(setup):
    library, root, database = setup
    data = weights()
    path = put(root / "text_encoders" / "qwen_te.safetensors", data)
    library.sync()
    (model,) = database.list_models()
    library._client, asked = hub({"Comfy-Org/Repack": repo(hashlib.sha256(data).hexdigest(), len(data))},
                                 named=["Comfy-Org/Repack"])

    await identify(library, model.id)

    after = database.get_model(model.id)
    assert after.provider == "huggingface" and after.lookup["source"] == "huggingface"
    assert after.identity["ref"]["path"] == "split_files/text_encoders/qwen_te.safetensors"
    assert after.meta["repo_id"] == "Comfy-Org/Repack" and after.identified
    assert "civitai.com/api/v1/model-versions/by-hash/" + hashlib.sha256(data).hexdigest() in asked
    record = sidecar.find_record(path, **library.places())
    assert json.loads(record.read_text("utf-8"))["source"]["repo_id"] == "Comfy-Org/Repack"


async def test_a_big_file_nobody_has_is_ruled_out_without_reading_it(setup, monkeypatch):
    library, root, database = setup
    monkeypatch.setattr("sfd.jobs.library.QUICK_ABOVE", 1024)
    put(root / "vae" / "mine.safetensors", weights())
    library.sync()
    (model,) = database.list_models()
    library._client, asked = hub({})
    monkeypatch.setattr(library, "_hash", lambda path, total: pytest.fail("read the whole file"))

    await identify(library, model.id)

    after = database.get_model(model.id)
    assert after.lookup["result"] == "not_found" and after.lookup["quick"] is True
    assert any("by-hash/" in a for a in asked), "Civitai was asked by the AutoV1"
    assert not any(a.endswith(hashlib.sha256(weights()).hexdigest()) for a in asked)


async def test_a_quick_look_that_finds_a_candidate_is_proven_by_the_hash(setup, monkeypatch):
    library, root, database = setup
    monkeypatch.setattr("sfd.jobs.library.QUICK_ABOVE", 1024)
    data = weights(3)
    put(root / "text_encoders" / "qwen_te.safetensors", data)
    library.sync()
    (model,) = database.list_models()
    library._client, asked = hub({"org/model": repo(hashlib.sha256(data).hexdigest(), len(data), "qwen_te.safetensors")},
                                 named=["org/model"])

    await identify(library, model.id)

    after = database.get_model(model.id)
    assert after.sha256 == hashlib.sha256(data).hexdigest(), "read, to prove it"
    assert after.lookup["result"] == "found" and after.meta["repo_id"] == "org/model"


async def test_a_candidate_the_hash_does_not_prove_is_not_taken(setup, monkeypatch):
    library, root, database = setup
    monkeypatch.setattr("sfd.jobs.library.QUICK_ABOVE", 1024)
    data = weights(4)
    put(root / "text_encoders" / "qwen_te.safetensors", data)
    library.sync()
    (model,) = database.list_models()
    library._client, _ = hub({"org/model": repo("ee" * 32, len(data), "qwen_te.safetensors")}, named=["org/model"])

    await identify(library, model.id)

    after = database.get_model(model.id)
    assert after.lookup["result"] == "not_found" and not after.identified
    assert after.state == states.PRESENT


# --- finding a missing model online ---------------------------------------------------------


def missing(tmp_path: Path, sha256: str | None = "ab" * 32):
    """A manager whose library remembers a model whose file is gone."""
    from sfd.jobs.manager import Manager

    root = tmp_path / "models"
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "d"), fetch_previews=False,
                        auto_start=False, _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    path = put(root / "text_encoders" / "qwen_te.safetensors", weights())
    manager.library.sync()
    model = database.model_at(path)
    # A note is what keeps a model found on disk in the library once its file is gone.
    database.update_model(model.id, sha256=sha256, hash_source="computed" if sha256 else None,
                          note="the good one")
    path.unlink()
    manager.library.sync()
    return manager, database, database.get_model(model.id)


async def test_a_missing_model_is_found_on_civitai_by_the_hash_it_left(tmp_path: Path):
    manager, database, model = missing(tmp_path)
    version = {"id": 5, "modelId": 6, "name": "v1", "model": {"name": "Encoder", "type": "Other"},
               "files": [{"id": 7, "name": "qwen_te.safetensors", "sizeKB": BIG / 1024,
                          "hashes": {"SHA256": "AB" * 32}}], "images": []}

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.host == "civitai.com" and request.url.path.endswith("ab" * 32):
            return httpx.Response(200, json=version)
        if request.url.host == "huggingface.co" and request.url.path == "/api/models":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"hits": []}) if "full-text" in request.url.path else httpx.Response(404)

    manager.library._client = httpx.AsyncClient(transport=httpx.MockTransport(answer))

    answer_ = await manager.find_online(model.id)

    (hit,) = answer_["found"]
    assert hit["source"] == "civitai" and hit["proven"] and "_identity" not in hit, \
        "what would be downloaded stays with the server"
    resolved = manager.resolve_found(answer_["token"], 0)
    first = resolved["rankings"][resolved["items"][0]["ranking"]][0]
    assert first["reason"].startswith("where it was")
    (task,) = manager.queue_resolved(resolved["token"], [0], Path(model.path).parent)["created"]
    assert task.identity == {"provider": "civitai", "ref": {"version_id": 5, "file_id": 7}}
    assert task.model_id == model.id and task.sha256 == "ab" * 32
    await manager.library.stop()
    database.close()


async def test_a_missing_model_is_found_on_the_hub_by_its_name_and_size(tmp_path: Path):
    manager, database, model = missing(tmp_path, sha256=None)
    manager.library._client, _ = hub({"org/model": repo("cd" * 32, BIG, "qwen_te.safetensors")},
                                     named=["org/model"])

    answer = await manager.find_online(model.id)

    hits = [h for h in answer["found"] if h["source"] == "huggingface"]
    assert hits and hits[0]["title"] == "org/model" and hits[0]["proven"] is False
    assert hits[0]["same_name"] is True
    assert "huggingface.co/models?search=qwen_te" in answer["search"]["huggingface"]
    await manager.library.stop()
    database.close()
