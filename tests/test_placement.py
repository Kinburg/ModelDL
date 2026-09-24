"""Asking where a download goes before it starts.

A link is resolved first, and nothing is queued until the page says which of its files go
where. What is tested is the order the folders are offered in — the library's own contents
first — and that the answer is taken exactly as given.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.core.types import FileIdentity
from sfd.jobs import db as states
from sfd.jobs.db import Database
from sfd.jobs.manager import Manager
from sfd.library.categories import Category
from sfd.library.classify import Verdict
from sfd.providers.civitai import CivitaiProvider
from sfd.providers.direct import DirectProvider
from sfd.providers.huggingface import HuggingFaceProvider
from sfd.providers.registry import Item, Resolution
from sfd.settings import Settings
from sfd.web.app import create_app


def put(path: Path, data: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(
        library_root=str(root),
        download_dir=str(tmp_path / "downloads"),
        auto_start=False,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    yield manager, root, database
    database.close()


def described(database: Database, path: Path, **values) -> None:
    """Say what a model on disk is, as the header reading would have."""
    database.update_model(database.model_at(path).id, **values)


def ranked(manager: Manager, category, base_model=None, filename="x.safetensors", **kwargs):
    rows = manager.library.rank_folders(manager.layout(), category, base_model, filename, **kwargs)
    return [(r["root"], r["relative"]) for r in rows], rows


# --- the order the folders are offered in ------------------------------------------


def test_where_the_same_kind_for_the_same_base_already_is_comes_first(setup):
    manager, root, database = setup
    krea = [put(root / "loras" / "mine" / f"{n}.safetensors") for n in ("a", "b")]
    pony = put(root / "loras" / "Pony" / "c.safetensors")
    manager.library.sync()
    for path in krea:
        described(database, path, category="lora", base_model="Krea 2")
    described(database, pony, category="lora", base_model="Pony")

    order, rows = ranked(manager, Category.LORA, "Krea2")

    assert order[0] == (0, "loras/mine"), "not the folder's name: what is in it"
    assert "2 Krea2 LoRAs here" in rows[0]["reason"], "Krea2 and Krea 2 are one base model"
    assert order.index((0, "loras/mine")) < order.index((0, "loras/Pony"))


def test_the_folder_the_version_you_have_is_in_comes_first(setup):
    manager, root, database = setup
    older = put(root / "loras" / "keepers" / "style_v1.safetensors")
    put(root / "loras" / "Krea 2" / "other.safetensors")
    manager.library.sync()
    described(database, older, category="lora", base_model="Krea 2",
              meta={"model_id": 77, "version_id": 1, "model_name": "Style"})

    order, rows = ranked(manager, Category.LORA, "Krea 2", "style_v2.safetensors",
                         meta={"model_id": 77, "version_id": 2})

    assert order[0] == (0, "loras/keepers")
    assert "the version you have is here" in rows[0]["reason"]


def test_where_a_file_was_comes_before_anything(setup):
    manager, root, database = setup
    put(root / "loras" / "Krea 2" / "a.safetensors")
    manager.library.sync()

    order, rows = ranked(manager, Category.LORA, "Krea 2",
                         hint=(root / "loras" / "old place", "where it was"))

    assert order[0] == (0, "loras/old place") and rows[0]["reason"].startswith("where it was")
    assert rows[0]["exists"] is False, "gone with the file, and made again when it lands"


def test_every_folder_of_the_library_is_offered(setup, tmp_path: Path):
    manager, root, database = setup
    shared = tmp_path / "Shared"
    put(shared / "loras" / "Krea2" / "x.safetensors")
    manager.settings.extra_roots = [str(shared)]
    manager.library.sync()
    described(database, shared / "loras" / "Krea2" / "x.safetensors", category="lora", base_model="Krea2")

    order, _rows = ranked(manager, Category.LORA, "Krea 2")

    assert order[0] == (1, "loras/Krea2"), "the only Krea 2 LoRAs are in the other folder"
    assert (0, "loras/Krea 2") in order, "and where the layout would put it is still there"


def test_a_word_of_the_filename_names_a_folder_only_within_its_kind(setup):
    manager, root, database = setup
    put(root / "text_encoders" / "krea2" / "te.safetensors")
    put(root / "sams" / "sam_vit.pth")
    manager.library.sync()

    _order, rows = ranked(manager, Category.LORA, None, "portrait_krea2.safetensors")
    by_place = {r["relative"]: r["reason"] for r in rows}
    assert "named in the file name" not in by_place.get("text_encoders/krea2", "")

    order, _rows = ranked(manager, Category.DETECTION, None, "mystery_sam_model.pt", confidence="low")
    assert order.index((0, "sams")) < 3, "when the classifier is unsure, the folder the name points at"


def test_the_folder_a_model_is_in_is_not_offered_to_move_it_to(setup):
    manager, root, database = setup
    path = put(root / "loras" / "a.safetensors")
    manager.library.sync()
    model = database.model_at(path)

    offered = manager.library.move_targets(model.id, manager.layout())["folders"]

    assert "loras" not in [f["relative"] for f in offered]


# --- resolving, then placing ----------------------------------------------------------


def resolving(monkeypatch, resolution: Resolution, verdict: Verdict | None = None):
    async def expand(text, client, **kwargs):
        return resolution

    async def classify(self, provider, item, client):
        return verdict or Verdict(Category.LORA, "high", "Civitai lists this as LORA", base_model="Krea 2")

    monkeypatch.setattr("sfd.jobs.manager.expand", expand)
    monkeypatch.setattr(Manager, "_classify", classify)


def civitai_item(file_id: int, filename: str, primary: bool = False) -> Item:
    return Item(
        identity=FileIdentity(provider="civitai", ref={"version_id": 5, "file_id": file_id}),
        filename=filename, size=1000, primary=primary,
        meta={"model_name": "Style", "version_name": "v2", "model_id": 3, "version_id": 5},
    )


def hf_item(path: str, repo: str = "org/model", base: str = "") -> Item:
    return Item(
        identity=FileIdentity(provider="huggingface", ref={
            "repo_id": repo, "repo_type": "model", "revision": "main", "path": path}),
        filename=path.rsplit("/", 1)[-1], size=10,
        meta={"repo_id": repo, "path": path},
        relative=path[len(base):] if base else path,
    )


async def test_a_resolved_link_queues_nothing_until_it_is_placed(setup, monkeypatch):
    manager, root, database = setup
    resolving(monkeypatch, Resolution(CivitaiProvider(), [civitai_item(7, "style.safetensors", True)], "Style / v2"))

    answer = await manager.resolve_links("https://civitai.com/models/3")

    assert database.list() == [], "a question, not a queue position"
    (item,) = answer["items"]
    assert item["checked"] and item["category"] == "lora" and item["model_name"] == "Style"
    assert answer["rankings"][item["ranking"]], "and where it could go"

    result = manager.queue_resolved(answer["token"], [0], root / "loras" / "Krea 2")

    (task,) = result["created"]
    assert task.dest == str(root / "loras" / "Krea 2" / "style.safetensors")
    assert task.state == states.PAUSED, "auto-start is off in this library"
    with pytest.raises(LookupError):
        manager.queue_resolved(answer["token"], [0], root / "loras")


async def test_only_the_files_picked_are_queued_and_the_folder_says_what_they_are(setup, monkeypatch):
    manager, root, database = setup
    resolving(monkeypatch, Resolution(CivitaiProvider(), [
        civitai_item(7, "style.safetensors", True),
        civitai_item(8, "style.fp8.safetensors"),
    ], "Style / v2"), Verdict(Category.OTHER, "low", "nothing identified this file"))

    answer = await manager.resolve_links("https://civitai.com/models/3")
    assert [i["checked"] for i in answer["items"]] == [True, False], "the version's own main file"

    (task,) = manager.queue_resolved(answer["token"], [1], root / "loras")["created"]

    assert task.filename == "style.fp8.safetensors"
    assert (task.category, task.confidence) == ("lora", "high"), "put in loras by a person"


async def test_a_repository_can_keep_its_own_folders(setup, monkeypatch):
    manager, root, database = setup
    files = [hf_item(p) for p in ("config.json", "model.safetensors", "tokenizer/tokenizer.json")]
    resolving(monkeypatch, Resolution(HuggingFaceProvider(), files, "org/model", folder_name="model"),
              Verdict(Category.LLM, "high", "transformer decoder layer names"))

    answer = await manager.resolve_links("org/model")
    assert all(i["checked"] for i in answer["items"]) and answer["structure"], \
        "one model with its configs: all of it, as the repository lays it out"

    created = manager.queue_resolved(answer["token"], [0, 1, 2], root / "LLM", keep_structure=True)["created"]

    assert sorted(t.dest for t in created) == sorted(str(p) for p in (
        root / "LLM" / "model" / "config.json",
        root / "LLM" / "model" / "model.safetensors",
        root / "LLM" / "model" / "tokenizer" / "tokenizer.json",
    ))


async def test_a_repository_of_quantisations_ticks_nothing(setup, monkeypatch):
    manager, root, database = setup
    quants = [hf_item(f"m-{q}.gguf", repo="org/m-GGUF") for q in ("Q4_K_M", "Q5_K_M", "Q8_0")]
    resolving(monkeypatch, Resolution(HuggingFaceProvider(), [hf_item("README.md"), *quants], "org/m-GGUF",
                                      folder_name="m-GGUF"))

    answer = await manager.resolve_links("org/m-GGUF")

    assert not any(i["checked"] for i in answer["items"]), "which of them is wanted is the question"


async def test_quantisations_of_one_model_are_read_once(setup, monkeypatch):
    """Twenty headers read to learn one thing kept the question waiting for seconds."""
    manager, root, database = setup
    files = [hf_item(f"m-{q}.gguf", repo="org/m-GGUF") for q in ("Q4_K_M", "Q8_0", "UD-IQ2_XXS", "BF16")]
    files.append(hf_item("mmproj-F16.gguf", repo="org/m-GGUF"))
    resolving(monkeypatch, Resolution(HuggingFaceProvider(), files, "org/m-GGUF"))
    read = []

    async def classify(self, provider, item, client):
        read.append(item.filename)
        return Verdict(Category.LLM, "high", "GGUF general.architecture is qwen3", base_model="qwen3")

    monkeypatch.setattr(Manager, "_classify", classify)

    answer = await manager.resolve_links("org/m-GGUF")

    assert len(read) == 2, read
    assert "mmproj-F16.gguf" in read, "a projector is a different file, and is read for itself"
    assert {i["category"] for i in answer["items"]} == {"llm"}


async def test_one_model_file_is_ticked_without_the_readme(setup, monkeypatch):
    manager, root, database = setup
    resolving(monkeypatch, Resolution(HuggingFaceProvider(), [
        hf_item("README.md"), hf_item(".gitattributes"), hf_item("flux.safetensors"),
    ], "org/model", folder_name="model"))

    answer = await manager.resolve_links("org/model")

    assert [i["checked"] for i in answer["items"]] == [False, False, True]
    assert answer["structure"] is False


async def test_a_file_already_in_the_library_is_not_ticked(setup, monkeypatch):
    manager, root, database = setup
    path = put(root / "loras" / "style.safetensors")
    database.add(state="done", source="s", provider="civitai", filename="style.safetensors",
                 identity={"provider": "civitai", "ref": {"version_id": 5, "file_id": 7}},
                 dest=str(path))
    manager.library.sync()
    resolving(monkeypatch, Resolution(CivitaiProvider(), [civitai_item(7, "style.safetensors", True)], "Style"))

    answer = await manager.resolve_links("https://civitai.com/models/3")

    (item,) = answer["items"]
    assert not item["checked"] and item["have"]["path"] == str(path)


# --- through the page ---------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    root = tmp_path / "models"
    (root / "loras").mkdir(parents=True)
    settings = Settings(
        library_root=str(root),
        download_dir=str(tmp_path / "downloads"),
        auto_start=False,
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database  # type: ignore[attr-defined]
        test.root = root  # type: ignore[attr-defined]
        test.settings = settings  # type: ignore[attr-defined]
        yield test


def direct(monkeypatch, name: str = "model.safetensors") -> None:
    resolving(monkeypatch, Resolution(DirectProvider(), [Item(
        identity=FileIdentity(provider="direct", ref={"url": f"https://example.com/{name}"}),
        filename=name, size=5,
    )], f"https://example.com/{name}"))


def test_the_page_resolves_then_places(client, monkeypatch):
    direct(monkeypatch)
    answer = client.post("/api/resolve", json={"source": "https://example.com/model.safetensors"}).json()
    assert client.database.list() == []

    body = client.post(f"/api/resolve/{answer['token']}/queue",
                       json={"files": [0], "root": 0, "folder": "loras/new place", "remember": True}).json()

    (task,) = body["tasks"]
    assert task["dest"] == str(client.root / "loras" / "new place" / "model.safetensors")
    assert body["remembered"] is None, "a folder that names no kind is not remembered as one"


def test_the_page_cannot_place_a_download_outside_the_library(client, monkeypatch):
    direct(monkeypatch)
    for folder in ("../elsewhere", "C:\\Windows", "/etc", "", "  "):
        token = client.post("/api/resolve", json={"source": "https://example.com/m"}).json()["token"]
        response = client.post(f"/api/resolve/{token}/queue", json={"files": [0], "root": 0, "folder": folder})
        assert response.status_code == 400, folder
    assert client.database.list() == []


def test_an_answer_to_a_link_nobody_resolved_is_refused(client):
    response = client.post("/api/resolve/made-up/queue", json={"files": [0], "root": 0, "folder": "loras"})
    assert response.status_code == 404


def test_smart_placement_is_off_until_turned_on(client):
    assert client.get("/api/settings").json()["settings"]["smart_placement"] is False
    client.put("/api/settings", json={"smart_placement": True})
    assert client.settings.smart_placement is True
