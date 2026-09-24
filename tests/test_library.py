"""The library: every model on disk, whether or not it was downloaded here.

What is tested is what the queue could never express: a model that went missing, a model
that was moved behind the app's back and has to be recognised where it landed, and a model
that arrived from somewhere else and has to be described from the file alone.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sfd.jobs import db as states
from sfd.jobs.db import Database
from sfd.jobs.library import Library
from sfd.library import details, relocate, scan, sidecar
from sfd.settings import Settings
from sfd.web.app import create_app


# --- builders -----------------------------------------------------------------


def safetensors(tensors: dict[str, tuple[str, list[int]]], metadata: dict | None = None) -> bytes:
    payload: dict = {
        name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    if metadata:
        payload["__metadata__"] = metadata
    blob = json.dumps(payload).encode()
    return struct.pack("<Q", len(blob)) + blob + b"\0" * 64


def gguf(kvs: dict[str, object]) -> bytes:
    def text(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 7) + struct.pack("<Q", len(kvs))
    for key, value in kvs.items():
        out += text(key)
        if isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)
        else:
            out += struct.pack("<I", 8) + text(str(value))
    return out + b"\0" * 64


LORA = safetensors(
    {"lora_unet_down.lora_down.weight": ("BF16", [32, 320]),
     "lora_unet_down.lora_up.weight": ("BF16", [320, 32])},
    {
        "ss_network_module": "networks.lora",
        "ss_network_dim": "32",
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_tag_frequency": json.dumps({"10_k2style": {"k2style": 40, "portrait": 12}}),
    },
)


def put(path: Path, data: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def setup(tmp_path: Path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    settings = Settings(
        library_root=str(library_root),
        download_dir=str(tmp_path / "downloads"),
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    events: list[dict] = []
    library = Library(settings, database, events.append)
    yield library, library_root, database, events
    database.close()


def done_task(database: Database, dest: Path, **overrides):
    values = {
        "state": "done",
        "source": "https://civitai.com/models/1",
        "provider": "civitai",
        "identity": {"provider": "civitai", "ref": {"version_id": 1, "file_id": 2}},
        "filename": dest.name,
        "dest": str(dest),
        "size": 7,
        "sha256": "ab" * 32,
        "category": "lora",
        "confidence": "high",
        "reason": "Civitai lists this as LORA",
        "meta": {"model_name": "Style", "version_name": "v1", "model_id": 1, "version_id": 1,
                 "trained_words": ["k2style"]},
    }
    values.update(overrides)
    return database.add(**values)


# --- the walk -------------------------------------------------------------------


def test_the_walk_finds_models_and_leaves_the_rest(tmp_path: Path):
    root = tmp_path / "models"
    put(root / "loras" / "a.safetensors")
    put(root / "loras" / ".cache" / "hidden.safetensors")
    put(root / "LLM" / "llama.cpp" / "models" / "ggml-vocab.gguf")
    put(root / "loras" / "a.preview.png")
    put(root / "text_encoders" / "tokenizer.bin", b"tiny")
    put(root / "loras" / "b.safetensors.part", b"half")
    (root / "hypernetworks").mkdir()

    found = scan.walk([root], excluded=[root / "LLM" / "llama.cpp"])

    names = sorted(entry.path.name for entry in found.files.values())
    assert names == ["a.safetensors"], "hidden, excluded and tiny .bin files are not models"
    assert [f.name for f in found.fragments] == ["b.safetensors.part"]
    folders = {(root_index, relative) for root_index, relative, _ in found.folders}
    assert (0, "loras") in folders
    assert (0, "hypernetworks") in folders, "a kind's folder is listed while still empty"
    assert not any(relative.startswith("LLM/llama.cpp") for _, relative, _ in found.folders)


def test_a_split_model_is_one_model(tmp_path: Path):
    root = tmp_path / "models"
    put(root / "LLM" / "big" / "model-00001-of-00002.safetensors", b"a" * 10)
    put(root / "LLM" / "big" / "model-00002-of-00002.safetensors", b"b" * 5)

    found = scan.walk([root])

    (entry,) = found.files.values()
    assert entry.path.name == "model-00001-of-00002.safetensors"
    assert entry.size == 15
    assert len(entry.parts) == 2


def test_a_folder_listed_twice_is_walked_once(tmp_path: Path):
    root = tmp_path / "models"
    put(root / "loras" / "a.safetensors")

    found = scan.walk([root, root / "loras"])

    assert len(found.files) == 1
    assert list(found.files.values())[0].root == 1, "the inner folder is walked as itself"


# --- keeping up with the disk ------------------------------------------------------


def test_a_first_walk_meets_every_model_and_every_download(setup):
    library, root, database, events = setup
    downloaded = put(root / "loras" / "style.safetensors")
    task = done_task(database, downloaded)
    put(root / "loras" / "foreign.safetensors", LORA)
    gone = root / "loras" / "gone.safetensors"
    lost = done_task(database, gone, identity={"provider": "civitai", "ref": {"version_id": 3, "file_id": 4}})

    library.sync()

    models = {m.filename: m for m in database.list_models()}
    assert models["style.safetensors"].origin == states.DOWNLOADED
    assert models["style.safetensors"].state == states.PRESENT
    assert models["style.safetensors"].sha256 == "ab" * 32
    assert models["foreign.safetensors"].origin == states.FOUND
    assert models["gone.safetensors"].state == states.MISSING, "a download whose file is gone"
    assert database.get(task.id).model_id == models["style.safetensors"].id
    assert database.get(lost.id).model_id == models["gone.safetensors"].id


def test_a_model_moved_in_explorer_is_recognised_where_it_landed(setup, tmp_path):
    """Same name, same size, one of each: the same file dragged somewhere else."""
    library, root, database, events = setup
    library.settings.sidecar_dir = str(tmp_path / "records")
    old = put(root / "loras" / "style.safetensors")
    task = done_task(database, old)
    library.sync()
    record = sidecar.write(
        old,
        details.classify(old.name),
        sidecar.Record(filename=old.name, note="the good one"),
        compat=False, triggers=False, **library.places(),
    )
    put(root / "loras" / "style.preview.png", b"\x89PNG")
    library.sync()
    (model,) = database.list_models()
    assert model.note == "the good one"

    new = root / "loras" / "Krea 2" / "style.safetensors"
    new.parent.mkdir()
    old.replace(new)
    library.sync()

    (model,) = database.list_models()
    assert model.state == states.PRESENT and model.path == str(new)
    assert model.note == "the good one", "the note came with it"
    assert not record.exists(), "the collected record followed into the new mirror"
    assert sidecar.find_record(new, **library.places()) is not None
    # The picture stayed where the model was: offered, not moved behind anyone's back.
    assert [Path(p["from"]).name for p in model.left_behind] == ["style.preview.png"]
    assert database.get(task.id).dest == str(new), "and the history says where it is now"
    assert any(e["type"] == "toast" for e in events)

    library.bring_back(model.id)
    assert (new.parent / "style.preview.png").is_file()
    assert database.get_model(model.id).left_behind == []


def test_two_candidates_are_a_question_not_a_guess(setup):
    library, root, database, events = setup
    old = put(root / "loras" / "style.safetensors")
    done_task(database, old)
    library.sync()
    old.unlink()
    put(root / "loras" / "a" / "style.safetensors")
    put(root / "loras" / "b" / "style.safetensors")

    library.sync()

    lost = next(m for m in database.list_models() if m.state == states.MISSING)
    assert len(library.candidates(lost)) == 2

    chosen = library.candidates(lost)[0]
    library.relink(lost.id, chosen.id)
    relinked = database.get_model(lost.id)
    assert relinked.state == states.PRESENT and relinked.path == chosen.path
    assert database.get_model(chosen.id) is None, "the two rows became one"


def test_a_found_model_that_disappears_goes_quietly(setup):
    """Nothing of anyone's was attached to it: keeping it, grey, forever, is rubbish."""
    library, root, database, events = setup
    path = put(root / "loras" / "passing.safetensors")
    library.sync()
    path.unlink()

    library.sync()

    assert database.list_models() == []


def test_a_found_model_with_a_note_is_kept_when_it_disappears(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "kept.safetensors")
    library.sync()
    (model,) = database.list_models()
    library.set_note(model.id, "weight 0.7")
    path.unlink()

    library.sync()

    (model,) = database.list_models()
    assert model.state == states.MISSING
    assert model.note == "weight 0.7"


def test_a_note_on_a_foreign_model_writes_it_a_record(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "foreign.safetensors", LORA)
    library.sync()
    (model,) = database.list_models()

    note, created = library.set_note(model.id, "fights the detailer")

    assert created and note == "fights the detailer"
    stored = json.loads(path.with_name(path.name + ".json").read_text("utf-8"))
    assert stored["note"] == "fights the detailer"
    assert stored["filename"] == "foreign.safetensors"


def test_a_note_edited_on_disk_is_read_back(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "foreign.safetensors")
    library.sync()
    (model,) = database.list_models()
    library.set_note(model.id, "first")
    record = path.with_name(path.name + ".json")
    data = json.loads(record.read_text("utf-8"))
    data["note"] = "edited by hand"
    record.write_text(json.dumps(data), encoding="utf-8")

    library.sync()

    assert database.get_model(model.id).note == "edited by hand"


def test_a_folder_taken_off_the_list_takes_its_found_models_with_it(setup, tmp_path):
    library, root, database, events = setup
    other = tmp_path / "shared"
    put(other / "loras" / "x.safetensors")
    library.settings.extra_roots = [str(other)]
    library.sync()
    assert len(database.list_models()) == 1

    library.settings.extra_roots = []
    library.sync()

    assert database.list_models() == []


def test_a_model_is_placed_under_the_folder_it_is_in(setup, tmp_path):
    library, root, database, events = setup
    other = tmp_path / "shared"
    put(other / "loras" / "Pony" / "x.safetensors")
    library.settings.extra_roots = [str(other)]
    library.sync()

    (model,) = database.list_models()
    summary = library.summary(model)

    assert summary["root"] == 1 and summary["relative"] == "loras/Pony"


# --- reading the files --------------------------------------------------------------


def test_a_lora_describes_itself(tmp_path: Path):
    path = put(tmp_path / "loras" / "style.safetensors", LORA)

    summary, sniffed = details.inspect(path)

    assert summary["format"] == "safetensors"
    assert summary["precision"] == "bf16"
    assert summary["params"] == 32 * 320 * 2
    assert summary["kohya"]["dim"] == "32"
    assert summary["tags"][0] == ["k2style", 40], "the most frequent training tag first"
    assert sniffed.metadata["ss_network_module"] == "networks.lora"


def test_a_gguf_says_its_architecture_and_quantisation(tmp_path: Path):
    path = put(tmp_path / "LLM" / "q.gguf", gguf({
        "general.architecture": "qwen3", "general.name": "Qwen3 8B", "general.file_type": 15,
        "qwen3.context_length": 32768, "tokenizer.ggml.model": "gpt2",
    }))

    summary, sniffed = details.inspect(path)

    assert summary["architecture"] == "qwen3"
    assert summary["quant"] == "Q4_K_M"
    assert summary["context"] == 32768
    assert summary["title"] == "Qwen3 8B"


def test_a_file_nobody_can_read_is_placed_by_its_folder(setup):
    library, root, database, events = setup
    put(root / "upscale_models" / "x4.pth", b"\x80\x02pickle")
    library.sync()
    library.inspect_pending()

    (model,) = database.list_models()
    assert model.category == "upscaler"
    assert model.confidence == "medium"
    assert "upscale_models" in model.reason
    assert model.header["format"] == "pickle"


def test_the_header_is_believed_over_the_folder_and_the_difference_is_noted(setup):
    library, root, database, events = setup
    put(root / "checkpoints" / "actually_a_lora.safetensors", LORA)
    library.sync()
    library.inspect_pending()

    (model,) = database.list_models()
    assert model.category == "lora"
    assert model.header.get("folder_says") == "checkpoint"


def test_the_base_model_folder_names_the_base_model(setup):
    """The layout files by base model; a folder named that way is the best answer there is
    for a file the service never described — better than the trainer's `sdxl_base_v1-0`."""
    library, root, database, events = setup
    put(root / "loras" / "Pony" / "style.safetensors", LORA)
    library.sync()
    library.inspect_pending()

    (model,) = database.list_models()
    assert model.base_model == "Pony"


def test_what_other_tools_left_beside_it_is_read(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors", LORA)
    path.with_name("style.civitai.info").write_text(json.dumps({
        "id": 11, "modelId": 22, "name": "v2", "baseModel": "Illustrious",
        "trainedWords": ["glossy"],
        "model": {"name": "Glossy Style", "type": "LORA", "nsfw": False},
        "files": [{"id": 33, "name": "style.safetensors", "sizeKB": 0.01, "primary": True,
                   "hashes": {"SHA256": "CD" * 32}}],
        "images": [{"url": "https://image.civitai.com/x/y/width=450/1.jpeg", "type": "image",
                    "nsfwLevel": 1, "meta": {"prompt": "a glossy cat"}}],
    }), encoding="utf-8")
    path.with_name("style.json").write_text(json.dumps({
        "activation text": "glossy, shiny", "preferred weight": 0.8, "sd version": "SDXL",
    }), encoding="utf-8")
    put(path.with_name("style.preview.png"), b"\x89PNG\r\n\x1a\n")

    library.sync()
    library.inspect_pending()

    (model,) = database.list_models()
    assert model.identified, "a .civitai.info says what it is, without asking anybody"
    assert model.title == "Glossy Style"
    assert model.base_model == "Illustrious"
    assert model.trigger_words == ["glossy"]
    assert model.extras["activation_text"] == "glossy, shiny"
    assert model.extras["image"] == "style.preview.png"
    assert model.preview_count == 1
    assert model.identity == {"provider": "civitai", "ref": {"version_id": 11, "file_id": 33}}


def test_a_folder_that_is_away_keeps_its_models(setup, tmp_path):
    """An unplugged drive says nothing about what is on it: nothing is marked missing,
    relinked elsewhere, or dropped because it could not be seen for a minute."""
    library, root, database, events = setup
    other = tmp_path / "usb"
    put(other / "loras" / "found.safetensors", LORA)
    downloaded = put(other / "loras" / "downloaded.safetensors")
    done_task(database, downloaded)
    library.settings.extra_roots = [str(other)]
    library.sync()
    library.inspect_pending()
    before = {m.filename: m for m in database.list_models()}
    database.update_model(before["found.safetensors"].id, sha256="aa" * 32, hash_source="computed")

    other.rename(tmp_path / "usb-unplugged")
    library.sync()

    after = {m.filename: m for m in database.list_models()}
    assert set(after) == set(before)
    assert all(m.state == states.PRESENT for m in after.values())
    assert after["found.safetensors"].sha256 == "aa" * 32, "and the hash that took a while is kept"


def test_a_model_is_not_moved_onto_a_missing_one(setup):
    """The row of a model that went missing still holds its path; a different file moved
    there would be taken for it coming back, with its history and its note."""
    library, root, database, events = setup
    gone = put(root / "checkpoints" / "foo.safetensors")
    done_task(database, gone)
    other = put(root / "loras" / "foo.safetensors", b"another file")
    library.sync()
    gone.unlink()
    library.sync()
    moving = next(m for m in database.list_models() if m.path == str(other))

    with pytest.raises(FileExistsError, match="missing model"):
        library.move(moving.id, root / "checkpoints")

    assert other.is_file(), "refused before anything moved"


def test_identifying_a_download_only_says_where_else_it_is(setup):
    """Where a file was downloaded from is a fact about it; Civitai having the same bytes
    does not change it."""
    library, root, database, events = setup
    path = put(root / "loras" / "from_the_hub.safetensors", LORA)
    done_task(database, path, provider="huggingface", identity={
        "provider": "huggingface",
        "ref": {"repo_id": "org/repo", "repo_type": "model", "revision": "main", "path": path.name},
    }, meta={"repo_id": "org/repo", "path": path.name})
    library.sync()
    (model,) = database.list_models()

    async def run():
        library._client = civitai(lambda request: httpx.Response(200, json={
            "id": 5, "modelId": 6, "name": "v3", "model": {"name": "Hub Style", "type": "LORA"},
            "files": [], "images": [],
        }))
        await library._identify(model.id, "cd" * 32)
        await library.stop()

    import asyncio
    asyncio.run(run())

    after = database.get_model(model.id)
    assert after.provider == "huggingface" and after.meta["repo_id"] == "org/repo"
    assert after.lookup["result"] == "found" and after.lookup["model_name"] == "Hub Style"


def test_a_sidecar_nobody_can_read_does_not_stop_the_others(setup):
    library, root, database, events = setup
    first = put(root / "loras" / "a.safetensors", LORA)
    first.with_name("a.civitai.info").write_text(json.dumps({"model": "not an object", "files": "x"}), "utf-8")
    put(root / "loras" / "b.safetensors", LORA)
    library.sync()

    library.inspect_pending()

    for model in database.list_models():
        assert model.sniffed_mtime == model.mtime, model.filename
    assert database.model_at(root / "loras" / "b.safetensors").header["format"] == "safetensors"


def test_relinking_to_a_download_keeps_what_the_download_knew(setup):
    library, root, database, events = setup
    old = put(root / "loras" / "style.safetensors")
    library.sync()
    (found,) = database.list_models()
    library.set_note(found.id, "mine")
    old.unlink()
    library.sync()
    new = put(root / "loras" / "copy" / "style.safetensors")
    task = done_task(database, new)
    library.sync()
    lost = database.get_model(found.id)
    candidate = database.model_at(new)
    if lost.state == states.PRESENT:
        pytest.skip("the walk linked them on its own")

    library.relink(lost.id, candidate.id)

    after = database.get_model(lost.id)
    assert after.origin == states.DOWNLOADED and after.sha256 == "ab" * 32
    assert after.note == "mine"
    assert database.get(task.id).model_id == lost.id


def test_forgetting_never_takes_a_sibling_s_files(setup):
    """`foo.safetensors` gone, `foo.ckpt` still there: `foo.png` may be the checkpoint's."""
    library, root, database, events = setup
    gone = put(root / "checkpoints" / "foo.safetensors")
    done_task(database, gone)
    put(root / "checkpoints" / "foo.ckpt")
    picture = put(root / "checkpoints" / "foo.png", b"\x89PNG")
    library.sync()
    gone.unlink()
    library.sync()
    lost = database.model_at(gone)

    assert picture not in library.leftovers(lost.id)
    library.forget(lost.id, cleanup=True)
    assert picture.is_file()


def test_a_model_that_came_back_is_not_forgotten(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors")
    done_task(database, path)
    library.sync()
    data = path.read_bytes()
    path.unlink()
    library.sync()
    (model,) = database.list_models()
    path.write_bytes(data)

    with pytest.raises(ValueError, match="on the disk"):
        library.forget(model.id, cleanup=True)


def test_a_split_model_that_cannot_all_move_moves_back(setup, monkeypatch):
    library, root, database, events = setup
    folder = root / "LLM" / "m"
    put(folder / "model-00001-of-00002.safetensors")
    put(folder / "model-00002-of-00002.safetensors")
    library.sync()
    (model,) = database.list_models()
    real = relocate.move
    calls = []

    def second_fails(path, target, **kwargs):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("the disk is full")
        return real(path, target, **kwargs)

    monkeypatch.setattr(relocate, "move", second_fails)
    with pytest.raises(OSError):
        library.move(model.id, root / "LLM" / "n")
    monkeypatch.setattr(relocate, "move", real)

    assert sorted(p.name for p in folder.iterdir()) == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    ]
    assert database.get_model(model.id).path == str(folder / "model-00001-of-00002.safetensors")


def test_making_another_folder_the_main_one_moves_the_records(setup, tmp_path):
    library, root, database, events = setup
    collected = tmp_path / "records"
    other = tmp_path / "shared"
    library.settings.sidecar_dir = str(collected)
    library.settings.extra_roots = [str(other)]
    mine = put(root / "loras" / "a.safetensors")
    theirs = put(other / "loras" / "a.safetensors")
    library.sync()
    for path, text in ((mine, "the old main one's"), (theirs, "the other one's")):
        library.set_note(database.model_at(path).id, text)
    before = (library.settings.library_path, tuple(library.settings.roots))

    library.settings.library_root, library.settings.extra_roots = str(other), [str(root)]
    library.remap_records(*before)
    library.sync()

    assert database.model_at(mine).note == "the old main one's"
    assert database.model_at(theirs).note == "the other one's"


def test_a_redownloaded_model_keeps_its_note(client):
    """Pasting the link of a model that was deleted in Explorer knows nothing of its note;
    the record does, and the download landing again must not wipe it."""
    import asyncio

    from sfd.core.types import FileIdentity

    path = put(client.root / "loras" / "style.safetensors")
    first = done_task(client.database, path)
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]
    client.post(f"/api/models/{model['id']}/note", json={"note": "weight 0.7"})
    again = client.database.add(
        state="running", source="s", provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 99, "file_id": 1}},
        filename=path.name, dest=str(path),
    )
    manager = client.app.state.manager

    asyncio.run(manager._write_sidecar(path, again, FileIdentity("civitai", {"version_id": 99, "file_id": 1})))

    record = json.loads(path.with_name(path.name + ".json").read_text("utf-8"))
    assert record["note"] == "weight 0.7"
    assert first.id != again.id


def test_a_request_from_another_site_changes_nothing(client):
    """The Host check keeps out a page that renamed itself into this server; a page that
    simply sends a request to 127.0.0.1 is kept out by where the browser says it came from."""
    put(client.root / "loras" / "a.safetensors")
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    for headers in ({"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"},
                    {"origin": "null"}):
        response = client.post("/api/models/delete", json={"ids": [model["id"]]}, headers=headers)
        assert response.status_code == 403, headers
    assert (client.root / "loras" / "a.safetensors").is_file()

    own = {"origin": "http://127.0.0.1:7788", "sec-fetch-site": "same-origin"}
    assert client.post("/api/library/rescan", headers=own).status_code == 200
    assert client.get("/api/library", headers={"origin": "https://evil.example"}).status_code == 200


def test_an_old_queue_is_copied_aside_before_it_is_changed(tmp_path: Path):
    import sqlite3

    path = tmp_path / "queue.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,"
        " updated_at REAL NOT NULL, state TEXT NOT NULL, source TEXT NOT NULL,"
        " label TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL, identity TEXT NOT NULL,"
        " filename TEXT NOT NULL DEFAULT '', size INTEGER, sha256 TEXT,"
        " dest TEXT NOT NULL DEFAULT '', downloaded INTEGER NOT NULL DEFAULT 0,"
        " category TEXT, confidence TEXT, reason TEXT, disagreement TEXT, base_model TEXT,"
        " meta TEXT NOT NULL DEFAULT '{}', error TEXT);"
        "CREATE UNIQUE INDEX tasks_identity ON tasks(identity);"
        "INSERT INTO tasks (created_at, updated_at, state, source, provider, identity)"
        " VALUES (1, 1, 'done', 's', 'direct', '{}');"
    )
    old.commit()
    old.close()

    Database(path).close()

    backup = tmp_path / "queue.db.before-library.bak"
    assert backup.is_file()
    kept = sqlite3.connect(backup)
    names = {r[0] for r in kept.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    kept.close()
    assert "tasks_identity" in names, "the copy is the queue as the older build left it"


# --- the records of more than one folder ---------------------------------------------


def test_each_folder_of_the_library_has_its_own_mirror(tmp_path: Path):
    collected = tmp_path / "records"
    primary = tmp_path / "models"
    other = tmp_path / "shared"
    roots = (primary, other)

    first = sidecar.record_path(primary / "loras" / "a.safetensors", collected, primary, roots)
    second = sidecar.record_path(other / "loras" / "a.safetensors", collected, primary, roots)

    assert first == collected / "loras" / "a.safetensors.json", "the library root as before"
    assert second.parent.parent.parent == collected / sidecar.ROOTS_FOLDER
    assert second != first, "two files of one name in two folders, two records"


def test_a_record_left_beside_the_model_is_still_found(tmp_path: Path):
    """Written before the records were collected elsewhere: still this model's record."""
    model = put(tmp_path / "models" / "loras" / "a.safetensors")
    beside = model.with_name(model.name + ".json")
    beside.write_text("{}", encoding="utf-8")

    found = sidecar.find_record(model, tmp_path / "records", tmp_path / "models")

    assert found == beside


def test_a_record_written_under_the_old_main_folder_is_still_found(tmp_path: Path):
    """Making another folder the main one moves where new records go, not the old ones."""
    collected = tmp_path / "records"
    first = tmp_path / "first"
    second = tmp_path / "second"
    model = put(first / "loras" / "a.safetensors")
    old = sidecar.record_path(model, collected, first)
    put(old, json.dumps({"filename": "a.safetensors", "note": "mine"}).encode())

    found = sidecar.find_record(model, collected, second, roots=(second, first))

    assert found == old


def test_another_folder_s_model_does_not_take_this_one_s_record(tmp_path: Path):
    """The same relative path in two folders of the library: one mirrored as if it were in
    the other lands on the other's record, which is not its to read, or to write a note
    into, or to delete."""
    collected = tmp_path / "records"
    primary = tmp_path / "models"
    other = tmp_path / "shared"
    mine = put(primary / "loras" / "a.safetensors")
    theirs = put(other / "loras" / "a.safetensors")
    record = sidecar.record_path(mine, collected, primary, (primary, other))
    put(record, json.dumps({"filename": "a.safetensors", "note": "the primary's"}).encode())

    assert sidecar.find_record(theirs, collected, primary, (primary, other)) is None
    assert sidecar.find_record(mine, collected, primary, (primary, other)) == record


def test_a_guess_that_names_another_file_is_not_taken(tmp_path: Path):
    collected = tmp_path / "records"
    model = put(tmp_path / "elsewhere" / "a.safetensors")
    put(collected / "a.safetensors.json", json.dumps({"filename": "b.safetensors"}).encode())

    assert sidecar.find_record(model, collected, tmp_path / "models") is None


def test_writing_a_record_again_keeps_its_note(tmp_path: Path):
    """The same file downloaded a second time rewrites its record; the note stays."""
    model = put(tmp_path / "loras" / "a.safetensors")
    verdict = details.classify(model.name)
    sidecar.write(model, verdict, sidecar.Record(filename=model.name, note="keep me"),
                  compat=False, triggers=False)

    sidecar.write(model, verdict, sidecar.Record(filename=model.name), compat=False, triggers=False)

    stored = json.loads(model.with_name(model.name + ".json").read_text("utf-8"))
    assert stored["note"] == "keep me"


def test_strays_take_the_model_s_new_name(tmp_path: Path):
    """Found under another name: what it left behind is offered under that name."""
    old = tmp_path / "loras" / "old_name.safetensors"
    put(old.with_name("old_name.preview.png"), b"\x89PNG")
    put(old.with_name("old_name.txt"), b"words")
    new = put(tmp_path / "loras" / "Krea 2" / "new_name.safetensors")

    pairs = relocate.strays(old, new)
    relocate.bring(new, pairs)

    assert sorted(p.name for p in new.parent.iterdir()) == [
        "new_name.preview.png", "new_name.safetensors", "new_name.txt",
    ]


def test_a_shared_stem_keeps_its_companions_where_they_are(tmp_path: Path):
    """`model.safetensors` beside `model.ckpt`: whose is `model.png`? Nobody's to take."""
    folder = tmp_path / "checkpoints"
    model = put(folder / "model.safetensors")
    put(folder / "model.ckpt")
    put(folder / "model.png", b"\x89PNG")

    relocate.move(model, tmp_path / "elsewhere")

    assert (folder / "model.png").is_file()


def test_a_record_whose_model_moved_elsewhere_is_not_rubbish(setup, tmp_path):
    library, root, database, events = setup
    collected = tmp_path / "records"
    library.settings.sidecar_dir = str(collected)
    put(root / "loras" / "Krea 2" / "style.safetensors")
    put(collected / "loras" / "style.safetensors.json", b'{"note": "keep me"}')
    library.sync()

    kinds = [item["kind"] for item in library.cleanup_scan()["items"]]

    assert "record" not in kinds


def test_a_file_downloaded_before_can_be_queued_again_once_it_is_gone(tmp_path: Path):
    """The old rule was one row per remote file, ever: a deleted model's link could never be
    pasted again. Only unfinished rows are unique now."""
    database = Database(tmp_path / "queue.db")
    identity = {"provider": "direct", "ref": {"url": "https://example.com/a"}}
    first = database.add(source="s", provider="direct", identity=identity, filename="a", state="done")
    again = database.add(source="s", provider="direct", identity=identity, filename="a")
    twice = database.add(source="s", provider="direct", identity=identity, filename="a")

    assert first is not None and again is not None
    assert twice is None, "but never two unfinished downloads of one file"
    database.close()


# --- changing a model ------------------------------------------------------------------


def test_renaming_changes_the_name_in_the_history_too(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "pytorch_lora_weights.safetensors")
    task = done_task(database, path)
    library.sync()
    (model,) = database.list_models()

    library.rename(model.id, "k2 outpaint")

    after = database.get(task.id)
    assert after.filename == "k2 outpaint.safetensors"
    assert after.original_filename == "pytorch_lora_weights.safetensors", "what it arrived as"
    assert database.get_model(model.id).filename == "k2 outpaint.safetensors"


def test_a_split_model_cannot_be_renamed(setup):
    library, root, database, events = setup
    put(root / "LLM" / "m" / "model-00001-of-00002.safetensors")
    put(root / "LLM" / "m" / "model-00002-of-00002.safetensors")
    library.sync()
    (model,) = database.list_models()

    with pytest.raises(ValueError):
        library.rename(model.id, "other")


def test_a_split_model_moves_whole(setup):
    library, root, database, events = setup
    folder = root / "LLM" / "m"
    put(folder / "model-00001-of-00002.safetensors")
    put(folder / "model-00002-of-00002.safetensors")
    (folder / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    library.sync()
    (model,) = database.list_models()

    library.move(model.id, root / "LLM" / "n")

    moved = sorted(p.name for p in (root / "LLM" / "n").iterdir())
    assert moved == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "model.safetensors.index.json"]
    assert len(database.get_model(model.id).parts) == 2


def test_deleting_a_model_leaves_its_history_marked(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors")
    task = done_task(database, path)
    library.sync()
    (model,) = database.list_models()

    library.delete_files(model.id)

    assert not path.exists()
    assert database.list_models() == []
    after = database.get(task.id)
    assert after.fate == states.DELETED and after.model_id is None


def test_forgetting_a_missing_model_can_take_what_it_left(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors")
    task = done_task(database, path)
    library.sync()
    (model,) = database.list_models()
    library.set_note(model.id, "worth keeping")
    preview = put(path.with_name("style.preview.png"), b"\x89PNG")
    path.unlink()
    library.sync()

    leftovers = {p.name for p in library.leftovers(model.id)}
    assert leftovers == {"style.safetensors.json", "style.preview.png"}

    library.forget(model.id, cleanup=True)

    assert not preview.exists()
    assert database.list_models() == []
    assert database.get(task.id).fate == states.FORGOTTEN


def test_a_model_still_on_disk_is_not_forgotten(setup):
    library, root, database, events = setup
    put(root / "loras" / "style.safetensors")
    library.sync()
    (model,) = database.list_models()

    with pytest.raises(ValueError):
        library.forget(model.id)


def test_pointing_a_model_at_a_file_of_another_size_asks_first(setup, tmp_path):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors")
    done_task(database, path)
    library.sync()
    (model,) = database.list_models()
    path.unlink()
    library.sync()
    elsewhere = put(tmp_path / "elsewhere" / "renamed.safetensors", b"different size")

    answer = library.link_to(model.id, elsewhere)

    assert "confirm" in answer, "a file of another size is a question"
    assert database.get_model(model.id).state == states.MISSING
    library.confirm_link(model.id, answer["confirm"]["token"])
    after = database.get_model(model.id)
    assert after.state == states.PRESENT and after.path == str(elsewhere)


# --- duplicates and rubbish -------------------------------------------------------------


BIG = 1024 * 1024 + 4096


def weights(seed: int = 0, size: int = BIG) -> bytes:
    """Bytes that look like nothing else: a different seed differs everywhere."""
    block = hashlib.sha256(f"seed {seed}".encode()).digest() * 2048
    return (block * (size // len(block) + 1))[:size]


def hashed(database: Database, *paths: Path) -> None:
    """Record the true hash of each file, as a hash job would."""
    for path in paths:
        model = database.model_at(path)
        database.update_model(model.id, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              hash_source="computed", hashed_mtime=model.mtime)


def test_the_same_file_twice_is_a_duplicate(setup):
    library, root, database, events = setup
    a = put(root / "loras" / "a.safetensors", weights())
    b = put(root / "loras" / "copy" / "a.safetensors", weights())
    library.sync()

    (group,) = library.duplicates()["groups"]
    assert group["status"] == "likely", "pieces that agree are a reason to hash, not proof"
    assert sorted(group["to_hash"]) == sorted(m.id for m in database.list_models())

    hashed(database, a, b)
    found = library.duplicates()

    (group,) = found["groups"]
    assert group["status"] == "same" and group["copies"] == 2 and not group["to_hash"]
    assert found["wasted"] == BIG


def test_files_of_one_size_are_not_copies_unless_they_look_alike(setup):
    """The old guess: a dozen LoRAs of one rank are a dozen files of one size."""
    library, root, database, events = setup
    for seed in range(3):
        put(root / "loras" / f"style{seed}.safetensors", weights(seed))
    library.sync()

    found = library.duplicates()

    assert found["groups"] == []
    assert found["shared_sizes"] == 1 and found["told_apart"] == 1


def test_pieces_that_agree_are_not_proof(setup):
    """Two files alike wherever they were sampled — merges that left the text encoder alone —
    are only as alike as their hashes say."""
    library, root, database, events = setup
    data = bytearray(weights())
    a = put(root / "checkpoints" / "merge_a.safetensors", bytes(data))
    data[200_000] ^= 0xFF
    b = put(root / "checkpoints" / "merge_b.safetensors", bytes(data))
    library.sync()

    assert [g["status"] for g in library.duplicates()["groups"]] == ["likely"]

    hashed(database, a, b)

    assert library.duplicates()["groups"] == []


def test_the_fingerprint_is_taken_in_the_background_once_per_version(setup, monkeypatch):
    from sfd.library import fingerprint

    library, root, database, events = setup
    path = put(root / "loras" / "a.safetensors", weights())
    library.sync()
    read = []
    real = fingerprint.sample
    monkeypatch.setattr(fingerprint, "sample", lambda p: read.append(p) or real(p))

    library.inspect_pending()
    library.inspect_pending()

    (model,) = database.list_models()
    assert read == [path], "read once, then trusted until the file changes"
    assert model.fingerprint and model.sampled_mtime == model.mtime

    put(path, weights(1))
    import os
    os.utime(path, (model.mtime + 10, model.mtime + 10))
    library.sync()
    library.inspect_pending()

    assert read == [path, path]
    assert database.get_model(model.id).fingerprint != model.fingerprint


def test_the_autov1_hash_is_the_one_a1111_showed(tmp_path: Path):
    from sfd.library import fingerprint

    data = weights()
    big = put(tmp_path / "big.safetensors", data)
    small = put(tmp_path / "small.safetensors", data[:4096])

    assert fingerprint.sample(big).autov1 == hashlib.sha256(data[0x100000:0x110000]).hexdigest()[:8]
    assert fingerprint.sample(small).autov1 is None, "too small to reach where it is taken"


def test_a_hard_link_is_not_a_copy(setup):
    import os

    library, root, database, events = setup
    a = put(root / "insightface" / "inswapper_128.onnx", weights())
    b = root / "simswap" / "inswapper_128.onnx"
    b.parent.mkdir(parents=True)
    try:
        os.link(a, b)
    except OSError:
        pytest.skip("this file system has no hard links")
    library.sync()
    hashed(database, a, b)

    assert library.duplicates()["groups"] == [], "one file under two names frees nothing"


def test_the_copy_kept_is_the_one_the_library_knows_most_about(setup):
    library, root, database, events = setup
    found = put(root / "simswap" / "inswapper_128.onnx", weights())
    downloaded = put(root / "insightface" / "inswapper_128.onnx", weights())
    done_task(database, downloaded, sha256=hashlib.sha256(weights()).hexdigest(), size=BIG)
    library.sync()
    hashed(database, found)

    (group,) = library.duplicates()["groups"]

    assert group["keep"] == database.model_at(downloaded).id
    assert "downloaded here" in group["keep_why"]
    assert "insightface" in group["caution"] and "simswap" in group["caution"], \
        "different folders are often different nodes, each reading its own copy"


def test_copies_in_one_kind_s_folder_carry_no_warning_and_packs_do(setup):
    """`vae` is read whole by its loader; the insightface packs are each loaded as a set."""
    library, root, database, events = setup
    vae = [put(root / "vae" / "ae.safetensors", weights(1)),
           put(root / "vae" / "flux" / "ae.safetensors", weights(1))]
    packs = [put(root / "insightface" / "models" / pack / "1k3d68.onnx", weights(2))
             for pack in ("antelopev2", "buffalo_l")]
    library.sync()
    hashed(database, *vae, *packs)

    cautions = {g["models"][0]["filename"]: g["caution"] for g in library.duplicates()["groups"]}

    assert cautions["ae.safetensors"] is None
    assert "antelopev2" in cautions["1k3d68.onnx"] and "buffalo_l" in cautions["1k3d68.onnx"]


def test_an_old_library_gains_the_new_columns(tmp_path: Path):
    import sqlite3

    Database(tmp_path / "queue.db").close()
    raw = sqlite3.connect(tmp_path / "queue.db")
    for column in ("fingerprint", "autov1", "sampled_mtime"):
        raw.execute(f"ALTER TABLE models DROP COLUMN {column}")
    raw.commit()
    raw.close()

    database = Database(tmp_path / "queue.db")
    model = database.add_model(path=str(tmp_path / "a.safetensors"), fingerprint="ab", autov1="cd")

    assert (model.fingerprint, model.autov1) == ("ab", "cd")
    database.close()


def test_the_page_is_told_which_copies_are_certain(client):
    put(client.root / "vae" / "ae.safetensors", weights())
    put(client.root / "vae" / "flux" / "ae.safetensors", weights())
    client.post("/api/library/rescan")

    body = client.get("/api/duplicates").json()

    (group,) = body["groups"]
    assert group["status"] == "likely" and len(group["models"]) == 2
    assert group["to_read"] == 2 * BIG


def test_the_cleanup_finds_what_belongs_to_nothing(setup):
    library, root, database, events = setup
    put(root / "loras" / "abandoned.safetensors.part", b"x" * 100)
    put(root / "loras" / "abandoned.safetensors.part.json", b"{}")
    put(root / "loras" / "gone.preview.png", b"\x89PNG")
    put(root / "loras" / "gone.txt", b"trigger")
    put(root / "loras" / "present.safetensors")
    put(root / "loras" / "present.preview.png", b"\x89PNG")
    library.sync()

    found = library.cleanup_scan()

    kinds = sorted((item["kind"], item["name"]) for item in found["items"])
    assert kinds == [("fragment", "abandoned.safetensors"), ("orphan", "gone")]
    orphan = next(i for i in found["items"] if i["kind"] == "orphan")
    assert sorted(f["name"] for f in orphan["files"]) == ["gone.preview.png", "gone.txt"]

    stale = found["token"]
    library.cleanup_scan()
    with pytest.raises(ValueError):
        library.cleanup_delete([item["id"] for item in found["items"]], stale)

    fresh = library.cleanup_scan()
    library.cleanup_delete([item["id"] for item in fresh["items"]], fresh["token"])

    assert sorted(p.name for p in (root / "loras").iterdir()) == [
        "present.preview.png", "present.safetensors",
    ]


def test_a_download_still_under_way_is_not_rubbish(setup):
    library, root, database, events = setup
    put(root / "loras" / "coming.safetensors.part", b"x")
    database.add(
        state="paused", source="s", provider="direct",
        identity={"provider": "direct", "ref": {"url": "u"}},
        filename="coming.safetensors", dest=str(root / "loras" / "coming.safetensors"),
    )
    library.sync()

    assert library.cleanup_scan()["items"] == []


# --- identifying -------------------------------------------------------------------------


def civitai(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_identifying_a_model_asks_civitai_by_its_hash(setup):
    library, root, database, events = setup
    library.settings.fetch_previews = False
    path = put(root / "loras" / "mystery.safetensors", LORA)
    digest = hashlib.sha256(LORA).hexdigest()
    library.sync()
    (model,) = database.list_models()
    asked = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(200, json={
            "id": 5, "modelId": 6, "name": "v3", "baseModel": "Flux.1 D",
            "trainedWords": ["m1st"],
            "model": {"name": "Mystery Style", "type": "LORA", "nsfw": False},
            "files": [{"id": 7, "name": "whatever.safetensors", "sizeKB": 1, "primary": True,
                       "hashes": {"SHA256": digest.upper()}}],
            "images": [],
        })

    library._client = civitai(answer)
    library.start()
    try:
        library.enqueue("identify", [model.id])
        for _ in range(200):
            if library.current_job is None and not library._jobs:
                after = database.get_model(model.id)
                if after.lookup.get("result"):
                    break
            import asyncio
            await asyncio.sleep(0.02)
    finally:
        await library.stop()

    after = database.get_model(model.id)
    assert asked == [f"/api/v1/model-versions/by-hash/{digest}"]
    assert after.sha256 == digest and after.hash_source == "computed"
    assert after.lookup["result"] == "found"
    assert after.title == "Mystery Style"
    assert after.identity == {"provider": "civitai", "ref": {"version_id": 5, "file_id": 7}}
    assert after.trigger_words == ["m1st"]
    record = json.loads(path.with_name(path.name + ".json").read_text("utf-8"))
    assert record["source"]["model_name"] == "Mystery Style"


async def test_a_model_civitai_does_not_know_says_so(setup):
    library, root, database, events = setup
    put(root / "loras" / "mystery.safetensors", LORA)
    library.sync()
    (model,) = database.list_models()
    library._client = civitai(lambda request: httpx.Response(404, json={"error": "nope"}))

    await library._identify(model.id, "00" * 32)

    assert database.get_model(model.id).lookup["result"] == "not_found"
    await library.stop()


async def test_a_newer_version_on_civitai_is_noticed(setup):
    library, root, database, events = setup
    path = put(root / "loras" / "style.safetensors")
    done_task(database, path)
    library.sync()
    (model,) = database.list_models()
    library._client = civitai(lambda request: httpx.Response(200, json={
        "id": 1, "modelVersions": [
            {"id": 9, "name": "v2", "baseModel": "Flux.1 D", "publishedAt": "2026-09-01"},
            {"id": 1, "name": "v1"},
        ],
    }))

    result = await library.check_updates([model.id])

    assert result == {"checked": 1, "updates": 1, "failed": 0}
    update = database.get_model(model.id).updates
    assert update["available"] and update["version_name"] == "v2"
    assert "modelVersionId=9" in update["page"]
    await library.stop()


# --- through the page --------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    library_root = tmp_path / "models"
    library_root.mkdir()
    settings = Settings(
        library_root=str(library_root),
        download_dir=str(tmp_path / "downloads"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database  # type: ignore[attr-defined]
        test.root = library_root  # type: ignore[attr-defined]
        test.settings = settings  # type: ignore[attr-defined]
        yield test


def test_the_page_is_given_the_whole_library(client):
    put(client.root / "loras" / "a.safetensors", LORA)
    client.post("/api/library/rescan")

    body = client.get("/api/library").json()

    assert [r["path"] for r in body["roots"]] == [str(client.root)]
    assert [m["filename"] for m in body["models"]] == ["a.safetensors"]
    assert body["models"][0]["relative"] == "loras"
    assert {"root": 0, "relative": "loras", "exists": True} in body["folders"]


def test_the_page_moves_a_model_into_another_folder_of_the_library(client, tmp_path):
    other = tmp_path / "shared"
    other.mkdir()
    client.settings.extra_roots = [str(other)]
    put(client.root / "loras" / "a.safetensors")
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    body = client.post(f"/api/models/{model['id']}/move",
                       json={"root": 1, "folder": "loras/new"}).json()

    assert body["ok"]
    assert (other / "loras" / "new" / "a.safetensors").is_file()


def test_the_page_cannot_move_a_model_out_of_the_library_by_name(client):
    put(client.root / "loras" / "a.safetensors")
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    for folder in ("../escape", "C:\\Windows", "/etc"):
        response = client.post(f"/api/models/{model['id']}/move", json={"root": 0, "folder": folder})
        assert response.status_code == 400, folder
    assert client.post(f"/api/models/{model['id']}/move",
                       json={"root": 7, "folder": "x"}).status_code == 400


def test_a_download_of_a_file_still_in_the_library_is_refused(client, monkeypatch):
    from sfd.core.types import FileIdentity
    from sfd.providers import registry
    from sfd.providers.direct import DirectProvider

    path = put(client.root / "loras" / "a.safetensors")
    identity = {"provider": "direct", "ref": {"url": "https://example.com/a.safetensors"}}
    done_task(client.database, path, provider="direct", identity=identity, sha256=None)
    client.post("/api/library/rescan")

    async def expand(text, client_, **kwargs):
        return registry.Resolution(
            provider=DirectProvider(),
            items=[registry.Item(identity=FileIdentity(**identity), filename="a.safetensors")],
            label=text,
        )

    monkeypatch.setattr("sfd.jobs.manager.expand", expand)
    response = client.post("/api/tasks", json={"source": "https://example.com/a.safetensors"})

    assert response.status_code == 409
    assert "already in your library" in response.json()["detail"]


def test_a_deleted_download_can_be_fetched_again_from_the_history(client):
    path = put(client.root / "loras" / "a.safetensors")
    task = done_task(client.database, path)
    client.post("/api/library/rescan")
    assert client.post(f"/api/history/{task.id}/redownload").status_code == 409, \
        "still in the library"

    (model,) = client.get("/api/library").json()["models"]
    client.request("DELETE", f"/api/models/{model['id']}/files")
    body = client.post(f"/api/history/{task.id}/redownload").json()

    assert body["ok"]
    again = client.database.get(body["task"]["id"])
    assert again.dest == str(path) and again.state != "done"


def test_hiding_a_folder_takes_it_out_of_the_library(client):
    put(client.root / "LLM" / "llama.cpp" / "models" / "vocab.gguf")
    put(client.root / "LLM" / "real.gguf")
    client.post("/api/library/rescan")
    assert len(client.get("/api/library").json()["models"]) == 2

    client.post("/api/folders/hide", json={"root": 0, "relative": "LLM/llama.cpp"})

    names = [m["filename"] for m in client.get("/api/library").json()["models"]]
    assert names == ["real.gguf"]
    body = client.post("/api/folders/unhide", json={"index": 0}).json()
    assert body["hidden"] == []


def test_a_folder_is_made_only_inside_the_library(client):
    assert client.post("/api/folders", json={"root": 0, "parent": "loras", "name": "Krea 2"}).json()["ok"]
    assert (client.root / "loras" / "Krea 2").is_dir()
    for name in ("../x", "a/b", "..", "", "con?"):
        assert client.post("/api/folders", json={"root": 0, "parent": "", "name": name}).status_code in (409, 422), name


def test_library_folders_are_added_and_taken_away(client, tmp_path, monkeypatch):
    import sfd.desktop

    other = tmp_path / "second"
    other.mkdir()
    monkeypatch.setattr(sfd.desktop, "pick_system_folder", lambda initial="", window=None: str(other))

    roots = client.post("/api/roots/add").json()["roots"]
    assert [r["path"] for r in roots] == [str(client.root), str(other)]

    roots = client.post("/api/roots/1/primary").json()["roots"]
    assert roots[0]["path"] == str(other) and client.settings.library_root == str(other)

    roots = client.post("/api/roots/1/remove").json()["roots"]
    assert [r["path"] for r in roots] == [str(other)]


def test_the_window_remembers_how_it_was_left(client):
    client.put("/api/ui", json={"prefs": {"left": 240, "open": ["0:loras"]}})
    client.put("/api/ui", json={"prefs": {"right": 320, "left": None}})

    assert client.get("/api/settings").json()["settings"]["ui"] == {
        "open": ["0:loras"], "right": 320,
    }
    too_much = {"prefs": {"blob": "x" * (70 * 1024)}}
    assert client.put("/api/ui", json=too_much).status_code == 413


def test_a_local_picture_is_served_for_a_model_that_has_one(client):
    path = put(client.root / "loras" / "a.safetensors")
    put(path.with_name("a.preview.png"), b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    response = client.get(f"/api/models/{model['id']}/preview/0")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_the_details_of_a_model(client):
    path = put(client.root / "loras" / "a.safetensors", LORA)
    task = done_task(client.database, path, size=len(LORA))
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    body = client.get(f"/api/models/{model['id']}").json()

    assert body["history"][0]["id"] == task.id
    assert body["page"] == "https://civitai.com/models/1?modelVersionId=1"
    assert body["meta"]["model_name"] == "Style"
    assert body["exists"] is True
