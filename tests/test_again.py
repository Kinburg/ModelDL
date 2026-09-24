"""Downloading a missing model again.

What the library kept when the file went — the link, the hash, the note — is enough to
fetch it again, and the file that arrives is that model back, whichever folder it is put in
this time: the same history, the same note, the same record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.core.types import FileIdentity
from sfd.jobs import db as states
from sfd.jobs.db import Database
from sfd.jobs.manager import Manager
from sfd.library import sidecar
from sfd.settings import Settings
from sfd.web.app import create_app

IDENTITY = {"provider": "civitai", "ref": {"version_id": 5, "file_id": 7}}


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
        sidecar_dir=str(tmp_path / "records"),
        auto_start=False,
        fetch_previews=False,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    yield manager, root, database
    database.close()


def gone(manager: Manager, database: Database, path: Path, note: str | None = "works at 0.8"):
    """A model downloaded here, annotated, and then deleted behind the app's back."""
    put(path)
    database.add(state="done", source="https://civitai.com/models/3", provider="civitai",
                 identity=IDENTITY, filename=path.name, dest=str(path), size=7,
                 sha256="ab" * 32, category="lora", confidence="high",
                 reason="Civitai lists this as LORA", base_model="Krea 2",
                 meta={"model_name": "Style", "model_id": 3, "version_id": 5, "host": "civitai.com"})
    manager.library.sync()
    model = database.model_at(path)
    if note:
        manager.library.set_note(model.id, note)
    path.unlink()
    manager.library.sync()
    model = database.get_model(model.id)
    assert model.state == states.MISSING
    return model


async def land(manager: Manager, database: Database, task_id: int) -> Path:
    """What the transfer does when the last byte arrives, without a transfer."""
    task = database.get(task_id)
    path = put(Path(task.dest))
    database.update(task.id, state=states.DONE, downloaded=7, finished_at=1.0)
    task = database.get(task.id)
    identity = FileIdentity(provider=task.identity["provider"], ref=task.identity["ref"])
    await manager._write_sidecar(path, task, identity)
    manager._landed(task, path)
    return path


def test_the_folder_it_was_in_is_offered_first(setup):
    manager, root, database = setup
    model = gone(manager, database, root / "loras" / "Krea 2" / "style.safetensors")

    answer = manager.resolve_again(model_id=model.id)

    (item,) = answer["items"]
    assert item["checked"] and item["filename"] == "style.safetensors"
    first = answer["rankings"][item["ranking"]][0]
    assert (first["root"], first["relative"]) == (0, "loras/Krea 2")
    assert first["reason"].startswith("where it was")


async def test_downloaded_again_elsewhere_it_is_the_same_model(setup):
    manager, root, database = setup
    model = gone(manager, database, root / "loras" / "Krea 2" / "style.safetensors")
    answer = manager.resolve_again(model_id=model.id)

    (task,) = manager.queue_resolved(answer["token"], [0], root / "loras" / "styles")["created"]
    assert task.model_id == model.id and task.note == "works at 0.8"
    path = await land(manager, database, task.id)

    back = database.get_model(model.id)
    assert back.state == states.PRESENT and back.path == str(path), "the same entry, not a second one"
    assert back.note == "works at 0.8"
    assert len(database.list_models()) == 1
    assert len(database.tasks_for_model(model.id)) == 2, "downloaded twice, one history"
    record = sidecar.find_record(path, **manager.library.places())
    assert record is not None and json.loads(record.read_text("utf-8"))["note"] == "works at 0.8"


async def test_a_walk_that_meets_the_file_first_changes_nothing(setup):
    manager, root, database = setup
    model = gone(manager, database, root / "loras" / "style.safetensors", note=None)
    answer = manager.resolve_again(model_id=model.id)
    (task,) = manager.queue_resolved(answer["token"], [0], root / "loras" / "elsewhere")["created"]
    put(Path(task.dest), b"new copy")
    manager.library.sync()

    path = await land(manager, database, task.id)

    assert [m.id for m in database.list_models()] == [model.id]
    assert database.get_model(model.id).path == str(path)


def test_several_go_back_where_they_were(setup):
    manager, root, database = setup
    a = gone(manager, database, root / "loras" / "a.safetensors", note=None)

    result = manager.redownload_models([a.id])

    (task,) = result["created"]
    assert task.dest == a.path and task.model_id == a.id
    assert result["failed"] == []


def test_a_file_nothing_says_the_source_of_is_not_offered_again(setup):
    manager, root, database = setup
    path = put(root / "loras" / "mystery.safetensors")
    manager.library.sync()
    found = database.model_at(path)
    manager.library.set_note(found.id, "kept for the note")
    path.unlink()
    manager.library.sync()

    with pytest.raises(ValueError, match="came from"):
        manager.resolve_again(model_id=found.id)
    assert manager.library.summary(database.get_model(found.id))["source"] is False


def test_a_model_still_on_disk_is_not_downloaded_again(setup):
    manager, root, database = setup
    path = put(root / "loras" / "here.safetensors")
    database.add(state="done", source="s", provider="civitai", identity=IDENTITY,
                 filename=path.name, dest=str(path), size=7)
    manager.library.sync()

    with pytest.raises(ValueError, match="still in your library"):
        manager.resolve_again(model_id=database.model_at(path).id)


def test_the_page_asks_where_before_downloading_again(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "d"), auto_start=False,
                        concurrent_downloads=1, _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as client:
        manager = client.app.state.manager
        model = gone(manager, database, root / "loras" / "style.safetensors", note=None)

        answer = client.post(f"/api/models/{model.id}/again").json()
        body = client.post(f"/api/resolve/{answer['token']}/queue",
                           json={"files": [0], "root": 0, "folder": "loras"}).json()

        (task,) = body["tasks"]
        assert task["model_id"] == model.id
