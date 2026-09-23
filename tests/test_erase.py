"""Deleting a model, and everything named after it.

Two things are being tested and only one of them is the unlink. The other is the set: a
delete that takes the model and leaves four files named after it in the folder has not
tidied anything, and an abandoned 40 GB `.part` — the usual reason anyone reaches for this
at all — is not next to the model but next to where the model was going to be.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.jobs.manager import Manager
from sfd.library import erase
from sfd.settings import Settings
from sfd.web.app import create_app

MODEL = "mystery.safetensors"


def place(folder: Path, name: str = MODEL, sidecars: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"weights")
    if sidecars:
        stem = path.stem
        (folder / f"{name}.json").write_text('{"schema": 1}', encoding="utf-8")
        (folder / f"{stem}.civitai.info").write_text("{}", encoding="utf-8")
        (folder / f"{stem}.txt").write_text("trigger word\n", encoding="utf-8")
        (folder / f"{stem}.preview.png").write_bytes(b"\x89PNG")
    return path


# --- the module -------------------------------------------------------------


def test_the_sidecars_go_with_the_model(tmp_path: Path):
    path = place(tmp_path / "loras")

    result = erase.erase(path)

    assert not result.failed and not result.missing
    assert len(result.deleted) == 5
    assert list((tmp_path / "loras").iterdir()) == []


def test_the_fragments_of_an_abandoned_download_go_too(tmp_path: Path):
    """The whole reason to delete anything: a `.part` is most of a file's size, and it is
    named after a model that does not exist, so nothing else will ever clear it."""
    folder = tmp_path / "loras"
    folder.mkdir()
    destination = folder / MODEL
    (folder / f"{MODEL}.part").write_bytes(b"most of it")
    (folder / f"{MODEL}.part.json").write_text("{}", encoding="utf-8")
    (folder / f"{MODEL}.part.corrupt").write_bytes(b"a failed checksum")

    result = erase.erase(destination)

    assert result.missing, "the model never landed, which is not a failure"
    assert len(result.deleted) == 3
    assert list(folder.iterdir()) == []


def test_a_collected_record_is_followed_into_its_mirror(tmp_path: Path):
    library = tmp_path / "library"
    collected = tmp_path / "meta"
    path = place(library / "loras", sidecars=False)
    record = collected / "loras" / f"{MODEL}.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"schema": 1}', encoding="utf-8")

    erase.erase(path, sidecar_dir=collected, library_root=library)

    assert not record.exists()
    assert not path.exists()


def test_a_model_that_will_not_delete_takes_nothing_with_it(tmp_path: Path, monkeypatch):
    """On Windows this is a loader holding the weights open. Deleting the record, the
    triggers and the preview of a model that is still sitting there is strictly worse than
    deleting nothing at all."""
    path = place(tmp_path / "loras")
    real = Path.unlink

    def refuse(self, *args, **kwargs):
        if self == path:
            raise PermissionError("the file is open in another program")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)

    with pytest.raises(OSError):
        erase.erase(path)

    assert path.is_file()
    assert (tmp_path / "loras" / f"{MODEL}.json").is_file()
    assert (tmp_path / "loras" / "mystery.preview.png").is_file()


def test_a_sidecar_that_will_not_go_is_named_rather_than_raised(tmp_path: Path, monkeypatch):
    """Past the model the set is gone whatever happens, so a stray file is a thing to
    report, not a reason to leave the rest of them lying around."""
    path = place(tmp_path / "loras")
    stubborn = tmp_path / "loras" / "mystery.txt"
    real = Path.unlink

    def refuse(self, *args, **kwargs):
        if self == stubborn:
            raise PermissionError("nope")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)

    result = erase.erase(path)

    assert [p.name for p, _ in result.failed] == ["mystery.txt"]
    assert not path.exists()
    assert not (tmp_path / "loras" / "mystery.preview.png").exists(), "the rest still went"


def test_the_list_shown_before_the_delete_leads_with_the_model(tmp_path: Path):
    """It is the file whose size is the reason anyone is looking, so it is not buried
    among four sidecars of a few kilobytes each."""
    path = place(tmp_path / "loras")

    found = erase.belongings(path)

    assert found[0] == path
    assert len(found) == 5


def test_nothing_on_disk_is_an_empty_list_and_not_an_error(tmp_path: Path):
    assert erase.belongings(tmp_path / "loras" / MODEL) == []


# --- through the page -------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir(parents=True)
    settings = Settings(
        library_root=str(library),
        download_dir=str(tmp_path / "downloads"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database  # type: ignore[attr-defined]
        test.library = library  # type: ignore[attr-defined]
        yield test


def finished(client, **overrides):
    path = place(client.library / "loras")
    values = {
        "state": "done",
        "source": "https://example.com/model.safetensors",
        "provider": "direct",
        "identity": {"provider": "direct", "ref": {"url": "https://example.com/m"}},
        "filename": MODEL,
        "category": "lora",
        "dest": str(path),
    }
    values.update(overrides)
    return client.database.add(**values)


def test_the_page_is_told_what_a_delete_would_take(client):
    task = finished(client)

    body = client.get(f"/api/tasks/{task.id}/files").json()

    assert [f["name"] for f in body["files"]][0] == MODEL
    assert len(body["files"]) == 5
    assert body["total"] == sum(f["size"] for f in body["files"])
    assert body["model"] == str(client.library / "loras" / MODEL)


def test_deleting_takes_the_files_and_leaves_the_history(client):
    task = finished(client)

    body = client.request("DELETE", f"/api/tasks/{task.id}/files").json()

    assert body["ok"] and len(body["deleted"]) == 5 and not body["failed"]
    assert list((client.library / "loras").iterdir()) == []
    # Off the Downloads list, since there is no file left for its card to be about — but
    # still in the history, which is about what arrived and when, and which is what
    # "download it again" needs.
    after = client.database.get(task.id)
    assert after.fate == "deleted" and after.archived
    assert after.model_id is None
    assert client.database.list_models() == [], "and the library forgot the model"


def test_deleting_a_download_that_never_finished_takes_its_row(client):
    """Never history: nothing arrived, so there is nothing to remember it by."""
    path = client.library / "loras" / MODEL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(MODEL + ".part").write_bytes(b"half")
    task = finished(client, state="failed")
    path.unlink()

    body = client.request("DELETE", f"/api/tasks/{task.id}/files").json()

    assert body["ok"]
    assert client.database.get(task.id) is None


def test_a_task_that_never_wrote_anything_lists_nothing(client):
    task = finished(client, dest="")

    assert client.get(f"/api/tasks/{task.id}/files").json() == {"files": [], "total": 0}


def test_deleting_is_refused_while_the_file_is_being_moved(client):
    """A copy running on a thread cannot be told the file underneath it has gone."""
    import threading

    task = finished(client)
    client.app.state.manager._moves[task.id] = threading.Event()

    response = client.request("DELETE", f"/api/tasks/{task.id}/files")

    assert response.status_code == 409
    assert (client.library / "loras" / MODEL).is_file()


def test_a_task_that_is_not_there_is_a_404(client):
    assert client.request("DELETE", "/api/tasks/9999/files").status_code == 404
    assert client.get("/api/tasks/9999/files").status_code == 404
