"""What you wrote about a file yourself.

The only field in a record that no service supplies and nothing can fetch again, which is
what decides where it lives. The queue row is cleared by *Clear finished*; the record is
not, follows the file through a move and a rename, and goes when the file goes. So the
record is the note, and the column beside the queue is a copy of it — the same relationship
`downloaded` has with the `.part.json` next to a half-finished file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.library import erase, relocate, sidecar
from sfd.settings import Settings
from sfd.web.app import create_app

MODEL = "mystery.safetensors"
NOTE = "too strong past 0.6, and it fights the detail slider"


def place(folder: Path, name: str = MODEL, record: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"weights")
    if record:
        (folder / f"{name}.json").write_text(
            json.dumps({"schema": 1, "filename": name}), encoding="utf-8"
        )
    return path


def written(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


# --- the record ---------------------------------------------------------------


def test_a_note_lands_in_the_record(tmp_path: Path):
    path = place(tmp_path / "loras")
    record = path.with_name(path.name + ".json")

    assert sidecar.annotate(record, NOTE) == NOTE

    stored = written(record)
    assert stored["note"] == NOTE
    assert stored["filename"] == MODEL, "and nothing else in the record was disturbed"


def test_an_emptied_box_removes_the_note_rather_than_storing_a_blank(tmp_path: Path):
    """A record carrying an empty string would put a blank note chip on the card forever."""
    path = place(tmp_path / "loras")
    record = path.with_name(path.name + ".json")
    sidecar.annotate(record, NOTE)

    assert sidecar.annotate(record, "   ") is None
    assert written(record)["note"] is None


def test_a_record_nobody_can_read_is_left_alone(tmp_path: Path):
    """More likely one somebody edited by hand than one that is genuinely broken, and
    overwriting it would throw away the very thing this field is for."""
    path = place(tmp_path / "loras", record=False)
    record = path.with_name(path.name + ".json")
    record.write_text("{ not json at all", encoding="utf-8")

    with pytest.raises(ValueError):
        sidecar.annotate(record, NOTE)

    assert record.read_text("utf-8") == "{ not json at all"


def test_a_fresh_record_carries_the_field_empty(tmp_path: Path):
    """Present from the first write, so nothing reading records has to guess whether the
    key is missing or the note is."""
    from sfd.library.categories import Category
    from sfd.library.classify import Verdict

    path = place(tmp_path / "loras", record=False)
    sidecar.write(
        path,
        Verdict(Category.LORA, "high", "the filename says so"),
        sidecar.Record(filename=MODEL),
        compat=False,
        triggers=False,
    )

    assert written(path.with_name(path.name + ".json"))["note"] is None


# --- the note goes where the file goes ----------------------------------------


def test_a_note_survives_a_move(tmp_path: Path):
    path = place(tmp_path / "loras")
    sidecar.annotate(path.with_name(path.name + ".json"), NOTE)

    result = relocate.move(path, tmp_path / "checkpoints")

    assert written(result.path.with_name(result.path.name + ".json"))["note"] == NOTE


def test_a_note_survives_a_rename(tmp_path: Path):
    path = place(tmp_path / "loras")
    sidecar.annotate(path.with_name(path.name + ".json"), NOTE)

    result = relocate.rename(path, "renamed")

    stored = written(result.path.with_name(result.path.name + ".json"))
    assert stored["note"] == NOTE
    assert stored["filename"] == "renamed.safetensors", "both were rewritten, not one"


def test_a_note_goes_when_the_file_goes(tmp_path: Path):
    path = place(tmp_path / "loras")
    sidecar.annotate(path.with_name(path.name + ".json"), NOTE)

    erase.erase(path)

    assert list((tmp_path / "loras").iterdir()) == []


# --- through the page ---------------------------------------------------------


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
        test.settings = settings  # type: ignore[attr-defined]
        yield test


def finished(client, record: bool = True, **overrides):
    path = place(client.library / "loras", record=record)
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


def test_the_page_writes_a_note_to_both_places(client):
    task = finished(client)

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE}).json()

    assert body == {"ok": True, "note": NOTE, "record_written": False}
    assert written(client.library / "loras" / f"{MODEL}.json")["note"] == NOTE
    # The copy the card draws and the filter box searches.
    assert client.database.get(task.id).note == NOTE


def test_a_file_with_no_record_gets_one_written_to_hold_the_note(client):
    """Sidecars turned off is a choice about clutter, not a choice to lose the one thing
    that cannot be fetched again."""
    task = finished(client, record=False)

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE}).json()

    assert body["record_written"] is True
    stored = written(client.library / "loras" / f"{MODEL}.json")
    assert stored["note"] == NOTE
    assert stored["filename"] == MODEL, "a real record, not a stub holding one field"
    # The compatibility files and the trigger .txt are a separate choice, and adding a note
    # is not the moment to overrule it.
    assert sorted(p.name for p in (client.library / "loras").iterdir()) == [
        MODEL, f"{MODEL}.json",
    ]


def test_clearing_a_note_that_was_never_written_writes_nothing(client):
    """Writing a whole record to hold a null would be an odd way to do nothing."""
    task = finished(client, record=False)

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": ""}).json()

    assert body == {"ok": True, "note": None, "record_written": False}
    assert not (client.library / "loras" / f"{MODEL}.json").exists()


def test_a_note_can_be_taken_off_again(client):
    task = finished(client)
    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": "  "}).json()

    assert body["note"] is None
    assert written(client.library / "loras" / f"{MODEL}.json")["note"] is None
    assert client.database.get(task.id).note is None


def test_the_note_outlives_the_queue_row(client):
    """The whole reason it is not kept in the database: clearing the list is the button
    people press after an evening of downloading, and taking a row out of the history the
    one they press when tidying that."""
    task = finished(client)
    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    client.post("/api/tasks/clear")
    assert client.database.get(task.id).archived, "off the list, into the history"

    client.delete(f"/api/history/{task.id}")

    assert client.database.get(task.id) is None
    assert written(client.library / "loras" / f"{MODEL}.json")["note"] == NOTE


def test_a_collected_record_holds_the_note_in_its_mirror(client):
    client.settings.sidecar_dir = str(client.library.parent / "meta")
    task = finished(client, record=False)

    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    collected = client.library.parent / "meta" / "loras" / f"{MODEL}.json"
    assert written(collected)["note"] == NOTE
    assert not (client.library / "loras" / f"{MODEL}.json").exists()


# --- before the file exists ---------------------------------------------------

IDENTITY = {"provider": "direct", "ref": {"url": "https://example.com/m"}}


def queued(client, **overrides):
    """A download that has not landed: no file, no record, nowhere yet to keep a note."""
    values = {
        "state": "running",
        "source": "https://example.com/model.safetensors",
        "provider": "direct",
        "identity": IDENTITY,
        "filename": MODEL,
        "category": "lora",
    }
    values.update(overrides)
    return client.database.add(**values)


def test_a_download_that_has_not_landed_yet_takes_a_note(client):
    """What you know about a model is in your head while you are queueing it, not an hour
    later when the bytes stop — so the note is taken then and waits in the row."""
    task = queued(client)

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE}).json()

    assert body == {"ok": True, "note": NOTE, "record_written": False}
    assert client.database.get(task.id).note == NOTE
    # Nothing was invented on disk to hold it: there is no file yet to put a record beside.
    assert not (client.library / "loras").exists()


def test_a_note_on_a_download_that_has_not_landed_can_be_taken_off_again(client):
    task = queued(client)
    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": "   "}).json()

    assert body == {"ok": True, "note": None, "record_written": False}
    assert client.database.get(task.id).note is None


async def test_the_note_written_while_it_downloaded_is_in_the_record_it_lands_with(client):
    """And it is read from the row, not from the snapshot the download started with: the
    note was written while the bytes were moving, long after that copy was taken."""
    task = queued(client)
    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})
    path = place(client.library / "loras", record=False)

    await client.app.state.manager._write_sidecar(path, task, _identity())

    stored = written(client.library / "loras" / f"{MODEL}.json")
    assert stored["note"] == NOTE
    assert stored["filename"] == MODEL, "a real record, not a stub holding one field"


async def test_records_turned_off_still_leaves_the_note_somewhere_to_land(client):
    """Records off is a choice about clutter, not a choice to lose the one field that
    cannot be fetched again."""
    client.settings.write_sidecars = False
    task = queued(client)
    client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})
    path = place(client.library / "loras", record=False)

    await client.app.state.manager._write_sidecar(path, task, _identity())

    assert written(client.library / "loras" / f"{MODEL}.json")["note"] == NOTE
    # Only the record. The compatibility files and the trigger .txt are a separate choice.
    assert sorted(p.name for p in (client.library / "loras").iterdir()) == [
        MODEL, f"{MODEL}.json",
    ]


async def test_records_turned_off_and_nothing_written_stays_off(client):
    """No note, no record: the checkbox means what it says for every other download."""
    client.settings.write_sidecars = False
    task = queued(client)
    path = place(client.library / "loras", record=False)

    await client.app.state.manager._write_sidecar(path, task, _identity())

    assert sorted(p.name for p in (client.library / "loras").iterdir()) == [MODEL]


def _identity():
    from sfd.core.types import FileIdentity

    return FileIdentity(provider="direct", ref={"url": "https://example.com/m"})


def test_a_note_about_a_file_that_is_gone_is_refused(client):
    task = finished(client)
    (client.library / "loras" / MODEL).unlink()

    response = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    assert response.status_code == 404


def test_a_note_longer_than_the_record_should_carry_is_refused(client):
    task = finished(client)

    response = client.post(f"/api/tasks/{task.id}/note", json={"note": "x" * 4001})

    assert response.status_code == 422
    assert client.database.get(task.id).note is None


def test_annotating_is_refused_while_the_file_is_being_moved(client):
    import threading

    task = finished(client)
    client.app.state.manager._moves[task.id] = threading.Event()

    response = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    assert response.status_code == 409


def test_an_identity_too_old_to_rebuild_a_link_still_takes_a_note(client):
    """The record has to be written for the note to have anywhere to live, and it is built
    from the stored identity. A ref this code can no longer turn into a URL once raised a
    KeyError from inside, which the endpoint reported as a task that did not exist."""
    task = finished(
        client, record=False,
        provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 1}},
    )

    body = client.post(f"/api/tasks/{task.id}/note", json={"note": NOTE})

    assert body.status_code == 200, body.text
    stored = written(client.library / "loras" / f"{MODEL}.json")
    assert stored["note"] == NOTE
    assert stored["source"]["url"] is None, "no link, rather than no record"


def test_a_task_that_is_not_there_is_a_404(client):
    assert client.post("/api/tasks/9999/note", json={"note": NOTE}).status_code == 404
