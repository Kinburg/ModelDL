"""Moving a model the classifier filed in the wrong place.

What makes this worth its own module is not the move — it is everything named after the
model that has to go with it. A checkpoint that arrives in its new folder without its
preview and its trigger words has been half-moved, and nothing about the folder shows it.
"""

from __future__ import annotations

import asyncio
import errno
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.jobs.manager import Manager
from sfd.library import relocate
from sfd.settings import Settings
from sfd.web.app import create_app

MODEL = "mystery.safetensors"


def place(folder: Path, name: str = MODEL, sidecars: bool = True) -> Path:
    """A model with the full set of files a finished download leaves beside it."""
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
    target = tmp_path / "checkpoints"

    result = relocate.move(path, target)

    assert result.path == target / MODEL
    assert result.path.read_bytes() == b"weights"
    assert not result.failed
    landed = sorted(p.name for p in target.iterdir())
    assert landed == [
        "mystery.civitai.info", "mystery.preview.png",
        MODEL, f"{MODEL}.json", "mystery.txt",
    ]
    # And nothing of the model is left in the folder it came from.
    assert not any(p.name.startswith("mystery") for p in (tmp_path / "loras").iterdir())


def test_a_model_of_that_name_already_there_stops_everything(tmp_path: Path):
    """The one outcome worse than a misfiled model is a silently replaced one."""
    path = place(tmp_path / "loras")
    target = tmp_path / "checkpoints"
    existing = place(target)
    existing.write_bytes(b"the other one")

    with pytest.raises(FileExistsError):
        relocate.move(path, target)

    assert path.is_file(), "the move was refused, so the original must still be there"
    assert (target / MODEL).read_bytes() == b"the other one"
    assert (tmp_path / "loras" / f"{MODEL}.json").is_file(), "no sidecar moved either"


def test_a_stranded_sidecar_is_named_rather_than_hidden(tmp_path: Path):
    """The model is across and the folder looks right; only the report says otherwise."""
    path = place(tmp_path / "loras")
    target = tmp_path / "checkpoints"
    target.mkdir()
    (target / "mystery.preview.png").write_bytes(b"a different picture")

    result = relocate.move(path, target)

    assert result.path == target / MODEL
    assert [p.name for p, _ in result.failed] == ["mystery.preview.png"]
    assert (tmp_path / "loras" / "mystery.preview.png").is_file()
    assert (target / "mystery.txt").is_file(), "the others still went"


def test_a_collected_record_follows_the_mirrored_tree(tmp_path: Path):
    """With `sidecar_dir` set the record is deliberately not beside the model, and dumping
    it there on the way past would undo the setting."""
    library = tmp_path / "library"
    collected = tmp_path / "meta"
    path = place(library / "loras", sidecars=False)
    record = collected / "loras" / f"{MODEL}.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"schema": 1}', encoding="utf-8")

    result = relocate.move(
        path, library / "checkpoints", sidecar_dir=collected, library_root=library
    )

    assert result.companions == [collected / "checkpoints" / f"{MODEL}.json"]
    assert not record.exists()
    assert not (library / "checkpoints" / f"{MODEL}.json").exists()


def test_moving_a_file_where_it_already_is_changes_nothing(tmp_path: Path):
    path = place(tmp_path / "loras")

    result = relocate.move(path, tmp_path / "loras")

    assert result.unchanged
    assert path.is_file()


def test_a_file_that_is_gone_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        relocate.move(tmp_path / "loras" / MODEL, tmp_path / "checkpoints")


# --- the drive boundary -----------------------------------------------------


def across_volumes(monkeypatch, blocked: Path) -> None:
    """Make a rename of `blocked` fail the way one across drives does, and only that one.

    There is no second drive in a test, and the branch that matters is reached by an errno
    rather than by a path, so this is the honest way to get at it.
    """
    real = os.replace

    def refuse(source, target, *args, **kwargs):
        if Path(source) == blocked:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(relocate.os, "replace", refuse)


def test_a_move_across_drives_copies_and_reports_as_it_goes(tmp_path: Path, monkeypatch):
    path = place(tmp_path / "loras")
    path.write_bytes(b"w" * (relocate.COPY_CHUNK * 2 + 5))
    across_volumes(monkeypatch, path)
    seen: list[tuple[int, int]] = []

    result = relocate.move(
        path, tmp_path / "elsewhere", progress=lambda done, total: seen.append((done, total))
    )

    assert result.path.read_bytes() == b"w" * (relocate.COPY_CHUNK * 2 + 5)
    assert not path.exists(), "a copy that leaves the original behind is not a move"
    assert seen and seen[-1][0] == seen[-1][1], "the last report has to say it finished"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert not list((tmp_path / "elsewhere").glob("*.moving")), "no fragment left behind"


def test_a_rename_reports_nothing_because_there_is_nothing_to_report(tmp_path: Path):
    """Within a drive the move is instant, and a progress bar that appears and vanishes in
    the same frame is worse than none."""
    path = place(tmp_path / "loras")
    seen = []

    relocate.move(path, tmp_path / "checkpoints", progress=lambda *a: seen.append(a))

    assert seen == []


def test_a_copy_that_fails_partway_leaves_neither_a_fragment_nor_a_hole(
    tmp_path: Path, monkeypatch
):
    """The moment of maximum danger: the bytes are written but the original is not yet
    deleted. Whatever goes wrong here, the model must still exist exactly once."""
    path = place(tmp_path / "loras")
    across_volumes(monkeypatch, path)
    monkeypatch.setattr(
        relocate.shutil, "copystat",
        lambda *a, **k: (_ for _ in ()).throw(OSError("the disk filled up")),
    )

    with pytest.raises(OSError):
        relocate.move(path, tmp_path / "elsewhere")

    assert path.read_bytes() == b"weights", "the original is the only copy, and it is intact"
    assert not list((tmp_path / "elsewhere").iterdir()), "and nothing was left at the target"


# --- through the page -------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    library = tmp_path / "library"
    (library / "checkpoints").mkdir(parents=True)
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


def test_the_page_can_move_a_finished_download(client):
    task = finished(client)

    body = client.post(f"/api/tasks/{task.id}/move", json={"folder": "checkpoints"}).json()

    assert body["ok"] and body["moved"] == 4 and not body["failed"]
    moved = client.library / "checkpoints" / MODEL
    assert moved.is_file()
    # The queue has to learn the new path, or Folder opens the one it left.
    assert client.database.get(task.id).dest == str(moved)


def test_the_card_stops_explaining_a_placement_nobody_kept(client):
    task = finished(client)

    client.post(f"/api/tasks/{task.id}/move", json={"folder": "checkpoints"})

    after = client.database.get(task.id)
    assert after.category == "checkpoint", "the folder names a kind, so it is now that kind"
    assert after.confidence == "high"
    assert "by hand" in after.reason


def test_a_folder_outside_the_library_is_refused(client):
    task = finished(client)

    for escape in ("../elsewhere", "C:\\Windows", "/etc", ""):
        response = client.post(f"/api/tasks/{task.id}/move", json={"folder": escape})
        assert response.status_code == 400, escape

    assert client.database.get(task.id).dest == str(client.library / "loras" / MODEL)


def test_only_a_finished_download_can_be_moved(client):
    task = finished(client, state="running")

    response = client.post(f"/api/tasks/{task.id}/move", json={"folder": "checkpoints"})

    assert response.status_code == 409
    assert (client.library / "loras" / MODEL).is_file()


def test_an_occupied_destination_is_reported_not_overwritten(client):
    task = finished(client)
    place(client.library / "checkpoints").write_bytes(b"the other one")

    response = client.post(f"/api/tasks/{task.id}/move", json={"folder": "checkpoints"})

    assert response.status_code == 409
    assert (client.library / "checkpoints" / MODEL).read_bytes() == b"the other one"


def test_remembering_the_folder_makes_it_the_home_of_that_kind(client):
    task = finished(client)

    client.post(
        f"/api/tasks/{task.id}/move", json={"folder": "checkpoints", "remember": True}
    )

    assert client.settings.layout_overrides["checkpoint"] == str(
        client.library / "checkpoints"
    )


# --- the system's own dialog ------------------------------------------------


def picks(monkeypatch, folder: Path | None):
    """Answer the folder dialog without one, since a modal window has nobody to answer it."""
    import sfd.desktop

    monkeypatch.setattr(
        sfd.desktop, "pick_system_folder", lambda initial="", window=None:
        str(folder) if folder is not None else None
    )


def test_the_dialog_can_land_a_model_outside_the_library(client, tmp_path, monkeypatch):
    """The whole point of the second endpoint: another drive is not under the library root,
    so the confined move cannot express it and refusing would be the wrong answer."""
    task = finished(client)
    elsewhere = tmp_path / "another-drive" / "checkpoints"
    picks(monkeypatch, elsewhere)

    body = client.post(f"/api/tasks/{task.id}/move-anywhere", json={}).json()

    assert body["ok"] and body["moved"] == 4
    assert (elsewhere / MODEL).read_bytes() == b"weights"
    assert client.database.get(task.id).dest == str(elsewhere / MODEL)


def test_dismissing_the_dialog_moves_nothing(client, monkeypatch):
    task = finished(client)
    picks(monkeypatch, None)

    body = client.post(f"/api/tasks/{task.id}/move-anywhere", json={}).json()

    assert body == {"ok": False, "cancelled": True}
    assert (client.library / "loras" / MODEL).is_file()


def test_the_request_cannot_name_the_folder_itself(client, tmp_path, monkeypatch):
    """What makes an unconfined destination safe. A caller that reaches this API without a
    person behind it can open a dialog; it cannot say where the file goes."""
    task = finished(client)
    chosen = tmp_path / "chosen-in-the-dialog"
    picks(monkeypatch, chosen)

    client.post(
        f"/api/tasks/{task.id}/move-anywhere",
        json={"folder": str(tmp_path / "asked-for-in-the-request"), "remember": False},
    )

    assert (chosen / MODEL).is_file()
    assert not (tmp_path / "asked-for-in-the-request").exists()


def test_an_outside_folder_can_still_become_the_home_of_its_kind(client, tmp_path, monkeypatch):
    """A drive that holds nothing but checkpoints is the usual reason to move one there."""
    task = finished(client)
    chosen = tmp_path / "another-drive" / "checkpoints"
    picks(monkeypatch, chosen)

    body = client.post(
        f"/api/tasks/{task.id}/move-anywhere", json={"remember": True}
    ).json()

    assert body["remembered"] == "checkpoint"
    assert client.settings.layout_overrides["checkpoint"] == str(chosen)


def test_a_folder_naming_no_kind_changes_no_mapping(client, tmp_path, monkeypatch):
    task = finished(client)
    picks(monkeypatch, tmp_path / "another-drive" / "Krea 2")

    body = client.post(
        f"/api/tasks/{task.id}/move-anywhere", json={"remember": True}
    ).json()

    assert body["remembered"] is None
    assert "checkpoint" not in client.settings.layout_overrides


def test_the_dialog_does_not_open_for_a_download_still_running(client, monkeypatch):
    """Checked before the picker, or a modal window appears only to be told it was pointless."""
    task = finished(client, state="running")
    opened = []
    import sfd.desktop
    monkeypatch.setattr(
        sfd.desktop, "pick_system_folder",
        lambda initial="", window=None: opened.append(initial),
    )

    response = client.post(f"/api/tasks/{task.id}/move-anywhere", json={})

    assert response.status_code == 409
    assert opened == []


# --- changing your mind halfway ---------------------------------------------


def test_a_stopped_copy_leaves_the_file_where_it_was(tmp_path: Path, monkeypatch):
    path = place(tmp_path / "loras")
    path.write_bytes(b"w" * (relocate.COPY_CHUNK * 3))
    across_volumes(monkeypatch, path)
    stop = threading.Event()

    # Set from inside the copy, which is where a person would set it from outside: after
    # some of the file is across and the rest is not.
    def halfway(copied, total):
        if copied >= relocate.COPY_CHUNK:
            stop.set()

    with pytest.raises(relocate.Cancelled):
        relocate.move(path, tmp_path / "elsewhere", progress=halfway, stop=stop)

    assert path.stat().st_size == relocate.COPY_CHUNK * 3, "the original is whole"
    assert not list((tmp_path / "elsewhere").iterdir()), "and the half-copy is gone"
    assert (tmp_path / "loras" / f"{MODEL}.json").is_file(), "sidecars never started"


def test_stopping_a_move_that_is_only_a_rename_does_not_break_it(tmp_path: Path):
    """Within a drive there is no copy loop to look at the flag, and the move is already
    over. Asking it to stop must not turn a completed move into a failure."""
    path = place(tmp_path / "loras")
    stop = threading.Event()
    stop.set()

    result = relocate.move(path, tmp_path / "checkpoints", stop=stop)

    assert result.path.is_file()


async def test_the_manager_hands_the_stop_through_to_the_copy(tmp_path: Path, monkeypatch):
    """The wiring that cannot be seen from either side alone: a request on the event loop
    reaching a copy that is already running on a thread, and the registry cleaning up."""
    library = tmp_path / "library"
    settings = Settings(
        library_root=str(library), download_dir=str(tmp_path / "downloads"),
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    path = place(library / "loras")
    task = database.add(
        state="done", source="https://example.com/m", provider="direct",
        identity={"provider": "direct", "ref": {"url": "https://example.com/m"}},
        filename=MODEL, category="lora", dest=str(path),
    )

    running = threading.Event()

    def blocks_until_told(*args, stop=None, **kwargs):
        running.set()
        assert stop is not None and stop.wait(5), "the stop never reached the thread"
        raise relocate.Cancelled("stopped")

    monkeypatch.setattr(relocate, "move", blocks_until_told)
    events = manager.subscribe()

    moving = asyncio.create_task(manager.move(task.id, library / "checkpoints"))
    await asyncio.to_thread(running.wait, 5)

    assert manager.stop_move(task.id) is True
    with pytest.raises(relocate.Cancelled):
        await moving

    assert manager.stop_move(task.id) is False, "the registry let go of the finished move"
    assert database.get(task.id).dest == str(path), "a stopped move changes no record"
    assert {"type": "moved", "id": task.id} == events.get_nowait()
    database.close()


def test_there_is_nothing_to_stop_when_nothing_is_moving(client):
    task = finished(client)

    body = client.post(f"/api/tasks/{task.id}/move/stop").json()

    assert body == {"ok": False}


async def test_one_file_cannot_be_moved_twice_at_once(tmp_path: Path, monkeypatch):
    """Two copies of one file racing towards the same destination is not a state worth
    reasoning about, so the second ask is refused rather than queued behind the first."""
    library = tmp_path / "library"
    settings = Settings(
        library_root=str(library), download_dir=str(tmp_path / "downloads"),
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    manager = Manager(settings, database)
    path = place(library / "loras")
    task = database.add(
        state="done", source="https://example.com/m", provider="direct",
        identity={"provider": "direct", "ref": {"url": "https://example.com/m"}},
        filename=MODEL, category="lora", dest=str(path),
    )

    running, release = threading.Event(), threading.Event()

    def blocks(*args, stop=None, **kwargs):
        running.set()
        release.wait(5)
        return relocate.Move(path=library / "checkpoints" / MODEL)

    monkeypatch.setattr(relocate, "move", blocks)
    first = asyncio.create_task(manager.move(task.id, library / "checkpoints"))
    await asyncio.to_thread(running.wait, 5)

    with pytest.raises(ValueError, match="already being moved"):
        await manager.move(task.id, library / "loras")

    release.set()
    await first
    database.close()
