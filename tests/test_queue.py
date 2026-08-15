"""The queue: persistence, state transitions, and the HTTP surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs import db as states
from sfd.jobs.db import Database
from sfd.settings import Settings
from sfd.web.app import create_app


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "queue.db")


def add(database: Database, **overrides):
    values = {
        "source": "https://example.com/model.safetensors",
        "provider": "direct",
        "identity": {"provider": "direct", "ref": {"url": "https://example.com/model.safetensors"}},
        "filename": "model.safetensors",
        "size": 1000,
        "dest": "/tmp/model.safetensors",
    }
    values.update(overrides)
    return database.add(**values)


# --- storage ----------------------------------------------------------------


def test_a_task_round_trips(database: Database):
    task = add(database)
    assert task is not None
    loaded = database.get(task.id)
    assert loaded.filename == "model.safetensors"
    assert loaded.identity["ref"]["url"] == "https://example.com/model.safetensors"
    assert loaded.state == states.PENDING


def test_the_same_file_is_not_queued_twice(database: Database):
    """Adding a link again should adopt the existing task, not race a second download
    onto the same path."""
    assert add(database) is not None
    assert add(database) is None
    assert len(database.list()) == 1


def test_different_files_coexist(database: Database):
    add(database)
    other = add(
        database,
        identity={"provider": "direct", "ref": {"url": "https://example.com/other.bin"}},
        filename="other.bin",
    )
    assert other is not None
    assert len(database.list()) == 2


def test_claim_hands_out_each_task_once(database: Database):
    first = add(database)
    second = add(
        database,
        identity={"provider": "direct", "ref": {"url": "https://example.com/b"}},
        filename="b",
    )
    claimed = {database.claim_next().id, database.claim_next().id}
    assert claimed == {first.id, second.id}
    assert database.claim_next() is None


def test_claim_skips_blocked_tasks(database: Database):
    add(database, state=states.BLOCKED)
    assert database.claim_next() is None


def test_a_task_running_when_the_process_died_becomes_pending_again(tmp_path: Path):
    """The point of persisting the queue: a crash must not lose a download."""
    path = tmp_path / "queue.db"
    first = Database(path)
    task = add(first)
    first.update(task.id, state=states.RUNNING, downloaded=500)
    first.close()

    reopened = Database(path)
    revived = reopened.get(task.id)
    assert revived.state == states.PENDING
    assert revived.downloaded == 500, "progress must survive the restart"


async def test_reconcile_reads_progress_back_from_the_part_file(tmp_path: Path):
    """After a crash the queue's counter is stale; the file's own state file is not.

    A 24 GB download shown as untouched reads exactly like the restart-from-zero this whole
    project exists to prevent, so the number has to come from the data.
    """
    from sfd.core.chunks import ChunkMap
    from sfd.core.state import PartState
    from sfd.core.types import FileIdentity
    from sfd.jobs.manager import Manager

    dest = tmp_path / "model.safetensors"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(bytes(4000))

    chunks = ChunkMap(size=4000, chunk_size=1000)
    chunks.chunks[0].done = 1000
    chunks.chunks[1].done = 400
    identity = FileIdentity(provider="direct", ref={"url": "https://example.com/m"})
    PartState.create(part, identity, _info(4000), chunks).flush(force=True)

    database = Database(tmp_path / "queue.db")
    task = add(database, dest=str(dest), size=4000)
    assert database.get(task.id).downloaded == 0

    Manager(Settings(), database).reconcile()
    assert database.get(task.id).downloaded == 1400


async def test_reconcile_notices_a_download_that_finished_during_the_gap(tmp_path: Path):
    from sfd.jobs.manager import Manager

    dest = tmp_path / "model.safetensors"
    dest.write_bytes(bytes(4000))

    database = Database(tmp_path / "queue.db")
    task = add(database, dest=str(dest), size=4000)
    Manager(Settings(), database).reconcile()

    updated = database.get(task.id)
    assert updated.state == states.DONE
    assert updated.downloaded == 4000


def _info(size: int):
    from sfd.core.types import RemoteFileInfo

    return RemoteFileInfo(filename="model.safetensors", size=size)


def test_clearing_finished_leaves_the_rest(database: Database):
    done = add(database, state=states.DONE)
    add(database, identity={"provider": "direct", "ref": {"url": "https://x/b"}}, filename="b")
    assert database.clear([states.DONE]) == 1
    assert [t.id for t in database.list()] != [done.id]
    assert len(database.list()) == 1


def test_dragging_changes_what_runs_next(database: Database):
    first = add(database)
    second = add(database, identity={"provider": "direct", "ref": {"url": "b"}}, filename="b")
    third = add(database, identity={"provider": "direct", "ref": {"url": "c"}}, filename="c")

    database.reorder([third.id, first.id, second.id])
    assert [t.id for t in database.list()] == [third.id, first.id, second.id]
    assert database.claim_next().id == third.id


def test_average_speed_uses_bytes_really_fetched(database: Database):
    """Dividing the file size by the elapsed time once reported 364 MB/s for a download
    that never happened, because the file was already on disk."""
    task = add(database, size=100_000_000)
    database.update(task.id, started_at=1000.0, finished_at=1010.0, transferred=50_000_000)

    loaded = database.get(task.id)
    assert loaded.duration == 10.0
    assert loaded.average_speed == 5_000_000


def test_a_skipped_file_reports_no_speed_rather_than_a_fictional_one(database: Database):
    task = add(database, size=100_000_000)
    database.update(task.id, started_at=1000.0, finished_at=1002.0, transferred=0)
    assert database.get(task.id).average_speed is None


def test_timing_is_absent_until_a_run_finishes(database: Database):
    task = add(database)
    assert database.get(task.id).duration is None
    database.update(task.id, started_at=1000.0)
    assert database.get(task.id).duration is None


def test_claiming_stamps_the_start_time(database: Database):
    add(database)
    claimed = database.claim_next()
    assert database.get(claimed.id).started_at is not None


def test_an_old_database_gains_the_new_columns(tmp_path: Path):
    """A queue from an earlier build must be migrated, not discarded."""
    import sqlite3

    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,"
        " updated_at REAL NOT NULL, state TEXT NOT NULL, source TEXT NOT NULL,"
        " label TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL, identity TEXT NOT NULL,"
        " filename TEXT NOT NULL DEFAULT '', size INTEGER, sha256 TEXT,"
        " dest TEXT NOT NULL DEFAULT '', downloaded INTEGER NOT NULL DEFAULT 0,"
        " category TEXT, confidence TEXT, reason TEXT, disagreement TEXT, base_model TEXT,"
        " meta TEXT NOT NULL DEFAULT '{}', error TEXT);"
        "INSERT INTO tasks (created_at, updated_at, state, source, provider, identity,"
        " filename, downloaded) VALUES (1, 1, 'pending', 'src', 'direct', '{}', 'old.bin', 42);"
    )
    connection.commit()
    connection.close()

    database = Database(path)
    task = database.list()[0]
    assert task.filename == "old.bin"
    assert task.downloaded == 42, "the existing queue must survive the migration"
    assert task.transferred == 0

    # Positions are backfilled, so anything added afterwards queues behind the old rows
    # instead of jumping ahead of them.
    assert task.position == task.id
    fresh = add(database, identity={"provider": "direct", "ref": {"url": "new"}}, filename="n")
    assert fresh.position > task.position
    assert [t.id for t in database.list()] == [task.id, fresh.id]


def test_progress_fraction_is_reported(database: Database):
    task = add(database, size=1000)
    database.update(task.id, downloaded=250)
    assert database.get(task.id).to_json()["fraction"] == 0.25


# --- HTTP -------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    # A settings file of its own. Without one, a test that saves settings rewrites the
    # user's real settings.json — which is how a live install once ended up pointing its
    # library at a pytest temp directory.
    settings = Settings(
        download_dir=str(tmp_path / "downloads"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database)) as test_client:
        test_client.database = database  # type: ignore[attr-defined]
        yield test_client


def test_the_page_is_served(client):
    assert client.get("/").status_code == 200


def test_listing_starts_empty(client):
    assert client.get("/api/tasks").json() == {"tasks": []}


def test_adding_nonsense_reports_why(client):
    response = client.post("/api/tasks", json={"source": "   "})
    assert response.status_code == 400


# Once a task is released, a worker claims it within milliseconds and may already have
# failed against the unreachable test URL. What these assert is that the command took
# effect — the task is no longer parked — not which instant it was caught in.
RELEASED = {states.PENDING, states.RUNNING, states.FAILED, states.DONE}


def test_pause_and_resume_move_the_state(client):
    task = add(client.database)
    assert client.post(f"/api/tasks/{task.id}/pause").status_code == 200
    assert client.database.get(task.id).state == states.PAUSED
    assert client.post(f"/api/tasks/{task.id}/resume").status_code == 200
    assert client.database.get(task.id).state in RELEASED


def test_confirming_a_blocked_task_can_correct_the_category(client):
    task = add(client.database, state=states.BLOCKED, category="checkpoint")
    response = client.post(
        f"/api/tasks/{task.id}/confirm", json={"category": "diffusion_model"}
    )
    assert response.status_code == 200
    updated = client.database.get(task.id)
    assert updated.category == "diffusion_model"
    assert updated.confidence == "high"
    assert updated.state in RELEASED, "confirming must release the task for download"


def test_an_unknown_category_is_rejected(client):
    task = add(client.database, state=states.BLOCKED)
    response = client.post(f"/api/tasks/{task.id}/confirm", json={"category": "nonsense"})
    assert response.status_code == 400


def test_start_all_releases_only_what_is_waiting(client):
    """Collecting links first, downloading later.

    Blocked tasks stay blocked: they are waiting on a decision about where the file belongs,
    and releasing them wholesale would file models by a guess the classifier already said it
    was not sure about.
    """
    held = [
        add(client.database, state=states.PAUSED,
            identity={"provider": "direct", "ref": {"url": f"held-{i}"}}, filename=f"h{i}")
        for i in range(3)
    ]
    unsure = add(client.database, state=states.BLOCKED,
                 identity={"provider": "direct", "ref": {"url": "unsure"}}, filename="u")
    finished = add(client.database, state=states.DONE,
                   identity={"provider": "direct", "ref": {"url": "done"}}, filename="d")

    assert client.post("/api/tasks/start-all").json()["released"] == 3
    for task in held:
        assert client.database.get(task.id).state in RELEASED
    assert client.database.get(unsure.id).state == states.BLOCKED
    assert client.database.get(finished.id).state == states.DONE


def test_start_all_on_an_empty_queue_is_harmless(client):
    assert client.post("/api/tasks/start-all").json()["released"] == 0


def test_auto_start_is_on_by_default():
    assert Settings().auto_start is True


def test_the_stored_record_can_be_read_back_through_the_page(client, tmp_path: Path):
    """Records collected into their own directory are tidy and hard to find.

    The card has to be able to fetch its own, or the trigger words end up somewhere the
    person who wanted them cannot reach.
    """
    from sfd.library.categories import Category
    from sfd.library.classify import Verdict
    from sfd.library.sidecar import Record, write

    library = tmp_path / "models"
    records = tmp_path / "records"
    model = library / "loras" / "style.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    write(
        model,
        Verdict(Category.LORA, "high", "training metadata names networks.lora",
                base_model="Pony"),
        Record(filename=model.name, provider="civitai", sha256="ab" * 32,
               meta={"trained_words": ["bl00m"], "model_id": 1, "version_id": 2}),
        sidecar_dir=records,
        library_root=library,
    )

    client.put("/api/settings", json={
        "library_root": str(library), "sidecar_dir": str(records),
    })
    task = add(client.database, dest=str(model), state=states.DONE)

    body = client.get(f"/api/tasks/{task.id}/record").json()
    assert body["record"]["usage"]["trigger_words"] == ["bl00m"]
    assert body["record"]["usage"]["base_model"] == "Pony"
    assert body["path"].endswith("style.safetensors.json")
    assert str(records) in body["path"]


def test_asking_for_a_record_that_was_never_written(client, tmp_path: Path):
    task = add(client.database, dest=str(tmp_path / "nothing.safetensors"))
    assert client.get(f"/api/tasks/{task.id}/record").status_code == 404


def test_asking_for_the_record_of_a_task_that_does_not_exist(client):
    assert client.get("/api/tasks/9999/record").status_code == 404


def test_removing_a_task(client):
    task = add(client.database)
    assert client.delete(f"/api/tasks/{task.id}").status_code == 200
    assert client.database.get(task.id) is None


def test_saving_settings_writes_only_to_its_own_file(client, tmp_path: Path):
    """Settings remember where they came from.

    Otherwise anything that saves — including a test — rewrites whatever settings.json
    happens to be in the working directory.
    """
    client.put("/api/settings", json={"library_root": str(tmp_path / "models")})
    assert (tmp_path / "settings.json").exists()
    assert not Path("settings.json").resolve().samefile(tmp_path / "settings.json") \
        if Path("settings.json").exists() else True


def test_the_settings_path_cannot_be_changed_through_the_api(client, tmp_path: Path):
    elsewhere = tmp_path / "hijacked.json"
    client.put("/api/settings", json={"_path": str(elsewhere), "connections": 3})
    client.put("/api/settings", json={"connections": 4})
    assert not elsewhere.exists()


def test_the_settings_path_is_not_sent_to_the_browser(client):
    assert "_path" not in client.get("/api/settings").json()["settings"]


def test_settings_never_send_the_tokens_back(client):
    client.put("/api/settings", json={"hf_token": "hf_secret", "connections": 8})
    body = client.get("/api/settings").json()["settings"]
    assert body["hf_token"] == ""
    assert body["hf_token_set"] is True
    assert body["connections"] == 8


def test_a_blank_token_field_does_not_erase_a_saved_one(client):
    """An empty box in the form means "leave it alone", not "forget my key"."""
    client.put("/api/settings", json={"civitai_token": "key"})
    client.put("/api/settings", json={"civitai_token": "", "connections": 4})
    body = client.get("/api/settings").json()["settings"]
    assert body["civitai_token_set"] is True
    assert body["connections"] == 4


def test_layout_endpoint_reports_the_mapping(client, tmp_path: Path):
    root = tmp_path / "models"
    (root / "loras").mkdir(parents=True)
    (root / "unet").mkdir()
    (root / "diffusion_models").mkdir()
    (root / "diffusion_models" / "m.safetensors").write_bytes(b"x")

    client.put("/api/settings", json={"library_root": str(root)})
    layout = client.get("/api/layout").json()
    assert layout["paths"]["lora"] == str(root / "loras")
    assert layout["paths"]["diffusion_model"] == str(root / "diffusion_models")
    assert layout["ambiguities"]["diffusion_model"] == ["unet"]
    assert layout["exists"]["lora"] is True
