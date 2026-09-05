"""The queue: persistence, state transitions, and the HTTP surface."""

from __future__ import annotations

import asyncio
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


def test_new_tasks_queue_behind_what_is_already_waiting(database: Database):
    first = add(database)
    second = add(database, identity={"provider": "direct", "ref": {"url": "b"}}, filename="b")
    assert [t.id for t in database.list()] == [first.id, second.id]


def test_queueing_at_the_top_jumps_the_line(database: Database):
    """With `queue_position: top`, the link just pasted is the one being waited on."""
    waiting = add(database)
    urgent = add(
        database,
        identity={"provider": "direct", "ref": {"url": "b"}},
        filename="b",
        position_mode=states.TOP,
    )
    assert [t.id for t in database.list()] == [urgent.id, waiting.id]
    assert database.claim_next().id == urgent.id


def test_a_batch_queued_at_the_top_keeps_its_own_order(database: Database):
    """One link expands into several files. Taking the top slot once per file would put
    each new one above the last and land the whole set reversed."""
    waiting = add(database)
    names = ["a.bin", "b.bin", "c.bin"]
    batch = [
        add(
            database,
            identity={"provider": "direct", "ref": {"url": name}},
            filename=name,
            position=position,
        )
        for name, position in zip(names, database.reserve_positions(len(names), states.TOP))
    ]
    assert [t.filename for t in database.list()] == [*names, "model.safetensors"]
    assert database.claim_next().id == batch[0].id


async def test_the_setting_decides_which_end_a_pasted_link_joins(tmp_path: Path, monkeypatch):
    """The end-to-end path the setting travels: settings → manager → the queue's order."""
    from sfd.jobs import manager as manager_module
    from sfd.jobs.manager import Manager
    from sfd.providers.direct import DirectProvider, make_identity
    from sfd.providers.registry import Item, Resolution

    async def fake_expand(text, client, **kwargs):
        return Resolution(
            provider=DirectProvider(),
            items=[
                Item(identity=make_identity(f"https://example.com/{name}"), filename=name)
                for name in ("one.bin", "two.bin")
            ],
            label="pack",
        )

    monkeypatch.setattr(manager_module, "expand", fake_expand)

    database = Database(tmp_path / "queue.db")
    add(database)
    settings = Settings(
        download_dir=str(tmp_path / "downloads"),
        queue_position=states.TOP,
        _path=str(tmp_path / "settings.json"),
    )
    await Manager(settings, database).add("https://example.com/pack")

    assert [t.filename for t in database.list()] == ["one.bin", "two.bin", "model.safetensors"]


def _manager(database: Database, **settings):
    from sfd.jobs.manager import Manager

    return Manager(Settings(**settings), database)


def test_a_failure_worth_retrying_books_another_attempt(database: Database):
    """The queue this is written for is one left running overnight: a router rebooting at
    3am should cost minutes, not the rest of the night."""
    from sfd.core.errors import TransportError

    task = add(database)
    manager = _manager(database)

    manager._fail(task, TransportError("connection reset"))
    failed = database.get(task.id)
    assert failed.state == states.FAILED
    assert failed.attempts == 1
    assert failed.retry_at is not None


def test_a_failure_no_wait_can_fix_is_left_alone(database: Database):
    """A missing token is answered by a person. Asking again in thirty seconds is noise."""
    from sfd.core.errors import AuthRequired

    task = add(database)
    _manager(database)._fail(task, AuthRequired("needs a token"))

    failed = database.get(task.id)
    assert failed.state == states.FAILED
    assert failed.retry_at is None


def test_retries_run_out(database: Database):
    from sfd.core.errors import TransportError
    from sfd.jobs.manager import RETRY_DELAYS

    task = add(database)
    manager = _manager(database)
    for _ in range(len(RETRY_DELAYS)):
        manager._fail(database.get(task.id), TransportError("down"))
        assert database.get(task.id).retry_at is not None

    manager._fail(database.get(task.id), TransportError("down"))
    exhausted = database.get(task.id)
    assert exhausted.attempts == len(RETRY_DELAYS) + 1
    assert exhausted.retry_at is None, "it stops asking rather than retrying forever"


def test_turning_auto_retry_off_stops_booking_them(database: Database):
    from sfd.core.errors import TransportError

    task = add(database)
    _manager(database, auto_retry=False)._fail(task, TransportError("down"))
    assert database.get(task.id).retry_at is None


def test_a_retry_asked_for_by_hand_forgives_the_attempts_spent(database: Database):
    """Otherwise a task that used its three tries at 3am gets one more forever, even after
    the thing that broke it is fixed."""
    from sfd.core.errors import TransportError

    task = add(database)
    manager = _manager(database)
    manager._fail(task, TransportError("down"))

    manager.retry(task.id)
    revived = database.get(task.id)
    assert revived.state == states.PENDING
    assert revived.attempts == 0
    assert revived.retry_at is None


def test_only_retries_that_have_come_round_are_released(database: Database):
    now = 1_000_000.0
    soon = add(database)
    database.update(soon.id, state=states.FAILED, retry_at=now - 1)
    later = add(database, identity={"provider": "direct", "ref": {"url": "b"}}, filename="b")
    database.update(later.id, state=states.FAILED, retry_at=now + 600)

    assert [t.id for t in database.due_retries(now)] == [soon.id]


async def test_the_retry_loop_puts_a_due_task_back_in_the_queue(database: Database, monkeypatch):
    import time

    from sfd.jobs import manager as manager_module

    monkeypatch.setattr(manager_module, "RETRY_POLL", 0.01)
    task = add(database)
    database.update(task.id, state=states.FAILED, retry_at=time.time() - 1, error="down")

    manager = _manager(database)
    loop = asyncio.create_task(manager._retry_loop())
    try:
        for _ in range(100):
            if database.get(task.id).state == states.PENDING:
                break
            await asyncio.sleep(0.01)
    finally:
        manager._stopping = True
        loop.cancel()

    revived = database.get(task.id)
    assert revived.state == states.PENDING
    assert revived.error is None and revived.retry_at is None


def test_pausing_calls_off_a_booked_retry(database: Database):
    from sfd.core.errors import TransportError

    task = add(database)
    manager = _manager(database)
    manager._fail(task, TransportError("down"))

    manager.pause(task.id)
    assert database.get(task.id).retry_at is None


async def test_the_worker_pool_follows_the_setting(tmp_path: Path):
    """A number that only takes effect after a restart reads as a broken control — the more
    so next to `connections`, which the very next download picks up."""
    from sfd.jobs.manager import Manager

    settings = Settings(concurrent_downloads=1)
    manager = Manager(settings, Database(tmp_path / "queue.db"))
    await manager.start()
    try:
        assert manager.workers == 1

        settings.concurrent_downloads = 3
        manager.resize()
        assert manager.workers == 3

        settings.concurrent_downloads = 1
        manager.resize()
        for _ in range(50):
            if manager.workers == 1:
                break
            await asyncio.sleep(0.01)
        assert manager.workers == 1, "workers past the limit retire when they next come up"
    finally:
        await manager.stop()


def test_saving_the_setting_resizes_the_pool(client):
    manager = client.app.state.manager
    assert manager.workers == 1

    client.put("/api/settings", json={"concurrent_downloads": 4})
    assert manager.workers == 4


def test_the_speed_ceiling_takes_effect_without_a_restart(client):
    """You reach for a speed limit precisely while something is downloading."""
    manager = client.app.state.manager
    assert manager.limiter.rate == 0

    client.put("/api/settings", json={"max_speed_kb": 512})
    assert manager.limiter.rate == 512 * 1024

    client.put("/api/settings", json={"max_speed_kb": 0})
    assert manager.limiter.rate == 0


async def test_a_file_the_disk_cannot_hold_fails_before_it_starts(tmp_path: Path, monkeypatch):
    """Preallocation is sparse, so nothing is reserved up front and a full disk would
    otherwise surface as an OSError from a chunk writer, forty gigabytes in."""
    from sfd.jobs import manager as manager_module
    from sfd.jobs.manager import Manager

    database = Database(tmp_path / "queue.db")
    task = add(database, size=40 * 1024**3, dest=str(tmp_path / "huge.safetensors"))
    monkeypatch.setattr(manager_module, "free_bytes", lambda path: 11 * 1024**3)

    await Manager(Settings(), database)._run(database.get(task.id))

    failed = database.get(task.id)
    assert failed.state == states.FAILED
    assert "40.0 GB" in failed.error and "11.0 GB" in failed.error


async def test_a_file_that_fits_is_not_stopped(tmp_path: Path, monkeypatch):
    from sfd.jobs import manager as manager_module
    from sfd.jobs.manager import Manager

    database = Database(tmp_path / "queue.db")
    task = add(database, size=1000, dest=str(tmp_path / "small.safetensors"))
    monkeypatch.setattr(manager_module, "free_bytes", lambda path: 40 * 1024**3)

    manager = Manager(Settings(), database)
    manager._require_space(database.get(task.id), tmp_path / "small.safetensors")


def test_an_unknown_size_is_not_treated_as_zero(tmp_path: Path, monkeypatch):
    """Civitai does not always say how big a file is. Refusing those would refuse most of
    what the tool is pointed at."""
    from sfd.jobs import manager as manager_module
    from sfd.jobs.manager import Manager

    database = Database(tmp_path / "queue.db")
    task = add(database, size=None, dest=str(tmp_path / "unknown.safetensors"))
    monkeypatch.setattr(manager_module, "free_bytes", lambda path: 0)

    Manager(Settings(), database)._require_space(database.get(task.id), tmp_path / "u.bin")


def test_the_space_endpoint_totals_what_is_left_to_fetch(client, tmp_path: Path):
    started = add(client.database, size=1000)
    client.database.update(started.id, downloaded=400)
    add(client.database, size=2000, state=states.DONE,
        identity={"provider": "direct", "ref": {"url": "done"}}, filename="done.bin")
    add(client.database, size=None, state=states.BLOCKED,
        identity={"provider": "direct", "ref": {"url": "unsure"}}, filename="unsure.bin")

    body = client.get("/api/space").json()
    assert body["needed"] == 600, "finished tasks are not still to fetch"
    assert body["unknown"] == 1
    assert body["free"] is None or body["free"] > 0


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


def test_trigger_words_reach_the_page_as_the_txt_file_has_them(database: Database):
    """Civitai often returns them comma-joined inside a single string. Passed through as it
    came, the card shows one chip with commas in it, and the words copied off that card would
    disagree with the `.txt` sitting beside the model."""
    task = add(database, meta={"trained_words": ["ohwx style, ohwx", "  spare  ", "OHWX"]})
    assert task.to_json()["trigger_words"] == ["ohwx style", "ohwx", "spare"]


def test_a_model_with_no_triggers_offers_none(database: Database):
    """Nothing for the page to copy, and nothing for it to draw a chip row for."""
    assert add(database).to_json()["trigger_words"] == []


# --- where a file came from -------------------------------------------------


def test_the_queue_says_which_service_each_file_came_off(database: Database):
    """The card header names the site rather than our provider name. Two files of the same
    name are told apart by where they came from before anything else about them."""
    assert add(database).to_json()["origin"] == "example.com"

    hub = add(
        database,
        provider="huggingface",
        identity={
            "provider": "huggingface",
            "ref": {"repo_id": "org/name", "repo_type": "model",
                    "revision": "main", "path": "model.safetensors"},
        },
        filename="hub.safetensors",
    )
    assert hub.to_json()["origin"] == "huggingface.co"


def test_a_mirror_is_named_as_itself(database: Database):
    """`civitai.red` is a mirror, and the task keeps talking to the domain the link came
    from — calling it "civitai" would hide the one thing that explains the difference."""
    task = add(
        database,
        provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 1, "file_id": 2}},
        meta={"host": "civitai.red"},
    )
    assert task.to_json()["origin"] == "civitai.red"
    plain = add(
        database,
        provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 3, "file_id": 4}},
        filename="other.safetensors",
    )
    assert plain.to_json()["origin"] == "civitai.com"


def test_a_direct_link_is_named_by_its_host_alone(database: Database):
    """Userinfo and the port are noise here, and a `user:pass@` left in would put a
    credential on screen next to the filename."""
    task = add(
        database,
        identity={
            "provider": "direct",
            "ref": {"url": "https://user:secret@www.files.example.com:8443/a/model.bin"},
        },
    )
    assert task.to_json()["origin"] == "files.example.com"


def test_a_row_with_no_service_to_name_says_nothing(database: Database):
    """A queue row from an older build carries no identity worth reading. A missing label is
    a missing badge on a card, not an error on the page."""
    task = add(database, provider="", identity={})
    assert task.to_json()["origin"] == ""


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
    # Under 127.0.0.1 rather than TestClient's default `testserver`: the app refuses names
    # it is not served under, and the tests should exercise the same path a browser takes.
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test_client:
        test_client.database = database  # type: ignore[attr-defined]
        yield test_client


def test_the_page_is_served(client):
    assert client.get("/").status_code == 200


def test_the_stylesheet_and_script_are_served(client):
    """The page is three files now. A missing mount leaves it rendering as plain text with
    no behaviour at all — which the page-is-served test above would not notice."""
    for asset in ("/static/styles.css", "/static/app.js"):
        assert client.get(asset).status_code == 200, asset


def test_a_request_under_someone_elses_name_is_refused(client):
    """DNS rebinding: a page can point a hostname it owns at 127.0.0.1 and then reach this
    API as same-origin. There is no authentication here, so the name is the check."""
    assert client.get("/api/tasks", headers={"host": "models.evil.example"}).status_code == 403
    assert client.put("/api/settings", json={"library_root": "C:/"},
                      headers={"host": "models.evil.example"}).status_code == 403


def test_the_names_this_server_answers_to(client):
    for name in ("127.0.0.1", "127.0.0.1:7788", "localhost:7788", "[::1]:7788"):
        assert client.get("/api/tasks", headers={"host": name}).status_code == 200, name


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


def test_a_setting_outside_its_bounds_is_refused(client):
    """Rejected here, or accepted and then failing inside a transfer hours later, as
    something that reads like a bug in the downloader rather than a typed-in zero."""
    for patch in ({"connections": 0}, {"connections": 999}, {"connections": "several"},
                  {"concurrent_downloads": 0}, {"queue_position": "sideways"},
                  {"hf_engine": "torrent"}, {"min_speed_kb": -1}):
        assert client.put("/api/settings", json=patch).status_code == 422, patch
    assert client.get("/api/settings").json()["settings"]["connections"] == Settings().connections


def test_a_patch_leaves_the_fields_it_does_not_mention_alone(client, tmp_path: Path):
    client.put("/api/settings", json={"library_root": str(tmp_path)})
    client.put("/api/settings", json={"connections": 8})

    body = client.get("/api/settings").json()["settings"]
    assert body["library_root"] == str(tmp_path)
    assert body["connections"] == 8


def test_every_setting_can_still_be_saved_through_the_page():
    """The request model lists its fields by hand. One added to the dataclass and forgotten
    here would be accepted by the form and silently dropped on the way in."""
    from dataclasses import fields

    from sfd.web.app import SettingsPatch

    storable = {f.name for f in fields(Settings) if not f.name.startswith("_")}
    assert storable == set(SettingsPatch.model_fields)


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
