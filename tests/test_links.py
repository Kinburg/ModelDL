"""One file under several names: hard links, made from proven copies and undone again.

A copy kept because two nodes each read their own folder takes its room twice; linked, every
path keeps working and the room is taken once. What is tested is that only copies the hashes
prove the same are ever linked, that nothing is left half done when a step is refused or
stopped, and that the way back gives every name its own bytes again.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.jobs.library import Library
from sfd.library import links, relocate
from sfd.settings import Settings
from sfd.web.app import create_app

BIG = 1024 * 1024 + 4096


def weights(seed: int = 0, size: int = BIG) -> bytes:
    block = hashlib.sha256(f"seed {seed}".encode()).digest() * 2048
    return (block * (size // len(block) + 1))[:size]


def put(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def same_file(a: Path, b: Path) -> bool:
    return os.stat(a).st_ino == os.stat(b).st_ino


@pytest.fixture(autouse=True)
def _hard_links_work(tmp_path: Path):
    probe = put(tmp_path / "probe", b"x")
    try:
        os.link(probe, tmp_path / "probe-link")
    except OSError:
        pytest.skip("this file system has no hard links")


# --- the file system -------------------------------------------------------------------


def test_a_copy_becomes_another_name_of_the_kept_file(tmp_path: Path):
    keep = put(tmp_path / "insightface" / "inswapper_128.onnx", weights())
    other = put(tmp_path / "simswap" / "inswapper_128.onnx", weights())

    links.link_over(keep, other)

    assert same_file(keep, other) and os.stat(keep).st_nlink == 2
    assert other.read_bytes() == weights()
    assert not list(tmp_path.rglob("*" + links.LINKING))


def test_a_copy_that_cannot_be_replaced_is_left_as_it_was(tmp_path: Path, monkeypatch):
    keep = put(tmp_path / "a" / "m.onnx", weights())
    other = put(tmp_path / "b" / "m.onnx", weights())
    monkeypatch.setattr(links.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("in use")))

    with pytest.raises(PermissionError):
        links.link_over(keep, other)

    assert not same_file(keep, other) and os.stat(keep).st_nlink == 1
    assert not list(tmp_path.rglob("*" + links.LINKING)), "the second name made for the swap goes"


def test_a_name_gets_its_own_bytes_back(tmp_path: Path):
    keep = put(tmp_path / "a" / "m.onnx", weights())
    other = tmp_path / "b" / "m.onnx"
    other.parent.mkdir()
    os.link(keep, other)

    links.separate(other)

    assert not same_file(keep, other)
    assert other.read_bytes() == keep.read_bytes()
    assert os.stat(other).st_mtime == os.stat(keep).st_mtime, "the dates go with the copy"


def test_a_copy_stopped_halfway_leaves_the_name_linked(tmp_path: Path):
    keep = put(tmp_path / "a" / "m.onnx", weights())
    other = tmp_path / "b" / "m.onnx"
    other.parent.mkdir()
    os.link(keep, other)
    stop = threading.Event()
    stop.set()

    with pytest.raises(relocate.Cancelled):
        links.separate(other, stop=stop)

    assert same_file(keep, other)
    assert not list(tmp_path.rglob("*" + links.SEPARATING))


def test_every_name_of_a_file_is_listed(tmp_path: Path):
    keep = put(tmp_path / "a" / "m.onnx", b"x")
    other = tmp_path / "b" / "m.onnx"
    other.parent.mkdir()
    os.link(keep, other)

    found = {os.path.normcase(str(p)) for p in links.names(keep)}

    if os.name == "nt":
        assert found == {os.path.normcase(str(keep)), os.path.normcase(str(other))}
    else:
        assert found == {os.path.normcase(str(keep))}


# --- the library ------------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "d"),
                        _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    library = Library(settings, database, lambda event: None)
    yield library, root, database
    database.close()


def copies(library: Library, database: Database, root: Path, count: int = 2, data: bytes | None = None):
    data = data or weights()
    paths = [put(root / folder / "inswapper_128.onnx", data) for folder in ("insightface", "simswap", "reactor")[:count]]
    library.sync()
    digest = hashlib.sha256(data).hexdigest()
    for path in paths:
        model = database.model_at(path)
        database.update_model(model.id, sha256=digest, hash_source="computed", hashed_mtime=model.mtime)
    return paths, [database.model_at(p).id for p in paths]


def test_proven_copies_are_linked_and_their_room_freed(setup):
    library, root, database = setup
    (a, b, c), (keep, *others) = copies(library, database, root, 3)

    result = library.link_copies(keep, others)

    assert [r["ok"] for r in result["results"]] == [True, True]
    assert result["freed"] == 2 * BIG
    assert same_file(a, b) and same_file(a, c) and os.stat(a).st_nlink == 3
    assert {database.get_model(i).links for i in (keep, *others)} == {3}


def test_a_link_keeps_every_hash_and_is_listed_as_linked(setup):
    library, root, database = setup
    _paths, (keep, other) = copies(library, database, root)
    library.link_copies(keep, [other])

    library.sync()
    found = library.duplicates()

    assert database.get_model(other).sha256, "a walk after linking must not take the hash for stale"
    assert found["groups"] == [], "one file under two names is not a copy any more"
    (linked,) = found["linked"]
    assert linked["names"] == 2 and linked["saved"] == BIG
    assert {m["id"] for m in linked["models"]} == {keep, other}


def test_copies_the_hashes_do_not_prove_are_not_linked(setup):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    database.update_model(other, sha256=None, hash_source=None)

    result = library.link_copies(keep, [other])

    assert result["results"][0]["ok"] is False and "confirm by hash" in result["results"][0]["error"]
    assert not same_file(a, b)


def test_a_file_changed_since_it_was_read_is_not_linked(setup):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    os.utime(b, (1_000_000, 1_000_000))

    result = library.link_copies(keep, [other])

    assert result["results"][0]["ok"] is False and "changed" in result["results"][0]["error"]
    assert not same_file(a, b)


def test_a_copy_on_another_drive_is_left_alone(setup, monkeypatch):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    monkeypatch.setattr(links, "same_volume", lambda x, y: False)

    result = library.link_copies(keep, [other])

    assert "another drive" in result["results"][0]["error"]
    assert not same_file(a, b)


def test_separating_needs_the_room_first(setup, monkeypatch):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    library.link_copies(keep, [other])
    monkeypatch.setattr("sfd.jobs.library.free_bytes", lambda path: 10)

    with pytest.raises(ValueError, match="free"):
        library.separate(other)

    assert same_file(a, b)


def test_separating_gives_the_name_its_own_copy_and_keeps_its_hash(setup):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    library.link_copies(keep, [other])

    library.separate(other)
    library.sync()

    assert not same_file(a, b)
    assert {database.get_model(i).links for i in (keep, other)} == {1}
    assert database.get_model(other).sha256, "the same bytes and dates, so the same hash"


def test_the_other_names_are_told_before_a_delete(setup):
    library, root, database = setup
    (a, b), (keep, other) = copies(library, database, root)
    library.link_copies(keep, [other])

    others = library.other_names(keep)

    assert others["count"] == 1
    if os.name == "nt":
        assert [os.path.normcase(p) for p in others["paths"]] == [os.path.normcase(str(b))]


def test_a_link_or_copy_left_by_a_crash_is_offered_by_the_cleanup(setup):
    library, root, database = setup
    put(root / "vae" / "ae.safetensors", weights())
    put(root / "vae" / "ae.safetensors.separating", b"half")
    library.sync()

    items = library.cleanup_scan()["items"]

    assert [(i["kind"], i["why"]) for i in items] == [("fragment", "an interrupted copy")]


# --- through the page -----------------------------------------------------------------------


def test_the_page_links_and_separates(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    settings = Settings(library_root=str(root), download_dir=str(tmp_path / "d"), concurrent_downloads=1,
                        _path=str(tmp_path / "settings.json"))
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as client:
        library = client.app.state.manager.library
        (a, b), (keep, other) = copies(library, database, root)

        body = client.post("/api/duplicates/link", json={"keep": keep, "ids": [other]}).json()
        assert body["freed"] == BIG and same_file(a, b)
        assert client.get(f"/api/models/{keep}/files").json()["others"]["count"] == 1

        body = client.post(f"/api/models/{other}/separate").json()
        assert body["ok"] and not same_file(a, b)
