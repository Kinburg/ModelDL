"""Where the app keeps its own files: beside the exe, wherever it is started from."""

from __future__ import annotations

from pathlib import Path

import pytest

from sfd import home
from sfd.home import locate


@pytest.fixture
def program(tmp_path: Path) -> Path:
    folder = tmp_path / "apps" / "ModelDL"
    folder.mkdir(parents=True)
    exe = folder / "ModelDL.exe"
    exe.write_bytes(b"MZ")
    return exe


def test_run_from_source_it_keeps_them_where_it_is_started(tmp_path: Path):
    found = locate(frozen=False, cwd=tmp_path)
    assert found.folder == tmp_path
    assert found.settings == tmp_path / "settings.json"
    assert found.db == tmp_path / "queue.db"


def test_the_built_program_keeps_them_beside_itself(program: Path, tmp_path: Path):
    """Started from a shortcut with another "Start in", or a terminal somewhere else, it
    still finds the settings it saved — and does not leave a new queue.db there."""
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    found = locate(frozen=True, executable=program, cwd=elsewhere)
    assert found.folder == program.parent
    assert found.settings == program.parent / "settings.json"
    assert found.db == program.parent / "queue.db"


def test_a_folder_it_cannot_write_to_sends_them_to_the_users_own(
    program: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(home, "writable", lambda folder: False)
    appdata = tmp_path / "LocalAppData"
    found = locate(frozen=True, executable=program, cwd=tmp_path, local_appdata=appdata)
    assert found.folder == appdata / "ModelDL"
    assert found.settings == appdata / "ModelDL" / "settings.json"


def test_files_already_beside_the_program_are_the_ones_meant(
    program: Path, tmp_path: Path, monkeypatch
):
    """A probe that fails for a passing reason — a scanner holding the new file — must not
    move a working install to an empty folder that looks like a lost history."""
    (program.parent / "queue.db").write_bytes(b"")
    monkeypatch.setattr(home, "writable", lambda folder: False)
    found = locate(frozen=True, executable=program, cwd=tmp_path, local_appdata=tmp_path / "x")
    assert found.folder == program.parent


def test_a_data_folder_named_on_the_command_line_wins(program: Path, tmp_path: Path):
    found = locate("portable-data", frozen=True, executable=program, cwd=tmp_path)
    assert found.folder == tmp_path / "portable-data"
    assert found.db == tmp_path / "portable-data" / "queue.db"

    absolute = tmp_path / "elsewhere"
    assert locate(str(absolute), frozen=False, cwd=program.parent).folder == absolute


def test_a_file_named_on_the_command_line_means_it_from_where_it_was_typed(
    program: Path, tmp_path: Path
):
    """Not from the data folder: the launcher moves into that one, and `--db test.db` typed
    in a terminal means the file in the terminal's folder."""
    found = locate(settings="mine.json", db="..\\other.db", frozen=True, executable=program,
                   cwd=tmp_path / "here")
    assert found.folder == program.parent
    assert found.settings == tmp_path / "here" / "mine.json"
    assert found.db == tmp_path / "other.db"


def test_writable_asks_by_making_a_file_and_leaves_nothing(tmp_path: Path):
    assert home.writable(tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert not home.writable(tmp_path / "missing")
