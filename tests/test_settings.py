"""Settings: where they are written, and what happens when they cannot be read."""

from __future__ import annotations

import json
from pathlib import Path

from sfd.settings import Settings


def test_a_missing_file_starts_from_defaults(tmp_path: Path):
    settings = Settings.load(tmp_path / "settings.json")
    assert settings.library_root == ""
    assert settings.error == ""


def test_round_trip(tmp_path: Path):
    path = tmp_path / "settings.json"
    original = Settings.load(path)
    original.library_root = str(tmp_path / "models")
    original.connections = 12
    original.save()

    reloaded = Settings.load(path)
    assert reloaded.library_root == str(tmp_path / "models")
    assert reloaded.connections == 12


def test_saving_returns_to_the_file_it_came_from(tmp_path: Path):
    path = tmp_path / "custom.json"
    settings = Settings.load(path)
    settings.connections = 5
    settings.save()
    assert path.exists()
    assert json.loads(path.read_text("utf-8"))["connections"] == 5


def test_a_utf8_bom_does_not_wipe_the_configuration(tmp_path: Path):
    """PowerShell's `-Encoding utf8` writes a BOM.

    Decoded as plain utf-8 the file fails on its very first byte, and a loader that shrugs
    that off leaves the library unconfigured with nothing said — downloads then quietly go
    somewhere else.
    """
    path = tmp_path / "settings.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"library_root": "F:/models"}).encode())

    settings = Settings.load(path)
    assert settings.library_root == "F:/models"
    assert settings.error == ""


def test_an_unreadable_file_is_reported_and_preserved(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", "utf-8")

    settings = Settings.load(path)
    assert settings.error, "a broken settings file must not pass silently"
    assert "library path is not configured" in settings.error
    # Moved aside, so the next save cannot overwrite whatever was in it.
    assert (tmp_path / "settings.json.broken").exists()


def test_a_json_array_is_not_accepted_as_settings(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", "utf-8")
    assert Settings.load(path).error


def test_unknown_keys_are_ignored(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"connections": 7, "from_a_future_version": True}), "utf-8")
    assert Settings.load(path).connections == 7


def test_internal_fields_are_never_serialised(tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings.load(path)
    settings.save()
    stored = json.loads(path.read_text("utf-8"))
    assert not any(key.startswith("_") for key in stored)


def test_environment_variables_win_over_the_file(monkeypatch, tmp_path: Path):
    settings = Settings.load(tmp_path / "settings.json")
    settings.hf_token = "from-file"
    assert settings.effective_hf_token == "from-file"

    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert settings.effective_hf_token == "from-env"


def test_apply_ignores_internals(tmp_path: Path):
    settings = Settings.load(tmp_path / "settings.json")
    settings.apply({"_path": "/somewhere/else", "_error": "spoofed", "connections": 3})
    assert settings.connections == 3
    assert settings._path == str(tmp_path / "settings.json")
    assert settings.error == ""
