"""Settings: where they are written, and what happens when they cannot be read."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sfd.settings import Settings, hf_login, hf_login_path


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


def test_new_downloads_join_the_bottom_unless_asked_otherwise(tmp_path: Path):
    """Where a pasted link lands is opt-in; without a choice the queue stays a queue."""
    path = tmp_path / "settings.json"
    settings = Settings.load(path)
    assert settings.queue_position == "bottom"

    settings.apply({"queue_position": "top"})
    settings.save()
    assert Settings.load(path).queue_position == "top"


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


def _login(tmp_path: Path, monkeypatch, token: str, stored: str | None = None) -> Path:
    """A token file as `hf auth login` leaves it, and the `stored_tokens` beside it."""
    path = tmp_path / "hf" / "token"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, "utf-8")
    if stored is not None:
        (path.parent / "stored_tokens").write_text(stored, "utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(path))
    return path


def test_the_hf_auth_login_token_comes_last(monkeypatch, tmp_path: Path):
    _login(tmp_path, monkeypatch, "from-login\n")
    settings = Settings.load(tmp_path / "settings.json")
    assert settings.effective_hf_token == "from-login"

    settings.hf_token = "from-file"
    assert settings.effective_hf_token == "from-file"

    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert settings.effective_hf_token == "from-env"


def test_the_login_is_looked_for_where_huggingface_hub_keeps_it(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_TOKEN_PATH")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    assert hf_login_path() == tmp_path / "hf-home" / "token"

    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hf_login_path() == tmp_path / "xdg" / "huggingface" / "token"

    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert hf_login_path() == tmp_path / "home" / ".cache" / "huggingface" / "token"


def test_an_expired_browser_login_is_left_out(monkeypatch, tmp_path: Path):
    """Only the `hf` tool renews a browser login's token, so one past its expiry is not
    sent — and the page can say that the login is what ran out."""
    stored = "[me]\nhf_token = hf_oauth\nrefresh_token = hf_refresh\nexpires_at = {}\n"
    _login(tmp_path, monkeypatch, "hf_oauth", stored.format(int(time.time()) - 60))
    settings = Settings.load(tmp_path / "settings.json")
    assert hf_login() == (None, True)
    assert settings.effective_hf_token is None
    assert settings.redacted()["hf_token_from_login"] is False
    assert settings.redacted()["hf_token_login_expired"] is True

    _login(tmp_path, monkeypatch, "hf_oauth", stored.format(int(time.time()) + 3600))
    assert hf_login() == ("hf_oauth", False)
    assert settings.effective_hf_token == "hf_oauth"


def test_a_pasted_login_token_does_not_expire(monkeypatch, tmp_path: Path):
    # What a login with a token from the website records: the token, and no expiry.
    _login(tmp_path, monkeypatch, "hf_pasted", "[pasted]\nhf_token = hf_pasted\n")
    assert hf_login() == ("hf_pasted", False)


def test_a_token_file_with_a_bom_still_reads(monkeypatch, tmp_path: Path):
    # PowerShell's `Set-Content -Encoding utf8` writes one.
    _login(tmp_path, monkeypatch, "﻿hf_bom\n")
    assert hf_login() == ("hf_bom", False)


def test_no_login_or_an_empty_one_is_no_token(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "nowhere" / "token"))
    assert hf_login() == (None, False)

    _login(tmp_path, monkeypatch, "  \n")
    assert hf_login() == (None, False)
    assert Settings.load(tmp_path / "settings.json").effective_hf_token is None


def test_the_page_learns_where_the_token_comes_from_never_the_token(monkeypatch, tmp_path: Path):
    _login(tmp_path, monkeypatch, "hf_secret_login")
    redacted = Settings.load(tmp_path / "settings.json").redacted()
    assert redacted["hf_token_from_login"] is True
    assert redacted["hf_token_login_expired"] is False
    assert "hf_secret_login" not in json.dumps(redacted)


def test_apply_ignores_internals(tmp_path: Path):
    settings = Settings.load(tmp_path / "settings.json")
    settings.apply({"_path": "/somewhere/else", "_error": "spoofed", "connections": 3})
    assert settings.connections == 3
    assert settings._path == str(tmp_path / "settings.json")
    assert settings.error == ""
