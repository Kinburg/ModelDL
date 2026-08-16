"""Tests for desktop integration, native dialogs, and utility endpoints."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sfd.desktop import JsApi, is_gui_available, open_system_path, pick_system_folder
from sfd.jobs.db import Database
from sfd.settings import Settings
from sfd.web.app import create_app


def test_is_gui_available_says_whether_pywebview_is_installed():
    from sfd import desktop

    assert is_gui_available() is (desktop.webview is not None)


def test_the_desktop_window_is_optional(monkeypatch):
    """pywebview is an extra, not a dependency: a headless or command-line install should
    not have to carry a GUI toolkit, and without it the server still runs in a browser."""
    from sfd import desktop

    monkeypatch.setattr(desktop, "webview", None)
    assert desktop.is_gui_available() is False
    with pytest.raises(RuntimeError, match="pywebview"):
        desktop.launch_desktop(None)


def test_the_page_is_found_inside_a_bundle(tmp_path, monkeypatch):
    """The one path that only breaks in the built exe, where nothing else notices."""
    import sys

    from sfd.web import app as web_app

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    bundled = tmp_path / "sfd" / "web" / "static"
    bundled.mkdir(parents=True)
    assert web_app._get_static_dir() == bundled

    # PyInstaller can be told to put the same folder at the root instead.
    import shutil

    shutil.rmtree(tmp_path / "sfd")
    flat = tmp_path / "static"
    flat.mkdir()
    assert web_app._get_static_dir() == flat


def test_the_page_is_found_when_running_from_source(monkeypatch):
    import sys

    from sfd.web import app as web_app

    monkeypatch.delattr(sys, "frozen", raising=False)
    found = web_app._get_static_dir()
    assert (found / "index.html").exists()
    assert (found / "app.js").exists()
    assert (found / "styles.css").exists()


def test_js_api_pick_folder():
    api = JsApi()
    mock_window = MagicMock()
    mock_window.create_file_dialog.return_value = ["F:\\Models"]
    api.set_window(mock_window)

    result = api.pick_folder()
    assert result == "F:\\Models"
    mock_window.create_file_dialog.assert_called_once()


def test_js_api_pick_folder_cancel():
    api = JsApi()
    mock_window = MagicMock()
    mock_window.create_file_dialog.return_value = None
    api.set_window(mock_window)

    result = api.pick_folder()
    assert result is None


def test_open_system_path_nonexistent():
    assert open_system_path("Z:\\NonExistentPath\\FakeFile.bin") is False


def test_open_system_path_existing_dir(tmp_path: Path):
    with patch("os.startfile", create=True) as mock_startfile:
        res = open_system_path(str(tmp_path))
        assert res is True
        mock_startfile.assert_called_once_with(str(tmp_path.resolve()))


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings.load(tmp_path / "settings.json")
    database = Database(tmp_path / "queue.db")
    test_client = TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788")
    test_client.database = database  # type: ignore[attr-defined]
    return test_client


def _queued(database: Database, dest: Path):
    return database.add(
        source="https://example.com/model.safetensors",
        provider="direct",
        identity={"provider": "direct", "ref": {"url": "https://example.com/model.safetensors"}},
        filename=dest.name,
        dest=str(dest),
    )


def test_revealing_a_download_takes_the_path_from_the_queue(client, tmp_path: Path):
    """Never from the request: this call hands a string to the shell, and the server has no
    authentication to decide whose string it is."""
    destination = tmp_path / "model.safetensors"
    task = _queued(client.database, destination)

    with patch("sfd.desktop.open_system_path", return_value=True) as mock_open:
        response = client.post(f"/api/tasks/{task.id}/reveal")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    mock_open.assert_called_once_with(str(destination))


def test_revealing_a_task_that_does_not_exist(client):
    assert client.post("/api/tasks/999/reveal").status_code == 404


def test_the_folder_picker_endpoint(client):
    with patch("sfd.desktop.pick_system_folder", return_value="C:\\SelectedDir") as mock_pick:
        response = client.post("/api/utils/pick-folder", json={"initial": "C:\\"})

    assert response.status_code == 200
    assert response.json() == {"path": "C:\\SelectedDir"}
    mock_pick.assert_called_once_with("C:\\")


def test_the_folder_picker_stays_off_the_event_loop(client):
    """The dialog blocks until someone answers it. Awaiting that on the loop freezes every
    running download for as long as the window is open; a sync handler gets a worker thread.
    """
    import inspect

    route = next(r for r in client.app.routes if getattr(r, "path", "") == "/api/utils/pick-folder")
    assert not inspect.iscoroutinefunction(route.endpoint)


def test_two_dialogs_cannot_open_at_once():
    """Modal to the person at the screen: a second one is not something they can answer."""
    import threading

    from sfd import desktop

    order: list[str] = []

    def slow_dialog(initial_dir: str, window):
        order.append("enter")
        time.sleep(0.05)
        order.append("leave")
        return None

    with patch.object(desktop, "_pick_folder", slow_dialog):
        threads = [
            threading.Thread(target=desktop.pick_system_folder, args=("",)) for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert order == ["enter", "leave", "enter", "leave"]
