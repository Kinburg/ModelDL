"""Tests for desktop integration, native dialogs, and utility endpoints."""

from __future__ import annotations

import socket
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


def test_a_port_we_cannot_have_moves_us_to_another_one():
    """Windows reserves blocks of ports for Hyper-V and WSL at boot, and a bind inside one
    fails outright. The port the person asked for is a preference, not a requirement."""
    from sfd import desktop

    taken, port = desktop.bind_local_port("127.0.0.1", 0)
    try:
        moved, chosen = desktop.bind_local_port("127.0.0.1", port)
        moved.close()
        assert chosen != port
    finally:
        taken.close()


def test_the_search_steps_over_a_whole_reserved_block():
    """The reserved blocks are a hundred ports wide, so the next port up is inside the same
    one. Stepping by a single port would scan the block instead of leaving it."""
    from sfd import desktop

    attempted = []
    real_socket = socket.socket

    class OnlyFarPortsWork(real_socket):
        def bind(self, address):
            attempted.append(address[1])
            if address[1] and address[1] < 8000:
                raise PermissionError(10013, "forbidden by its access permissions")
            return super().bind(address)

    with patch.object(desktop.socket, "socket", OnlyFarPortsWork):
        sock, chosen = desktop.bind_local_port("127.0.0.1", 7788)
    sock.close()

    assert chosen >= 8000
    assert max(attempted) - min(attempted) > 100


def test_no_bindable_port_at_all_is_reported_as_such():
    from sfd import desktop

    real_socket = socket.socket

    class NothingBinds(real_socket):
        def bind(self, address):
            raise PermissionError(10013, "forbidden by its access permissions")

    with patch.object(desktop.socket, "socket", NothingBinds):
        with pytest.raises(RuntimeError, match="could not bind a port"):
            desktop.bind_local_port("127.0.0.1", 7788)


def test_the_window_wears_the_application_icon():
    """From source there is nothing to inherit an icon from, so the file has to be found."""
    from sfd import desktop

    icon = desktop.window_icon()
    assert icon is not None and icon.endswith(".ico")
    assert Path(icon).is_file()


def test_a_missing_icon_leaves_the_choice_to_pywebview(monkeypatch, tmp_path):
    """Inside the built exe the logo is not on disk; pywebview then takes the icon out of
    the executable, which is the same image. Handing it a path that is not there would
    only make it look."""
    from sfd import desktop

    monkeypatch.setattr(desktop, "__file__", str(tmp_path / "desktop.py"))
    assert desktop.window_icon() is None


class _Event:
    """Stands in for a .NET event: `core.ContextMenuRequested += handler`."""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


def _fake_webview():
    from types import SimpleNamespace

    core = SimpleNamespace(
        Settings=SimpleNamespace(AreDefaultContextMenusEnabled=False),
        ContextMenuRequested=_Event(),
    )
    return SimpleNamespace(CoreWebView2=core), SimpleNamespace(IsSuccess=True)


def _fake_edge():
    from types import SimpleNamespace

    class EdgeChrome:
        def on_webview_ready(self, sender, args):
            self.pywebview_ran = True

    return SimpleNamespace(EdgeChrome=EdgeChrome)


def test_a_text_field_gets_its_right_click_menu_back():
    """pywebview ties WebView2's default context menus to debug mode, so a normal run has
    none at all and copy/paste are keyboard-only. The setting is flipped where pywebview
    flips it — on the UI thread, once the WebView2 core exists — without losing the rest of
    what that handler does."""
    from sfd import desktop

    edge = _fake_edge()
    assert desktop.enable_text_context_menus(edge) is True

    chrome = edge.EdgeChrome()
    sender, args = _fake_webview()
    chrome.on_webview_ready(sender, args)

    assert chrome.pywebview_ran is True
    assert sender.CoreWebView2.Settings.AreDefaultContextMenusEnabled is True
    assert desktop.suppress_page_context_menu in sender.CoreWebView2.ContextMenuRequested.handlers


def test_the_right_click_menu_is_the_editing_one_not_the_browsers():
    """Over a text field or a selection the menu is what the right-click was reached for;
    on the page itself it is Edge's own — reload, save as, print — in a window that is not
    a browser."""
    from types import SimpleNamespace

    from sfd import desktop

    def menu_for(**target):
        args = SimpleNamespace(ContextMenuTarget=SimpleNamespace(**target), Handled=False)
        desktop.suppress_page_context_menu(None, args)
        return not args.Handled

    assert menu_for(IsEditable=True, HasSelection=False) is True
    assert menu_for(IsEditable=False, HasSelection=True) is True
    assert menu_for(IsEditable=False, HasSelection=False) is False


def test_a_right_click_we_cannot_read_keeps_the_default_menu():
    """Whatever this is, a menu nobody asked for is a smaller thing than an exception
    thrown back into the UI thread."""
    from types import SimpleNamespace

    from sfd import desktop

    class Unreadable:
        @property
        def ContextMenuTarget(self):
            raise RuntimeError("interop said no")

    desktop.suppress_page_context_menu(None, Unreadable())
    desktop.suppress_page_context_menu(None, SimpleNamespace())


def test_a_webview_that_will_not_answer_still_opens_the_window():
    """The menu is worth a try and nothing more: if the core or the event is not there —
    an older WebView2 runtime has the setting but not the filter — the window opens anyway."""
    from types import SimpleNamespace

    from sfd import desktop

    edge = _fake_edge()
    desktop.enable_text_context_menus(edge)
    chrome = edge.EdgeChrome()

    class NoCore:
        @property
        def CoreWebView2(self):
            raise RuntimeError("initialization raced us")

    chrome.on_webview_ready(NoCore(), SimpleNamespace(IsSuccess=True))
    assert chrome.pywebview_ran is True


def test_the_menu_is_only_wired_once():
    """launch_desktop can run twice in a process — tests do it — and a handler wrapped
    around itself would set the same settings again on every navigation."""
    from sfd import desktop

    edge = _fake_edge()
    assert desktop.enable_text_context_menus(edge) is True
    assert desktop.enable_text_context_menus(edge) is False


def test_no_webview2_interop_leaves_the_window_as_it_was():
    """On a machine where the interop DLLs will not import — or a pywebview that has moved
    on from this handler — there is nothing to patch and nothing to report."""
    from sfd import desktop

    assert desktop.enable_text_context_menus(object()) is False
