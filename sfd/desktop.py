"""Native desktop window runner using pywebview (WebView2 on Windows).

Runs the FastAPI / Uvicorn server in a background thread and opens a standalone
desktop window with system integration (native folder pickers, Explorer reveal,
graceful server shutdown on window close).
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import webview
except ImportError:
    webview = None  # type: ignore[assignment]

import uvicorn
from fastapi import FastAPI


class JsApi:
    """Methods exposed to JavaScript via window.pywebview.api."""

    def __init__(self) -> None:
        self._window: Any = None

    def set_window(self, window: Any) -> None:
        self._window = window

    def pick_folder(self, initial_dir: str = "") -> str | None:
        """Open a native Windows/macOS folder picker dialog."""
        return pick_system_folder(initial_dir, self._window)


# A folder dialog is modal to the person in front of the screen, so two at once is not
# something they can answer. The HTTP endpoint runs in a worker thread and could otherwise
# raise a second one behind the first.
_DIALOG = threading.Lock()


def pick_system_folder(initial_dir: str = "", window: Any = None) -> str | None:
    """Open a native folder picker dialog using pywebview or fallback."""
    with _DIALOG:
        return _pick_folder(initial_dir, window)


def _pick_folder(initial_dir: str, window: Any) -> str | None:
    init_path = str(Path(initial_dir).resolve()) if initial_dir and Path(initial_dir).exists() else ""

    # 1. Use pywebview window dialog if available
    if window is not None:
        try:
            result = window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=init_path or None,
            )
            if result:
                # pywebview returns a tuple/list of strings
                return str(result[0])
            return None
        except Exception:
            pass

    # 2. Fallback using tkinter if in standard Python. Only from the main thread: Tk owns
    # the thread its root was created on, and the browser-mode endpoint answers from a
    # worker thread, where creating one is a coin flip between working and hanging the
    # request. There the OS dialog below is the one that runs.
    if threading.current_thread() is threading.main_thread():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(initialdir=init_path or None)
            root.destroy()
            return str(folder) if folder else None
        except Exception:
            pass

    # 3. Windows PowerShell FolderBrowserDialog fallback
    if platform.system() == "Windows":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    f"$f.SelectedPath = '{init_path}'; "
                    "$f.Description = 'Select Folder'; "
                    "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
                    "Write-Output $f.SelectedPath }"
                ),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=30,
            )
            out = proc.stdout.strip()
            return out if out and Path(out).is_dir() else None
        except Exception:
            pass

    return None


def open_system_path(path: str) -> bool:
    """Reveal a file in Explorer or open a directory in the default file manager."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            # If the file doesn't exist, try its parent directory
            if p.parent.exists():
                p = p.parent
            else:
                return False

        system = platform.system()
        if system == "Windows":
            if p.is_file():
                # /select,<path> highlights the downloaded file in Explorer
                subprocess.Popen(["explorer.exe", f"/select,{p}"])
            else:
                os.startfile(str(p))
            return True
        elif system == "Darwin":  # macOS
            if p.is_file():
                subprocess.Popen(["open", "-R", str(p)])
            else:
                subprocess.Popen(["open", str(p)])
            return True
        else:  # Linux / BSD
            target = str(p.parent if p.is_file() else p)
            subprocess.Popen(["xdg-open", target])
            return True
    except Exception:
        return False


class ServerThread(threading.Thread):
    """Runs Uvicorn in a background thread with clean shutdown support."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        super().__init__(daemon=True, name="UvicornServerThread")
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def wait_for_server(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait until the HTTP server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def is_gui_available() -> bool:
    """Check if pywebview is available in the current environment."""
    return webview is not None


def launch_desktop(
    app: FastAPI,
    host: str = "127.0.0.1",
    port: int = 7788,
    width: int = 1120,
    height: int = 780,
    debug: bool = False,
) -> int:
    """Launch the FastAPI server and open the native pywebview desktop window."""
    if not is_gui_available():
        raise RuntimeError(
            "pywebview is not installed. Install it with: pip install pywebview"
        )

    server_thread = ServerThread(app, host=host, port=port)
    server_thread.start()

    if not wait_for_server(host, port):
        server_thread.stop()
        raise RuntimeError(f"Server failed to start on {host}:{port}")

    url = f"http://{host}:{port}"
    api = JsApi()

    # Create native standalone window
    window = webview.create_window(
        title="ModelDL",
        url=url,
        width=width,
        height=height,
        min_size=(760, 540),
        js_api=api,
        background_color="#14161a",  # Match dark theme to avoid white flash
        text_select=True,
    )
    api.set_window(window)

    try:
        # On Windows, EdgeChromium (WebView2) provides full modern web support
        gui = "edgechromium" if sys.platform == "win32" else None
        webview.start(debug=debug, gui=gui)
    finally:
        # Window closed by user -> cleanly shutdown backend server
        server_thread.stop()
        server_thread.join(timeout=3.0)

    return 0
