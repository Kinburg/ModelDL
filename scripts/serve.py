"""Run the local web interface.

    run.cmd                        (Windows — sets everything up on first run)
    .venv/bin/python scripts/serve.py

Binds to 127.0.0.1 only. The server holds your tokens and has no authentication, so it must
not be reachable from the rest of the network.
"""

from __future__ import annotations

import argparse
import socket
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import uvicorn
except ModuleNotFoundError as exc:  # pragma: no cover - a setup problem, not a code path
    # The obvious command — `python scripts/serve.py` — uses whatever Python is on PATH,
    # which is usually not the one the dependencies were installed into. A bare
    # ModuleNotFoundError names the missing package and hides the actual mistake.
    venv = ROOT / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    print(
        f"{exc.name} is not installed in this interpreter ({sys.executable}).\n",
        file=sys.stderr,
    )
    if venv.exists():
        print(f"The project has its own environment. Use it:\n\n  {venv} {' '.join(sys.argv)}\n",
              file=sys.stderr)
    else:
        print("Set the project up first:\n\n  run.cmd\n\nor by hand:\n\n"
              "  py -3 -m venv .venv\n"
              f"  {venv} -m pip install -e \".[hf]\"\n", file=sys.stderr)
    raise SystemExit(1) from None

from sfd.desktop import bind_local_port, is_gui_available, launch_desktop  # noqa: E402
from sfd.jobs.db import Database  # noqa: E402
from sfd.settings import Settings  # noqa: E402
from sfd.web.app import create_app  # noqa: E402


def serve(app, host: str, port: int, sock: socket.socket) -> None:
    """Run the server on a socket we already hold."""
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    uvicorn.Server(config).run(sockets=[sock])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7788, help="port to bind to (default: 7788)")
    parser.add_argument("--db", default="queue.db", help="path to SQLite database (default: queue.db)")
    parser.add_argument("--settings", default="settings.json", help="path to settings file")
    parser.add_argument("--browser", "--open", action="store_true", help="open in default web browser instead of desktop window")
    parser.add_argument("--no-gui", "--headless", action="store_true", help="run server without opening window or browser")
    parser.add_argument("--debug", action="store_true", help="enable devtools in desktop window")
    args = parser.parse_args()

    settings = Settings.load(Path(args.settings))
    say = lambda line: print(line, flush=True)  # noqa: E731

    host = "127.0.0.1"
    try:
        sock, port = bind_local_port(host, args.port)
    except RuntimeError as exc:
        say(f"{exc}")
        if sys.platform == "win32":
            # The likeliest cause on Windows, and the one with no visible symptom: nothing
            # is listening on the port, it is simply reserved out from under us.
            say("")
            say("Windows may have reserved the port for Hyper-V, WSL or Docker.")
            say("The reserved ranges:")
            say("")
            say("  netsh interface ipv4 show excludedportrange protocol=tcp")
            say("")
            say("Pick a port outside them with --port.")
        return 1

    # A fresh database per attempt. Shutting the app down closes its connection, so the
    # browser fallback below cannot reuse the one the desktop attempt already spent.
    build_app = lambda: create_app(settings, Database(args.db))  # noqa: E731

    url = f"http://{host}:{port}"

    say(f"ModelDL on {url}")
    if port != args.port:
        say(f"\n  !! port {args.port} was not available — using {port} instead\n")
    if settings.error:
        say(f"\n  !! {settings.error}\n")
    if settings.library_root:
        say(f"library: {settings.library_root} ({settings.profile})")
    else:
        say(f"library: not configured — files go to {settings.download_dir}/")
    say(f"engine : {settings.hf_engine}"
        f"{' (falls back to the other)' if settings.hf_fallback else ''}")
    if not settings.auto_start:
        say("note   : auto-start is off — added links wait until you press Start all")

    # Mode 1: Headless / server only
    if args.no_gui:
        serve(build_app(), host, port, sock)
        return 0

    # Mode 2: Explicit browser mode or GUI not available
    if args.browser or not is_gui_available():
        if not is_gui_available() and not args.browser:
            say("pywebview not available — opening in default browser")
        webbrowser.open(url)
        serve(build_app(), host, port, sock)
        return 0

    # Mode 3: Native desktop standalone window (default)
    try:
        return launch_desktop(app=build_app(), host=host, port=port, debug=args.debug, sock=sock)
    except Exception as exc:
        say(f"Could not open desktop window ({exc}), opening browser fallback...")
        # That attempt ran the app's lifespan and took the socket down with it. Both have
        # to be built again; neither survives a shutdown. The close is for the case where
        # it failed before the server ever adopted the socket -- rebinding a port we still
        # hold ourselves would only push us onto the next one.
        try:
            sock.close()
        except OSError:
            pass
        sock, port = bind_local_port(host, port)
        webbrowser.open(f"http://{host}:{port}")
        serve(build_app(), host, port, sock)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
