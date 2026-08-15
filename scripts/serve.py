"""Run the local web interface.

    run.cmd                        (Windows — sets everything up on first run)
    .venv/bin/python scripts/serve.py

Binds to 127.0.0.1 only. The server holds your tokens and has no authentication, so it must
not be reachable from the rest of the network.
"""

from __future__ import annotations

import argparse
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

from sfd.jobs.db import Database  # noqa: E402
from sfd.settings import Settings  # noqa: E402
from sfd.web.app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--db", default="queue.db")
    parser.add_argument("--settings", default="settings.json")
    parser.add_argument("--open", action="store_true", help="open a browser on start")
    args = parser.parse_args()

    settings = Settings.load(Path(args.settings))
    app = create_app(settings, Database(args.db))

    url = f"http://127.0.0.1:{args.port}"
    # flush=True throughout: redirected output is buffered, so a process that is killed
    # rather than exiting takes its startup log with it — leaving an empty file at exactly
    # the moment you need to know what happened.
    say = lambda line: print(line, flush=True)  # noqa: E731

    say(f"ModelDL on {url}")
    if settings.error:
        # Loud, because the consequence is quiet: files land somewhere else and nothing
        # else would say why.
        say(f"\n  !! {settings.error}\n")
    if settings.library_root:
        say(f"library: {settings.library_root} ({settings.profile})")
    else:
        say(f"library: not configured — files go to {settings.download_dir}/")
    say(f"engine : {settings.hf_engine}"
        f"{' (falls back to the other)' if settings.hf_fallback else ''}")
    if not settings.auto_start:
        say("note   : auto-start is off — added links wait until you press Start all")
    if args.open:
        webbrowser.open(url)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
