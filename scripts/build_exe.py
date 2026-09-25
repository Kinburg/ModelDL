"""Build ModelDL for Windows with PyInstaller.

Usage:
    .venv/Scripts/python scripts/build_exe.py [--portable] [--console]

Without --portable the result is one file, dist/ModelDL.exe. With it, the portable folder —
ModelDL.exe beside an _internal folder, quicker to start since nothing is unpacked into
%TEMP% first — packed as dist/ModelDL-portable.zip. The name carries no version, so a link
to the latest release's copy stays the same from one release to the next; README.txt inside
says which one it is.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
# PyInstaller empties the folder it builds the portable variant in, so that is a folder of
# its own and never dist/, where a built ModelDL.exe keeps its settings and history.
STAGE = ROOT / "build" / "portable"
NAME = "ModelDL"

# Beside the exe in the archive. The window is .NET (WebView2 through pythonnet), and files
# unpacked from a downloaded archive carry Windows' mark of coming from the internet, which
# makes .NET refuse to load them: the window would fail and the app open in a browser.
CONFIG = """\
<?xml version="1.0" encoding="utf-8"?>
<!-- Lets .NET load this app's own assemblies from _internal after the archive they came in
     was downloaded: unpacking passes Windows' "from the internet" mark on to every file. -->
<configuration>
  <runtime>
    <loadFromRemoteSources enabled="true"/>
  </runtime>
</configuration>
"""

README = """\
ModelDL {version}, portable

Keep this folder together: ModelDL.exe runs from the files in _internal beside it.

Settings (tokens among them), the download history and the cache of sample pictures are
kept in this folder, beside ModelDL.exe, and so are downloads while no library folder is
set. Move or copy the folder and they go with it. In a folder that cannot be written to,
such as Program Files, they are kept in %LOCALAPPDATA%\\ModelDL instead.

To update, delete _internal and unpack the new archive over this folder. Everything the app
wrote itself stays: settings.json, queue.db, previews, downloads.

Windows warns about an unrecognised app the first time it starts, as the program is not
code-signed: More info, then Run anyway.

https://github.com/Kinburg/ModelDL
"""


def version() -> str:
    sys.path.insert(0, str(ROOT))
    from sfd import __version__

    return __version__


def pack(folder: Path, archive: Path, version: str) -> Path:
    """Zip a built portable folder as ModelDL/…, the program's own files and nothing else.

    Only ModelDL.exe and _internal are taken from the folder. The app keeps its settings,
    tokens included, beside the exe: a folder that was ever started would otherwise hand
    them to whoever downloads the archive.
    """
    exe = folder / f"{NAME}.exe"
    internal = folder / "_internal"
    if not exe.is_file() or not internal.is_dir():
        raise FileNotFoundError(f"{folder} does not hold a built {NAME}.exe and its _internal")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(exe, f"{NAME}/{exe.name}")
        for path in sorted(internal.rglob("*")):
            if path.is_file():
                zipped.write(path, f"{NAME}/{path.relative_to(folder).as_posix()}")
        zipped.writestr(f"{NAME}/{NAME}.exe.config", CONFIG.replace("\n", "\r\n"))
        zipped.writestr(f"{NAME}/README.txt", README.format(version=version).replace("\n", "\r\n"))
    os.replace(partial, archive)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help=f"build the portable folder and pack it as dist/{NAME}-portable.zip",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="build with console window visible (useful for debugging)",
    )
    args = parser.parse_args()

    print("=== Building ModelDL Executable ===")

    # Ensure PyInstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Installing it into virtualenv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])

    spec_file = ROOT / "model_dl.spec"
    if not spec_file.exists():
        print(f"Error: Spec file not found at {spec_file}", file=sys.stderr)
        return 1

    # The spec reads the variant from the environment: PyInstaller takes no options of its
    # own alongside a spec file.
    env = dict(os.environ)
    env["MODELDL_ONEDIR"] = "1" if args.portable else "0"
    env["MODELDL_CONSOLE"] = "1" if args.console else "0"

    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm"]
    if args.portable:
        cmd += ["--distpath", str(STAGE), "--workpath", str(ROOT / "build" / "portable-work")]
    cmd.append(str(spec_file))

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    if args.portable:
        product = pack(STAGE / NAME, DIST / f"{NAME}-portable.zip", version())
        # The archive is the product. A folder left here would be run sooner or later, and
        # the next build empties it — along with whatever the app had written into it.
        shutil.rmtree(STAGE)
    else:
        product = DIST / f"{NAME}.exe"

    if not product.exists():
        print("\nBuild finished. Check dist/ directory.")
        return 0
    size_mb = product.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 45)
    print("  BUILD SUCCESSFUL!")
    print(f"  {'Archive' if args.portable else 'Executable'}: {product}")
    print(f"  Size: {size_mb:.1f} MB")
    print("=" * 45 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
