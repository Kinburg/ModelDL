"""Build script to package ModelDL as a standalone single-file executable (ModelDL.exe).

Usage:
    .venv/Scripts/python scripts/build_exe.py [--console]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
    ]

    if args.console:
        # Override console setting in spec or build command
        cmd.extend(["--console", str(ROOT / "scripts" / "serve.py")])
    else:
        cmd.append(str(spec_file))

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode == 0:
        exe_path = ROOT / "dist" / "ModelDL.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 45)
            print("  BUILD SUCCESSFUL!")
            print(f"  Executable: {exe_path}")
            print(f"  Size: {size_mb:.1f} MB")
            print("=" * 45 + "\n")
        else:
            print("\nBuild finished. Check dist/ directory.")
    else:
        print(f"\nBuild failed with exit code {result.returncode}", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
