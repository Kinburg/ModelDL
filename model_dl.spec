# -*- mode: python ; coding: utf-8 -*-
#
# Built by scripts/build_exe.py, which picks the variant through the environment — a spec
# file takes no options of its own on the PyInstaller command line:
#   MODELDL_ONEDIR=1   the portable folder, ModelDL.exe beside _internal, instead of one exe
#   MODELDL_CONSOLE=1  with a console window, for seeing what the app prints
import os
from pathlib import Path
import sys

block_cipher = None

ROOT = Path.cwd()
ONEDIR = os.environ.get("MODELDL_ONEDIR") == "1"
CONSOLE = os.environ.get("MODELDL_CONSOLE") == "1"

datas = [
    (str(ROOT / "sfd" / "web" / "static"), "sfd/web/static"),
]

hiddenimports = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "fastapi",
    "starlette",
    "starlette.responses",
    "starlette.routing",
    "pydantic",
    "httpx",
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr_loader",
    "pythonnet",
    "tkinter",
    "tkinter.filedialog",
]

a = Analysis(
    ["scripts/serve.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "pytest_asyncio"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One file carries everything and unpacks it into %TEMP% on every start; the folder variant
# keeps it unpacked in _internal beside the exe, which COLLECT puts together.
bundled = [] if ONEDIR else [a.binaries, a.zipfiles, a.datas]

exe = EXE(
    pyz,
    a.scripts,
    *bundled,
    [],
    exclude_binaries=ONEDIR,
    name="ModelDL",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=CONSOLE,  # Windowed GUI application, unless asked otherwise
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "sfd" / "logo" / "logo.ico"),
)

if ONEDIR:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="ModelDL",
    )
