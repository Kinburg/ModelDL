"""Packing the portable folder: the program goes into the archive, the user's files do not."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_exe.py"


@pytest.fixture(scope="module")
def build_exe():
    spec = importlib.util.spec_from_file_location("build_exe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def built(tmp_path: Path) -> Path:
    """A portable folder as PyInstaller leaves it — and as a run of it would leave it."""
    folder = tmp_path / "ModelDL"
    (folder / "_internal" / "webview" / "lib").mkdir(parents=True)
    (folder / "ModelDL.exe").write_bytes(b"MZ program")
    (folder / "_internal" / "python314.dll").write_bytes(b"dll")
    (folder / "_internal" / "webview" / "lib" / "Microsoft.Web.WebView2.Core.dll").write_bytes(b"dll")
    return folder


def test_the_archive_holds_the_program_in_a_folder_of_its_own(build_exe, built, tmp_path):
    archive = build_exe.pack(built, tmp_path / "out" / "ModelDL-portable.zip", "1.2.3")
    names = set(zipfile.ZipFile(archive).namelist())
    assert names == {
        "ModelDL/ModelDL.exe",
        "ModelDL/ModelDL.exe.config",
        "ModelDL/README.txt",
        "ModelDL/_internal/python314.dll",
        "ModelDL/_internal/webview/lib/Microsoft.Web.WebView2.Core.dll",
    }
    assert not archive.with_name(archive.name + ".partial").exists()


def test_what_a_run_wrote_beside_the_exe_stays_out(build_exe, built, tmp_path):
    """Settings hold the tokens. A folder that was started once, packed as it stands, would
    hand them to everybody who downloads the release."""
    (built / "settings.json").write_text('{"hf_token": "hf_secret"}')
    (built / "queue.db").write_bytes(b"history")
    (built / "previews").mkdir()
    (built / "previews" / "sample.jpg").write_bytes(b"jpg")
    (built / "downloads").mkdir()
    (built / "downloads" / "model.safetensors").write_bytes(b"weights")

    archive = build_exe.pack(built, tmp_path / "portable.zip", "1.2.3")
    names = zipfile.ZipFile(archive).namelist()
    assert not [n for n in names if not n.startswith("ModelDL/_internal/")
                and n not in ("ModelDL/ModelDL.exe", "ModelDL/ModelDL.exe.config", "ModelDL/README.txt")]
    assert all(b"hf_secret" not in zipfile.ZipFile(archive).read(n) for n in names)


def test_the_config_lets_dotnet_load_files_unpacked_from_a_download(build_exe, built, tmp_path):
    """Without it the window's .NET half refuses every assembly Windows marked as coming
    from the internet — all of them, once the archive is unpacked by Explorer."""
    archive = zipfile.ZipFile(build_exe.pack(built, tmp_path / "portable.zip", "1.2.3"))
    config = archive.read("ModelDL/ModelDL.exe.config").decode()
    assert '<loadFromRemoteSources enabled="true"/>' in config
    readme = archive.read("ModelDL/README.txt").decode()
    assert readme.startswith("ModelDL 1.2.3, portable")
    assert "\r\n" in readme


def test_a_folder_without_a_built_program_is_refused(build_exe, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        build_exe.pack(tmp_path / "empty", tmp_path / "portable.zip", "1.2.3")
    assert not (tmp_path / "portable.zip").exists()
