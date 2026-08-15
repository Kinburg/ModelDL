"""The subprocess engine: its protocol, its environment, and its failure modes.

The worker is replaced with a stub script so these run offline and deterministically. What
they check is the contract between the two halves — the part that breaks silently.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from sfd.core.errors import TransferFailed
from sfd.engines.hf_hub import HfHubEngine, HfHubOptions


def stub_worker(tmp_path: Path, body: str) -> Path:
    """A stand-in worker that speaks the same JSON protocol."""
    script = tmp_path / "hf_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys, time

            def emit(**p):
                sys.stdout.write(json.dumps(p) + "\\n"); sys.stdout.flush()

            job = json.loads(sys.stdin.read())
            """
        )
        + textwrap.dedent(body),
        "utf-8",
    )
    return script


def engine_using(script: Path, **option_kwargs) -> HfHubEngine:
    engine = HfHubEngine("token-value", HfHubOptions(**option_kwargs))
    # The engine locates its worker beside itself; point it at the stub instead.
    engine._worker_override = script  # type: ignore[attr-defined]
    return engine


# --- environment ------------------------------------------------------------


def test_the_token_travels_in_the_environment_not_the_command_line():
    """A command line is readable by every process on the machine; the environment is not."""
    env = HfHubOptions().environment("hf_secret")
    assert env["HF_TOKEN"] == "hf_secret"


def test_no_token_means_the_variable_is_removed(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "inherited-from-the-parent")
    assert "HF_TOKEN" not in HfHubOptions().environment(None)


@pytest.mark.parametrize(
    "kwargs, variable",
    [
        ({"disable_xet": True}, "HF_HUB_DISABLE_XET"),
        ({"sequential_writes": True}, "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"),
        ({"high_performance": True}, "HF_XET_HIGH_PERFORMANCE"),
    ],
)
def test_xet_switches_reach_the_child(kwargs, variable):
    """These are read once at import, which is the whole reason for a separate process."""
    assert HfHubOptions(**kwargs).environment(None)[variable] == "1"
    assert variable not in HfHubOptions().environment(None)


def test_output_is_unbuffered_so_progress_arrives_while_it_happens():
    assert HfHubOptions().environment(None)["PYTHONUNBUFFERED"] == "1"


# --- the protocol -----------------------------------------------------------


async def test_progress_and_completion_are_reported(tmp_path: Path):
    script = stub_worker(tmp_path, """
        emit(e="progress", bytes=1000)
        emit(e="progress", bytes=5000)
        emit(e="done", paths=[job["local_dir"] + "/model.bin"], bytes=5000)
        """)

    seen = []
    result = await engine_using(script).download_repo(
        "org/repo", tmp_path / "out", total=5000, on_progress=lambda s: seen.append(s)
    )

    assert [s.downloaded for s in seen] == [1000, 5000]
    assert seen[-1].total == 5000
    assert result.transferred == 5000
    assert result.paths[0].name == "model.bin"


async def test_an_error_line_becomes_an_exception(tmp_path: Path):
    script = stub_worker(tmp_path, """
        emit(e="error", message="RemoteEntryNotFoundError: 404")
        sys.exit(1)
        """)
    with pytest.raises(TransferFailed, match="404"):
        await engine_using(script).download_repo("org/repo", tmp_path / "out")


async def test_a_crash_without_a_message_still_fails_loudly(tmp_path: Path):
    script = stub_worker(tmp_path, """
        sys.exit(3)
        """)
    with pytest.raises(TransferFailed, match="exited with 3"):
        await engine_using(script).download_repo("org/repo", tmp_path / "out")


async def test_finishing_without_naming_a_file_is_a_failure(tmp_path: Path):
    """Exit zero having produced nothing is not success."""
    script = stub_worker(tmp_path, """
        emit(e="done", paths=[], bytes=0)
        """)
    with pytest.raises(TransferFailed, match="no files"):
        await engine_using(script).download_repo("org/repo", tmp_path / "out")


async def test_noise_on_stdout_is_ignored(tmp_path: Path):
    """Libraries print. A stray line must not take the download down with it."""
    script = stub_worker(tmp_path, """
        print("some library felt chatty")
        emit(e="progress", bytes=10)
        print("{ not json either")
        emit(e="done", paths=[job["local_dir"] + "/f.bin"], bytes=10)
        """)
    result = await engine_using(script).download_repo("org/repo", tmp_path / "out")
    assert result.transferred == 10


async def test_the_job_reaches_the_child_intact(tmp_path: Path):
    script = stub_worker(tmp_path, """
        with open(job["local_dir"] + "/job.json", "w") as fh:
            json.dump(job, fh)
        emit(e="done", paths=[job["local_dir"] + "/f.bin"], bytes=0)
        """)
    out = tmp_path / "out"
    out.mkdir()
    await engine_using(script).download_repo(
        "org/repo", out, revision="refs/pr/3", repo_type="dataset",
        allow_patterns=["*.gguf"], ignore_patterns=["*.md"],
    )

    job = json.loads((out / "job.json").read_text("utf-8"))
    assert job["repo_id"] == "org/repo"
    assert job["revision"] == "refs/pr/3"
    assert job["repo_type"] == "dataset"
    assert job["allow_patterns"] == ["*.gguf"]
    assert job["ignore_patterns"] == ["*.md"]


async def test_a_single_file_is_staged_away_from_its_destination(tmp_path: Path):
    """The client insists on reproducing the repo layout, so it cannot write straight to
    the destination the library layout picked."""
    script = stub_worker(tmp_path, """
        emit(e="done", paths=[job["target"]], bytes=1)
        """)
    destination = tmp_path / "loras" / "Pony" / "final.safetensors"
    result = await engine_using(script).download_file(
        "org/repo", "nested/dir/model.safetensors", destination
    )
    assert result.paths == [destination]


# --- cancellation -----------------------------------------------------------


async def test_cancelling_stops_the_child(tmp_path: Path):
    script = stub_worker(tmp_path, """
        emit(e="progress", bytes=1)
        time.sleep(60)
        """)
    engine = engine_using(script)
    task = asyncio.create_task(engine.download_repo("org/repo", tmp_path / "out"))
    await asyncio.sleep(1.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # No orphan left behind holding sockets and disk.
    assert engine._process is None
