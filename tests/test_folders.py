"""Choosing a real folder when the classifier is not sure.

The canonical categories cannot express "put this in `sams`" — `categories.py` refuses to
claim the folders custom nodes bring, on purpose — so the picker offers the tree as it
really is, and these tests are mostly about the order it offers it in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.library import folders
from sfd.library.categories import Category
from sfd.library.layout import adopt
from sfd.settings import Settings
from sfd.web.app import create_app


def build(root: Path, tree: dict[str, int]) -> None:
    """Create folders holding `n` model-shaped files each."""
    for relative, count in tree.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"model{index}.safetensors").write_bytes(b"x")


def offered(root: Path, category=None, base_model=None, filename=None) -> list[str]:
    return [f.relative for f in folders.offer(adopt(root), category, base_model, filename)]


# --- ranking ----------------------------------------------------------------


def test_the_layouts_own_answer_comes_first(tmp_path: Path):
    build(tmp_path, {"loras": 3, "checkpoints": 40, "sams": 2})
    assert offered(tmp_path, Category.LORA)[0] == "loras"


def test_the_base_model_folder_is_offered_ahead_of_the_bare_kind(tmp_path: Path):
    """With grouping on, `loras/Pony` is where the file would actually have gone, so it is
    the row that has to be first — the bare `loras` is a different decision."""
    build(tmp_path, {"loras/Pony": 12, "loras/Flux": 4})
    ranked = offered(tmp_path, Category.LORA, "Pony")
    assert ranked[0] == "loras/Pony"
    assert "loras" in ranked and "loras/Flux" in ranked


def test_folders_we_refuse_to_claim_are_still_offered(tmp_path: Path):
    """The whole point: a SAM checkpoint has a right home, and it is not `other`."""
    build(tmp_path, {"sams": 4, "insightface": 1, "loras": 0})
    assert "sams" in offered(tmp_path, Category.DETECTION)


def test_a_folder_named_in_the_filename_ranks_near_the_top(tmp_path: Path):
    """The classifier is unsure precisely because the file gives it nothing to go on. The
    library's own folder names are the signal that is left."""
    build(tmp_path, {"ultralytics": 6, "sams": 1, "loras/Pony": 9})
    ranked = offered(
        tmp_path, Category.DETECTION, "Pony", filename="mystery_sam_model.safetensors"
    )
    assert ranked[1] == "sams", "after the layout's own answer, before everything else"
    assert ranked.index("sams") < ranked.index("loras/Pony")


def test_only_a_whole_folder_name_counts_as_named_in_the_file(tmp_path: Path):
    """Otherwise one word puts a style LoRA in `style_models` — the confident misfiling this
    dialog exists to let a person correct."""
    build(tmp_path, {"loras": 3, "style_models": 1})
    ranked = offered(tmp_path, Category.LORA, filename="some_style_lora.safetensors")
    assert ranked.index("loras") < ranked.index("style_models")


def test_the_runner_up_of_an_ambiguous_kind_ranks_high(tmp_path: Path):
    """`unet` and `diffusion_models` both mean the same thing; adoption picks one by what is
    in use, and this is where that call gets overridden by hand."""
    build(tmp_path, {"diffusion_models": 10, "unet": 2})
    ranked = offered(tmp_path, Category.DIFFUSION_MODEL)
    assert ranked[:2] == ["diffusion_models", "unet"]


def test_folders_in_use_outrank_empty_ones(tmp_path: Path):
    build(tmp_path, {"aardvark": 0, "zebra": 7})
    ranked = offered(tmp_path)
    assert ranked.index("zebra") < ranked.index("aardvark")


def test_files_below_a_folder_count_towards_it(tmp_path: Path):
    """A library where every model sits in a base-model subfolder would otherwise report
    every top-level folder as empty and sort them alphabetically."""
    build(tmp_path, {"checkpoints/Krea 2": 5})
    by_name = {f.relative: f for f in folders.offer(adopt(tmp_path))}
    assert by_name["checkpoints"].models == 5
    assert by_name["checkpoints/Krea 2"].models == 5


def test_a_kind_with_no_folder_yet_is_offered_last_but_offered(tmp_path: Path):
    build(tmp_path, {"loras": 1})
    ranked = offered(tmp_path, Category.LORA)
    assert "embeddings" in ranked, "the file may be the first embedding in this library"
    assert ranked.index("embeddings") > ranked.index("loras")

    not_yet = next(f for f in folders.offer(adopt(tmp_path)) if f.relative == "embeddings")
    assert not_yet.exists is False


def test_a_base_model_subfolder_carries_no_kind_of_its_own(tmp_path: Path):
    """Which is what stops "remember this" from sending every checkpoint into one base
    model's folder."""
    build(tmp_path, {"checkpoints/Krea 2": 1})
    by_name = {f.relative: f for f in folders.offer(adopt(tmp_path))}
    assert by_name["checkpoints"].category is Category.CHECKPOINT
    assert by_name["checkpoints/Krea 2"].category is None


def test_hidden_and_cache_folders_are_left_out(tmp_path: Path):
    build(tmp_path, {".git": 1, "__pycache__": 1, "loras": 1})
    ranked = offered(tmp_path)
    assert ".git" not in ranked and "__pycache__" not in ranked


def test_nothing_is_offered_for_a_root_that_does_not_exist(tmp_path: Path):
    ranked = folders.offer(adopt(tmp_path / "missing"))
    assert all(not f.exists for f in ranked), "only the not-created-yet homes"


# --- resolving a typed path -------------------------------------------------


def test_a_typed_folder_resolves_under_the_root(tmp_path: Path):
    assert folders.resolve_inside(tmp_path, "sams") == (tmp_path / "sams").resolve()
    assert folders.resolve_inside(tmp_path, "loras/Pony") == (tmp_path / "loras/Pony").resolve()
    # Windows separators arrive from a Windows browser.
    assert folders.resolve_inside(tmp_path, r"loras\Pony") == (tmp_path / "loras/Pony").resolve()


def test_a_path_that_escapes_the_root_is_refused(tmp_path: Path):
    for attempt in ("..", "../outside", "loras/../..", "/etc", "C:\\Windows", "", "   "):
        assert folders.resolve_inside(tmp_path, attempt) is None, attempt


def test_the_root_itself_is_not_a_folder_choice(tmp_path: Path):
    """Dropping a model at the top of the library is the one outcome the whole layout
    exists to avoid."""
    assert folders.resolve_inside(tmp_path, ".") is None


# --- through the page -------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    library = tmp_path / "library"
    build(library, {"loras": 2, "checkpoints": 5, "sams": 1})
    settings = Settings(
        library_root=str(library),
        download_dir=str(tmp_path / "downloads"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database  # type: ignore[attr-defined]
        test.library = library  # type: ignore[attr-defined]
        test.settings = settings  # type: ignore[attr-defined]
        yield test


def unsure(database: Database, **overrides):
    values = {
        "state": "blocked",
        "source": "https://example.com/model.safetensors",
        "provider": "direct",
        "identity": {"provider": "direct", "ref": {"url": "https://example.com/m"}},
        "filename": "mystery.safetensors",
        "category": "lora",
        "base_model": "Pony",
        "dest": "unset",
    }
    values.update(overrides)
    return database.add(**values)


def test_the_page_is_offered_the_real_tree(client):
    task = unsure(client.database)
    body = client.get(f"/api/tasks/{task.id}/folders").json()

    assert body["category"] == "lora"
    offered_now = [f["relative"] for f in body["folders"]]
    assert offered_now[0] == "loras/Pony", "where it would have gone, grouping included"
    assert "sams" in offered_now
    assert body["folders"][0]["reason"]


def test_a_chosen_folder_is_taken_exactly_as_given(client):
    """Nothing appended, not even base-model grouping: a path quietly extended underneath
    the person who picked it is how a successful download goes missing."""
    task = unsure(client.database)
    client.post(f"/api/tasks/{task.id}/confirm", json={"folder": "sams"})

    filed = client.database.get(task.id)
    assert filed.dest == str(client.library / "sams" / "mystery.safetensors")
    # A released task is claimed within milliseconds and may already have failed against the
    # unreachable test URL; what matters is that it is no longer parked.
    assert filed.state != "blocked"


def test_a_folder_that_names_a_kind_also_says_what_the_file_is(client):
    task = unsure(client.database, category="other")
    client.post(f"/api/tasks/{task.id}/confirm", json={"folder": "checkpoints"})

    filed = client.database.get(task.id)
    assert filed.category == "checkpoint"
    assert filed.confidence == "high"


def test_an_unclaimed_folder_leaves_the_guess_standing(client):
    """`sams` means nothing to us as a kind, so inventing a better-sounding category for the
    record would be making it up."""
    task = unsure(client.database, category="detection")
    client.post(f"/api/tasks/{task.id}/confirm", json={"folder": "sams"})

    assert client.database.get(task.id).category == "detection"


def test_a_folder_outside_the_library_is_refused(client):
    task = unsure(client.database)
    for attempt in ("../elsewhere", "C:\\Windows\\System32", "/etc"):
        response = client.post(f"/api/tasks/{task.id}/confirm", json={"folder": attempt})
        assert response.status_code == 400, attempt
    assert client.database.get(task.id).state == "blocked"


def test_remembering_a_choice_moves_the_kind_for_good(client):
    task = unsure(client.database)
    client.post(
        f"/api/tasks/{task.id}/confirm", json={"folder": "loras", "remember": True}
    )

    assert client.settings.layout_overrides["lora"] == str(client.library / "loras")
    # And the mapping the page shows agrees, or it would disagree with where files go.
    assert client.get("/api/layout").json()["paths"]["lora"] == str(client.library / "loras")


def test_remembering_is_ignored_for_a_folder_that_names_no_kind(client):
    """`checkpoints/Krea 2` as the home of every future checkpoint is not what was asked."""
    task = unsure(client.database, category="checkpoint")
    client.post(
        f"/api/tasks/{task.id}/confirm",
        json={"folder": "checkpoints/Krea 2", "remember": True},
    )

    assert client.settings.layout_overrides == {}
    assert client.database.get(task.id).dest.endswith(str(Path("Krea 2") / "mystery.safetensors"))


def test_the_categories_still_work_on_their_own(client):
    """The folder is an addition, not a replacement — the old answer still has to land."""
    task = unsure(client.database, category="other")
    client.post(f"/api/tasks/{task.id}/confirm", json={"category": "vae"})

    filed = client.database.get(task.id)
    assert filed.category == "vae"
    assert filed.dest == str(client.library / "vae" / "mystery.safetensors")
