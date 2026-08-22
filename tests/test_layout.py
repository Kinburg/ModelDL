"""Adopting an existing model tree, and deciding where a file lands in it."""

from __future__ import annotations

import json
from pathlib import Path

from sfd.library.categories import Category
from sfd.library.classify import Verdict
from sfd.library.layout import adopt, flat
from sfd.library.sidecar import Record, read, write


def make_tree(root: Path, folders: dict[str, int]) -> None:
    """Create folders, each seeded with `n` model-shaped files."""
    for name, count in folders.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (directory / f"m{i}.safetensors").write_bytes(b"x")


# --- adoption ---------------------------------------------------------------


def test_existing_folders_are_used_as_they_are(tmp_path: Path):
    make_tree(tmp_path, {"loras": 3, "checkpoints": 1, "vae": 2})
    layout = adopt(tmp_path)
    assert layout.paths[Category.LORA] == tmp_path / "loras"
    assert layout.paths[Category.CHECKPOINT] == tmp_path / "checkpoints"
    assert layout.paths[Category.VAE] == tmp_path / "vae"


def test_synonyms_are_resolved_by_which_one_is_actually_used(tmp_path: Path):
    """ComfyUI installs accumulate both `unet` and `diffusion_models`.

    Taken from a real tree where `diffusion_models` held seventeen files and `unet` was
    empty: what the user does beats what any naming convention prefers.
    """
    make_tree(tmp_path, {"unet": 0, "diffusion_models": 17})
    layout = adopt(tmp_path)
    assert layout.paths[Category.DIFFUSION_MODEL] == tmp_path / "diffusion_models"
    assert layout.ambiguities[Category.DIFFUSION_MODEL] == ["unet"]


def test_the_populated_folder_wins_even_against_the_profile_default(tmp_path: Path):
    make_tree(tmp_path, {"unet": 12, "diffusion_models": 0})
    layout = adopt(tmp_path)
    assert layout.paths[Category.DIFFUSION_MODEL] == tmp_path / "unet"


def test_the_profile_default_breaks_a_tie_between_empty_folders(tmp_path: Path):
    make_tree(tmp_path, {"clip": 0, "text_encoders": 0})
    layout = adopt(tmp_path)
    assert layout.paths[Category.TEXT_ENCODER] == tmp_path / "text_encoders"


def test_missing_categories_fall_back_to_profile_names(tmp_path: Path):
    make_tree(tmp_path, {"loras": 1})
    layout = adopt(tmp_path)
    assert layout.paths[Category.CONTROLNET] == tmp_path / "controlnet"
    assert layout.paths[Category.LLM] == tmp_path / "LLM"


def test_unrecognised_folders_are_left_alone(tmp_path: Path):
    """A real install is full of custom-node directories we must not claim."""
    make_tree(tmp_path, {"loras": 1, "reactor": 3, "SEEDVR2": 2, "fishaudioS2": 1})
    layout = adopt(tmp_path)
    assert set(layout.ambiguities) == set()
    assert tmp_path / "reactor" not in layout.paths.values()


def test_detector_folders_that_are_not_interchangeable_stay_unclaimed(tmp_path: Path):
    """SAM checkpoints, face detectors and YOLO models are read by different nodes.

    Nothing in a bare .pt distinguishes them, so folding them into one category would file a
    segmenter under `ultralytics` with full confidence. Better to claim only what we can
    actually target.
    """
    make_tree(tmp_path, {"ultralytics": 2, "sams": 3, "sam2": 1, "facedetection": 4})
    layout = adopt(tmp_path)
    assert layout.paths[Category.DETECTION] == tmp_path / "ultralytics"
    assert Category.DETECTION not in layout.ambiguities
    for unclaimed in ("sams", "sam2", "facedetection"):
        assert tmp_path / unclaimed not in layout.paths.values()


def test_a1111_profile_uses_its_own_names(tmp_path: Path):
    layout = adopt(tmp_path, profile="a1111")
    assert layout.paths[Category.CHECKPOINT] == tmp_path / "Stable-diffusion"
    assert layout.paths[Category.LORA] == tmp_path / "Lora"


# --- placement --------------------------------------------------------------


def test_base_model_becomes_a_subfolder_where_it_helps(tmp_path: Path):
    make_tree(tmp_path, {"loras": 1})
    layout = adopt(tmp_path)
    verdict = Verdict(Category.LORA, "high", "because", base_model="Flux.1 D")
    assert layout.destination(verdict, "x.safetensors") == (
        tmp_path / "loras" / "Flux.1 D" / "x.safetensors"
    )


def test_base_model_is_not_used_where_it_would_only_scatter_files(tmp_path: Path):
    make_tree(tmp_path, {"vae": 1})
    layout = adopt(tmp_path)
    verdict = Verdict(Category.VAE, "high", "because", base_model="SDXL 1.0")
    assert layout.destination(verdict, "ae.safetensors") == (
        tmp_path / "vae" / "ae.safetensors"
    )


def test_base_model_grouping_can_be_switched_off(tmp_path: Path):
    layout = adopt(tmp_path)
    layout.group_by_base_model = False
    verdict = Verdict(Category.LORA, "high", "because", base_model="Pony")
    assert layout.destination(verdict, "x.safetensors").parent.name == "loras"


def test_a_base_model_name_cannot_escape_the_folder(tmp_path: Path):
    layout = adopt(tmp_path)
    verdict = Verdict(Category.LORA, "high", "r", base_model="../../etc/passwd")
    destination = layout.destination(verdict, "x.safetensors")
    assert ".." not in destination.parts
    assert tmp_path in destination.parents


def test_flat_layout_puts_everything_in_one_place(tmp_path: Path):
    layout = flat(tmp_path)
    verdict = Verdict(Category.LORA, "high", "r", base_model="Pony")
    assert layout.destination(verdict, "x.safetensors") == tmp_path / "x.safetensors"


def test_layout_survives_a_round_trip(tmp_path: Path):
    make_tree(tmp_path, {"unet": 0, "diffusion_models": 5, "loras": 2})
    layout = adopt(tmp_path)
    restored = type(layout).from_dict(json.loads(json.dumps(layout.to_dict())))
    assert restored.paths == layout.paths
    assert restored.ambiguities == layout.ambiguities
    assert restored.root == layout.root


# --- sidecars ---------------------------------------------------------------


def test_sidecar_keeps_the_things_that_are_otherwise_lost(tmp_path: Path):
    model = tmp_path / "style.safetensors"
    model.write_bytes(b"weights")
    verdict = Verdict(Category.LORA, "high", "training metadata names networks.lora",
                      base_model="Pony")
    record = Record(
        filename=model.name,
        provider="civitai",
        source_url="https://civitai.com/api/download/models/1558543?fileId=1458454",
        sha256="df7c" + "0" * 60,
        size=57420828,
        meta={
            "model_id": 1234, "version_id": 5678,
            "trained_words": ["abstract painting", "oil on canvas"],
            "model_name": "Abstract Painting", "version_name": "v1",
        },
    )
    write(model, verdict, record, raw={"id": 5678, "files": []})

    saved = read(model)
    assert saved["usage"]["trigger_words"] == ["abstract painting", "oil on canvas"]
    assert saved["usage"]["base_model"] == "Pony"
    assert saved["integrity"]["sha256"].startswith("df7c")
    assert saved["source"]["page"] == "https://civitai.com/models/1234?modelVersionId=5678"
    assert saved["classification"]["category"] == "lora"

    # The filename existing tools already look for.
    assert (tmp_path / "style.civitai.info").exists()


def test_sidecar_keeps_the_prompts_the_samples_were_made_with(tmp_path: Path):
    """Trigger words say which tokens wake a LoRA up. The sample prompts say what a working
    prompt around them looks like, and they exist nowhere once the model page is gone."""
    model = tmp_path / "ollie.safetensors"
    model.write_bytes(b"weights")
    record = Record(
        filename=model.name,
        provider="civitai",
        meta={
            "previews": [
                {
                    "url": "https://image.civitai.com/b/one/width=450/a.jpeg",
                    "type": "image", "nsfw": False,
                    "meta": {"prompt": "sick ollie, a cat on a skateboard", "seed": 1},
                },
                {"url": "https://image.civitai.com/b/two/width=450/b.mp4", "type": "video"},
            ]
        },
    )
    write(model, Verdict(Category.LORA, "high", "named as one"), record)

    kept = read(model)["previews"]
    assert kept[0]["meta"]["prompt"] == "sick ollie, a cat on a skateboard"
    assert kept[1]["type"] == "video" and kept[1]["nsfw"] is False


def test_sidecar_records_a_disagreement_for_later(tmp_path: Path):
    model = tmp_path / "ollie.safetensors"
    model.write_bytes(b"weights")
    verdict = Verdict(
        Category.DIFFUSION_MODEL, "high", "denoiser weights only",
        base_model="Krea 2", disagreement=Category.CHECKPOINT,
    )
    write(model, verdict, Record(filename=model.name, provider="civitai"))

    saved = read(model)
    assert saved["classification"]["category"] == "diffusion_model"
    assert saved["classification"]["service_called_it"] == "checkpoint"


def test_no_civitai_info_for_a_huggingface_download(tmp_path: Path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")
    write(
        model,
        Verdict(Category.LLM, "high", "arch"),
        Record(filename=model.name, provider="huggingface", meta={"repo_id": "org/name"}),
        raw={"irrelevant": True},
    )
    assert not (tmp_path / "m.civitai.info").exists()
    assert read(model)["source"]["page"] == "https://huggingface.co/org/name"


def test_reading_a_missing_sidecar_is_not_an_error(tmp_path: Path):
    assert read(tmp_path / "nothing.safetensors") is None


def test_records_can_be_collected_away_from_the_models(tmp_path: Path):
    """Keeping the library free of .json clutter, without losing which model is which."""
    library = tmp_path / "models"
    records = tmp_path / "records"
    model = library / "loras" / "Pony" / "style.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")

    write(
        model,
        Verdict(Category.LORA, "high", "r"),
        Record(filename=model.name, provider="civitai"),
        sidecar_dir=records,
        library_root=library,
    )

    # The folder structure is mirrored, so the record can still be traced to its model.
    assert (records / "loras" / "Pony" / "style.safetensors.json").exists()
    assert not (model.parent / "style.safetensors.json").exists()
    assert read(model, sidecar_dir=records, library_root=library)["filename"] == "style.safetensors"


def test_collected_records_of_identically_named_models_do_not_collide(tmp_path: Path):
    """Two `model.safetensors` in different categories are different files.

    Flattening them into one records directory would leave one of them overwritten.
    """
    library = tmp_path / "models"
    records = tmp_path / "records"
    written = []
    for category, folder in ((Category.LORA, "loras"), (Category.VAE, "vae")):
        model = library / folder / "model.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"x")
        written.append(
            write(
                model, Verdict(category, "high", "r"),
                Record(filename=model.name, provider="direct"),
                sidecar_dir=records, library_root=library,
            )
        )

    assert written[0] != written[1]
    assert all(p.exists() for p in written)


def test_compat_files_are_never_moved(tmp_path: Path):
    """A1111 and ComfyUI model managers look for these beside the model and nowhere else."""
    library = tmp_path / "models"
    model = library / "loras" / "style.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")

    write(
        model, Verdict(Category.LORA, "high", "r"),
        Record(filename=model.name, provider="civitai"),
        raw={"id": 1}, sidecar_dir=tmp_path / "records", library_root=library,
    )
    assert (model.parent / "style.civitai.info").exists()


def test_trigger_words_land_in_a_txt_the_loaders_read(tmp_path: Path):
    """`<model>.txt` is activation text: loaders paste it into the prompt verbatim.

    So it holds the trigger words and nothing else — a URL in here would end up inside
    someone's generation.
    """
    model = tmp_path / "style.safetensors"
    model.write_bytes(b"x")
    write(
        model, Verdict(Category.LORA, "high", "r"),
        Record(
            filename=model.name, provider="civitai",
            source_url="https://civitai.com/api/download/models/1",
            meta={"trained_words": ["bl00m", "bloom girl"], "model_id": 1, "version_id": 2},
        ),
    )

    text = (tmp_path / "style.txt").read_text("utf-8")
    assert text.strip() == "bl00m, bloom girl"
    assert "http" not in text
    # The link is still recorded, just not where a prompt would pick it up.
    assert read(model)["source"]["page"].startswith("https://")


def test_a_comma_joined_trigger_string_is_split_and_tidied(tmp_path: Path):
    """Civitai often returns every word inside one string, with trailing separators."""
    from sfd.library.sidecar import normalise_triggers

    assert normalise_triggers(["abstractionism, brush stroke, traditional media, "]) == [
        "abstractionism", "brush stroke", "traditional media",
    ]
    assert normalise_triggers(["bl00m", "BL00M", " bl00m "]) == ["bl00m"]
    assert normalise_triggers([]) == []
    assert normalise_triggers(None) == []


def test_no_txt_for_a_model_without_triggers(tmp_path: Path):
    model = tmp_path / "checkpoint.safetensors"
    model.write_bytes(b"x")
    write(
        model, Verdict(Category.CHECKPOINT, "high", "r"),
        Record(filename=model.name, provider="civitai", meta={"trained_words": []}),
    )
    assert not (tmp_path / "checkpoint.txt").exists()


def test_the_trigger_txt_can_be_turned_off(tmp_path: Path):
    model = tmp_path / "style.safetensors"
    model.write_bytes(b"x")
    write(
        model, Verdict(Category.LORA, "high", "r"),
        Record(filename=model.name, provider="civitai", meta={"trained_words": ["bl00m"]}),
        triggers=False,
    )
    assert not (tmp_path / "style.txt").exists()


def test_compat_files_can_be_turned_off(tmp_path: Path):
    model = tmp_path / "style.safetensors"
    model.write_bytes(b"x")
    write(
        model, Verdict(Category.LORA, "high", "r"),
        Record(filename=model.name, provider="civitai"), raw={"id": 1}, compat=False,
    )
    assert not (tmp_path / "style.civitai.info").exists()
    assert read(model) is not None
