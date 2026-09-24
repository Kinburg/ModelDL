"""What can be learned about a model on disk without asking anybody.

For a file this app downloaded, the service already said what it is. For one that arrived
some other way — copied off another machine, fetched by a browser, put there by another tool
— the answer has to come from the file itself and from what other tools left beside it, and
it is more than one might expect:

  The header.   Every tensor's name, type and shape, so the kind of model (the classifier's
                job), its precision and its size in parameters. GGUF adds the architecture
                and the quantisation outright.
  The trainer.  kohya writes its whole training configuration into a LoRA: the base model,
                the network's rank, and how often each tag appeared in the captions — which
                for most LoRAs is as close to the trigger words as a file can get.
  The spec.     `modelspec.*`, Stability's metadata standard: a title, an author, a
                description, sometimes the trigger phrase and a thumbnail.
  Neighbours.   A `.civitai.info` from Civitai Helper is the whole Civitai record; A1111
                writes its description and activation text to `<model>.json`; LoRA Manager
                and Stability Matrix have their own. Any picture named after the model.

Everything here reads, nothing writes, and nothing touches the network. Reads are bounded:
a header is at most a few megabytes, a sidecar at most a few more.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from ..providers.civitai import _describe_files
from .categories import ALIASES, Category
from .classify import Verdict, classify
from .sniff import Sniff, _Reader, _Truncated

MAX_HEADER = 32 * 1024 * 1024
MAX_SIDECAR = 8 * 1024 * 1024
GGUF_FIRST_READ = 256 * 1024
GGUF_MAX_READ = 8 * 1024 * 1024
TOP_TAGS = 16
MAX_TEXT = 2000

# llama.cpp's `llama_ftype`: what `general.file_type` means.
GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE",
}

# safetensors dtypes, as a person would write them.
DTYPES = {
    "F64": "fp64", "F32": "fp32", "F16": "fp16", "BF16": "bf16",
    "F8_E4M3": "fp8 e4m3", "F8_E5M2": "fp8 e5m2", "F8_E8M0": "fp8 e8m0",
    "F4": "fp4", "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}

# What kohya writes into a LoRA, under shorter names.
KOHYA = {
    "ss_network_module": "network",
    "ss_network_dim": "dim",
    "ss_network_alpha": "alpha",
    "ss_base_model_version": "base_model",
    "ss_sd_model_name": "trained_on",
    "ss_output_name": "output_name",
    "ss_resolution": "resolution",
    "ss_num_train_images": "images",
    "ss_epoch": "epoch",
    "ss_num_epochs": "epochs",
    "ss_steps": "steps",
    "ss_learning_rate": "learning_rate",
    "ss_training_comment": "comment",
}

MODELSPEC = (
    "title", "architecture", "author", "description", "date", "resolution",
    "trigger_phrase", "usage_hint", "license", "implementation", "prediction_type",
)

IMAGE_SUFFIXES = (
    ".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp", ".preview.gif",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
)


# --- the header ---------------------------------------------------------------


def inspect(path: Path) -> tuple[dict[str, Any], Sniff]:
    """A summary of the file's own header, and the raw reading the classifier takes."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if head[:4] == b"GGUF":
                return _gguf(handle)
            if len(head) == 8:
                summary = _safetensors(handle, head)
                if summary is not None:
                    return summary
    except OSError as exc:
        return {"format": None, "error": str(exc)}, Sniff()

    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return {"format": "onnx"}, Sniff()
    if head[:4] == b"PK\x03\x04" or head[:1] == b"\x80":
        # A pickle, zipped or not. Nothing in it is read: unpickling is how a model file
        # runs code, and the header of a pickle is not a header.
        return {"format": "pickle"}, Sniff()
    return {"format": None}, Sniff()


def _safetensors(handle, head: bytes) -> tuple[dict[str, Any], Sniff] | None:
    (length,) = struct.unpack("<Q", head)
    if not 2 <= length <= MAX_HEADER:
        return None
    try:
        payload = json.loads(handle.read(length).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_meta = payload.pop("__metadata__", None)
    metadata = {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}

    weights: dict[str, int] = {}
    params = 0
    for info in payload.values():
        if not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        try:
            count = math.prod(int(n) for n in shape) if shape else 1
        except (TypeError, ValueError):
            continue
        params += count
        dtype = str(info.get("dtype") or "?")
        weights[dtype] = weights.get(dtype, 0) + count

    summary: dict[str, Any] = {
        "format": "safetensors",
        "tensors": len(payload),
        "params": params,
        "precision": _precision(weights, params),
    }
    summary.update(_from_metadata(metadata))
    sniffed = Sniff(format="safetensors", tensor_names=list(payload), metadata=metadata)
    return summary, sniffed


def _precision(weights: dict[str, int], total: int) -> str | None:
    """The type most of the weights are stored in, which is what "fp8" or "bf16" in a
    filename claims and not always what the file holds."""
    if not total or not weights:
        return None
    ranked = sorted(weights.items(), key=lambda item: -item[1])
    first, share = ranked[0]
    label = DTYPES.get(first, first.lower())
    if share / total >= 0.9 or len(ranked) == 1:
        return label
    second = DTYPES.get(ranked[1][0], ranked[1][0].lower())
    return f"{label} + {second}"


def _from_metadata(metadata: dict[str, str]) -> dict[str, Any]:
    found: dict[str, Any] = {}

    kohya = {short: _clip(metadata[long]) for long, short in KOHYA.items() if metadata.get(long)}
    if kohya:
        found["kohya"] = kohya
    tags = _tags(metadata.get("ss_tag_frequency"))
    if tags:
        found["tags"] = tags

    spec = {
        name: _clip(metadata[f"modelspec.{name}"])
        for name in MODELSPEC if metadata.get(f"modelspec.{name}")
    }
    if spec:
        found["modelspec"] = spec
    if metadata.get("modelspec.thumbnail", "").startswith("data:image/"):
        found["thumbnail"] = True

    title = spec.get("title") or kohya.get("output_name")
    if title:
        found["title"] = title
    if spec.get("trigger_phrase"):
        found["trigger_phrase"] = spec["trigger_phrase"]
    if spec.get("description"):
        found["description"] = spec["description"]
    return found


def _tags(raw: str | None) -> list[list[Any]]:
    """The tags a LoRA was trained on, most frequent first.

    `ss_tag_frequency` is `{dataset folder: {tag: count}}`. Summed across folders and
    sorted, the top of it is usually the trigger word and the tags that described every
    image — which is what someone writing a prompt for the LoRA needs to know.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    totals: dict[str, int] = {}
    for folder in data.values():
        if not isinstance(folder, dict):
            continue
        for tag, count in folder.items():
            word = str(tag).strip()
            if not word:
                continue
            try:
                totals[word] = totals.get(word, 0) + int(count)
            except (TypeError, ValueError):
                continue
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [[tag, count] for tag, count in ranked[:TOP_TAGS]]


def _gguf(handle) -> tuple[dict[str, Any], Sniff]:
    """Everything in a GGUF before its tokenizer: the part that says what the model is.

    The tokenizer comes after, as arrays of a hundred thousand strings, and nothing in it
    helps answer the question — so reading stops there. It is read in growing steps because
    how far "there" is depends on the model.
    """
    budget = GGUF_FIRST_READ
    while True:
        handle.seek(0)
        data = handle.read(budget)
        try:
            return _parse_gguf(data)
        except _Truncated as exc:
            if budget >= GGUF_MAX_READ or len(data) < budget:
                return _parse_gguf(data, partial=True)
            budget = min(GGUF_MAX_READ, max(budget * 2, exc.needed))


def _parse_gguf(data: bytes, partial: bool = False) -> tuple[dict[str, Any], Sniff]:
    reader = _Reader(data)
    summary: dict[str, Any] = {"format": "gguf"}
    values: dict[str, Any] = {}
    try:
        reader.skip(4)
        version = reader.u32()
        tensors = reader.u64()
        count = reader.u64()
        summary["tensors"] = tensors
        if version not in (1, 2, 3):
            return summary, Sniff(format="gguf")
        for _ in range(min(count, 4096)):
            name = reader.string()
            if name.startswith("tokenizer."):
                break
            value = reader.value()
            if isinstance(value, (str, int, float, bool)):
                values[name] = value
    except _Truncated:
        if not partial:
            raise
    except (UnicodeDecodeError, struct.error, ValueError):
        pass

    arch = str(values.get("general.architecture") or "") or None
    summary["architecture"] = arch
    for key, name in (("general.name", "name"), ("general.size_label", "size_label"),
                      ("general.basename", "basename"), ("general.finetune", "finetune"),
                      ("general.author", "author"), ("general.license", "license")):
        if values.get(key) not in (None, ""):
            summary[name] = _clip(str(values[key]))
    file_type = values.get("general.file_type")
    if isinstance(file_type, int) and not isinstance(file_type, bool):
        summary["quant"] = GGUF_FILE_TYPES.get(file_type, f"type {file_type}")
        summary["precision"] = summary["quant"]
    if arch:
        for key, name in (("context_length", "context"), ("block_count", "layers"),
                          ("expert_count", "experts")):
            value = values.get(f"{arch}.{key}")
            if isinstance(value, int) and not isinstance(value, bool):
                summary[name] = value
    if summary.get("name"):
        summary["title"] = summary["name"]

    metadata = {k: str(v) for k, v in values.items() if k.startswith("general.")}
    return summary, Sniff(format="gguf", metadata=metadata, architecture=arch)


def thumbnail(path: Path) -> tuple[bytes, str] | None:
    """The picture a safetensors file carries inside it, as `modelspec.thumbnail`."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) < 8:
                return None
            (length,) = struct.unpack("<Q", head)
            if not 2 <= length <= MAX_HEADER:
                return None
            payload = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    uri = ((payload or {}).get("__metadata__") or {}).get("modelspec.thumbnail", "")
    match = re.match(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", str(uri), re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(2), validate=False), match.group(1).lower()
    except (binascii.Error, ValueError):
        return None


# --- the neighbours -----------------------------------------------------------


def local_image(path: Path) -> Path | None:
    """A picture named after the model, the way model managers look for one."""
    for suffix in IMAGE_SUFFIXES:
        candidate = path.with_name(path.stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def neighbours(path: Path, size: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """What other tools wrote beside the model: (extras, meta).

    `meta` is filled only from a Civitai record — `.civitai.info`, or the copy inside a LoRA
    Manager file — and has the same shape as the meta of a model downloaded from Civitai
    here, so everything that works for one works for the other: the samples with their
    prompts, the trigger words, the page link.
    """
    extras: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    image = local_image(path)
    if image is not None:
        extras["image"] = image.name

    info = _read_json(path.with_name(path.stem + ".civitai.info"))
    if info:
        meta = _civitai_meta(info, path, size)
        extras["sources"] = ["civitai.info"]

    a1111 = _read_json(path.with_name(path.stem + ".json"))
    if a1111 and ("activation text" in a1111 or "sd version" in a1111 or "preferred weight" in a1111):
        if a1111.get("activation text"):
            extras["activation_text"] = _clip(str(a1111["activation text"]))
        if a1111.get("negative text"):
            extras["negative_text"] = _clip(str(a1111["negative text"]))
        if a1111.get("preferred weight") not in (None, "", 0):
            extras["preferred_weight"] = a1111["preferred weight"]
        if a1111.get("description"):
            extras["description"] = _clip(str(a1111["description"]))
        if a1111.get("notes"):
            extras["their_notes"] = _clip(str(a1111["notes"]))
        if a1111.get("sd version") and str(a1111["sd version"]).lower() != "unknown":
            extras["sd_version"] = str(a1111["sd version"])
        extras.setdefault("sources", []).append("A1111")

    manager = _read_json(path.with_name(path.stem + ".metadata.json"))
    if manager:
        if manager.get("model_name"):
            extras["model_name"] = _clip(str(manager["model_name"]))
        if manager.get("base_model"):
            extras["base_model"] = str(manager["base_model"])
        if manager.get("notes"):
            extras["their_notes"] = _clip(str(manager["notes"]))
        if isinstance(manager.get("tags"), list):
            extras["their_tags"] = [str(t) for t in manager["tags"][:20]]
        civitai = manager.get("civitai")
        if isinstance(civitai, dict) and not meta and (civitai.get("id") or civitai.get("files")):
            meta = _civitai_meta(civitai, path, size)
        extras.setdefault("sources", []).append("LoRA Manager")

    matrix = _read_json(path.with_name(path.stem + ".cm-info.json"))
    if matrix:
        pick = lambda *names: next((matrix[n] for n in names if matrix.get(n)), None)  # noqa: E731
        if pick("ModelName", "modelName"):
            extras["model_name"] = _clip(str(pick("ModelName", "modelName")))
        if pick("VersionName", "ModelVersionName", "versionName"):
            extras["version_name"] = _clip(str(pick("VersionName", "ModelVersionName", "versionName")))
        if pick("BaseModel", "baseModel"):
            extras["base_model"] = str(pick("BaseModel", "baseModel"))
        words = pick("TrainedWords", "trainedWords")
        if words:
            extras["trained_words"] = words
        extras.setdefault("sources", []).append("Stability Matrix")

    return extras, meta


def _civitai_meta(version: dict[str, Any], path: Path, size: int | None) -> dict[str, Any]:
    """The meta a Civitai download of this very file would have carried."""
    try:
        files = _describe_files(version)
    except Exception:  # noqa: BLE001 - someone else's file, in whatever shape it is in
        files = []
    chosen = None
    name = path.name.lower()
    for entry in files:
        if entry.filename.lower() == name:
            chosen = entry
            break
    if chosen is None and size:
        for entry in files:
            if entry.size and abs(entry.size - size) <= 4096:
                chosen = entry
                break
    if chosen is None:
        chosen = next((f for f in files if f.primary), files[0] if files else None)
    if chosen is None:
        model = version.get("model")
        model = model if isinstance(model, dict) else {}
        return {
            "source": "civitai",
            "version_id": version.get("id"),
            "version_name": version.get("name"),
            "model_id": version.get("modelId"),
            "model_name": model.get("name"),
            "model_type": model.get("type"),
            "base_model": version.get("baseModel"),
            "trained_words": version.get("trainedWords") or [],
        }
    meta = dict(chosen.meta)
    meta["host"] = "civitai.com"
    meta["file_id"] = chosen.file_id
    if chosen.sha256:
        meta["sha256"] = chosen.sha256
    return meta


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_SIDECAR:
            return None
        data = json.loads(path.read_text("utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


# --- deciding what it is ------------------------------------------------------


# Kinds one file is legitimately used as, depending on what loads it: a language model
# as an image model's text encoder, a vision tower as part of one, a multimodal projector
# kept beside the language model it belongs to. Between these, the folder is right.
INTERCHANGEABLE = frozenset({Category.LLM, Category.TEXT_ENCODER, Category.CLIP_VISION})


def interchangeable(a: Category | None, b: Category | None) -> bool:
    return a in INTERCHANGEABLE and b in INTERCHANGEABLE


def folder_kind(path: Path, roots: list[Path]) -> tuple[Category | None, str | None, str | None]:
    """What the folders a model sits in say about it: (kind, that folder's name, the
    folder under it).

    `loras/Krea 2/style.safetensors` says LoRA, and — the layout files base models into
    subfolders of a kind — "Krea 2" is very likely its base model.
    """
    relative = None
    for root in roots:
        try:
            relative = path.relative_to(root)
            break
        except ValueError:
            continue
    parts = list(relative.parts[:-1]) if relative is not None else [path.parent.name]
    for index, part in enumerate(parts):
        kind = ALIASES.get(part.lower())
        if kind is not None:
            below = parts[index + 1] if index + 1 < len(parts) else None
            if below is not None and ALIASES.get(below.lower()) is not None:
                below = None
            return kind, part, below
    return None, None, None


def judge(
    path: Path,
    sniffed: Sniff,
    summary: dict[str, Any],
    meta: dict[str, Any],
    extras: dict[str, Any],
    roots: list[Path],
) -> Verdict:
    """What a found model is, from everything above.

    The same cascade as a download — the header over the service's label — with one more
    witness a download does not have: the folder. Somebody put the file there, and for the
    files the header cannot read (a `.pth`, an `.onnx`) that is the best evidence there is.
    It never overrules a header that is sure.
    """
    verdict = classify(path.name, meta or None, sniffed)
    kind, folder, below = folder_kind(path, roots)
    if kind is not None and (verdict.confidence == "low" or verdict.category is Category.OTHER):
        verdict = Verdict(
            kind, "medium", f"it is in the {folder} folder", base_model=verdict.base_model
        )
    elif kind is not None and verdict.category is not kind:
        if interchangeable(verdict.category, kind):
            # Qwen loaded as the text encoder of an image model is a language model by its
            # weights and a text encoder by its job, and the job is what the folder says.
            verdict = Verdict(
                kind, "high", f"{verdict.reason}; used as {kind.value.replace('_', ' ')} "
                              f"here — it is in the {folder} folder",
                base_model=verdict.base_model,
            )
        elif verdict.confidence == "high":
            summary["folder_says"] = kind.value

    if not verdict.base_model or (meta.get("base_model") is None and below):
        verdict.base_model = (
            meta.get("base_model")
            or extras.get("base_model")
            or below
            or extras.get("sd_version")
            or verdict.base_model
            or (summary.get("modelspec") or {}).get("architecture")
        )
    return verdict
