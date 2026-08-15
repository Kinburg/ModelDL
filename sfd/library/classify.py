"""Deciding what a model is, and how sure we are.

Three sources of evidence, strongest first:

1. **The service's own metadata.** Civitai states the type outright, so there is nothing to
   infer. HuggingFace has no equivalent field, but `library_name`, `pipeline_tag` and tags
   narrow it a long way.

2. **The file's own header.** Tensor names are close to unforgeable: a LoRA has
   `lora_down.weight` keys and nothing else does. This is what catches the cases naming
   conventions get wrong — most importantly that `.gguf` is not a synonym for "language
   model", since ComfyUI ships quantised Flux and Wan in the same container.

3. **The filename.** Last resort, and treated as such.

Nothing here decides silently. Every verdict carries a confidence and the reason behind it,
so the caller can file a confident answer automatically and put an uncertain one in front of
the user instead of guessing on their behalf.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .categories import Category
from .sniff import Sniff


@dataclass(slots=True)
class Verdict:
    category: Category
    confidence: str            # "high" | "medium" | "low"
    reason: str
    base_model: str | None = None
    # What the service called it, when that differs from what the file turned out to be.
    # Not an error — the two answer different questions — but the user should see it.
    disagreement: Category | None = None

    @property
    def needs_confirmation(self) -> bool:
        return self.confidence != "high" or self.disagreement is not None


# --- 1. provider metadata ---------------------------------------------------

_CIVITAI_TYPES = {
    "checkpoint": Category.CHECKPOINT,
    "lora": Category.LORA,
    "locon": Category.LORA,
    "dora": Category.LORA,
    "lycoris": Category.LORA,
    "textualinversion": Category.EMBEDDING,
    "controlnet": Category.CONTROLNET,
    "vae": Category.VAE,
    "upscaler": Category.UPSCALER,
    "hypernetwork": Category.HYPERNETWORK,
    "motionmodule": Category.MOTION_MODULE,
    "aestheticgradient": Category.EMBEDDING,
    "poses": Category.OTHER,
    "wildcards": Category.OTHER,
    "workflows": Category.OTHER,
    "other": Category.OTHER,
}

# GGUF architectures that are image or video models rather than language models. This list
# is the whole reason header sniffing exists: by extension alone they are indistinguishable.
_DIFFUSION_ARCHITECTURES = {
    "flux", "sd3", "sdxl", "sd1", "wan", "ltxv", "hunyuan-video", "hyvid",
    "hidream", "qwen_image", "cosmos", "mochi", "aura", "chroma", "lumina2",
}
_TEXT_ENCODER_ARCHITECTURES = {"t5", "t5encoder", "clip", "umt5", "clip_vision"}


def from_provider(meta: dict[str, Any]) -> Verdict | None:
    """Civitai names the type; HuggingFace has to be inferred from its tags."""
    declared = str(meta.get("model_type") or "").strip().lower()
    if declared:
        category = _CIVITAI_TYPES.get(declared.replace(" ", "").replace("-", ""))
        if category is not None:
            return Verdict(
                category,
                "high",
                f"Civitai lists this as {meta['model_type']}",
                base_model=meta.get("base_model"),
            )

    library = str(meta.get("library_name") or "").lower()
    pipeline = str(meta.get("pipeline_tag") or "").lower()
    tags = {str(t).lower() for t in (meta.get("tags") or [])}

    if library == "peft" or "lora" in tags or any(t.startswith("base_model:adapter:") for t in tags):
        return Verdict(Category.LORA, "high", "the repo is tagged as an adapter")
    if library == "transformers" and pipeline in {"text-generation", "text2text-generation"}:
        return Verdict(Category.LLM, "high", f"transformers repo for {pipeline}")
    if pipeline in {"text-to-image", "image-to-image", "text-to-video", "image-to-video"}:
        return Verdict(Category.CHECKPOINT, "medium", f"pipeline tag is {pipeline}")
    if library == "sentence-transformers":
        return Verdict(Category.LLM, "medium", "sentence-transformers repo")
    return None


# --- 2. the file's own header -----------------------------------------------

_TENSOR_RULES: list[tuple[Category, str, tuple[str, ...]]] = [
    # LoRA adapters are unmistakable: the decomposition shows up in every key.
    (Category.LORA, "tensor names carry LoRA decomposition weights",
     ("lora_down.weight", "lora_up.weight", ".lora_a.weight", ".lora_b.weight",
      "lora_unet_", "lora_te_", ".hada_w1_a", ".lokr_w1")),
    (Category.CONTROLNET, "tensor names contain a ControlNet hint block",
     ("control_model.", "input_hint_block", "controlnet_cond_embedding")),
    (Category.IPADAPTER, "tensor names contain IP-Adapter projection layers",
     ("image_proj.", "ip_adapter.")),
    (Category.CLIP_VISION, "tensor names contain a vision tower",
     ("vision_model.encoder.layers", "visual.transformer.resblocks")),
    (Category.HYPERNETWORK, "tensor names look like a hypernetwork", ("model.linear1",)),
]

# A full checkpoint bundles the denoiser with a text encoder and a VAE; a bare diffusion
# model does not. Both contain `model.diffusion_model.`, so the companions decide.
_CHECKPOINT_COMPANIONS = ("first_stage_model.", "cond_stage_model.", "conditioner.")
_DIFFUSION_PREFIXES = (
    "model.diffusion_model.", "double_blocks.", "single_blocks.",
    "joint_blocks.", "diffusion_model.",
)
_VAE_PREFIXES = ("encoder.down.", "decoder.up.", "quant_conv", "post_quant_conv")
_LLM_PATTERN = re.compile(r"^(model\.)?layers\.\d+\.(self_attn|mlp)\.")
_TEXT_ENCODER_PREFIXES = ("text_model.encoder.layers", "encoder.block.0.layer")


def from_header(sniff: Sniff) -> Verdict | None:
    if sniff.format == "gguf":
        return _from_gguf(sniff)
    if sniff.format == "safetensors":
        return _from_safetensors(sniff)
    return None


def _from_gguf(sniff: Sniff) -> Verdict | None:
    arch = (sniff.architecture or "").strip().lower()
    if not arch:
        return None
    if arch in _DIFFUSION_ARCHITECTURES:
        return Verdict(
            Category.DIFFUSION_MODEL, "high",
            f"GGUF general.architecture is {arch}, an image or video model",
            base_model=arch,
        )
    if arch in _TEXT_ENCODER_ARCHITECTURES:
        return Verdict(
            Category.TEXT_ENCODER, "high", f"GGUF general.architecture is {arch}",
            base_model=arch,
        )
    return Verdict(
        Category.LLM, "high", f"GGUF general.architecture is {arch}", base_model=arch
    )


def _from_safetensors(sniff: Sniff) -> Verdict | None:
    names = sniff.tensor_names
    if not names:
        return None
    lowered = [n.lower() for n in names]

    # kohya writes its training configuration into the header; when present it is decisive.
    module = sniff.metadata.get("ss_network_module", "")
    if module:
        return Verdict(
            Category.LORA, "high", f"training metadata names {module}",
            base_model=sniff.metadata.get("ss_base_model_version"),
        )

    for category, reason, needles in _TENSOR_RULES:
        if any(any(needle in name for needle in needles) for name in lowered):
            return Verdict(category, "high", reason)

    has_diffusion = _any_startswith(lowered, _DIFFUSION_PREFIXES)
    has_companions = _any_startswith(lowered, _CHECKPOINT_COMPANIONS)
    if has_diffusion and has_companions:
        return Verdict(Category.CHECKPOINT, "high",
                       "contains a denoiser together with a text encoder and VAE")
    if has_diffusion:
        return Verdict(Category.DIFFUSION_MODEL, "high",
                       "contains denoiser weights and nothing else")

    if _any_startswith(lowered, _VAE_PREFIXES) and not has_diffusion:
        return Verdict(Category.VAE, "high", "only encoder/decoder weights are present")

    if any(_LLM_PATTERN.match(name) for name in lowered):
        return Verdict(Category.LLM, "high", "transformer decoder layer names")

    if _any_startswith(lowered, _TEXT_ENCODER_PREFIXES):
        return Verdict(Category.TEXT_ENCODER, "high", "text encoder layer names")

    # A single tiny tensor is the shape of a textual inversion.
    if len(names) <= 2 and any("emb_params" in n or "string_to_param" in n for n in lowered):
        return Verdict(Category.EMBEDDING, "high", "a single embedding tensor")

    return None


# --- 3. filename ------------------------------------------------------------

_NAME_RULES: list[tuple[Category, str, tuple[str, ...]]] = [
    (Category.VAE, "filename mentions a VAE", ("vae", "ae.safetensors", "sdxl_vae")),
    (Category.TEXT_ENCODER, "filename names a known text encoder",
     ("t5xxl", "t5_", "clip_l", "clip_g", "umt5", "text_encoder")),
    (Category.CLIP_VISION, "filename names a vision encoder", ("clip_vision", "clip-vit")),
    (Category.LORA, "filename says lora", ("lora", "locon", "lycoris")),
    (Category.CONTROLNET, "filename says controlnet", ("controlnet", "control_v11", "t2i-adapter")),
    (Category.IPADAPTER, "filename says ip-adapter", ("ip-adapter", "ip_adapter")),
    (Category.UPSCALER, "filename looks like an upscaler",
     ("esrgan", "ultrasharp", "realesr", "swinir", "4x-", "x4-", "remacri")),
    (Category.EMBEDDING, "filename says embedding", ("embedding", "textual_inversion")),
    (Category.MOTION_MODULE, "filename says animatediff", ("animatediff", "mm_sd", "motion_module")),
    (Category.DETECTION, "filename names a detector", ("yolo", "sam_", "sam2", "bbox", "face_yolo")),
]


def from_filename(filename: str) -> Verdict | None:
    lowered = filename.lower()
    for category, reason, needles in _NAME_RULES:
        if any(needle in lowered for needle in needles):
            return Verdict(category, "low", reason)
    if lowered.endswith((".pth", ".pt")) and "yolo" not in lowered:
        return Verdict(Category.UPSCALER, "low", "a bare .pth is usually an upscaler")
    return None


# --- the cascade ------------------------------------------------------------


def classify(
    filename: str,
    provider_meta: dict[str, Any] | None = None,
    header: Sniff | None = None,
) -> Verdict:
    """Best available answer, always with its provenance attached.

    The file's own header outranks the service's label, which is not the obvious ordering.
    The reason is that the two answer different questions. A site's category is what the
    uploader picked from a dropdown; the header is what the weights actually are — and it is
    the weights that decide which loader can open the file.

    Civitai lists modern Krea and Flux models as "Checkpoint" although many contain only
    denoiser weights, with no VAE and no text encoder bundled. Filed as checkpoints they
    land in a folder whose loader cannot open them. The header says `diffusion_model`, and
    the header is right about where the file belongs.

    The service still wins when the header is silent or unsure, and it remains the only
    source for base model and trigger words.
    """
    header_says = from_header(header) if header is not None else None
    service_says = from_provider(provider_meta) if provider_meta else None

    if header_says is not None and header_says.confidence == "high":
        chosen = header_says
        if service_says is not None and service_says.category is not chosen.category:
            chosen.disagreement = service_says.category
            chosen.reason += (
                f"; the service lists it as {service_says.category.value}, but the file's "
                f"contents decide which loader can open it"
            )
    elif service_says is not None:
        chosen = service_says
    elif header_says is not None:
        chosen = header_says
    else:
        chosen = from_filename(filename) or Verdict(
            Category.OTHER, "low",
            "nothing identified this file — the service gave no type, the header was "
            "unreadable, and the name says nothing",
        )

    return _with_base_model(chosen, provider_meta, header)


def _with_base_model(
    verdict: Verdict, meta: dict[str, Any] | None, header: Sniff | None
) -> Verdict:
    """Fill in the base model for subfoldering — service first, header second.

    The opposite order to the category, and for a good reason. The service names the
    ecosystem a model belongs to: Pony, Illustrious, Flux.1 D. Training metadata names the
    technical starting point, so a Pony LoRA records `sdxl_base_v1-0`. Both are true, but
    only one is useful for filing: Pony LoRAs do not work on plain SDXL, and a folder that
    mixes them is a folder nobody can use.
    """
    if meta and meta.get("base_model"):
        verdict.base_model = str(meta["base_model"])
    elif not verdict.base_model and header is not None:
        verdict.base_model = (
            header.metadata.get("ss_base_model_version") or header.architecture
        )
    return verdict


def _any_startswith(names: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefixes) for name in names)
