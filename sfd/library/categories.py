"""The canonical set of model kinds, and where each one lives.

The names here are internal and stable. Where they land on disk is a per-profile decision —
ComfyUI, A1111 and llama.cpp all disagree — so nothing outside `layout` should hardcode a
directory.
"""

from __future__ import annotations

from enum import Enum


class Category(str, Enum):
    CHECKPOINT = "checkpoint"            # full pipeline: unet + text encoder + vae
    DIFFUSION_MODEL = "diffusion_model"  # unet / transformer on its own (Flux, Wan, SD3)
    LORA = "lora"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"        # CLIP-L, T5, UMT5
    CLIP_VISION = "clip_vision"
    CONTROLNET = "controlnet"
    EMBEDDING = "embedding"              # textual inversion
    UPSCALER = "upscaler"
    IPADAPTER = "ipadapter"
    STYLE_MODEL = "style_model"
    HYPERNETWORK = "hypernetwork"
    MOTION_MODULE = "motion_module"
    LLM = "llm"
    DETECTION = "detection"              # ultralytics, face detection, segmentation
    OTHER = "other"


# Default subdirectory per profile. A profile is seeded from whatever folders already exist
# in the user's root, so these are only the fallback when nothing matches by name.
COMFYUI: dict[Category, str] = {
    Category.CHECKPOINT: "checkpoints",
    Category.DIFFUSION_MODEL: "diffusion_models",
    Category.LORA: "loras",
    Category.VAE: "vae",
    Category.TEXT_ENCODER: "text_encoders",
    Category.CLIP_VISION: "clip_vision",
    Category.CONTROLNET: "controlnet",
    Category.EMBEDDING: "embeddings",
    Category.UPSCALER: "upscale_models",
    Category.IPADAPTER: "ipadapter",
    Category.STYLE_MODEL: "style_models",
    Category.HYPERNETWORK: "hypernetworks",
    Category.MOTION_MODULE: "animatediff_models",
    Category.LLM: "LLM",
    Category.DETECTION: "ultralytics",
    Category.OTHER: "other",
}

A1111: dict[Category, str] = {
    Category.CHECKPOINT: "Stable-diffusion",
    Category.DIFFUSION_MODEL: "Stable-diffusion",
    Category.LORA: "Lora",
    Category.VAE: "VAE",
    Category.TEXT_ENCODER: "text_encoder",
    Category.CLIP_VISION: "clip_vision",
    Category.CONTROLNET: "ControlNet",
    Category.EMBEDDING: "embeddings",
    Category.UPSCALER: "ESRGAN",
    Category.IPADAPTER: "ipadapter",
    Category.STYLE_MODEL: "style_models",
    Category.HYPERNETWORK: "hypernetworks",
    Category.MOTION_MODULE: "animatediff",
    Category.LLM: "LLM",
    Category.DETECTION: "detection",
    Category.OTHER: "other",
}

PROFILES = {"comfyui": COMFYUI, "a1111": A1111}

# Folder names seen in the wild that mean the same thing. Used when adopting an existing
# tree: the user's own layout wins over our defaults.
ALIASES: dict[str, Category] = {
    "checkpoints": Category.CHECKPOINT,
    "stable-diffusion": Category.CHECKPOINT,
    "diffusion_models": Category.DIFFUSION_MODEL,
    "unet": Category.DIFFUSION_MODEL,
    "loras": Category.LORA,
    "lora": Category.LORA,
    "lycoris": Category.LORA,
    "vae": Category.VAE,
    "text_encoders": Category.TEXT_ENCODER,
    "text_encoder": Category.TEXT_ENCODER,
    "clip": Category.TEXT_ENCODER,
    "clip_vision": Category.CLIP_VISION,
    "controlnet": Category.CONTROLNET,
    "embeddings": Category.EMBEDDING,
    "embedding": Category.EMBEDDING,
    "upscale_models": Category.UPSCALER,
    "esrgan": Category.UPSCALER,
    "ipadapter": Category.IPADAPTER,
    "style_models": Category.STYLE_MODEL,
    "hypernetworks": Category.HYPERNETWORK,
    "animatediff_models": Category.MOTION_MODULE,
    "animatediff": Category.MOTION_MODULE,
    "motion_module": Category.MOTION_MODULE,
    "llm": Category.LLM,
    "ultralytics": Category.DETECTION,
    "detection": Category.DETECTION,
    # Deliberately absent: `sams`, `sam2`, `facedetection`, `insightface`, `reactor` and the
    # rest of the custom-node folders. They hold detectors too, but segmenters, face
    # detectors and YOLO models are read by different nodes and are not interchangeable —
    # and a bare .pt file gives us nothing to tell them apart with. Claiming those folders
    # would mean filing a SAM checkpoint under ultralytics with full confidence. Files we
    # cannot place go to `other` and ask.
}
