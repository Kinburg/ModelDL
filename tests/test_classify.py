"""Model identification: header sniffing and the classification cascade."""

from __future__ import annotations

import json
import struct

import pytest

from sfd.library.categories import Category
from sfd.library.classify import classify, from_filename, from_header, from_provider
from sfd.library.sniff import sniff


# --- builders ---------------------------------------------------------------


def safetensors(tensors: list[str], metadata: dict | None = None) -> bytes:
    payload = {
        name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for name in tensors
    }
    if metadata:
        payload["__metadata__"] = metadata
    blob = json.dumps(payload).encode()
    return struct.pack("<Q", len(blob)) + blob + b"\0" * 32


def _gguf_str(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(kvs: dict[str, str], version: int = 3) -> bytes:
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", 0)               # tensor count
    out += struct.pack("<Q", len(kvs))
    for key, value in kvs.items():
        out += _gguf_str(key) + struct.pack("<I", 8) + _gguf_str(value)
    return out + b"\0" * 32


# --- GGUF -------------------------------------------------------------------


def test_gguf_language_model():
    result = sniff(gguf({"general.architecture": "qwen3", "general.name": "Qwen3"}))
    assert result.format == "gguf" and result.architecture == "qwen3"
    assert from_header(result).category is Category.LLM


@pytest.mark.parametrize("arch", ["flux", "wan", "sd3", "ltxv", "qwen_image"])
def test_gguf_image_models_are_not_language_models(arch):
    """The trap the whole sniffing layer exists for.

    ComfyUI ships quantised Flux and Wan as .gguf. Filing those under LLM by extension puts
    a diffusion model in a folder no image tool looks in.
    """
    verdict = from_header(sniff(gguf({"general.architecture": arch})))
    assert verdict.category is Category.DIFFUSION_MODEL
    assert verdict.confidence == "high"


def test_gguf_truncated_header_asks_for_more_rather_than_guessing():
    blob = gguf({"general.architecture": "llama"})
    result = sniff(blob[:12])
    assert result.format == "gguf"
    assert result.needs_bytes and result.needs_bytes > 12
    assert not result.understood


def test_gguf_with_an_unreadable_body_is_still_recognised_as_gguf():
    result = sniff(b"GGUF" + struct.pack("<I", 3) + b"\xff" * 64)
    assert result.format == "gguf"


# --- safetensors ------------------------------------------------------------


def test_lora_by_tensor_names():
    blob = safetensors(["lora_unet_down_blocks_0.lora_down.weight",
                        "lora_unet_down_blocks_0.lora_up.weight"])
    assert from_header(sniff(blob)).category is Category.LORA


def test_lora_by_training_metadata():
    blob = safetensors(
        ["some.opaque.tensor"],
        {"ss_network_module": "networks.lora", "ss_base_model_version": "sdxl_base_v1-0"},
    )
    verdict = from_header(sniff(blob))
    assert verdict.category is Category.LORA
    assert verdict.base_model == "sdxl_base_v1-0"


def test_full_checkpoint_has_a_denoiser_and_its_companions():
    blob = safetensors([
        "model.diffusion_model.input_blocks.0.0.weight",
        "first_stage_model.encoder.down.0.block.0.norm1.weight",
        "cond_stage_model.transformer.text_model.embeddings.weight",
    ])
    assert from_header(sniff(blob)).category is Category.CHECKPOINT


def test_bare_diffusion_model_is_not_a_checkpoint():
    """Same denoiser prefix, no text encoder or VAE — a different folder entirely."""
    blob = safetensors([
        "double_blocks.0.img_attn.qkv.weight",
        "single_blocks.0.linear1.weight",
    ])
    assert from_header(sniff(blob)).category is Category.DIFFUSION_MODEL


def test_standalone_vae():
    blob = safetensors([
        "encoder.down.0.block.0.norm1.weight",
        "decoder.up.0.block.0.norm1.weight",
        "quant_conv.weight",
    ])
    assert from_header(sniff(blob)).category is Category.VAE


def test_language_model_in_safetensors():
    blob = safetensors([
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ])
    assert from_header(sniff(blob)).category is Category.LLM


def test_text_encoder():
    blob = safetensors(["text_model.encoder.layers.0.self_attn.q_proj.weight"])
    assert from_header(sniff(blob)).category is Category.TEXT_ENCODER


def test_truncated_safetensors_header_reports_what_it_needs():
    blob = safetensors(["model.diffusion_model.x"])
    result = sniff(blob[:16])
    assert result.format == "safetensors"
    assert result.needs_bytes == struct.unpack("<Q", blob[:8])[0] + 8
    assert not result.understood


def test_garbage_is_not_forced_into_a_format():
    assert sniff(b"not a model at all, just some bytes").format is None
    assert sniff(b"").format is None
    # A plausible length prefix followed by non-JSON must not be claimed as safetensors.
    assert sniff(struct.pack("<Q", 20) + b"x" * 20).format is None


def test_absurd_header_length_is_rejected():
    assert sniff(struct.pack("<Q", 2**60) + b"x" * 64).format is None


# --- provider metadata ------------------------------------------------------


def test_civitai_type_is_taken_at_face_value():
    verdict = from_provider({"model_type": "LORA", "base_model": "Flux.1 D"})
    assert verdict.category is Category.LORA
    assert verdict.confidence == "high"
    assert verdict.base_model == "Flux.1 D"


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("Checkpoint", Category.CHECKPOINT),
        ("TextualInversion", Category.EMBEDDING),
        ("Controlnet", Category.CONTROLNET),
        ("VAE", Category.VAE),
        ("Upscaler", Category.UPSCALER),
        ("DoRA", Category.LORA),
        ("Workflows", Category.OTHER),
    ],
)
def test_the_civitai_type_enum(declared, expected):
    assert from_provider({"model_type": declared}).category is expected


def test_huggingface_tags():
    assert from_provider(
        {"library_name": "transformers", "pipeline_tag": "text-generation"}
    ).category is Category.LLM
    assert from_provider({"library_name": "peft"}).category is Category.LORA
    assert from_provider(
        {"tags": ["base_model:adapter:black-forest-labs/FLUX.1-dev"]}
    ).category is Category.LORA


def test_unknown_provider_metadata_yields_nothing_rather_than_a_guess():
    assert from_provider({"model_type": "SomethingNew"}) is None
    assert from_provider({}) is None


# --- the cascade ------------------------------------------------------------


def test_the_header_outranks_the_service_label():
    """Taken from a real case: Civitai lists Sick Ollie Krea2 as a Checkpoint, but the file
    holds only denoiser weights — no VAE, no text encoder. Filed as a checkpoint it lands in
    a folder whose loader cannot open it. The header decides.
    """
    blob = safetensors([
        "model.diffusion_model.blocks.0.attn.gate.weight",
        "model.diffusion_model.blocks.0.attn.qknorm.knorm.scale",
    ])
    verdict = classify(
        "sickOllie_krea2.bf16.safetensors",
        {"model_type": "Checkpoint", "base_model": "Krea 2"},
        sniff(blob),
    )
    assert verdict.category is Category.DIFFUSION_MODEL
    assert verdict.disagreement is Category.CHECKPOINT
    assert verdict.base_model == "Krea 2"          # still taken from the service
    assert verdict.needs_confirmation, "a disagreement should be shown, not buried"


def test_a_genuine_full_checkpoint_is_not_downgraded():
    blob = safetensors([
        "model.diffusion_model.input_blocks.0.0.weight",
        "first_stage_model.encoder.down.0.block.0.norm1.weight",
        "cond_stage_model.transformer.text_model.embeddings.weight",
    ])
    verdict = classify("sd15.safetensors", {"model_type": "Checkpoint"}, sniff(blob))
    assert verdict.category is Category.CHECKPOINT
    assert verdict.disagreement is None


def test_the_service_names_the_base_model_even_though_the_header_wins_the_category():
    """Opposite priorities, on purpose.

    A Pony LoRA records `sdxl_base_v1-0` in its training metadata — technically true, and
    useless for filing, because Pony LoRAs do not work on plain SDXL. Civitai calls it
    "Pony", which is the folder a person actually wants.
    """
    blob = safetensors(
        ["x.lora_down.weight"],
        {"ss_network_module": "networks.lora", "ss_base_model_version": "sdxl_base_v1-0"},
    )
    verdict = classify("style.safetensors", {"base_model": "Pony"}, sniff(blob))
    assert verdict.category is Category.LORA
    assert verdict.base_model == "Pony"


def test_training_metadata_supplies_the_base_model_when_the_service_is_silent():
    blob = safetensors(
        ["x.lora_down.weight"],
        {"ss_network_module": "networks.lora", "ss_base_model_version": "sdxl_base_v1-0"},
    )
    assert classify("style.safetensors", None, sniff(blob)).base_model == "sdxl_base_v1-0"


def test_agreement_leaves_no_disagreement_note():
    blob = safetensors(["lora_unet_x.lora_down.weight"])
    verdict = classify("x.safetensors", {"model_type": "LORA"}, sniff(blob))
    assert verdict.category is Category.LORA
    assert verdict.disagreement is None
    assert not verdict.needs_confirmation


def test_the_service_decides_when_the_header_is_silent():
    """An upscaler in .pth has no header we can read; the label is all there is."""
    verdict = classify("4x-thing.pth", {"model_type": "Upscaler"}, sniff(b"garbage"))
    assert verdict.category is Category.UPSCALER
    assert verdict.confidence == "high"


def test_header_outranks_the_filename():
    """A misleading name loses to what is actually inside the file."""
    blob = safetensors(["lora_unet_x.lora_down.weight"])
    verdict = classify("definitely_a_vae.safetensors", {}, sniff(blob))
    assert verdict.category is Category.LORA
    assert verdict.confidence == "high"


def test_filename_is_used_only_when_nothing_else_speaks():
    verdict = classify("t5xxl_fp16.safetensors", None, None)
    assert verdict.category is Category.TEXT_ENCODER
    assert verdict.confidence == "low"
    assert verdict.needs_confirmation


def test_a_total_unknown_is_reported_as_such():
    verdict = classify("mystery.bin", None, None)
    assert verdict.category is Category.OTHER
    assert verdict.needs_confirmation
    assert "nothing identified" in verdict.reason


def test_every_verdict_explains_itself():
    blob = safetensors(["model.layers.0.self_attn.q_proj.weight"])
    for verdict in (
        classify("x.gguf", {"model_type": "Checkpoint"}, None),
        classify("x.safetensors", None, sniff(blob)),
        classify("4x-UltraSharp.pth", None, None),
        classify("mystery.bin", None, None),
    ):
        assert verdict.reason
        assert verdict.confidence in {"high", "medium", "low"}


def test_filename_rules():
    assert from_filename("4x-UltraSharp.pth").category is Category.UPSCALER
    assert from_filename("ip-adapter_sdxl.safetensors").category is Category.IPADAPTER
    assert from_filename("control_v11p_sd15_canny.pth").category is Category.CONTROLNET
    assert from_filename("ae.safetensors").category is Category.VAE
    assert from_filename("anonymous.bin") is None
