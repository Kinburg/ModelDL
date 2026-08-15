"""Civitai reference parsing and variant handling.

The fixture mirrors a real response: one model version carrying five quantisations that all
share a single filename.
"""

from __future__ import annotations

import pytest

from sfd.core.errors import AccessDenied
from sfd.providers.civitai import (
    CivitaiProvider,
    CivitaiRef,
    _describe_files,
    is_civitai_url,
    make_identity,
    parse_ref,
)

VERSION = {
    "id": 3154408,
    "name": "Krea2",
    "modelId": 2676616,
    "baseModel": "Krea 2",
    "trainedWords": ["sick ollie"],
    "model": {"name": "Sick Ollie", "type": "Checkpoint", "nsfw": False},
    "files": [
        {
            "id": 3035672, "primary": True, "type": "Model",
            "name": "sickOllie_krea2.safetensors", "sizeKB": 12845056,
            "metadata": {"format": "SafeTensor", "size": "full", "fp": "int8"},
            "hashes": {"SHA256": "AFDE753B7F8B13A529CF9D10A2422C490E40A130C9157C9B56743107F511EED8"},
        },
        {
            "id": 3036948, "primary": False, "type": "Model",
            "name": "sickOllie_krea2.safetensors", "sizeKB": 25036554,
            "metadata": {"format": "SafeTensor", "size": "full", "fp": "bf16"},
            "hashes": {"SHA256": "956C3A9AFE1230D5ACD322C3D76DE912A7813BD0B65CE528695101DA96C1CC5B"},
        },
        {
            "id": 3035266, "primary": False, "type": "Model",
            "name": "sickOllie_krea2.safetensors", "sizeKB": 12519178,
            "metadata": {"format": "SafeTensor", "size": "full", "fp": "fp8"},
            "hashes": {"SHA256": "FB51603FBB376F7DC96D40FF520BE23B75164FF356198891479DDD45858D0542"},
        },
    ],
}


# --- parsing ----------------------------------------------------------------


def test_model_page_with_a_version():
    ref = parse_ref("https://civitai.com/models/2676616/sick-ollie?modelVersionId=3154408")
    assert ref.model_id == 2676616
    assert ref.version_id == 3154408
    assert ref.file_id is None


def test_model_page_without_a_version():
    ref = parse_ref("https://civitai.com/models/2676616")
    assert ref.model_id == 2676616 and ref.version_id is None


def test_download_link_from_a_variant_button():
    ref = parse_ref("https://civitai.com/api/download/models/3154408?fileId=3036948")
    assert ref.version_id == 3154408 and ref.file_id == 3036948
    assert not ref.wants_primary


def test_download_link_with_precision_parameters():
    ref = parse_ref(
        "https://civitai.com/api/download/models/3154408"
        "?type=Model&format=SafeTensor&size=full&fp=bf16"
    )
    assert dict(ref.variant) == {
        "fp": "bf16", "format": "SafeTensor", "size": "full", "type": "Model"
    }
    assert not ref.wants_primary


def test_bare_download_link_means_the_primary_file():
    """Not "every variant" — that turns one click into a five-fold download."""
    ref = parse_ref("https://civitai.com/api/download/models/3154408")
    assert ref.wants_primary and ref.file_id is None and not ref.variant


def test_the_red_mirror_is_accepted_and_preserved():
    """civitai.red is a full mirror of the same API.

    People use it because .com is unreachable where they are, so rewriting their link back
    to .com would send them somewhere they cannot get to.
    """
    ref = parse_ref("https://civitai.red/models/2676616/sick-ollie?modelVersionId=3154408")
    assert ref.host == "civitai.red"
    assert ref.model_id == 2676616 and ref.version_id == 3154408
    assert ref.page_url.startswith("https://civitai.red/")

    download = parse_ref("https://civitai.red/api/download/models/3154408?fileId=3036948")
    assert download.host == "civitai.red"
    assert download.endpoint == "https://civitai.red"


def test_the_default_host_is_still_com():
    assert parse_ref("https://civitai.com/models/1").host == "civitai.com"
    assert parse_ref("https://www.civitai.com/models/1").host == "civitai.com"
    assert is_civitai_url("https://civitai.red/models/1")


def test_the_mirror_is_not_part_of_the_identity():
    """The same file from either domain is one download, not two."""
    from_com = parse_ref("https://civitai.com/api/download/models/3?fileId=4")
    from_red = parse_ref("https://civitai.red/api/download/models/3?fileId=4")
    assert make_identity(from_com).key() == make_identity(from_red).key()


def test_the_provider_talks_to_the_host_it_was_given():
    assert CivitaiProvider(host="civitai.red")._api == "https://civitai.red/api/v1"
    assert CivitaiProvider()._api == "https://civitai.com/api/v1"
    # An unknown host must not be trusted into the URL builder.
    assert CivitaiProvider(host="evil.example.com").host == "civitai.com"


def test_air_identifier():
    ref = parse_ref("urn:air:sdxl:checkpoint:civitai:2676616@3154408")
    assert ref.model_id == 2676616 and ref.version_id == 3154408


@pytest.mark.parametrize(
    "url",
    ["https://huggingface.co/org/name", "https://civitai.com/", "", "https://example.com/x"],
)
def test_rejects_other_things(url):
    with pytest.raises(ValueError):
        parse_ref(url)


def test_is_civitai_url():
    assert is_civitai_url("https://civitai.com/models/1")
    assert is_civitai_url("urn:air:sdxl:checkpoint:civitai:1@2")
    assert not is_civitai_url("https://huggingface.co/org/name")


def test_identity_needs_a_specific_file():
    with pytest.raises(ValueError):
        make_identity(CivitaiRef(version_id=3154408))
    identity = make_identity(CivitaiRef(version_id=3154408, file_id=3036948))
    assert set(identity.ref) == {"version_id", "file_id"}


# --- filename disambiguation ------------------------------------------------


def test_variants_sharing_a_name_get_distinct_filenames():
    """The API returns one identical `name` for all five quantisations.

    Written as-is, each download would overwrite the previous one and leave a single file
    whose precision nobody can determine afterwards.
    """
    files = _describe_files(VERSION)
    names = [f.filename for f in files]
    assert names == [
        "sickOllie_krea2.int8.safetensors",
        "sickOllie_krea2.bf16.safetensors",
        "sickOllie_krea2.fp8.safetensors",
    ]
    assert len(set(names)) == len(names)


def test_a_unique_name_is_left_alone():
    version = {**VERSION, "files": [VERSION["files"][0]]}
    assert _describe_files(version)[0].filename == "sickOllie_krea2.safetensors"


def test_catalogue_metadata_is_carried_through():
    """Type, base model and trigger words are what make the file usable later."""
    entry = _describe_files(VERSION)[1]
    assert entry.meta["model_type"] == "Checkpoint"
    assert entry.meta["base_model"] == "Krea 2"
    assert entry.meta["trained_words"] == ["sick ollie"]
    assert entry.meta["precision"] == "bf16"
    assert entry.sha256 == entry.sha256.lower() and len(entry.sha256) == 64
    assert entry.size == 25036554 * 1024


# --- selection --------------------------------------------------------------


async def test_file_id_selects_exactly_one(monkeypatch):
    files = await _list(CivitaiRef(version_id=3154408, file_id=3036948))
    assert [f.file_id for f in files] == [3036948]


async def test_precision_parameters_select_exactly_one():
    ref = parse_ref(
        "https://civitai.com/api/download/models/3154408?format=SafeTensor&fp=fp8"
    )
    files = await _list(ref)
    assert [f.filename for f in files] == ["sickOllie_krea2.fp8.safetensors"]


async def test_bare_download_link_selects_the_primary():
    files = await _list(parse_ref("https://civitai.com/api/download/models/3154408"))
    assert [f.primary for f in files] == [True]


async def test_model_page_offers_every_variant():
    files = await _list(CivitaiRef(version_id=3154408, model_id=2676616))
    assert len(files) == 3


async def test_an_unmatched_variant_is_an_error_not_a_silent_full_list():
    ref = parse_ref("https://civitai.com/api/download/models/3154408?fp=nf4")
    with pytest.raises(AccessDenied):
        await _list(ref)


async def _list(ref: CivitaiRef):
    provider = CivitaiProvider()
    provider._versions[3154408] = VERSION      # skip the network
    return await provider.list_files(ref, client=None)  # type: ignore[arg-type]
