"""Link parsing.

People paste whatever URL they happen to be looking at — the `/blob/` page from search, the
`/resolve/` link the download button makes, a `/tree/` folder, or just `org/name`. All of
them point at the same thing and all of them have to work.
"""

from __future__ import annotations

import pytest

from sfd.providers.huggingface import HfRef, is_hf_url, make_identity, parse_ref


def test_blob_page_url():
    ref = parse_ref(
        "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/blob/main/Qwen3-1.7B-Q4_K_M.gguf"
    )
    assert ref.repo_id == "unsloth/Qwen3-1.7B-GGUF"
    assert ref.repo_type == "model"
    assert ref.revision == "main"
    assert ref.path == "Qwen3-1.7B-Q4_K_M.gguf"
    assert ref.is_file


def test_blob_and_resolve_are_the_same_file():
    blob = parse_ref("https://huggingface.co/org/name/blob/main/model.safetensors")
    resolve = parse_ref("https://huggingface.co/org/name/resolve/main/model.safetensors")
    assert blob == resolve
    assert make_identity(blob).key() == make_identity(resolve).key()


def test_nested_path_is_kept_whole():
    ref = parse_ref("https://huggingface.co/org/name/blob/main/unet/diffusion_model.safetensors")
    assert ref.path == "unet/diffusion_model.safetensors"


def test_repo_root_has_no_path():
    ref = parse_ref("https://huggingface.co/org/name")
    assert ref.path is None
    assert ref.is_directory and not ref.is_file


def test_tree_url_is_a_directory_not_a_file():
    """A folder must not become a download task pointing at a path that serves nothing."""
    ref = parse_ref("https://huggingface.co/org/name/tree/main/subdir")
    assert ref.path == "subdir"
    assert ref.is_directory and not ref.is_file
    with pytest.raises(ValueError):
        make_identity(ref)


def test_blob_url_at_the_same_path_is_a_file():
    ref = parse_ref("https://huggingface.co/org/name/blob/main/subdir")
    assert ref.is_file and not ref.is_directory


def test_tree_at_the_repo_root():
    ref = parse_ref("https://huggingface.co/org/name/tree/main")
    assert ref.path is None and ref.is_directory


def test_dataset_and_space_repo_types():
    assert parse_ref("https://huggingface.co/datasets/org/name/blob/main/f.parquet").repo_type == "dataset"
    assert parse_ref("https://huggingface.co/spaces/org/name").repo_type == "space"


def test_legacy_single_segment_repo():
    # Repos predating organisations, like "gpt2", have no owner segment.
    assert parse_ref("https://huggingface.co/gpt2").repo_id == "gpt2"
    assert parse_ref("https://huggingface.co/gpt2/blob/main/config.json").repo_id == "gpt2"


def test_bare_repo_id():
    assert parse_ref("org/name") == HfRef("org/name", is_directory=True)
    assert parse_ref("gpt2") == HfRef("gpt2", is_directory=True)


def test_hf_co_shorthand_and_missing_scheme():
    assert parse_ref("hf.co/org/name/blob/main/f.bin").repo_id == "org/name"
    assert parse_ref("huggingface.co/org/name").repo_id == "org/name"


def test_non_main_revision():
    ref = parse_ref("https://huggingface.co/org/name/blob/fp16/model.safetensors")
    assert ref.revision == "fp16"


def test_pull_request_revision_spans_three_segments():
    # refs/pr/3 is a branch name containing slashes; naive parsing would read the revision
    # as "refs" and the path as "pr/3/model.safetensors".
    ref = parse_ref("https://huggingface.co/org/name/blob/refs/pr/3/model.safetensors")
    assert ref.revision == "refs/pr/3"
    assert ref.path == "model.safetensors"


def test_percent_encoded_segments():
    ref = parse_ref("https://huggingface.co/org/name/blob/main/dir%20with%20space/a%2Bb.gguf")
    assert ref.path == "dir with space/a+b.gguf"


def test_query_and_fragment_are_ignored():
    ref = parse_ref("https://huggingface.co/org/name/blob/main/f.gguf?download=true#sha")
    assert ref.path == "f.gguf"


@pytest.mark.parametrize(
    "url",
    [
        "https://civitai.com/models/1234",
        "https://example.com/org/name/blob/main/f.bin",
        "",
        "   ",
        "a/b/c/d/e",
    ],
)
def test_rejects_things_that_are_not_hub_references(url):
    with pytest.raises(ValueError):
        parse_ref(url)


def test_is_hf_url():
    assert is_hf_url("https://huggingface.co/org/name")
    assert is_hf_url("hf.co/org/name")
    assert not is_hf_url("https://civitai.com/models/1")


# --- URL construction -------------------------------------------------------


def test_download_url_uses_resolve():
    ref = parse_ref("https://huggingface.co/org/name/blob/main/model.safetensors")
    assert ref.download_url() == (
        "https://huggingface.co/org/name/resolve/main/model.safetensors"
    )


def test_download_url_for_a_dataset():
    ref = parse_ref("https://huggingface.co/datasets/org/name/blob/main/train.parquet")
    assert ref.download_url() == (
        "https://huggingface.co/datasets/org/name/resolve/main/train.parquet"
    )


def test_download_url_escapes_awkward_names():
    ref = HfRef("org/name", "model", "refs/pr/3", "dir with space/a+b.gguf")
    url = ref.download_url()
    assert "refs%2Fpr%2F3" in url
    assert "dir%20with%20space/a%2Bb.gguf" in url


def test_whole_repo_cannot_be_a_single_transfer():
    with pytest.raises(ValueError):
        parse_ref("https://huggingface.co/org/name").download_url()
    with pytest.raises(ValueError):
        make_identity(parse_ref("https://huggingface.co/org/name"))


def test_a_link_can_be_rebuilt_from_an_identity():
    """Identities hold no URL, so records that need one have to reconstruct it."""
    from sfd.providers.civitai import CivitaiRef
    from sfd.providers.civitai import make_identity as civitai_identity
    from sfd.providers.direct import make_identity as direct_identity
    from sfd.providers.registry import source_url

    hf_identity = make_identity(parse_ref("https://huggingface.co/org/name/blob/main/f.gguf"))
    assert source_url(hf_identity) == "https://huggingface.co/org/name/resolve/main/f.gguf"

    cv_identity = civitai_identity(CivitaiRef(version_id=1, file_id=2))
    assert source_url(cv_identity) == "https://civitai.com/api/download/models/1?fileId=2"

    assert source_url(direct_identity("https://example.com/x.bin")) == "https://example.com/x.bin"


def test_identity_excludes_the_token_and_the_url():
    """Resume must survive a re-signed URL and a rotated token."""
    identity = make_identity(parse_ref("https://huggingface.co/org/name/blob/main/f.gguf"))
    assert set(identity.ref) == {"repo_id", "repo_type", "revision", "path"}
    assert "huggingface.co" not in identity.key()
