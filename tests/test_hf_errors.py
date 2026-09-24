"""What the Hub's refusals turn into: a message that says what to do next."""

from __future__ import annotations

import pytest

from sfd.core.errors import AccessDenied, AuthRequired
from sfd.providers.huggingface import _raise_for_hf_headers, parse_ref

REF = parse_ref("https://huggingface.co/org/flux/resolve/main/flux.safetensors")
GATED = {"x-error-code": "GatedRepo", "x-error-message": "Access to model org/flux is restricted."}


def test_a_gated_repo_without_a_token_says_where_a_token_comes_from():
    with pytest.raises(AuthRequired) as caught:
        _raise_for_hf_headers(401, GATED, REF, authenticated=False)
    message = str(caught.value)
    assert REF.page_url in message
    assert "Settings" in message and "$HF_TOKEN" in message and "hf auth login" in message


def test_a_gated_repo_with_a_token_asks_for_the_terms_to_be_accepted():
    with pytest.raises(AccessDenied) as caught:
        _raise_for_hf_headers(403, GATED, REF, authenticated=True)
    assert REF.page_url in str(caught.value)
    assert "accept the terms" in str(caught.value)
