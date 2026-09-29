import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    """Stub servers answer at once, and a 429 one of them sends is the answer under test:
    waiting it out would only slow the suite down. tests/test_polite.py turns it back on."""
    from sfd.core import polite

    monkeypatch.setattr(polite, "ENABLED", False)


@pytest.fixture(autouse=True)
def _no_hf_credentials(monkeypatch, tmp_path_factory):
    """Keep the machine's own HuggingFace token out of every test.

    Without a token the lookup falls back to the file `hf auth login` writes. A test that
    expects no token must not quietly get the real one — nor send it to a stub server.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path_factory.getbasetemp() / "no-hf-login" / "token"))
