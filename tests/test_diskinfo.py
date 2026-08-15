from __future__ import annotations

import pytest

from sfd.core.diskinfo import _classify, recommend_connections
from sfd.core.types import DiskKind


@pytest.mark.parametrize(
    "media, spindle, bus, expected",
    [
        ("ssd", "0", "nvme", DiskKind.SSD),
        ("ssd", "0", "sata", DiskKind.SSD),
        ("hdd", "7200", "sata", DiskKind.HDD),
        # MediaType is unset but the drive still admits to spinning.
        ("unspecified", "5400", "sata", DiskKind.HDD),
        # "I don't know" is a valid answer and must not be dressed up as a guess: a SATA
        # drive with no media type and no spindle speed could be either.
        ("unspecified", "0", "sata", DiskKind.UNKNOWN),
        ("unspecified", "4294967295", "sata", DiskKind.UNKNOWN),
        # NVMe cannot be mechanical.
        ("unspecified", "0", "nvme", DiskKind.SSD),
    ],
)
def test_classification(media, spindle, bus, expected):
    assert _classify(media, spindle, bus) is expected


def test_hdd_gets_throttled_hard(tmp_path):
    assert recommend_connections(tmp_path, 8, 20 * 1024**3, DiskKind.HDD) == 2


def test_unknown_is_treated_conservatively(tmp_path):
    assert recommend_connections(tmp_path, 8, 20 * 1024**3, DiskKind.UNKNOWN) == 6


def test_ssd_keeps_what_it_was_given(tmp_path):
    assert recommend_connections(tmp_path, 8, 20 * 1024**3, DiskKind.SSD) == 8


def test_small_files_do_not_get_a_connection_fleet(tmp_path):
    # One chunk of work cannot usefully occupy eight sockets.
    assert recommend_connections(tmp_path, 8, 4 * 1024**2, DiskKind.SSD) == 1


def test_never_returns_zero(tmp_path):
    assert recommend_connections(tmp_path, 0, 0, DiskKind.HDD) == 1
