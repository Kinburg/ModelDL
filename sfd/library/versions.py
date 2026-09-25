"""Which of a model's newer versions is an update, and which is only newer.

A Civitai model is a page of versions, newest first, and "listed above mine" used to be
taken for "a newer version of mine". On a real library of 96 Civitai files that called 22
of them out of date, and 7 were. The other fifteen were newer only in the sense of being
further up the page:

  A collection.     Bob's Bobs: Penelope, Dianda, Kay, then Erica — every version another
                    character, none of them a new Penelope.
  Another base.     A LoRA for Krea 2, then the same idea trained for Qwen 2.1. Whoever
                    runs Krea 2 has nothing to update to.
  A variant.        ACE-Step v1.5 XL Base and v1.5 XL Turbo, published the same minute;
                    Lightning at 4 steps and at 8. Siblings, not successors.
  Already here.     v2 was downloaded last week, and v1 beside it is kept, not out of date.

What tells an update apart is in the names. An update is a version for the same base model
whose name carries a higher version number than any version of it already here. On that
library the rule kept exactly the seven and none of the fifteen. A version whose name holds
no number — Final, 2511 — is never counted, only listed among the other new versions, which
is the right way to be wrong: a quiet line that could have been a badge, rather than a badge
that should have been nothing.

The checks themselves are made by the library; this module only reads what came back, so
that the rule can be tested on the answers without asking anyone anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

# What a check can conclude about one file of the library.
UPDATE = "update"      # a newer version of it: counted, and offered for download
OTHER = "other"        # newer versions on its page, none of them an update of this one
CURRENT = "current"    # nothing newer
GONE = "gone"          # taken down: the page or this version of it is not there any more
FAILED = "error"       # the question could not be answered this time

# How many of the other new versions a record keeps by name. A collection of forty
# characters is "and 37 more" after the first few, not forty names in every row.
OTHERS_KEPT = 8

# `v1.0`, `V3`, `ver 2`, `version 2.1` — after a separator or at the start — or a dotted
# `2.0` standing on its own. Not every number in a name is a version: `Qwen2.1` is a base
# model, `4steps` a setting and `(2511)` a date, and none of them is taken for one.
_NUMBER = re.compile(
    r"(?:^|[\s_\-(\[])v(?:er(?:sion)?)?\.?\s?(\d+(?:\.\d+)*)"
    r"|(?:^|\s)(\d+\.\d+(?:\.\d+)*)(?=\s|$)",
    re.IGNORECASE,
)


def number(name: str | None) -> tuple[int, ...] | None:
    """The version number a version's name carries, or None when it carries none."""
    match = _NUMBER.search(name or "")
    if not match:
        return None
    return tuple(int(part) for part in (match.group(1) or match.group(2)).split("."))


def higher(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether `a` comes after `b`. The shorter is padded: v2 is v2.0, and v2.0.1 is after."""
    width = max(len(a), len(b))
    return tuple(a) + (0,) * (width - len(a)) > tuple(b) + (0,) * (width - len(b))


@dataclass(slots=True)
class Standing:
    """Where one version stands among the versions of its model's page."""

    # Why nothing can be said: the version is not published any more.
    gone: str | None = None
    # The update — the highest-numbered version that is one — and how many versions are.
    pick: dict[str, Any] | None = None
    count: int = 0
    # Further up the page than this one, not here, and not an update of it.
    others: list[dict[str, Any]] = field(default_factory=list)


def assess(versions: list[dict[str, Any]], mine: int, have: set[int]) -> Standing:
    """Where version `mine` stands among `versions`, in the order the API lists them.

    `have` is every version of this model the library holds, which are nothing to download
    whatever they are. The number to beat is the highest of them for this base model, not
    this one's own, so that v1 is not out of date against v2 while v3 is here as well.
    """
    ids = [version.get("id") for version in versions]
    if mine not in ids:
        return Standing(gone="this version is no longer published")
    at = ids.index(mine)
    own = versions[at]
    here = set(have) | {mine}
    base = own.get("baseModel")
    updates: list[dict[str, Any]] = []
    best = number(own.get("name"))
    # A version named without a number — a character of a collection, a VAE — has no line
    # of versions to be behind in.
    if best is not None:
        for version in versions:
            if version.get("id") in here and version.get("baseModel") == base:
                found = number(version.get("name"))
                if found is not None and higher(found, best):
                    best = found
        for version in versions:
            if version.get("id") in here or version.get("baseModel") != base:
                continue
            found = number(version.get("name"))
            if found is not None and higher(found, best):
                updates.append(version)
    pick = None
    for version in updates:
        # The first of equals wins, and the API lists the newest first.
        if pick is None or higher(number(version.get("name")) or (), number(pick.get("name")) or ()):
            pick = version
    counted = {version.get("id") for version in updates}
    others = [
        version for version in versions[:at]
        if version.get("id") not in here and version.get("id") not in counted
    ]
    return Standing(pick=pick, count=len(updates), others=others)


def access(version: dict[str, Any]) -> dict[str, Any] | None:
    """What it takes to download a version: nothing (None), or buying it — for good, or
    until its early access ends, after which it is free like any other.

    Civitai says so in `paidAccess`: `{"permanent": true}` for a version that is sold, and
    `{"permanent": false, "endsAt": …}` for early access. The price is not in its API, only
    on its page, and whether this account has bought it is not in it either: that only the
    download link itself answers — see `CivitaiProvider.owned`.
    """
    paid = version.get("paidAccess")
    if not isinstance(paid, dict):
        return None
    if paid.get("permanent"):
        return {"permanent": True, "until": None}
    until = paid.get("endsAt")
    if until and _passed(str(until)):
        return None
    return {"permanent": False, "until": until}


def _passed(moment: str) -> bool:
    try:
        when = datetime.fromisoformat(moment.replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when <= datetime.now(timezone.utc)


def find_file(files: list[dict[str, Any]] | None, file_id: Any = None, sha256: str | None = None):
    """The entry of a version's `files` that is the file here: by its id, or by its hash."""
    for entry in files or []:
        if file_id is not None and entry.get("id") == file_id:
            return entry
    if sha256:
        for entry in files or []:
            digest = (entry.get("hashes") or {}).get("SHA256")
            if isinstance(digest, str) and digest.lower() == sha256.lower():
                return entry
    return None


def choose_file(files: list[dict[str, Any]] | None, like: dict[str, Any] | None) -> dict[str, Any] | None:
    """Which file of a new version to fetch in place of the one here.

    A version can carry several — bf16 and int8, pruned and full, the VAE that goes with it
    — and the one wanted is the counterpart of the file already here: the same kind, format,
    precision and size, then as much of that as the new version has, in that order. With
    nothing to compare against, it is the file the version's own download button gives.
    """
    listed = [entry for entry in files or [] if entry.get("id") is not None]
    if not listed:
        return None

    def traits(entry: dict[str, Any]) -> tuple[Any, ...]:
        meta = entry.get("metadata") or {}
        return (entry.get("type"), meta.get("format"), meta.get("fp"), meta.get("size"))

    if like is not None:
        wanted = traits(like)
        for depth in (4, 3, 2, 1):
            same = [entry for entry in listed if traits(entry)[:depth] == wanted[:depth]]
            if same:
                chosen = next((entry for entry in same if entry.get("primary")), same[0])
                return _described(chosen, exact=depth == 4)
    chosen = next((entry for entry in listed if entry.get("primary")), listed[0])
    return _described(chosen, exact=False)


def _described(entry: dict[str, Any], exact: bool) -> dict[str, Any]:
    size = entry.get("sizeKB")
    digest = (entry.get("hashes") or {}).get("SHA256")
    return {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "size": int(size * 1024) if isinstance(size, (int, float)) and size else None,
        "sha256": digest.lower() if isinstance(digest, str) else None,
        "primary": bool(entry.get("primary")),
        # The same kind, format, precision and size as the file here — not merely the
        # version's main file.
        "exact": exact,
    }


# --- the record a check leaves ---------------------------------------------------


def covered(skipped: dict[str, Any] | None, pick: dict[str, Any] | None) -> bool:
    """Whether a version skipped earlier still answers for `pick`.

    Skipping v3 says no to v3 and to everything before it, not to v4: the update is counted
    again as soon as one higher than the skipped one comes out. A file on the Hub has no
    numbers, only content, and skipping it skips exactly that content.
    """
    if not skipped or not pick:
        return False
    if "sha256" in skipped:
        return skipped["sha256"] == pick.get("sha256")
    if pick.get("number") and skipped.get("number"):
        return not higher(pick["number"], skipped["number"])
    return skipped.get("id") is not None and skipped.get("id") == pick.get("id")


def status(record: dict[str, Any]) -> str:
    """What a record says, worked out from what is in it — so that skipping a version, or
    counting it again, needs no second question to the service."""
    if record.get("gone"):
        return GONE
    if record.get("error"):
        return FAILED
    pick = record.get("update")
    if pick and not covered(record.get("skipped"), pick):
        return UPDATE
    if pick or record.get("others"):
        return OTHER
    return CURRENT


def normalise(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """A stored record as the page reads it, or None for nothing known.

    Records written before the rule above — `available` and the newest version, flat — were
    made by the rule this module replaces, and most of what they called newer is not. They
    are taken for not checked yet, which the next check puts right.
    """
    if not record or "status" not in record:
        return None
    return record


def skip(record: dict[str, Any]) -> dict[str, Any] | None:
    """The record with its update skipped, or None when it has no update to skip."""
    pick = record.get("update")
    if record.get("status") != UPDATE or not pick:
        return None
    changed = dict(record)
    if "id" in pick:
        changed["skipped"] = {"id": pick.get("id"), "name": pick.get("name"), "number": pick.get("number")}
    else:
        changed["skipped"] = {"sha256": pick.get("sha256"), "name": pick.get("name")}
    changed["status"] = status(changed)
    return changed


def unskip(record: dict[str, Any]) -> dict[str, Any] | None:
    """The record counting its update again, or None when nothing was skipped."""
    if not record.get("skipped"):
        return None
    changed = {key: value for key, value in record.items() if key != "skipped"}
    changed["status"] = status(changed)
    return changed


def target(record: dict[str, Any] | None) -> Any:
    """What an update record points at: a version, or content on the Hub."""
    pick = (record or {}).get("update") or {}
    return pick.get("id") if "id" in pick else pick.get("sha256")
