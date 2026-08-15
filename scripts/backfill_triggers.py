"""Write the `<model>.txt` activation files for models downloaded before this existed.

    python scripts/backfill_triggers.py F:\\ComfyUI\\Shared [--sidecar-dir DIR] [--apply]

Reads the trigger words out of the JSON records already written beside (or alongside) each
model, and writes the `.txt` that A1111 extensions and ComfyUI loaders read. Defaults to a
dry run, because it writes into a live model library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfd.library.sidecar import normalise_triggers, write_trigger_text  # noqa: E402

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin"}


def records(root: Path, sidecar_dir: Path | None):
    """Every JSON record we wrote, wherever it was configured to live."""
    for base in filter(None, (sidecar_dir, root)):
        yield from base.rglob("*.json")


def model_for(record: Path, root: Path, sidecar_dir: Path | None) -> Path | None:
    """Map a record back to the model it describes."""
    stem = record.with_suffix("")           # strip the .json, leaving model.safetensors
    if stem.suffix.lower() not in MODEL_SUFFIXES:
        return None
    if sidecar_dir and sidecar_dir in record.parents:
        return root / stem.relative_to(sidecar_dir)
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="model library root")
    parser.add_argument("--sidecar-dir", help="where the .json records were collected")
    parser.add_argument("--apply", action="store_true", help="actually write the files")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace .txt files that already exist")
    args = parser.parse_args()

    root = Path(args.library)
    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else None
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    written = skipped = 0
    for record in records(root, sidecar_dir):
        model = model_for(record, root, sidecar_dir)
        if model is None or not model.exists():
            continue
        try:
            data = json.loads(record.read_text("utf-8-sig"))
        except (OSError, ValueError):
            continue

        words = normalise_triggers((data.get("usage") or {}).get("trigger_words"))
        if not words:
            continue

        target = model.with_name(model.stem + ".txt")
        if target.exists() and not args.overwrite:
            skipped += 1
            continue

        print(f"{'write' if args.apply else 'would write'}  {target}")
        print(f"          {', '.join(words)}")
        if args.apply:
            write_trigger_text(model, words)
        written += 1

    print(f"\n{written} file(s){'' if args.apply else ' would be written — pass --apply'}"
          f"{f', {skipped} already present' if skipped else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
