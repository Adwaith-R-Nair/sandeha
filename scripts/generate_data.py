"""Generate the dataset, or resume an interrupted run.

Usage: scripts/long_job.sh data uv run scripts/generate_data.py --config configs/data.yaml

A copy of the manifest goes to a tracked path, because data/ isn't tracked and the checksums of
the locked test sets should be on record in git.
"""

import argparse
from pathlib import Path

from sandeha.data.generate import generate, load_config
from sandeha.data.store import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest-copy", type=Path, default=ROOT / "results" / "data_manifest.json")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted changes, which the manifest records. Not for the real dataset",
    )
    args = parser.parse_args()

    manifest = generate(
        load_config(args.config),
        args.root,
        allow_dirty=args.allow_dirty,
        log=lambda line: print(line, flush=True),
    )
    write_json_atomic(args.manifest_copy, manifest)


if __name__ == "__main__":
    main()
