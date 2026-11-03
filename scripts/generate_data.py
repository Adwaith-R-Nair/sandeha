"""Generate the dataset, or resume an interrupted run.

Usage: scripts/long_job.sh data uv run scripts/generate_data.py --config configs/data.yaml

A copy of the manifest goes to a tracked path, because data/ isn't tracked and the checksums of
the locked test sets should be on record in git.

Refuses to start until scripts/check_travel.py --control-sample has been measured for this config
and committed: docs/PREREGISTRATION.md requires those figures to be logged first. Committed follows
from the check for a clean tree, since an untracked result in results/ makes the tree dirty.
"""

import argparse
from pathlib import Path

from sandeha.data.generate import generate, load_config
from sandeha.data.store import write_json_atomic
from sandeha.data.travel import ControlsNotReady, check_controls, controls_gate_applies

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
    parser.add_argument("--controls", type=Path, default=ROOT / "results" / "travel_controls.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if controls_gate_applies(args.root, args.allow_dirty):
        try:
            check_controls(config, args.controls)
        except ControlsNotReady as error:
            raise SystemExit(str(error))
    manifest = generate(
        config,
        args.root,
        allow_dirty=args.allow_dirty,
        log=lambda line: print(line, flush=True),
    )
    write_json_atomic(args.manifest_copy, manifest)


if __name__ == "__main__":
    main()
