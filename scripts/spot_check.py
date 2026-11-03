"""The spot-check after generation: stats, NaN count, manifest, and a sample scene from each split.

Writes results/spot_check.json and results/figures/spot_check.png, and exits with status 1 if a
check fails. The figure is for a person to inspect. The travel rule is a separate step.

Usage: uv run python scripts/spot_check.py --data-root data
"""

import argparse
import json
from pathlib import Path

from sandeha.data.generate import git_state
from sandeha.data.spotcheck import run_spot_check

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "spot_check.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "results" / "figures" / "spot_check.png")
    args = parser.parse_args()

    commit, dirty = git_state()
    result = {"git_commit": commit, "git_dirty": dirty, **run_spot_check(args.data_root, args.figure)}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    if not result["passed"]:
        raise SystemExit("the spot-check failed: see the result above")


if __name__ == "__main__":
    main()
