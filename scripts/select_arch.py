"""Train the U-Net and the FNO, and write the winner on validation rollout error.

    scripts/long_job.sh select_arch uv run python scripts/select_arch.py

Restart the same command after a stop: a finished model is skipped, and a partial one resumes.
--dry-run does the same on tiny data in a temporary directory, in about a minute, and writes
nothing in the repo.
"""

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from sandeha.train.campaign import dry_run_config, make_dry_run_data, select_architecture

DRY_RUN_MINUTES = 0.2
ARCHS = ("unet", "fno")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("runs/select"))
    parser.add_argument("--results", type=Path, default=Path("results/arch_selection.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="Where a dry run writes. Default: a temporary directory.")
    args = parser.parse_args()
    if args.out and not args.dry_run:
        parser.error("--out is for --dry-run")

    configs = {arch: yaml.safe_load(Path(f"configs/train_{arch}.yaml").read_text()) for arch in ARCHS}
    if not args.dry_run:
        report = select_architecture(configs, args.run_root, args.results)
        print(json.dumps(report, indent=1))
        print(f"winner: {report['winner']}. Record it in docs/DECISIONS.md.")
        return

    with tempfile.TemporaryDirectory() as temporary:
        out = args.out or Path(temporary)
        out.mkdir(parents=True, exist_ok=True)
        root = make_dry_run_data(out / "data")
        configs = {arch: dry_run_config(cfg, root, DRY_RUN_MINUTES) for arch, cfg in configs.items()}
        report = select_architecture(configs, out / "select", out / "arch_selection.json")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
