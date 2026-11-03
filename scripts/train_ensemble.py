"""Train and check the ensemble with the winning architecture, or its two ablations.

    scripts/long_job.sh ensemble uv run python scripts/train_ensemble.py
    scripts/long_job.sh ablations uv run python scripts/train_ensemble.py --ablations

Restart the same command after a stop: a finished member is skipped, and a partial one resumes.
If time runs short, stop with --seeds 0 1 2 and record the change. Exits with 1 if a member failed
its check twice. --dry-run does the same on tiny data in a temporary directory, in under a minute,
and writes nothing in the repo.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from sandeha.train.campaign import (
    DRY_RUN_CHECK,
    dry_run_config,
    make_dry_run_data,
    train_ablations,
    train_ensemble,
)

ABLATIONS = ("no_divergence", "no_pushforward")
DRY_RUN_MEMBER_MINUTES = 0.05
DRY_RUN_ABLATION_MINUTES = 0.1


def overlays() -> dict[str, dict]:
    return {name: yaml.safe_load(Path(f"configs/ablation_{name}.yaml").read_text()) for name in ABLATIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=Path("results/arch_selection.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/ensemble.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results/ensemble.json"))
    parser.add_argument("--ablations", action="store_true", help="Train the ablations, not the ensemble.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="Where a dry run writes. Default: a temporary directory.")
    args = parser.parse_args()
    if args.out and not args.dry_run:
        parser.error("--out is for --dry-run")

    settings = yaml.safe_load(args.config.read_text())
    if args.seeds:
        settings["seeds"] = args.seeds

    if args.dry_run:
        with tempfile.TemporaryDirectory() as temporary:
            out = args.out or Path(temporary)
            out.mkdir(parents=True, exist_ok=True)
            root = make_dry_run_data(out / "data")
            cfg = dry_run_config(yaml.safe_load(Path("configs/train_unet.yaml").read_text()), root, 0)
            if args.ablations:
                minutes = {"max_minutes": DRY_RUN_ABLATION_MINUTES}
                train_ablations(cfg, {name: o | minutes for name, o in overlays().items()}, out / "ablation")
                return
            settings |= DRY_RUN_CHECK | {"max_minutes": DRY_RUN_MEMBER_MINUTES}
            report = train_ensemble(cfg, settings, out / "ensemble", out / "ensemble.json")
            print(json.dumps(report, indent=1))
            return

    arch = json.loads(args.selection.read_text())["winner"]
    cfg = yaml.safe_load(Path(f"configs/train_{arch}.yaml").read_text())
    if args.ablations:
        print(train_ablations(cfg, overlays(), args.run_root / "ablation"))
        return
    report = train_ensemble(cfg, settings, args.run_root / "ensemble", args.results)
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
