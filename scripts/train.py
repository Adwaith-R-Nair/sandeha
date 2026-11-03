"""Train one model from a config. Restart a stopped run with --resume."""

import argparse
from pathlib import Path

import yaml

from sandeha.train.loop import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_minutes is not None:
        cfg["max_minutes"] = args.max_minutes
    print(train(cfg, args.run_dir, resume=args.resume))


if __name__ == "__main__":
    main()
