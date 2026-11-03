"""Train the public track's members. Run again to resume, and finished members are left alone."""

import argparse
from pathlib import Path

import yaml

from sandeha.well.train import train_members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/well.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/well"))
    args = parser.parse_args()
    for path in train_members(yaml.safe_load(args.config.read_text()), args.run_root):
        print(path)


if __name__ == "__main__":
    main()
