"""Doubt on the public track. Without --final, fixes the choices on valid. With it, runs test."""

import argparse
from pathlib import Path

import yaml

from sandeha.well.evaluate import LABEL, evaluate_final, select_on_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/well.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/well"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--final", action="store_true", help="open the official test split")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = [args.run_root / f"member_{seed}" / "best.pt" for seed in cfg["seeds"]]
    selection_path = args.results / "well_selection.json"
    print(LABEL)
    if args.final:
        result_path = args.results / "exp_well.json"
        figure = evaluate_final(cfg, paths, selection_path, result_path)["evidence"]["doubt"]
        print(result_path, figure["within_time_spearman"])
    else:
        selection = select_on_valid(cfg, paths, selection_path)
        print(selection_path, selection["aggregation"], selection["aggregation_figures"])


if __name__ == "__main__":
    main()
