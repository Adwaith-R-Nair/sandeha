"""Choose the doubt aggregation, hand-off length and frozen threshold, on validation only.

    scripts/long_job.sh select_handoff uv run python scripts/select_handoff.py

Reads the trained ensemble from results/ensemble.json and writes results/handoff_selection.json.
Restart the same command after a stop: finished rollout sets are kept in runs/handoff_selection.
--dry-run does the same on the CPU, with small random models on a tiny solved split in a temporary
directory, and writes nothing in the repo.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from sandeha.data.dataset import Normalizer
from sandeha.data.generate import git_state
from sandeha.data.store import open_split, read_manifest, sha256_file, write_json_atomic
from sandeha.ensemble import Ensemble
from sandeha.handoff_selection import HANDOFF_FRAMES, PERCENTILES, Window, make_tiny_val, select_handoff
from sandeha.models.factory import build_model
from sandeha.physics.solver import SmokeSolver
from sandeha.train.loop import load_checkpoint

DRY_RUN_MODEL = {"arch": "unet", "base_width": 8, "levels": 2}
DRY_RUN_WINDOW = Window(1, 6)
DRY_RUN_PERCENTILES = (50, 90)
DRY_RUN_RESAMPLES = 50


def solver_for(root: Path, device: str) -> tuple[SmokeSolver, int]:
    config = read_manifest(root)["config"]
    solver = SmokeSolver(
        config["res"],
        device,
        config["dt"],
        cg_tol=config.get("cg_tol", 1e-5),
        cg_max_iter=config.get("cg_max_iter", 2000),
    )
    return solver, config["steps_per_frame"]


def trained_ensemble(report_path: Path, normalizer: Normalizer, device: str) -> tuple[Ensemble, list[str], bool]:
    """The ensemble in the training report, each checkpoint checked against it."""
    if not report_path.exists():
        raise SystemExit(f"{report_path} is missing: train the ensemble first, with scripts/train_ensemble.py")
    report = json.loads(report_path.read_text())
    if not report["complete"]:
        raise SystemExit(f"{report_path} says the ensemble isn't complete, so there is nothing to select with")
    models, hashes, amp = [], [], False
    for member in report["members"]:
        path = Path(member["checkpoint"])
        if sha256_file(path) != member["checkpoint_sha256"]:
            raise SystemExit(f"{path} doesn't match its checksum in {report_path}")
        saved = load_checkpoint(path, map_location=device)
        model = build_model(saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        models.append(model)
        hashes.append(member["checkpoint_sha256"])
        amp = bool(saved["config"]["mixed_precision"]) and device == "cuda"
    return Ensemble(models, normalizer), hashes, amp


def provenance(root: Path, hashes: list[str]) -> dict:
    commit, dirty = git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "data_manifest_sha256": sha256_file(root / "manifest.json"),
        "checkpoint_sha256": hashes,
    }


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report)
    print(json.dumps(report["choice"], indent=1))
    print(f"written to {path}. Commit it before any test split opens.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ensemble", type=Path, default=Path("results/ensemble.json"))
    parser.add_argument("--results", type=Path, default=Path("results/handoff_selection.json"))
    parser.add_argument("--progress", type=Path, default=Path("runs/handoff_selection/progress.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="Where a dry run writes. Default: a temporary directory.")
    args = parser.parse_args()
    if args.out and not args.dry_run:
        parser.error("--out is for --dry-run")

    if args.dry_run:
        with tempfile.TemporaryDirectory() as temporary:
            out = args.out or Path(temporary)
            out.mkdir(parents=True, exist_ok=True)
            store, normalizer, config = make_tiny_val(out / "val")
            models = []
            for seed in range(3):
                torch.manual_seed(seed)
                models.append(build_model(DRY_RUN_MODEL))
            report = select_handoff(
                Ensemble(models, normalizer),
                SmokeSolver(config["res"], "cpu", config["dt"]),
                store,
                progress_path=out / "progress.json",
                window=DRY_RUN_WINDOW,
                percentiles=DRY_RUN_PERCENTILES,
                substeps=config["steps_per_frame"],
                n_resamples=DRY_RUN_RESAMPLES,
                provenance={"dry_run": True},
            )
            write(report, out / "handoff_selection.json")
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    normalizer = Normalizer.load(args.data_root / "normalizer.json")
    ensemble, hashes, amp = trained_ensemble(args.ensemble, normalizer, device)
    solver, substeps = solver_for(args.data_root, device)
    report = select_handoff(
        ensemble,
        solver,
        open_split(args.data_root, "val"),
        progress_path=args.progress,
        percentiles=PERCENTILES,
        handoff_frames=HANDOFF_FRAMES,
        substeps=substeps,
        amp=amp,
        provenance=provenance(args.data_root, hashes),
    )
    write(report, args.results)


if __name__ == "__main__":
    main()
