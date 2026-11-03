"""Run the experiments. Without --final, only validation opens.

    scripts/long_job.sh experiments_val uv run python scripts/run_experiments.py
    scripts/long_job.sh experiments_final uv run python scripts/run_experiments.py --final

Reads results/ensemble.json and results/handoff_selection.json, and any ablation under
runs/ablation. A plain run writes to runs/experiments/val_results, and --final to results/. Restart the
same command after a stop: finished rollout sets are kept in runs/experiments. A set made by other code
runs again, with the old one moved to superseded/ there. An ablation that finishes later joins the run.

--dry-run does all of this on the CPU, in a temporary directory or --out: small random models on one
tiny solved split under every set's name. It opens no real split and writes nothing in the repo. It
also makes its own tiny hand-off selection, and takes a threshold even if that chose **never**, so that
every path runs.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
import yaml
from select_handoff import DRY_RUN_MODEL, provenance, solver_for, trained_ensemble

from sandeha.data.dataset import Normalizer
from sandeha.data.generate import git_state
from sandeha.data.store import read_manifest, sha256_file
from sandeha.ensemble import Ensemble
from sandeha.eval.experiments import EXPERIMENTS, Evaluation, make_tiny_sets, open_eval_sets, run_experiments
from sandeha.handoff_selection import Window, select_handoff
from sandeha.models.factory import build_model
from sandeha.physics.solver import SmokeSolver
from sandeha.train.loop import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
ABLATIONS = ("no_divergence", "no_pushforward")
DRY_RUN_CONFIG = {
    "start": 1,
    "primary_end": 6,
    "long_end": 7,
    "n_resamples": 50,
    "usable_error": 1.0,
    "rollout_lengths": [1, 3, 5],
    "timing_scenes": 1,
    "timing_frames": 1,
    "default_handoff_frames": 2,
    "random_seeds": [0, 1],
    "ensemble_sizes": [2, 3],
    "sweep": {"periods": [3], "rates": [0.2]},
}


def _random_model(seed: int):
    torch.manual_seed(seed)
    return build_model(DRY_RUN_MODEL)


def _digest(model) -> str:
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def code_sha256(root: Path = REPO) -> str:
    """The code that makes the rollouts, as it stands on disk. Docs, results and the demo don't count."""
    sources = [path for path in sorted((root / "src" / "sandeha").rglob("*.py")) if "demo" not in path.parts]
    named = [root / "scripts" / "run_experiments.py", root / "scripts" / "select_handoff.py", root / "uv.lock"]
    digest = hashlib.sha256()
    for path in [*sources, *(path for path in named if path.exists())]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _forced_threshold(selection: dict) -> dict:
    """A dry run only: the run with the largest share, so that the baselines and the oracle are run."""
    runs = [run for run in selection["runs"] if run["tau"] is not None]
    chosen = max(runs, key=lambda run: run["solver_share"])
    fields = ("aggregation", "handoff_frames", "tau", "set_score", "solver_share", "forced_handoffs")
    return {**selection, "choice": {**selection["choice"], "kind": "dry_run_forced", **{k: chosen[k] for k in fields}}}


def dry_run(out: Path, names) -> None:
    stores, normalizer, config = make_tiny_sets(out / "data")
    ensemble = Ensemble([_random_model(seed) for seed in range(3)], normalizer)
    solver = SmokeSolver(config["res"], "cpu", config["dt"])
    window = Window(DRY_RUN_CONFIG["start"], DRY_RUN_CONFIG["primary_end"])
    selection = select_handoff(
        ensemble,
        solver,
        stores["val"],
        progress_path=out / "selection_progress.json",
        window=window,
        percentiles=(50, 90),
        handoff_frames=(2,),
        substeps=config["steps_per_frame"],
        n_resamples=DRY_RUN_CONFIG["n_resamples"],
        provenance={"dry_run": True},
    )
    commit, dirty = git_state()
    ablations = {name: _random_model(10 + i) for i, name in enumerate(ABLATIONS)}
    evaluation = Evaluation(
        ensemble=ensemble,
        solver=solver,
        coarse_solver=SmokeSolver(config["coarse_res"], "cpu", config["dt"]),
        substeps=config["steps_per_frame"],
        amp=False,
        selection=_forced_threshold(selection),
        config=DRY_RUN_CONFIG,
        provenance={
            "git_commit": commit,
            "git_dirty": dirty,
            "code_sha256": code_sha256(),
            "data_manifest_sha256": sha256_file(stores["val"].directory / "fields.npy"),
            "checkpoint_sha256": [_digest(model) for model in ensemble.models],
            "ablation_checkpoint_sha256": {name: _digest(model) for name, model in ablations.items()},
            "config": DRY_RUN_CONFIG,
            "dry_run": True,
        },
        progress_dir=out / "progress",
        ablation_models=ablations,
    )
    run_experiments(evaluation, stores, stores["val"], names, out, final=False)


def _ablation_models(run_root: Path, device: str) -> tuple[dict, dict]:
    """The reduced trainings that have finished. Any other is said to be missing, not guessed at."""
    models, hashes = {}, {}
    for name in ABLATIONS:
        path = run_root / "ablation" / name / "best.pt"
        if not path.exists():
            print(f"{path} is missing, so that ablation isn't measured.")
            continue
        saved = load_checkpoint(path, map_location=device)
        model = build_model(saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        models[name] = model
        hashes[name] = sha256_file(path)
    return models, hashes


def real_run(args) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.selection.exists():
        raise SystemExit(f"{args.selection} is missing: run scripts/select_handoff.py first, and commit it.")
    selection = json.loads(args.selection.read_text())
    config = yaml.safe_load(args.config.read_text())
    normalizer = Normalizer.load(args.data_root / "normalizer.json")
    ensemble, hashes, amp = trained_ensemble(args.ensemble, normalizer, device)
    ablations, ablation_hashes = _ablation_models(args.run_root, device)
    solver, substeps = solver_for(args.data_root, device)
    data = read_manifest(args.data_root)["config"]
    coarse = SmokeSolver(data["coarse_res"], device, data["dt"], data.get("cg_tol", 1e-5), data.get("cg_max_iter", 2000))
    stores, reference = open_eval_sets(args.data_root, args.final)
    mode = "final" if args.final else "val"
    evaluation = Evaluation(
        ensemble=ensemble,
        solver=solver,
        coarse_solver=coarse,
        substeps=substeps,
        amp=amp,
        selection=selection,
        config=config,
        provenance={
            **provenance(args.data_root, hashes),
            "code_sha256": code_sha256(),
            "ablation_checkpoint_sha256": ablation_hashes,
            "config": config,
            "selection_sha256": sha256_file(args.selection),
        },
        progress_dir=args.progress / mode,
        ablation_models=ablations,
    )
    out = args.results if args.final else args.progress / "val_results"
    run_experiments(evaluation, stores, reference, args.experiments, out, final=args.final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ensemble", type=Path, default=Path("results/ensemble.json"))
    parser.add_argument("--selection", type=Path, default=Path("results/handoff_selection.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--progress", type=Path, default=Path("runs/experiments"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--final", action="store_true", help="Open the test splits. Run this once.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="Where a dry run writes. Default: a temporary directory.")
    args = parser.parse_args()
    if args.out and not args.dry_run:
        parser.error("--out is for --dry-run")
    if args.final and args.dry_run:
        parser.error("--final and --dry-run don't go together")

    if args.dry_run:
        with tempfile.TemporaryDirectory() as temporary:
            out = args.out or Path(temporary)
            out.mkdir(parents=True, exist_ok=True)
            dry_run(out, args.experiments)
        return
    real_run(args)


if __name__ == "__main__":
    main()
