"""The training runs of docs/PLAN.md Task 6: architecture choice, ensemble and ablations.

Every run is restartable: a finished run is skipped, a partial one resumes from its ``last.pt``,
and a member's check is kept beside its checkpoint.
"""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from sandeha.data.dataset import Normalizer
from sandeha.data.generate import generate, git_state
from sandeha.data.store import open_split, sha256_file, write_json_atomic
from sandeha.models.factory import build_model
from sandeha.train.loop import (
    ERROR_CAP,
    capped_mean,
    load_checkpoint,
    rollout_error,
    rollout_frame_errors,
    train,
)

# Tiny enough for the CPU, for the scripts' --dry-run.
DRY_RUN_DATA = {
    "seed": 12,
    "res": 64,
    "coarse_res": 32,
    "dt": 0.01,
    "steps_per_frame": 4,
    "frames": 8,
    "batch_size": 2,
    "fine_frames": [2],
    "scenes": {"train": 3, "val": 2, "test_a": 1, "test_b": 1, "test_c": 2},
}
DRY_RUN_MODELS = {
    "unet": {"arch": "unet", "base_width": 8, "levels": 2},
    "fno": {"arch": "fno", "width": 8, "modes": 4, "layers": 2},
}
DRY_RUN_CHECK = {"check_start": 1, "check_steps": 6}


def dry_run_config(cfg: dict, root: Path, minutes: float) -> dict:
    """A real training config cut down to the tiny data, on the CPU, for ``minutes``."""
    return cfg | {
        "data_root": str(root),
        "device": "cpu",
        "model": DRY_RUN_MODELS[cfg["model"]["arch"]],
        "batch_size": 2,
        "num_workers": 0,
        "mixed_precision": False,
        "max_minutes": minutes,
        "timing_steps": 5,
        "eval_every": 20,
        "checkpoint_every": 10,
        "log_every": 10,
        "rollout_start": 1,
        "rollout_steps": 4,
        "eval_batch_size": 2,
    }


def make_dry_run_data(root: Path) -> Path:
    generate(DRY_RUN_DATA, root, allow_dirty=True, log=lambda _: None)
    return root


class Persistence(nn.Module):
    """Predicts no change, so a rollout stays at its first frame."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x[:, :3])


def _resolved(cfg: dict) -> tuple[str, bool]:
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, bool(cfg["mixed_precision"]) and device == "cuda"


def _provenance(cfg: dict) -> dict:
    commit, dirty = git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "data_manifest_sha256": sha256_file(Path(cfg["data_root"]) / "manifest.json"),
    }


def is_finished(run_dir: Path) -> bool:
    last, best = Path(run_dir) / "last.pt", Path(run_dir) / "best.pt"
    if not (last.exists() and best.exists()):
        return False
    saved = load_checkpoint(last)
    return saved["step"] >= saved["total_steps"]


def run_member(cfg: dict, run_dir: Path) -> Path:
    """Skip a finished run, resume a partial one, or start a new one. Returns its best checkpoint."""
    run_dir = Path(run_dir)
    if is_finished(run_dir):
        print(f"{run_dir}: finished, skipped", flush=True)
        return run_dir / "best.pt"
    state = "resuming" if (run_dir / "last.pt").exists() else "starting"
    print(f"{run_dir}: {state}", flush=True)
    best = train(cfg, run_dir, resume=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def check_member(best_path: Path, cfg: dict, start: int, steps: int) -> dict:
    """Finite rollouts on validation, and an improvement over persistence, on the same frames.

    A frame counts as blown up as in docs/PREREGISTRATION.md: not finite, or above the cap.
    """
    device, amp = _resolved(cfg)
    root = Path(cfg["data_root"])
    normalizer = Normalizer.load(root / "normalizer.json")
    store = open_split(root, "val")
    saved = load_checkpoint(best_path, map_location=device)
    model = build_model(saved["config"]["model"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    window = (store, normalizer, start, steps, cfg["eval_batch_size"], device, amp)
    errors = rollout_frame_errors(model, *window)
    blown_up = int((~errors.isfinite() | (errors > ERROR_CAP)).sum())
    error = capped_mean(errors)
    persistence = capped_mean(rollout_frame_errors(Persistence(), *window))

    reasons = []
    if blown_up:
        reasons.append(f"blew up on {blown_up} of {errors.numel()} frames")
    if not error < persistence:
        reasons.append(f"no improvement over persistence: {error:.4f} against {persistence:.4f}")
    return {
        "checkpoint_step": saved["step"],
        "start": start,
        "steps": steps,
        "scenes": store.n_scenes,
        "blown_up_frames": blown_up,
        "finite": not blown_up,
        "error": error,
        "persistence_error": persistence,
        "improved": error < persistence,
        "passed": not reasons,
        "reason": "; ".join(reasons) or None,
    }


def _trained_and_checked(cfg: dict, seed: int, settings: dict, run_root: Path) -> tuple[Path, dict]:
    run_dir = Path(run_root) / f"seed_{seed}"
    best = run_member(cfg | {"seed": seed, "max_minutes": settings["max_minutes"]}, run_dir)
    kept = run_dir / "check.json"
    if kept.exists():
        return best, json.loads(kept.read_text())
    result = check_member(best, cfg, settings["check_start"], settings["check_steps"])
    write_json_atomic(kept, result)
    return best, result


def train_ensemble(cfg: dict, settings: dict, run_root: Path, results_path: Path) -> dict:
    """Train and check each seed from nothing, and retrain a failure once. Report after each one."""
    seeds = settings["seeds"]
    report = {
        "arch": cfg["model"]["arch"],
        "max_minutes": settings["max_minutes"],
        "seeds": seeds,
        "split": "val",
        "members": [],
        "complete": False,
        **_provenance(cfg),
    }
    for seed in seeds:
        trained_seed, because = seed, None
        best, check = _trained_and_checked(cfg, seed, settings, run_root)
        if not check["passed"]:
            because = check["reason"]
            trained_seed = seed + settings["retry_seed_offset"]
            print(f"member {seed} failed its check, retraining as seed {trained_seed}: {because}", flush=True)
            best, check = _trained_and_checked(cfg, trained_seed, settings, run_root)
        report["members"].append(
            {
                "seed": seed,
                "trained_seed": trained_seed,
                "retrained_because": because,
                "passed": check["passed"],
                "checkpoint": str(best),
                "checkpoint_sha256": sha256_file(best),
                "check": check,
            }
        )
        report["complete"] = len(report["members"]) == len(seeds) and all(
            member["passed"] for member in report["members"]
        )
        write_json_atomic(Path(results_path), report)
        state = "passed" if check["passed"] else f"FAILED: {check['reason']}"
        print(f"member {seed} finished (seed {trained_seed}), check {state}", flush=True)
    return report


def pick_winner(candidates: dict) -> str:
    """The lowest validation error. A tie goes to the U-Net, the expected backbone."""
    valid = [name for name, found in candidates.items() if "val_error" in found]
    return min(valid, key=lambda name: (candidates[name]["val_error"], name != "unet"))


def _candidate(cfg: dict, run_dir: Path, best: Path) -> dict:
    saved, last = load_checkpoint(best), load_checkpoint(run_dir / "last.pt")
    return {
        "val_error": saved["val_error"],
        "best_step": saved["step"],
        "total_steps": last["total_steps"],
        "skipped_steps": last["skipped_steps"],
        "seconds_per_step": last["seconds_per_step"],
        "parameters": sum(p.numel() for p in build_model(cfg["model"]).parameters()),
        "checkpoint": str(best),
        "checkpoint_sha256": sha256_file(best),
    }


def select_architecture(configs: dict[str, dict], run_root: Path, results_path: Path) -> dict:
    """Train each architecture and write the winner on validation rollout error.

    The U-Net must train. An FNO that fails is recorded and the U-Net goes on (docs/DESIGN.md, 5.2).
    """
    candidates = {}
    for name, cfg in configs.items():
        run_dir = Path(run_root) / name
        try:
            candidates[name] = _candidate(cfg, run_dir, run_member(cfg, run_dir))
        except RuntimeError as failure:
            if name == "unet":
                raise
            print(f"{name} failed, continuing without it: {failure}", flush=True)
            candidates[name] = {"failed": str(failure)}

    cfg = configs["unet"]
    device, amp = _resolved(cfg)
    root = Path(cfg["data_root"])
    report = {
        "winner": pick_winner(candidates),
        "split": "val",
        "measure": "mean RMS rollout error in normalised field units, on fluid cells",
        "rollout_start": cfg["rollout_start"],
        "rollout_steps": cfg["rollout_steps"],
        "seed": cfg["seed"],
        "max_minutes": cfg.get("max_minutes"),
        "candidates": candidates,
        "persistence_error": rollout_error(
            Persistence(), open_split(root, "val"), Normalizer.load(root / "normalizer.json"),
            cfg["rollout_start"], cfg["rollout_steps"], cfg["eval_batch_size"], device, amp,
        ),
        **_provenance(cfg),
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(results_path), report)
    return report


def train_ablations(cfg: dict, overlays: dict[str, dict], run_root: Path) -> dict[str, Path]:
    return {name: run_member(cfg | overlay, Path(run_root) / name) for name, overlay in overlays.items()}
