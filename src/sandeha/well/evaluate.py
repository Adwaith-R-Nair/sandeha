"""Doubt on the public dataset: choices fixed on the official valid split, then the test split.

The aggregation, the bad-frame threshold and the calibration edges are fixed on valid and saved
before the test split opens, and the final evaluation refuses other checkpoints than those.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from sandeha.data.generate import git_state
from sandeha.data.store import sha256_file, write_json_atomic
from sandeha.models.factory import build_model
from sandeha.train.loop import load_checkpoint
from sandeha.well.data import NORMALIZER_FILE, WellNormalizer, WellSplit, check_counts, load_well_normalizer
from sandeha.well.doubt import (
    AGGREGATIONS,
    auroc,
    calibration,
    decile_edges,
    evidence_measure,
    evidence_rollouts,
    is_bad,
    risk_coverage,
    single_rollout,
    spearman,
    within_time_auroc,
)

LABEL = (
    "Published scientific simulation benchmark (The Well, turbulent_radiative_layer_2D), "
    "not measured data."
)
NOT_EVIDENCE = "Only within_time_spearman counts as evidence. The other measures don't by themselves."
NO_RISE = "not applicable: this dataset has one test set, with no familiar and unfamiliar sets to compare"


def _device_settings(cfg: dict) -> tuple[str, bool]:
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, bool(cfg["mixed_precision"]) and device == "cuda"


def _load_members(paths: list[Path], normalizer: WellNormalizer, device: str) -> list[nn.Module]:
    models = []
    for path in paths:
        saved = load_checkpoint(path, map_location=device)
        if saved["normalizer"] != normalizer.to_dict():
            raise ValueError(f"{path} was trained with a different normaliser")
        model = build_model(saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        models.append(model.eval())
    return models


def _checkpoint_hashes(paths: list[Path]) -> dict[str, str]:
    return {Path(path).parent.name: sha256_file(Path(path)) for path in paths}


def _provenance(cfg: dict, paths: list[Path], split: WellSplit) -> dict:
    commit, dirty = git_state()
    root = Path(cfg["data_root"])
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "config": cfg,
        "checkpoints_sha256": _checkpoint_hashes(paths),
        "data_sha256": {
            **{f"{split.split}/{path.name}": sha256_file(path) for path in split.files},
            NORMALIZER_FILE: sha256_file(root / NORMALIZER_FILE),
        },
    }


def _rollouts(cfg: dict, paths: list[Path], split: WellSplit) -> dict:
    device, amp = _device_settings(cfg)
    normalizer = load_well_normalizer(cfg["data_root"])
    models = _load_members(paths, normalizer, device)
    evidence = cfg["evidence"]
    steps = evidence["secondary_last"] - evidence["start"]
    settings = (normalizer, evidence["start"], steps, cfg["eval_batch_size"], device, amp)
    # The lowest-seed member gives the no-ensemble signal, so ``paths`` are in seed order.
    return {
        "ensemble": evidence_rollouts(models, split, *settings),
        "single": single_rollout(models[0], split, *settings),
    }


def _window(cfg: dict, values: Tensor, last_frame: int) -> Tensor:
    """Columns for frames ``first`` to ``last_frame``, where column k is frame ``start + k + 1``."""
    evidence = cfg["evidence"]
    return values[:, evidence["first"] - evidence["start"] - 1 : last_frame - evidence["start"]]


def _evidence_figure(cfg: dict, scores: Tensor, errors: Tensor) -> dict:
    evidence = cfg["evidence"]
    return evidence_measure(scores, errors, evidence["bootstrap"], evidence["seed"])


def _measures(cfg: dict, scores: Tensor, errors: Tensor, threshold: float, edges: list[float]) -> dict:
    bad = is_bad(errors, threshold)
    return {
        "within_time_spearman": _evidence_figure(cfg, scores, errors),
        "pooled_spearman": spearman(scores.ravel(), errors.ravel()),
        "within_time_auroc": within_time_auroc(scores, bad, cfg["evidence"]["min_each_class"]),
        "pooled_auroc": auroc(scores.ravel(), bad.ravel()),
        "bad_frames": int(bad.sum()),
        "frames": bad.numel(),
        "non_finite_frames": int((~torch.isfinite(scores)).sum()),
        "capped_frames": int((errors >= 10).sum()),
        "mean_error": errors.double().mean().item(),
        "calibration": calibration(scores, errors, edges, threshold),
        "risk_coverage": risk_coverage(scores, errors),
    }


def select_on_valid(cfg: dict, paths: list[Path], selection_path: Path) -> dict:
    """Fix the aggregation, bad-frame threshold and calibration edges. Opens valid only."""
    split = WellSplit(cfg["data_root"], "valid")
    check_counts(split, cfg["expected"])
    runs = _rollouts(cfg, paths, split)
    last = cfg["evidence"]["last"]
    errors = _window(cfg, runs["ensemble"]["errors"], last)
    scores = {name: _window(cfg, runs["ensemble"]["scores"][name], last) for name in AGGREGATIONS}
    figures = {name: _evidence_figure(cfg, scores[name], errors) for name in AGGREGATIONS}
    # The first one wins a tie. An aggregation with no figure can't lead one that has one.
    leader = max(AGGREGATIONS, key=lambda name: -np.inf if figures[name]["value"] is None else figures[name]["value"])
    percentile = cfg["evidence"]["bad_percentile"]
    selection = {
        "label": LABEL,
        "split": split.split,
        "aggregation": leader,
        "aggregation_figures": figures,
        "bad_threshold": float(np.percentile(errors.double().numpy(), percentile)),
        "doubt_deciles": decile_edges(scores[leader]),
        "divergence_deciles": decile_edges(_window(cfg, runs["single"]["scores"], last)),
        "provenance": _provenance(cfg, paths, split),
    }
    write_json_atomic(Path(selection_path), selection)
    return selection


def _window_results(cfg: dict, runs: dict, selection: dict, last_frame: int) -> dict:
    ensemble, single = runs["ensemble"], runs["single"]
    aggregation = selection["aggregation"]
    return {
        "note": NOT_EVIDENCE,
        "doubt": _measures(
            cfg,
            _window(cfg, ensemble["scores"][aggregation], last_frame),
            _window(cfg, ensemble["errors"], last_frame),
            selection["bad_threshold"],
            selection["doubt_deciles"],
        ),
        "no_ensemble_divergence": _measures(
            cfg,
            _window(cfg, single["scores"], last_frame),
            _window(cfg, single["errors"], last_frame),
            selection["bad_threshold"],
            selection["divergence_deciles"],
        ),
    }


def evaluate_final(cfg: dict, paths: list[Path], selection_path: Path, result_path: Path) -> dict:
    """Run the frozen choices on the official test split, and write the result."""
    selection_path = Path(selection_path)
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} is missing: run the selection on valid first")
    selection = json.loads(selection_path.read_text())
    if selection["provenance"]["checkpoints_sha256"] != _checkpoint_hashes(paths):
        raise ValueError("these checkpoints aren't the ones the selection was made with")
    split = WellSplit(cfg["data_root"], "test", True)
    check_counts(split, cfg["expected"])
    runs = _rollouts(cfg, paths, split)
    evidence = cfg["evidence"]
    result = {
        "label": LABEL,
        "split": split.split,
        "trajectories": split.n_trajectories,
        "evidence_figure": "evidence.doubt.within_time_spearman",
        "aggregation": selection["aggregation"],
        "bad_threshold": selection["bad_threshold"],
        "evidence": _window_results(cfg, runs, selection, evidence["last"]),
        "secondary": _window_results(cfg, runs, selection, evidence["secondary_last"]),
        "rise_on_unfamiliar": NO_RISE,
        "provenance": _provenance(cfg, paths, split) | {"selection_sha256": sha256_file(selection_path)},
    }
    write_json_atomic(Path(result_path), result)
    return result
