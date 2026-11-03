"""The hand-off selection on validation, by the steps in docs/PREREGISTRATION.md.

Runs in the order the preregistration gives: the aggregations left in, their candidate thresholds,
a hand-off rollout set for each, and the choice. Validation's evidence rollouts also fix the
bad-frame threshold and the calibration edges. Every finished rollout set is kept in a progress
file, so a stopped run restarts where it was.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sandeha.data.dataset import Normalizer
from sandeha.data.generate import _simulate_batch
from sandeha.data.store import SplitStore, write_json_atomic
from sandeha.ensemble import Ensemble, doubt_score
from sandeha.hybrid import AlwaysAI, DoubtThreshold, HybridSimulator
from sandeha.physics.scenes import sample_scene
from sandeha.train.loop import ERROR_CAP

AGGREGATIONS = ("mean", "p95", "p99")
HANDOFF_FRAMES = (2, 5, 10)
PERCENTILES = (50, 60, 70, 80, 90, 95, 98, 99)
MAX_SOLVER_SHARE = 0.25
BAD_FRAME_PERCENTILE = 80
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_RESAMPLES = 2000
SCORE_DECIMALS = 6
_FLOOR_SHARE = 0.01  # Of a channel's training standard deviation.


@dataclass(frozen=True)
class Window:
    """Rollouts start from the stored frame ``start``, and are scored on frames after it to ``end``."""

    start: int = 20
    end: int = 70

    @property
    def frames(self) -> int:
        return self.end - self.start


def frame_error(pred: np.ndarray, ref: np.ndarray, solid: np.ndarray, floors) -> float:
    """The preregistration's per-frame error of fields ``(3,H,W)``, capped.

    ``solid`` is the mask on the way into the frame, and ``floors`` is 1% of each channel's training
    standard deviation. A frame with no fluid cells has nothing to compare, and scores zero.
    """
    fluid = ~solid
    if not fluid.any():
        return 0.0
    total = 0.0
    for channel in range(3):
        predicted = pred[channel][fluid].astype(np.float64)
        if not np.isfinite(predicted).all():
            return ERROR_CAP
        reference = ref[channel][fluid].astype(np.float64)
        rms = np.sqrt(np.mean(np.square(predicted - reference)))
        total += rms / max(reference.std(), floors[channel])
    error = total / 3
    return float(error) if error <= ERROR_CAP else ERROR_CAP


def _average_ranks(column: np.ndarray) -> np.ndarray:
    ordered = np.sort(column)
    return (np.searchsorted(ordered, column, "left") + np.searchsorted(ordered, column, "right") + 1) / 2


def _time_correlations(doubt: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """The Spearman correlation at each rollout time, NaN where it is undefined. Both ``(scenes, times)``."""
    doubt = np.where(np.isnan(doubt), np.inf, doubt)  # Non-finite ranks above all, and ties.
    found = np.full(doubt.shape[1], np.nan)
    for time_index in range(doubt.shape[1]):
        a, b = doubt[:, time_index], errors[:, time_index]
        if (a == a[0]).all() or (b == b[0]).all():
            continue
        a = _average_ranks(a)
        b = _average_ranks(b)
        a, b = a - a.mean(), b - b.mean()
        found[time_index] = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    return found


def evidence_measure(doubt: np.ndarray, errors: np.ndarray) -> tuple[float, int]:
    """The mean correlation over rollout times, and how many were undefined. NaN if all of them were."""
    correlations = _time_correlations(doubt, errors)
    undefined = int(np.isnan(correlations).sum())
    if undefined == len(correlations):
        return math.nan, undefined
    return float(np.nanmean(correlations)), undefined


def _resampled_measures(doubt: np.ndarray, errors: np.ndarray, resamples: np.ndarray) -> np.ndarray:
    return np.array([evidence_measure(doubt[i], errors[i])[0] for i in resamples])


def _interval(values: np.ndarray) -> list[float] | None:
    values = values[~np.isnan(values)]
    if not len(values):
        return None
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def aggregations_left_in(
    doubts: dict[str, np.ndarray], errors: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES
) -> dict:
    """Step 1. The leader, and each other aggregation whose paired interval from it includes zero.

    A resample where either side has no figure leaves that difference out, and an aggregation with
    no interval at all can't show that it stays. A tie for the lead goes to the earlier aggregation.
    If none has a figure, none is left in and only **never** remains, so a choice still exists.
    """
    n_scenes = errors.shape[0]
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_scenes, (n_resamples, n_scenes))
    report = {}
    resampled = {}
    for name in AGGREGATIONS:
        measure, undefined = evidence_measure(doubts[name], errors)
        resampled[name] = _resampled_measures(doubts[name], errors, resamples)
        report[name] = {
            "measure": measure,
            "ci": _interval(resampled[name]),
            "undefined_times": undefined,
            "non_finite_frames": int((~np.isfinite(doubts[name])).sum()),
        }
    figures = [name for name in AGGREGATIONS if not math.isnan(report[name]["measure"])]
    if not figures:
        return {"leader": None, "left_in": [], "by_aggregation": report, "n_resamples": n_resamples}
    leader = max(figures, key=lambda name: (report[name]["measure"], -AGGREGATIONS.index(name)))

    left_in = []
    for name in AGGREGATIONS:
        entry = report[name]
        entry["difference_from_leader"] = None
        entry["difference_ci"] = None
        if name == leader:
            left_in.append(name)
        elif name in figures:
            entry["difference_from_leader"] = entry["measure"] - report[leader]["measure"]
            interval = _interval(resampled[name] - resampled[leader])
            entry["difference_ci"] = interval
            if interval is not None and interval[0] <= 0 <= interval[1]:
                left_in.append(name)
    return {"leader": leader, "left_in": left_in, "by_aggregation": report, "n_resamples": n_resamples}


def candidate_thresholds(scores: np.ndarray, percentiles=PERCENTILES) -> list[dict]:
    """Step 2. A percentile that isn't finite is left out, and equal values count once."""
    scores = np.where(np.isnan(scores), np.inf, scores).ravel()
    candidates, seen = [], set()
    with np.errstate(invalid="ignore"):
        values = np.percentile(scores, percentiles)
    for percentile, value in zip(percentiles, values, strict=True):
        tau = float(value)
        if math.isfinite(tau) and tau not in seen:
            seen.add(tau)
            candidates.append({"percentile": percentile, "tau": tau})
    return candidates


def choose_run(runs: list[dict]) -> dict:
    """Step 4. Among runs within the share, the lowest score after rounding, then the tie order.

    **never** has no aggregation or hand-off length. It goes first in a tie, since a run tying it at
    the same share gained nothing over no hand-off.
    """

    def key(run: dict) -> tuple:
        never = run["tau"] is None
        order = -1 if never else AGGREGATIONS.index(run["aggregation"])
        return (
            round(run["set_score"], SCORE_DECIMALS),
            run["solver_share"],
            order,
            0 if never else run["handoff_frames"],
        )

    return min((run for run in runs if run["solver_share"] <= MAX_SOLVER_SHARE), key=key)


def _scene_inputs(store, scene: int, window: Window, device) -> dict:
    fine = store.fine_scene(scene)

    def mask(array) -> torch.Tensor:
        return torch.from_numpy(array)[None, None]

    fields = np.array(store.fields[scene, window.start], dtype=np.float32)
    has_late = fine.late_obstacle is not None
    return {
        "state": torch.from_numpy(fields)[None].to(device),
        "obstacle": mask(fine.obstacle),
        "inflow": mask(fine.inflow),
        "buoyancy": fine.buoyancy,
        "frame": window.start,
        "late_obstacle": mask(fine.late_obstacle) if has_late else None,
        "late_frame": fine.late_frame,
    }


def _rollout(sim: HybridSimulator, store, scene: int, window: Window, floors, device, scores_by=()) -> dict:
    """One scene's rollout over the window: each frame's error, source and doubt scores."""
    sim.reset(**_scene_inputs(store, scene, window, device))
    errors, solver_frames, forced = [], 0, 0
    scores = {name: [] for name in scores_by}
    while sim.frame < window.end:
        first = sim.frame + 1
        info = sim.step()
        for name in scores_by:
            solid = torch.from_numpy(store.obstacle(scene, first - 1))[None, None].to(device)
            scores[name].append(float(doubt_score(info.doubt_map, solid, name)[0]))
        forced += int(info.forced)
        for k, fields in enumerate(info.frames):
            frame = first + k
            if frame > window.end:  # A hand-off running past the window counts only inside it.
                break
            ref = np.array(store.fields[scene, frame], dtype=np.float32)
            solid = store.obstacle(scene, frame - 1)
            errors.append(frame_error(fields[0].detach().cpu().numpy(), ref, solid, floors))
            solver_frames += int(info.source == "solver")
    return {"errors": errors, "solver_frames": solver_frames, "forced": forced, "scores": scores}


def _set_result(rollouts: list[dict], n_frames: int, n_resamples: int) -> dict:
    scene_scores = np.array([np.mean(r["errors"]) for r in rollouts])
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(scene_scores), (n_resamples, len(scene_scores))
    )
    return {
        "set_score": float(scene_scores.mean()),
        "set_score_ci": _interval(scene_scores[resamples].mean(axis=1)),
        "scene_scores": scene_scores.tolist(),
        "solver_share": sum(r["solver_frames"] for r in rollouts) / (n_frames * len(rollouts)),
        "forced_handoffs": sum(r["forced"] for r in rollouts),
        "blown_up_rollouts": sum(max(r["errors"]) >= ERROR_CAP for r in rollouts),
    }


def _decile_edges(scores: np.ndarray) -> list[float]:
    """The calibration bins' edges. Non-finite scores go in the top bin, and equal edges merge."""
    finite = scores[np.isfinite(scores)]
    return sorted(set(np.percentile(finite, range(10, 100, 10)).tolist())) if len(finite) else []


def _without_non_finite(value):
    """Non-finite numbers aren't JSON, so a report holds null in their place."""
    if isinstance(value, dict):
        return {key: _without_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_non_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def make_tiny_val(directory: Path, n_scenes: int = 3, seed: int = 12) -> tuple[SplitStore, Normalizer, dict]:
    """A tiny validation split solved on the CPU, for a dry run and for tests: no real split opens.

    Its normaliser comes from its own frames, which only a dry run may do.
    """
    config = {"res": 64, "coarse_res": 32, "dt": 0.01, "steps_per_frame": 4, "frames": 8, "fine_frames": [2]}
    rng = np.random.default_rng(seed)
    scenes = [sample_scene(rng, config["res"], "val") for _ in range(n_scenes)]
    result = _simulate_batch(config, scenes, torch.device("cpu"))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "fields.npy", result["fields"])
    np.savez(
        directory / "statics.npz",
        obstacle=result["obstacle"],
        late_obstacle=result["late_obstacle"],
        inflow=result["inflow"],
        fine_obstacle=np.stack([scene.obstacle for scene in scenes]),
        fine_late_obstacle=result["fine_late_obstacle"],
        fine_inflow=np.stack([scene.inflow for scene in scenes]),
        buoyancy=np.array([scene.buoyancy for scene in scenes]),
        late_frame=np.array([-1 if scene.late_frame is None else scene.late_frame for scene in scenes]),
    )
    fields = result["fields"].astype(np.float64)
    change = fields[:, 1:] - fields[:, :-1]
    normalizer = Normalizer(
        field_mean=fields.mean(axis=(0, 1, 3, 4)).tolist(),
        field_std=fields.std(axis=(0, 1, 3, 4)).tolist(),
        delta_std=np.sqrt(np.square(change).mean(axis=(0, 1, 3, 4))).tolist(),
        inflow_scale=float(result["inflow"].max()),
        buoyancy_mean=float(np.mean([scene.buoyancy for scene in scenes])),
        buoyancy_std=float(np.std([scene.buoyancy for scene in scenes])) or 1.0,
        fitted_on="val",
        n_scenes=n_scenes,
    )
    return SplitStore(directory, "val"), normalizer, config


def select_handoff(
    ensemble: Ensemble,
    solver,
    store,
    *,
    progress_path: Path,
    window: Window = Window(),
    percentiles=PERCENTILES,
    handoff_frames=HANDOFF_FRAMES,
    substeps: int = 4,
    amp: bool = False,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    provenance: dict | None = None,
    log=print,
) -> dict:
    """The selection's report. Only the validation split is accepted, and nothing here opens another."""
    if store.split != "val":
        raise ValueError(f"hand-off selection is made on validation only, not {store.split}")
    device = solver.device
    floors = [_FLOOR_SHARE * std for std in ensemble.normalizer.field_std]
    n_scenes = store.n_scenes
    provenance = provenance or {}
    commit = provenance.get("git_commit")
    settings = {
        "window": asdict(window),
        "percentiles": list(percentiles),
        "handoff_frames": list(handoff_frames),
        "substeps": substeps,
        "n_scenes": n_scenes,
    }
    # The commit stays out of this, so that a restart survives a commit touching only the docs. Each
    # run records its own instead, and the report lists them, so that mixed code can be seen.
    fingerprint = settings | {k: v for k, v in provenance.items() if k not in ("git_commit", "git_dirty")}

    progress_path = Path(progress_path)
    progress = {"fingerprint": fingerprint, "evidence": None, "runs": {}}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != fingerprint:
            raise ValueError(f"{progress_path} holds runs from other settings. Remove it to start again.")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    if progress["evidence"] is None:
        began = time.perf_counter()
        sim = HybridSimulator(ensemble, solver, AlwaysAI(), 1, substeps, amp=amp)
        rollouts = [
            _rollout(sim, store, scene, window, floors, device, scores_by=AGGREGATIONS)
            for scene in range(n_scenes)
        ]
        progress["evidence"] = {
            "errors": [r["errors"] for r in rollouts],
            "doubts": {name: [r["scores"][name] for r in rollouts] for name in AGGREGATIONS},
            "result": {**_set_result(rollouts, window.frames, n_resamples), "git_commit": commit},
        }
        write_json_atomic(progress_path, progress)
        log(f"evidence rollouts: {time.perf_counter() - began:.0f} s")

    errors = np.array(progress["evidence"]["errors"], dtype=np.float64)
    doubts = {
        name: np.array(progress["evidence"]["doubts"][name], dtype=np.float64) for name in AGGREGATIONS
    }

    aggregation = aggregations_left_in(doubts, errors, n_resamples)
    log(f"aggregations left in: {aggregation['left_in']}, leader {aggregation['leader']}")

    candidates = {
        name: candidate_thresholds(doubts[name], percentiles) for name in aggregation["left_in"]
    }
    plan = [
        (name, length, candidate)
        for name in aggregation["left_in"]
        for length in handoff_frames
        for candidate in candidates[name]
    ]
    for number, (name, length, candidate) in enumerate(plan, start=1):
        key = f"{name}|{length}|{candidate['tau']!r}"
        if key in progress["runs"]:
            continue
        began = time.perf_counter()
        sim = HybridSimulator(
            ensemble, solver, DoubtThreshold(candidate["tau"]), length, substeps, doubt_method=name, amp=amp
        )
        rollouts = [_rollout(sim, store, scene, window, floors, device) for scene in range(n_scenes)]
        progress["runs"][key] = {
            "aggregation": name,
            "handoff_frames": length,
            "tau": candidate["tau"],
            "percentile": candidate["percentile"],
            "git_commit": commit,
            **_set_result(rollouts, window.frames, n_resamples),
        }
        write_json_atomic(progress_path, progress)
        found = progress["runs"][key]
        log(
            f"run {number}/{len(plan)} {key}: score {found['set_score']:.6f}, "
            f"share {found['solver_share']:.3f}, {time.perf_counter() - began:.0f} s"
        )

    never = {
        "aggregation": None,
        "handoff_frames": None,
        "tau": None,
        "percentile": None,
        **progress["evidence"]["result"],
    }
    runs = [never, *(progress["runs"][f"{n}|{h}|{c['tau']!r}"] for n, h, c in plan)]
    chosen = choose_run(runs)
    return _without_non_finite(
        {
            "split": "val",
            **settings,
            "n_resamples": n_resamples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            **provenance,
            "run_commits": sorted({run["git_commit"] for run in runs if run["git_commit"]}),
            "bad_frame_threshold": float(np.percentile(errors, BAD_FRAME_PERCENTILE)),
            "calibration_edges": {
                name: _decile_edges(doubts[name]) for name in AGGREGATIONS
            },
            "aggregation": aggregation,
            "candidates": candidates,
            "max_solver_share": MAX_SOLVER_SHARE,
            "runs": runs,
            "choice": {
                "kind": "never" if chosen["tau"] is None else "threshold",
                "gain_on_validation": chosen["tau"] is not None,
                **{name: chosen[name] for name in ("aggregation", "handoff_frames", "tau")},
                **{name: chosen[name] for name in ("set_score", "solver_share", "forced_handoffs")},
                "never_set_score": never["set_score"],
            },
        }
    )
