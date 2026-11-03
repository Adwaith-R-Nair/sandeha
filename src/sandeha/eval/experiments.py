"""The experiments of docs/DESIGN.md sections 6.1 to 6.4, by the rules in docs/PREREGISTRATION.md.

Without ``final`` only ``val`` opens. Each finished rollout set is kept in a progress directory, so a
stopped run restarts where it was, and the experiments share the rollouts they have in common.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sandeha.data.dataset import Normalizer
from sandeha.data.store import SplitStore, open_split, write_json_atomic
from sandeha.ensemble import Ensemble
from sandeha.eval.metrics import (
    _as_ranked,
    bootstrap_ci,
    decile_edges,
    doubt_figures,
    evidence_measure,
    has_contact,
    join_summaries,
    rise_on_unfamiliar,
    side_of,
    usable_frames,
    window_summary,
)
from sandeha.eval.rollout import (
    CoarseSolverSim,
    IndependentSim,
    PersistenceSim,
    RolloutTrace,
    oracle_error_of,
    rollout,
    timed_reference,
)
from sandeha.handoff_selection import (
    AGGREGATIONS,
    BOOTSTRAP_SEED,
    Window,
    _without_non_finite,
    make_tiny_val,
)
from sandeha.hybrid import AlwaysAI, AlwaysSolver, DoubtThreshold, FixedSchedule, HybridSimulator, Oracle, RandomBudget
from sandeha.train.loop import ERROR_CAP

TEST_SETS = ("test_a", "test_b", "test_c")
MATCH_TOLERANCE = 0.01  # One percentage point of solver share.
ORACLE_CANDIDATES = 9
BISECTION_STEPS = 30
REQUIRED_PROVENANCE = ("git_commit", "data_manifest_sha256", "checkpoint_sha256", "config")
EXPERIMENTS = ("accuracy", "doubt", "handoff", "ablation")
ABLATION_NOTE = (
    "Each reduced training is one seed trained for less time than the ensemble's members, and the "
    "plan names no baseline with that shorter time, so its comparison with the full member isn't "
    "only the effect of what was removed."
)
SIZES_NOTE = (
    "Sizes come from the lowest-seed members. Doubt is a population spread, so its absolute scale "
    "differs by size, and only rankings are compared."
)
BUDGET_ZERO = "A budget of 0 means there is no hand-off to compare."


class MissingProvenance(ValueError):
    """A result lacks what traces it to its code, data and models."""


def _missing_provenance(result: dict) -> list[str]:
    return [name for name in REQUIRED_PROVENANCE if name not in result or result[name] in (None, "", [], {})]


def check_provenance(result: dict) -> None:
    missing = _missing_provenance(result)
    if missing:
        raise MissingProvenance(f"result has no provenance for: {', '.join(missing)}")


def with_provenance(result: dict, provenance: dict) -> dict:
    joined = {**result, **provenance}
    check_provenance(joined)
    return joined


def open_eval_sets(root: Path, final: bool) -> tuple[dict, object]:
    """The sets to measure, and validation, which fixes the no-ensemble signal's calibration edges."""
    if not final:
        store = open_split(root, "val")
        return {"val": store}, store
    return {name: open_split(root, name, final_eval=True) for name in TEST_SETS}, open_split(root, "val")


def make_tiny_sets(directory: Path) -> tuple[dict, Normalizer, dict]:
    """One tiny solved split under every set's name, so that a dry run takes each set's path.

    All of them are the same small files, with no real split opened. The ``test_c`` copy's buoyancies
    are overwritten to give it both sides, which its solver and models then simply use.
    """
    val, normalizer, config = make_tiny_val(Path(directory) / "val", n_scenes=4)
    stores = {"val": val}
    for name in TEST_SETS:
        stores[name] = SplitStore(val.directory, name)
    stores["test_c"]._statics = {**stores["test_c"]._statics, "buoyancy": np.array([0.5, 0.6, 4.0, 4.2])}
    return stores, normalizer, config


# Budgets, baseline schedules and the oracle's search, worked out before any rollout runs


def _scheduled_frames(policy, handoff_frames: int, window: Window) -> int:
    """How many window frames a policy's schedule gives the solver, with no forced hand-offs."""
    frame, solver = window.start + 1, 0
    while frame <= window.end:
        if policy.decide(frame, 0.0, {}):
            solver += min(handoff_frames, window.end - frame + 1)
            frame += handoff_frames
        else:
            frame += 1
    return solver


def fixed_share(period: int, handoff_frames: int, n_scenes: int, window: Window) -> float:
    start = window.start + 1
    solver = sum(
        _scheduled_frames(FixedSchedule.for_scene(period, i, n_scenes, start), handoff_frames, window)
        for i in range(n_scenes)
    )
    return solver / (window.frames * n_scenes)


def _nearest(items: list[dict], budget: float) -> dict:
    """The schedule nearest the budget, with ties going to the longer period."""
    return min(items, key=lambda item: (round(abs(item["scheduled_share"] - budget), 12), -item["period"]))


def choose_fixed_schedules(budget: float, handoff_frames: int, n_scenes: int, window: Window) -> list[dict]:
    """The nearest schedule or, if it's over a point away, the nearest on each side of the budget."""
    periods = range(handoff_frames, window.frames * n_scenes + handoff_frames + 1)
    items = [
        {"period": period, "scheduled_share": fixed_share(period, handoff_frames, n_scenes, window)}
        for period in periods
    ]
    nearest = _nearest(items, budget)
    if abs(nearest["scheduled_share"] - budget) <= MATCH_TOLERANCE:
        return [{**nearest, "role": "nearest"}]
    chosen = []
    for role, side in (
        ("below", [item for item in items if item["scheduled_share"] < budget]),
        ("above", [item for item in items if item["scheduled_share"] > budget]),
    ):
        if side:
            chosen.append({**_nearest(side, budget), "role": role})
    return chosen


def random_share(rate: float, handoff_frames: int, scenes, window: Window, seeds) -> float:
    solver = sum(
        _scheduled_frames(RandomBudget(rate, seed, scene), handoff_frames, window)
        for scene in scenes
        for seed in seeds
    )
    return solver / (window.frames * len(scenes) * len(seeds))


def find_random_rate(budget: float, handoff_frames: int, scenes, window: Window, seeds) -> float:
    """The rate whose scheduled share on these streams is nearest the budget, by bisection."""
    low, high = 0.0, 1.0
    best, best_gap = 0.0, abs(budget)
    for _ in range(BISECTION_STEPS):
        rate = (low + high) / 2
        share = random_share(rate, handoff_frames, scenes, window, seeds)
        gap = abs(share - budget)
        if gap < best_gap or (gap == best_gap and rate < best):
            best, best_gap = rate, gap
        if share > budget:
            high = rate
        else:
            low = rate
    return best


def oracle_search(budget: float, errors, run_share: Callable[[float], float]) -> dict:
    """The preregistered search for the oracle's ``tau``. A budget of 0 isn't run at all."""
    if budget == 0:
        return {"run": False, "statement": "A budget of 0 means the oracle isn't run."}
    tau = float(np.percentile(np.asarray(errors, dtype=np.float64), 100 - 100 * budget))
    low, high = 0.0, ERROR_CAP
    candidates = []
    for _ in range(ORACLE_CANDIDATES):
        share = run_share(tau)
        candidates.append({"tau": tau, "share": share})
        if abs(share - budget) <= MATCH_TOLERANCE:
            break
        if share > budget:
            low = tau
        else:
            high = tau
        tau = (low + high) / 2
    last = candidates[-1]
    if abs(last["share"] - budget) <= MATCH_TOLERANCE:
        kept = last
    else:
        kept = min(candidates, key=lambda c: (round(abs(c["share"] - budget), 12), c["share"], c["tau"]))
    return {
        "run": True,
        "candidates": candidates,
        "kept": kept,
        "unmatched": abs(kept["share"] - budget) > MATCH_TOLERANCE,
    }


# Runs and what is reported of them


def _finish_run(run: dict, n_resamples: int) -> dict:
    scores = run["scene_scores"]
    run["set_score"] = float(np.mean(scores))
    run["set_score_ci"] = list(bootstrap_ci(scores, n_resamples))
    run["achieved_share"] = run["solver_frames"] / run["total_frames"]
    run["blown_up_share"] = run["blown_up"] / run["n_rollouts"]
    run["ms_per_frame"] = run["ms"] / run["frames_made"]
    return run


def combine_runs(first: dict, second: dict, n_resamples: int = 2000) -> dict:
    """Two sides' runs as one, with the scenes in order, so that a paired difference still pairs."""
    joined = {"scene_scores": list(first["scene_scores"]) + list(second["scene_scores"])}
    for name in ("solver_frames", "total_frames", "forced_handoffs", "blown_up", "n_rollouts", "ms", "frames_made"):
        joined[name] = first[name] + second[name]
    if "secondary" in first and "secondary" in second:
        joined["secondary"] = {
            window: join_summaries(first["secondary"][window], second["secondary"][window])
            for window in first["secondary"]
        }
    return _finish_run(joined, n_resamples)


def compare_to_baseline(frozen_scores, budget: float, baseline: dict, n_resamples: int) -> dict:
    """The paired difference, doubt less baseline. Doubt beats only one matched or above the budget."""
    difference = np.asarray(frozen_scores) - np.asarray(baseline["scene_scores"])
    low, high = bootstrap_ci(difference, n_resamples)
    share = baseline["achieved_share"]
    if abs(share - budget) <= MATCH_TOLERANCE:
        status = "matched"
    else:
        status = "above" if share > budget else "below"
    return {
        "status": status,
        "difference": float(difference.mean()),
        "difference_ci": [low, high],
        "beats": bool(high < 0 and status != "below"),
    }


@dataclass
class Evaluation:
    """Everything an experiment reads. ``config`` is ``configs/eval.yaml``, or a dry run's own."""

    ensemble: Ensemble
    solver: object
    coarse_solver: object
    substeps: int
    amp: bool
    selection: dict
    config: dict
    provenance: dict
    progress_dir: Path
    ablation_models: dict = field(default_factory=dict)
    log: Callable = print

    def __post_init__(self):
        self.window = Window(self.config["start"], self.config["primary_end"])
        self.long_end = self.config["long_end"]
        self.n_resamples = self.config["n_resamples"]
        self.floors = [0.01 * std for std in self.ensemble.normalizer.field_std]
        choice = self.selection["choice"]
        self.tau = choice["tau"]
        self.handoff_frames = choice["handoff_frames"] or self.config["default_handoff_frames"]
        # The chosen aggregation. If **never** won, the one that led step 1 stands in for the figures.
        self.aggregation = choice["aggregation"] or self.selection["aggregation"]["leader"] or AGGREGATIONS[0]
        self._memory = {}
        fingerprint = {
            "config": self.config,
            "choice": choice,
            **{k: v for k, v in self.provenance.items() if k not in ("git_commit", "git_dirty", "config")},
        }
        self.progress_dir = Path(self.progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        path = self.progress_dir / "fingerprint.json"
        if path.exists() and json.loads(path.read_text()) != fingerprint:
            raise ValueError(f"{self.progress_dir} holds rollouts from other settings. Remove it to start again.")
        write_json_atomic(path, fingerprint)

    # Rollout sets

    def run_set(self, store, key: str, make_sim, end: int, scenes=None, scores_by=()) -> list[RolloutTrace]:
        """The set's traces, in the order of ``scenes``, computed once. ``make_sim(position, scene)``."""
        scenes = list(range(store.n_scenes)) if scenes is None else list(scenes)
        memory_key = (store.split, key)
        if memory_key in self._memory:
            return self._memory[memory_key]
        path = self.progress_dir / store.split / f"{key}.json"
        if path.exists():
            traces = [RolloutTrace(**item) for item in json.loads(path.read_text())]
        else:
            traces = [
                rollout(
                    make_sim(position, scene),
                    scene,
                    store,
                    self.window.start,
                    end - self.window.start,
                    floors=self.floors,
                    scores_by=scores_by,
                )
                for position, scene in enumerate(scenes)
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, [trace.to_dict() for trace in traces])
            ms = sum(trace.ms for trace in traces)
            self.log(f"{store.split}/{key}: {len(traces)} rollouts, {ms / 1000:.1f} s stepping")
        self._memory[memory_key] = traces
        return traces

    def hybrid(self, policy, handoff_frames: int = 1, method: str = "mean", ensemble=None, error_of=None):
        return HybridSimulator(
            ensemble or self.ensemble,
            self.solver,
            policy,
            handoff_frames,
            self.substeps,
            doubt_method=method,
            amp=self.amp,
            error_of=error_of,
        )

    def single_member(self) -> Ensemble:
        return Ensemble(self.ensemble.models[:1], self.ensemble.normalizer)

    def evidence(self, store) -> list[RolloutTrace]:
        return self.run_set(
            store, "ensemble", lambda *_: self.hybrid(AlwaysAI()), self.long_end, scores_by=AGGREGATIONS
        )

    def single(self, store) -> list[RolloutTrace]:
        return self.run_set(
            store, "single", lambda *_: self.hybrid(AlwaysAI(), ensemble=self.single_member()), self.long_end
        )

    def independent(self, store) -> list[RolloutTrace]:
        return self.run_set(
            store,
            "independent",
            lambda *_: IndependentSim(self.ensemble, self.solver, self.substeps, self.amp),
            self.long_end,
            scores_by=AGGREGATIONS,
        )

    # What is reported

    def summary(self, traces: list[RolloutTrace]) -> dict:
        """A run's primary figures over the window. A blown-up frame stays at the cap, and none is left out."""
        frames = self.window.frames
        primary = [trace.errors[:frames] for trace in traces]
        return _finish_run(
            {
                "scene_scores": [float(np.mean(errors)) for errors in primary],
                "solver_frames": sum(sum(trace.solver_flags[:frames]) for trace in traces),
                "total_frames": frames * len(traces),
                "forced_handoffs": sum(trace.forced for trace in traces),
                "blown_up": sum(max(errors) >= ERROR_CAP for errors in primary),
                "n_rollouts": len(traces),
                "ms": sum(trace.ms for trace in traces),
                "frames_made": sum(trace.frames_made for trace in traces),
            },
            self.n_resamples,
        )

    def secondary(self, traces: list[RolloutTrace]) -> dict:
        """Secondary metrics by window. A window longer than a run's rollouts covers what it has."""
        frames = self.window.frames
        return {
            "primary_window": window_summary([frame for t in traces for frame in t.secondary[:frames]]),
            "long_window": window_summary([frame for t in traces for frame in t.secondary]),
        }

    def report(self, traces: list[RolloutTrace]) -> dict:
        return {**self.summary(traces), "secondary": self.secondary(traces)}


def groups_of(store) -> dict[str, list[int]]:
    """The scenes to report apart. ``test_c`` has a side per scene, and a pooled figure is built on top."""
    if store.split != "test_c":
        return {"all": list(range(store.n_scenes))}
    sides = {"low": [], "high": []}
    for scene in range(store.n_scenes):
        sides[side_of(store.buoyancy(scene))].append(scene)
    return {name: scenes for name, scenes in sides.items() if scenes}


def _by_group(store, measure: Callable[[list[int], str], dict]) -> dict:
    groups = groups_of(store)
    found = {name: measure(scenes, name) for name, scenes in groups.items()}
    if store.split == "test_c":
        found["pooled"] = measure([s for scenes in groups.values() for s in scenes], "pooled")
    return found


def _picked(traces: list, scenes: list[int]) -> list:
    return [traces[scene] for scene in scenes]


# Accuracy


def _standard_runs(ev: Evaluation, store) -> dict:
    return {
        "ensemble": ev.evidence(store),
        "single_model": ev.single(store),
        "persistence": ev.run_set(store, "persistence", lambda *_: PersistenceSim(ev.solver), ev.long_end),
        "coarse_solver": ev.run_set(
            store, "coarse_solver", lambda *_: CoarseSolverSim(ev.solver, ev.coarse_solver, ev.substeps), ev.long_end
        ),
        # One long hand-off, so that the solver keeps its own fine state all the way.
        "solver_only": ev.run_set(
            store,
            "solver_only",
            lambda *_: ev.hybrid(AlwaysSolver(), ev.long_end - ev.window.start),
            ev.long_end,
        ),
    }


def _accuracy_of(ev: Evaluation, traces: list[RolloutTrace]) -> dict:
    errors = np.array([trace.errors for trace in traces], dtype=np.float64)
    means = np.cumsum(errors, axis=1) / np.arange(1, errors.shape[1] + 1)
    lengths = [length for length in ev.config["rollout_lengths"] if length <= errors.shape[1]]
    usable = [usable_frames(trace.errors, ev.config["usable_error"]) for trace in traces]
    return {
        **ev.report(traces),
        "error_by_time": errors.mean(axis=0).tolist(),
        "mean_error_by_length": means.mean(axis=0).tolist(),
        "by_length": {
            str(length): {
                "mean_error": float(means[:, length - 1].mean()),
                "ci95": list(bootstrap_ci(means[:, length - 1], ev.n_resamples)),
            }
            for length in lengths
        },
        "usable_frames": {"mean": float(np.mean(usable)), "median": float(np.median(usable)), "limit": ev.config["usable_error"]},
    }


def _late_counts(store, scenes, window: Window) -> dict:
    """Scenes whose late obstacle changes the solid inside the window, after it, or never."""
    late = [store.scene(scene).late_frame for scene in scenes]
    inside = sum(frame is not None and frame < window.end for frame in late)
    after = sum(frame is not None and frame >= window.end for frame in late)
    return {"inside_window": inside, "after_window": after, "none": len(late) - inside - after}


def _no_contact(store, scenes, window: Window) -> dict:
    missing = 0
    for scene in scenes:
        density = np.array(store.fields[scene, : window.end + 1, 2], dtype=np.float32)
        masks = np.stack([store.obstacle(scene, frame) for frame in range(window.end + 1)])
        missing += not has_contact(density, masks)
    return {"count": missing, "of": len(scenes), "rule": "coarse rule on the stored reference, from the preregistration"}


def accuracy_experiment(ev: Evaluation, store) -> dict:
    runs = _standard_runs(ev, store)

    def measure(scenes, _group):
        found = {
            "scenes": len(scenes),
            "late_obstacles": _late_counts(store, scenes, ev.window),
            "policies": {name: _accuracy_of(ev, _picked(traces, scenes)) for name, traces in runs.items()},
        }
        if store.split == "test_c":
            found["no_contact_scenes"] = _no_contact(store, scenes, ev.window)
        return found

    timing = list(range(min(ev.config["timing_scenes"], store.n_scenes)))
    return {
        "groups": _by_group(store, measure),
        "reference_ms_per_frame": timed_reference(
            ev.solver, store, timing, ev.window.start, ev.config["timing_frames"], ev.substeps
        ),
    }


# Doubt


def _matrix(rows, frames: int) -> np.ndarray:
    return np.array([list(row)[:frames] for row in rows], dtype=np.float64)


def _no_ensemble_signal(traces: list[RolloutTrace], frames: int) -> np.ndarray:
    return _matrix(([frame["divergence_rms"] for frame in trace.secondary] for trace in traces), frames)


def _no_ensemble_edges(ev: Evaluation, reference) -> list[float]:
    """Validation's deciles of the no-ensemble signal, frozen for it as doubt's are."""
    return decile_edges(_no_ensemble_signal(ev.single(reference), ev.window.frames))


def doubt_experiment(ev: Evaluation, store, reference) -> dict:
    evidence, independent, single = ev.evidence(store), ev.independent(store), ev.single(store)
    edges = ev.selection["calibration_edges"]
    no_ensemble_edges = _no_ensemble_edges(ev, reference)
    threshold = ev.selection["bad_frame_threshold"]
    windows = {"primary_window": ev.window.frames, "long_window": ev.long_end - ev.window.start}

    def signal(traces, name, frames):
        return _matrix((trace.scores[name] for trace in traces), frames)

    def errors_of(traces, frames):
        return _matrix((trace.errors for trace in traces), frames)

    def measure(scenes, _group):
        found = {"scenes": len(scenes), "aggregation": ev.aggregation}
        chosen, separate, one = _picked(evidence, scenes), _picked(independent, scenes), _picked(single, scenes)
        for window, frames in windows.items():
            errors = errors_of(chosen, frames)
            found[window] = {
                "aggregations": {
                    name: doubt_figures(signal(chosen, name, frames), errors, edges[name], threshold, ev.n_resamples)
                    for name in AGGREGATIONS
                },
                "no_ensemble": doubt_figures(
                    _no_ensemble_signal(one, frames), errors_of(one, frames), no_ensemble_edges, threshold, ev.n_resamples
                ),
                "independent_secondary": doubt_figures(
                    signal(separate, ev.aggregation, frames),
                    errors_of(separate, frames),
                    edges[ev.aggregation],
                    threshold,
                    ev.n_resamples,
                ),
            }
        ranked = _as_ranked(signal(chosen, ev.aggregation, windows["long_window"]))
        found["median_doubt_by_time"] = np.median(ranked, axis=0).tolist()
        return found

    return {"groups": _by_group(store, measure)}


def rise_experiment(ev: Evaluation, stores: dict) -> dict:
    """Doubt on Tests B and C against Test A. There's nothing to compare without Test A."""
    if "test_a" not in stores:
        return {"statement": "Test A wasn't opened, so doubt on unfamiliar sets isn't compared."}
    frames = ev.window.frames

    def doubt(store, scenes=None):
        traces = ev.evidence(store)
        traces = traces if scenes is None else _picked(traces, scenes)
        return _matrix((trace.scores[ev.aggregation] for trace in traces), frames)

    familiar = doubt(stores["test_a"])
    found = {}
    if "test_b" in stores:
        found["test_b"] = rise_on_unfamiliar(doubt(stores["test_b"]), familiar, ev.n_resamples, BOOTSTRAP_SEED)
    if "test_c" in stores:
        found["test_c"] = {
            name: rise_on_unfamiliar(doubt(stores["test_c"], scenes), familiar, ev.n_resamples, BOOTSTRAP_SEED)
            for name, scenes in _by_group(stores["test_c"], lambda scenes, _: scenes).items()
        }
    return found


# Hand-off


def _frozen_traces(ev: Evaluation, store) -> list[RolloutTrace]:
    if ev.tau is None:
        return ev.evidence(store)
    policy = lambda *_: ev.hybrid(DoubtThreshold(ev.tau), ev.handoff_frames, ev.aggregation)  # noqa: E731
    return ev.run_set(store, "frozen", policy, ev.long_end)


def _fixed_runs(ev, store, group, scenes, frozen, budget) -> list[dict]:
    out = []
    start = ev.window.start + 1
    for schedule in choose_fixed_schedules(budget, ev.handoff_frames, len(scenes), ev.window):
        period = schedule["period"]
        traces = ev.run_set(
            store,
            f"fixed_{group}_p{period}",
            lambda position, _scene, period=period: ev.hybrid(
                FixedSchedule.for_scene(period, position, len(scenes), start), ev.handoff_frames, ev.aggregation
            ),
            ev.long_end,
            scenes,
        )
        run = {**schedule, **ev.report(traces)}
        out.append({**run, **compare_to_baseline(frozen["scene_scores"], budget, run, ev.n_resamples)})
    return out


def _random_runs(ev, store, group, scenes, frozen, budget) -> list[dict]:
    seeds = ev.config["random_seeds"]
    rate = find_random_rate(budget, ev.handoff_frames, scenes, ev.window, seeds)
    by_seed = [
        ev.run_set(
            store,
            f"random_{group}_s{seed}_r{rate:.10f}",
            lambda _position, scene, seed=seed: ev.hybrid(RandomBudget(rate, seed, scene), ev.handoff_frames, ev.aggregation),
            ev.long_end,
            scenes,
        )
        for seed in seeds
    ]
    runs = [ev.summary(traces) for traces in by_seed]
    joined = _finish_run(
        {
            "scene_scores": np.mean([run["scene_scores"] for run in runs], axis=0).tolist(),
            "solver_frames": sum(run["solver_frames"] for run in runs),
            "total_frames": sum(run["total_frames"] for run in runs),
            "forced_handoffs": sum(run["forced_handoffs"] for run in runs),
            "blown_up": sum(run["blown_up"] for run in runs),
            "n_rollouts": sum(run["n_rollouts"] for run in runs),
            "ms": sum(run["ms"] for run in runs),
            "frames_made": sum(run["frames_made"] for run in runs),
        },
        ev.n_resamples,
    )
    run = {
        "rate": rate,
        "role": "nearest",
        "seeds": list(seeds),
        "scheduled_share": random_share(rate, ev.handoff_frames, scenes, ev.window, seeds),
        **joined,
        "secondary": ev.secondary([trace for traces in by_seed for trace in traces]),
    }
    return [{**run, **compare_to_baseline(frozen["scene_scores"], budget, run, ev.n_resamples)}]


def _oracle_run(ev, store, group, scenes, frozen, budget, evidence) -> dict:
    def make(tau):
        return lambda _position, scene: ev.hybrid(
            Oracle(tau), ev.handoff_frames, ev.aggregation, error_of=oracle_error_of(store, scene, ev.floors)
        )

    def run_share(tau):
        traces = ev.run_set(store, f"oracle_{group}_search_{tau:.12g}", make(tau), ev.window.end, scenes)
        return ev.summary(traces)["achieved_share"]

    errors = _matrix((trace.errors for trace in _picked(evidence, scenes)), ev.window.frames)
    found = oracle_search(budget, errors, run_share)
    if not found["run"]:
        return found
    tau = found["kept"]["tau"]
    run = ev.report(ev.run_set(store, f"oracle_{group}_{tau:.12g}", make(tau), ev.long_end, scenes))
    difference = np.asarray(frozen["scene_scores"]) - np.asarray(run["scene_scores"])
    return {
        **found,
        **run,
        "descriptive": True,
        "difference_from_doubt": float(difference.mean()),
        "difference_from_doubt_ci": list(bootstrap_ci(difference, ev.n_resamples)),
    }


def _handoff_side(ev, store, group, scenes, frozen_traces, evidence) -> dict:
    picked = _picked(frozen_traces, scenes)
    frozen = ev.report(picked)
    budget = frozen["achieved_share"]
    report = {"budget": budget, "frozen": frozen, "handoff_frames": ev.handoff_frames, "scenes": len(scenes)}
    if budget == 0:
        return {
            **report,
            "statement": BUDGET_ZERO,
            "fixed": [],
            "random": [],
            "oracle": oracle_search(0.0, [], None),
        }
    report["fixed"] = _fixed_runs(ev, store, group, scenes, frozen, budget)
    report["random"] = _random_runs(ev, store, group, scenes, frozen, budget)
    report["oracle"] = _oracle_run(ev, store, group, scenes, frozen, budget, evidence)
    return report


def _pooled_baselines(ev, low, high, kind: str, frozen, budget) -> list[dict]:
    """Joins each side's runs by role. A side with a budget of 0 has no hand-off to schedule."""
    roles = sorted({run["role"] for side in (low, high) for run in side[kind]})

    def pick(side, role):
        if not side[kind]:
            return side["frozen"]
        by_role = {run["role"]: run for run in side[kind]}
        return by_role.get(role) or by_role.get("nearest") or side[kind][0]

    out = []
    for role in roles:
        run = combine_runs(pick(low, role), pick(high, role), ev.n_resamples)
        out.append({"role": role, **run, **compare_to_baseline(frozen["scene_scores"], budget, run, ev.n_resamples)})
    return out


def _pooled_handoff(ev, low, high) -> dict:
    frozen = combine_runs(low["frozen"], high["frozen"], ev.n_resamples)
    budget = frozen["achieved_share"]
    report = {"budget": budget, "frozen": frozen, "handoff_frames": ev.handoff_frames, "scenes": frozen["n_rollouts"]}
    if budget == 0:
        return {**report, "statement": BUDGET_ZERO, "fixed": [], "random": [], "oracle": oracle_search(0.0, [], None)}
    report["fixed"] = _pooled_baselines(ev, low, high, "fixed", frozen, budget)
    report["random"] = _pooled_baselines(ev, low, high, "random", frozen, budget)
    sides = [side["oracle"] if side["oracle"]["run"] else side["frozen"] for side in (low, high)]
    joined = combine_runs(*sides, ev.n_resamples)
    difference = np.asarray(frozen["scene_scores"]) - np.asarray(joined["scene_scores"])
    report["oracle"] = {
        "run": True,
        **joined,
        "descriptive": True,
        "difference_from_doubt": float(difference.mean()),
        "difference_from_doubt_ci": list(bootstrap_ci(difference, ev.n_resamples)),
    }
    return report


def _sweep(ev: Evaluation, store) -> list[dict]:
    """Descriptive curves on every policy. They choose nothing."""
    start = ev.window.start + 1
    h = ev.handoff_frames
    n = store.n_scenes
    points = []

    def point(policy, value, traces):
        points.append(
            {
                "policy": policy,
                "value": value,
                "descriptive": True,
                "groups": _by_group(
                    store, lambda scenes, _: _without_scene_scores(ev.summary(_picked(traces, scenes)))
                ),
            }
        )

    runs = _standard_runs(ev, store)
    for name in ("ensemble", "single_model", "solver_only"):
        point(name, None, runs[name])
    for candidate in ev.selection["candidates"].get(ev.aggregation, []):
        tau = candidate["tau"]
        point(
            "doubt",
            tau,
            ev.run_set(
                store,
                f"sweep_doubt_{tau:.12g}",
                lambda *_, tau=tau: ev.hybrid(DoubtThreshold(tau), h, ev.aggregation),
                ev.window.end,
            ),
        )
    for period in ev.config["sweep"]["periods"]:
        if period >= h:
            point(
                "fixed",
                period,
                ev.run_set(
                    store,
                    f"sweep_fixed_{period}",
                    lambda position, _scene, period=period: ev.hybrid(
                        FixedSchedule.for_scene(period, position, n, start), h, ev.aggregation
                    ),
                    ev.window.end,
                ),
            )
    seed = ev.config["random_seeds"][0]
    for rate in ev.config["sweep"]["rates"]:
        point(
            "random",
            rate,
            ev.run_set(
                store,
                f"sweep_random_{rate:g}",
                lambda _position, scene, rate=rate: ev.hybrid(RandomBudget(rate, seed, scene), h, ev.aggregation),
                ev.window.end,
            ),
        )
    return points


def _without_scene_scores(run: dict) -> dict:
    return {key: value for key, value in run.items() if key != "scene_scores"}


def handoff_experiment(ev: Evaluation, store) -> dict:
    frozen = _frozen_traces(ev, store)
    evidence = ev.evidence(store)
    groups = {
        name: _handoff_side(ev, store, name, scenes, frozen, evidence)
        for name, scenes in groups_of(store).items()
    }
    if store.split == "test_c":
        if len(groups) == 2:
            groups["pooled"] = _pooled_handoff(ev, groups["low"], groups["high"])
        else:  # All scenes on one side: its run is the pooled one.
            groups["pooled"] = next(iter(groups.values()))
    return {
        "choice": ev.selection["choice"],
        "groups": groups,
        "sweep": _sweep(ev, store),
    }


# Ablations


def ablation_experiment(ev: Evaluation, store) -> dict:
    full = ev.single(store)
    trained = {
        name: ev.run_set(
            store,
            f"ablation_{name}",
            lambda *_, model=model: ev.hybrid(AlwaysAI(), ensemble=Ensemble([model], ev.ensemble.normalizer)),
            ev.window.end,
        )
        for name, model in ev.ablation_models.items()
    }
    sizes = {}
    for size in ev.config["ensemble_sizes"]:
        if size > len(ev.ensemble.models):
            continue
        members = Ensemble(ev.ensemble.models[:size], ev.ensemble.normalizer)
        sizes[size] = (
            ev.evidence(store)
            if size == len(ev.ensemble.models)
            else ev.run_set(
                store,
                f"ablation_size_{size}",
                lambda *_, members=members: ev.hybrid(AlwaysAI(), ensemble=members),
                ev.window.end,
                scores_by=AGGREGATIONS,
            )
        )

    def measure(scenes, _group):
        base = ev.summary(_picked(full, scenes))
        reduced = {}
        for name, traces in trained.items():
            run = ev.summary(_picked(traces, scenes))
            difference = np.asarray(run["scene_scores"]) - np.asarray(base["scene_scores"])
            reduced[name] = {
                **run,
                "difference_from_full_member": float(difference.mean()),
                "difference_ci": list(bootstrap_ci(difference, ev.n_resamples)),
            }
        by_size = {}
        for size, traces in sizes.items():
            picked = _picked(traces, scenes)
            frames = ev.window.frames
            by_size[str(size)] = {
                **ev.summary(picked),
                "evidence": evidence_measure(
                    _matrix((t.scores[ev.aggregation] for t in picked), frames),
                    _matrix((t.errors for t in picked), frames),
                    ev.n_resamples,
                    BOOTSTRAP_SEED,
                ),
            }
        return {"full_member": base, "reduced_trainings": reduced, "ensemble_sizes": by_size}

    return {"groups": _by_group(store, measure)}


# Running them


def run_experiments(ev: Evaluation, stores: dict, reference, names, out: Path, final: bool) -> dict[str, Path]:
    """Writes one result file per experiment, each carrying its provenance."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    measures = {
        "accuracy": lambda store: accuracy_experiment(ev, store),
        "doubt": lambda store: doubt_experiment(ev, store, reference),
        "handoff": lambda store: handoff_experiment(ev, store),
        "ablation": lambda store: ablation_experiment(ev, store),
    }
    written = {}
    for name in names:
        found = {"experiment": name, "final": final, "sets": {split: measures[name](store) for split, store in stores.items()}}
        if name == "doubt":
            found["rise_on_unfamiliar"] = rise_experiment(ev, stores)
        if name == "ablation":
            found["notes"] = [ABLATION_NOTE, SIZES_NOTE]
            found["ablations_run"] = sorted(ev.ablation_models)
        result = with_provenance(found, {"selection_choice": ev.selection["choice"], **ev.provenance})
        written[name] = out / f"exp_{name}.json"
        write_json_atomic(written[name], _without_non_finite(result))
        ev.log(f"wrote {written[name]}")
    return written
