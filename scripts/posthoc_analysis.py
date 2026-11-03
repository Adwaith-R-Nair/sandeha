"""POST-HOC, exploratory analysis of why doubt-based hand-off did not beat the baselines.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/posthoc_analysis.py

None of this was preregistered, and none of it changes a headline result. It reads saved rollouts and
result files only: no model is loaded, no solver runs and no test split is opened. Its figures explain
results on the same test sets they were seen on, so they suggest reasons and don't confirm them.

Gain means the no-hand-off ensemble's error less the policy's error, so a positive gain is better.
Intervals are 95% percentile intervals from 2000 resamples of whole scenes, seed 20261103.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_RESAMPLES = 2000
SEED = 20261103
WINDOW = 50  # Frames 21 to 70 are indices 0 to 49 of a trace.
HORIZON = 100  # Frames 21 to 120.
HANDOFF_FRAMES = 2
TIME_EDGES = (0, 10, 20, 30, 40, 50)
TIME_LABELS = ("21-30", "31-40", "41-50", "51-60", "61-70")
EVENT_EDGES = (0, 5, 10, 20, 30, 40, 50)
EVENT_LABELS = ("21-25", "26-30", "31-40", "41-50", "51-60", "61-70")
EVENT_LAGS = (0, 1, 2, 5, 10, 20, 30)
RANDOM_RATE_LIMIT = 0.2  # Above this, hand-offs fill nearly every frame and timing can't vary.
PREDICTION_SHARE_LIMIT = 0.4  # A rate of 0.2 with 2-frame hand-offs is about this share.
GROUPS = (("test_a", "all"), ("test_b", "all"), ("test_c", "low"), ("test_c", "high"))
INPUT_FILES = (
    "exp_handoff.json",
    "exp_doubt.json",
    "exp_accuracy.json",
    "exp_ablation.json",
    "handoff_selection.json",
    "data_manifest.json",
)

BLUE, ORANGE, GREEN, AMBER, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


@dataclass
class Runs:
    """One rollout set, as arrays with one row per scene of the group."""

    errors: np.ndarray
    flags: np.ndarray
    ms: np.ndarray
    frames_made: np.ndarray

    @property
    def score(self) -> np.ndarray:
        return self.errors[:, :WINDOW].mean(axis=1)

    @property
    def window_share(self) -> float:
        return float(self.flags[:, :WINDOW].mean())

    @property
    def full_share(self) -> float:
        return float(self.flags.mean())

    @property
    def ms_per_frame(self) -> float:
        return float(self.ms.sum() / self.frames_made.sum())


def resamples(n_scenes: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n_scenes, (N_RESAMPLES, n_scenes))


def mean_ci(values) -> dict:
    """The mean of per-scene values and its interval. Equal lengths share resamples, so it's paired."""
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values[resamples(len(values))].mean(axis=1), [2.5, 97.5])
    return {"mean": float(values.mean()), "ci": [float(low), float(high)], "scenes": len(values)}


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis, ties sharing their mean."""
    order = np.sort(values, axis=-1)
    if values.ndim == 1:
        return (np.searchsorted(order, values, "left") + np.searchsorted(order, values, "right") + 1) / 2
    return np.stack([ranks(row) for row in values])


def spearman(a, b) -> float:
    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_ci(a, b) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    found = [spearman(a[pick], b[pick]) for pick in resamples(len(a))]
    low, high = np.nanpercentile(found, [2.5, 97.5])
    return {"spearman": spearman(a, b), "ci": [float(low), float(high)], "scenes": len(a)}


def within_time_spearman(a: np.ndarray, b: np.ndarray) -> dict:
    """The mean over columns of the Spearman across rows, with whole rows resampled."""

    def measure(x, y):
        rx, ry = ranks(x.T), ranks(y.T)
        rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
        bottom = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
        return float(np.mean((rx * ry).sum(axis=1)[bottom > 0] / bottom[bottom > 0]))

    found = [measure(a[pick], b[pick]) for pick in resamples(len(a))]
    low, high = np.percentile(found, [2.5, 97.5])
    return {"mean_over_times": measure(a, b), "ci": [float(low), float(high)], "scenes": len(a)}


def handoff_starts(flags) -> list[int]:
    """Indices where a hand-off begins. Two contiguous hand-offs show as a run, split every 2 frames."""
    starts, i = [], 0
    while i < len(flags):
        if flags[i]:
            starts.append(i)
            i += HANDOFF_FRAMES
        else:
            i += 1
    return starts


def load_traces(path: Path) -> list[dict]:
    keep = ("errors", "solver_flags", "ms", "frames_made", "scores")
    traces = json.loads(path.read_text())["traces"]
    for trace in traces:
        divergence = [frame["divergence_rms"] for frame in trace["secondary"]]
        for key in list(trace):
            if key not in keep:
                del trace[key]
        trace["divergence_rms"] = divergence
    return traces


def to_runs(traces: list[dict]) -> Runs:
    return Runs(
        np.array([t["errors"] for t in traces], dtype=np.float64),
        np.array([t["solver_flags"] for t in traces], dtype=bool),
        np.array([t["ms"] for t in traces], dtype=np.float64),
        np.array([t["frames_made"] for t in traces], dtype=np.float64),
    )


class Store:
    """Every saved rollout file of a test set, picked by group."""

    def __init__(self, root: Path, split: str):
        self.split = split
        self.traces = {
            path.stem: load_traces(path)
            for path in sorted((root / split).glob("*.json"))
            if "search" not in path.stem and not path.stem.startswith("ablation")
        }
        self.n = len(self.traces["ensemble"])
        frozen = to_runs(self.traces["frozen"])
        if split == "test_c":
            # A scene's side is in the split's statics, which stay closed here. The frozen policy used
            # no solver on the low side and some on every high scene, and main() checks this against
            # the side's saved score and the manifest's side counts.
            used = frozen.flags.any(axis=1)
            self.scenes = {"low": np.flatnonzero(~used), "high": np.flatnonzero(used)}
        else:
            self.scenes = {"all": np.arange(self.n)}

    def names(self, *prefixes: str) -> list[str]:
        return [name for name in self.traces if name.startswith(prefixes)]

    def has(self, name: str, group: str) -> bool:
        return len(self.traces[name]) in (self.n, len(self.scenes[group])) and (
            len(self.traces[name]) == self.n or f"_{group}_" in name
        )

    def runs(self, name: str, group: str) -> Runs:
        traces = self.traces[name]
        if len(traces) != self.n:  # A run on one side alone holds that side's scenes, in order.
            return to_runs(traces)
        return to_runs([traces[i] for i in self.scenes[group]])

    def p95(self, group: str) -> np.ndarray:
        return np.array([self.traces["ensemble"][i]["scores"]["p95"] for i in self.scenes[group]])

    def signal(self, group: str) -> np.ndarray:
        return np.array([self.traces["single"][i]["divergence_rms"] for i in self.scenes[group]])

    def policy_names(self, group: str) -> dict[str, list[str]]:
        return {
            "doubt": ["frozen"],
            "fixed": self.names(f"fixed_{group}_"),
            "random": self.names(f"random_{group}_"),
            "oracle": self.names(f"oracle_{group}_"),
        }

    def exogenous_names(self, group: str) -> list[str]:
        """Runs whose hand-off times were set with no knowledge of the flow."""
        names = self.names(f"fixed_{group}_", f"random_{group}_", "sweep_fixed", "sweep_random")
        return [name for name in names if self.has(name, group)]

    def fixed_curve_names(self, group: str) -> list[str]:
        return [name for name in self.names(f"fixed_{group}_", "sweep_fixed") if self.has(name, group)]

    def random_regression_names(self, group: str) -> list[str]:
        """Random runs at a rate low enough that hand-off times can still differ."""
        names = []
        for name in self.names(f"random_{group}_", "sweep_random"):
            rate = float(name.rsplit("_", 1)[1].lstrip("r"))
            if rate <= RANDOM_RATE_LIMIT and self.has(name, group):
                names.append(name)
        return names


def check_inputs(stores: dict, handoff: dict, accuracy: dict, manifest: dict) -> dict:
    """Stop if the saved rollouts don't give back the reported figures, or the counterfactual fails."""
    tau = handoff["selection_choice"]["tau"]
    largest, mismatched, compared = 0.0, 0, 0
    for split, group in GROUPS:
        store = stores[split]
        report = handoff["sets"][split]["groups"][group]
        ensemble = store.runs("ensemble", group)
        frozen = store.runs("frozen", group)
        assert np.isclose(frozen.score.mean(), report["frozen"]["set_score"], atol=1e-9), (split, group)
        assert np.isclose(frozen.window_share, report["budget"], atol=1e-9), (split, group)
        score = accuracy["sets"][split]["groups"][group]["policies"]["ensemble"]["set_score"]
        assert np.isclose(ensemble.score.mean(), score, atol=1e-9), (split, group)
        if split == "test_c":
            assert len(store.scenes[group]) == manifest["splits"]["test_c"]["sides"][group]
        names = store.policy_names(group)
        for entry in report["fixed"]:
            found = store.runs(f"fixed_{group}_p{entry['period']}", group)
            assert np.isclose(found.score.mean(), entry["set_score"], atol=1e-9), (split, group)
        if report["random"]:
            mean = np.mean([store.runs(name, group).score for name in names["random"]], axis=0).mean()
            assert np.isclose(mean, report["random"][0]["set_score"], atol=1e-9), (split, group)
        if report["oracle"].get("run"):
            found = store.runs(names["oracle"][0], group)
            assert np.isclose(found.score.mean(), report["oracle"]["set_score"], atol=1e-9), (split, group)
        p95 = store.p95(group)
        for name in store.exogenous_names(group) + names["oracle"] + ["frozen"]:
            runs = store.runs(name, group)
            for row in range(len(runs.errors)):
                on = np.flatnonzero(runs.flags[row])
                first = int(on[0]) if len(on) else runs.errors.shape[1]
                if first:
                    largest = max(largest, float(np.abs(runs.errors[row, :first] - ensemble.errors[row, :first]).max()))
                compared += 1
        for row in range(len(p95)):
            over = np.flatnonzero(p95[row] > tau)
            on = np.flatnonzero(frozen.flags[row])
            mismatched += (int(over[0]) if len(over) else -1) != (int(on[0]) if len(on) else -1)
    assert largest == 0.0, "a hand-off rollout left the ensemble's path before its first hand-off"
    assert mismatched == 0, "the frozen policy's first hand-off isn't the first frame over tau"
    return {
        "saved_scores_reproduce_reported_scores": True,
        "rollouts_compared": compared,
        "largest_error_difference_before_first_handoff": largest,
        "scenes_where_first_handoff_is_not_first_frame_over_tau": mismatched,
    }


def timing_of(runs: Runs) -> dict:
    window = runs.flags[:, :WINDOW]
    starts = [handoff_starts(row) for row in window]
    every = np.array([s for row in starts for s in row], dtype=np.float64)
    gaps = np.array([g for row in starts for g in np.diff(row)], dtype=np.float64)
    firsts = np.array([row[0] for row in starts if row], dtype=np.float64)
    per_scene = window.sum(axis=1)
    top = max(1, round(len(per_scene) / 6))
    return {
        "window_share": runs.window_share,
        "share_frames_21_to_120": runs.full_share,
        "solver_frames_frames_21_to_70": int(window.sum()),
        "solver_frames_frames_71_to_120": int(runs.flags[:, WINDOW:].sum()),
        "solver_frames_by_bin": dict(
            zip(TIME_LABELS, [int(window[:, a:b].sum()) for a, b in zip(TIME_EDGES[:-1], TIME_EDGES[1:], strict=True)], strict=True)
        ),
        "mean_start_frame": float(every.mean() + 21) if len(every) else None,
        "median_first_handoff_frame": float(np.median(firsts) + 21) if len(firsts) else None,
        "scenes": len(per_scene),
        "scenes_with_no_handoff_in_window": int((per_scene == 0).sum()),
        "largest_sixth_of_scenes_share_of_solver_frames": float(np.sort(per_scene)[-top:].sum() / max(1, per_scene.sum())),
        "handoffs": len(every),
        "share_of_gaps_immediately_following": float((gaps == HANDOFF_FRAMES).mean()) if len(gaps) else None,
        "median_gap": float(np.median(gaps)) if len(gaps) else None,
        "ms_per_frame": runs.ms_per_frame,
    }


def timing_section(store: Store, group: str) -> dict:
    out = {}
    for policy, names in store.policy_names(group).items():
        for name in names:
            label = policy if policy in ("doubt", "oracle") else name
            out[label] = timing_of(store.runs(name, group))
    return out


def first_handoff_events(store: Store, group: str, names: list[str]) -> dict:
    """Each distinct (scene, first hand-off frame), with its error against the ensemble's.

    Up to its first hand-off a rollout is exactly the ensemble's rollout (checked in check_inputs), so
    the ensemble's error is the exact counterfactual until the next hand-off. Two runs that first hand
    off at the same frame of a scene are one experiment, so they count once, with the longest clean span.
    """
    ensemble = store.runs("ensemble", group).errors
    events: dict[tuple[int, int], np.ndarray] = {}
    for name in names:
        runs = store.runs(name, group)
        for row in range(len(runs.errors)):
            starts = handoff_starts(runs.flags[row])
            if not starts:
                continue
            first = starts[0]
            end = starts[1] if len(starts) > 1 else runs.errors.shape[1]
            delta = runs.errors[row, first:end] - ensemble[row, first:end]
            if (row, first) not in events or len(delta) > len(events[(row, first)]):
                events[(row, first)] = delta
    return events


def event_table(events: dict, n_scenes: int) -> dict:
    table = {}
    picks = resamples(n_scenes)
    for a, b, label in zip(EVENT_EDGES[:-1], EVENT_EDGES[1:], EVENT_LABELS, strict=True):
        row = {}
        for lag in EVENT_LAGS:
            totals, counts = np.zeros(n_scenes), np.zeros(n_scenes)
            for (scene, first), delta in events.items():
                if a <= first < b and lag < len(delta):
                    totals[scene] += delta[lag]
                    counts[scene] += 1
            if (counts > 0).sum() < 3:
                row[f"lag_{lag}"] = {"events": int(counts.sum()), "scenes": int((counts > 0).sum()), "mean": None}
                continue
            resampled = totals[picks].sum(axis=1) / np.maximum(counts[picks].sum(axis=1), 1)
            low, high = np.percentile(resampled, [2.5, 97.5])
            row[f"lag_{lag}"] = {
                "events": int(counts.sum()),
                "scenes": int((counts > 0).sum()),
                "mean_error_change": float(totals.sum() / counts.sum()),
                "ci": [float(low), float(high)],
            }
        table[label] = row
    return table


def error_around_handoff(events: dict, ensemble: np.ndarray) -> dict:
    """Error before the hand-off, and at its last frame with and without it, by start frame."""
    out = {}
    for a, b, label in zip(EVENT_EDGES[:-1], EVENT_EDGES[1:], EVENT_LABELS, strict=True):
        before, without, with_handoff = [], [], []
        for (scene, first), delta in events.items():
            if a <= first < b and len(delta) >= HANDOFF_FRAMES:
                before.append(ensemble[scene, first - 1] if first else 0.0)
                without.append(ensemble[scene, first + 1])
                with_handoff.append(ensemble[scene, first + 1] + delta[1])
        if before:
            before_, without_, with_ = np.array(before), np.array(without), np.array(with_handoff)
            out[label] = {
                "events": len(before),
                "error_frame_before": float(before_.mean()),
                "error_two_frames_on_without_handoff": float(without_.mean()),
                "error_two_frames_on_with_handoff": float(with_.mean()),
                "share_of_events_where_error_fell_below_the_frame_before": float((with_ < before_).mean()),
            }
    return out


def event_section(store: Store, group: str) -> dict:
    ensemble = store.runs("ensemble", group)
    exogenous = first_handoff_events(store, group, store.exogenous_names(group))
    doubt_names = [name for name in store.names("frozen", "sweep_doubt") if store.has(name, group)]
    doubt = first_handoff_events(store, group, doubt_names)
    oracle = first_handoff_events(store, group, store.policy_names(group)["oracle"])
    n = len(ensemble.errors)
    p95 = store.p95(group)

    # Does a frame's doubt or error say which start frames gain most, among starts in the same bin?
    within_bin = {}
    for a, b, label in zip(EVENT_EDGES[:-1], EVENT_EDGES[1:], EVENT_LABELS, strict=True):
        for lag in (1, 5, 10):
            pairs = [
                (delta[lag], p95[scene, first], ensemble.errors[scene, first - 1] if first else 0.0)
                for (scene, first), delta in exogenous.items()
                if a <= first < b and lag < len(delta)
            ]
            if len(pairs) >= 15:
                change, doubts, errors = np.array(pairs).T
                within_bin.setdefault(label, {})[f"lag_{lag}"] = {
                    "events": len(pairs),
                    "spearman_error_change_with_doubt": spearman(change, doubts),
                    "spearman_error_change_with_error_before": spearman(change, errors),
                }

    # A doubt-chosen first frame against unplanned first frames near it in the same scene.
    by_scene: dict[int, list] = {}
    for scene in range(n):
        for (s, first), delta in doubt.items():
            if s != scene or len(delta) < HANDOFF_FRAMES:
                continue
            near = [
                other[1]
                for (s2, f2), other in exogenous.items()
                if s2 == scene and abs(f2 - first) <= 3 and f2 != first and len(other) >= HANDOFF_FRAMES
            ]
            if near:
                by_scene.setdefault(scene, []).append(delta[1] - float(np.mean(near)))
    nearby = [float(np.mean(found)) for found in by_scene.values()]

    return {
        "note": "Negative change means the hand-off lowered error against the same scene's ensemble rollout.",
        "distinct_events": {"exogenous": len(exogenous), "doubt": len(doubt), "oracle": len(oracle)},
        "exogenous_error_change_by_start_bin": event_table(exogenous, n),
        "doubt_error_change_by_start_bin_selected_not_random": event_table(doubt, n),
        "oracle_error_change_by_start_bin_selected_not_random": event_table(oracle, n),
        "error_around_exogenous_handoffs": error_around_handoff(exogenous, ensemble.errors),
        "within_start_bin_spearman": within_bin,
        "doubt_start_less_nearby_exogenous_start_lag_1": mean_ci(nearby) if len(nearby) >= 3 else {"scenes": len(nearby)},
    }


def counts_by_bin(flags: np.ndarray) -> np.ndarray:
    return np.histogram(handoff_starts(flags[:WINDOW]), bins=TIME_EDGES)[0].astype(np.float64)


def solve_by_scene(xs: list[np.ndarray], ys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Least squares, and its coefficients over resampled scenes. xs[s], ys[s] are scene s's rows."""
    xtx = np.stack([x.T @ x for x in xs])
    xty = np.stack([x.T @ y for x, y in zip(xs, ys, strict=True)])
    fit = np.linalg.lstsq(xtx.sum(axis=0), xty.sum(axis=0), rcond=None)[0]
    picks = resamples(len(xs))
    found = np.stack([np.linalg.lstsq(xtx[p].sum(axis=0), xty[p].sum(axis=0), rcond=None)[0] for p in picks])
    return fit, found


def regression_section(store: Store, group: str) -> dict:
    """What a hand-off gains by when it starts, from random runs, whose times know nothing of the flow."""
    names = store.random_regression_names(group)
    ensemble = store.runs("ensemble", group).score
    n = len(ensemble)
    xs, ys = [[] for _ in range(n)], [[] for _ in range(n)]
    within_x, within_y = [[] for _ in range(n)], [[] for _ in range(n)]
    for name in names:
        runs = store.runs(name, group)
        for row in range(n):
            gain = ensemble[row] - runs.score[row]
            xs[row].append(counts_by_bin(runs.flags[row]))
            ys[row].append(gain)
            starts = handoff_starts(runs.flags[row, :WINDOW])
            if starts:
                pairs = int((np.diff(starts) == HANDOFF_FRAMES).sum())
                within_x[row].append([len(starts), len(starts) ** 2, (np.mean(starts) + 21) / 10, pairs])
                within_y[row].append(gain)
    x_scenes, y_scenes = [np.array(x) for x in xs], [np.array(y) for y in ys]
    beta, found = solve_by_scene(x_scenes, y_scenes)
    low, high = np.percentile(found, [2.5, 97.5], axis=0)

    def predict(runs: Runs) -> np.ndarray:
        return np.array([counts_by_bin(row) for row in runs.flags])

    predictions = {}
    policy_names = store.policy_names(group)
    fixed_name = next(iter(policy_names["fixed"]), None)
    doubt_counts = predict(store.runs("frozen", group))
    if fixed_name and store.runs("frozen", group).window_share > PREDICTION_SHARE_LIMIT:
        predictions = {"statement": "Not given: this budget is far above the rates the regression was fitted on."}
    elif fixed_name:
        fixed = store.runs(fixed_name, group)
        fixed_counts = predict(fixed)
        doubt_run = store.runs("frozen", group)
        difference = found @ (doubt_counts.mean(axis=0) - fixed_counts.mean(axis=0))
        predictions = {
            "fixed_run": fixed_name,
            "fixed_predicted_gain": float((fixed_counts @ beta).mean()),
            "fixed_actual_gain": float((ensemble - fixed.score).mean()),
            "doubt_predicted_gain": float((doubt_counts @ beta).mean()),
            "doubt_actual_gain": float((ensemble - doubt_run.score).mean()),
            "doubt_less_fixed_predicted_from_start_times_alone": float(
                beta @ (doubt_counts.mean(axis=0) - fixed_counts.mean(axis=0))
            ),
            "doubt_less_fixed_predicted_ci": [float(v) for v in np.percentile(difference, [2.5, 97.5])],
            "doubt_less_fixed_actual_gain": float((fixed.score - doubt_run.score).mean()),
        }
        if policy_names["oracle"]:
            oracle = store.runs(policy_names["oracle"][0], group)
            predictions["oracle_predicted_gain"] = float((predict(oracle) @ beta).mean())
            predictions["oracle_actual_gain"] = float((ensemble - oracle.score).mean())

    # Same scene, similar budget: did rollouts that happened to hand off later gain less?
    demeaned_x, demeaned_y = [], []
    for x, y in zip(within_x, within_y, strict=True):
        if len(x) >= 2:
            x, y = np.array(x, dtype=np.float64), np.array(y)
            demeaned_x.append(x - x.mean(axis=0))
            demeaned_y.append(y - y.mean())
    slopes, found_slopes = solve_by_scene(demeaned_x, demeaned_y)
    slope_low, slope_high = np.percentile(found_slopes, [2.5, 97.5], axis=0)
    terms = ("per_handoff", "per_handoff_squared", "per_10_frames_later_mean_start", "per_immediately_following_pair")
    return {
        "runs": names,
        "rollouts": int(sum(len(y) for y in y_scenes)),
        "window_gain_per_handoff_by_start_bin": {
            label: {"gain": float(beta[i]), "ci": [float(low[i]), float(high[i])]} for i, label in enumerate(TIME_LABELS)
        },
        "predictions_from_start_times": predictions,
        "same_scene_regression_on_random_runs": {
            term: {"coefficient": float(slopes[i]), "ci": [float(slope_low[i]), float(slope_high[i])]}
            for i, term in enumerate(terms)
        },
    }


def allocation_of(store: Store, group: str, policy: Runs, fixed: Runs | None) -> dict:
    """Split a policy's gain by which scenes got solver time, and how that time fell inside a scene.

    The even-spacing estimate reads each scene's own gain against share from its fixed-schedule runs at
    the share the policy gave that scene. A single rollout at each share is noisy, and a share past the
    largest schedule needs an assumption, so it is given both held flat and extended in a straight line.
    """
    ensemble = store.runs("ensemble", group).score
    n = len(ensemble)
    names = store.fixed_curve_names(group)
    curves = [store.runs(name, group) for name in names]
    shares = np.vstack([np.zeros(n), *[r.flags[:, :WINDOW].mean(axis=1) for r in curves]])
    gains = np.vstack([np.zeros(n), *[ensemble - r.score for r in curves]])
    target = policy.flags[:, :WINDOW].mean(axis=1)
    flat, extended = np.zeros(n), np.zeros(n)
    for scene in range(n):
        order = np.argsort(shares[:, scene], kind="stable")
        x, y = shares[order, scene], gains[order, scene]
        flat[scene] = np.interp(target[scene], x, y)
        slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-9)
        extended[scene] = flat[scene] + (slope * (target[scene] - x[-1]) if target[scene] > x[-1] else 0.0)
    actual = ensemble - policy.score
    chosen = target > 0
    out = {
        "scene_gain_curve_runs": names,
        "policy_gain": mean_ci(actual),
        "scenes_given_solver_time": int(chosen.sum()),
        "scenes_given_none": int((~chosen).sum()),
        "even_spacing_at_policy_scene_shares_held_flat": mean_ci(flat),
        "even_spacing_at_policy_scene_shares_extended": mean_ci(extended),
        "policy_less_even_spacing_held_flat": mean_ci(actual - flat),
        "policy_less_even_spacing_extended": mean_ci(actual - extended),
        "share_in_scenes_given_solver_time": float(target[chosen].mean()) if chosen.any() else 0.0,
        "spearman_scene_share_with_ensemble_scene_score": spearman(target, ensemble),
    }
    if fixed is not None:
        fixed_gain = ensemble - fixed.score
        out["fixed_gain"] = mean_ci(fixed_gain)
        out["policy_less_fixed"] = mean_ci(actual - fixed_gain)
        out["even_spacing_held_flat_less_fixed"] = mean_ci(flat - fixed_gain)
        out["even_spacing_extended_less_fixed"] = mean_ci(extended - fixed_gain)
        if chosen.any():
            out["in_scenes_given_solver_time"] = {
                "policy_gain": mean_ci(actual[chosen]),
                "fixed_gain": mean_ci(fixed_gain[chosen]),
            }
        if (~chosen).any():
            out["in_scenes_given_none"] = {"policy_gain": mean_ci(actual[~chosen]), "fixed_gain": mean_ci(fixed_gain[~chosen])}
            out["share_of_fixed_set_gain_from_scenes_given_none"] = float(fixed_gain[~chosen].sum() / fixed_gain.sum())
        if chosen.any() and (~chosen).any():
            found = []
            for pick in resamples(n):
                inside = chosen[pick]
                if inside.any() and (~inside).any():
                    found.append(fixed_gain[pick][~inside].mean() - fixed_gain[pick][inside].mean())
            out["fixed_gain_in_scenes_given_none_less_given_time"] = {
                "mean": float(fixed_gain[~chosen].mean() - fixed_gain[chosen].mean()),
                "ci": [float(v) for v in np.percentile(found, [2.5, 97.5])],
            }
    return out


def allocation_section(store: Store, group: str) -> dict:
    names = store.policy_names(group)
    if not names["fixed"]:
        return {"statement": "No baseline ran on this group, since its budget was 0."}
    fixed = store.runs(names["fixed"][0], group)
    out = {
        "fixed_run": names["fixed"][0],
        "doubt": allocation_of(store, group, store.runs("frozen", group), fixed),
        "oracle": allocation_of(store, group, store.runs(names["oracle"][0], group), fixed),
    }
    seeds = [store.runs(name, group).score.mean() for name in names["random"]]
    out["random_set_score_by_seed"] = [float(v) for v in seeds]
    out["random_set_score_seed_sd"] = float(np.std(seeds, ddof=1))
    random_scene = np.mean([store.runs(name, group).score for name in names["random"]], axis=0)
    out["fixed_less_random_score"] = mean_ci(fixed.score - random_scene)
    # One schedule is one chaotic draw, so read the budget off the other periods as well.
    ensemble = store.runs("ensemble", group).score
    others = sorted(
        (store.runs(name, group).window_share, float((ensemble - store.runs(name, group).score).mean()))
        for name in store.fixed_curve_names(group)
        if name != names["fixed"][0]
    )
    budget = store.runs("frozen", group).window_share
    shares, gains = np.array(others).T
    out["fixed_gain_at_budget_read_from_other_periods"] = (
        float(np.interp(budget, shares, gains)) if shares[0] <= budget <= shares[-1] else None
    )
    return out


def gain_curve_section(store: Store, group: str) -> dict:
    ensemble = store.runs("ensemble", group).score
    points = {}
    for name in sorted(store.fixed_curve_names(group)):
        runs = store.runs(name, group)
        gain = ensemble - runs.score
        points[name] = {
            "window_share": runs.window_share,
            "set_score": float(runs.score.mean()),
            "gain": mean_ci(gain),
            "gain_per_10_points_of_share": float(gain.mean() / runs.window_share * 0.1) if runs.window_share else None,
        }
    return points


def doubt_signal_section(store: Store, group: str, manifest: dict, tau: float) -> dict:
    p95 = store.p95(group)[:, :WINDOW]
    ensemble = store.runs("ensemble", group)
    errors = ensemble.errors[:, :WINDOW]
    logged = np.log(p95)
    timeless = logged - logged.mean(axis=0, keepdims=True)
    scene_part = timeless.mean(axis=1, keepdims=True)
    over = p95 > tau
    following = over[:, :-1]
    next_over = over[:, 1:]
    growth = errors[:, 5:] - errors[:, :-5]
    travel = np.array(manifest["splits"][store.split]["stats"]["travel_by_scene"])[store.scenes[group]]
    out = {
        "tau": tau,
        "share_of_frames_over_tau": float(over.mean()),
        "share_over_tau_by_bin": dict(
            zip(TIME_LABELS, [float(over[:, a:b].mean()) for a, b in zip(TIME_EDGES[:-1], TIME_EDGES[1:], strict=True)], strict=True)
        ),
        "median_doubt": float(np.median(p95)),
        "doubt_percentiles_5_50_95": [float(v) for v in np.percentile(p95, [5, 50, 95])],
        "largest_doubt": float(p95.max()),
        "share_of_scenes_ever_over_tau": float(over.any(axis=1).mean()),
        "share_of_scene_frames_over_tau_by_scene_sorted": [float(v) for v in np.sort(over.mean(axis=1))],
        "chance_over_tau_given_over_at_frame_before": float(next_over[following].mean()) if following.any() else None,
        "chance_over_tau_given_under_at_frame_before": float(next_over[~following].mean()) if (~following).any() else None,
        "share_of_log_doubt_variance_after_time_that_is_between_scenes": float(scene_part.var() / timeless.var()),
        "within_time_spearman_with_error": within_time_spearman(p95, errors),
        "within_time_spearman_with_error_growth_over_next_5_frames": within_time_spearman(p95[:, :-5], growth),
        "within_time_spearman_error_with_its_own_growth_over_next_5_frames": within_time_spearman(errors[:, :-5], growth),
        "scene_mean_doubt_with_ensemble_score": spearman_ci(p95.mean(axis=1), errors.mean(axis=1)),
        "scene_mean_doubt_with_largest_travel": spearman_ci(p95.mean(axis=1), travel),
        "ensemble_score_with_largest_travel": spearman_ci(errors.mean(axis=1), travel),
    }
    if store.has("sweep_fixed_10", group):
        gain = ensemble.score - store.runs("sweep_fixed_10", group).score
        out["scene_gain_at_period_10_with"] = {
            "mean_doubt": spearman_ci(p95.mean(axis=1), gain),
            "ensemble_score": spearman_ci(errors.mean(axis=1), gain),
            "mean_single_model_divergence": spearman_ci(store.signal(group)[:, :WINDOW].mean(axis=1), gain),
        }
    return out


def test_c_section(store: Store, handoff: dict, accuracy: dict, signals: dict) -> dict:
    report = accuracy["sets"]["test_c"]
    out = {"reference_ms_per_frame": report["reference_ms_per_frame"]}
    for group in ("low", "high"):
        policies = report["groups"][group]["policies"]
        found = {
            "doubt": signals[group]["doubt_percentiles_5_50_95"],
            "largest_doubt": signals[group]["largest_doubt"],
            "tau": signals[group]["tau"],
            "share_of_frames_over_tau": signals[group]["share_of_frames_over_tau"],
            "share_of_scene_frames_over_tau_by_scene_sorted": signals[group]["share_of_scene_frames_over_tau_by_scene_sorted"],
            "ensemble_score": policies["ensemble"]["set_score"],
            "coarse_solver_score": policies["coarse_solver"]["set_score"],
            "solver_only_score": policies["solver_only"]["set_score"],
            "solver_only_ms_per_frame": policies["solver_only"]["ms_per_frame"],
            "frozen_score": handoff["sets"]["test_c"]["groups"][group]["frozen"]["set_score"],
            "frozen_share": handoff["sets"]["test_c"]["groups"][group]["budget"],
            "frozen_ms_per_frame": handoff["sets"]["test_c"]["groups"][group]["frozen"]["ms_per_frame"],
            "unused_solver_gain_from_fixed_schedules": {
                name: {"share": point["window_share"], "gain": point["gain"]}
                for name, point in gain_curve_section(store, group).items()
            },
        }
        out[group] = found
    return out


def cost_section(store: Store, group: str) -> dict:
    """Milliseconds by what a rollout did, and whether the frozen policy's cost is explained by counts."""
    names = store.policy_names(group)
    full = ["frozen", *names["fixed"], *names["random"]]
    rows, totals = [], []
    for name in full:
        runs = store.runs(name, group)
        for row in range(len(runs.errors)):
            ai = float((~runs.flags[row]).sum())
            solver = runs.frames_made[row] - ai
            handoffs = len(handoff_starts(runs.flags[row]))
            rows.append([ai + handoffs, solver])
            totals.append(runs.ms[row])
    (per_step, per_solver), *_ = np.linalg.lstsq(np.array(rows), np.array(totals), rcond=None)

    def explain(name):
        runs = store.runs(name, group)
        ai = (~runs.flags).sum(axis=1).astype(np.float64)
        solver = runs.frames_made - ai
        handoffs = np.array([len(handoff_starts(row)) for row in runs.flags])
        predicted = (per_step * (ai + handoffs) + per_solver * solver).sum() / runs.frames_made.sum()
        return {
            "ms_per_frame": runs.ms_per_frame,
            "predicted_from_counts": float(predicted),
            "window_share": runs.window_share,
            "share_frames_21_to_120": runs.full_share,
        }

    out = {
        "note": "Saved ms cover frames 21 to 120, while solver share and the budget cover frames 21 to 70.",
        "ms_per_ensemble_step": float(per_step),
        "ms_per_solver_frame": float(per_solver) if any(row[1] for row in rows) else None,
        "doubt": explain("frozen"),
    }
    if names["fixed"]:
        out["fixed"] = explain(names["fixed"][0])
    if names["random"]:
        out["random_seed_0"] = explain(names["random"][0])
    if names["oracle"]:
        out["oracle_not_in_fit_as_its_steps_also_read_the_reference"] = explain(names["oracle"][0])

    # Sweep runs stop at frame 70, so there the frames timed are the frames budgeted.
    tau = next((name for name in store.names("sweep_doubt_0.01979") if store.has(name, group)), None)
    curve = sorted(
        ((store.runs(n, group).window_share, store.runs(n, group).ms_per_frame) for n in store.names("sweep_fixed") if store.has(n, group))
    )
    if tau and curve:
        runs = store.runs(tau, group)
        shares, costs = np.array(curve).T
        out["runs_stopping_at_frame_70"] = {
            "doubt_at_frozen_tau_share": runs.window_share,
            "doubt_at_frozen_tau_ms_per_frame": runs.ms_per_frame,
            "fixed_sweep_ms_per_frame_at_that_share": (
                float(np.interp(runs.window_share, shares, costs)) if shares[0] <= runs.window_share <= shares[-1] else None
            ),
            "fixed_sweep_points": [[float(a), float(b)] for a, b in curve],
        }
    return out


def secondary_section(store: Store, handoff: dict, accuracy: dict, split: str, group: str) -> dict:
    report = handoff["sets"][split]["groups"][group]
    accuracy_group = accuracy["sets"][split]["groups"][group]["policies"]

    def pick(run):
        found = {}
        for name, window in run["secondary"].items():
            found[name] = {
                "vorticity_correlation": window["vorticity_correlation"]["mean"],
                "divergence_rms": window["divergence_rms"]["mean"],
                "top_difference": window["top_difference"]["mean"],
                "centroid_difference": window["centroid_difference"]["mean"],
                "total_smoke_ratio": window["total_smoke_ratio"]["mean"],
                "spectrum_log_ratio_highest_band": window["spectrum_log_ratio"]["mean"][-1],
            }
        return found

    out = {"ensemble": pick(accuracy_group["ensemble"]), "solver_only": pick(accuracy_group["solver_only"]), "doubt": pick(report["frozen"])}
    for entry in report["fixed"]:
        out[f"fixed_{entry['role']}"] = pick(entry)
    for entry in report["random"]:
        out["random"] = pick(entry)
    if report["oracle"].get("run"):
        out["oracle"] = pick(report["oracle"])

    # Error past the primary window, where each policy's solver use differs, unlike inside it.
    long_errors = {"ensemble": store.runs("ensemble", group), "doubt": store.runs("frozen", group)}
    names = store.policy_names(group)
    if names["fixed"]:
        long_errors["fixed"] = store.runs(names["fixed"][0], group)
    if names["oracle"]:
        long_errors["oracle"] = store.runs(names["oracle"][0], group)
    if names["random"]:
        long_errors["random_seed_0"] = store.runs(names["random"][0], group)
    out["mean_error_frames_71_to_120"] = {
        name: {
            "error": float(runs.errors[:, WINDOW:HORIZON].mean()),
            "solver_share_frames_71_to_120": float(runs.flags[:, WINDOW:HORIZON].mean()),
        }
        for name, runs in long_errors.items()
    }
    return out


def ablation_section(ablation: dict, split: str, group: str) -> dict:
    found = ablation["sets"][split]["groups"][group]
    return {
        "full_member": found["full_member"]["set_score"],
        "reduced_trainings": {
            name: {
                "set_score": run["set_score"],
                "difference_from_full_member": run["difference_from_full_member"],
                "ci": run["difference_ci"],
            }
            for name, run in found["reduced_trainings"].items()
        },
        "ensemble_sizes": {
            size: {"set_score": run["set_score"], "within_time_spearman": run["evidence"]["value"]}
            for size, run in found["ensemble_sizes"].items()
        },
    }


def handoff_length_section(selection: dict) -> dict:
    """Validation's runs at each hand-off length, for the frozen aggregation. Validation only."""
    never = selection["choice"]["never_set_score"]
    out = {}
    for run in selection["runs"]:
        if run["aggregation"] == selection["choice"]["aggregation"]:
            out.setdefault(str(run["handoff_frames"]), []).append(
                {"share": run["solver_share"], "set_score": run["set_score"], "gain": never - run["set_score"]}
            )
    return {"split": "val", "never_set_score": never, "by_handoff_frames": out}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_timing(stores: dict, out: Path) -> None:
    panels = (("test_a", "all", "Test A"), ("test_b", "all", "Test B"), ("test_c", "high", "Test C, high side"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    frames = np.arange(21, 21 + WINDOW)
    for ax, (split, group, title) in zip(axes, panels, strict=True):
        store = stores[split]
        names = store.policy_names(group)
        fixed = names["fixed"][-1] if group == "high" else names["fixed"][0]
        random_flags = np.concatenate([store.runs(name, group).flags for name in names["random"]])
        series = (
            ("Doubt-based", store.runs("frozen", group).flags, BLUE),
            ("Fixed schedule", store.runs(fixed, group).flags, ORANGE),
            ("Random", random_flags, GREEN),
            ("Oracle", store.runs(names["oracle"][0], group).flags, AMBER),
        )
        for label, flags, color in series:
            ax.plot(frames, np.convolve(flags[:, :WINDOW].mean(axis=0), np.ones(5) / 5, "same"), color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Frame")
    axes[0].set_ylabel("Share of scenes on the solver (5-frame mean)")
    axes[0].legend()
    fig.suptitle("POST-HOC: when each policy used the solver", fontsize=10)
    _save(fig, out / "posthoc_timing.png")


def figure_event_study(events: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    bins = (("21-25", BLUE), ("26-30", ORANGE), ("31-40", GREEN), ("41-50", AMBER))
    for ax, key, title in zip(axes, ("test_a/all", "test_b/all"), ("Test A", "Test B"), strict=True):
        table = events[key]["exogenous_error_change_by_start_bin"]
        for label, color in bins:
            lags, means, lows, highs = [], [], [], []
            for lag in EVENT_LAGS:
                entry = table[label][f"lag_{lag}"]
                if entry.get("mean_error_change") is not None:
                    lags.append(lag)
                    means.append(entry["mean_error_change"])
                    lows.append(entry["ci"][0])
                    highs.append(entry["ci"][1])
            ax.fill_between(lags, lows, highs, color=color, alpha=0.12, linewidth=0)
            ax.plot(lags, means, color=color, marker="o", markersize=4, label=f"Frames {label}")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Frames since the hand-off began")
    axes[0].set_ylabel("Error change against the ensemble\n(below 0 is better)")
    axes[0].legend(title="First hand-off starts at")
    fig.suptitle("POST-HOC: effect of a first hand-off at a time set without looking at the flow", fontsize=10)
    _save(fig, out / "posthoc_first_handoff.png")


def figure_allocation(allocation: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    groups = ("Scenes doubt\ngave solver time", "Scenes doubt\ngave none")
    for ax, key, title in zip(axes, ("test_a/all", "test_b/all"), ("Test A", "Test B"), strict=True):
        found = allocation[key]["doubt"]
        sides = (found["in_scenes_given_solver_time"], found["in_scenes_given_none"])
        for offset, policy, label, color in ((-0.19, "policy_gain", "Doubt-based", BLUE), (0.19, "fixed_gain", "Fixed schedule", ORANGE)):
            means = [side[policy]["mean"] for side in sides]
            errs = [[m - side[policy]["ci"][0] for m, side in zip(means, sides, strict=True)], [side[policy]["ci"][1] - m for m, side in zip(means, sides, strict=True)]]
            ax.bar(np.arange(2) + offset, means, 0.36, color=color, label=label, yerr=errs, ecolor=MUTED, capsize=3)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(2), [f"{g}\n(n={side['policy_gain']['scenes']})" for g, side in zip(groups, sides, strict=True)])
        ax.set_title(title)
    axes[0].set_ylabel("Gain in scene score")
    axes[0].legend()
    fig.suptitle("POST-HOC: where the gain came from at equal solver budget", fontsize=10)
    _save(fig, out / "posthoc_allocation.png")


def figure_share_curves(stores: dict, handoff: dict, accuracy: dict, out: Path) -> None:
    panels = (("test_a", "all", "Test A"), ("test_b", "all", "Test B"), ("test_c", "high", "Test C, high side"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (split, group, title) in zip(axes, panels, strict=True):
        store = stores[split]
        points = sorted((store.runs(n, group).window_share, store.runs(n, group).score.mean()) for n in store.fixed_curve_names(group))
        shares, scores = np.array(points).T
        ax.plot(shares, scores, color=ORANGE, marker="o", markersize=5, label="Fixed schedules")
        report = handoff["sets"][split]["groups"][group]
        ax.plot(report["budget"], report["frozen"]["set_score"], "o", color=BLUE, markersize=9, label="Doubt-based, frozen")
        ax.plot(report["oracle"]["achieved_share"], report["oracle"]["set_score"], "D", color=AMBER, markersize=7, label="Oracle")
        policies = accuracy["sets"][split]["groups"][group]["policies"]
        ax.axhline(policies["ensemble"]["set_score"], color=GREEN, linewidth=1.5, label="Ensemble, no hand-off")
        ax.axhline(policies["solver_only"]["set_score"], color=PURPLE, linewidth=1.5, label="Solver only, one run")
        ax.set_title(title)
        ax.set_xlabel("Solver share, frames 21 to 70")
    axes[0].set_ylabel("Set score (lower is better)")
    axes[0].legend()
    fig.suptitle("POST-HOC: repeated 2-frame hand-offs stay far above one continuous solver run", fontsize=10)
    _save(fig, out / "posthoc_share_curves.png")


def figure_test_c_doubt(stores: dict, tau: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    frames = np.arange(21, 21 + WINDOW)
    for group, label, color in (("low", "Low side", BLUE), ("high", "High side", ORANGE)):
        p95 = stores["test_c"].p95(group)[:, :WINDOW]
        low, middle, high = np.percentile(p95, [5, 50, 95], axis=0)
        ax.fill_between(frames, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(frames, middle, color=color, label=label)
    ax.axhline(tau, color=INK, linewidth=1.2, linestyle="--", label="Frozen threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Doubt score (median, 5th to 95th percentile)")
    ax.set_ylim(2e-3, 1.2e-1)
    ax.legend(loc="lower right", ncols=3)
    ax.set_title("POST-HOC: one threshold sat above one side and below the other")
    _save(fig, out / "posthoc_test_c_doubt.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/experiments/final"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    read = lambda name: json.loads((args.results / name).read_text())  # noqa: E731
    handoff, accuracy, ablation, selection, manifest = (
        read("exp_handoff.json"),
        read("exp_accuracy.json"),
        read("exp_ablation.json"),
        read("handoff_selection.json"),
        read("data_manifest.json"),
    )
    tau = handoff["selection_choice"]["tau"]
    stores = {split: Store(args.runs, split) for split in ("test_a", "test_b", "test_c")}
    checks = check_inputs(stores, handoff, accuracy, manifest)

    sections = {
        name: {}
        for name in ("timing", "first_handoff", "start_time_regression", "allocation", "gain_curves", "doubt_signal", "cost", "secondary", "ablation")
    }
    signals = {}
    for split, group in GROUPS:
        key = f"{split}/{group}"
        store = stores[split]
        sections["timing"][key] = timing_section(store, group)
        sections["first_handoff"][key] = event_section(store, group)
        sections["start_time_regression"][key] = regression_section(store, group)
        sections["allocation"][key] = allocation_section(store, group)
        sections["gain_curves"][key] = gain_curve_section(store, group)
        signals[group] = sections["doubt_signal"][key] = doubt_signal_section(store, group, manifest, tau)
        sections["cost"][key] = cost_section(store, group)
        sections["secondary"][key] = secondary_section(store, handoff, accuracy, split, group)
        sections["ablation"][key] = ablation_section(ablation, split, group)

    result = {
        "label": "POST-HOC, exploratory. Not preregistered, and chosen after the results were seen.",
        "limits": [
            "Explanations are fitted to the same test sets that showed the results, so they need new data to confirm.",
            "Several comparisons were tried, and the intervals aren't corrected for that.",
            "Hand-off rollouts save no doubt scores, so doubt is known only on no-hand-off rollouts.",
            "Each fixed schedule is one rollout per scene, and flows are chaotic, so one schedule's score carries noise.",
            "The saved rollouts can't separate chaos from plain carrying forward of an error already in the state.",
            "A late obstacle's scene and frame aren't in these files, so its effect on hand-off isn't analysed here.",
        ],
        "inputs": {
            name: hashlib.sha256((args.results / name).read_bytes()).hexdigest() for name in INPUT_FILES
        },
        "git_commit_when_run": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "bootstrap": {"resamples": N_RESAMPLES, "seed": SEED, "unit": "whole scenes"},
        "checks": checks,
        **sections,
        "test_c_sides": test_c_section(stores["test_c"], handoff, accuracy, signals),
        "handoff_length_on_validation": handoff_length_section(selection),
    }
    (args.results / "posthoc_analysis.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")

    figures = args.results / "figures"
    figure_timing(stores, figures)
    figure_event_study(sections["first_handoff"], figures)
    figure_allocation(sections["allocation"], figures)
    figure_share_curves(stores, handoff, accuracy, figures)
    figure_test_c_doubt(stores, tau, figures)


if __name__ == "__main__":
    main()
