"""Hand-off selection on validation, step by step, as docs/PREREGISTRATION.md sets it out."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import sandeha.handoff_selection as selection
from sandeha.data.store import SplitStore
from sandeha.ensemble import Ensemble
from sandeha.handoff_selection import (
    Window,
    aggregations_left_in,
    candidate_thresholds,
    choose_run,
    evidence_measure,
    frame_error,
    make_tiny_val,
    select_handoff,
)
from sandeha.models.factory import build_model
from sandeha.physics.solver import SmokeSolver

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_val(tmp_path_factory.mktemp("selection") / "val")


# The primary metric


def test_frame_error_is_zero_for_identical_and_ignores_solid():
    ref = np.random.default_rng(0).normal(size=(3, 8, 8))
    solid = np.zeros((8, 8), bool)
    solid[0, 0] = True
    pred = ref.copy()
    pred[:, 0, 0] = 1e9
    assert frame_error(pred, ref, solid, [0.01, 0.01, 0.01]) == 0.0


def test_frame_error_is_per_channel_rms_over_reference_std():
    ref = np.zeros((3, 4, 4))
    ref[:, ::2] = 1.0  # Standard deviation 0.5 in each channel.
    pred = ref + np.array([0.1, 0.2, 0.3]).reshape(3, 1, 1)
    want = (0.1 / 0.5 + 0.2 / 0.5 + 0.3 / 0.5) / 3
    assert frame_error(pred, ref, np.zeros((4, 4), bool), [0.01, 0.01, 0.01]) == pytest.approx(want)


def test_frame_error_uses_the_floor_on_a_nearly_empty_frame():
    ref = np.zeros((3, 4, 4))
    pred = np.full((3, 4, 4), 0.01)
    assert frame_error(pred, ref, np.zeros((4, 4), bool), [0.02, 0.02, 0.02]) == pytest.approx(0.5)


def test_frame_error_caps_blow_ups():
    ref = np.zeros((3, 4, 4))
    ref[:, ::2] = 1.0
    solid = np.zeros((4, 4), bool)
    nan = ref.copy()
    nan[2, 1, 1] = math.nan
    assert frame_error(nan, ref, solid, [0.01] * 3) == 10.0
    assert frame_error(ref + 1e6, ref, solid, [0.01] * 3) == 10.0


def test_frame_error_with_no_fluid_is_zero():
    assert frame_error(np.ones((3, 4, 4)), np.zeros((3, 4, 4)), np.ones((4, 4), bool), [0.01] * 3) == 0.0


# Step 1: which aggregations stay in


def test_evidence_measure_is_the_mean_correlation_within_times():
    errors = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    measure, undefined = evidence_measure(errors, errors)
    assert measure == pytest.approx(1.0)
    assert undefined == 0
    opposite, _ = evidence_measure(-errors, errors)
    assert opposite == pytest.approx(-1.0)


def test_evidence_measure_leaves_undefined_times_out_and_counts_them():
    doubt = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    errors = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    measure, undefined = evidence_measure(doubt, errors)
    assert measure == pytest.approx(1.0)
    assert undefined == 1
    measure, undefined = evidence_measure(np.ones((3, 2)), errors)
    assert math.isnan(measure)
    assert undefined == 2


def test_evidence_measure_ranks_non_finite_doubt_highest():
    doubt = np.array([[1.0], [math.inf], [2.0], [math.nan]])
    errors = np.array([[1.0], [3.5], [2.0], [3.5]])
    measure, _ = evidence_measure(doubt, errors)
    assert measure == pytest.approx(1.0)


def test_aggregations_left_in_keeps_the_leader_and_those_not_shown_worse():
    rng = np.random.default_rng(0)
    errors = rng.normal(size=(30, 10))
    doubts = {
        "mean": errors + 0.01 * rng.normal(size=errors.shape),
        "p95": errors + 0.02 * rng.normal(size=errors.shape),
        "p99": rng.normal(size=errors.shape),
    }
    report = aggregations_left_in(doubts, errors, n_resamples=200)
    assert report["leader"] == "mean"
    assert report["left_in"] == ["mean", "p95"]
    low, high = report["by_aggregation"]["p99"]["difference_ci"]
    assert high < 0
    assert report["by_aggregation"]["mean"]["difference_ci"] is None


def test_aggregations_left_in_is_repeatable_and_a_tie_goes_to_the_earlier():
    errors = np.random.default_rng(1).normal(size=(10, 5))
    doubts = {name: errors.copy() for name in ("p99", "p95", "mean")}
    first = aggregations_left_in(doubts, errors, n_resamples=50)
    assert first["leader"] == "mean"
    assert first["left_in"] == ["mean", "p95", "p99"]
    assert first == aggregations_left_in(doubts, errors, n_resamples=50)


# Step 2: candidate thresholds


def test_candidate_thresholds_are_percentiles_without_repeats_or_non_finite():
    scores = np.arange(1.0, 101.0)
    found = candidate_thresholds(scores, (50, 90))
    assert [c["percentile"] for c in found] == [50, 90]
    assert [c["tau"] for c in found] == pytest.approx([50.5, 90.1])

    assert len(candidate_thresholds(np.ones(100), (50, 60, 70))) == 1

    scores = np.concatenate([np.arange(1.0, 91.0), np.full(10, math.inf)])
    found = candidate_thresholds(scores, (50, 95, 99))
    assert [c["percentile"] for c in found] == [50]


# Step 4: the choice


def run(aggregation, handoff_frames, tau, score, share):
    return {
        "aggregation": aggregation,
        "handoff_frames": handoff_frames,
        "tau": tau,
        "set_score": score,
        "solver_share": share,
    }


NEVER = run(None, None, None, 0.5, 0.0)


def test_choice_is_the_lowest_score_within_the_share():
    chosen = choose_run([NEVER, run("mean", 2, 1.0, 0.4, 0.2), run("mean", 5, 0.5, 0.3, 0.26)])
    assert chosen["set_score"] == 0.4


def test_choice_counts_a_share_of_exactly_a_quarter():
    assert choose_run([NEVER, run("p95", 2, 1.0, 0.4, 0.25)])["set_score"] == 0.4


def test_choice_goes_to_never_when_nothing_beats_it():
    assert choose_run([NEVER, run("mean", 2, 1.0, 0.6, 0.2)]) is NEVER


def test_choice_compares_scores_rounded_to_six_decimals_then_lower_share():
    chosen = choose_run(
        [NEVER, run("mean", 2, 1.0, 0.4000004, 0.2), run("mean", 5, 1.5, 0.3999996, 0.21)]
    )
    assert chosen["handoff_frames"] == 2


def test_choice_tie_goes_to_aggregation_order_then_smaller_handoff():
    runs = [NEVER, run("p99", 2, 1.0, 0.4, 0.2), run("p95", 5, 1.0, 0.4, 0.2), run("p95", 2, 2.0, 0.4, 0.2)]
    assert choose_run(runs)["aggregation"] == "p95"
    assert choose_run(runs)["handoff_frames"] == 2
    assert choose_run(runs + [run("mean", 10, 3.0, 0.4, 0.2)])["aggregation"] == "mean"


def test_a_tie_with_never_goes_to_never():
    assert choose_run([run("mean", 2, 1.0, 0.5, 0.0), NEVER]) is NEVER


# The whole selection, on tiny data with small random models


def tiny_ensemble(normalizer) -> Ensemble:
    models = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        models.append(build_model({"arch": "unet", "base_width": 8, "levels": 2}))
    return Ensemble(models, normalizer)


def run_selection(tiny, tmp_path, **options):
    store, normalizer, _ = tiny
    settings = {"window": Window(1, 5), "percentiles": (50, 90), "handoff_frames": (2,), "n_resamples": 20}
    return select_handoff(
        tiny_ensemble(normalizer),
        SmokeSolver(64, "cpu", 0.01),
        store,
        progress_path=tmp_path / "progress.json",
        log=lambda _: None,
        **(settings | options),
    )


def test_select_handoff_on_tiny_data_records_each_step(tiny, tmp_path):
    report = run_selection(tiny, tmp_path, provenance={"git_commit": "abc"})
    assert report["split"] == "val"
    assert report["git_commit"] == "abc"
    assert report["window"] == {"start": 1, "end": 5}
    assert set(report["aggregation"]["by_aggregation"]) == {"mean", "p95", "p99"}

    chosen = report["choice"]
    assert chosen["kind"] in ("never", "threshold")
    assert chosen["solver_share"] <= 0.25
    assert report["bad_frame_threshold"] > 0
    assert set(report["calibration_edges"]) == {"mean", "p95", "p99"}

    runs = report["runs"]
    never = [r for r in runs if r["tau"] is None]
    assert len(never) == 1
    assert never[0]["solver_share"] == 0
    assert never[0]["forced_handoffs"] == 0
    finite = [r for r in runs if r["tau"] is not None]
    assert finite
    assert {r["aggregation"] for r in finite} == set(report["aggregation"]["left_in"])
    assert all(len(r["scene_scores"]) == 3 for r in runs)
    assert all(r["set_score"] == pytest.approx(np.mean(r["scene_scores"])) for r in runs)
    # The report holds valid JSON, with no bare infinities.
    json.loads(json.dumps(report, allow_nan=False))


def test_select_handoff_counts_a_share_only_inside_the_window(tiny, tmp_path):
    # A hand-off of 10 frames, starting inside a 4-frame window, still counts only what is inside.
    report = run_selection(tiny, tmp_path, handoff_frames=(10,))
    assert all(0 <= r["solver_share"] <= 1 for r in report["runs"])
    assert any(r["solver_share"] > 0 for r in report["runs"])


def test_select_handoff_resumes_finished_runs(tiny, tmp_path, monkeypatch):
    first = run_selection(tiny, tmp_path)
    calls = []
    original = selection._rollout

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "_rollout", counted)
    assert run_selection(tiny, tmp_path) == first
    assert not calls


def test_select_handoff_restarts_across_a_commit_and_lists_both(tiny, tmp_path):
    run_selection(tiny, tmp_path, percentiles=(50,), provenance={"git_commit": "one"})
    report = run_selection(tiny, tmp_path, percentiles=(50,), provenance={"git_commit": "one", "git_dirty": True})
    assert report["run_commits"] == ["one"]
    report = run_selection(tiny, tmp_path, percentiles=(50,), provenance={"git_commit": "two"})
    assert report["git_commit"] == "two"
    assert report["run_commits"] == ["one"]


def test_select_handoff_refuses_progress_from_other_settings(tiny, tmp_path):
    run_selection(tiny, tmp_path, provenance={"checkpoint_sha256": ["a"]})
    with pytest.raises(ValueError, match="other settings"):
        run_selection(tiny, tmp_path, provenance={"checkpoint_sha256": ["b"]})


class AllNaN(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 3, *x.shape[2:]), math.nan)


def test_a_wholly_blown_up_ensemble_leaves_only_never(tiny, tmp_path):
    store, normalizer, _ = tiny
    report = select_handoff(
        Ensemble([AllNaN(), AllNaN()], normalizer),
        SmokeSolver(64, "cpu", 0.01),
        store,
        progress_path=tmp_path / "progress.json",
        window=Window(1, 5),
        n_resamples=5,
        log=lambda _: None,
    )
    assert report["aggregation"]["left_in"] == []
    assert report["choice"]["kind"] == "never"
    assert report["choice"]["set_score"] == 10.0
    assert report["runs"][0]["blown_up_rollouts"] == 3
    assert report["calibration_edges"]["mean"] == []
    assert report["aggregation"]["by_aggregation"]["mean"]["non_finite_frames"] == 12
    json.loads(json.dumps(report, allow_nan=False))


def test_selection_refuses_any_split_but_validation(tiny, tmp_path):
    store, normalizer, _ = tiny
    for split in ("train", "test_b"):
        with pytest.raises(ValueError, match="validation"):
            select_handoff(
                tiny_ensemble(normalizer),
                SmokeSolver(64, "cpu", 0.01),
                SplitStore(store.directory, split),
                progress_path=tmp_path / "other.json",
                window=Window(1, 5),
            )


def run_script(*arguments):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "select_handoff.py"), *arguments],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )


def test_script_dry_run_writes_only_where_it_is_told(tmp_path):
    done = run_script("--dry-run", "--out", str(tmp_path))
    assert done.returncode == 0, done.stderr
    report = json.loads((tmp_path / "handoff_selection.json").read_text())
    assert report["split"] == "val"
    assert report["choice"]["kind"] in ("never", "threshold")


def test_script_refuses_without_a_trained_ensemble(tmp_path):
    done = run_script(
        "--ensemble", str(tmp_path / "missing.json"), "--results", str(tmp_path / "out.json")
    )
    assert done.returncode != 0
    assert "train the ensemble first" in done.stderr
    assert not (tmp_path / "out.json").exists()
