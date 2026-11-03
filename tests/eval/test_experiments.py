"""Rollouts, baselines, the oracle's search, provenance and the split lock, as preregistered."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sandeha.eval.experiments as experiments
from sandeha.data.store import sha256_file
from sandeha.ensemble import Ensemble
from sandeha.eval.experiments import (
    Evaluation,
    MissingProvenance,
    _pooled_baselines,
    check_provenance,
    choose_fixed_schedules,
    combine_runs,
    compare_to_baseline,
    find_random_rate,
    fixed_share,
    handoff_experiment,
    make_tiny_sets,
    oracle_search,
    open_eval_sets,
    random_share,
    with_provenance,
)
from sandeha.eval.rollout import CoarseSolverSim, IndependentSim, PersistenceSim, rollout
from sandeha.handoff_selection import Window, choose_run
from sandeha.hybrid import AlwaysAI, HybridSimulator
from sandeha.models.factory import build_model
from sandeha.physics.solver import SmokeSolver

REPO = Path(__file__).resolve().parents[2]
WINDOW = Window(20, 70)
SMALL_MODEL = {"arch": "unet", "base_width": 8, "levels": 2}


class Exploding(nn.Module):
    def forward(self, x):
        return torch.full_like(x[:, :3], torch.nan)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_sets(tmp_path_factory.mktemp("sets"))


def floors_of(normalizer):
    return [0.01 * std for std in normalizer.field_std]


def small_ensemble(normalizer, n=3):
    models = []
    for seed in range(n):
        torch.manual_seed(seed)
        models.append(build_model(SMALL_MODEL))
    return Ensemble(models, normalizer)


# Rollouts


def test_blown_up_rollout_counts_as_max_error(tiny):
    stores, normalizer, config = tiny
    solver = SmokeSolver(config["res"], "cpu", config["dt"])
    sim = HybridSimulator(Ensemble([Exploding()], normalizer), solver, AlwaysAI(), 1, 4)
    trace = rollout(sim, 0, stores["val"], 1, 5, floors=floors_of(normalizer))
    assert trace.errors == [10.0] * 5  # Every frame stays, each at the cap.
    assert trace.blown_up
    assert np.mean(trace.errors) == 10.0


def test_rollout_stops_at_the_horizon_and_records_sources(tiny):
    stores, normalizer, config = tiny
    solver = SmokeSolver(config["res"], "cpu", config["dt"])
    sim = HybridSimulator(small_ensemble(normalizer), solver, AlwaysAI(), 1, 4)
    trace = rollout(sim, 1, stores["val"], 1, 3, floors=floors_of(normalizer), scores_by=("mean", "p95"))
    assert len(trace.errors) == len(trace.solver_flags) == 3
    assert not any(trace.solver_flags)
    assert len(trace.scores["p95"]) == 3
    assert trace.frames_made == 3 and trace.ms > 0
    assert len(trace.secondary) == 3


def test_baseline_sims_step_like_the_hybrid(tiny):
    stores, normalizer, config = tiny
    fine = SmokeSolver(config["res"], "cpu", config["dt"])
    coarse = SmokeSolver(config["coarse_res"], "cpu", config["dt"])
    floors = floors_of(normalizer)
    store = stores["val"]

    persistence = rollout(PersistenceSim(fine), 0, store, 1, 2, floors=floors)
    assert persistence.errors[0] > 0

    solved = rollout(CoarseSolverSim(fine, coarse, 4), 0, store, 1, 2, floors=floors)
    assert all(np.isfinite(solved.errors))
    assert solved.errors[0] < 10

    independent = rollout(IndependentSim(small_ensemble(normalizer), fine, 4), 0, store, 1, 2, floors=floors, scores_by=("mean",))
    assert len(independent.scores["mean"]) == 2
    assert independent.scores["mean"][0] > 0


# Selection


def test_selection_steps_and_tie_breaks():
    def run(score, share, aggregation="mean", frames=2, tau=1.0):
        return {"set_score": score, "solver_share": share, "aggregation": aggregation, "handoff_frames": frames, "tau": tau}

    never = run(0.5, 0.0, None, None, None)
    over_budget = run(0.1, 0.3)
    assert choose_run([never, over_budget]) is never
    low_share = run(0.4, 0.1)
    assert choose_run([never, low_share, run(0.4000001, 0.05)]) is not low_share  # Rounded, then share.
    assert choose_run([run(0.4, 0.1, "p95"), run(0.4, 0.1, "mean")])["aggregation"] == "mean"
    assert choose_run([run(0.4, 0.1, "mean", 5), run(0.4, 0.1, "mean", 2)])["handoff_frames"] == 2


# Baselines


def test_baseline_schedule_nearest_the_budget():
    # With a 2-frame hand-off, period 10 schedules about 20%.
    chosen = choose_fixed_schedules(0.2, 2, 30, WINDOW)
    assert len(chosen) == 1
    assert chosen[0]["period"] == 10
    assert chosen[0]["role"] == "nearest"
    assert abs(chosen[0]["scheduled_share"] - 0.2) <= 0.01


def test_baseline_schedule_tie_goes_to_the_longer_period(monkeypatch):
    shares = {2: 0.30, 3: 0.20}
    monkeypatch.setattr(experiments, "fixed_share", lambda period, *a: shares.get(period, 0.01))
    chosen = choose_fixed_schedules(0.25, 2, 1, Window(0, 2))
    assert [item["period"] for item in chosen if item["role"] != "nearest"] == [3, 2]


def test_fixed_share_counts_only_inside_the_window():
    # One scene, hand-offs of 10 from frame 21 every 10 frames: every window frame is the solver's.
    assert fixed_share(10, 10, 1, WINDOW) == 1.0
    # A hand-off running past the window counts only inside it.
    assert fixed_share(48, 5, 1, WINDOW) == pytest.approx((5 + 2) / 50)


def test_unmatched_budget_runs_both_neighbours():
    # A hand-off of 10 can only schedule 100%, 91% (period 11), ... well apart near the top.
    chosen = choose_fixed_schedules(0.955, 10, 1, WINDOW)
    roles = {item["role"]: item for item in chosen}
    assert set(roles) == {"below", "above"}
    assert roles["below"]["scheduled_share"] < 0.955 < roles["above"]["scheduled_share"]
    assert abs(roles["below"]["scheduled_share"] - 0.955) > 0.01


def test_random_rate_reaches_the_budget_on_its_streams():
    rate = find_random_rate(0.2, 2, list(range(10)), WINDOW, (0, 1, 2, 3, 4))
    assert abs(random_share(rate, 2, list(range(10)), WINDOW, (0, 1, 2, 3, 4)) - 0.2) < 0.01


def test_comparison_needs_an_interval_below_zero_and_enough_budget():
    frozen = np.array([0.1, 0.2, 0.15, 0.12, 0.11, 0.13])
    matched = {"scene_scores": (frozen + 0.1).tolist(), "achieved_share": 0.20}
    assert compare_to_baseline(frozen, 0.20, matched, 100)["beats"]
    assert compare_to_baseline(frozen, 0.20, matched, 100)["status"] == "matched"

    above = {"scene_scores": (frozen + 0.1).tolist(), "achieved_share": 0.30}
    assert compare_to_baseline(frozen, 0.20, above, 100)["beats"]
    assert compare_to_baseline(frozen, 0.20, above, 100)["status"] == "above"

    below = {"scene_scores": (frozen + 0.1).tolist(), "achieved_share": 0.10}
    found = compare_to_baseline(frozen, 0.20, below, 100)
    assert found["status"] == "below" and not found["beats"]
    assert found["difference_ci"][1] < 0  # The interval is reported all the same.

    worse = {"scene_scores": (frozen - 0.1).tolist(), "achieved_share": 0.20}
    assert not compare_to_baseline(frozen, 0.20, worse, 100)["beats"]


def test_pooled_runs_join_the_sides_in_order():
    low = {"scene_scores": [1.0], "solver_frames": 10, "total_frames": 50, "forced_handoffs": 1, "blown_up": 0, "n_rollouts": 1, "ms": 5.0, "frames_made": 50}
    high = {"scene_scores": [2.0, 3.0], "solver_frames": 30, "total_frames": 100, "forced_handoffs": 2, "blown_up": 1, "n_rollouts": 2, "ms": 5.0, "frames_made": 50}
    joined = combine_runs(low, high)
    assert joined["scene_scores"] == [1.0, 2.0, 3.0]
    assert joined["achieved_share"] == pytest.approx(40 / 150)
    assert joined["blown_up_share"] == pytest.approx(1 / 3)
    assert joined["forced_handoffs"] == 3


# The oracle


def test_oracle_tau_search_keeps_the_nearest_share():
    errors = np.linspace(0, 1, 101)
    # This share never comes within 1 point: 0.5 below tau 0.5, 0.05 above it.
    def run_share(tau):
        return 0.5 if tau < 0.5 else 0.05

    found = oracle_search(0.2, errors, run_share)
    assert found["run"]
    assert len(found["candidates"]) == 9  # The percentile and 8 bisections.
    assert found["candidates"][0]["tau"] == pytest.approx(np.percentile(errors, 80))
    assert found["kept"]["share"] == 0.05  # Nearest the budget: 0.15 from it, against 0.3.
    assert found["unmatched"]

    # The search stops at the first share within a point.
    shares = iter([0.5, 0.205])
    stopped = oracle_search(0.2, errors, lambda tau: next(shares))
    assert len(stopped["candidates"]) == 2
    assert stopped["kept"]["share"] == 0.205
    assert not stopped["unmatched"]
    # A share above the budget raises the lower end: the second candidate is above the first.
    assert stopped["candidates"][1]["tau"] > stopped["candidates"][0]["tau"]


def test_oracle_tau_search_ties_go_to_the_lower_share():
    errors = np.linspace(0, 1, 101)
    found = oracle_search(0.2, errors, lambda tau: 0.3 if tau < 5 else 0.1)
    assert found["kept"]["share"] == 0.1


def test_oracle_is_not_run_on_a_budget_of_zero():
    def run_share(tau):
        raise AssertionError("the oracle must not run")

    found = oracle_search(0.0, np.ones(10), run_share)
    assert not found["run"]
    assert "budget of 0" in found["statement"]


# Test C's sides


def _run(scores, solver_frames, role=None):
    run = {
        "scene_scores": scores, "solver_frames": solver_frames, "total_frames": 50 * len(scores), "forced_handoffs": 0,
        "blown_up": 0, "n_rollouts": len(scores), "ms": 1.0, "frames_made": 50 * len(scores),
    }
    return {**run, "role": role} if role else run


def test_pooled_baseline_uses_no_handoff_for_a_side_with_no_budget():
    low = {"frozen": _run([0.5], 0), "fixed": []}
    high = {"frozen": _run([0.4], 10), "fixed": [_run([0.6], 10, "below"), _run([0.7], 15, "above")]}
    pooled = _pooled_baselines(_FakeEv, low, high, "fixed", {"scene_scores": [0.5, 0.4]}, 0.1)
    assert [run["role"] for run in pooled] == ["above", "below"]
    assert pooled[1]["scene_scores"] == [0.5, 0.6]  # In the order the frozen run pairs with.
    assert pooled[1]["achieved_share"] == pytest.approx(0.1)


class _FakeEv:
    n_resamples = 20


def test_handoff_on_test_c_runs_each_side_and_pools_them(tiny, tmp_path):
    stores, normalizer, config = tiny
    store = stores["test_c"]
    settings = {
        "start": 1, "primary_end": 6, "long_end": 7, "n_resamples": 20, "usable_error": 1.0, "rollout_lengths": [1],
        "timing_scenes": 1, "timing_frames": 1, "default_handoff_frames": 2, "random_seeds": [0],
        "ensemble_sizes": [2], "sweep": {"periods": [3], "rates": [0.2]},
    }
    selection = {
        "choice": {"kind": "threshold", "aggregation": "mean", "handoff_frames": 2, "tau": 0.0},
        "aggregation": {"leader": "mean"},
        "candidates": {},
        "bad_frame_threshold": 0.5,
        "calibration_edges": {"mean": [0.1], "p95": [0.1], "p99": [0.1]},
    }
    evaluation = Evaluation(
        ensemble=small_ensemble(normalizer),
        solver=SmokeSolver(config["res"], "cpu", config["dt"]),
        coarse_solver=SmokeSolver(config["coarse_res"], "cpu", config["dt"]),
        substeps=4, amp=False, selection=selection, config=settings,
        provenance={"data_manifest_sha256": "x", "checkpoint_sha256": ["y"]},
        progress_dir=tmp_path, log=lambda _: None,
    )
    groups = handoff_experiment(evaluation, store)["groups"]
    assert set(groups) == {"low", "high", "pooled"}
    for name in ("low", "high"):
        assert groups[name]["budget"] == 1.0  # A threshold of 0 hands off every frame.
        assert groups[name]["fixed"] and groups[name]["random"] and groups[name]["oracle"]["run"]
    pooled = groups["pooled"]
    assert pooled["frozen"]["n_rollouts"] == 4
    assert "long_window" in pooled["frozen"]["secondary"]
    assert pooled["fixed"][0]["status"] in ("matched", "above", "below")
    assert len(pooled["fixed"][0]["scene_scores"]) == 4
    assert len(pooled["random"][0]["scene_scores"]) == 4
    assert pooled["oracle"]["descriptive"]
    assert len(pooled["oracle"]["scene_scores"]) == 4


# Provenance and the lock


def test_results_carry_provenance():
    provenance = {"git_commit": "abc", "data_manifest_sha256": "def", "checkpoint_sha256": ["0"], "config": {"a": 1}}
    result = with_provenance({"sets": {}}, provenance)
    for name in provenance:
        assert result[name] == provenance[name]
    check_provenance(result)

    for name in provenance:
        partial = {key: value for key, value in provenance.items() if key != name}
        with pytest.raises(MissingProvenance, match=name):
            with_provenance({"sets": {}}, partial)
    with pytest.raises(MissingProvenance, match="checkpoint_sha256"):
        with_provenance({"sets": {}}, provenance | {"checkpoint_sha256": []})


def test_experiments_use_val_unless_final(monkeypatch):
    opened = []

    def recording(root, split, final_eval=False):
        opened.append((split, final_eval))
        return split

    monkeypatch.setattr(experiments, "open_split", recording)
    stores, reference = open_eval_sets(Path("nowhere"), final=False)
    assert opened == [("val", False)]
    assert list(stores) == ["val"]
    assert reference == "val"

    opened.clear()
    stores, reference = open_eval_sets(Path("nowhere"), final=True)
    assert list(stores) == ["test_a", "test_b", "test_c"]
    assert ("test_a", True) in opened and ("test_c", True) in opened
    assert reference == "val"  # Only for the frozen edges of the no-ensemble signal.


# The whole path, on tiny data and small random models


def test_plain_run_reads_the_files_a_real_run_reads(tiny, tmp_path):
    """Everything but the real data: manifest, normaliser, checkpoints, report and selection on disk."""
    stores, normalizer, config = tiny
    root = tmp_path / "data"
    (root / "val").mkdir(parents=True)
    for name in ("fields.npy", "statics.npz"):
        shutil.copy(stores["val"].directory / name, root / "val" / name)
    normalizer.save(root / "normalizer.json")
    files = {name: sha256_file(root / "val" / name) for name in ("fields.npy", "statics.npz")}
    (root / "manifest.json").write_text(
        json.dumps({
            "config": config,
            "normalizer_sha256": sha256_file(root / "normalizer.json"),
            "splits": {"val": {"files": files}},
        })
    )

    def checkpoint(path, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        torch.save({"config": {"model": SMALL_MODEL, "mixed_precision": False}, "model": build_model(SMALL_MODEL).state_dict()}, path)
        return path

    members = [checkpoint(tmp_path / f"m{seed}" / "best.pt", seed) for seed in range(2)]
    (tmp_path / "ensemble.json").write_text(
        json.dumps({"complete": True, "members": [{"checkpoint": str(m), "checkpoint_sha256": sha256_file(m)} for m in members]})
    )
    checkpoint(tmp_path / "runs" / "ablation" / "no_divergence" / "best.pt", 7)

    (tmp_path / "selection.json").write_text(
        json.dumps({
            "choice": {"kind": "threshold", "aggregation": "mean", "handoff_frames": 2, "tau": 0.0},
            "aggregation": {"leader": "mean"},
            "candidates": {"mean": [{"percentile": 50, "tau": 0.0}]},
            "bad_frame_threshold": 0.5,
            "calibration_edges": {"mean": [0.1], "p95": [0.1], "p99": [0.1]},
        })
    )
    settings = {
        "start": 1, "primary_end": 6, "long_end": 7, "n_resamples": 20, "usable_error": 1.0, "rollout_lengths": [1],
        "timing_scenes": 1, "timing_frames": 1, "default_handoff_frames": 2, "random_seeds": [0],
        "ensemble_sizes": [2], "sweep": {"periods": [3], "rates": [0.2]},
    }
    (tmp_path / "eval.yaml").write_text(json.dumps(settings))
    done = subprocess.run(
        [
            sys.executable, "scripts/run_experiments.py", "--config", str(tmp_path / "eval.yaml"),
            "--data-root", str(root), "--ensemble", str(tmp_path / "ensemble.json"),
            "--selection", str(tmp_path / "selection.json"), "--run-root", str(tmp_path / "runs"),
            "--progress", str(tmp_path / "progress"), "--device", "cpu", "--results", str(tmp_path / "results"),
        ],
        cwd=REPO, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    assert "no_pushforward" in done.stdout  # A missing ablation is said to be missing.
    assert not (tmp_path / "results").exists()  # Without --final, nothing lands where results go.
    written = tmp_path / "progress" / "val_results"
    for name in ("accuracy", "doubt", "handoff", "ablation"):
        result = json.loads((written / f"exp_{name}.json").read_text())
        check_provenance(result)
        assert list(result["sets"]) == ["val"] and not result["final"]
    ablation = json.loads((written / "exp_ablation.json").read_text())
    assert ablation["ablations_run"] == ["no_divergence"]


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("dry")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, "scripts/run_experiments.py", "--dry-run", "--out", str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    return out


def test_dry_run_writes_every_result_with_provenance(dry_run):
    for name in ("accuracy", "doubt", "handoff", "ablation"):
        result = json.loads((dry_run / f"exp_{name}.json").read_text())
        check_provenance(result)
        assert result["dry_run"] is True
        assert set(result["sets"]) == {"val", "test_a", "test_b", "test_c"}


def test_dry_run_reports_test_c_per_side_and_pooled(dry_run):
    for name in ("accuracy", "doubt", "handoff"):
        result = json.loads((dry_run / f"exp_{name}.json").read_text())
        assert set(result["sets"]["test_c"]["groups"]) == {"low", "high", "pooled"}
    accuracy = json.loads((dry_run / "exp_accuracy.json").read_text())
    assert "no_contact_scenes" in accuracy["sets"]["test_c"]["groups"]["low"]
    handoff = json.loads((dry_run / "exp_handoff.json").read_text())
    pooled = handoff["sets"]["test_c"]["groups"]["pooled"]
    assert pooled["frozen"]["n_rollouts"] == 4
    assert all(item["descriptive"] for item in handoff["sets"]["val"]["sweep"])


def test_dry_run_handoff_reports_baselines_and_the_oracle(dry_run):
    group = json.loads((dry_run / "exp_handoff.json").read_text())["sets"]["val"]["groups"]["all"]
    assert group["budget"] > 0
    assert group["fixed"] and group["random"]
    for baseline in group["fixed"] + group["random"]:
        assert "achieved_share" in baseline and "status" in baseline
    assert "candidates" in group["oracle"]
    assert "beats" not in group["oracle"]  # The oracle doesn't enter the comparison.


def test_figures_and_results_come_from_a_dry_run(dry_run, tmp_path):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    figures = tmp_path / "figures"
    done = subprocess.run(
        [sys.executable, "scripts/make_figures.py", "--results", str(dry_run), "--out", str(figures)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert len(list(figures.glob("*.png"))) >= 6

    out = tmp_path / "RESULTS.md"
    out.write_text("# Results\n\nMy analysis.\n")
    done = subprocess.run(
        [sys.executable, "scripts/render_results.py", "--results", str(dry_run), "--out", str(out)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    text = out.read_text()
    assert "My analysis." in text and "Dry run" in text and "test_c" in text
    first = text
    subprocess.run(
        [sys.executable, "scripts/render_results.py", "--results", str(dry_run), "--out", str(out)],
        cwd=REPO, env=env, check=True, capture_output=True, timeout=300,
    )
    assert out.read_text() == first  # Rendering replaces its block, and keeps the analysis.


def test_render_results_refuses_missing_provenance(tmp_path):
    (tmp_path / "exp_accuracy.json").write_text(json.dumps({"sets": {}, "git_commit": "abc"}))
    out = tmp_path / "RESULTS.md"
    done = subprocess.run(
        [sys.executable, "scripts/render_results.py", "--results", str(tmp_path), "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode != 0
    assert "provenance" in done.stderr
    assert not out.exists()
