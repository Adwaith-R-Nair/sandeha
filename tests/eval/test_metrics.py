"""Known answers for the metrics, by the rules in docs/PREREGISTRATION.md."""

import math

import numpy as np
import pytest

from sandeha.eval.metrics import (
    auroc,
    bootstrap_ci,
    calibration,
    evidence_measure,
    frame_secondary,
    has_contact,
    join_summaries,
    nrmse,
    rise_on_unfamiliar,
    risk_coverage,
    side_of,
    spearman,
    usable_frames,
    window_summary,
)

INF = float("inf")
NAN = float("nan")


def test_nrmse_zero_for_identical():
    ref = np.random.default_rng(0).normal(size=(3, 8, 8))
    solid = np.zeros((8, 8), bool)
    assert nrmse(ref, ref, solid) == 0.0


def test_nrmse_counts_fluid_cells_only_and_caps_non_finite():
    ref = np.random.default_rng(0).normal(size=(3, 8, 8))
    solid = np.zeros((8, 8), bool)
    solid[0, 0] = True
    pred = ref.copy()
    pred[:, 0, 0] = np.nan
    assert nrmse(pred, ref, solid) == 0.0
    pred[0, 3, 3] = np.nan
    assert nrmse(pred, ref, solid) == 10.0


def test_spearman_known():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1)


def test_auroc_known():
    assert auroc([3, 2, 1, 0], [True, True, False, False]) == 1
    assert auroc([0, 1, 2, 3], [True, True, False, False]) == 0
    assert auroc([1, 1, 1, 1], [True, True, False, False]) == 0.5


def test_bootstrap_contains_mean():
    values = np.random.default_rng(1).normal(size=30)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high


def test_bootstrap_is_paired_by_shared_resamples():
    a = np.random.default_rng(2).normal(size=20)
    assert bootstrap_ci(a, n=200) == bootstrap_ci(a, n=200)
    low, high = bootstrap_ci(a - a, n=200)
    assert low == high == 0


def test_non_finite_doubt_ranks_highest():
    scores = np.array([[0.1], [0.2], [0.3], [NAN], [INF]])
    errors = np.array([[1.0], [2.0], [3.0], [10.0], [10.0]])

    assert spearman(scores[:, 0], errors[:, 0]) == pytest.approx(1)
    assert auroc(scores[:, 0], errors[:, 0] >= 10) == 1

    bins = calibration(scores, errors, [0.15, 0.25], bad_threshold=5.0)
    assert sum(item["frames"] for item in bins) == 5  # Nothing is dropped.
    assert bins[-1]["frames"] == 3
    assert bins[-1]["bad_share"] == pytest.approx(2 / 3)

    curve = risk_coverage(scores, errors)
    assert curve["pooled"]["achieved"][-1] == 1
    assert curve["pooled"]["mean_error"][-1] == pytest.approx(errors.mean())
    # The two non-finite frames tie, so they are kept or left out together.
    assert curve["pooled"]["achieved"][5] == 0.6
    assert curve["pooled"]["achieved"][6] == 1
    assert curve["pooled"]["mean_error"][0] == 1.0


def test_undefined_time_is_left_out_and_counted():
    scores = np.array([[1.0, 5.0, 1.0], [2.0, 5.0, 2.0], [3.0, 5.0, 3.0]])
    errors = np.array([[1.0, 1.0, 10.0], [2.0, 2.0, 10.0], [3.0, 3.0, 10.0]])
    found = evidence_measure(scores, errors, 20, 0)
    assert found["value"] == pytest.approx(1)
    assert found["undefined_times"] == 2
    assert found["times"] == 3

    nothing = evidence_measure(scores[:, 1:], errors[:, 1:], 20, 0)
    assert nothing["value"] is None
    assert nothing["ci95"] is None


def test_no_contact_count_uses_the_coarse_rule():
    density = np.zeros((4, 8, 8))
    masks = np.zeros((4, 8, 8), bool)
    masks[:] = False
    masks[:, 4, 4] = True
    # Smoke on the box's edge and far from the solid touches nothing.
    density[1, 0, 0] = 1.0
    assert not has_contact(density, masks)
    # Thin smoke beside the solid is below the 0.01 of the rule.
    density[2, 3, 3] = 0.01
    assert not has_contact(density, masks)
    # Smoke inside the solid isn't a cell of fluid beside it.
    density[2, 4, 4] = 1.0
    assert not has_contact(density, masks)
    # Diagonal counts, since the neighbourhood is 3x3, and the mask is the one on the way in.
    density[2, 3, 3] = 0.02
    assert has_contact(density, masks)
    density[2, 3, 3] = 0.0
    density[3, 5, 5] = 0.5
    masks[2] = False  # Frame 3 is judged by the mask on the way into it, index 2.
    assert not has_contact(density, masks)
    masks[2, 4, 4] = True
    assert has_contact(density, masks)
    # The box's own walls don't count, however close smoke comes to the edge.
    edge = np.zeros((2, 8, 8))
    edge[1, 0, 3] = 1.0
    assert not has_contact(edge, np.zeros((2, 8, 8), bool))


def test_rise_counts_ties_as_half():
    familiar = np.array([[1.0, 1.0], [2.0, 2.0]])
    other = np.array([[3.0, 1.0], [3.0, 2.0]])
    found = rise_on_unfamiliar(other, familiar, 50, 0)
    # First time: always higher. Second: 1 v 1 half, 1 v 2 no, 2 v 1 yes, 2 v 2 half: 0.5.
    assert found["value"] == pytest.approx(0.75)
    assert found["ci95"] is not None


def test_rise_ranks_non_finite_doubt_highest():
    assert rise_on_unfamiliar(np.array([[NAN]]), np.array([[1e9]]), 10, 0)["value"] == 1
    assert rise_on_unfamiliar(np.array([[NAN]]), np.array([[INF]]), 10, 0)["value"] == 0.5


def test_usable_frames_stop_at_the_first_crossing():
    assert usable_frames([0.1, 0.2, 2.0, 0.1], 1.0) == 2
    assert usable_frames([0.1, 0.2], 1.0) == 2


def test_side_comes_from_buoyancy():
    assert side_of(0.5) == "low"
    assert side_of(4.0) == "high"


def test_frame_secondary_on_identical_fields():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 32, 32))
    ref[2] = np.abs(ref[2])
    solid = np.zeros((32, 32), bool)
    found = frame_secondary(ref, ref, solid)
    assert found["vorticity_correlation"] == pytest.approx(1)
    assert found["total_smoke_ratio"] == pytest.approx(1)
    assert found["centroid_difference"] == pytest.approx(0)
    assert found["top_difference"] == pytest.approx(0)
    assert all(value == pytest.approx(0) for value in found["spectrum_log_ratio"])
    assert found["divergence_rms"] > 0
    assert found["divergence_max"] >= found["divergence_rms"]


def test_frame_secondary_of_a_blown_up_frame_keeps_it_in_the_average():
    ref = np.random.default_rng(0).normal(size=(3, 16, 16))
    pred = np.full_like(ref, np.nan)
    solid = np.zeros((16, 16), bool)
    found = frame_secondary(pred, ref, solid)
    assert found["vorticity_correlation"] == 0.0  # A blown-up frame can't flatter this average.
    assert found["divergence_rms"] == INF
    assert math.isnan(found["total_smoke_ratio"])

    summary = window_summary([found, frame_secondary(ref, ref, solid)])
    assert summary["vorticity_correlation"]["mean"] == pytest.approx(0.5)
    assert summary["divergence_rms"]["non_finite_frames"] == 1


def test_joined_summaries_match_summarising_together():
    rng = np.random.default_rng(1)
    solid = np.zeros((16, 16), bool)
    ref = rng.normal(size=(3, 16, 16))
    frames = [frame_secondary(rng.normal(size=(3, 16, 16)), ref, solid) for _ in range(3)]
    frames.append(frame_secondary(np.full_like(ref, np.nan), ref, solid))
    together = window_summary(frames)
    joined = join_summaries(window_summary(frames[:1]), window_summary(frames[1:]))
    assert joined["divergence_rms"]["mean"] == pytest.approx(together["divergence_rms"]["mean"])
    assert joined["spectrum_log_ratio"]["mean"] == pytest.approx(together["spectrum_log_ratio"]["mean"])
    assert joined["divergence_rms"]["non_finite_frames"] == 1
