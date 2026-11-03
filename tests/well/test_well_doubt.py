"""The section 6.2 measures on known values, and the evidence rollouts."""

import math

import pytest
import torch
from torch import nn

from sandeha.well.data import WellNormalizer
from sandeha.well.doubt import (
    auroc,
    calibration,
    evidence_measure,
    evidence_rollouts,
    risk_coverage,
    single_rollout,
    spearman,
    within_time_auroc,
)

INF = math.inf


def test_spearman_known():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1)  # Ties share a rank.


def test_spearman_undefined_when_tied():
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([INF, INF, INF], [1, 2, 3]) is None


def test_non_finite_score_ranks_highest():
    assert spearman([float("nan"), 1, 2], [10, 1, 2]) == pytest.approx(1)
    assert auroc([float("nan"), 0.1, 0.2], [True, False, False]) == 1


def test_auroc_known():
    assert auroc([3, 2, 1, 0], [True, True, False, False]) == 1
    assert auroc([0, 1, 2, 3], [True, True, False, False]) == 0
    assert auroc([1, 1, 1, 1], [True, True, False, False]) == 0.5  # Ties count as half.
    assert auroc([1, 2], [True, True]) is None


def test_evidence_measure_counts_undefined_times():
    # Scores agree with errors at time 0, and every error is tied at time 1.
    scores = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    errors = torch.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    out = evidence_measure(scores, errors, bootstrap=20, seed=0)
    assert out["value"] == pytest.approx(1)
    assert out["undefined_times"] == 1
    assert out["ci95"][0] <= 1.0001


def test_evidence_measure_with_no_time_left_has_no_figure():
    out = evidence_measure(torch.ones(4, 2), torch.ones(4, 2), bootstrap=5, seed=0)
    assert out["value"] is None and out["ci95"] is None and out["undefined_times"] == 2


def test_within_time_auroc_leaves_out_thin_times():
    scores = torch.arange(12.0)[:, None].repeat(1, 2)
    bad = torch.zeros(12, 2, dtype=torch.bool)
    bad[6:, 0] = True  # Six of each at time 0, and one bad frame at time 1.
    bad[11, 1] = True
    out = within_time_auroc(scores, bad, min_each_class=5)
    assert out["value"] == 1
    assert out["excluded_times"] == 1
    assert out["times_no_bad_or_all_bad"] == 0


def test_within_time_auroc_says_when_no_time_qualifies():
    out = within_time_auroc(torch.arange(9.0)[:, None], torch.arange(9)[:, None] > 4, min_each_class=5)
    assert out["value"] is None and out["excluded_times"] == 1


def test_calibration_merges_equal_edges_and_keeps_non_finite():
    scores = torch.tensor([[0.5], [1.5], [2.5], [INF]])
    errors = torch.tensor([[1.0], [2.0], [3.0], [10.0]])
    bins = calibration(scores, errors, [1.0, 1.0, 2.0], bad_threshold=2.5)
    assert [b["frames"] for b in bins] == [1, 1, 2]
    assert bins[-1]["mean_error"] == pytest.approx(6.5)
    assert bins[-1]["bad_share"] == pytest.approx(1)


def test_risk_coverage_keeps_ties_together():
    scores = torch.tensor([[1.0], [1.0], [2.0], [INF]])
    errors = torch.tensor([[1.0], [3.0], [5.0], [10.0]])
    out = risk_coverage(scores, errors)
    assert out["coverage"][0] == pytest.approx(0.1)
    assert out["pooled"]["achieved"][0] == pytest.approx(0.5)
    assert out["pooled"]["mean_error"][0] == pytest.approx(2)
    assert out["pooled"]["mean_error"][-1] == pytest.approx(19 / 4)
    assert out["within_time"]["mean_error"][-1] == pytest.approx(19 / 4)


class _Blowup(nn.Module):
    def forward(self, x):
        return torch.full_like(x, float("nan"))


class _Constant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


class _Frames:
    """A split of constant-in-time zeros."""

    n_trajectories, n_steps, grid = 3, 8, (6, 6)

    def read(self, trajectory, start, stop):
        return torch.zeros(stop - start, 4, 6, 6).numpy()


def _unit_normalizer():
    return WellNormalizer([0.0] * 4, [1.0] * 4, [1.0] * 4, "train", 3)


def test_shared_mean_rollout_has_known_doubt_and_error():
    models = [_Constant(-1.0), _Constant(1.0)]
    out = evidence_rollouts(models, _Frames(), _unit_normalizer(), start=1, steps=2, batch_size=2)
    assert out["errors"].shape == (3, 2)
    assert out["errors"].abs().max() == 0  # The members' mean is no change.
    for name in ("mean", "p95", "p99"):
        assert torch.allclose(out["scores"][name], torch.ones(3, 2))  # std of -1 and +1, population.


def test_blown_up_rollout_counts_as_max_error_and_stays():
    out = evidence_rollouts([_Blowup(), _Blowup()], _Frames(), _unit_normalizer(), start=0, steps=3, batch_size=2)
    assert out["errors"].shape == (3, 3)
    assert (out["errors"] == 10).all()
    assert torch.isinf(out["scores"]["mean"]).all()


def test_single_rollout_gives_divergence_and_error():
    out = single_rollout(_Constant(0.0), _Frames(), _unit_normalizer(), start=0, steps=2, batch_size=2)
    assert out["errors"].shape == (3, 2)
    assert (out["scores"] == 0).all()
    blown = single_rollout(_Blowup(), _Frames(), _unit_normalizer(), start=0, steps=2, batch_size=2)
    assert (blown["errors"] == 10).all() and torch.isinf(blown["scores"]).all()


class _Ramp(nn.Module):
    """Sets one velocity channel to the cell's index along one axis."""

    def __init__(self, channel: int, axis: int):
        super().__init__()
        self.channel, self.axis = channel, axis

    def forward(self, x):
        out = torch.zeros_like(x)
        shape = [1, 1, 1, 1]
        shape[self.axis] = -1
        out[:, self.channel] = torch.arange(x.shape[self.axis], dtype=x.dtype).view(shape)[:, 0]
        return out


@pytest.mark.parametrize("channel, axis", [(2, 2), (3, 3)])
def test_divergence_differentiates_each_velocity_along_its_own_axis(channel, axis):
    # velocity_x along x, the first grid axis, and velocity_y along y, the second.
    out = single_rollout(_Ramp(channel, axis), _Frames(), _unit_normalizer(), start=0, steps=1, batch_size=3)
    assert out["scores"] == pytest.approx(torch.ones(3, 1))


@pytest.mark.parametrize("channel, axis", [(2, 3), (3, 2)])
def test_divergence_ignores_shear(channel, axis):
    out = single_rollout(_Ramp(channel, axis), _Frames(), _unit_normalizer(), start=0, steps=1, batch_size=3)
    assert (out["scores"] == 0).all()
