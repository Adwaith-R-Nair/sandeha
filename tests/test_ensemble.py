"""The ensemble's shared-input step, its doubt map, and the doubt score of a frame."""

import pytest
import torch
from torch import nn

from sandeha.data.dataset import Normalizer
from sandeha.ensemble import Ensemble, doubt_score
from sandeha.models.factory import apply_step, build_model


@pytest.fixture
def normalizer() -> Normalizer:
    return Normalizer(
        field_mean=[0.1, 0.2, 0.3],
        field_std=[1.0, 2.0, 4.0],
        delta_std=[0.5, 0.25, 0.125],
        inflow_scale=5.0,
        buoyancy_mean=1.0,
        buoyancy_std=1.0,
        fitted_on="train",
        n_scenes=1,
    )


class Constant(nn.Module):
    """A member whose predicted change, in normalised delta units, is the same everywhere."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 3, *x.shape[2:]), self.value)


def model_input(obstacle=None, batch: int = 2, size: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(batch, 6, size, size)
    x[:, 3] = 0.0 if obstacle is None else obstacle
    return x


def test_identical_members_zero_doubt(normalizer):
    # Five equal values: their explicit mean can round, and doubt must still be exactly zero.
    out = Ensemble([Constant(0.1) for _ in range(5)], normalizer).step(model_input())
    assert out.doubt_map.shape == (2, 1, 8, 8)
    assert (out.doubt_map == 0).all()


def test_identical_real_models_zero_doubt(normalizer):
    model = build_model({"arch": "unet", "base_width": 8, "levels": 2, "groups": 4})
    out = Ensemble([model, model, model], normalizer).step(model_input())
    assert out.mean.shape == (2, 3, 8, 8)
    assert (out.doubt_map == 0).all()


def test_disagreement_raises_doubt(normalizer):
    x = model_input()
    close = Ensemble([Constant(0.0), Constant(0.1)], normalizer).step(x).doubt_map
    far = Ensemble([Constant(0.0), Constant(1.0)], normalizer).step(x).doubt_map
    assert (close > 0).all()
    assert (far > close).all()


def test_doubt_is_the_mean_over_channels_of_each_normalised_std(normalizer):
    out = Ensemble([Constant(-1.0), Constant(1.0)], normalizer).step(model_input())
    # A change of plus or minus one delta_std gives a population std of delta_std in physical units,
    # which is then expressed in units of field_std.
    expected = (0.5 / 1.0 + 0.25 / 2.0 + 0.125 / 4.0) / 3
    assert out.doubt_map == pytest.approx(torch.full((2, 1, 8, 8), expected), abs=1e-5)


def test_mean_matches_manual_mean(normalizer):
    x = model_input()
    models = [build_model({"arch": "unet", "base_width": 8, "levels": 2, "groups": 4}) for _ in range(3)]
    out = Ensemble(models, normalizer).step(x)
    with torch.no_grad():
        manual = torch.stack([apply_step(model, x, normalizer) for model in models])
    assert out.members.shape == (3, 2, 3, 8, 8)
    assert torch.equal(out.members, manual)
    assert torch.equal(out.mean, manual.mean(dim=0))
    assert not out.mean.requires_grad


def test_every_member_steps_from_the_same_input(normalizer):
    seen = []

    class Recording(Constant):
        def forward(self, x):
            seen.append(x.clone())
            return super().forward(x)

    x = model_input()
    Ensemble([Recording(0.0), Recording(1.0)], normalizer).step(x)
    assert len(seen) == 2
    assert torch.equal(seen[0], x) and torch.equal(seen[1], x)


def test_score_ignores_solid_cells():
    doubt_map = torch.ones(2, 1, 8, 8)
    solid = torch.zeros(2, 1, 8, 8)
    solid[:, :, :4] = 1
    doubt_map[solid != 0] = 1e6
    for method in ("mean", "p95", "p99"):
        assert doubt_score(doubt_map, solid, method).tolist() == [1.0, 1.0]


def test_score_aggregations_have_known_values():
    doubt_map = torch.arange(101, dtype=torch.float32).view(1, 1, 1, 101)
    fluid = torch.zeros_like(doubt_map)
    assert doubt_score(doubt_map, fluid, "mean").item() == pytest.approx(50.0)
    assert doubt_score(doubt_map, fluid, "p95").item() == pytest.approx(95.0)
    assert doubt_score(doubt_map, fluid, "p99").item() == pytest.approx(99.0)


def test_scores_are_per_frame_with_different_masks():
    doubt_map = torch.stack([torch.full((1, 4, 4), 2.0), torch.full((1, 4, 4), 3.0)])
    solid = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    solid[1, 0, :3] = True
    assert doubt_score(doubt_map, solid, "p95").tolist() == [2.0, 3.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("method", ["mean", "p95", "p99"])
def test_one_non_finite_fluid_cell_makes_the_score_infinite(method, bad):
    doubt_map = torch.ones(2, 1, 8, 8)
    doubt_map[0, 0, 3, 3] = bad
    score = doubt_score(doubt_map, torch.zeros(2, 1, 8, 8), method)
    assert score[0].item() == float("inf")
    assert score[1].item() == 1.0


def test_non_finite_in_a_solid_cell_is_ignored():
    doubt_map = torch.ones(1, 1, 8, 8)
    doubt_map[0, 0, 3, 3] = float("nan")
    solid = torch.zeros(1, 1, 8, 8)
    solid[0, 0, 3, 3] = 1
    assert doubt_score(doubt_map, solid, "p99").item() == 1.0


def test_blown_up_member_gives_non_finite_score_and_keeps_solids_at_zero(normalizer):
    solid = torch.zeros(8, 8)
    solid[:2] = 1
    x = model_input(solid)
    out = Ensemble([Constant(0.0), Constant(float("nan"))], normalizer).step(x)
    assert (out.doubt_map[:, :, :2] == 0).all()
    assert (doubt_score(out.doubt_map, x[:, 3:4], "mean") == float("inf")).all()


@pytest.mark.parametrize("method", ["mean", "p95", "p99"])
def test_frame_with_no_fluid_cells_scores_zero(method):
    score = doubt_score(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), method)
    assert score.tolist() == [0.0]


def test_one_member_doubt_is_zero(normalizer):
    x = model_input()
    model = Constant(0.3)
    out = Ensemble([model], normalizer).step(x)
    assert (out.doubt_map == 0).all()
    assert torch.equal(out.mean, apply_step(model, x, normalizer))
    assert (doubt_score(out.doubt_map, x[:, 3:4], "p99") == 0).all()


def test_no_members_and_unknown_method_are_refused(normalizer):
    with pytest.raises(ValueError):
        Ensemble([], normalizer)
    with pytest.raises(ValueError):
        doubt_score(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), "max")


def test_mask_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        doubt_score(torch.ones(2, 1, 4, 4), torch.zeros(1, 4, 4), "mean")


def test_members_are_put_in_evaluation_mode(normalizer):
    model = Constant(0.0)
    assert model.training
    Ensemble([model], normalizer)
    assert not model.training
