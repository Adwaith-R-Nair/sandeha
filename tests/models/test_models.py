import pytest
import torch
from torch import nn

from sandeha.data.dataset import Normalizer
from sandeha.models.factory import apply_step, build_model

CONFIGS = {
    "unet": {"arch": "unet", "base_width": 32, "levels": 4},
    "fno": {"arch": "fno", "width": 48, "modes": 24, "layers": 4},
}


def normalizer() -> Normalizer:
    return Normalizer(
        field_mean=[0.1, 0.2, 0.3],
        field_std=[1.0, 2.0, 3.0],
        delta_std=[0.5, 0.25, 0.125],
        inflow_scale=5.0,
        buoyancy_mean=1.0,
        buoyancy_std=1.0,
        fitted_on="train",
        n_scenes=1,
    )


class Constant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x[:, :3], self.value)


def state_with_wall() -> tuple[torch.Tensor, torch.Tensor]:
    n = normalizer()
    obstacle = torch.zeros(2, 1, 16, 16)
    obstacle[:, :, 4:8, 4:8] = 1
    fields = torch.ones(2, 3, 16, 16)
    return n.make_input(fields, obstacle, torch.zeros_like(obstacle), torch.ones(2)), obstacle


@pytest.mark.parametrize("arch", CONFIGS)
def test_model_output_shape(arch):
    model = build_model(CONFIGS[arch])
    assert model(torch.zeros(2, 6, 128, 128)).shape == (2, 3, 128, 128)


def test_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="resnet"):
        build_model({"arch": "resnet"})


def test_fno_sees_position():
    """The box isn't periodic, so a shifted input must not give just a shifted output."""
    model = build_model({"arch": "fno", "width": 8, "modes": 4, "layers": 1})
    x = torch.randn(1, 6, 16, 16)
    shifted = model(torch.roll(x, 3, dims=-1))
    assert not torch.allclose(torch.roll(model(x), 3, dims=-1), shifted, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fno_runs_under_mixed_precision():
    model = build_model({"arch": "fno", "width": 8, "modes": 4, "layers": 2}).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(torch.zeros(1, 6, 32, 32).cuda()).float().pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_solid_cells_zero_after_apply_step():
    x, obstacle = state_with_wall()
    out = apply_step(Constant(1.0), x, normalizer())
    assert (out[obstacle.expand_as(out) == 1] == 0).all()


def test_apply_step_adds_the_decoded_change_in_physical_units():
    x, obstacle = state_with_wall()
    out = apply_step(Constant(2.0), x, normalizer())
    fluid = obstacle[0, 0] == 0
    # A physical 1, plus 2 times each channel's change scale.
    for channel, expected in enumerate([2.0, 1.5, 1.25]):
        assert torch.allclose(out[0, channel][fluid], torch.tensor(expected))


def test_a_non_finite_prediction_stays_out_of_solid_cells_only():
    x, obstacle = state_with_wall()
    out = apply_step(Constant(float("nan")), x, normalizer())
    solid = obstacle.expand_as(out) == 1
    assert (out[solid] == 0).all()
    assert out[~solid].isnan().all()


def test_apply_step_returns_full_precision_under_autocast():
    class Half(Constant):
        def forward(self, x):
            return super().forward(x).half()

    x, _ = state_with_wall()
    assert apply_step(Half(1.0), x, normalizer()).dtype == torch.float32
