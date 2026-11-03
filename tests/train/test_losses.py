import torch

from sandeha.train.losses import divergence, step_loss


def grid(n: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
    return x, y


def clear(n: int = 16) -> torch.Tensor:
    return torch.zeros(1, 1, n, n)


def test_divergence_of_uniform_flow_is_zero():
    velocity = torch.stack([torch.full((16, 16), 2.0), torch.full((16, 16), -3.0)])[None]
    obstacle = clear()
    obstacle[:, :, 5:9, 5:9] = 1
    assert (divergence(velocity, obstacle) == 0).all()


def test_divergence_known_answer_and_axes():
    """Index ``(row, column)`` is ``(y, x)``, so ``u`` varies along columns and ``v`` along rows."""
    x, y = grid()
    outward = divergence(torch.stack([x, y])[None], clear())
    assert outward.shape == (1, 1, 16, 16)
    assert torch.allclose(outward[:, :, 1:-1, 1:-1], torch.tensor(2.0))
    assert (divergence(torch.stack([y, x])[None], clear()) == 0).all()


def test_divergence_leaves_out_solids_and_their_neighbours():
    x, y = grid()
    obstacle = clear()
    obstacle[:, :, 8, 8] = 1
    result = divergence(torch.stack([x, y])[None], obstacle)
    assert result[0, 0, 8, 8] == 0
    assert result[0, 0, 8, 7] == 0  # Its stencil reaches into the solid.
    assert result[0, 0, 4, 4] == 2


def test_step_loss_is_mse_on_fluid_cells():
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    obstacle = torch.zeros(1, 1, 8, 8)
    obstacle[:, :, :4] = 1
    target[:, :, :4] = 100.0  # Solid: must not count.
    target[:, :, 4:] = 2.0
    out = step_loss(pred, target, obstacle, lambda_div=0.0)
    assert torch.allclose(out["mse"], torch.tensor(4.0))
    assert torch.allclose(out["loss"], torch.tensor(4.0))


def test_step_loss_adds_weighted_divergence_and_reports_the_target_floor():
    x, y = grid()
    outward = torch.stack([x, y, torch.zeros_like(x)])[None]
    still = torch.zeros_like(outward)
    out = step_loss(outward, outward, clear(), lambda_div=0.5)
    assert out["mse"] == 0
    assert torch.allclose(out["div"], torch.tensor(4.0))
    assert torch.allclose(out["div_target"], torch.tensor(4.0))
    assert torch.allclose(out["loss"], torch.tensor(2.0))
    assert step_loss(still, outward, clear(), lambda_div=0.5)["div"] == 0


def test_step_loss_measures_divergence_after_undoing_channel_scales():
    """``u`` and ``v`` have different scales, and only the physical change has true divergence."""
    x, y = grid()
    # A physical change of (x, -y), which has no divergence, scaled by 2 and 0.5.
    scaled = torch.stack([x / 2.0, -y / 0.5, torch.zeros_like(x)])[None]
    assert step_loss(scaled, scaled, clear(), 0.1)["div"] > 1
    out = step_loss(scaled, scaled, clear(), 0.1, velocity_scale=(2.0, 0.5))
    assert out["div"] == 0
