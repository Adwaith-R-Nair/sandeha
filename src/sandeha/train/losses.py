"""The one-step training loss."""

import torch
from torch import Tensor


def _stencil(obstacle: Tensor) -> Tensor:
    """Cells whose four neighbours and own cell are fluid and inside the box, ``(B,1,H,W)``."""
    fluid = obstacle == 0
    inside = torch.zeros_like(fluid)
    inside[..., 1:-1, 1:-1] = (
        fluid[..., 1:-1, 1:-1]
        & fluid[..., 1:-1, 2:]
        & fluid[..., 1:-1, :-2]
        & fluid[..., 2:, 1:-1]
        & fluid[..., :-2, 1:-1]
    )
    return inside


def divergence(velocity: Tensor, obstacle: Tensor) -> Tensor:
    """Central differences of ``(B,2,H,W)`` in grid units, ``(B,1,H,W)``, zero outside fluid.

    Only cells with their whole stencil in fluid are measured, since a difference across a wall or
    the box's edge says nothing about the flow. This is not the solver's divergence on its faces:
    a projected field has a nonzero floor here (``docs/LLD.md``).
    """
    du = (velocity[:, 0, 1:-1, 2:] - velocity[:, 0, 1:-1, :-2]) / 2
    dv = (velocity[:, 1, 2:, 1:-1] - velocity[:, 1, :-2, 1:-1]) / 2
    result = torch.zeros_like(velocity[:, :1])
    result[:, 0, 1:-1, 1:-1] = du + dv
    return result.masked_fill(~_stencil(obstacle), 0.0)


def _mean_squared_divergence(delta: Tensor, obstacle: Tensor, velocity_scale: tuple[float, float]) -> Tensor:
    scale = delta.new_tensor(velocity_scale).view(1, 2, 1, 1)
    measured = divergence(delta[:, :2] * scale, obstacle)
    return measured.square().sum() / _stencil(obstacle).sum().clamp(min=1)


def step_loss(
    pred: Tensor,
    target: Tensor,
    obstacle: Tensor,
    lambda_div: float,
    velocity_scale: tuple[float, float] = (1.0, 1.0),
) -> dict:
    """MSE of the normalised change on fluid cells, plus the divergence of the predicted change.

    ``pred`` and ``target`` are ``(B,3,H,W)``. ``velocity_scale`` undoes the different scales of
    ``u`` and ``v``, since only the change in physical proportion has a true divergence.
    ``div_target`` is the same measure on the target: the floor that ``div`` can't sensibly go below.
    """
    fluid = obstacle == 0
    mse = ((pred - target).square() * fluid).sum() / (fluid.sum().clamp(min=1) * pred.shape[1])
    div = _mean_squared_divergence(pred, obstacle, velocity_scale)
    with torch.no_grad():
        div_target = _mean_squared_divergence(target, obstacle, velocity_scale)
    return {"loss": mse + lambda_div * div, "mse": mse, "div": div, "div_target": div_target}
