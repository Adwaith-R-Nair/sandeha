"""Building a model from its config, and taking one step with it."""

import torch
from torch import Tensor, nn

from sandeha.models.fno import FNO2d
from sandeha.models.unet import UNet


def build_model(cfg: dict) -> nn.Module:
    """A model taking ``(B,6,H,W)`` to ``(B,3,H,W)``, the normalised change over one step."""
    settings = dict(cfg)
    arch = settings.pop("arch")
    if arch == "unet":
        return UNet(**settings)
    if arch == "fno":
        return FNO2d(**settings)
    raise ValueError(f"unknown architecture {arch}")


def apply_step(model: nn.Module, x: Tensor, normalizer) -> Tensor:
    """The next physical fields ``(B,3,H,W)`` from a model input, with solid cells forced to zero.

    Solid cells are filled rather than multiplied, so that a blown-up prediction can't put a NaN
    inside a solid, and the mask is the one in the input.
    """
    current = normalizer.decode(x[:, :3])
    # Under mixed precision the model's output is half precision, far too coarse for the state.
    change = normalizer.decode_delta(model(x).float())
    return (current + change).masked_fill(x[:, 3:4] != 0, 0.0)
