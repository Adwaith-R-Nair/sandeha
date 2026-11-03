"""The ensemble's step from a shared input, its doubt map, and the doubt score of a frame."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from sandeha.models.factory import apply_step

_QUANTILES = {"p95": 0.95, "p99": 0.99}


@dataclass
class EnsembleOut:
    """One ensemble step.

    - ``mean`` ``(B,3,H,W)``: the mean next fields, in physical units.
    - ``doubt_map`` ``(B,1,H,W)``: the members' spread, in units of the field normaliser.
    - ``members`` ``(K,B,3,H,W)``: each member's next fields, in physical units.
    """

    mean: Tensor
    doubt_map: Tensor
    members: Tensor


class Ensemble:
    """Models that all step from one input, so that a rollout feeds every member the shared mean."""

    def __init__(self, models: list[nn.Module], normalizer):
        if not models:
            raise ValueError("an ensemble needs at least one member")
        self.models = list(models)
        self.normalizer = normalizer
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def step(self, x: Tensor) -> EnsembleOut:
        """One step from a model input ``(B,6,H,W)``."""
        members = torch.stack([apply_step(model, x, self.normalizer) for model in self.models])
        # Taken relative to the first member, so equal members give exactly zero without relying on how
        # a backend rounds their mean. The variance is the population's, so one member has zero.
        encoded = self.normalizer.encode(members)
        variance = (encoded - encoded[:1]).var(dim=0, unbiased=False)
        doubt_map = variance.sqrt().mean(dim=1, keepdim=True)
        return EnsembleOut(mean=members.mean(dim=0), doubt_map=doubt_map, members=members)


def doubt_score(doubt_map: Tensor, obstacle: Tensor, method: Literal["mean", "p95", "p99"]) -> Tensor:
    """One score per frame ``(B,)`` from the doubt map over its fluid cells.

    ``obstacle`` ``(B,1,H,W)`` is the mask in force on the way into the frame, nonzero where solid.
    A non-finite fluid cell makes the score positive infinity under every method, as a percentile
    could otherwise pass over it. A frame with no fluid cells has nothing to doubt, and scores zero.
    """
    if method != "mean" and method not in _QUANTILES:
        raise ValueError(f"unknown doubt aggregation {method}")
    if obstacle.shape != doubt_map.shape:
        raise ValueError(f"obstacle {obstacle.shape} doesn't match doubt map {doubt_map.shape}")
    scores = doubt_map.new_zeros(doubt_map.shape[0])
    for i, (frame, mask) in enumerate(zip(doubt_map, obstacle)):
        fluid = frame[mask == 0]
        if fluid.numel() == 0:
            continue
        if not torch.isfinite(fluid).all():
            scores[i] = torch.inf
        elif method == "mean":
            scores[i] = fluid.mean()
        else:
            scores[i] = torch.quantile(fluid, _QUANTILES[method])
    return scores
