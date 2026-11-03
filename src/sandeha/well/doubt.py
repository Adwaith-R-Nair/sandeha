"""Evidence rollouts on the public dataset, and the measures of design section 6.2.

The rules are those of ``docs/PREREGISTRATION.md``: every member steps from the shared mean, a
blown-up frame counts as the cap of 10 and stays, and a non-finite score ranks above all finite
ones and ties with other non-finite ones. The figure that counts as evidence is the within-time
Spearman correlation.
"""

import numpy as np
import torch
from torch import Tensor, nn

from sandeha.train.losses import divergence
from sandeha.train.loop import ERROR_CAP, amp_context
from sandeha.well.data import WellNormalizer

AGGREGATIONS = ("mean", "p95", "p99")
COVERAGES = tuple(range(1, 11))  # In tenths.


def well_step(model: nn.Module, x: Tensor, normalizer: WellNormalizer) -> Tensor:
    """The next physical frames from normalised ones, with the model predicting the change."""
    return normalizer.decode(x) + normalizer.decode_delta(model(x).float())


def _capped(errors: Tensor) -> Tensor:
    return torch.nan_to_num(errors, nan=ERROR_CAP, posinf=ERROR_CAP).clamp(max=ERROR_CAP)


def _frame_error(state: Tensor, truth: Tensor, normalizer: WellNormalizer) -> Tensor:
    squares = (normalizer.encode(state) - normalizer.encode(truth)).square()
    return _capped(squares.mean(dim=(1, 2, 3)).sqrt())


def _true_frames(split, first: int, last: int, start: int, stop: int, device: str) -> Tensor:
    frames = np.stack([split.read(i, start, stop) for i in range(first, last)])
    return torch.from_numpy(frames).to(device)


def _flat_non_finite(doubt: Tensor) -> tuple[Tensor, Tensor]:
    flat = doubt.flatten(1)
    return torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0), ~torch.isfinite(flat).all(dim=1)


def _doubt_scores(doubt_map: Tensor) -> dict[str, Tensor]:
    flat, broken = _flat_non_finite(doubt_map)
    scores = {
        "mean": flat.mean(dim=1),
        "p95": torch.quantile(flat, 0.95, dim=1),
        "p99": torch.quantile(flat, 0.99, dim=1),
    }
    return {name: score.masked_fill(broken, torch.inf) for name, score in scores.items()}


@torch.no_grad()
def evidence_rollouts(
    models: list[nn.Module],
    split,
    normalizer: WellNormalizer,
    start: int,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    amp: bool = False,
) -> dict:
    """Shared-mean rollouts of every trajectory from frame ``start``.

    Returns ``errors`` and each aggregation's ``scores``, all ``(trajectories, steps)``, where
    column ``k`` is frame ``start + k + 1``. The doubt map is the square root of the per-channel
    member variance, in normalised units, averaged across channels, as in ``sandeha.ensemble``.
    """
    errors, scores = [], {name: [] for name in AGGREGATIONS}
    for first in range(0, split.n_trajectories, batch_size):
        last = min(first + batch_size, split.n_trajectories)
        frames = _true_frames(split, first, last, start, start + steps + 1, device)
        state = frames[:, 0]
        batch_errors, batch_scores = [], {name: [] for name in AGGREGATIONS}
        for k in range(steps):
            x = normalizer.encode(state)
            with amp_context(device, amp):
                members = torch.stack([well_step(model, x, normalizer) for model in models])
            state = members.mean(dim=0)
            doubt_map = normalizer.encode(members).var(dim=0, unbiased=False).sqrt().mean(dim=1)
            batch_errors.append(_frame_error(state, frames[:, k + 1], normalizer).cpu())
            for name, score in _doubt_scores(doubt_map).items():
                batch_scores[name].append(score.cpu())
        errors.append(torch.stack(batch_errors, dim=1))
        for name in AGGREGATIONS:
            scores[name].append(torch.stack(batch_scores[name], dim=1))
    return {"errors": torch.cat(errors), "scores": {name: torch.cat(scores[name]) for name in AGGREGATIONS}}


@torch.no_grad()
def single_rollout(
    model: nn.Module,
    split,
    normalizer: WellNormalizer,
    start: int,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    amp: bool = False,
) -> dict:
    """One model's rollout: ``errors``, and ``scores``, the RMS divergence of its output velocity."""
    errors, scores = [], []
    for first in range(0, split.n_trajectories, batch_size):
        last = min(first + batch_size, split.n_trajectories)
        frames = _true_frames(split, first, last, start, start + steps + 1, device)
        state = frames[:, 0]
        batch_errors, batch_scores = [], []
        for k in range(steps):
            with amp_context(device, amp):
                state = well_step(model, normalizer.encode(state), normalizer)
            batch_errors.append(_frame_error(state, frames[:, k + 1], normalizer).cpu())
            # ``divergence`` differentiates its first channel along the last axis, which here is y.
            velocity = state[:, [3, 2]]
            measured = divergence(velocity, torch.zeros_like(velocity[:, :1]))
            interior = (velocity.shape[-2] - 2) * (velocity.shape[-1] - 2)
            flat, broken = _flat_non_finite(measured)
            broken |= _flat_non_finite(velocity)[1]
            rms = (flat.square().sum(dim=1) / interior).sqrt()
            batch_scores.append(rms.masked_fill(broken, torch.inf).cpu())
        errors.append(torch.stack(batch_errors, dim=1))
        scores.append(torch.stack(batch_scores, dim=1))
    return {"errors": torch.cat(errors), "scores": torch.cat(scores)}


def _as_ranked(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array), array, np.inf)


def _average_ranks(array: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2)[inverse]


def spearman(x, y) -> float | None:
    """The rank correlation with tied values sharing a rank, or None if either side is all tied."""
    x, y = _as_ranked(x), _as_ranked(y)
    if (x == x[0]).all() or (y == y[0]).all():
        return None
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def auroc(score, label) -> float | None:
    """The chance a positive outranks a negative, ties as half, or None without both classes."""
    score, label = _as_ranked(score), np.asarray(label, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _mean_within_time(scores: np.ndarray, errors: np.ndarray) -> tuple[float | None, int]:
    values = [spearman(scores[:, t], errors[:, t]) for t in range(scores.shape[1])]
    defined = [value for value in values if value is not None]
    return (float(np.mean(defined)) if defined else None), len(values) - len(defined)


def evidence_measure(scores: Tensor, errors: Tensor, bootstrap: int, seed: int) -> dict:
    """The mean over times of the correlation across trajectories, with a 95% interval.

    Trajectories are resampled. A time with no defined correlation leaves the mean, in that
    resample too, and a resample with no time left leaves the interval.
    """
    scores, errors = np.asarray(scores), np.asarray(errors)
    value, undefined = _mean_within_time(scores, errors)
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(bootstrap):
        chosen = rng.integers(0, len(scores), len(scores))
        figure, _ = _mean_within_time(scores[chosen], errors[chosen])
        if figure is not None:
            resampled.append(figure)
    interval = None
    if value is not None and resampled:
        interval = [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))]
    return {
        "value": value,
        "ci95": interval,
        "undefined_times": undefined,
        "times": scores.shape[1],
        "resamples_with_no_figure": bootstrap - len(resampled),
    }


def is_bad(errors: Tensor, threshold: float) -> Tensor:
    """Above the threshold, or at the cap whatever the threshold."""
    return (errors > threshold) | (errors >= ERROR_CAP)


def within_time_auroc(scores: Tensor, bad: Tensor, min_each_class: int) -> dict:
    """The mean AUROC over times with enough of each class, and how many times were left out."""
    values, excluded, one_class = [], 0, 0
    for t in range(scores.shape[1]):
        n_bad = int(bad[:, t].sum())
        n_good = len(bad) - n_bad
        one_class += n_bad == 0 or n_good == 0
        if n_bad < min_each_class or n_good < min_each_class:
            excluded += 1
        else:
            values.append(auroc(scores[:, t], bad[:, t]))
    return {
        "value": float(np.mean(values)) if values else None,
        "excluded_times": excluded,
        "times_no_bad_or_all_bad": one_class,
    }


def decile_edges(scores: Tensor) -> list[float]:
    """The deciles of finite and non-finite scores together. An edge that isn't finite is left out."""
    ranked = _as_ranked(scores).ravel()
    with np.errstate(invalid="ignore"):
        edges = np.percentile(ranked, range(10, 100, 10))
    return [float(edge) for edge in edges if np.isfinite(edge)]


def calibration(scores: Tensor, errors: Tensor, edges: list[float], bad_threshold: float) -> list[dict]:
    """Frames, mean error and share of bad frames in bins set by frozen edges, equal edges merged."""
    ranked, errors = _as_ranked(scores).ravel(), np.asarray(errors, dtype=np.float64).ravel()
    bad = np.asarray(is_bad(torch.as_tensor(errors), bad_threshold))
    bins = np.searchsorted(np.unique(edges), ranked, side="left")
    out = []
    for i in range(len(np.unique(edges)) + 1):
        chosen = bins == i
        any_frames = bool(chosen.any())
        out.append(
            {
                "frames": int(chosen.sum()),
                "mean_error": float(errors[chosen].mean()) if any_frames else None,
                "bad_share": float(bad[chosen].mean()) if any_frames else None,
            }
        )
    return out


def _covered(scores: np.ndarray, errors: np.ndarray) -> tuple[list[float], list[float]]:
    ordered = np.sort(scores)
    achieved, mean_error = [], []
    for tenths in COVERAGES:
        keep = -(-tenths * len(scores) // 10)
        kept = scores <= ordered[keep - 1]
        achieved.append(float(kept.mean()))
        mean_error.append(float(errors[kept].mean()))
    return achieved, mean_error


def risk_coverage(scores: Tensor, errors: Tensor) -> dict:
    """The mean error of the lowest-doubt frames at each coverage, pooled and within each time.

    Tied frames are kept together, so the achieved coverage can pass the one asked for.
    """
    ranked, errors = _as_ranked(scores), np.asarray(errors, dtype=np.float64)
    pooled = _covered(ranked.ravel(), errors.ravel())
    by_time = [_covered(ranked[:, t], errors[:, t]) for t in range(ranked.shape[1])]
    return {
        "coverage": [tenths / 10 for tenths in COVERAGES],
        "pooled": {"achieved": pooled[0], "mean_error": pooled[1]},
        "within_time": {
            "achieved": np.mean([item[0] for item in by_time], axis=0).tolist(),
            "mean_error": np.mean([item[1] for item in by_time], axis=0).tolist(),
        },
    }
