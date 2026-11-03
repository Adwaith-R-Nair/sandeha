"""The measures of docs/PREREGISTRATION.md, on plain arrays.

The ranking measures are one copy, in ``well/doubt.py``, shared with the public track. They rank a
non-finite score as positive infinity, so no frame is ever dropped from a figure.
"""

import numpy as np
import torch

from sandeha.handoff_selection import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, frame_error
from sandeha.physics.scenes import TEST_C_BUOYANCY
from sandeha.train.loop import ERROR_CAP
from sandeha.train.losses import _stencil, divergence
from sandeha.well.doubt import (  # noqa: F401  The eval package's names for the shared copy.
    _as_ranked,
    auroc,
    calibration,
    decile_edges,
    evidence_measure,
    is_bad,
    risk_coverage,
    spearman,
    within_time_auroc,
)

CONTACT_DENSITY = 0.01
PLUME_DENSITY = 0.01
_SMALLEST_FLOOR = (1e-12, 1e-12, 1e-12)
_NO_DIVISION_BY_ZERO = 1e-30


def nrmse(pred, ref, obstacle, floors=_SMALLEST_FLOOR) -> float:
    """The per-frame error of fields ``(3,H,W)`` over fluid cells, capped, with non-finite at the cap."""
    return frame_error(np.asarray(pred), np.asarray(ref), np.asarray(obstacle, dtype=bool), floors)


def bootstrap_ci(
    values_by_scene, n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """The 95% percentile interval of the mean, resampling scenes.

    Two calls on the same number of scenes use the same resamples, so a paired difference is the
    interval of the differences themselves.
    """
    values = np.asarray(values_by_scene, dtype=np.float64)
    resamples = np.random.default_rng(seed).integers(0, len(values), (n, len(values)))
    low, high = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def rise_on_unfamiliar(other, familiar, n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """The chance a frame from ``other`` has higher doubt than one from ``familiar``, ties as half.

    Both are ``(scenes, times)``. The figure is the mean over times, with scenes resampled within
    each set. Non-finite doubt ranks above all finite doubt and ties with itself.
    """
    a, b = _as_ranked(other), _as_ranked(familiar)
    # Mean over times of each pair of scenes' chance, so that a resample only indexes this matrix.
    wins = (a[:, None] > b[None]).astype(np.float64) + 0.5 * (a[:, None] == b[None])
    pairs = wins.mean(axis=2)
    rng = np.random.default_rng(seed)
    resampled = [
        pairs[np.ix_(rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b)))].mean() for _ in range(n)
    ]
    return {
        "value": float(pairs.mean()),
        "ci95": [float(x) for x in np.percentile(resampled, [2.5, 97.5])],
        "times": a.shape[1],
    }


def usable_frames(errors, limit: float) -> int:
    """How many frames pass before the error first goes above ``limit``."""
    for frame, error in enumerate(errors):
        if error > limit:
            return frame
    return len(errors)


def side_of(buoyancy: float) -> str:
    """A ``test_c`` scene's side."""
    return "low" if buoyancy <= TEST_C_BUOYANCY[0][1] else "high"


def has_contact(density, masks) -> bool:
    """The preregistration's coarse rule, on ``(frames,H,W)`` from frame 0.

    Frame ``f`` from 1 on has contact if its density is above 0.01 in a cell not solid in the mask
    in force on the way into it, ``masks[f - 1]``, with a solid cell of that mask in its 3x3
    neighbourhood. The box's own walls aren't in a mask, so they don't count.
    """
    density, masks = np.asarray(density), np.asarray(masks, dtype=bool)
    for frame in range(1, len(density)):
        solid = masks[frame - 1]
        padded = np.pad(solid, 1)
        near = np.zeros_like(solid)
        for dy in range(3):
            for dx in range(3):
                near |= padded[dy : dy + solid.shape[0], dx : dx + solid.shape[1]]
        if ((density[frame] > CONTACT_DENSITY) & near & ~solid).any():
            return True
    return False


def _vorticity(fields: np.ndarray) -> np.ndarray:
    return np.gradient(fields[1], axis=1) - np.gradient(fields[0], axis=0)


def _band_power(vorticity: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Power in octaves of wavenumber, 1 to 2, 2 to 4, and so on up to half the grid."""
    res = vorticity.shape[0]
    power = np.abs(np.fft.rfft2(np.where(solid, 0.0, vorticity))) ** 2
    ky, kx = np.meshgrid(np.fft.fftfreq(res, 1 / res), np.fft.rfftfreq(res, 1 / res), indexing="ij")
    radius = np.hypot(kx, ky)
    edges = 2 ** np.arange(int(np.log2(res)))
    return np.array([power[(radius >= low) & (radius < 2 * low)].sum() for low in edges])


def _plume(density: np.ndarray, solid: np.ndarray) -> tuple[float, float, float]:
    """Total smoke, the centroid's height and the height of the top, with rows counting upward."""
    res = density.shape[0]
    smoke = np.where(solid, 0.0, np.clip(density, 0.0, None))
    total = float(smoke.sum())
    heights = (np.arange(res) + 0.5) / res
    centroid = float((smoke.sum(axis=1) * heights).sum() / total) if total > 0 else 0.0
    rows = np.nonzero((smoke > PLUME_DENSITY).any(axis=1))[0]
    return total, centroid, float(heights[rows[-1]]) if len(rows) else 0.0


def frame_secondary(pred, ref, solid) -> dict:
    """One frame's secondary metrics. Every policy gets the same operators, at the neural grid.

    A blown-up frame's vorticity correlation is 0, so that it stays in the average and can't flatter
    it, and its divergence is infinite. Its other figures have no value, and a window's summary
    counts them beside its mean.
    """
    pred, ref, solid = np.asarray(pred, dtype=np.float64), np.asarray(ref, dtype=np.float64), np.asarray(solid, dtype=bool)
    fluid = ~solid
    bands = len(_band_power(_vorticity(ref), solid))
    if not np.isfinite(pred[:, fluid]).all():
        return {
            "divergence_max": np.inf,
            "divergence_rms": np.inf,
            "vorticity_correlation": 0.0,
            "spectrum_log_ratio": [np.nan] * bands,
            "total_smoke_ratio": np.nan,
            "centroid_difference": np.nan,
            "top_difference": np.nan,
        }

    pred = np.where(solid, 0.0, pred)
    velocity = torch.from_numpy(pred[:2])[None]
    mask = torch.from_numpy(solid)[None, None]
    measured = divergence(velocity, mask).numpy()
    cells = max(int(_stencil(mask).sum()), 1)

    predicted, referenced = _vorticity(pred), _vorticity(ref)
    a, b = predicted[fluid], referenced[fluid]
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    ratio = (_band_power(predicted, solid) + _NO_DIVISION_BY_ZERO) / (_band_power(referenced, solid) + _NO_DIVISION_BY_ZERO)
    mine, theirs = _plume(pred[2], solid), _plume(ref[2], solid)
    return {
        "divergence_max": float(np.abs(measured).max()),
        "divergence_rms": float(np.sqrt(np.square(measured).sum() / cells)),
        "vorticity_correlation": correlation,
        "spectrum_log_ratio": np.log(ratio).tolist(),
        "total_smoke_ratio": mine[0] / theirs[0] if theirs[0] > 0 else np.nan,
        "centroid_difference": mine[1] - theirs[1],
        "top_difference": mine[2] - theirs[2],
    }


def window_summary(frames: list[dict]) -> dict:
    """The mean of each secondary metric over frames, and the number with no finite value."""
    summary = {}
    for name in frames[0]:
        values = np.array([frame[name] for frame in frames], dtype=np.float64)
        finite = np.isfinite(values)
        if values.ndim == 1:
            mean = float(values[finite].mean()) if finite.any() else None
        else:
            counts = finite.sum(axis=0)
            totals = np.where(finite, values, 0.0).sum(axis=0)
            mean = [float(t / c) if c else None for t, c in zip(totals, counts, strict=True)]
            finite = finite.all(axis=1)
        summary[name] = {"mean": mean, "frames": len(frames), "non_finite_frames": int((~finite).sum())}
    return summary


def _weighted(first, second, weights):
    if first is None or second is None:
        return first if second is None else second
    total = sum(weights)
    return (first * weights[0] + second * weights[1]) / total if total else None


def join_summaries(first: dict, second: dict) -> dict:
    """Two ``window_summary`` results as one, as if their frames had been summarised together."""
    joined = {}
    for name, a in first.items():
        b = second[name]
        weights = [a["frames"] - a["non_finite_frames"], b["frames"] - b["non_finite_frames"]]
        if isinstance(a["mean"], list):
            mean = [_weighted(x, y, weights) for x, y in zip(a["mean"], b["mean"], strict=True)]
        else:
            mean = _weighted(a["mean"], b["mean"], weights)
        joined[name] = {
            "mean": mean,
            "frames": a["frames"] + b["frames"],
            "non_finite_frames": a["non_finite_frames"] + b["non_finite_frames"],
        }
    return joined


def bad_frames(errors, threshold: float) -> np.ndarray:
    return is_bad(np.asarray(errors, dtype=np.float64), threshold)


def doubt_figures(scores, errors, edges, bad_threshold: float, n: int, seed: int = BOOTSTRAP_SEED) -> dict:
    """A signal's figures against error, both ``(scenes, times)``.

    Only ``evidence`` counts as evidence. The pooled figures, calibration and risk-coverage are
    labelled as not being evidence by themselves.
    """
    scores, errors = np.asarray(scores, dtype=np.float64), np.asarray(errors, dtype=np.float64)
    bad = bad_frames(errors, bad_threshold)
    both_classes = bad.any(axis=0) & ~bad.all(axis=0)
    return {
        "evidence": evidence_measure(scores, errors, n, seed),
        "non_finite_frames": int((~np.isfinite(scores)).sum()),
        "pooled_spearman_not_evidence": spearman(scores.ravel(), errors.ravel()),
        "within_time_auroc": within_time_auroc(scores, bad, 5),
        "pooled_auroc_not_evidence": auroc(scores.ravel(), bad.ravel()),
        "bad_share": float(bad.mean()),
        "times_with_both_classes": int(both_classes.sum()),
        "calibration_not_evidence": calibration(scores, errors, edges, bad_threshold),
        "risk_coverage_not_evidence": risk_coverage(scores, errors),
    }


__all__ = [
    "ERROR_CAP",
    "auroc",
    "bad_frames",
    "bootstrap_ci",
    "calibration",
    "decile_edges",
    "doubt_figures",
    "evidence_measure",
    "frame_secondary",
    "has_contact",
    "join_summaries",
    "nrmse",
    "rise_on_unfamiliar",
    "risk_coverage",
    "side_of",
    "spearman",
    "usable_frames",
    "window_summary",
    "within_time_auroc",
]
