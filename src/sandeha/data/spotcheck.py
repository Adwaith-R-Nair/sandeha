"""The spot-check after generation: draw a sample of each split, and check what was stored.

This opens test splits under the exception in ``docs/PREREGISTRATION.md``: it is an integrity check
on the data, and nothing about the data, ranges or models is tuned from it. The scenes drawn are
fixed in advance, not picked after looking: each split's scene 0, and its first late-obstacle scene.

The travel rule is a separate step, ``scripts/check_travel.py --data-root``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sandeha.data.dataset import Normalizer
from sandeha.data.store import open_split, read_manifest
from sandeha.physics.scenes import SPLITS

CHANNELS = ("u", "v", "density")
FRAMES_SHOWN = (0, 20, 45, 70, 120)


def count_non_finite(fields) -> int:
    """How many values aren't finite. Goes scene by scene, as a split may not fit in memory."""
    return sum(int(np.size(scene) - np.count_nonzero(np.isfinite(scene))) for scene in fields)


def frames_to_show(frames: int) -> list[int]:
    """The planned frames that a scene has. A shorter scene shows its first and last."""
    shown = [frame for frame in FRAMES_SHOWN if frame < frames]
    return shown if frames > FRAMES_SHOWN[-1] else [0, frames - 1]


def _channel_stats(fields) -> dict:
    """Range, mean and RMS of each channel, over all scenes and frames."""
    channels = {}
    for k, name in enumerate(CHANNELS):
        low, high, total, squares, count = np.inf, -np.inf, 0.0, 0.0, 0
        for scene in fields:
            values = np.asarray(scene[:, k], dtype=np.float64)
            low, high = min(low, float(values.min())), max(high, float(values.max()))
            total += float(values.sum())
            squares += float(np.square(values).sum())
            count += values.size
        channels[name] = {"min": low, "max": high, "mean": total / count, "rms": (squares / count) ** 0.5}
    return channels


def _check_split(store, entry: dict, config: dict) -> dict:
    expected = (config["scenes"][store.split], entry["n_frames"], 3, config["coarse_res"], config["coarse_res"])
    stats = entry["stats"]
    return {
        "n_scenes": store.n_scenes,
        "n_frames": store.n_frames,
        "shape_matches_manifest": tuple(store.fields.shape) == expected and entry["n_frames"] == config["frames"],
        "non_finite_values": count_non_finite(store.fields),
        "channels": _channel_stats(store.fields),
        "unconverged_item_steps": stats["unconverged_item_steps"],
        "max_density_in_solid": stats["max_density_in_solid"],
        "late_scenes": entry["late_scenes"],
        "scenes_over_thinnest_wall": stats["scenes_over_thinnest_wall"],
    }


def _render(rows: list[tuple[str, object, int]], frames: int, path: Path) -> None:
    """One row per scene, with density at each frame shown and the solid in force in grey."""
    shown = frames_to_show(frames)
    figure, axes = plt.subplots(
        len(rows), len(shown), figsize=(1.9 * len(shown), 1.9 * len(rows)), squeeze=False
    )
    for row, (label, store, scene) in enumerate(rows):
        top = max(float(np.max(store.fields[scene, frame, 2])) for frame in shown) or 1.0
        for column, frame in enumerate(shown):
            axis = axes[row, column]
            density = np.asarray(store.fields[scene, frame, 2], dtype=np.float32)
            axis.imshow(density, origin="lower", cmap="magma", vmin=0, vmax=top)
            overlay = np.zeros((*density.shape, 4))
            overlay[store.obstacle(scene, max(frame - 1, 0))] = (0.6, 0.6, 0.6, 1.0)
            axis.imshow(overlay, origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"frame {frame}", fontsize=9)
        axes[row, 0].set_ylabel(label, fontsize=8)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)


def run_spot_check(root: Path, figure_path: Path) -> dict:
    """Check every split and draw its sample scenes. A changed file raises, from the lock's checks."""
    root = Path(root)
    manifest = read_manifest(root)
    Normalizer.load(root / "normalizer.json")
    config = manifest["config"]
    splits, rows = {}, []
    for split in SPLITS:
        store = open_split(root, split, final_eval=True)
        entry = manifest["splits"][split]
        splits[split] = _check_split(store, entry, config)
        rendered = [0]
        rows.append((f"{split} scene 0", store, 0))
        if entry["late_scenes"]:
            first_late = next(i for i in range(store.n_scenes) if store.scene(i).late_frame is not None)
            if first_late != 0:
                rendered.append(first_late)
                late_frame = store.scene(first_late).late_frame
                rows.append((f"{split} scene {first_late}, late at {late_frame}", store, first_late))
        splits[split]["rendered_scenes"] = rendered
    _render(rows, config["frames"], figure_path)
    return {
        "manifest_git_commit": manifest["git_commit"],
        "manifest_git_dirty": manifest["git_dirty"],
        "timing_seconds": manifest["timing"]["total_seconds"],
        "peak_gpu_mib": manifest["peak_gpu_mib"],
        "splits": splits,
        "passed": all(
            entry["non_finite_values"] == 0 and entry["shape_matches_manifest"] for entry in splits.values()
        ),
    }
