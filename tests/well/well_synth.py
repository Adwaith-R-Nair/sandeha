"""Writes a tiny dataset in The Well's on-disk layout."""

from pathlib import Path

import h5py
import numpy as np

GRID = (16, 24)
STEPS = 14
FILES = ("0.03", "0.10", "1.00")
PER_FILE = {"train": 2, "valid": 2, "test": 2}


def write_file(path: Path, trajectories: int, steps: int, grid: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (trajectories, steps, *grid)
    x = np.linspace(0, 2 * np.pi, grid[0], dtype=np.float32)[:, None]
    y = np.linspace(0, 2 * np.pi, grid[1], dtype=np.float32)[None, :]
    t = np.arange(steps, dtype=np.float32)[None, :, None, None] / 4
    phase = rng.uniform(0, 6, (trajectories, 1, 1, 1)).astype(np.float32)
    wave = np.sin(x + t + phase) * np.cos(y - t + phase)
    with h5py.File(path, "w") as file:
        file.attrs["dataset_name"] = "turbulent_radiative_layer_2D"
        file.attrs["n_trajectories"] = trajectories
        dimensions = file.create_group("dimensions")
        dimensions["time"] = np.arange(steps, dtype=np.float32)
        dimensions["x"] = np.linspace(0, 1, grid[0], dtype=np.float32)
        dimensions["y"] = np.linspace(0, 1, grid[1], dtype=np.float32)
        file.create_group("boundary_conditions")
        file.create_group("scalars")["tcool"] = 0.1
        t0 = file.create_group("t0_fields")
        t0["density"] = (30 + 20 * wave + rng.normal(0, 0.1, shape)).astype(np.float32)
        t0["pressure"] = (1 + 0.1 * wave + rng.normal(0, 0.001, shape)).astype(np.float32)
        velocity = np.stack([0.05 * wave, -0.05 * wave], axis=-1) + rng.normal(0, 1e-3, (*shape, 2))
        file.create_group("t1_fields")["velocity"] = velocity.astype(np.float32)
        file.create_group("t2_fields")
