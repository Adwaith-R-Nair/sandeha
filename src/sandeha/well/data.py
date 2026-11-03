"""The public dataset: The Well's ``turbulent_radiative_layer_2D``, read with h5py.

This is a published scientific simulation benchmark, not measured data.

Layout under the root, as downloaded:

    data/<split>/*.hdf5   One file per cooling time. Splits are train, valid and test.
                          t0_fields/density and t0_fields/pressure: (trajectory, time, x, y)
                          t1_fields/velocity: (trajectory, time, x, y, 2)
    normalizer.json       Written here, fitted on train only.

Frames are ``(4, x, y)``, with channels in the order of ``CHANNELS``. Grid, step and trajectory
counts come from the files, are checked between them, and are compared with the config by
``check_counts``. Files are opened for each read, so that loader workers share no HDF5 state.
"""

import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from sandeha.data.store import write_json_atomic

SPLITS = ("train", "valid", "test")
TEST_SPLIT = "test"
CHANNELS = ("density", "pressure", "velocity_x", "velocity_y")
NORMALIZER_FILE = "normalizer.json"


class WellCountError(ValueError):
    """The files disagree with each other, or with the counts in the config."""


class WellSplit:
    """One official split. The test split needs ``final_eval=True``, checked before any reading."""

    def __init__(self, root: Path, split: str, final_eval: bool = False):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        if split == TEST_SPLIT and not final_eval:
            raise PermissionError(
                f"{split} is a locked test split, read only by the final evaluation (final_eval=True)"
            )
        self.split = split
        self.files = sorted((Path(root) / "data" / split).glob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"no HDF5 files in {Path(root) / 'data' / split}")
        self._index: list[tuple[Path, int]] = []
        shapes = set()
        for path in self.files:
            trajectories, steps, grid = self._read_counts(path)
            shapes.add((steps, grid))
            self._index += [(path, i) for i in range(trajectories)]
        if len(shapes) != 1:
            raise WellCountError(f"files in {split} have different steps and grids: {sorted(shapes)}")
        self.n_steps, self.grid = shapes.pop()
        self.n_trajectories = len(self._index)

    @staticmethod
    def _read_counts(path: Path) -> tuple[int, int, tuple[int, int]]:
        with h5py.File(path, "r") as file:
            density = file["t0_fields/density"].shape
            pressure = file["t0_fields/pressure"].shape
            velocity = file["t1_fields/velocity"].shape
            if len(density) != 4 or pressure != density or velocity != (*density, 2):
                raise WellCountError(f"{path}: density {density}, pressure {pressure}, velocity {velocity}")
            trajectories, steps, *grid = density
            stated = {
                "trajectories": (file.attrs.get("n_trajectories", trajectories), trajectories),
                "time": (file["dimensions/time"].shape[-1], steps),
                "x": (file["dimensions/x"].shape[-1], grid[0]),
                "y": (file["dimensions/y"].shape[-1], grid[1]),
            }
        for name, (claimed, found) in stated.items():
            if claimed != found:
                raise WellCountError(f"{path}: states {claimed} for {name}, and its fields have {found}")
        return trajectories, steps, tuple(grid)

    def read(self, trajectory: int, start: int, stop: int) -> np.ndarray:
        """Frames ``start`` to ``stop - 1`` of a trajectory, float32 ``(T, 4, x, y)``."""
        path, local = self._index[trajectory]
        with h5py.File(path, "r") as file:
            density = file["t0_fields/density"][local, start:stop]
            pressure = file["t0_fields/pressure"][local, start:stop]
            velocity = file["t1_fields/velocity"][local, start:stop]
        return np.stack(
            [density, pressure, velocity[..., 0], velocity[..., 1]], axis=1, dtype=np.float32
        )


def check_counts(split: WellSplit, expected: dict) -> None:
    """Raise unless the split's grid, steps and trajectories are those expected."""
    if list(split.grid) != list(expected["grid"]):
        raise WellCountError(f"{split.split} has grid {split.grid}, expected {expected['grid']}")
    if split.n_steps != expected["steps"]:
        raise WellCountError(f"{split.split} has {split.n_steps} steps, expected {expected['steps']}")
    wanted = expected["trajectories"][split.split]
    if split.n_trajectories != wanted:
        raise WellCountError(f"{split.split} has {split.n_trajectories} trajectories, expected {wanted}")


def _channel_view(values: list[float], x: Tensor) -> Tensor:
    if x.dim() < 3 or x.shape[-3] != len(CHANNELS):
        raise ValueError(f"expected {len(CHANNELS)} channels in dimension -3, got shape {x.shape}")
    return torch.tensor(values, dtype=x.dtype, device=x.device).view(len(CHANNELS), 1, 1)


class WellNormalizer:
    """Scales for frames of shape ``(..., 4, x, y)``, as ``Normalizer`` does for the smoke fields."""

    def __init__(
        self,
        field_mean: list[float],
        field_std: list[float],
        delta_rms: list[float],
        fitted_on: str,
        n_trajectories: int,
    ):
        self.field_mean = field_mean
        self.field_std = field_std
        self.delta_rms = delta_rms
        self.fitted_on = fitted_on
        self.n_trajectories = n_trajectories

    def encode(self, x: Tensor) -> Tensor:
        return (x - _channel_view(self.field_mean, x)) / _channel_view(self.field_std, x)

    def decode(self, x: Tensor) -> Tensor:
        return x * _channel_view(self.field_std, x) + _channel_view(self.field_mean, x)

    def encode_delta(self, x: Tensor) -> Tensor:
        return x / _channel_view(self.delta_rms, x)

    def decode_delta(self, x: Tensor) -> Tensor:
        return x * _channel_view(self.delta_rms, x)

    def to_dict(self) -> dict:
        return {
            "field_mean": self.field_mean,
            "field_std": self.field_std,
            "delta_rms": self.delta_rms,
            "fitted_on": self.fitted_on,
            "n_trajectories": self.n_trajectories,
        }


def fit_well_normalizer(split: WellSplit) -> WellNormalizer:
    """Fit on the train split. Any other split is refused."""
    if split.split != "train":
        raise ValueError(f"the normaliser is fitted on train only, not {split.split}")
    count = change_count = 0
    total, squares, change_squares = np.zeros(4), np.zeros(4), np.zeros(4)
    for trajectory in range(split.n_trajectories):
        frames = split.read(trajectory, 0, split.n_steps).astype(np.float64)
        change = frames[1:] - frames[:-1]
        count += frames.size // 4
        change_count += change.size // 4
        total += frames.sum(axis=(0, 2, 3))
        squares += (frames * frames).sum(axis=(0, 2, 3))
        change_squares += (change * change).sum(axis=(0, 2, 3))
    mean = total / count
    std = np.sqrt(squares / count - mean * mean)
    delta = np.sqrt(change_squares / change_count)
    if (std == 0).any() or (delta == 0).any():
        raise ValueError("a channel has no spread in train, so it can't be normalised")
    return WellNormalizer(mean.tolist(), std.tolist(), delta.tolist(), split.split, split.n_trajectories)


def load_well_normalizer(root: Path) -> WellNormalizer:
    """The normaliser saved under ``root``, fitting it on train if there is none yet."""
    path = Path(root) / NORMALIZER_FILE
    if not path.exists():
        write_json_atomic(path, fit_well_normalizer(WellSplit(root, "train")).to_dict())
    normalizer = WellNormalizer(**json.loads(path.read_text()))
    if normalizer.fitted_on != "train":
        raise ValueError(f"{path} was fitted on {normalizer.fitted_on}, not train")
    return normalizer


class WellStepDataset(Dataset):
    """Each item starts at one frame of one trajectory, and holds the next ``unroll`` frames.

    An item is a dict: ``input`` ``(4,x,y)``, ``target_seq`` ``(unroll,4,x,y)``, both through
    ``encode``, and ``trajectory`` and ``frame``, where the item came from.
    """

    def __init__(
        self,
        root: Path,
        split: Literal["train", "valid", "test"],
        final_eval: bool = False,
        unroll: int = 1,
    ):
        self.split = WellSplit(root, split, final_eval)
        if not 1 <= unroll < self.split.n_steps:
            raise ValueError(f"unroll {unroll} doesn't fit in {self.split.n_steps} frames")
        self.normalizer = load_well_normalizer(root)
        self.unroll = unroll
        self.windows = self.split.n_steps - unroll

    def __len__(self) -> int:
        return self.split.n_trajectories * self.windows

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory, frame = divmod(index, self.windows)
        frames = self.normalizer.encode(
            torch.from_numpy(self.split.read(trajectory, frame, frame + self.unroll + 1))
        )
        return {"input": frames[0], "target_seq": frames[1:], "trajectory": trajectory, "frame": frame}
