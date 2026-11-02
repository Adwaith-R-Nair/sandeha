"""The normaliser, fitted on train only, and the training items built from a split.

Physical fields are scaled twice over, because one neural step changes them far less than their
overall spread: ``encode`` for a field, and ``encode_delta`` for the change over one step.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _channel_view(values: list[float], x: Tensor) -> Tensor:
    if x.dim() < 3 or x.shape[-3] != 3:
        raise ValueError(f"expected channels u, v, density in dimension -3, got shape {x.shape}")
    return torch.tensor(values, dtype=x.dtype, device=x.device).view(3, 1, 1)


class Normalizer:
    """Scales for fields of shape ``(..., 3, H, W)``, and the inputs built from them.

    ``delta_std`` is the root mean square change over one neural step, around zero rather than
    around its mean, so that a model predicting no change predicts exactly zero.
    """

    def __init__(
        self,
        field_mean: list[float],
        field_std: list[float],
        delta_std: list[float],
        inflow_scale: float,
        buoyancy_mean: float,
        buoyancy_std: float,
        fitted_on: str,
        n_scenes: int,
    ):
        self.field_mean = field_mean
        self.field_std = field_std
        self.delta_std = delta_std
        self.inflow_scale = inflow_scale
        self.buoyancy_mean = buoyancy_mean
        self.buoyancy_std = buoyancy_std
        self.fitted_on = fitted_on
        self.n_scenes = n_scenes

    def encode(self, x: Tensor) -> Tensor:
        return (x - _channel_view(self.field_mean, x)) / _channel_view(self.field_std, x)

    def decode(self, x: Tensor) -> Tensor:
        return x * _channel_view(self.field_std, x) + _channel_view(self.field_mean, x)

    def encode_delta(self, x: Tensor) -> Tensor:
        return x / _channel_view(self.delta_std, x)

    def decode_delta(self, x: Tensor) -> Tensor:
        return x * _channel_view(self.delta_std, x)

    def make_input(self, fields: Tensor, obstacle: Tensor, inflow: Tensor, buoyancy: Tensor) -> Tensor:
        """The model's input ``(..., 6, H, W)``: u, v, density, obstacle, inflow, buoyancy.

        ``fields`` is physical ``(..., 3, H, W)``. ``obstacle`` and ``inflow`` are
        ``(..., 1, H, W)``, and ``buoyancy`` has shape ``(...)``.
        """
        lift = (buoyancy - self.buoyancy_mean) / self.buoyancy_std
        lift = lift.view(*lift.shape, 1, 1, 1).expand_as(obstacle).to(fields.dtype)
        return torch.cat(
            [self.encode(fields), obstacle.to(fields.dtype), inflow / self.inflow_scale, lift], dim=-3
        )

    def to_dict(self) -> dict:
        return {
            "field_mean": self.field_mean,
            "field_std": self.field_std,
            "delta_std": self.delta_std,
            "inflow_scale": self.inflow_scale,
            "buoyancy_mean": self.buoyancy_mean,
            "buoyancy_std": self.buoyancy_std,
            "fitted_on": self.fitted_on,
            "n_scenes": self.n_scenes,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls(**json.loads(Path(path).read_text()))


def fit_normalizer(store) -> Normalizer:
    """Fit on fluid cells of the train split. Any other split is refused.

    A frame counts the cells that were fluid on the way into it, and a change counts the cells
    fluid under the mask in its input, so cells a late obstacle just covered are left out.
    """
    if store.split != "train":
        raise ValueError(f"the normaliser is fitted on train only, not {store.split}")
    count = 0
    total = np.zeros(3)
    squares = np.zeros(3)
    change_count = 0
    change_squares = np.zeros(3)
    peak = 0.0
    lifts = []
    for i in range(store.n_scenes):
        fields = np.asarray(store.fields[i], dtype=np.float64)
        masks = np.stack([store.obstacle(i, frame) for frame in range(store.n_frames)])
        into = np.concatenate([masks[:1], masks[:-1]])
        fluid = ~into[:, None]
        count += int(fluid.sum())
        total += (fields * fluid).sum(axis=(0, 2, 3))
        squares += (fields * fields * fluid).sum(axis=(0, 2, 3))

        step_fluid = ~masks[:-1, None]
        change = fields[1:] - fields[:-1]
        change_count += int(step_fluid.sum())
        change_squares += (change * change * step_fluid).sum(axis=(0, 2, 3))

        peak = max(peak, float(store.inflow(i).max()))
        lifts.append(store.buoyancy(i))

    mean = total / count
    std = np.sqrt(squares / count - mean * mean)
    delta = np.sqrt(change_squares / change_count)
    buoyancy_std = float(np.std(lifts))
    if (std == 0).any() or (delta == 0).any() or peak == 0 or buoyancy_std == 0:
        raise ValueError("a channel has no spread in train, so it can't be normalised")
    return Normalizer(
        field_mean=mean.tolist(),
        field_std=std.tolist(),
        delta_std=delta.tolist(),
        inflow_scale=peak,
        buoyancy_mean=float(np.mean(lifts)),
        buoyancy_std=buoyancy_std,
        fitted_on=store.split,
        n_scenes=store.n_scenes,
    )


class StepDataset(Dataset):
    """Each item starts at one frame of one scene, and holds the next ``unroll`` frames.

    An item is a dict:

    - ``input`` ``(6,H,W)``: the model's input at the start frame.
    - ``target_seq`` ``(unroll,3,H,W)``: the following frames, through ``encode``.
    - ``obstacle`` ``(1,H,W)``: the mask in ``input``, as 0 or 1.
    - ``obstacle_seq`` ``(unroll,1,H,W)``: the mask for each step of the unroll, whose first entry
      is ``obstacle``. It changes during the unroll if a late obstacle appears there.
    - ``inflow`` ``(1,H,W)`` and ``buoyancy`` ``()``: physical values, for ``make_input``.
    - ``scene`` and ``frame``: where the item came from.
    """

    def __init__(self, store, normalizer: Normalizer, unroll: int = 1):
        if not 1 <= unroll < store.n_frames:
            raise ValueError(f"unroll {unroll} doesn't fit in {store.n_frames} frames")
        self.store = store
        self.normalizer = normalizer
        self.unroll = unroll
        self.windows = store.n_frames - unroll  # Start frames per scene.

    def __len__(self) -> int:
        return self.store.n_scenes * self.windows

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        scene, frame = divmod(index, self.windows)
        store = self.store
        fields = torch.from_numpy(
            np.array(store.fields[scene, frame : frame + self.unroll + 1], dtype=np.float32)
        )
        masks = torch.from_numpy(
            np.stack([store.obstacle(scene, frame + k) for k in range(self.unroll)])
        )[:, None].float()
        inflow = torch.from_numpy(store.inflow(scene))[None].float()
        buoyancy = torch.tensor(store.buoyancy(scene), dtype=torch.float32)
        return {
            "input": self.normalizer.make_input(fields[0], masks[0], inflow, buoyancy),
            "target_seq": self.normalizer.encode(fields[1:]),
            "obstacle": masks[0],
            "obstacle_seq": masks,
            "inflow": inflow,
            "buoyancy": buoyancy,
            "scene": scene,
            "frame": frame,
        }
