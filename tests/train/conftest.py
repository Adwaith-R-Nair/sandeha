"""A tiny finished dataset and a training config small enough for the CPU."""

import pytest

from sandeha.data.generate import generate


@pytest.fixture(scope="session")
def train_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("train_tiny") / "data"
    config = {
        "seed": 12,
        "res": 64,
        "coarse_res": 32,
        "dt": 0.01,
        "steps_per_frame": 4,
        "frames": 8,
        "batch_size": 2,
        "fine_frames": [2],
        "scenes": {"train": 3, "val": 2, "test_a": 1, "test_b": 1, "test_c": 2},
    }
    generate(config, root, allow_dirty=True)
    return root


@pytest.fixture
def cfg(train_root) -> dict:
    return {
        "data_root": str(train_root),
        "device": "cpu",
        "seed": 3,
        "model": {"arch": "unet", "base_width": 8, "levels": 2},
        "batch_size": 2,
        "num_workers": 0,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "lambda_div": 0.1,
        "mixed_precision": False,
        "warmup_fraction": 0.25,
        "pushforward_prob": 0.5,
        "pushforward_max": 3,
        "total_steps": 8,
        "eval_every": 4,
        "checkpoint_every": 3,
        "log_every": 1,
        "rollout_start": 1,
        "rollout_steps": 4,
        "eval_batch_size": 2,
    }
