"""Tiny datasets shared by the data tests. Nothing here approaches the planned size."""

import pytest

from sandeha.data.generate import generate


def tiny_config(**changes) -> dict:
    config = {
        "seed": 11,
        "res": 64,
        "coarse_res": 32,
        "dt": 0.01,
        "steps_per_frame": 4,
        "frames": 6,
        "batch_size": 2,
        "fine_frames": [2],
        "scenes": {"train": 3, "val": 2, "test_a": 1, "test_b": 1, "test_c": 2},
    }
    config.update(changes)
    return config


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    """A finished tiny dataset. Tests must not change it, and should copy it to do so."""
    root = tmp_path_factory.mktemp("tiny") / "data"
    generate(tiny_config(), root, allow_dirty=True)
    return root
