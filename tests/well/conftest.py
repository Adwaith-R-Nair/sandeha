"""Fixtures for the public-dataset track."""


import pytest
import torch

from well_synth import FILES, GRID, PER_FILE, STEPS, write_file


@pytest.fixture
def well_root(tmp_path):
    root = tmp_path / "well"
    seed = 0
    for split, trajectories in PER_FILE.items():
        for name in FILES:
            seed += 1
            path = root / "data" / split / f"turbulent_radiative_layer_tcool_{name}.hdf5"
            write_file(path, trajectories, STEPS, GRID, seed)
    return root


@pytest.fixture
def well_cfg(well_root):
    return {
        "data_root": str(well_root),
        "device": "cpu",
        "expected": {
            "grid": list(GRID),
            "steps": STEPS,
            "trajectories": {split: n * len(FILES) for split, n in PER_FILE.items()},
        },
        "seeds": [0, 1, 2],
        "seed": 0,
        "model": {"arch": "unet", "base_width": 4, "levels": 2, "groups": 2, "inputs": 4, "outputs": 4},
        "batch_size": 2,
        "num_workers": 0,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "warmup_fraction": 0.2,
        "pushforward_prob": 0.5,
        "pushforward_max": 2,
        "total_steps": 6,
        "eval_every": 3,
        "checkpoint_every": 2,
        "log_every": 1,
        "rollout_start": 2,
        "rollout_steps": 3,
        "eval_batch_size": 2,
        "evidence": {
            "start": 2,
            "first": 3,
            "last": 8,
            "secondary_last": 13,
            "bad_percentile": 80,
            "min_each_class": 5,
            "bootstrap": 50,
            "seed": 0,
        },
    }


@pytest.fixture(autouse=True)
def _few_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)
