"""The reader of the public dataset: its shapes, its counts, and the lock on its test split."""

from pathlib import Path

import pytest
import torch
import yaml

from sandeha.well.data import (
    CHANNELS,
    WellCountError,
    WellSplit,
    WellStepDataset,
    check_counts,
    fit_well_normalizer,
    load_well_normalizer,
)
from well_synth import FILES, GRID, PER_FILE, STEPS, write_file

REPO = Path(__file__).resolve().parents[2]
REAL_ROOT = REPO / "data" / "well"


def test_reads_expected_shapes(well_root):
    dataset = WellStepDataset(well_root, "train")
    trajectories = PER_FILE["train"] * len(FILES)
    assert len(dataset) == trajectories * (STEPS - 1)
    item = dataset[STEPS]  # The second trajectory's second frame.
    assert item["input"].shape == (len(CHANNELS), *GRID)
    assert item["target_seq"].shape == (1, len(CHANNELS), *GRID)
    assert item["input"].dtype == torch.float32
    assert (item["trajectory"], item["frame"]) == (1, 1)


def test_item_is_the_frame_and_the_next_one(well_root):
    dataset = WellStepDataset(well_root, "valid")
    frames = torch.from_numpy(dataset.split.read(1, 3, 5))
    item = dataset[1 * (STEPS - 1) + 3]
    assert torch.allclose(item["input"], dataset.normalizer.encode(frames[0]))
    assert torch.allclose(item["target_seq"][0], dataset.normalizer.encode(frames[1]))


def test_channels_are_density_pressure_and_velocity(well_root):
    split = WellSplit(well_root, "train")
    frames = split.read(0, 0, 2)
    assert frames.shape == (2, 4, *GRID)
    assert frames[:, 0].mean() > 10  # density in the fixture
    assert 0.5 < frames[:, 1].mean() < 2  # pressure in the fixture
    assert abs(frames[:, 2:].mean()) < 0.1  # velocity in the fixture


def test_test_split_guarded(well_root, tmp_path):
    with pytest.raises(PermissionError):
        WellStepDataset(well_root, "test")
    with pytest.raises(PermissionError):
        WellSplit(well_root, "test")
    assert len(WellStepDataset(well_root, "test", final_eval=True)) > 0
    # The lock comes first, so a refused split isn't read at all.
    with pytest.raises(PermissionError):
        WellSplit(tmp_path / "nothing", "test")


def test_unknown_split_refused(well_root):
    with pytest.raises(ValueError):
        WellSplit(well_root, "val")


def test_missing_split_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        WellSplit(tmp_path, "train")


def test_normaliser_comes_from_train(well_root):
    trained = fit_well_normalizer(WellSplit(well_root, "train"))
    held_out = WellStepDataset(well_root, "test", final_eval=True).normalizer
    assert held_out.to_dict() == trained.to_dict()
    assert held_out.fitted_on == "train"
    with pytest.raises(ValueError):
        fit_well_normalizer(WellSplit(well_root, "valid"))
    assert load_well_normalizer(well_root).to_dict() == trained.to_dict()


def test_normaliser_scales_train_to_unit_spread(well_root):
    split = WellSplit(well_root, "train")
    normalizer = fit_well_normalizer(split)
    frames = torch.cat([torch.from_numpy(split.read(i, 0, STEPS)) for i in range(split.n_trajectories)])
    encoded = normalizer.encode(frames)
    assert encoded.mean(dim=(0, 2, 3)).abs().max() < 1e-3
    assert (encoded.std(dim=(0, 2, 3)) - 1).abs().max() < 1e-2
    assert torch.allclose(normalizer.decode(encoded), frames, atol=1e-3, rtol=1e-3)


def test_counts_are_read_and_confirmed(well_root, well_cfg):
    split = WellSplit(well_root, "valid")
    assert (split.grid, split.n_steps) == (GRID, STEPS)
    assert split.n_trajectories == PER_FILE["valid"] * len(FILES)
    check_counts(split, well_cfg["expected"])
    with pytest.raises(WellCountError, match="trajectories"):
        check_counts(split, well_cfg["expected"] | {"trajectories": {"valid": 99}})
    with pytest.raises(WellCountError, match="grid"):
        check_counts(split, well_cfg["expected"] | {"grid": [128, 384]})
    with pytest.raises(WellCountError, match="steps"):
        check_counts(split, well_cfg["expected"] | {"steps": 101})


def test_files_that_disagree_are_refused(well_root):
    write_file(well_root / "data" / "valid" / "other.hdf5", 1, STEPS, (GRID[0], GRID[1] + 1), 99)
    with pytest.raises(WellCountError):
        WellSplit(well_root, "valid")


def test_real_file_shapes():
    if not (REAL_ROOT / "data" / "train").is_dir():
        pytest.skip("the public dataset isn't downloaded")
    cfg = yaml.safe_load((REPO / "configs" / "well.yaml").read_text())
    for split in ("train", "valid"):
        check_counts(WellSplit(REAL_ROOT, split), cfg["expected"])
    check_counts(WellSplit(REAL_ROOT, "test", final_eval=True), cfg["expected"])
