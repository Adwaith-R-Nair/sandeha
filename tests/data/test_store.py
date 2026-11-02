"""Checks on the on-disk format and the lock on the test splits."""

import json
import shutil

import numpy as np
import pytest

from sandeha.data.store import TEST_SPLITS, TamperedDataError, open_split, write_atomic


def test_open_split_refuses_test_without_final_eval(tiny_root):
    for split in TEST_SPLITS:
        with pytest.raises(PermissionError):
            open_split(tiny_root, split)
        with pytest.raises(PermissionError):
            open_split(tiny_root, split, final_eval=False)


def test_test_splits_are_exactly_the_three_planned():
    assert set(TEST_SPLITS) == {"test_a", "test_b", "test_c"}


@pytest.mark.parametrize("split", TEST_SPLITS)
def test_final_eval_opens_a_test_split(tiny_root, split):
    assert open_split(tiny_root, split, final_eval=True).split == split


@pytest.mark.parametrize("split", ["train", "val"])
def test_train_and_val_open_without_final_eval(tiny_root, split):
    assert open_split(tiny_root, split).split == split


def test_lock_is_checked_before_the_files_are_touched(tiny_root, tmp_path):
    """A locked split must refuse even when its files can't be read at all."""
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    shutil.rmtree(root / "test_b")
    with pytest.raises(PermissionError):
        open_split(root, "test_b")


def test_unknown_split_is_refused(tiny_root):
    with pytest.raises(ValueError):
        open_split(tiny_root, "testing")


def test_no_manifest_means_no_data(tiny_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    (root / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        open_split(root, "train")


@pytest.mark.parametrize(
    "split, name",
    [
        ("test_b", "fields.npy"),
        ("test_b", "statics.npz"),
        ("test_b", "fine.npz"),
        ("test_a", "fields.npy"),
        ("train", "fields.npy"),
        ("val", "statics.npz"),
    ],
)
def test_manifest_detects_tampering(tiny_root, tmp_path, split, name):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    path = root / split / name
    content = bytearray(path.read_bytes())
    content[-1] ^= 1  # One bit, at the end, far from any header.
    path.write_bytes(content)
    with pytest.raises(TamperedDataError):
        open_split(root, split, final_eval=True)


def test_a_missing_file_counts_as_tampering(tiny_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    (root / "val" / "fine.npz").unlink()
    with pytest.raises(TamperedDataError):
        open_split(root, "val")


def test_other_splits_still_open_when_one_is_tampered(tiny_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    path = root / "val" / "fields.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    assert open_split(root, "train").n_scenes == 3


def test_fields_are_a_read_only_float16_memmap(tiny_root):
    store = open_split(tiny_root, "train")
    assert isinstance(store.fields, np.memmap)
    assert store.fields.dtype == np.float16
    assert store.fields.shape == (3, 6, 3, 32, 32)
    assert not store.fields.flags.writeable
    assert store.n_scenes == 3
    assert store.n_frames == 6
    assert len(store) == 3


def test_scene_is_the_fine_scene_downsampled(tiny_root):
    store = open_split(tiny_root, "train")
    for i in range(store.n_scenes):
        coarse = store.scene(i)
        fine = store.fine_scene(i)
        assert coarse.obstacle.shape == (32, 32)
        assert fine.obstacle.shape == (64, 64)
        blocks = fine.obstacle.reshape(32, 2, 32, 2)
        # A coarse cell is solid if any of its fine cells is.
        assert np.array_equal(coarse.obstacle, blocks.any(axis=(1, 3)))
        expected = fine.inflow.reshape(32, 2, 32, 2).mean(axis=(1, 3))
        assert np.allclose(coarse.inflow, expected)
        assert coarse.buoyancy == fine.buoyancy
        assert coarse.late_frame == fine.late_frame


def test_accessors_match_the_scene(tiny_root):
    store = open_split(tiny_root, "val")
    scene = store.scene(1)
    assert np.array_equal(store.inflow(1), scene.inflow)
    assert store.buoyancy(1) == scene.buoyancy
    assert store.obstacle(1, 0).dtype == bool
    assert np.array_equal(store.obstacle(1, 0), scene.obstacle)


def test_fine_state_only_where_stored(tiny_root):
    store = open_split(tiny_root, "train")
    state = store.fine_state(0, 2)
    assert state.velocity.shape == (1, 2, 64, 64)
    assert state.density.shape == (1, 1, 64, 64)
    assert state.mac is None
    with pytest.raises(KeyError):
        store.fine_state(0, 3)


def test_scene_index_out_of_range_is_refused(tiny_root):
    store = open_split(tiny_root, "train")
    for call in (store.scene, store.inflow, store.buoyancy):
        with pytest.raises(IndexError):
            call(3)
    with pytest.raises(IndexError):
        store.obstacle(0, 6)


def test_manifest_names_every_required_item(tiny_root):
    manifest = json.loads((tiny_root / "manifest.json").read_text())
    assert manifest["config"]["seed"] == 11
    assert len(manifest["git_commit"]) == 40
    assert isinstance(manifest["git_dirty"], bool)
    assert set(manifest["splits"]) == {"train", "val", "test_a", "test_b", "test_c"}
    for split, entry in manifest["splits"].items():
        assert len(entry["seeds"]) == manifest["config"]["scenes"][split]
        assert len(entry["sha256"]) == 64
        assert set(entry["files"]) == {"fields.npy", "statics.npz", "fine.npz"}
        assert "late_scenes" in entry
    assert entry["sides"]["low"] + entry["sides"]["high"] == 2  # test_c is last.
    assert len(manifest["normalizer_sha256"]) == 64


def test_an_interrupted_write_leaves_no_file_and_keeps_the_old_one(tmp_path):
    def torn(file):
        file.write(b"part of a scene")
        raise OSError("disk full")

    path = tmp_path / "00000.npz"
    with pytest.raises(OSError):
        write_atomic(path, torn)
    assert not path.exists()

    path.write_bytes(b"finished")
    with pytest.raises(OSError):
        write_atomic(path, torn)
    assert path.read_bytes() == b"finished"
