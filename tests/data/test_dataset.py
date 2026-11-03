"""Checks on the normaliser and on what a training item holds."""

import json

import numpy as np
import pytest
import torch

from sandeha.data.dataset import Normalizer, StepDataset, fit_normalizer
from sandeha.data.generate import generate
from sandeha.data.store import open_split
from sandeha.physics import scenes

from conftest import tiny_config


@pytest.fixture(scope="module")
def normalizer(tiny_root):
    return Normalizer.load(tiny_root / "normalizer.json")


@pytest.fixture(scope="module")
def planned_grid_root(tmp_path_factory):
    """The planned 256 and 128 grids, with very few scenes and frames."""
    root = tmp_path_factory.mktemp("planned_grid") / "data"
    config = tiny_config(
        res=256,
        coarse_res=128,
        frames=4,
        fine_frames=[],
        scenes={"train": 2, "val": 1, "test_a": 1, "test_b": 1, "test_c": 1},
    )
    generate(config, root, allow_dirty=True)
    return root


@pytest.fixture(scope="module")
def late_root(tmp_path_factory):
    """Every scene in the late-obstacle splits has one, early enough to be seen."""
    root = tmp_path_factory.mktemp("late") / "data"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scenes, "LATE_SHARE", 1.0)
        patch.setattr(scenes, "LATE_FRAMES", (2, 3))
        generate(tiny_config(), root, allow_dirty=True)
    return root


def expected_stats(store):
    """The normaliser's figures, worked out separately from the fitting code."""
    values = [[], [], []]
    deltas = [[], [], []]
    for i in range(store.n_scenes):
        for frame in range(store.n_frames):
            solid = store.obstacle(i, max(frame - 1, 0))
            for c in range(3):
                values[c].append(np.asarray(store.fields[i, frame, c], dtype=np.float64)[~solid])
        for frame in range(store.n_frames - 1):
            solid = store.obstacle(i, frame)
            for c in range(3):
                change = np.asarray(store.fields[i, frame + 1, c], dtype=np.float64) - np.asarray(
                    store.fields[i, frame, c], dtype=np.float64
                )
                deltas[c].append(change[~solid])
    values = [np.concatenate(v) for v in values]
    deltas = [np.concatenate(d) for d in deltas]
    return (
        [v.mean() for v in values],
        [v.std() for v in values],
        [np.sqrt((d * d).mean()) for d in deltas],
    )


def test_normalizer_uses_train_only(tiny_root, normalizer):
    train = open_split(tiny_root, "train")
    mean, std, delta = expected_stats(train)
    assert normalizer.fitted_on == "train"
    assert np.allclose(normalizer.field_mean, mean, rtol=1e-6)
    assert np.allclose(normalizer.field_std, std, rtol=1e-6)
    assert np.allclose(normalizer.delta_std, delta, rtol=1e-6)
    buoyancies = [train.buoyancy(i) for i in range(train.n_scenes)]
    assert normalizer.buoyancy_mean == pytest.approx(np.mean(buoyancies))
    assert normalizer.buoyancy_std == pytest.approx(np.std(buoyancies))

    # The figures from another split are different, so the match above says something.
    other_mean, other_std, _ = expected_stats(open_split(tiny_root, "val"))
    assert not np.allclose(normalizer.field_std, other_std, rtol=1e-3)


def test_normalizer_applies_a_late_obstacle_on_the_planned_frames(late_root):
    """Frame scales use the mask on the way in, and change scales the mask in the input.

    Tiny scenes in ``tiny_root`` end before any late obstacle, so only this root tells them apart.
    """
    train = open_split(late_root, "train")
    assert all(train.scene(i).late_frame is not None for i in range(train.n_scenes))
    mean, std, delta = expected_stats(train)
    fitted = Normalizer.load(late_root / "normalizer.json")
    assert np.allclose(fitted.field_mean, mean, rtol=1e-6)
    assert np.allclose(fitted.field_std, std, rtol=1e-6)
    assert np.allclose(fitted.delta_std, delta, rtol=1e-6)


def test_inflow_scale_is_the_peak_of_train(tiny_root, normalizer):
    peaks = lambda store: [float(store.inflow(i).max()) for i in range(store.n_scenes)]
    train = peaks(open_split(tiny_root, "train"))
    assert normalizer.inflow_scale == pytest.approx(max(train))
    # Another split's peak, or one scene's, is different, so the match above says something.
    assert max(peaks(open_split(tiny_root, "val"))) != pytest.approx(max(train))
    assert min(train) != pytest.approx(max(train))


@pytest.mark.parametrize("split", ["val", "test_a", "test_b", "test_c"])
def test_normalizer_refuses_to_fit_on_any_other_split(tiny_root, split):
    store = open_split(tiny_root, split, final_eval=True)
    with pytest.raises(ValueError):
        fit_normalizer(store)


def test_fit_matches_the_saved_normalizer(tiny_root, normalizer):
    fitted = fit_normalizer(open_split(tiny_root, "train"))
    assert fitted.to_dict() == normalizer.to_dict()


def test_a_constant_channel_is_refused(tiny_root):
    store = open_split(tiny_root, "train")

    class Still:
        split = "train"
        n_scenes = store.n_scenes
        n_frames = store.n_frames
        fields = np.zeros_like(store.fields)
        obstacle = staticmethod(store.obstacle)
        inflow = staticmethod(store.inflow)
        buoyancy = staticmethod(store.buoyancy)

    with pytest.raises(ValueError):
        fit_normalizer(Still())


def test_normalizer_round_trips(tmp_path, normalizer):
    x = torch.randn(4, 3, 8, 8)
    assert torch.allclose(normalizer.decode(normalizer.encode(x)), x, atol=1e-5)
    assert torch.allclose(normalizer.decode_delta(normalizer.encode_delta(x)), x, atol=1e-5)

    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    assert Normalizer.load(path).to_dict() == normalizer.to_dict()
    json.loads(path.read_text())


def test_encode_works_by_channel(normalizer):
    x = torch.zeros(3, 2, 2)
    encoded = normalizer.encode(x)
    for c in range(3):
        assert encoded[c, 0, 0].item() == pytest.approx(
            -normalizer.field_mean[c] / normalizer.field_std[c]
        )


def test_encode_refuses_the_wrong_channel_count(normalizer):
    with pytest.raises(ValueError):
        normalizer.encode(torch.zeros(1, 6, 4, 4))


def test_delta_scale_is_not_the_field_scale(normalizer):
    """One neural step changes the fields far less than their overall spread."""
    x = torch.ones(3, 2, 2)
    encoded = normalizer.encode_delta(x)
    for c in range(3):
        assert encoded[c, 0, 0].item() == pytest.approx(1.0 / normalizer.delta_std[c])


def test_item_shapes(planned_grid_root):
    store = open_split(planned_grid_root, "train")
    dataset = StepDataset(store, Normalizer.load(planned_grid_root / "normalizer.json"))
    item = dataset[0]
    assert item["input"].shape == (6, 128, 128)
    assert item["target_seq"].shape == (1, 3, 128, 128)
    assert item["obstacle"].shape == (1, 128, 128)
    assert item["inflow"].shape == (1, 128, 128)
    assert item["buoyancy"].shape == ()
    assert item["input"].dtype == torch.float32
    assert item["target_seq"].dtype == torch.float32


def test_item_channels_are_in_the_planned_order(planned_grid_root):
    store = open_split(planned_grid_root, "train")
    normalizer = Normalizer.load(planned_grid_root / "normalizer.json")
    dataset = StepDataset(store, normalizer)
    # Scene 1, frame 2.
    item = dataset[dataset.windows + 2]
    assert item["scene"] == 1 and item["frame"] == 2
    x = item["input"]

    fields = torch.from_numpy(np.array(store.fields[1, 2], dtype=np.float32))
    assert torch.allclose(normalizer.decode(x[:3]), fields, atol=1e-4)
    assert torch.equal(x[3] > 0.5, torch.from_numpy(store.obstacle(1, 2)))
    assert set(x[3].unique().tolist()) <= {0.0, 1.0}
    inflow = torch.from_numpy(store.inflow(1))
    assert torch.allclose(x[4], inflow / normalizer.inflow_scale)
    expected = (store.buoyancy(1) - normalizer.buoyancy_mean) / normalizer.buoyancy_std
    assert torch.allclose(x[5], torch.full_like(x[5], expected))
    assert item["buoyancy"].item() == pytest.approx(store.buoyancy(1))
    assert torch.equal(item["inflow"][0], inflow)


def test_every_window_is_there_once(tiny_root, normalizer):
    store = open_split(tiny_root, "train")
    for unroll in (1, 3):
        dataset = StepDataset(store, normalizer, unroll=unroll)
        assert dataset.windows == store.n_frames - unroll
        assert len(dataset) == store.n_scenes * dataset.windows
        seen = {(dataset[i]["scene"], dataset[i]["frame"]) for i in range(len(dataset))}
        assert len(seen) == len(dataset)
        with pytest.raises(IndexError):
            dataset[len(dataset)]


def test_unroll_must_fit_in_a_scene(tiny_root, normalizer):
    store = open_split(tiny_root, "train")
    for unroll in (0, store.n_frames):
        with pytest.raises(ValueError):
            StepDataset(store, normalizer, unroll=unroll)


def test_targets_are_the_following_frames(tiny_root, normalizer):
    store = open_split(tiny_root, "train")
    dataset = StepDataset(store, normalizer, unroll=3)
    item = dataset[1]
    assert item["target_seq"].shape == (3, 3, 32, 32)
    for k in range(3):
        fields = torch.from_numpy(np.array(store.fields[0, 1 + k + 1], dtype=np.float32))
        assert torch.allclose(normalizer.decode(item["target_seq"][k]), fields, atol=1e-4)


def test_make_input_rebuilds_an_item_input(tiny_root, normalizer):
    """Pushforward rollouts must be able to build later inputs exactly as the dataset does."""
    store = open_split(tiny_root, "train")
    item = StepDataset(store, normalizer)[2]
    fields = normalizer.decode(item["input"][:3])
    rebuilt = normalizer.make_input(fields, item["obstacle"], item["inflow"], item["buoyancy"])
    assert torch.allclose(rebuilt, item["input"], atol=1e-5)

    batched = normalizer.make_input(
        fields[None], item["obstacle"][None], item["inflow"][None], item["buoyancy"][None]
    )
    assert batched.shape == (1, 6, 32, 32)
    assert torch.allclose(batched[0], item["input"], atol=1e-5)


def test_late_obstacle_changes_the_mask_at_its_frame(late_root):
    store = open_split(late_root, "train")
    for i in range(store.n_scenes):
        scene = store.scene(i)
        last = scene.late_frame
        assert last in (2, 3)
        assert np.array_equal(store.obstacle(i, last - 1), scene.obstacle)
        both = scene.obstacle | scene.late_obstacle
        assert np.array_equal(store.obstacle(i, last), both)
        assert np.array_equal(store.obstacle(i, store.n_frames - 1), both)


def test_obstacle_seq_follows_a_late_obstacle(late_root):
    store = open_split(late_root, "train")
    normalizer = Normalizer.load(late_root / "normalizer.json")
    dataset = StepDataset(store, normalizer, unroll=3)
    late = store.scene(0).late_frame
    item = dataset[late - 1]  # Scene 0. The event falls inside the unroll.
    assert item["frame"] == late - 1
    assert item["obstacle_seq"].shape == (3, 1, 32, 32)
    assert torch.equal(item["obstacle_seq"][0], item["obstacle"])
    assert torch.equal(item["obstacle"][0] > 0.5, torch.from_numpy(store.obstacle(0, late - 1)))
    assert torch.equal(item["obstacle_seq"][1][0] > 0.5, torch.from_numpy(store.obstacle(0, late)))
    assert not torch.equal(item["obstacle_seq"][0], item["obstacle_seq"][1])


def test_every_target_is_clear_of_the_mask_in_its_input(late_root):
    """apply_step forces solid cells to zero, so the target must be zero there too."""
    store = open_split(late_root, "train")
    for i in range(store.n_scenes):
        for frame in range(store.n_frames - 1):
            target = np.asarray(store.fields[i, frame + 1])
            assert np.all(target[:, store.obstacle(i, frame)] == 0.0)
