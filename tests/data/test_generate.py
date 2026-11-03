"""Checks on data generation: seeds, resuming, what is stored, and what stops a run."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from sandeha.data import generate as generate_module
from sandeha.data.generate import GenerationError, generate, load_config, scene_seed, travel_cells
from sandeha.data.store import open_split, sha256_file
from sandeha.physics import scenes
from sandeha.physics.scenes import SPLITS, sample_scene
from sandeha.physics.solver import SmokeSolver, SmokeState

from conftest import tiny_config

REPO = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda")


class Stop(Exception):
    pass


def manifest_of(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def hashes(root: Path) -> dict:
    manifest = manifest_of(root)
    result = {split: entry["sha256"] for split, entry in manifest["splits"].items()}
    result["normalizer"] = manifest["normalizer_sha256"]
    return result


def stop_after(count: int):
    """A callback that interrupts a run once ``count`` scenes have been written."""
    written = []

    def callback(split, index):
        written.append((split, index))
        if len(written) == count:
            raise Stop

    return callback


def scene_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/scenes/*.npz"))


@pytest.fixture(scope="module")
def uninterrupted(tiny_root):
    return hashes(tiny_root)


def test_no_scene_shared_between_splits(tiny_root):
    manifest = manifest_of(tiny_root)
    seeds = [seed for entry in manifest["splits"].values() for seed in entry["seeds"]]
    assert len(seeds) == len(set(seeds))
    # Seeds alone could agree while the content doesn't, or the reverse, so check content too.
    obstacles = []
    for split in SPLITS:
        store = open_split(tiny_root, split, final_eval=True)
        obstacles += [store.scene(i).obstacle.tobytes() for i in range(store.n_scenes)]
    assert len(obstacles) == len(set(obstacles))


def test_seeds_are_disjoint_at_the_planned_size():
    config = load_config(REPO / "configs" / "data.yaml")
    seeds = [
        scene_seed(config["seed"], split, i)
        for split in SPLITS
        for i in range(config["scenes"][split])
    ]
    assert len(seeds) == 360
    assert len(seeds) == len(set(seeds))


def test_a_scene_seed_does_not_depend_on_split_sizes(tiny_root):
    """Scaling train down must not reroll the test sets."""
    manifest = manifest_of(tiny_root)
    assert manifest["splits"]["test_b"]["seeds"] == [scene_seed(11, "test_b", 0)]
    assert scene_seed(11, "train", 0) != scene_seed(12, "train", 0)


def test_scene_uses_its_own_seed(tiny_root):
    manifest = manifest_of(tiny_root)
    store = open_split(tiny_root, "val")
    for i, seed in enumerate(manifest["splits"]["val"]["seeds"]):
        expected = sample_scene(np.random.default_rng(seed), 64, "val")
        assert np.array_equal(store.fine_scene(i).obstacle, expected.obstacle)
        assert np.array_equal(store.fine_scene(i).inflow, expected.inflow)
        assert store.buoyancy(i) == expected.buoyancy


def test_config_file_matches_the_plan():
    config = load_config(REPO / "configs" / "data.yaml")
    assert config["res"] == 256
    assert config["coarse_res"] == 128
    assert config["steps_per_frame"] == 4
    assert config["frames"] == 121
    assert config["scenes"] == {"train": 240, "val": 30, "test_a": 30, "test_b": 30, "test_c": 30}


@pytest.mark.parametrize(
    "change",
    [
        {"coarse_res": 30},
        {"frames": 1},
        {"batch_size": 0},
        {"fine_frames": [6]},
        {"scenes": {"train": 3}},
    ],
)
def test_bad_config_is_refused(tmp_path, change):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config(**change)))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_resume_generation(tmp_path, uninterrupted, count):
    """Interrupted at a batch boundary or inside a batch, the result is the same to the byte."""
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(count))
    assert not (root / "manifest.json").exists()
    assert len(scene_files(root)) == count

    generate(tiny_config(), root, allow_dirty=True)
    assert hashes(root) == uninterrupted


def test_resume_does_not_redo_finished_scenes(tmp_path, uninterrupted):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(4))
    written = []
    generate(
        tiny_config(), root, allow_dirty=True, on_scene=lambda split, i: written.append((split, i))
    )
    # Train's two batches and val's first scene were finished. The rest of val's batch re-runs.
    assert written[0] == ("val", 1)
    assert not any(split == "train" for split, _ in written)
    assert hashes(root) == uninterrupted


def test_resume_ignores_a_torn_temporary_file(tmp_path, uninterrupted):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(1))
    torn = root / "train" / "scenes" / "00001.npz.tmp"
    torn.write_bytes(b"not a finished scene")
    generate(tiny_config(), root, allow_dirty=True)
    assert not torn.exists()
    assert hashes(root) == uninterrupted


def test_killed_process_resumes_to_the_same_bytes(tmp_path):
    """The real thing: SIGKILL a separate process mid-run, then a third process finishes it."""
    config_path = tmp_path / "config.yaml"
    config = tiny_config(batch_size=1, scenes={"train": 4, "val": 1, "test_a": 1, "test_b": 1, "test_c": 1})
    config_path.write_text(yaml.safe_dump(config))

    def command(root):
        return [
            sys.executable,
            str(REPO / "scripts" / "generate_data.py"),
            "--config",
            str(config_path),
            "--root",
            str(root),
            "--allow-dirty",
            "--manifest-copy",
            str(root.parent / f"{root.name}_manifest_copy.json"),
        ]

    whole = tmp_path / "whole"
    subprocess.run(command(whole), check=True, capture_output=True)

    cut = tmp_path / "cut"
    process = subprocess.Popen(command(cut), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    limit = time.monotonic() + 90
    while len(list(cut.glob("*/scenes/*.npz"))) < 2:
        assert process.poll() is None, "the run ended before it could be killed"
        assert time.monotonic() < limit, "no scenes appeared"
        time.sleep(0.02)
    process.send_signal(signal.SIGKILL)
    process.wait()
    assert not (cut / "manifest.json").exists()

    subprocess.run(command(cut), check=True, capture_output=True)
    assert hashes(cut) == hashes(whole)

    copy = json.loads((tmp_path / "cut_manifest_copy.json").read_text())
    assert copy == manifest_of(cut)


@pytest.mark.parametrize(
    "torn",
    [b'{"split": "train", "sta', b"\x00\x00\x00\x00", b'{"split": "tr\xff'],
    ids=["partial", "zeros", "not_text"],
)
def test_resume_survives_a_torn_timing_line(tmp_path, uninterrupted, torn):
    """A power loss or a full disk can leave part of a line. Timing mustn't cost the dataset."""
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(4))
    timing = root / "timing.jsonl"
    whole = timing.read_bytes()
    assert whole
    timing.write_bytes(whole + torn)

    # The next batch's line must not join on to the torn one, and be lost with it.
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(3))
    assert timing.read_bytes().startswith(whole + b'{"split": "val"')

    manifest = generate(tiny_config(), root, allow_dirty=True)
    assert hashes(root) == uninterrupted
    assert manifest["timing"]["unreadable_lines"] == 0


def test_an_unreadable_line_inside_the_timing_file_is_counted(tmp_path, uninterrupted):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(4))
    timing = root / "timing.jsonl"
    timing.write_bytes(b'{"split": "tr\n[1, 2]\n{"split": "train"}\n' + timing.read_bytes())

    manifest = generate(tiny_config(), root, allow_dirty=True)
    assert hashes(root) == uninterrupted
    assert manifest["timing"]["unreadable_lines"] == 3
    assert manifest["timing"]["by_split"]["train"]["scenes"] == 3


def test_a_missing_timing_file_costs_only_the_timing(tmp_path, uninterrupted):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(9))  # The last scene.
    (root / "timing.jsonl").unlink()
    manifest = generate(tiny_config(), root, allow_dirty=True)
    assert hashes(root) == uninterrupted
    assert manifest["timing"]["by_split"] == {}


def test_a_timing_line_is_on_disk_before_the_next_batch(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(generate_module.os, "fsync", lambda descriptor: synced.append(descriptor))
    generate_module._record_timing(tmp_path, {"split": "train"})
    assert len(synced) == 1


def test_resume_refuses_a_changed_config(tmp_path):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(1))
    with pytest.raises(GenerationError):
        generate(tiny_config(dt=0.02), root, allow_dirty=True)


def test_resume_refuses_a_changed_commit(tmp_path, monkeypatch):
    root = tmp_path / "data"
    with pytest.raises(Stop):
        generate(tiny_config(), root, allow_dirty=True, on_scene=stop_after(1))
    monkeypatch.setattr(generate_module, "git_state", lambda: ("f" * 40, False))
    with pytest.raises(GenerationError):
        generate(tiny_config(), root, allow_dirty=True)


def test_finished_data_is_never_regenerated(tmp_path, tiny_root):
    root = tmp_path / "data"
    generate(tiny_config(frames=3), root, allow_dirty=True)
    path = root / "test_b" / "fields.npy"
    before = path.stat().st_mtime_ns, sha256_file(path)

    written = []
    generate(tiny_config(frames=3), root, allow_dirty=True, on_scene=lambda *a: written.append(a))
    assert written == []
    assert (path.stat().st_mtime_ns, sha256_file(path)) == before

    with pytest.raises(GenerationError):
        generate(tiny_config(frames=4), root, allow_dirty=True)
    assert (path.stat().st_mtime_ns, sha256_file(path)) == before


def test_scene_files_are_removed_once_finished(tiny_root):
    assert scene_files(tiny_root) == []
    assert list(tiny_root.glob("**/*.tmp")) == []


def test_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_module, "git_state", lambda: ("a" * 40, True))
    with pytest.raises(GenerationError):
        generate(tiny_config(), tmp_path / "data")
    assert not (tmp_path / "data").exists()

    generate(tiny_config(frames=3), tmp_path / "allowed", allow_dirty=True)
    manifest = manifest_of(tmp_path / "allowed")
    assert manifest["git_dirty"] is True
    assert manifest["git_commit"] == "a" * 40


def test_clean_tree_needs_no_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_module, "git_state", lambda: ("a" * 40, False))
    generate(tiny_config(frames=3), tmp_path / "data")
    assert manifest_of(tmp_path / "data")["git_dirty"] is False


def test_fields_match_a_direct_solver_run(tmp_path, monkeypatch):
    """Pins down the timing of a late obstacle, the downsampling and the solid-cell rule.

    The solid in force from frame f to f+1 includes the late obstacle once f >= late_frame, so
    frame late_frame is the last one made under the old solid.
    """
    monkeypatch.setattr(scenes, "LATE_SHARE", 1.0)
    monkeypatch.setattr(scenes, "LATE_FRAMES", (2, 2))
    root = tmp_path / "data"
    config = tiny_config(
        frames=5,
        batch_size=1,
        scenes={"train": 2, "val": 1, "test_a": 1, "test_b": 1, "test_c": 1},
    )
    generate(config, root, allow_dirty=True)

    store = open_split(root, "train")
    seed = manifest_of(root)["splits"]["train"]["seeds"][1]
    scene = sample_scene(np.random.default_rng(seed), 64, "train")
    assert scene.late_frame == 2

    old = torch.from_numpy(scene.obstacle)[None, None].to(DEVICE)
    new = old | torch.from_numpy(scene.late_obstacle)[None, None].to(DEVICE)
    inflow = torch.from_numpy(scene.inflow)[None, None].to(DEVICE)
    lift = torch.tensor([scene.buoyancy], device=DEVICE, dtype=torch.float32)
    solver = SmokeSolver(64, DEVICE, 0.01)
    state = SmokeState(torch.zeros(1, 2, 64, 64, device=DEVICE), torch.zeros(1, 1, 64, 64, device=DEVICE))
    travel = []

    def stored_view(state, solid):
        pooled = F.avg_pool2d(torch.cat([state.velocity, state.density], dim=1), 2)
        coarse_solid = F.max_pool2d(solid.float(), 2) > 0
        return (pooled * ~coarse_solid).half().cpu().numpy()[0]

    assert np.array_equal(np.asarray(store.fields[1, 0]), stored_view(state, old))
    for frame in range(1, 5):
        solid = new if frame - 1 >= 2 else old
        for _ in range(4):
            state = solver.step(state, solid, inflow, lift)
            u, v = state.mac
            sideways = torch.maximum(u[..., :-1].abs(), u[..., 1:].abs())
            upwards = torch.maximum(v[..., :-1, :].abs(), v[..., 1:, :].abs())
            speed = (sideways**2 + upwards**2).sqrt().max().item()
            travel.append((speed * 0.01 * 64, frame - 1))
        assert np.array_equal(np.asarray(store.fields[1, frame]), stored_view(state, solid)), frame

    stats = manifest_of(root)["splits"]["train"]["stats"]
    peak, peak_frame = max(travel, key=lambda entry: entry[0])
    assert stats["travel_by_scene"][1] == pytest.approx(peak)
    assert stats["travel_frame_by_scene"][1] == peak_frame


def test_stored_values_are_real_flow(tiny_root):
    store = open_split(tiny_root, "train")
    fields = np.asarray(store.fields)
    assert np.isfinite(fields).all()
    assert np.all(fields[:, 0] == 0)  # Starts still and empty.
    assert np.any(fields[:, -1, 2] > 0)  # Smoke has entered.
    assert np.all(fields[:, :, 2] >= 0)
    assert np.any(fields[:, -1, 1] > 0)  # Buoyancy has lifted it.


def test_split_rules_reach_the_store(tiny_root):
    b = open_split(tiny_root, "test_b", final_eval=True)
    assert b.scene(0).late_obstacle is None and b.scene(0).late_frame is None
    c = open_split(tiny_root, "test_c", final_eval=True)
    assert not 1.0 <= c.buoyancy(0) <= 3.0


def test_non_finite_state_stops_the_run_and_stores_nothing(tmp_path, monkeypatch):
    real_step = SmokeSolver.step
    calls = []

    def step(self, state, obstacle, inflow, buoyancy):
        out = real_step(self, state, obstacle, inflow, buoyancy)
        calls.append(1)
        if len(calls) == 6:
            out.density[1, 0, 3, 3] = float("nan")
        return out

    monkeypatch.setattr(SmokeSolver, "step", step)
    root = tmp_path / "data"
    with pytest.raises(GenerationError, match="not finite"):
        generate(tiny_config(), root, allow_dirty=True)
    assert scene_files(root) == []
    assert not (root / "manifest.json").exists()


def test_float16_overflow_stops_the_run(tmp_path, monkeypatch):
    """A value too large for float16 turns infinite only on storing, so check the stored form."""
    real_step = SmokeSolver.step

    def step(self, state, obstacle, inflow, buoyancy):
        out = real_step(self, state, obstacle, inflow, buoyancy)
        out.density[0] = 1e6
        return out

    monkeypatch.setattr(SmokeSolver, "step", step)
    with pytest.raises(GenerationError, match="not finite"):
        generate(tiny_config(), tmp_path / "data", allow_dirty=True)


def test_unconverged_solves_are_recorded(tmp_path):
    root = tmp_path / "data"
    generate(tiny_config(frames=4, cg_max_iter=1), root, allow_dirty=True)
    stats = manifest_of(root)["splits"]["train"]["stats"]
    assert stats["unconverged_item_steps"] > 0


def test_manifest_records_solver_health(tiny_root):
    manifest = manifest_of(tiny_root)
    stats = manifest["splits"]["train"]["stats"]
    assert stats["unconverged_item_steps"] == 0
    assert stats["item_steps"] == 3 * 5 * 4
    assert stats["max_density_in_solid"] == 0
    assert 0 < stats["max_travel_cells_per_step"] < 64
    assert stats["max_cg_iterations"] > 0
    assert stats["max_divergence"] > 0


def test_travel_is_a_distance_across_both_directions():
    """A wall's thickness is a distance, so a diagonal flow has to count both components."""
    u, v = torch.zeros(2, 1, 8, 9), torch.zeros(2, 1, 9, 8)
    u[0], v[0] = 3.0, 4.0  # A uniform diagonal flow.
    u[1, 0, 2, 3] = -6.0  # One face in cells (2, 2) and (2, 3).
    v[1, 0, 3, 3] = 8.0  # One face in cells (2, 3) and (3, 3).
    assert travel_cells((u, v), dt=0.01, res=8).tolist() == pytest.approx([0.4, 0.8])


def test_travel_counts_only_the_faces_of_one_cell():
    """The fastest faces of two far-apart cells don't add up."""
    u, v = torch.zeros(1, 1, 8, 9), torch.zeros(1, 1, 9, 8)
    u[0, 0, 0, 0] = 3.0
    v[0, 0, 8, 7] = 4.0
    assert travel_cells((u, v), dt=0.01, res=8).tolist() == pytest.approx([0.32])


def test_manifest_records_each_scene_s_travel(tiny_root):
    """A split's largest figure can't say which scene to look at."""
    for split, entry in manifest_of(tiny_root)["splits"].items():
        stats = entry["stats"]
        assert len(stats["travel_by_scene"]) == entry["n_scenes"], split
        assert len(stats["travel_frame_by_scene"]) == entry["n_scenes"], split
        assert max(stats["travel_by_scene"]) == pytest.approx(stats["max_travel_cells_per_step"])
        assert stats["thinnest_wall_cells"] == pytest.approx(0.04 * 64)
        assert all(0 <= frame < 5 for frame in stats["travel_frame_by_scene"])


def test_manifest_names_the_scenes_above_the_thinnest_wall(tmp_path, monkeypatch):
    config = tiny_config(frames=3)
    generate(config, tmp_path / "first", allow_dirty=True)
    travel = manifest_of(tmp_path / "first")["splits"]["train"]["stats"]["travel_by_scene"]
    limit = (min(travel) + max(travel)) / 2
    assert min(travel) < limit < max(travel)

    # Sampling keeps its own constant, so this moves only the limit, and the scenes stay the same.
    monkeypatch.setattr(generate_module, "MIN_THICKNESS", limit / 64)
    generate(config, tmp_path / "second", allow_dirty=True)
    stats = manifest_of(tmp_path / "second")["splits"]["train"]["stats"]
    assert stats["travel_by_scene"] == travel
    assert stats["scenes_over_thinnest_wall"] == [i for i, value in enumerate(travel) if value > limit]


def test_no_scene_is_named_when_travel_stays_under_the_thinnest_wall(tiny_root):
    stats = manifest_of(tiny_root)["splits"]["train"]["stats"]
    assert max(stats["travel_by_scene"]) < stats["thinnest_wall_cells"]
    assert stats["scenes_over_thinnest_wall"] == []


def test_manifest_records_how_float16_compares_with_the_change_per_frame(tiny_root):
    """float16 rounds each value, so the rounding must stay small against what the model learns."""
    stats = manifest_of(tiny_root)["splits"]["train"]["stats"]
    assert len(stats["rounding_rms"]) == 3
    assert len(stats["frame_change_rms"]) == 3
    for rounding, change in zip(stats["rounding_rms"], stats["frame_change_rms"]):
        assert 0 < rounding < change


def test_manifest_records_realised_late_share_and_test_c_sides(tmp_path, monkeypatch):
    monkeypatch.setattr(scenes, "LATE_SHARE", 1.0)
    monkeypatch.setattr(scenes, "LATE_FRAMES", (2, 3))
    root = tmp_path / "data"
    generate(tiny_config(frames=4), root, allow_dirty=True)
    splits = manifest_of(root)["splits"]
    assert splits["train"]["late_scenes"] == 3
    assert splits["train"]["late_share"] == 1.0
    assert set(splits["train"]["late_frames"]) <= {2, 3}
    assert splits["test_b"]["late_scenes"] == 0
    assert splits["test_c"]["sides"]["low"] + splits["test_c"]["sides"]["high"] == 2
    assert "sides" not in splits["train"]


def test_manifest_counts_test_c_sides_from_buoyancy(tmp_path):
    """An odd count can't split evenly, so swapping the sides would change these numbers."""
    root = tmp_path / "data"
    config = tiny_config(frames=2, fine_frames=[], scenes={"train": 2, "val": 1, "test_a": 1, "test_b": 1, "test_c": 3})
    generate(config, root, allow_dirty=True)
    store = open_split(root, "test_c", final_eval=True)
    low = sum(store.buoyancy(i) < 1.0 for i in range(3))
    assert manifest_of(root)["splits"]["test_c"]["sides"] == {"low": low, "high": 3 - low}


def test_manifest_records_timing(tiny_root):
    timing = manifest_of(tiny_root)["timing"]
    for key in ("sampling", "solving", "writing", "assembling", "hashing", "normalizer"):
        assert timing["seconds"][key] >= 0
    assert timing["seconds"]["solving"] > 0
    per_split = timing["by_split"]["train"]
    assert per_split["scenes"] == 3
    assert per_split["solving"] > 0
    assert timing["unreadable_lines"] == 0


def test_missing_gpu_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GenerationError, match="GPU"):
        generate(tiny_config(), tmp_path / "data", allow_dirty=True)
