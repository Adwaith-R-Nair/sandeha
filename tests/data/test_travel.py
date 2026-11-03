"""Checks on the spot-check's rule for travel above the thinnest wall."""

import copy
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sandeha.data import generate as generate_module
from sandeha.data import travel
from sandeha.data.generate import _simulate_batch, scene_seed
from sandeha.data.travel import (
    ControlsNotReady,
    apply_travel_rule,
    check_controls,
    controls_gate_applies,
    fork_frame,
    measure_control_sample,
    render_restart,
    restart_runs,
    step_size_difference,
    travel_by_frame,
)
from sandeha.physics import scenes
from sandeha.physics.scenes import SPLITS, TRAIN_BUOYANCY, Scene, sample_scene
from sandeha.physics.solver import SmokeSolver

from conftest import tiny_config

REPO = Path(__file__).resolve().parents[2]


def a_scene(split: str = "test_b", seed: int = 3) -> Scene:
    return sample_scene(np.random.default_rng(seed), 64, split)


def fake_manifest(travel_values: list[float], named: list[int], frames: list[int] | None = None) -> dict:
    """One split with these travels. The other splits have one quiet scene and name nothing."""
    config = tiny_config(frames=121)
    splits = {}
    for split in SPLITS:
        values, chosen = (travel_values, named) if split == "test_b" else ([1.0], [])
        count = len(values)
        splits[split] = {
            "seeds": list(range(100, 100 + count)),
            "stats": {
                "max_travel_cells_per_step": max(values),
                "travel_by_scene": values,
                "travel_frame_by_scene": frames if frames and split == "test_b" else [40] * count,
                "thinnest_wall_cells": 2.56,
                "scenes_over_thinnest_wall": chosen,
            },
        }
    return {"git_commit": "a" * 40, "config": config, "splits": splits}


def fake_figures(monkeypatch, figures: dict[tuple[int, int], float]) -> list:
    """Stand in for the restart. A scene is known by its seed, and its figure by (seed, fork)."""
    calls = []
    by_obstacle = {}
    for seed, _ in figures:
        scene = sample_scene(np.random.default_rng(seed), 64, "test_b")
        by_obstacle[scene.obstacle.tobytes()] = seed

    def difference(scene, config, fork):
        seed = by_obstacle[scene.obstacle.tobytes()]
        calls.append((seed, fork))
        return [0.0, figures[seed, fork]]

    monkeypatch.setattr(travel, "step_size_difference", difference)
    return calls


@pytest.mark.parametrize("split", ["test_b", "test_c"])
def test_restart_works_on_a_scene_with_no_late_obstacle(split):
    """Every test_b and test_c scene is like this, and so are most controls elsewhere."""
    scene = a_scene(split)
    assert scene.late_obstacle is None
    figures = step_size_difference(scene, tiny_config(), 2)
    assert len(figures) == travel.FORK_FRAMES
    assert all(math.isfinite(value) and value > 0 for value in figures)


def test_restart_applies_a_late_obstacle_at_its_frame():
    plain = a_scene()
    middle = np.zeros_like(plain.obstacle)
    middle[30:40, 25:40] = True
    def with_late(late_frame):
        return Scene(plain.obstacle, plain.inflow, plain.buoyancy, middle & ~plain.obstacle, late_frame)

    config = tiny_config()
    before = step_size_difference(plain, config, 8)
    assert step_size_difference(with_late(50), config, 8) == before
    assert step_size_difference(with_late(10), config, 8) != before
    assert step_size_difference(with_late(5), config, 8) != step_size_difference(with_late(10), config, 8)


def test_restart_reproduces_the_frames_generation_stores(monkeypatch):
    """On either side of the late obstacle, the restart's own step size is the data's run."""
    monkeypatch.setattr(scenes, "LATE_SHARE", 1.0)
    monkeypatch.setattr(scenes, "LATE_FRAMES", (6, 6))
    config = tiny_config(frames=13, batch_size=1, fine_frames=[])
    scene = sample_scene(np.random.default_rng(5), 64, "train")
    assert scene.late_frame == 6
    device = torch.device("cuda")
    stored = _simulate_batch(config, [scene], device)["fields"][0, :, 2]
    old = torch.from_numpy(scene.obstacle)[None, None].to(device)
    new = old | torch.from_numpy(scene.late_obstacle)[None, None].to(device)

    def as_stored(density, frame):
        solid = new if frame - 1 >= scene.late_frame else old
        coarse_solid = F.max_pool2d(solid.float(), 2) > 0
        return (F.avg_pool2d(density[None, None], 2) * ~coarse_solid).half().cpu().numpy()[0, 0]

    for fork in (4, 5, 6, 7):
        runs = restart_runs(scene, config, fork)
        assert runs.shared_frames[-1] == fork
        for frame, density in zip(runs.shared_frames, runs.shared):
            assert np.array_equal(as_stored(density, frame), stored[frame]), (fork, frame)
        assert runs.frames == list(range(fork + 1, fork + 1 + travel.FORK_FRAMES))
        for frame, density in zip(runs.frames, runs.coarse):
            assert np.array_equal(as_stored(density, frame), stored[frame]), (fork, frame)
        assert not np.array_equal(
            as_stored(runs.fine[-1], runs.frames[-1]), stored[runs.frames[-1]]
        ), "the other step size should differ"


def test_both_step_sizes_cover_the_same_physical_time(monkeypatch):
    """Half the time at dt / 2 would make the figure mostly a difference in elapsed time."""
    steps = []

    class Counting(SmokeSolver):
        def __init__(self, res, device, dt, **settings):
            super().__init__(res, device, dt, **settings)
            self.counted_dt = dt

        def step(self, *args):
            steps.append(self.counted_dt)
            return super().step(*args)

    monkeypatch.setattr(travel, "SmokeSolver", Counting)
    step_size_difference(a_scene(), tiny_config(), 3)
    shared, coarse, fine = steps[:12], steps[12:32], steps[32:]
    assert shared == [0.01] * 12, "3 frames of 4 steps, up to the restart"
    assert coarse == [0.01] * 20, "5 frames of 4 steps"
    assert fine == [0.005] * 40, "the same 5 frames, at half the step"


def test_the_difference_is_measured_over_the_fluid_of_each_frame():
    plain = a_scene()
    middle = np.zeros_like(plain.obstacle)
    middle[30:40, 25:40] = True
    scene = Scene(plain.obstacle, plain.inflow, plain.buoyancy, middle & ~plain.obstacle, 6)
    config = tiny_config()
    figures = step_size_difference(scene, config, 4)
    runs = restart_runs(scene, config, 4)
    solid = torch.from_numpy(scene.obstacle | scene.late_obstacle).to(runs.coarse[0].device)
    last = ~solid
    expected = (runs.coarse[-1] - runs.fine[-1])[last].norm() / runs.fine[-1][last].norm()
    assert figures[-1] == pytest.approx(float(expected))


def test_a_rendering_shows_the_frames_around_the_spike(tmp_path):
    scene = a_scene()
    path = tmp_path / "figures" / "scene.png"
    shown = render_restart(scene, tiny_config(), 3, path, title="a scene")
    assert path.read_bytes().startswith(b"\x89PNG")
    assert shown == [1, 2, 3, 4, 5, 6, 7, 8]


def test_a_rendering_at_the_start_of_a_scene_has_no_frames_before_it(tmp_path):
    shown = render_restart(a_scene(), tiny_config(), 0, tmp_path / "scene.png", title="a scene")
    assert shown == [0, 1, 2, 3, 4, 5]


def test_restart_uses_the_config_s_solver_settings():
    scene = a_scene()
    assert step_size_difference(scene, tiny_config(cg_tol=1e-1), 2) != step_size_difference(scene, tiny_config(), 2)


def test_travel_by_frame_takes_scenes_with_and_without_a_late_obstacle(monkeypatch):
    monkeypatch.setattr(scenes, "LATE_SHARE", 1.0)
    monkeypatch.setattr(scenes, "LATE_FRAMES", (1, 1))
    batch = [a_scene("train"), a_scene("test_b")]
    assert batch[0].late_obstacle is not None and batch[1].late_obstacle is None
    values, converged = travel_by_frame(batch, tiny_config())
    assert values.shape == (2, 5)
    assert (values > 0).any(axis=1).all()
    assert converged.all()


@pytest.mark.parametrize("frame, frames, expected", [(40, 121, 40), (115, 121, 115), (116, 121, 115), (119, 121, 115), (3, 6, 0)])
def test_a_restart_stays_inside_the_scene(frame, frames, expected):
    assert fork_frame(frame, frames) == expected


def test_the_rule_moves_a_restart_back_from_the_scene_s_end(monkeypatch):
    """Frames 119 and 117 of 121 would run past the end, so both restart from frame 115."""
    manifest = fake_manifest([3.0, 1.0], [0], frames=[119, 117])
    calls = fake_figures(monkeypatch, {(100, 115): 0.1, (101, 115): 0.1})
    entry = apply_travel_rule(manifest)["splits"]["test_b"]
    assert sorted(calls) == [(100, 115), (101, 115)]
    assert entry["named"][0]["fork_frame"] == 115
    assert entry["controls"][0]["fork_frame"] == 115


def test_a_figure_is_the_largest_of_its_frames_wherever_that_falls(monkeypatch):
    manifest = fake_manifest([3.0, 1.0], [0])
    fake_figures(monkeypatch, {(100, 40): 0.3, (101, 40): 0.2})
    original = travel.step_size_difference

    def spike_first(scene, config, fork):
        return list(reversed(original(scene, config, fork)))

    monkeypatch.setattr(travel, "step_size_difference", spike_first)
    entry = apply_travel_rule(manifest)["splits"]["test_b"]
    assert entry["named"][0]["figure"] == 0.3
    assert entry["controls"][0]["figure"] == 0.2


def test_test_c_figures_state_each_scene_s_side(monkeypatch):
    config = tiny_config(frames=121)
    seeds = {}
    for seed in range(100, 200):
        side = "low" if sample_scene(np.random.default_rng(seed), 64, "test_c").buoyancy < TRAIN_BUOYANCY[0] else "high"
        seeds.setdefault(side, seed)
    assert set(seeds) == {"low", "high"}
    manifest = fake_manifest([1.0], [])
    manifest["config"] = config
    manifest["splits"]["test_c"] = {
        "seeds": [seeds["high"], seeds["low"]],
        "stats": {
            "max_travel_cells_per_step": 3.0,
            "travel_by_scene": [3.0, 1.0],
            "travel_frame_by_scene": [40, 40],
            "thinnest_wall_cells": 2.56,
            "scenes_over_thinnest_wall": [0],
        },
    }
    monkeypatch.setattr(travel, "step_size_difference", lambda scene, config, fork: [0.1])
    entry = apply_travel_rule(manifest)["splits"]["test_c"]
    assert entry["named"][0]["side"] == "high"
    assert entry["controls"][0]["side"] == "low"


def test_nothing_named_passes_with_no_restart(monkeypatch):
    def refuse(*args):
        raise AssertionError("no scene is named, so nothing should run")

    monkeypatch.setattr(travel, "step_size_difference", refuse)
    result = apply_travel_rule(fake_manifest([1.0, 2.0], []))
    assert result["data_accepted"] is True
    assert result["splits"]["test_b"]["passed"] is True
    assert result["splits"]["test_b"]["largest_travel"] == 2.0
    assert result["splits"]["test_b"]["margin_to_thinnest_wall"] == pytest.approx((2.56 - 2.0) / 2.56)


@pytest.mark.parametrize("figure, passed", [(0.2, True), (0.2001, False)])
def test_named_scene_against_twice_the_largest_control(monkeypatch, figure, passed):
    """Scene 4 is named. Scenes 1, 2 and 0 are lowest. Scene 3 is higher, so it has no say."""
    manifest = fake_manifest([1.2, 1.0, 1.1, 2.0, 3.0], [4], frames=[10, 11, 12, 13, 14])
    calls = fake_figures(
        monkeypatch,
        {(104, 14): figure, (101, 11): 0.05, (102, 12): 0.1, (100, 10): 0.07},
    )
    result = apply_travel_rule(manifest)
    entry = result["splits"]["test_b"]
    assert sorted(calls) == [(100, 10), (101, 11), (102, 12), (104, 14)]
    assert entry["largest_control_figure"] == 0.1
    assert [control["scene"] for control in entry["controls"]] == [1, 2, 0]
    assert entry["named"][0]["figure"] == figure
    assert entry["named"][0]["passed"] is passed
    assert entry["passed"] is passed
    assert result["data_accepted"] is passed


def test_a_named_scene_is_not_its_own_control(monkeypatch):
    manifest = fake_manifest([3.0, 1.0, 3.5, 1.1, 1.2], [0, 2])
    calls = fake_figures(
        monkeypatch,
        {(seed, 40): 0.1 for seed in range(100, 105)},
    )
    entry = apply_travel_rule(manifest)["splits"]["test_b"]
    assert [control["scene"] for control in entry["controls"]] == [1, 3, 4]
    assert [named["scene"] for named in entry["named"]] == [0, 2]
    assert len(calls) == 5


def test_a_split_with_no_control_is_not_accepted(monkeypatch):
    calls = fake_figures(monkeypatch, {(100, 40): 0.1})
    result = apply_travel_rule(fake_manifest([3.0], [0]))
    assert result["splits"]["test_b"]["passed"] is False
    assert result["data_accepted"] is False
    assert calls == []


@pytest.mark.parametrize("named_figure, control_figure", [(float("nan"), 0.1), (0.1, float("nan"))])
def test_a_figure_that_is_not_a_number_does_not_pass(monkeypatch, named_figure, control_figure):
    manifest = fake_manifest([3.0, 1.0], [0])
    fake_figures(monkeypatch, {(100, 40): named_figure, (101, 40): control_figure})
    assert apply_travel_rule(manifest)["data_accepted"] is False


def named_copy(tiny_root, tmp_path, split: str, named: int) -> Path:
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"][split]["stats"]["scenes_over_thinnest_wall"] = [named]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return root


@pytest.mark.parametrize("split", ["test_c", "train"])
def test_the_rule_runs_on_a_dataset_s_named_scene(tiny_root, tmp_path, split):
    """The command the spot-check uses, with real restarts. test_c has no late obstacles."""
    root = named_copy(tiny_root, tmp_path, split, 1)
    # Nothing but the manifest is read, so these can't be a way into a locked split's fields.
    for other in SPLITS:
        for name in ("fields.npy", "statics.npz", "fine.npz"):
            (root / other / name).unlink()
    out = tmp_path / "rule.json"
    process = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_travel.py"), "--data-root", str(root), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    result = json.loads(out.read_text())
    assert process.returncode == (0 if result["data_accepted"] else 1), process.stderr
    entry = result["splits"][split]
    assert entry["named"][0]["scene"] == 1
    # Rule 4 logs a rendering of each named scene, beside the result and not in the repository.
    assert (tmp_path / entry["named"][0]["rendering"]).read_bytes().startswith(b"\x89PNG")
    assert len(entry["controls"]) >= 1
    assert all(control["scene"] != 1 for control in entry["controls"])
    assert math.isfinite(entry["named"][0]["figure"])
    assert result["seed"] == 11
    assert len(result["data_git_commit"]) == 40


def test_the_rule_needs_no_gpu_time_when_nothing_is_named(tiny_root):
    manifest = json.loads((tiny_root / "manifest.json").read_text())
    untouched = copy.deepcopy(manifest)
    result = apply_travel_rule(manifest)
    assert manifest == untouched
    assert result["data_accepted"] is True
    assert set(result["splits"]) == set(SPLITS)
    for split, entry in result["splits"].items():
        assert entry["largest_travel"] == manifest["splits"][split]["stats"]["max_travel_cells_per_step"]


def test_control_sample_covers_every_split_and_all_kinds_of_scene():
    config = tiny_config(frames=7)
    result = measure_control_sample(config, seed=777123, scenes_per_split=4)
    assert set(result["splits"]) == set(SPLITS)
    for split, entry in result["splits"].items():
        assert len(entry["scene_travel"]) == 4
        assert len(entry["controls"]) == travel.CONTROLS
        lowest = sorted(entry["scene_travel"])[: travel.CONTROLS]
        assert [control["largest_travel"] for control in entry["controls"]] == lowest
        assert all(math.isfinite(control["figure"]) for control in entry["controls"])
        for control in entry["controls"]:
            assert control["fork_frame"] == fork_frame(control["largest_travel_frame"], 7)
    # test_b has no late obstacles, so this is where the old tool stopped.
    assert result["splits"]["test_b"]["late_scenes"] == 0


def test_control_sample_gives_test_c_controls_for_each_side():
    """The lowest travel overall is likely all low side, so that alone can't judge the high side."""
    config = tiny_config(frames=7)
    entry = measure_control_sample(config, seed=777123, scenes_per_split=6)["splits"]["test_c"]
    assert set(entry["sides"]) == {"low", "high"}
    assert sum(len(side["scenes"]) for side in entry["sides"].values()) == 6
    for name, side in entry["sides"].items():
        assert len(side["controls"]) == min(travel.CONTROLS, len(side["scenes"]))
        assert all(control["side"] == name for control in side["controls"])
        travels = [entry["scene_travel"][i] for i in side["scenes"]]
        assert [c["largest_travel"] for c in side["controls"]] == sorted(travels)[: travel.CONTROLS]


def write_controls(path: Path, config: dict) -> Path:
    result = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config": config,
        "splits": {split: {"largest_control_figure": 0.1} for split in SPLITS},
    }
    path.write_text(json.dumps(result))
    return path


def test_generation_needs_the_controls_to_be_measured(tmp_path):
    with pytest.raises(ControlsNotReady, match="control-sample"):
        check_controls(tiny_config(), tmp_path / "missing.json")


def test_measured_controls_open_the_gate(tmp_path):
    config = tiny_config(frames=7)
    result = measure_control_sample(config, seed=777123, scenes_per_split=4)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps({"git_commit": "a" * 40, "git_dirty": False, **result}))
    check_controls(config, path)
    check_controls({**config, "scenes": {split: 1 for split in SPLITS}}, path)


@pytest.mark.parametrize(
    "change",
    [
        {"dt": 0.005, "steps_per_frame": 8},
        {"res": 128, "coarse_res": 64},
        {"frames": 61},
        {"cg_tol": 1e-3},
    ],
)
def test_controls_from_other_physics_do_not_open_the_gate(tmp_path, change):
    path = write_controls(tmp_path / "controls.json", tiny_config())
    with pytest.raises(ControlsNotReady, match="different"):
        check_controls(tiny_config(**change), path)


def test_controls_with_a_dirty_tree_or_no_figure_do_not_open_the_gate(tmp_path):
    path = write_controls(tmp_path / "controls.json", tiny_config())
    result = json.loads(path.read_text())
    path.write_text(json.dumps({**result, "git_dirty": True}))
    with pytest.raises(ControlsNotReady, match="commit"):
        check_controls(tiny_config(), path)

    result["splits"]["test_b"]["largest_control_figure"] = float("nan")
    path.write_text(json.dumps(result))
    with pytest.raises(ControlsNotReady, match="test_b"):
        check_controls(tiny_config(), path)

    del result["splits"]["test_c"]
    path.write_text(json.dumps(result))
    with pytest.raises(ControlsNotReady, match="test_c"):
        check_controls(tiny_config(), path)


def test_only_a_scratch_run_outside_the_data_root_skips_the_gate(tmp_path):
    assert controls_gate_applies(tmp_path, allow_dirty=False) is True
    assert controls_gate_applies(tmp_path, allow_dirty=True) is False
    assert controls_gate_applies(REPO / "data", allow_dirty=True) is True
    assert controls_gate_applies(REPO / "data", allow_dirty=False) is True


def test_the_generation_command_stops_at_the_gate(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(tiny_config()))
    root = tmp_path / "data"
    process = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_data.py"),
            "--config",
            str(config_path),
            "--root",
            str(root),
            "--controls",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "control-sample" in process.stderr
    assert not root.exists()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recheck_measures_one_saved_scene_on_the_current_measure():
    check_travel = load_script("check_travel")
    config = tiny_config(frames=7)
    saved = {
        "git_commit": "b" * 40,
        "thinnest_wall_cells": 2.56,
        "largest_travel": 1.5,
        "scenes": [
            {"scene_index": 2, "largest_travel": 1.0, "largest_travel_frame": 1},
            {"scene_index": 5, "largest_travel": 1.5, "largest_travel_frame": 3},
        ],
    }
    result = check_travel.recheck_largest(config, saved)
    assert result["scene_index"] == 5
    assert result["saved_travel_old_measure"] == 1.5
    assert result["saved_git_commit"] == "b" * 40
    assert result["travel_current_measure"] > 0
    assert result["above_thinnest_wall"] is (result["travel_current_measure"] > 2.56)
    scene = sample_scene(np.random.default_rng(scene_seed(check_travel.SEED, "train", 5)), 64, "train")
    values, _ = travel_by_frame([scene], config)
    assert result["travel_current_measure"] == float(values.max())


def test_the_estimate_sample_measures_its_scene_files(tmp_path, monkeypatch):
    estimate = load_script("estimate_generation")
    monkeypatch.setattr(generate_module, "git_state", lambda: ("a" * 40, False))
    config = tiny_config()
    sample = estimate.run_sample(config, config["scenes"], 2, tmp_path)
    scenes_total = sum(config["scenes"].values())
    assert sample["scene_file_bytes"] > 0
    disk = estimate.peak_disk(sample, scenes_total * 4)
    assert disk == (sample["scene_file_bytes"] + sum(e["bytes"] for e in sample["splits"].values())) * 4
