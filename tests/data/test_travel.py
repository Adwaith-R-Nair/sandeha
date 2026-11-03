"""Checks on the spot-check's rule for travel above the thinnest wall."""

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sandeha.data import travel
from sandeha.data.travel import (
    apply_travel_rule,
    fork_frame,
    measure_control_sample,
    step_size_difference,
    travel_by_frame,
)
from sandeha.physics import scenes
from sandeha.physics.scenes import SPLITS, Scene, sample_scene

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
    # test_b has no late obstacles, so this is where the old tool stopped.
    assert result["splits"]["test_b"]["late_scenes"] == 0
