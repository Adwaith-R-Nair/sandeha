"""Checks on the spot-check: its rendering, its counts, and what makes it fail."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sandeha.data.spotcheck import count_non_finite, frames_to_show, run_spot_check
from sandeha.data.store import TamperedDataError
from sandeha.physics import scenes
from sandeha.physics.scenes import SPLITS

from conftest import tiny_config

REPO = Path(__file__).resolve().parents[2]


def test_non_finite_values_are_counted_in_every_scene():
    fields = np.zeros((2, 3, 3, 4, 4), dtype=np.float16)
    assert count_non_finite(fields) == 0
    fields[1, 2, 0, 3, 3] = np.nan
    fields[0, 0, 2, 0, 0] = np.inf
    fields[0, 1, 1, 1, 1] = -np.inf
    assert count_non_finite(fields) == 3


def test_frames_shown_stay_inside_the_scene():
    assert frames_to_show(121) == [0, 20, 45, 70, 120]
    assert frames_to_show(6) == [0, 5]


def test_spot_check_covers_every_split_and_passes_on_good_data(tiny_root, tmp_path):
    figure = tmp_path / "spot.png"
    result = run_spot_check(tiny_root, figure)
    assert result["passed"] is True
    assert set(result["splits"]) == set(SPLITS)
    manifest = json.loads((tiny_root / "manifest.json").read_text())
    assert result["manifest_git_commit"] == manifest["git_commit"]
    for split, entry in result["splits"].items():
        assert entry["non_finite_values"] == 0
        assert entry["n_scenes"] == tiny_config()["scenes"][split]
        assert entry["unconverged_item_steps"] == 0
        assert set(entry["channels"]) == {"u", "v", "density"}
        assert entry["channels"]["density"]["max"] > 0
        assert entry["rendered_scenes"][0] == 0
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_spot_check_renders_the_first_late_obstacle_scene_as_well(tmp_path, monkeypatch):
    from sandeha.data.generate import generate

    monkeypatch.setattr(scenes, "LATE_SHARE", 0.5)
    monkeypatch.setattr(scenes, "LATE_FRAMES", (2, 2))
    root = tmp_path / "data"
    config = tiny_config(scenes={"train": 6, "val": 1, "test_a": 1, "test_b": 1, "test_c": 1})
    generate(config, root, allow_dirty=True)
    manifest = json.loads((root / "manifest.json").read_text())
    train = manifest["splits"]["train"]
    late = [
        i
        for i, seed in enumerate(train["seeds"])
        if scenes.sample_scene(np.random.default_rng(seed), 64, "train").late_frame is not None
    ]
    assert late and late[0] != 0, "this test needs a first scene with no late obstacle"
    result = run_spot_check(root, tmp_path / "spot.png")
    assert result["splits"]["train"]["rendered_scenes"] == [0, late[0]]
    # test_b has no late obstacles, so it has only the one to show.
    assert result["splits"]["test_b"]["rendered_scenes"] == [0]


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest["config"]["scenes"].update(train=99),
        lambda manifest: manifest["splits"]["train"].update(n_frames=99),
    ],
)
def test_spot_check_fails_when_the_shape_is_not_the_manifest_s(tiny_root, tmp_path, change):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    manifest = json.loads((root / "manifest.json").read_text())
    change(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = run_spot_check(root, tmp_path / "spot.png")
    assert result["passed"] is False
    assert result["splits"]["train"]["shape_matches_manifest"] is False


def test_spot_check_fails_on_a_scene_with_a_nan(tiny_root, tmp_path, monkeypatch):
    from sandeha.data import spotcheck

    monkeypatch.setattr(spotcheck, "count_non_finite", lambda fields: 1)
    assert run_spot_check(tiny_root, tmp_path / "spot.png")["passed"] is False


def test_spot_check_refuses_a_changed_file(tiny_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    path = root / "test_b" / "fields.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(TamperedDataError):
        run_spot_check(root, tmp_path / "spot.png")


def test_spot_check_refuses_a_changed_normalizer(tiny_root, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(tiny_root, root)
    path = root / "normalizer.json"
    path.write_text(path.read_text().replace("0.", "1.", 1))
    with pytest.raises(TamperedDataError):
        run_spot_check(root, tmp_path / "spot.png")


def test_the_spot_check_command_writes_its_result(tiny_root, tmp_path):
    out = tmp_path / "spot.json"
    figure = tmp_path / "spot.png"
    process = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "spot_check.py"),
            "--data-root",
            str(tiny_root),
            "--out",
            str(out),
            "--figure",
            str(figure),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(out.read_text())["passed"] is True
    assert figure.exists()
