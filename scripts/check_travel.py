"""Check travel per step against the thinnest wall, and how a step size changes the flow.

Three uses, each writing one JSON result:

- With ``--data-root``, apply the rule in ``docs/PREREGISTRATION.md`` to a finished dataset: this
  is the spot-check's step. It reads the manifest and reruns named scenes and controls from their
  seeds, and exits with status 1 if the data isn't accepted. Writes results/travel_rule.json, and a
  rendering of each named scene to results/figures/, as the rule's step 4 requires.
- With ``--control-sample``, measure the rule's controls on scenes of every kind and split, from a
  seed that isn't the dataset's. Writes results/travel_controls.json.
- With ``--recheck-largest``, rerun the largest-travel scene of results/travel_check.json on the
  current measure, alone. Writes results/travel_recheck.json. Takes under a minute.
- With none, the sample behind results/travel_check.json: ordinary late-obstacle train scenes,
  where a late obstacle can throw up a one-off spike in speed, from the same seed.

Usage: uv run python scripts/check_travel.py [--data-root data | --control-sample | --recheck-largest]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sandeha.data.generate import git_state, load_config, scene_seed
from sandeha.data.store import read_manifest
from sandeha.data.travel import (
    CONTROLS,
    FORK_FRAMES,
    apply_travel_rule,
    fork_frame,
    measure_control_sample,
    render_named_scenes,
    step_size_difference,
    travel_by_frame,
)
from sandeha.physics.scenes import MIN_THICKNESS, sample_scene

ROOT = Path(__file__).resolve().parents[1]
SEED = 777123  # Not the dataset's seed.
LATE_SCENES = 62
BATCH = 16
SCENES_PER_SPLIT = 16


def write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=1) + "\n")


def late_sample(config: dict, commit: str, dirty: bool) -> dict:
    res, frames = config["res"], config["frames"]
    wall = MIN_THICKNESS * res

    chosen = []
    index = 0
    while len(chosen) < LATE_SCENES:
        scene = sample_scene(np.random.default_rng(scene_seed(SEED, "train", index)), res, "train")
        if scene.late_obstacle is not None:
            chosen.append((index, scene))
        index += 1

    rows = []
    for start in range(0, LATE_SCENES, BATCH):
        group = chosen[start : start + BATCH]
        travel, converged = travel_by_frame([s for _, s in group], config)
        for k, (scene_index, scene) in enumerate(group):
            # The event's own spike comes in the frame out of late_frame, and may last a little.
            outside = np.ones(frames - 1, dtype=bool)
            outside[scene.late_frame : scene.late_frame + 3] = False
            rows.append(
                {
                    "scene_index": scene_index,
                    "late_frame": scene.late_frame,
                    "largest_travel": float(travel[k].max()),
                    "largest_travel_frame": int(travel[k].argmax()),
                    "largest_travel_outside_event": float(travel[k][outside].max()),
                    "converged": bool(converged[k]),
                }
            )
        print(f"{start + len(group)} of {LATE_SCENES}", flush=True)

    by_travel = sorted(rows, key=lambda row: row["largest_travel"])
    scenes = dict(chosen)
    forks = []
    for row in by_travel[-1:] + by_travel[:CONTROLS]:
        fork = fork_frame(row["largest_travel_frame"], frames)
        difference = step_size_difference(scenes[row["scene_index"]], config, fork)
        forks.append({**row, "density_difference_by_frame": difference, "figure": max(difference)})

    largest = by_travel[-1]["largest_travel"]
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": SEED,
        "dt": config["dt"],
        "thinnest_wall_cells": wall,
        "draws_to_find_late_scenes": index,
        "scenes": rows,
        "largest_travel": largest,
        "second_largest_travel": by_travel[-2]["largest_travel"],
        "margin_to_thinnest_wall": (wall - largest) / wall,
        "scenes_above_thinnest_wall": sum(row["largest_travel"] > wall for row in rows),
        "all_converged": all(row["converged"] for row in rows),
        "step_size_forks": {
            "frames": FORK_FRAMES,
            "largest_travel_scene": forks[0],
            "controls": forks[1:],
            "largest_control_figure": max(fork["figure"] for fork in forks[1:]),
        },
    }


def recheck_largest(config: dict, saved: dict) -> dict:
    """The saved sample's largest-travel scene, run alone on the current measure.

    Alone, not in its first batch of 16, so its arithmetic isn't exactly that run's.
    """
    row = max(saved["scenes"], key=lambda scene: scene["largest_travel"])
    seed = scene_seed(SEED, "train", row["scene_index"])
    scene = sample_scene(np.random.default_rng(seed), config["res"], "train")
    travel, converged = travel_by_frame([scene], config)
    wall = MIN_THICKNESS * config["res"]
    largest = float(travel[0].max())
    return {
        "saved_git_commit": saved["git_commit"],
        "scene_index": row["scene_index"],
        "late_frame": scene.late_frame,
        "saved_travel_old_measure": row["largest_travel"],
        "saved_frame_old_measure": row["largest_travel_frame"],
        "travel_current_measure": largest,
        "frame_current_measure": int(travel[0].argmax()),
        "thinnest_wall_cells": wall,
        "margin_to_thinnest_wall": (wall - largest) / wall,
        "above_thinnest_wall": largest > wall,
        "converged": bool(converged[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="apply the rule to this finished dataset")
    parser.add_argument("--control-sample", action="store_true", help="measure controls on every kind of scene")
    parser.add_argument("--recheck-largest", action="store_true", help="rerun the saved sample's largest scene")
    parser.add_argument("--out", type=Path, help="where to write the result")
    args = parser.parse_args()
    commit, dirty = git_state()

    if args.data_root is not None:
        # The new manifest's tracked copy isn't committed until the spot-check ends, so a tree
        # with that change is the normal case here. The result records the state it ran in.
        manifest = read_manifest(args.data_root)
        result = {"git_commit": commit, "git_dirty": dirty, **apply_travel_rule(manifest)}
        out = args.out or ROOT / "results" / "travel_rule.json"
        render_named_scenes(result, manifest, out.parent / "figures", out.parent)
        write_result(out, result)
        print(json.dumps(result, indent=1))
        if not result["data_accepted"]:
            raise SystemExit("not accepted: see the rule in docs/PREREGISTRATION.md")
        return

    if dirty:
        raise SystemExit("commit first: this result has to trace to a commit")
    config = load_config(ROOT / "configs" / "data.yaml")
    if args.control_sample:
        result = {"git_commit": commit, "git_dirty": dirty, **measure_control_sample(config, SEED, SCENES_PER_SPLIT)}
        write_result(args.out or ROOT / "results" / "travel_controls.json", result)
    elif args.recheck_largest:
        saved = json.loads((ROOT / "results" / "travel_check.json").read_text())
        result = {"git_commit": commit, "git_dirty": dirty, **recheck_largest(config, saved)}
        write_result(args.out or ROOT / "results" / "travel_recheck.json", result)
    else:
        result = late_sample(config, commit, dirty)
        write_result(args.out or ROOT / "results" / "travel_check.json", result)
    print({key: value for key, value in result.items() if key != "scenes"})


if __name__ == "__main__":
    main()
