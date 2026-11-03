"""Measure travel per step, on ordinary late-obstacle train scenes, and how a step size changes it.

A late obstacle can throw up a one-off spike in speed. Each scene's largest travel is compared with
the thinnest wall. Then the largest-travel scene and low-travel controls are each restarted from
the frame of their largest travel, at the data's step size and at half of it, and their density is
compared over the next frames. Starting both from one state leaves out the chaotic difference that
builds up over a whole run. ``docs/PREREGISTRATION.md`` uses that figure for scenes above the wall.

The master seed is not the dataset's, so no locked scene is drawn. Writes results/travel_check.json.

Usage: uv run python scripts/check_travel.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from sandeha.data.generate import git_state, load_config, scene_seed
from sandeha.physics.scenes import MIN_THICKNESS, Scene, sample_scene
from sandeha.physics.solver import SmokeSolver, SmokeState

ROOT = Path(__file__).resolve().parents[1]
SEED = 777123  # Not the dataset's seed.
LATE_SCENES = 62
BATCH = 16
FORK_FRAMES = 5
CONTROLS = 3
DEVICE = torch.device("cuda")


def empty_state(count: int, res: int) -> SmokeState:
    return SmokeState(
        torch.zeros(count, 2, res, res, device=DEVICE), torch.zeros(count, 1, res, res, device=DEVICE)
    )


def travel_by_frame(batch: list[Scene], res: int, dt: float, frames: int, substeps: int) -> tuple:
    """Each scene's largest travel in each frame, and whether all of its solves converged."""

    def tensor(arrays):
        return torch.from_numpy(np.stack(arrays))[:, None].to(DEVICE)

    obstacle, late = tensor([s.obstacle for s in batch]), tensor([s.late_obstacle for s in batch])
    inflow = tensor([s.inflow for s in batch])
    late_frame = torch.tensor([s.late_frame for s in batch], device=DEVICE).view(-1, 1, 1, 1)
    lift = torch.tensor([s.buoyancy for s in batch], device=DEVICE, dtype=torch.float32)
    solver = SmokeSolver(res, DEVICE, dt)
    state = empty_state(len(batch), res)
    travel = torch.zeros(len(batch), frames - 1, device=DEVICE)
    converged = torch.ones(len(batch), dtype=torch.bool, device=DEVICE)
    for frame in range(frames - 1):
        solid = obstacle | (late & (late_frame <= frame))
        for _ in range(substeps):
            state = solver.step(state, solid, inflow, lift)
            converged &= solver.last_solve.converged
            u, v = state.mac
            fastest = torch.maximum(u.abs().amax(dim=(1, 2, 3)), v.abs().amax(dim=(1, 2, 3)))
            travel[:, frame] = torch.maximum(travel[:, frame], fastest * dt * res)
    return travel.cpu().numpy(), converged.cpu().numpy()


def step_size_difference(scene: Scene, res: int, dt: float, substeps: int, fork: int) -> list[float]:
    """RMS density difference over fluid cells, relative to RMS density, from a shared state."""
    obstacle = torch.from_numpy(scene.obstacle)[None, None].to(DEVICE)
    late = torch.from_numpy(scene.late_obstacle)[None, None].to(DEVICE)
    inflow = torch.from_numpy(scene.inflow)[None, None].to(DEVICE)
    lift = torch.tensor([scene.buoyancy], device=DEVICE, dtype=torch.float32)

    def solid_at(frame: int) -> torch.Tensor:
        return obstacle | late if scene.late_frame <= frame else obstacle

    solver = SmokeSolver(res, DEVICE, dt)
    shared = empty_state(1, res)
    for frame in range(fork):
        for _ in range(substeps):
            shared = solver.step(shared, solid_at(frame), inflow, lift)

    def carry_on(step: float, count: int) -> list[torch.Tensor]:
        fine = SmokeSolver(res, DEVICE, step)
        state, out = shared, []
        for frame in range(fork, fork + FORK_FRAMES):
            for _ in range(count):
                state = fine.step(state, solid_at(frame), inflow, lift)
            out.append(state.density[0, 0])
        return out

    before = shared.density.clone()
    coarse, fine = carry_on(dt, substeps), carry_on(dt / 2, 2 * substeps)
    assert torch.equal(before, shared.density), "a step must not change the state it starts from"
    result = []
    for k, (a, b) in enumerate(zip(coarse, fine)):
        fluid = ~solid_at(fork + k)[0, 0]
        result.append(float((a - b)[fluid].norm() / b[fluid].norm()))
    return result


def main() -> None:
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("commit first: this result has to trace to a commit")
    config = load_config(ROOT / "configs" / "data.yaml")
    res, dt, substeps, frames = config["res"], config["dt"], config["steps_per_frame"], config["frames"]
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
        travel, converged = travel_by_frame([s for _, s in group], res, dt, frames, substeps)
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
        fork = row["largest_travel_frame"]
        scene = scenes[row["scene_index"]]
        difference = step_size_difference(scene, res, dt, substeps, min(fork, frames - 1 - FORK_FRAMES))
        forks.append({**row, "density_difference_by_frame": difference, "figure": max(difference)})

    largest = by_travel[-1]["largest_travel"]
    result = {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": SEED,
        "dt": dt,
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
    path = ROOT / "results" / "travel_check.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print({key: value for key, value in result.items() if key != "scenes"})


if __name__ == "__main__":
    main()
