"""The spot-check's rule for travel above the thinnest wall (``docs/PREREGISTRATION.md``).

A scene named in the manifest is restarted from the frame of its largest travel, at the data's
step size and at half of it, and the difference in density is compared with that of the split's
quietest scenes. Starting both from one state leaves out the chaotic difference that builds up over
a whole run. Scenes come from their seeds, so the rule reads the manifest and no split's files.
"""

import math

import numpy as np
import torch

from sandeha.data.generate import scene_seed, travel_cells
from sandeha.physics.scenes import SPLITS, Scene, sample_scene
from sandeha.physics.solver import SmokeSolver, SmokeState

FORK_FRAMES = 5
CONTROLS = 3
PASS_FACTOR = 2.0  # A named scene passes at up to this many times the largest control.
DEVICE = torch.device("cuda")


def _empty_state(count: int, res: int) -> SmokeState:
    return SmokeState(
        torch.zeros(count, 2, res, res, device=DEVICE), torch.zeros(count, 1, res, res, device=DEVICE)
    )


def _solver(config: dict, dt: float) -> SmokeSolver:
    return SmokeSolver(
        config["res"],
        DEVICE,
        dt,
        cg_tol=config.get("cg_tol", 1e-5),
        cg_max_iter=config.get("cg_max_iter", 2000),
    )


def travel_by_frame(batch: list[Scene], config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each scene's largest travel in each frame, and whether all of its solves converged."""
    res, frames, substeps = config["res"], config["frames"], config["steps_per_frame"]

    def tensor(arrays) -> torch.Tensor:
        return torch.from_numpy(np.stack(arrays))[:, None].to(DEVICE)

    obstacle = tensor([s.obstacle for s in batch])
    late = tensor([np.zeros((res, res), bool) if s.late_obstacle is None else s.late_obstacle for s in batch])
    # Frames beyond the scene never arrive, so a scene with no late obstacle can use the same rule.
    late_frame = torch.tensor(
        [frames if s.late_frame is None else s.late_frame for s in batch], device=DEVICE
    ).view(-1, 1, 1, 1)
    inflow = tensor([s.inflow for s in batch])
    lift = torch.tensor([s.buoyancy for s in batch], device=DEVICE, dtype=torch.float32)
    solver = _solver(config, config["dt"])
    state = _empty_state(len(batch), res)
    travel = torch.zeros(len(batch), frames - 1, device=DEVICE)
    converged = torch.ones(len(batch), dtype=torch.bool, device=DEVICE)
    for frame in range(frames - 1):
        solid = obstacle | (late & (late_frame <= frame))
        for _ in range(substeps):
            state = solver.step(state, solid, inflow, lift)
            converged &= solver.last_solve.converged
            travel[:, frame] = torch.maximum(travel[:, frame], travel_cells(state.mac, config["dt"], res))
    return travel.cpu().numpy(), converged.cpu().numpy()


def fork_frame(frame: int, frames: int) -> int:
    """The frame to restart from: the one given, moved back if the comparison would leave the scene."""
    return max(min(frame, frames - 1 - FORK_FRAMES), 0)


def step_size_difference(scene: Scene, config: dict, fork: int) -> list[float]:
    """RMS density difference over fluid cells, relative to RMS density, from a shared state."""
    dt, substeps = config["dt"], config["steps_per_frame"]
    obstacle = torch.from_numpy(scene.obstacle)[None, None].to(DEVICE)
    inflow = torch.from_numpy(scene.inflow)[None, None].to(DEVICE)
    lift = torch.tensor([scene.buoyancy], device=DEVICE, dtype=torch.float32)

    def solid_at(frame: int) -> torch.Tensor:
        if scene.late_obstacle is None or frame < scene.late_frame:
            return obstacle
        return obstacle | torch.from_numpy(scene.late_obstacle)[None, None].to(DEVICE)

    solver = _solver(config, dt)
    shared = _empty_state(1, config["res"])
    for frame in range(fork):
        for _ in range(substeps):
            shared = solver.step(shared, solid_at(frame), inflow, lift)

    def carry_on(step: float, count: int) -> list[torch.Tensor]:
        fine = _solver(config, step)
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


def _largest(values: list[float]) -> float:
    """The largest value, or NaN if any is. Python's ``max`` can pass over a NaN."""
    return float(np.max(values))


def _figure(scene: Scene, config: dict, frame: int) -> dict:
    fork = fork_frame(frame, config["frames"])
    by_frame = step_size_difference(scene, config, fork)
    return {"fork_frame": fork, "density_difference_by_frame": by_frame, "figure": _largest(by_frame)}


def _judge_split(split: str, entry: dict, config: dict) -> dict:
    stats = entry["stats"]
    travel, frames = stats["travel_by_scene"], stats["travel_frame_by_scene"]
    wall, chosen = stats["thinnest_wall_cells"], stats["scenes_over_thinnest_wall"]
    largest = stats["max_travel_cells_per_step"]
    result = {
        "largest_travel": largest,
        "thinnest_wall_cells": wall,
        "margin_to_thinnest_wall": (wall - largest) / wall,
        "named": [],
        "controls": [],
        "passed": True,
    }
    if not chosen:
        return result

    def measure(i: int) -> dict:
        scene = sample_scene(np.random.default_rng(entry["seeds"][i]), config["res"], split)
        return {"scene": i, "largest_travel": travel[i], **_figure(scene, config, frames[i])}

    others = sorted((i for i in range(len(travel)) if i not in chosen), key=lambda i: travel[i])
    result["controls"] = [measure(i) for i in others[:CONTROLS]]
    if not result["controls"]:
        result["passed"] = False  # Nothing to compare with, so nothing shows that it's fine.
        return result
    limit = PASS_FACTOR * _largest([control["figure"] for control in result["controls"]])
    result["largest_control_figure"] = limit / PASS_FACTOR
    for i in chosen:
        named = measure(i)
        # A figure that isn't a number compares false, so it doesn't pass.
        named["passed"] = bool(named["figure"] <= limit)
        result["named"].append(named)
    result["passed"] = math.isfinite(limit) and all(named["passed"] for named in result["named"])
    return result


def apply_travel_rule(manifest: dict) -> dict:
    """Apply the rule to each split of a finished dataset, from its manifest alone."""
    config = manifest["config"]
    splits = {split: _judge_split(split, manifest["splits"][split], config) for split in SPLITS}
    return {
        "data_git_commit": manifest["git_commit"],
        "seed": config["seed"],
        "dt": config["dt"],
        "fork_frames": FORK_FRAMES,
        "pass_factor": PASS_FACTOR,
        "splits": splits,
        "data_accepted": all(entry["passed"] for entry in splits.values()),
    }


def measure_control_sample(config: dict, seed: int, scenes_per_split: int) -> dict:
    """The rule's controls on ordinary scenes of every kind, from a seed that isn't the dataset's."""
    splits = {}
    batch_size = config["batch_size"]
    for split in SPLITS:
        batch = [
            sample_scene(np.random.default_rng(scene_seed(seed, split, i)), config["res"], split)
            for i in range(scenes_per_split)
        ]
        largest, frames, converged = [], [], []
        for start in range(0, scenes_per_split, batch_size):
            values, ok = travel_by_frame(batch[start : start + batch_size], config)
            largest += values.max(axis=1).tolist()
            frames += values.argmax(axis=1).tolist()
            converged += ok.tolist()
        lowest = sorted(range(scenes_per_split), key=lambda i: largest[i])[:CONTROLS]
        controls = [
            {
                "scene": i,
                "late_frame": batch[i].late_frame,
                "largest_travel": largest[i],
                "largest_travel_frame": frames[i],
                **_figure(batch[i], config, frames[i]),
            }
            for i in lowest
        ]
        splits[split] = {
            "scene_travel": largest,
            "late_scenes": sum(scene.late_obstacle is not None for scene in batch),
            "all_converged": all(converged),
            "controls": controls,
            "largest_control_figure": _largest([control["figure"] for control in controls]),
        }
    return {"seed": seed, "scenes_per_split": scenes_per_split, "fork_frames": FORK_FRAMES, "splits": splits}
