"""The spot-check's rule for travel above the thinnest wall (``docs/PREREGISTRATION.md``).

A scene named in the manifest is restarted from the frame of its largest travel, at the data's
step size and at half of it, and the difference in density is compared with that of the split's
quietest scenes. Starting both from one state leaves out the chaotic difference that builds up over
a whole run. Scenes come from their seeds, so the rule reads the manifest and no split's files.

The controls must be measured before generation, and ``check_controls`` is what enforces that.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sandeha.data.generate import scene_seed, travel_cells
from sandeha.data.store import DATA_ROOT
from sandeha.physics.scenes import SPLITS, TRAIN_BUOYANCY, Scene, sample_scene
from sandeha.physics.solver import SmokeSolver, SmokeState

FORK_FRAMES = 5
FRAMES_BEFORE_FORK = 2  # Shown in a rendering, so that it shows the flow going into the spike.
CONTROLS = 3
PASS_FACTOR = 2.0  # A named scene passes at up to this many times the largest control.
DEVICE = torch.device("cuda")
# What a control's figure depends on. A sample with other values says nothing about this dataset.
CONTROL_PHYSICS = ("res", "dt", "steps_per_frame", "frames", "cg_tol", "cg_max_iter")


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


@dataclass
class RestartRuns:
    """Density at the reference grid, from one state, at the data's step size and at half of it."""

    shared_frames: list[int]  # The frames up to the restart, which both runs have in common.
    shared: list[torch.Tensor]
    frames: list[int]  # The frames after the restart.
    coarse: list[torch.Tensor]
    fine: list[torch.Tensor]
    solid: list[torch.Tensor]  # The solid in force on the way into each of ``frames``.


def restart_runs(scene: Scene, config: dict, fork: int) -> RestartRuns:
    """Run a scene alone up to ``fork``, then on for ``FORK_FRAMES`` at both step sizes."""
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
    shared_frames = [frame for frame in range(fork + 1) if frame >= fork - FRAMES_BEFORE_FORK]
    history = [shared.density[0, 0]] if 0 in shared_frames else []
    for frame in range(fork):
        for _ in range(substeps):
            shared = solver.step(shared, solid_at(frame), inflow, lift)
        if frame + 1 in shared_frames:
            history.append(shared.density[0, 0])

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
    return RestartRuns(
        shared_frames,
        history,
        list(range(fork + 1, fork + 1 + FORK_FRAMES)),
        coarse,
        fine,
        [solid_at(frame)[0, 0] for frame in range(fork, fork + FORK_FRAMES)],
    )


def step_size_difference(scene: Scene, config: dict, fork: int) -> list[float]:
    """RMS density difference over fluid cells, relative to RMS density, from a shared state."""
    runs = restart_runs(scene, config, fork)
    result = []
    for a, b, solid in zip(runs.coarse, runs.fine, runs.solid):
        fluid = ~solid
        result.append(float((a - b)[fluid].norm() / b[fluid].norm()))
    return result


def render_restart(scene: Scene, config: dict, fork: int, path: Path, title: str) -> list[int]:
    """Draw the frames around a restart at both step sizes, and return the frames shown.

    Top row, the data's step size. Middle row, half of it. Bottom row, the difference, on its own
    scale so that a small leak can be seen. Solid is grey. A frame before the restart has one run.
    """
    runs = restart_runs(scene, config, fork)
    shown = runs.shared_frames + runs.frames
    coarse = runs.shared + runs.coarse
    fine = runs.shared + runs.fine
    top = max(float(density.max()) for density in coarse + fine) or 1.0
    late = len(runs.shared)
    gaps = [(a - b).abs() for a, b in zip(runs.coarse, runs.fine)]
    gap_top = max(float(gap.max()) for gap in gaps) or 1.0
    solids = [torch.from_numpy(scene.obstacle)] * late + [solid.cpu() for solid in runs.solid]

    figure, axes = plt.subplots(3, len(shown), figsize=(1.7 * len(shown), 5.4), squeeze=False)
    for column, frame in enumerate(shown):
        rows = (coarse[column], fine[column], gaps[column - late] if column >= late else None)
        for row, values in enumerate(rows):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if values is None:
                axis.axis("off")
                continue
            limit = gap_top if row == 2 else top
            axis.imshow(values.cpu().numpy(), origin="lower", cmap="magma", vmin=0, vmax=limit)
            overlay = np.zeros((*values.shape, 4))
            overlay[solids[column].numpy()] = (0.6, 0.6, 0.6, 1.0)
            axis.imshow(overlay, origin="lower")
        axes[0, column].set_title(f"frame {frame}", fontsize=9)
    dt = config["dt"]
    axes[0, 0].set_ylabel(f"dt={dt}", fontsize=9)
    axes[1, 0].set_ylabel(f"dt={dt / 2}", fontsize=9)
    axes[2, late].set_ylabel(f"difference, top {gap_top:.3g}", fontsize=9)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return shown


def _largest(values: list[float]) -> float:
    """The largest value, or NaN if any is. Python's ``max`` can pass over a NaN."""
    return float(np.max(values))


def _figure(scene: Scene, config: dict, frame: int) -> dict:
    fork = fork_frame(frame, config["frames"])
    by_frame = step_size_difference(scene, config, fork)
    return {"fork_frame": fork, "density_difference_by_frame": by_frame, "figure": _largest(by_frame)}


def _side(scene: Scene) -> str:
    """A test_c scene's side, by the same test as the manifest's count of sides."""
    return "low" if scene.buoyancy < TRAIN_BUOYANCY[0] else "high"


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
        side = {"side": _side(scene)} if split == "test_c" else {}
        return {"scene": i, "largest_travel": travel[i], **side, **_figure(scene, config, frames[i])}

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


def render_named_scenes(result: dict, manifest: dict, directory: Path, relative_to: Path) -> None:
    """Rule 4: draw the frames around each named scene's spike, and add each path to ``result``."""
    config = manifest["config"]
    for split, entry in result["splits"].items():
        for named in entry["named"]:
            seed = manifest["splits"][split]["seeds"][named["scene"]]
            scene = sample_scene(np.random.default_rng(seed), config["res"], split)
            path = Path(directory) / f"travel_rule_{split}_{named['scene']:05d}.png"
            title = f"{split} scene {named['scene']}: travel {named['largest_travel']:.2f}, figure {named['figure']:.3g}"
            render_restart(scene, config, named["fork_frame"], path, title)
            named["rendering"] = str(path.relative_to(relative_to))


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
        measured = {}

        def control(i: int) -> dict:
            if i not in measured:
                side = {"side": _side(batch[i])} if split == "test_c" else {}
                measured[i] = {
                    "scene": i,
                    "late_frame": batch[i].late_frame,
                    "largest_travel": largest[i],
                    "largest_travel_frame": frames[i],
                    **side,
                    **_figure(batch[i], config, frames[i]),
                }
            return measured[i]

        def lowest(indices) -> list[dict]:
            return [control(i) for i in sorted(indices, key=lambda i: largest[i])[:CONTROLS]]

        controls = lowest(range(scenes_per_split))
        splits[split] = {
            "scene_travel": largest,
            "late_scenes": sum(scene.late_obstacle is not None for scene in batch),
            "all_converged": all(converged),
            "controls": controls,
            "largest_control_figure": _largest([control["figure"] for control in controls]),
        }
        if split == "test_c":
            # The rule's controls are the lowest overall, which may all be one side, while a named
            # scene is likely the other. Each side's own figures show whether that matters.
            sides = {"low": [], "high": []}
            for i, scene in enumerate(batch):
                sides[_side(scene)].append(i)
            splits[split]["sides"] = {
                side: {"scenes": indices, "controls": lowest(indices)} for side, indices in sides.items()
            }
    used = {key: config[key] for key in CONTROL_PHYSICS if key in config}
    return {
        "seed": seed,
        "scenes_per_split": scenes_per_split,
        "fork_frames": FORK_FRAMES,
        "config": used,
        "splits": splits,
    }


class ControlsNotReady(RuntimeError):
    """The rule's controls haven't been measured for this config, so generation can't start."""


def controls_gate_applies(root: Path, allow_dirty: bool) -> bool:
    """Everything but a scratch run, which ``--allow-dirty`` outside the data root declares."""
    return not allow_dirty or Path(root).resolve() == Path(DATA_ROOT).resolve()


def check_controls(config: dict, path: Path) -> None:
    """Raise unless ``path`` holds controls measured on committed code, for this config's physics."""
    path = Path(path)
    if not path.exists():
        raise ControlsNotReady(
            f"{path} is missing. Measure and log the controls before generating, with: "
            "scripts/long_job.sh travel-controls uv run python scripts/check_travel.py --control-sample"
        )
    result = json.loads(path.read_text())
    if result.get("git_dirty", True):
        raise ControlsNotReady(f"{path} was not measured on committed code, so it can't be traced")
    # A setting left out of a config takes the solver's default, so compare what the two give.
    used = result.get("config", {})
    settings = {"cg_tol": 1e-5, "cg_max_iter": 2000}
    different = [
        key
        for key in CONTROL_PHYSICS
        if used.get(key, settings.get(key)) != config.get(key, settings.get(key))
    ]
    if different:
        raise ControlsNotReady(f"{path} was measured with different {different}, so it doesn't judge this dataset")
    for split in SPLITS:
        figure = result["splits"].get(split, {}).get("largest_control_figure", math.nan)
        if not math.isfinite(figure):
            raise ControlsNotReady(f"{path} has no finite control figure for {split}")
