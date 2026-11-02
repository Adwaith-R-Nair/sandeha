"""Measure the scene sampler, and run a few scenes through the solver for one full length.

Answers three questions before data is generated: do the shape families cover similar areas,
do the fastest flows stay clear of jumping the thinnest wall, and does smoke reach the obstacles.
Writes results/scene_check.json, results/figures/scene_check.png and scene_check_flow.png.
Usage: uv run python scripts/check_scenes.py
"""

import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sandeha.physics import scenes
from sandeha.physics.scenes import (
    MIN_THICKNESS,
    SPLITS,
    TEST_C_BUOYANCY,
    TEST_C_INFLOW,
    TRAIN_BUOYANCY,
    TRAIN_INFLOW,
    sample_scene,
)
from sandeha.physics.solver import SmokeSolver, SmokeState

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda")
RES = 256
DT = 0.01  # The value the solver checks used. Task 4 fixes the one used for generation.
STEPS_PER_FRAME = 4
FRAMES = 121
BATCH = 8
SEED = 20261103
STATS_SCENES = 200
AREA_DRAWS = 300
TOUCH_DENSITY = 0.01
FLOW_FRAME = 60
GALLERY_COLUMNS = 6


def with_physics(scene, buoyancy, strength):
    """The same geometry at chosen physics, to test the corners of the ranges."""
    inflow = scene.inflow / scene.inflow.max() * strength
    return scenes.Scene(scene.obstacle, inflow, buoyancy, scene.late_obstacle, scene.late_frame)


def spread(values):
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def solid_ever(scene):
    return scene.obstacle if scene.late_obstacle is None else scene.obstacle | scene.late_obstacle


def measure_family_areas():
    x, y = scenes._grid(RES)
    x, y = x - 0.5, y - 0.5
    rng = np.random.default_rng(SEED)
    return {
        name: spread([builder(rng, x, y).mean() for _ in range(AREA_DRAWS)])
        for name, builder in scenes._BUILDERS.items()
    }


def measure_splits():
    result = {}
    for index, split in enumerate(SPLITS):
        rng = np.random.default_rng([SEED, index])
        start = time.perf_counter()
        batch = [sample_scene(rng, RES, split) for _ in range(STATS_SCENES)]
        elapsed = time.perf_counter() - start
        result[split] = {
            "scenes": STATS_SCENES,
            "seconds_per_scene": elapsed / STATS_SCENES,
            "solid_fraction": spread([solid_ever(s).mean() for s in batch]),
            "late_share": float(np.mean([s.late_obstacle is not None for s in batch])),
            "buoyancy": spread([s.buoyancy for s in batch]),
            "inflow_strength": spread([float(s.inflow.max()) for s in batch]),
        }
    return result


def run_group(batch):
    """Run a batch for a full scene length, and report speed, solver health and plume reach."""
    tensor = lambda arrays: torch.from_numpy(np.stack(arrays))[:, None].to(DEVICE)
    empty = np.zeros((RES, RES), dtype=bool)
    obstacle = tensor([s.obstacle for s in batch])
    late = tensor([empty if s.late_obstacle is None else s.late_obstacle for s in batch])
    late_step = torch.tensor(
        [10**9 if s.late_frame is None else s.late_frame * STEPS_PER_FRAME for s in batch],
        device=DEVICE,
    ).view(-1, 1, 1, 1)
    inflow = tensor([s.inflow.astype(np.float32) for s in batch])
    lift = torch.tensor([s.buoyancy for s in batch], dtype=torch.float32, device=DEVICE)
    state = SmokeState(
        velocity=torch.zeros(len(batch), 2, RES, RES, device=DEVICE),
        density=torch.zeros(len(batch), 1, RES, RES, device=DEVICE),
    )
    around = F.max_pool2d(obstacle.float(), 3, stride=1, padding=1).bool() & ~obstacle

    solver = SmokeSolver(RES, DEVICE, DT)
    steps = (FRAMES - 1) * STEPS_PER_FRAME
    travel = np.zeros(steps)
    iterations = []
    divergence = []
    unconverged = 0
    first_touch = torch.full((len(batch),), -1, device=DEVICE)
    density_in_solid = 0.0
    flow = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        solid = obstacle | (late & (step >= late_step))
        state = solver.step(state, solid, inflow, lift)
        report = solver.last_solve
        iterations.append(int(report.iterations.max()))
        divergence.append(float(report.residual.max()))
        unconverged += int((~report.converged).sum())
        u, v = state.mac
        travel[step] = float(torch.maximum(u.abs().max(), v.abs().max())) * DT * RES
        density_in_solid = max(density_in_solid, float((state.density * solid).abs().max()))
        frame = step // STEPS_PER_FRAME + 1
        touched = ((state.density > TOUCH_DENSITY) & around).any(dim=(1, 2, 3))
        first_touch = torch.where((first_touch < 0) & touched, frame, first_touch)
        if frame == FLOW_FRAME and flow is None:
            flow = state.density[:, 0].cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    touch = first_touch.cpu().numpy()
    return (
        {
            "scenes": len(batch),
            "steps": steps,
            "seconds": elapsed,
            "item_steps_per_second": len(batch) * steps / elapsed,
            "largest_travel_cells_per_step": float(travel.max()),
            "mean_of_travel_cells_per_step": float(travel.mean()),
            "thinnest_wall_cells": MIN_THICKNESS * RES,
            "steps_with_travel_above_thinnest_wall": int((travel > MIN_THICKNESS * RES).sum()),
            "largest_cg_iterations": max(iterations),
            "mean_of_largest_cg_iterations": float(np.mean(iterations)),
            "largest_divergence": max(divergence),
            "unconverged_item_steps": unconverged,
            "largest_density_in_solid": density_in_solid,
            "all_finite": bool(state.density.isfinite().all() and state.velocity.isfinite().all()),
            "largest_density": float(state.density.max()),
            "first_frame_smoke_touches_obstacle": [int(t) for t in touch],
            "scenes_where_smoke_never_touches": int((touch < 0).sum()),
        },
        flow,
    )


def group_scenes():
    def draw(split, index):
        rng = np.random.default_rng([SEED, 100 + index])
        return [sample_scene(rng, RES, split) for _ in range(BATCH)]

    low = (TEST_C_BUOYANCY[0][0], TEST_C_INFLOW[0][0])
    high = (TEST_C_BUOYANCY[1][1], TEST_C_INFLOW[1][1])
    return {
        "train": draw("train", 0),
        "test_b": draw("test_b", 1),
        "train_top_corner": [
            with_physics(s, TRAIN_BUOYANCY[1], TRAIN_INFLOW[1]) for s in draw("train", 2)
        ],
        "test_c_low_corner": [with_physics(s, *low) for s in draw("test_c", 3)],
        "test_c_high_corner": [with_physics(s, *high) for s in draw("test_c", 4)],
    }


def render_gallery(path):
    figure, axes = plt.subplots(
        len(SPLITS), GALLERY_COLUMNS, figsize=(2 * GALLERY_COLUMNS, 2.1 * len(SPLITS))
    )
    for row, split in enumerate(SPLITS):
        rng = np.random.default_rng([SEED, 200 + row])
        for column in range(GALLERY_COLUMNS):
            scene = sample_scene(rng, RES, split)
            share = (scene.inflow / scene.inflow.max())[..., None]
            image = 1 - share * (1 - np.array([1.0, 0.4, 0.05]))
            image[scene.obstacle] = (0.15, 0.3, 0.45)
            if scene.late_obstacle is not None:
                image[scene.late_obstacle & ~scene.obstacle] = (0.55, 0.75, 0.9)
            axis = axes[row, column]
            axis.imshow(image, origin="lower")
            axis.set_title(f"b={scene.buoyancy:.2f} s={scene.inflow.max():.1f}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(split)
    figure.suptitle("Scenes by split. Dark: obstacle. Pale: late obstacle. Orange: inflow.")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)


def render_flows(path, groups, flows):
    columns = 4
    figure, axes = plt.subplots(len(groups), columns, figsize=(2.2 * columns, 2.2 * len(groups)))
    for row, (name, batch) in enumerate(groups.items()):
        top = max(float(flows[name][:columns].max()), 1e-6)
        for column in range(columns):
            axis = axes[row, column]
            axis.imshow(flows[name][column], origin="lower", cmap="magma", vmin=0, vmax=top)
            overlay = np.zeros((RES, RES, 4))
            overlay[batch[column].obstacle] = (0.35, 0.8, 0.9, 1.0)
            axis.imshow(overlay, origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=8)
    figure.suptitle(f"Density at frame {FLOW_FRAME}, each row on its own scale. Up is up.")
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def git_commit():
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"head": head, "working_tree_changed": bool(dirty)}


def main():
    groups = group_scenes()
    runs = {}
    flows = {}
    for name, batch in groups.items():
        runs[name], flows[name] = run_group(batch)
        print(name, json.dumps(runs[name]), flush=True)

    result = {
        "git": git_commit(),
        "device": torch.cuda.get_device_name(DEVICE),
        "settings": {
            "res": RES,
            "dt": DT,
            "steps_per_frame": STEPS_PER_FRAME,
            "frames": FRAMES,
            "batch": BATCH,
            "seed": SEED,
            "touch_density": TOUCH_DENSITY,
            "min_thickness": MIN_THICKNESS,
        },
        "family_area_fraction": measure_family_areas(),
        "splits": measure_splits(),
        "runs": runs,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20,
    }
    figures = ROOT / "results" / "figures"
    render_gallery(figures / "scene_check.png")
    render_flows(figures / "scene_check_flow.png", groups, flows)

    output = ROOT / "results" / "scene_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("family_area_fraction", "splits")}, indent=2))


if __name__ == "__main__":
    main()
