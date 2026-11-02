"""Measure the reference solver and render a check run.

Writes results/solver_check.json and results/figures/solver_check.png.
Usage: uv run python scripts/check_solver.py
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

from sandeha.physics.solver import DENSITY_DECAY, SmokeSolver, SmokeState

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda")
DT = 0.01
LIFT = 2.0
WARM_UP_STEPS = 100
TIMED_STEPS = 100
CONVERGENCE_TIMES = tuple(0.25 * i for i in range(1, 13))  # Up to t = 3.
LONG_RUN_STEPS = 300
PLAN_DIVERGENCE_LIMIT = 1e-3  # The plan's limit, which it sets at 64x64.
STRIP_TIMES = (0.25, 0.75, 1.25, 1.75, 2.25, 3.0)


def coords(res):
    centres = (torch.arange(res, device=DEVICE, dtype=torch.float32) + 0.5) / res
    y, x = torch.meshgrid(centres, centres, indexing="ij")
    return x, y


def disc(res, cx, cy, r):
    x, y = coords(res)
    return (((x - cx) ** 2 + (y - cy) ** 2) < r**2)[None, None]


def soft_inflow(res):
    x, y = coords(res)
    return 5.0 * torch.exp(-((x - 0.5) ** 2 + (y - 0.15) ** 2) / 0.08**2)[None, None]


def hard_inflow(res):
    return 5.0 * disc(res, 0.5, 0.15, 0.08).float()


def still(res, batch=1):
    return SmokeState(
        velocity=torch.zeros(batch, 2, res, res, device=DEVICE),
        density=torch.zeros(batch, 1, res, res, device=DEVICE),
    )


def measure_speed(res, batch):
    obstacle = disc(res, 0.5, 0.55, 0.1).expand(batch, -1, -1, -1)
    inflow = soft_inflow(res).expand(batch, -1, -1, -1)
    lift = torch.full((batch,), LIFT, device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)
    state = still(res, batch)
    for _ in range(WARM_UP_STEPS):
        state = solver.step(state, obstacle, inflow, lift)

    iterations = []
    residuals = []
    all_converged = True
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        state = solver.step(state, obstacle, inflow, lift)
        report = solver.last_solve
        iterations.append(report.iterations.float().mean())
        residuals.append(report.residual.max())
        all_converged = all_converged and bool(report.converged.all())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "res": res,
        "batch": batch,
        "steps_per_second": TIMED_STEPS / elapsed,
        "item_steps_per_second": batch * TIMED_STEPS / elapsed,
        "mean_cg_iterations": float(torch.stack(iterations).mean()),
        "max_divergence": float(torch.stack(residuals).max()),
        "all_converged": all_converged,
    }


def density_at_times(res, inflow, times):
    obstacle = disc(res, 0.5, 0.55, 0.1)
    lift = torch.tensor([LIFT], device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)
    state = still(res)
    wanted = {round(t / DT): t for t in times}
    fields = {}
    for step in range(1, max(wanted) + 1):
        state = solver.step(state, obstacle, inflow(res), lift)
        if step in wanted:
            fields[wanted[step]] = state.density.clone()
    return fields


def relative_error(a, b):
    return float((a - b).norm() / b.norm())


def measure_convergence():
    table = []
    for name, inflow in (("soft", soft_inflow), ("hard", hard_inflow)):
        runs = {res: density_at_times(res, inflow, CONVERGENCE_TIMES) for res in (64, 128, 256)}
        for t in CONVERGENCE_TIMES:
            fine = runs[256][t]
            table.append(
                {
                    "inflow": name,
                    "time": t,
                    "error_128_vs_256": relative_error(runs[128][t], F.avg_pool2d(fine, 2)),
                    "error_64_vs_256": relative_error(runs[64][t], F.avg_pool2d(fine, 4)),
                    "ratio_128_to_64": relative_error(runs[128][t], F.avg_pool2d(fine, 2))
                    / relative_error(runs[64][t], F.avg_pool2d(fine, 4)),
                }
            )
    return table


def divergence_size(solver, state, obstacle):
    div = solver.divergence(state, obstacle)
    fluid = div[obstacle == 0]
    return {"max": float(fluid.abs().max()), "rms": float(fluid.pow(2).mean().sqrt())}


def measure_long_run():
    """Divergence over a whole run, and of each velocity view of its final state."""
    res = 256
    obstacle = disc(res, 0.5, 0.55, 0.1)
    inflow = soft_inflow(res)
    lift = torch.tensor([LIFT], device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)
    state = still(res)
    residuals = []
    limits = []
    all_converged = True
    for _ in range(LONG_RUN_STEPS):
        state = solver.step(state, obstacle, inflow, lift)
        u, v = state.mac
        fastest = max(float(u.abs().max()), float(v.abs().max()))
        residuals.append(float(solver.last_solve.residual[0]))
        limits.append(solver.cg_tol * fastest * res)
        all_converged = all_converged and bool(solver.last_solve.converged[0])

    coarse = SmokeSolver(128, DEVICE, DT)
    random_solver = SmokeSolver(64, DEVICE, DT)
    open_box = torch.zeros(1, 1, 64, 64, dtype=torch.bool, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    random_velocity = 2 * torch.rand(1, 2, 64, 64, device=DEVICE, generator=generator) - 1
    projected = random_solver.project(
        SmokeState(random_velocity, torch.zeros(1, 1, 64, 64, device=DEVICE)), open_box
    )
    return {
        "res": res,
        "steps": LONG_RUN_STEPS,
        "cg_tol": solver.cg_tol,
        "max_divergence": max(residuals),
        "max_physical_limit": max(limits),
        "steps_above_plan_limit": sum(r > PLAN_DIVERGENCE_LIMIT for r in residuals),
        "plan_limit": PLAN_DIVERGENCE_LIMIT,
        "all_converged": all_converged,
        "max_divergence_steps_101_to_200": max(residuals[100:200]),
        "view_divergence": {
            "plume_256_faces": divergence_size(solver, state, obstacle),
            "plume_256_cell_centred": divergence_size(
                solver, SmokeState(state.velocity, state.density), obstacle
            ),
            "plume_128_downsampled_cell_centred": divergence_size(
                coarse,
                SmokeState(F.avg_pool2d(state.velocity, 2), F.avg_pool2d(state.density, 2)),
                disc(128, 0.5, 0.55, 0.1),
            ),
            "random_64_faces": divergence_size(random_solver, projected, open_box),
            "random_64_cell_centred": divergence_size(
                random_solver, SmokeState(projected.velocity, projected.density), open_box
            ),
        },
    }


def render_strip(path):
    res = 256
    fields = density_at_times(res, soft_inflow, STRIP_TIMES)
    solid = disc(res, 0.5, 0.55, 0.1)[0, 0].cpu().numpy()
    top = max(float(f.max()) for f in fields.values())

    figure, axes = plt.subplots(1, len(STRIP_TIMES), figsize=(3 * len(STRIP_TIMES), 3.3))
    for axis, t in zip(axes, STRIP_TIMES):
        axis.imshow(
            fields[t][0, 0].cpu().numpy(), origin="lower", cmap="magma", vmin=0, vmax=top
        )
        overlay = np.zeros((res, res, 4))
        overlay[solid] = (0.35, 0.8, 0.9, 1.0)
        axis.imshow(overlay, origin="lower")
        axis.set_title(f"t = {t}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Reference solver at 256x256: density, with the obstacle in blue. Up is up.")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
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
    result = {
        "git": git_commit(),
        "device": torch.cuda.get_device_name(DEVICE),
        "settings": {
            "dt": DT,
            "buoyancy": LIFT,
            "density_decay": DENSITY_DECAY,
            "warm_up_steps": WARM_UP_STEPS,
            "timed_steps": TIMED_STEPS,
        },
        "speed": [measure_speed(res, batch) for res in (256, 128) for batch in (1, 8, 16)],
        "convergence": measure_convergence(),
        "long_run": measure_long_run(),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20,
    }
    render_strip(ROOT / "results" / "figures" / "solver_check.png")

    output = ROOT / "results" / "solver_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
