"""Measure the demo's AI-mode frame rate, and choose how many members it shows.

    uv run python scripts/bench_demo.py

Runs the demo's own simulation in AI mode for 60 s per rung, with 5 members first. Below 15 FPS it goes
down the fallback ladder, one member fewer at a time, and stops at the first size to reach the target.
Writes results/demo_bench.json, whose chosen_members the demo then uses and shows on screen. Fewer than
2 members is never tried, as one has no doubt. Run it on the trained ensemble, with the GPU otherwise idle.

--random-weights measures untrained models, which isn't a result, so it needs its own --out.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

from sandeha.data.generate import git_state
from sandeha.data.store import write_json_atomic
from sandeha.demo.server import BENCH_PATH, OOM_MESSAGE, StartupError, add_arguments, build_simulation, check_device

TARGET_FPS = 15.0
LADDER = (5, 4, 3, 2)
WARM_UP_SECONDS = 2.0


def measure(args: argparse.Namespace, members: int) -> dict:
    """One rung: frames made per second in AI mode, after a warm-up, with each one encoded to send."""
    args.members = members
    sim = build_simulation(args, need_selection=False)
    sim.preflight()
    sim.submit({"type": "mode", "mode": "ai"})
    warm_up_ends = time.perf_counter() + WARM_UP_SECONDS
    while time.perf_counter() < warm_up_ends:
        sim.advance()
    first_frame, first_solver = sim._frames_made, sim._solver_frames
    began = time.perf_counter()
    while time.perf_counter() - began < args.seconds:
        sim.advance()
    elapsed = time.perf_counter() - began
    frames = sim._frames_made - first_frame
    rung = {
        "members": members,
        "seconds": elapsed,
        "frames": frames,
        "fps": frames / elapsed,
        "solver_frames": sim._solver_frames - first_solver,  # Rescues from a blow-up. They slow a run.
        "checkpoint_sha256": sim.checkpoint_sha256,
    }
    del sim
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"{members} members: {rung['fps']:.1f} FPS, {frames} frames in {elapsed:.1f} s", flush=True)
    return rung


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, help="Where to write. Default, and only for the trained ensemble: " + str(BENCH_PATH))
    args = parser.parse_args()
    if args.members is not None:
        parser.error("--members is set by the ladder here")
    if args.random_weights and args.out is None:
        parser.error("--random-weights isn't a result: name somewhere other than results/ with --out")

    rungs = []
    try:
        check_device(args.device)
        for members in LADDER:
            rungs.append(measure(args, members))
            if rungs[-1]["fps"] >= TARGET_FPS:
                break
    except torch.cuda.OutOfMemoryError:
        print(OOM_MESSAGE, file=sys.stderr)
        return 1
    except StartupError as error:
        print(error, file=sys.stderr)
        return 1

    commit, dirty = git_state()
    best = rungs[-1]
    report = {
        "target_fps": TARGET_FPS,
        "target_met": best["fps"] >= TARGET_FPS,
        "chosen_members": best["members"],
        "rungs": rungs,
        "weights": "random" if args.random_weights else "trained",
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else args.device,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    out = args.out or BENCH_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, report)
    print(f"written to {out}: {best['members']} members, {best['fps']:.1f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
