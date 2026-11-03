"""Estimate the time to generate the planned dataset, from a small sample of it.

Runs the real generator at the planned grid, frame count and config, but with few scenes and a
different master seed, so that none of the locked scenes are drawn. Everything goes to a temporary
directory, which is deleted afterwards. Writes results/generation_estimate.json.

The estimate adds up the planned scene counts at each split's measured cost per scene, then the
steps that scale with size: joining files, checksums and the normaliser.

Usage: uv run python scripts/estimate_generation.py
"""

import copy
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from sandeha.data.generate import _simulate_batch, generate, git_state, load_config, scene_seed
from sandeha.physics.scenes import SPLITS, TRAIN_BUOYANCY, sample_scene

ROOT = Path(__file__).resolve().parents[1]
ESTIMATE_SEED = 900001  # Not the dataset's seed.
BATCHES = (8, 16)
SCENES_PER_BATCH_TEST = 16
SCENES_PER_SPLIT = 16
TIME_BUDGET_SECONDS = 1.5 * 3600
GB = 10**9


def gpu_reading() -> dict:
    query = "temperature.gpu,clocks.sm,clocks_throttle_reasons.active,power.draw"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return dict(zip(query.split(","), output.split(", ")))


def run_sample(config: dict, scenes: dict[str, int], batch: int, parent: Path) -> dict:
    sample = copy.deepcopy(config)
    sample["seed"] = ESTIMATE_SEED
    sample["batch_size"] = batch
    sample["scenes"] = scenes
    root = Path(tempfile.mkdtemp(dir=parent))
    scene_bytes = []

    def weigh(split: str, index: int) -> None:
        scene_bytes.append((root / split / "scenes" / f"{index:05d}.npz").stat().st_size)

    began = time.perf_counter()
    manifest = generate(sample, root, on_scene=weigh, log=lambda line: print(line, flush=True))
    manifest["wall_seconds"] = time.perf_counter() - began
    manifest["scene_file_bytes"] = sum(scene_bytes)
    shutil.rmtree(root)
    return manifest


def peak_disk(sample: dict, planned_scenes: int) -> float:
    """Bytes at the end of generation, when a split's scene files and final files both exist."""
    finished = sum(entry["bytes"] for entry in sample["splits"].values())
    per_scene = (sample["scene_file_bytes"] + finished) / sum(sample["config"]["scenes"].values())
    return per_scene * planned_scenes


def travel_by_side(sample: dict, res: int) -> dict:
    """Each test_c sample scene's travel, by its side, from the seeds and the manifest's figures."""
    entry = sample["splits"]["test_c"]
    sides = {"low": [], "high": []}
    for seed, travel in zip(entry["seeds"], entry["stats"]["travel_by_scene"]):
        scene = sample_scene(np.random.default_rng(seed), res, "test_c")
        sides["low" if scene.buoyancy < TRAIN_BUOYANCY[0] else "high"].append(travel)
    return sides


def compare_batch(config: dict, batch: int) -> dict:
    """Sample and solve the same train scenes at one batch size. Nothing is stored."""
    sample = {**config, "batch_size": batch}
    seeds = [scene_seed(ESTIMATE_SEED, "train", i) for i in range(SCENES_PER_BATCH_TEST)]
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for start in range(0, len(seeds), batch):
        group = [
            sample_scene(np.random.default_rng(seed), config["res"], "train")
            for seed in seeds[start : start + batch]
        ]
        _simulate_batch(sample, group, torch.device("cuda"))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    return {
        "scenes": len(seeds),
        "seconds_per_scene": seconds / len(seeds),
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def scene_cost(manifest: dict, split: str) -> dict:
    timing = manifest["timing"]["by_split"][split]
    per_scene = (timing["sampling"] + timing["solving"] + timing["writing"]) / timing["scenes"]
    steps = manifest["config"]["steps_per_frame"] * (manifest["config"]["frames"] - 1)
    return {
        "scenes_measured": timing["scenes"],
        "seconds_per_scene": per_scene,
        "sampling_seconds_per_scene": timing["sampling"] / timing["scenes"],
        "solving_seconds_per_scene": timing["solving"] / timing["scenes"],
        "writing_seconds_per_scene": timing["writing"] / timing["scenes"],
        "item_steps_per_second": steps * timing["scenes"] / timing["solving"],
    }


def main() -> None:
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("commit first: this result has to trace to a commit")
    config = load_config(ROOT / "configs" / "data.yaml")
    planned = config["scenes"]
    parent = ROOT / "runs"  # On the same disk as data/, unlike a RAM-backed temporary directory.
    parent.mkdir(exist_ok=True)
    gpu_before = gpu_reading()
    began = time.perf_counter()

    comparison = {str(batch): compare_batch(config, batch) for batch in BATCHES}
    best = min(BATCHES, key=lambda batch: comparison[str(batch)]["seconds_per_scene"])
    if best != config["batch_size"]:
        raise SystemExit(f"configs/data.yaml has batch_size {config['batch_size']}, but {best} is faster: {comparison}")

    # Every split, at the configured batch size.
    sample = run_sample(config, {split: SCENES_PER_SPLIT for split in SPLITS}, config["batch_size"], parent)
    gpu_after = gpu_reading()

    costs = {split: scene_cost(sample, split) for split in SPLITS}
    simulate = {split: costs[split]["seconds_per_scene"] * planned[split] for split in SPLITS}

    sample_bytes = sum(entry["bytes"] for entry in sample["splits"].values())
    sample_scenes = sum(sample["config"]["scenes"].values())
    planned_scenes = sum(planned.values())
    planned_bytes = sample_bytes / sample_scenes * planned_scenes
    seconds = sample["timing"]["seconds"]
    scale_by_size = planned_scenes / sample_scenes
    assembling = seconds["assembling"] * scale_by_size
    hashing = seconds["hashing"] * scale_by_size
    normalizer = seconds["normalizer"] * planned["train"] / sample["config"]["scenes"]["train"]
    total = sum(simulate.values()) + assembling + hashing + normalizer

    rates = [costs[split]["seconds_per_scene"] for split in SPLITS]
    result = {
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config,
        "estimate_seed": ESTIMATE_SEED,
        "gpu": sample["gpu"],
        "torch_version": sample["torch_version"],
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "measurement_wall_seconds": time.perf_counter() - began,
        "batch_comparison_train": comparison,
        "chosen_batch_size": config["batch_size"],
        "peak_gpu_mib": sample["peak_gpu_mib"],
        "per_split": costs,
        "sample_seconds": seconds,
        "sample_wall_seconds": sample["wall_seconds"],
        "sample_bytes": sample_bytes,
        "sample_solver_health": {split: sample["splits"][split]["stats"] for split in SPLITS},
        "sample_scenes_over_thinnest_wall": sum(
            len(sample["splits"][split]["stats"]["scenes_over_thinnest_wall"]) for split in SPLITS
        ),
        "sample_largest_travel": max(sample["splits"][split]["stats"]["max_travel_cells_per_step"] for split in SPLITS),
        "sample_test_c_travel_by_side": travel_by_side(sample, config["res"]),
        "planned_scenes": planned,
        "planned_bytes": planned_bytes,
        "planned_gb": planned_bytes / GB,
        "planned_peak_disk_gb": peak_disk(sample, planned_scenes) / GB,
        "disk_free_gb": shutil.disk_usage(parent).free / GB,
        "estimate_seconds": {
            "simulate_by_split": simulate,
            "assembling": assembling,
            "hashing": hashing,
            "normalizer": normalizer,
            "total": total,
        },
        "estimate_hours": total / 3600,
        "if_every_scene_cost_the_slowest_split_hours": (
            max(rates) * planned_scenes + assembling + hashing + normalizer
        )
        / 3600,
        "if_every_scene_cost_the_fastest_split_hours": (
            min(rates) * planned_scenes + assembling + hashing + normalizer
        )
        / 3600,
        "budget_hours": TIME_BUDGET_SECONDS / 3600,
        "within_budget": total <= TIME_BUDGET_SECONDS,
        "not_included": [
            "the spot-check: scripts/spot_check.py and scripts/check_travel.py --data-root",
            "the controls before generation: scripts/check_travel.py --control-sample",
            "sustained thermal throttling beyond the length of this sample",
            "disk cache effects: this sample's files fit in memory, and the planned ones may not",
        ],
    }
    path = ROOT / "results" / "generation_estimate.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["estimate_seconds"], indent=1))
    print(f"estimate: {result['estimate_hours']:.2f} h, {result['planned_gb']:.2f} GB; wrote {path}")


if __name__ == "__main__":
    main()
