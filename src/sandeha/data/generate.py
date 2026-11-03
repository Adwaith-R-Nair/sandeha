"""Generate the dataset: run scenes through the reference solver and store them.

Safe to stop and restart at any point. Each scene gets a seed of its own and goes to its own file
in one atomic step. Scenes are batched by fixed index ranges, and a batch with any scene missing
runs again whole, so a resumed run does the same arithmetic as an uninterrupted one.

Resuming needs the same config and commit. Finished data is never regenerated: replacing a locked
test set should take a deliberate step, so it means deleting the data root by hand.
"""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sandeha.data.dataset import fit_normalizer
from sandeha.data.store import (
    SPLIT_FILES,
    SplitStore,
    sha256_file,
    split_digest,
    sync_directory,
    write_atomic,
    write_json_atomic,
)
from sandeha.physics.scenes import MIN_THICKNESS, SPLITS, TRAIN_BUOYANCY, sample_scene
from sandeha.physics.solver import SmokeSolver, SmokeState

REPO = Path(__file__).resolve().parents[3]
FORMAT_VERSION = 1
_REQUIRED = ("seed", "res", "coarse_res", "dt", "steps_per_frame", "frames", "batch_size", "fine_frames", "scenes")


class GenerationError(RuntimeError):
    """Generation can't go on, or can't safely resume."""


def validate_config(config: dict) -> dict:
    missing = [key for key in _REQUIRED if key not in config]
    if missing:
        raise ValueError(f"config is missing {missing}")
    if config["res"] % config["coarse_res"] != 0:
        raise ValueError("coarse_res must divide res")
    if config["frames"] < 2:
        raise ValueError("a scene needs at least 2 frames")
    if config["batch_size"] < 1 or config["steps_per_frame"] < 1:
        raise ValueError("batch_size and steps_per_frame must be positive")
    if set(config["scenes"]) != set(SPLITS):
        raise ValueError(f"scenes must list exactly {SPLITS}")
    if any(not 0 <= frame < config["frames"] for frame in config["fine_frames"]):
        raise ValueError("fine_frames must be frames of the scene")
    return config


def load_config(path: Path) -> dict:
    return validate_config(yaml.safe_load(Path(path).read_text()))


def scene_seed(master: int, split: str, index: int) -> int:
    """A seed depending only on these three, so that resizing one split rerolls no other."""
    sequence = np.random.SeedSequence([master, SPLITS.index(split), index])
    return int(sequence.generate_state(1, np.uint64)[0])


def git_state() -> tuple[str, bool]:
    """The current commit, and whether the working tree differs from it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def travel_cells(mac: tuple[torch.Tensor, torch.Tensor], dt: float, res: int) -> torch.Tensor:
    """Each item's largest distance moved in one step, in cells, shape ``(B,)``.

    A wall's thickness is a distance in any direction, so both components count together. Taking
    each cell's fastest faces gives an upper bound, where averaging them to the centre could read
    low at a sharp spike, and a scene near the wall should be named rather than missed.
    """
    u, v = mac
    sideways = torch.maximum(u[..., :-1].abs(), u[..., 1:].abs())
    upwards = torch.maximum(v[..., :-1, :].abs(), v[..., 1:, :].abs())
    return (sideways.square() + upwards.square()).sqrt().amax(dim=(1, 2, 3)) * dt * res


def _simulate_batch(config: dict, batch: list, device: torch.device) -> dict:
    """Run scenes together, and return per-scene arrays of what is to be stored."""
    res, factor = config["res"], config["res"] // config["coarse_res"]
    frames, count = config["frames"], len(batch)

    def tensor(arrays) -> torch.Tensor:
        return torch.from_numpy(np.stack(arrays))[:, None].to(device)

    obstacle = tensor([scene.obstacle for scene in batch])
    late = tensor(
        [np.zeros((res, res), bool) if s.late_obstacle is None else s.late_obstacle for s in batch]
    )
    # Frames beyond the scene never arrive, so a scene with no late obstacle can use the same rule.
    late_frame = torch.tensor(
        [frames if s.late_frame is None else s.late_frame for s in batch], device=device
    )
    inflow = tensor([scene.inflow for scene in batch])
    lift = torch.tensor([scene.buoyancy for scene in batch], device=device, dtype=torch.float32)

    solver = SmokeSolver(
        res,
        device,
        config["dt"],
        cg_tol=config.get("cg_tol", 1e-5),
        cg_max_iter=config.get("cg_max_iter", 2000),
    )
    state = SmokeState(
        torch.zeros(count, 2, res, res, device=device), torch.zeros(count, 1, res, res, device=device)
    )

    fields = torch.empty(count, frames, 3, res // factor, res // factor, dtype=torch.float16, device=device)
    fine_states = torch.empty(count, len(config["fine_frames"]), 3, res, res, dtype=torch.float32)
    rounding = torch.zeros(count, 3, device=device, dtype=torch.float64)
    change = torch.zeros(count, 3, device=device, dtype=torch.float64)
    fluid_terms = torch.zeros(count, device=device, dtype=torch.float64)
    change_terms = torch.zeros(count, device=device, dtype=torch.float64)
    stats = {
        "unconverged_item_steps": torch.zeros(count, device=device, dtype=torch.long),
        "item_steps": torch.zeros(count, device=device, dtype=torch.long),
        "max_cg_iterations": torch.zeros(count, device=device, dtype=torch.long),
        "max_divergence": torch.zeros(count, device=device),
        "max_travel_cells_per_step": torch.zeros(count, device=device),
        "max_travel_frame": torch.zeros(count, device=device, dtype=torch.long),
        "max_density_in_solid": torch.zeros(count, device=device),
    }

    previous = None
    solid_into = obstacle  # The solid in force on the way into the current frame.
    for frame in range(frames):
        coarse_solid = F.max_pool2d(solid_into.float(), factor) > 0
        view = F.avg_pool2d(torch.cat([state.velocity, state.density], dim=1), factor) * ~coarse_solid
        stored = view.half()
        fields[:, frame] = stored

        # Finite as stored, since a value too large for float16 only turns infinite here.
        finite = stored.isfinite().all(dim=(1, 2, 3))
        if not finite.all():
            bad = [i for i, ok in enumerate(finite.tolist()) if not ok]
            raise GenerationError(f"state is not finite at frame {frame}, batch items {bad}")

        fluid = ~coarse_solid
        rounding += ((stored.float() - view).double() ** 2 * fluid).sum(dim=(2, 3))
        fluid_terms += fluid.sum(dim=(1, 2, 3))
        if previous is not None:
            change += ((view - previous).double() ** 2 * fluid).sum(dim=(2, 3))
            change_terms += fluid.sum(dim=(1, 2, 3))
        previous = view

        stats["max_density_in_solid"] = torch.maximum(
            stats["max_density_in_solid"], (state.density * solid_into).amax(dim=(1, 2, 3))
        )
        if frame in config["fine_frames"]:
            fine = torch.cat([state.velocity, state.density], dim=1).cpu()
            fine_states[:, config["fine_frames"].index(frame)] = fine
        if frame == frames - 1:
            break

        solid = obstacle | (late & (late_frame <= frame).view(-1, 1, 1, 1))
        for _ in range(config["steps_per_frame"]):
            state = solver.step(state, solid, inflow, lift)
            report = solver.last_solve
            stats["unconverged_item_steps"] += ~report.converged
            stats["item_steps"] += 1
            stats["max_cg_iterations"] = torch.maximum(stats["max_cg_iterations"], report.iterations)
            stats["max_divergence"] = torch.maximum(stats["max_divergence"], report.residual)
            travel = travel_cells(state.mac, config["dt"], res)
            furthest = travel > stats["max_travel_cells_per_step"]
            stats["max_travel_frame"][furthest] = frame
            stats["max_travel_cells_per_step"][furthest] = travel[furthest]
        solid_into = solid

    pooled_obstacle = F.max_pool2d(obstacle.float(), factor) > 0
    pooled_late = F.max_pool2d(late.float(), factor) > 0
    pooled_inflow = F.avg_pool2d(inflow, factor)
    return {
        "fields": fields.cpu().numpy(),
        "fine_states": fine_states.numpy(),
        "obstacle": pooled_obstacle[:, 0].cpu().numpy(),
        "late_obstacle": pooled_late[:, 0].cpu().numpy(),
        "inflow": pooled_inflow[:, 0].cpu().numpy(),
        "fine_late_obstacle": late[:, 0].cpu().numpy(),
        "rounding_rms": (rounding / fluid_terms[:, None]).sqrt().cpu().numpy(),
        "frame_change_rms": (change / change_terms[:, None].clamp(min=1)).sqrt().cpu().numpy(),
        **{name: value.cpu().numpy() for name, value in stats.items()},
    }


_STAT_NAMES = (
    "unconverged_item_steps",
    "item_steps",
    "max_cg_iterations",
    "max_divergence",
    "max_travel_cells_per_step",
    "max_travel_frame",
    "max_density_in_solid",
    "rounding_rms",
    "frame_change_rms",
)


def _write_scene(path: Path, seed: int, scene, result: dict, index: int) -> None:
    arrays = {
        "seed": np.array(seed, dtype=np.uint64),
        "buoyancy": np.array(scene.buoyancy, dtype=np.float64),
        "late_frame": np.array(-1 if scene.late_frame is None else scene.late_frame),
        "fine_obstacle": scene.obstacle,
        "fine_inflow": scene.inflow,
    }
    names = ("fields", "fine_states", "obstacle", "late_obstacle", "inflow", "fine_late_obstacle")
    for name in names + _STAT_NAMES:
        arrays[name] = result[name][index]
    write_atomic(path, lambda file: np.savez(file, **arrays))


def _record_timing(root: Path, entry: dict) -> None:
    with open(root / "timing.jsonl", "a") as file:
        file.write(json.dumps(entry) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _drop_torn_timing(root: Path, log: Callable[[str], None]) -> None:
    """Cut anything after the last full line, so that the next line starts on a line of its own."""
    path = root / "timing.jsonl"
    if not path.exists():
        return
    content = path.read_bytes()
    whole = content.rfind(b"\n") + 1
    if whole == len(content):
        return
    with open(path, "rb+") as file:
        file.truncate(whole)
        os.fsync(file.fileno())
    log(f"dropped {len(content) - whole} bytes of a torn last line from {path}")


def _generate_split(config, root, split, device, on_scene, log) -> None:
    directory = root / split / "scenes"
    directory.mkdir(parents=True, exist_ok=True)
    for torn in directory.glob("*.tmp"):
        torn.unlink()

    total = config["scenes"][split]
    for start in range(0, total, config["batch_size"]):
        indices = range(start, min(start + config["batch_size"], total))
        paths = {i: directory / f"{i:05d}.npz" for i in indices}
        if all(path.exists() for path in paths.values()):
            continue

        began = time.perf_counter()
        seeds = [scene_seed(config["seed"], split, i) for i in indices]
        batch = [sample_scene(np.random.default_rng(seed), config["res"], split) for seed in seeds]
        sampled = time.perf_counter()
        result = _simulate_batch(config, batch, device)
        torch.cuda.synchronize()
        solved = time.perf_counter()

        for position, i in enumerate(indices):
            if not paths[i].exists():
                _write_scene(paths[i], seeds[position], batch[position], result, position)
                if on_scene is not None:
                    on_scene(split, i)
        finished = time.perf_counter()
        _record_timing(
            root,
            {
                "split": split,
                "start": start,
                "scenes": len(indices),
                "sampling": sampled - began,
                "solving": solved - sampled,
                "writing": finished - solved,
            },
        )
        rate = len(indices) * (config["frames"] - 1) * config["steps_per_frame"] / (solved - sampled)
        log(f"{split}: scenes {start} to {indices[-1]} of {total}, {rate:.1f} item steps/s")


def _assemble_split(config: dict, root: Path, split: str, fine_frames: list[int]) -> dict:
    """Join a split's scene files into its final files. Safe to repeat."""
    directory = root / split
    paths = [directory / "scenes" / f"{i:05d}.npz" for i in range(config["scenes"][split])]
    coarse = config["coarse_res"]
    temporary = directory / "fields.npy.tmp"
    fields = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(paths), config["frames"], 3, coarse, coarse),
    )

    names = ("obstacle", "late_obstacle", "inflow", "fine_obstacle", "fine_late_obstacle", "fine_inflow")
    static = {name: [] for name in names + ("buoyancy", "late_frame", "seed")}
    states = []
    collected = {name: [] for name in _STAT_NAMES}
    for i, path in enumerate(paths):
        with np.load(path) as scene:
            fields[i] = scene["fields"]
            for name in static:
                static[name].append(scene[name])
            states.append(scene["fine_states"])
            for name in _STAT_NAMES:
                collected[name].append(scene[name])
    fields.flush()
    del fields
    with open(temporary, "rb+") as file:
        os.fsync(file.fileno())
    os.replace(temporary, directory / "fields.npy")
    sync_directory(directory)

    arrays = {name: np.stack(values) for name, values in static.items()}
    write_atomic(directory / "statics.npz", lambda file: np.savez(file, **arrays))
    write_atomic(
        directory / "fine.npz",
        lambda file: np.savez(file, frames=np.array(fine_frames, dtype=np.int64), states=np.stack(states)),
    )

    travel = np.array(collected["max_travel_cells_per_step"], dtype=np.float64)
    thinnest_wall = MIN_THICKNESS * config["res"]
    late_frames = [int(frame) for frame in arrays["late_frame"] if frame >= 0]
    entry = {
        "n_scenes": len(paths),
        "n_frames": config["frames"],
        "seeds": [int(seed) for seed in arrays["seed"]],
        "late_scenes": len(late_frames),
        "late_share": len(late_frames) / len(paths),
        "late_frames": late_frames,
        "stats": {
            "unconverged_item_steps": int(np.sum(collected["unconverged_item_steps"])),
            "item_steps": int(np.sum(collected["item_steps"])),
            "max_cg_iterations": int(np.max(collected["max_cg_iterations"])),
            "max_divergence": float(np.max(collected["max_divergence"])),
            "max_travel_cells_per_step": float(np.max(travel)),
            "travel_by_scene": travel.tolist(),
            "travel_frame_by_scene": [int(frame) for frame in collected["max_travel_frame"]],
            "thinnest_wall_cells": thinnest_wall,
            "scenes_over_thinnest_wall": np.flatnonzero(travel > thinnest_wall).tolist(),
            "max_density_in_solid": float(np.max(collected["max_density_in_solid"])),
            # Each scene's figure, averaged as squares, so that the result is an overall RMS.
            "rounding_rms": np.sqrt(np.mean(np.square(collected["rounding_rms"]), axis=0)).tolist(),
            "frame_change_rms": np.sqrt(np.mean(np.square(collected["frame_change_rms"]), axis=0)).tolist(),
        },
    }
    if split == "test_c":
        low = int(np.sum(arrays["buoyancy"] < TRAIN_BUOYANCY[0]))
        entry["sides"] = {"low": low, "high": len(paths) - low}
    return entry


def _read_timing_line(line: str) -> dict | None:
    """One batch's entry, or None if the line can't be one."""
    try:
        entry = json.loads(line)
        return {
            "split": entry["split"],
            "scenes": int(entry["scenes"]),
            **{key: float(entry[key]) for key in ("sampling", "solving", "writing")},
        }
    except (ValueError, KeyError, TypeError):
        return None


def _summarise_timing(root: Path, extra: dict) -> dict:
    """Add up every batch recorded, including those run before a restart.

    Timing is a record and not part of the data, so a line that can't be read costs that batch's
    figures and no more. The number of such lines is reported.
    """
    seconds = {"sampling": 0.0, "solving": 0.0, "writing": 0.0, **extra}
    by_split = {}
    unreadable = 0
    path = root / "timing.jsonl"
    # Not all bytes of a damaged file are text, and those lines fail to parse anyway.
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    for line in lines:
        entry = _read_timing_line(line)
        if entry is None:
            unreadable += 1
            continue
        split = by_split.setdefault(
            entry["split"], {"scenes": 0, "sampling": 0.0, "solving": 0.0, "writing": 0.0}
        )
        split["scenes"] += entry["scenes"]
        for key in ("sampling", "solving", "writing"):
            seconds[key] += entry[key]
            split[key] += entry[key]
    return {
        "seconds": seconds,
        "total_seconds": sum(seconds.values()),
        "by_split": by_split,
        "unreadable_lines": unreadable,
    }


def _remove_scene_files(root: Path) -> None:
    """Remove what the finished files replace. Also run on a rerun, in case a crash came first."""
    for split in SPLITS:
        directory = root / split / "scenes"
        if directory.exists():
            shutil.rmtree(directory)


def generate(
    config: dict,
    root: Path,
    *,
    allow_dirty: bool = False,
    on_scene: Callable[[str, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Generate or resume the dataset under ``root``, and return its manifest.

    ``on_scene(split, index)`` is called after each scene is safely on disk. Non-finite data stops
    the run, with no scene stored from that batch. An unconverged solve doesn't stop it, and is
    counted in the manifest for the spot-check to judge.
    """
    validate_config(config)
    root = Path(root)
    if not torch.cuda.is_available():
        raise GenerationError("no GPU is available, and the dataset is generated on the GPU")
    device = torch.device("cuda")

    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise GenerationError("the working tree has uncommitted changes, so the data couldn't be traced to a commit")

    marker = root / "generation.json"
    if marker.exists():
        earlier = json.loads(marker.read_text())
        if earlier["config"] != json.loads(json.dumps(config)) or earlier["git_commit"] != commit:
            raise GenerationError(
                f"{root} holds a run with a different config or commit, so it can't be resumed or reused"
            )
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(marker, {"config": config, "git_commit": commit})

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        log(f"{manifest_path} exists, so this dataset is finished and its files were not changed")
        _remove_scene_files(root)
        return json.loads(manifest_path.read_text())

    _drop_torn_timing(root, log)
    torch.cuda.reset_peak_memory_stats(device)
    for split in SPLITS:
        _generate_split(config, root, split, device, on_scene, log)
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20

    began = time.perf_counter()
    splits = {split: _assemble_split(config, root, split, config["fine_frames"]) for split in SPLITS}
    assembled = time.perf_counter()
    for split, entry in splits.items():
        entry["files"] = {name: sha256_file(root / split / name) for name in SPLIT_FILES}
        entry["sha256"] = split_digest(entry["files"])
        entry["bytes"] = sum((root / split / name).stat().st_size for name in SPLIT_FILES)
    hashed = time.perf_counter()
    # The files were only just checksummed, so read train directly, not through open_split.
    normalizer = fit_normalizer(SplitStore(root / "train", "train"))
    write_json_atomic(root / "normalizer.json", normalizer.to_dict())
    fitted = time.perf_counter()

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_mib": peak_memory,
        "splits": splits,
        "normalizer_sha256": sha256_file(root / "normalizer.json"),
        "timing": _summarise_timing(
            root,
            {"assembling": assembled - began, "hashing": hashed - assembled, "normalizer": fitted - hashed},
        ),
    }
    write_json_atomic(manifest_path, manifest)
    _remove_scene_files(root)
    log(f"finished in {manifest['timing']['total_seconds']:.1f} s of work: {manifest_path}")
    return manifest
