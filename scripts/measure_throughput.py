"""Measure training speed and memory for a model at each batch size, on the real training split.

Run it when nothing else is using the GPU, since the figures are meant to set training's length:

    uv run python scripts/measure_throughput.py

It also measures the divergence floor on the training targets, which docs/LLD.md asks for before
lambda_div is judged. Writes results/throughput.json.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

from sandeha.data.dataset import Normalizer, StepDataset
from sandeha.data.store import open_split, write_json_atomic
from sandeha.models.factory import build_model
from sandeha.train.loop import _sync, _time_forward, make_loader, training_step

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [REPO / "configs" / "train_unet.yaml", REPO / "configs" / "train_fno.yaml"]
FLOOR_BATCHES = 20


def measure(cfg: dict, batch_size: int, steps: int, device: str) -> dict:
    amp = bool(cfg["mixed_precision"]) and device == "cuda"
    root = Path(cfg["data_root"])
    normalizer = Normalizer.load(root / "normalizer.json")
    dataset = StepDataset(open_split(root, "train"), normalizer, unroll=cfg["pushforward_max"] + 1)
    loader = iter(make_loader(dataset, batch_size, cfg["seed"], 0, cfg["num_workers"]))

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def one_step(pushed: int) -> dict:
        optimizer.zero_grad(set_to_none=True)
        out = training_step(model, next(loader), normalizer, pushed, cfg["lambda_div"], device, amp)
        scaler.scale(out["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        return out

    one_step(cfg["pushforward_max"])  # The deepest step sets the peak, and warms everything up.
    _sync(device)
    began = time.perf_counter()
    for _ in range(steps):
        one_step(0)
    _sync(device)
    seconds = (time.perf_counter() - began) / steps
    batch = next(loader)
    floor = sum(
        training_step(model, next(loader), normalizer, 0, 0.0, device, amp)["div_target"].item()
        for _ in range(FLOOR_BATCHES)
    ) / FLOOR_BATCHES
    result = {
        "arch": cfg["model"]["arch"],
        "batch_size": batch_size,
        "device": device,
        "steps": steps,
        "steps_per_second": 1 / seconds,
        "items_per_second": batch_size / seconds,
        "seconds_per_forward": _time_forward(model, batch, device, amp),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else None,
        "lambda_div": cfg["lambda_div"],
        "div_target_mean_squared": floor,
    }
    del loader, model, optimizer
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", help="default: both planned models")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "throughput.json")
    args = parser.parse_args()

    other_memory_gb = None
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        other_memory_gb = (total - free) / 1024**3
    runs = []
    for path in args.config or DEFAULT_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        for batch_size in args.batch_sizes:
            try:
                runs.append(measure(cfg, batch_size, args.steps, args.device))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                runs.append({"arch": cfg["model"]["arch"], "batch_size": batch_size, "out_of_memory": True})
            print(json.dumps(runs[-1]))
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    write_json_atomic(args.out, {"git_commit": commit, "gpu_memory_in_use_at_start_gb": other_memory_gb, "runs": runs})


if __name__ == "__main__":
    main()
