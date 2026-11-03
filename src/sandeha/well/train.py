"""Training one U-Net on the public dataset, resumable as ``sandeha.train.loop.train`` is.

The loss is the MSE of the normalised change, with a pushforward as in the main track. There's no
divergence penalty, because this flow is compressible.
"""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn

from sandeha.models.factory import build_model
from sandeha.train.loop import (
    TIMING_WARMUP_STEPS,
    _cosine_lr,
    _rng_state,
    _set_rng_state,
    _sync,
    _time_forward,
    amp_context,
    budget_steps,
    load_checkpoint,
    make_loader,
    save_checkpoint,
)
from sandeha.well.data import WellSplit, WellNormalizer, WellStepDataset, check_counts
from sandeha.well.doubt import single_rollout, well_step


def training_step(
    model: nn.Module, batch: dict, normalizer: WellNormalizer, pushed: int, device: str = "cpu", amp: bool = False
) -> Tensor:
    """The loss after ``pushed`` steps on the model's own output, with no gradient through them."""
    x = batch["input"].to(device, non_blocking=True)
    if pushed:
        with torch.no_grad(), amp_context(device, amp):
            for _ in range(pushed):
                x = normalizer.encode(well_step(model, x, normalizer))
    with amp_context(device, amp):
        pred = model(x)
    with torch.autocast("cuda", enabled=False) if amp else nullcontext():
        truth = normalizer.decode(batch["target_seq"][:, pushed].to(device, non_blocking=True))
        target = normalizer.encode_delta(truth - normalizer.decode(x))
        return (pred.float() - target).square().mean()


def train_member(cfg: dict, run_dir: Path, resume: bool, *, on_step=None) -> Path:
    """Train to ``cfg["total_steps"]``, or as many as ``cfg["max_minutes"]`` allows.

    Returns the checkpoint with the lowest validation rollout error. With ``resume``, a run
    continues from ``last.pt`` if there is one. Without it, an existing run is refused.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists() and not resume:
        raise FileExistsError(f"{last_path} exists: pass resume, or use another directory")

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["mixed_precision"]) and device == "cuda"
    root = Path(cfg["data_root"])
    valid = WellSplit(root, "valid")
    check_counts(valid, cfg["expected"])
    dataset = WellStepDataset(root, "train", unroll=cfg["pushforward_max"] + 1)
    check_counts(dataset.split, cfg["expected"])
    normalizer = dataset.normalizer

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    saved = None
    step, skipped, best_error = 0, 0, math.inf
    total: int | None = cfg.get("total_steps")
    seconds_per_step: float | None = None
    if resume and last_path.exists():
        saved = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, skipped, best_error = saved["step"], saved["skipped_steps"], saved["best_error"]
        total, seconds_per_step = saved["total_steps"], saved["seconds_per_step"]
        saved["rng_cpu"] = saved["rng_cpu"].cpu()
        if "rng_cuda" in saved:
            saved["rng_cuda"] = [state.cpu() for state in saved["rng_cuda"]]
        if step >= total:
            return best_path

    def evaluate() -> float:
        nonlocal best_error
        model.eval()
        errors = single_rollout(
            model, valid, normalizer, cfg["rollout_start"], cfg["rollout_steps"],
            cfg["eval_batch_size"], device, amp,
        )["errors"]
        model.train()
        error = errors.double().mean().item()
        if error < best_error:
            best_error = error
            save_checkpoint(
                best_path,
                {
                    "model": model.state_dict(),
                    "step": step,
                    "val_error": error,
                    "config": cfg,
                    "normalizer": normalizer.to_dict(),
                },
            )
        return error

    def save_last() -> None:
        save_checkpoint(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "skipped_steps": skipped,
                "best_error": best_error,
                "total_steps": total,
                "seconds_per_step": seconds_per_step,
                "config": cfg,
                **_rng_state(),
            },
        )

    timing_steps = cfg.get("timing_steps", 50)
    warmup = None if total is None else cfg["warmup_fraction"] * total
    timing_began = None
    batches = iter(make_loader(dataset, cfg["batch_size"], cfg["seed"], step, cfg["num_workers"]))
    if saved is not None:  # After the loader starts, which draws from the generator.
        _set_rng_state(saved, device)
    with open(run_dir / "metrics.jsonl", "a", buffering=1) as log:
        for batch in batches:
            lr = _cosine_lr(cfg["lr"], step, total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            pushed = 0
            if warmup is not None and step >= warmup and torch.rand(1).item() < cfg["pushforward_prob"]:
                pushed = int(torch.randint(1, cfg["pushforward_max"] + 1, (1,)).item())

            optimizer.zero_grad(set_to_none=True)
            loss = training_step(model, batch, normalizer, pushed, device, amp)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                skipped += 1
            step += 1

            record = {}
            if total is None and step == TIMING_WARMUP_STEPS:
                _sync(device)
                timing_began = time.perf_counter()
            if total is None and step == max(timing_steps, TIMING_WARMUP_STEPS + 1):
                _sync(device)
                seconds_per_step = (time.perf_counter() - timing_began) / (step - TIMING_WARMUP_STEPS)
                forward_seconds = _time_forward(model, batch, device, amp)
                began = time.perf_counter()
                record["val_error"] = evaluate()
                eval_seconds = time.perf_counter() - began
                total = max(
                    step,
                    budget_steps(
                        cfg["max_minutes"] * 60, seconds_per_step, forward_seconds, eval_seconds,
                        cfg["eval_every"], cfg["warmup_fraction"], cfg["pushforward_prob"],
                        cfg["pushforward_max"],
                    ),
                )
                warmup = cfg["warmup_fraction"] * total
                record |= {"total_steps": total, "seconds_per_step": seconds_per_step}
            elif total is not None and (step % cfg["eval_every"] == 0 or step == total):
                record["val_error"] = evaluate()

            if step % cfg["log_every"] == 0 or record:
                record |= {"step": step, "lr": lr, "pushed": pushed, "skipped_steps": skipped, "loss": loss.item()}
                log.write(json.dumps(record) + "\n")
            if total is not None and (step % cfg["checkpoint_every"] == 0 or step == total):
                save_last()
            if on_step is not None:
                on_step(step)
            if total is not None and step >= total:
                break
    return best_path


def train_members(cfg: dict, run_root: Path) -> list[Path]:
    """Train or resume one member per seed, and return their best checkpoints."""
    return [
        train_member(cfg | {"seed": seed}, Path(run_root) / f"member_{seed}", resume=True)
        for seed in cfg["seeds"]
    ]
