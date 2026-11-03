"""Resumable training for one model.

``last.pt`` holds all that a restart needs, and a run killed at any moment, even inside a save,
continues from the last whole one. On the CPU a resumed run ends with the same weights as an
unbroken one. Batches depend only on the seed and step, so no loader state needs saving.
"""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sandeha.data.dataset import Normalizer, StepDataset
from sandeha.data.store import open_split, write_atomic
from sandeha.models.factory import apply_step, build_model
from sandeha.train.losses import step_loss

# A rollout frame's error above this, or not finite, counts as this (docs/PREREGISTRATION.md).
ERROR_CAP = 10.0
TIMING_WARMUP_STEPS = 2


def amp_context(device: str, amp: bool):
    return torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()


def save_checkpoint(path: Path, state: dict) -> None:
    write_atomic(Path(path), lambda file: torch.save(state, file))


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


class _Batches:
    """Endless shuffled batches of indices, fixed by the seed, and able to start at any step."""

    def __init__(self, size: int, batch_size: int, seed: int, start_step: int):
        if batch_size > size:
            raise ValueError(f"batch size {batch_size} is larger than the {size} items")
        self.size, self.batch_size, self.seed, self.start_step = size, batch_size, seed, start_step

    def __iter__(self):
        per_epoch = self.size // self.batch_size
        epoch, position = divmod(self.start_step, per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            order = torch.randperm(self.size, generator=generator)
            for i in range(position, per_epoch):
                yield order[i * self.batch_size : (i + 1) * self.batch_size].tolist()
            epoch, position = epoch + 1, 0


def make_loader(dataset, batch_size: int, seed: int, start_step: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=_Batches(len(dataset), batch_size, seed, start_step),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _velocity_scale(normalizer: Normalizer) -> tuple[float, float]:
    u, v = normalizer.delta_std[:2]
    middle = (u + v) / 2
    return u / middle, v / middle


def training_step(
    model: nn.Module,
    batch: dict,
    normalizer: Normalizer,
    pushed: int,
    lambda_div: float,
    device: str = "cpu",
    amp: bool = False,
) -> dict:
    """The loss after ``pushed`` steps on the model's own output, with no gradient through them.

    The target is the change that takes the pushed state to the true next frame.
    """
    x = batch["input"].to(device, non_blocking=True)
    inflow = batch["inflow"].to(device, non_blocking=True)
    buoyancy = batch["buoyancy"].to(device, non_blocking=True)
    masks = batch["obstacle_seq"].to(device, non_blocking=True)
    if pushed:
        with torch.no_grad(), amp_context(device, amp):
            for k in range(pushed):
                state = apply_step(model, x, normalizer)
                x = normalizer.make_input(state, masks[:, k + 1], inflow, buoyancy)
    with amp_context(device, amp):
        pred = model(x)
    # The loss runs in full precision: squared divergences underflow in half.
    with torch.autocast("cuda", enabled=False) if amp else nullcontext():
        truth = normalizer.decode(batch["target_seq"][:, pushed].to(device, non_blocking=True))
        target = normalizer.encode_delta(truth - normalizer.decode(x[:, :3]))
        return step_loss(pred.float(), target, masks[:, pushed], lambda_div, _velocity_scale(normalizer))


def capped_mean(errors: Tensor) -> float:
    errors = torch.nan_to_num(errors, nan=ERROR_CAP, posinf=ERROR_CAP).clamp(max=ERROR_CAP)
    return errors.double().mean().item()


@torch.no_grad()
def rollout_frame_errors(
    model: nn.Module,
    store,
    normalizer: Normalizer,
    start: int,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    amp: bool = False,
) -> Tensor:
    """The RMS error ``(scenes, steps)`` in normalised field units on fluid cells, uncapped."""
    if start + steps >= store.n_frames:
        raise ValueError(f"a rollout of {steps} from frame {start} doesn't fit in {store.n_frames} frames")
    errors = []
    for first in range(0, store.n_scenes, batch_size):
        scenes = range(first, min(first + batch_size, store.n_scenes))
        fields = np.array(store.fields[scenes.start : scenes.stop, start : start + steps + 1], dtype=np.float32)
        fields = torch.from_numpy(fields).to(device)
        inflow = torch.from_numpy(np.stack([store.inflow(i) for i in scenes]))[:, None].float().to(device)
        buoyancy = torch.tensor([store.buoyancy(i) for i in scenes], dtype=torch.float32, device=device)
        state = fields[:, 0]
        scene_errors = []
        for k in range(steps):
            solid = np.stack([store.obstacle(i, start + k) for i in scenes])
            solid = torch.from_numpy(solid)[:, None].to(device)
            x = normalizer.make_input(state, solid.float(), inflow, buoyancy)
            with amp_context(device, amp):
                state = apply_step(model, x, normalizer)
            squares = (normalizer.encode(state) - normalizer.encode(fields[:, k + 1])).square()
            fluid = ~solid
            scene_errors.append(((squares * fluid).sum(dim=(1, 2, 3)) / (3 * fluid.sum(dim=(1, 2, 3)))).sqrt().cpu())
        errors.append(torch.stack(scene_errors, dim=1))
    return torch.cat(errors)


def rollout_error(
    model: nn.Module,
    store,
    normalizer: Normalizer,
    start: int,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    amp: bool = False,
) -> float:
    """Mean over scenes and frames of the rollout error.

    A frame that blows up counts as ``ERROR_CAP`` and stays in the mean, so that a blown-up model
    can't be chosen as best.
    """
    errors = rollout_frame_errors(model, store, normalizer, start, steps, batch_size, device, amp)
    return capped_mean(errors)


def budget_steps(
    seconds: float,
    step_seconds: float,
    forward_seconds: float,
    eval_seconds: float,
    eval_every: int,
    warmup_fraction: float,
    pushforward_prob: float,
    pushforward_max: int,
) -> int:
    """How many steps fit in ``seconds``, counting pushed passes and validation."""
    pushed = (1 - warmup_fraction) * pushforward_prob * (1 + pushforward_max) / 2
    return int(seconds / (step_seconds + pushed * forward_seconds + eval_seconds / eval_every))


def _rng_state() -> dict:
    state = {"rng_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict, device: str) -> None:
    torch.set_rng_state(state["rng_cpu"])
    if device == "cuda" and "rng_cuda" in state:
        torch.cuda.set_rng_state_all(state["rng_cuda"])


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _cosine_lr(base: float, step: int, total: int | None) -> float:
    if total is None:
        return base
    return base * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def _time_forward(model: nn.Module, batch: dict, device: str, amp: bool, passes: int = 5) -> float:
    x = batch["input"].to(device)
    with torch.no_grad(), amp_context(device, amp):
        _sync(device)
        began = time.perf_counter()
        for _ in range(passes):
            model(x)
        _sync(device)
    return (time.perf_counter() - began) / passes


def train(cfg: dict, run_dir: Path, resume: bool, *, on_step=None) -> Path:
    """Train to ``cfg["total_steps"]``, or as many as ``cfg["max_minutes"]`` allows.

    Returns the checkpoint with the lowest validation rollout error. With ``resume``, a run
    continues from ``last.pt`` if there is one. Without it, an existing run is refused.
    ``on_step(step)`` is called after each finished step.
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
    normalizer = Normalizer.load(root / "normalizer.json")
    val_store = open_split(root, "val")
    # One start frame's targets for each pushed step, and one more for the step with gradients.
    dataset = StepDataset(open_split(root, "train"), normalizer, unroll=cfg["pushforward_max"] + 1)

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
        error = rollout_error(
            model, val_store, normalizer, cfg["rollout_start"], cfg["rollout_steps"],
            cfg["eval_batch_size"], device, amp,
        )
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
            out = training_step(model, batch, normalizer, pushed, cfg["lambda_div"], device, amp)
            if torch.isfinite(out["loss"]):
                scaler.scale(out["loss"]).backward()
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
                # The steps so far give the speed, and were ordinary training steps.
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
                record |= {"total_steps": total, "seconds_per_step": seconds_per_step,
                           "seconds_per_forward": forward_seconds, "seconds_per_validation": eval_seconds}
            elif total is not None and (step % cfg["eval_every"] == 0 or step == total):
                record["val_error"] = evaluate()

            if step % cfg["log_every"] == 0 or record:
                record |= {
                    "step": step, "lr": lr, "pushed": pushed, "skipped_steps": skipped,
                    **{key: value.item() for key, value in out.items()},
                }
                log.write(json.dumps(record) + "\n")
            # Steps before the total is known aren't saved, so a restart there just times again.
            if total is not None and (step % cfg["checkpoint_every"] == 0 or step == total):
                save_last()
            if on_step is not None:
                on_step(step)
            if total is not None and step >= total:
                break
    return best_path
