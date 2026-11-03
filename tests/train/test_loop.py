import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from sandeha.data.dataset import Normalizer, StepDataset
from sandeha.data.store import open_split
from sandeha.models.factory import build_model
from sandeha.train.loop import (
    amp_context,
    budget_steps,
    load_checkpoint,
    make_loader,
    rollout_error,
    save_checkpoint,
    train,
    training_step,
)

REPO = Path(__file__).resolve().parents[2]


def first_batch(root, batch_size=2, unroll=4) -> tuple[dict, Normalizer]:
    normalizer = Normalizer.load(Path(root) / "normalizer.json")
    dataset = StepDataset(open_split(root, "train"), normalizer, unroll=unroll)
    return next(iter(make_loader(dataset, batch_size, seed=0, start_step=0, workers=0))), normalizer


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"arch": "unet", "base_width": 32, "levels": 4},
        {"arch": "fno", "width": 24, "modes": 8, "layers": 4},
    ],
    ids=["unet", "fno"],
)
def test_overfits_one_batch(train_root, model_cfg):
    torch.manual_seed(0)
    batch, normalizer = first_batch(train_root)
    model = build_model(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        out = training_step(model, batch, normalizer, pushed=0, lambda_div=0.1)
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0] / 10, (losses[0], losses[-1])


def test_only_the_last_pushforward_step_gets_gradients(train_root):
    batch, normalizer = first_batch(train_root)
    model = build_model({"arch": "unet", "base_width": 8, "levels": 2})
    calls = []
    model.register_forward_hook(lambda _m, _i, out: calls.append(out.requires_grad))
    training_step(model, batch, normalizer, pushed=3, lambda_div=0.1)["loss"].backward()
    assert calls == [False, False, False, True]


def test_pushforward_target_is_the_true_next_frame_from_the_pushed_state(train_root):
    """If the model predicts the given target, the step lands on the stored frame."""

    class Zero(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return x[:, :3] * self.weight

    batch, normalizer = first_batch(train_root)
    out = training_step(Zero(), batch, normalizer, pushed=2, lambda_div=0.0)
    # With no change predicted, the pushed state is the start frame, so the target is frames 0 to 3.
    start = normalizer.decode(batch["input"][:, :3])
    truth = normalizer.decode(batch["target_seq"][:, 2])
    fluid = 1 - batch["obstacle_seq"][:, 2]
    expected = (normalizer.encode_delta(truth - start) ** 2 * fluid).sum() / (3 * fluid.sum())
    assert torch.allclose(out["mse"], expected, rtol=1e-4)


def test_rollout_error_known_answer_and_capped_on_blow_up(train_root):
    class Still(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[:, :3])

    class Nan(nn.Module):
        def forward(self, x):
            return torch.full_like(x[:, :3], float("nan"))

    class Huge(nn.Module):
        def forward(self, x):
            return torch.full_like(x[:, :3], 1e30)

    store = open_split(train_root, "val")
    normalizer = Normalizer.load(train_root / "normalizer.json")
    for model in (Nan(), Huge()):
        error = rollout_error(model, store, normalizer, 1, 3, batch_size=2, device="cpu")
        assert error == 10.0

    # With no change predicted, each frame's error is against the start frame, in the store's units.
    fields = torch.from_numpy(store.fields[:, 1:4].astype("float32"))
    expected = []
    for scene in range(store.n_scenes):
        solid = torch.from_numpy(store.obstacle(scene, 1))
        for k in (1, 2):
            diff = normalizer.encode(fields[scene, k]) - normalizer.encode(fields[scene, 0] * ~solid)
            expected.append(diff[:, ~solid].square().mean().sqrt())
    got = rollout_error(Still(), store, normalizer, 1, 2, batch_size=2, device="cpu")
    assert got == pytest.approx(torch.stack(expected).mean().item(), rel=1e-4)


def test_budget_steps_covers_training_pushforward_and_validation():
    # 1 s a step, 0.5 of pushed passes at 2 extra on average after a warm-up of 0.2, at 0.5 s each,
    # and a 10 s validation every 100 steps: 1 + 0.8*0.5*2*0.5 + 0.1 = 1.5 s a step.
    steps = budget_steps(
        seconds=1500, step_seconds=1.0, forward_seconds=0.5, eval_seconds=10,
        eval_every=100, warmup_fraction=0.2, pushforward_prob=0.5, pushforward_max=3,
    )
    assert steps == 1000


def test_max_minutes_sets_total_steps_from_a_measurement(cfg, tmp_path):
    del cfg["total_steps"]
    cfg |= {"max_minutes": 0.05, "timing_steps": 6}
    train(cfg, tmp_path, resume=False)
    state = load_checkpoint(tmp_path / "last.pt")
    assert state["total_steps"] >= 6
    assert state["step"] == state["total_steps"]
    assert state["seconds_per_step"] > 0


def test_train_keeps_the_lowest_validation_error_and_the_schedule_ends(cfg, tmp_path):
    best = train(cfg, tmp_path, resume=False)
    assert best == tmp_path / "best.pt"
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    errors = [r["val_error"] for r in records if "val_error" in r]
    assert len(errors) == 2  # At steps 4 and 8.
    saved = load_checkpoint(best)
    assert saved["val_error"] == min(errors)
    assert saved["normalizer"] == Normalizer.load(Path(cfg["data_root"]) / "normalizer.json").to_dict()
    lrs = [r["lr"] for r in records if "lr" in r]
    assert lrs[0] == pytest.approx(3e-4) and lrs[-1] < lrs[0] / 10
    model = build_model(saved["config"]["model"])
    model.load_state_dict(saved["model"])


def test_pushforward_waits_for_the_warm_up(cfg, tmp_path):
    cfg |= {"pushforward_prob": 1.0, "total_steps": 8}
    train(cfg, tmp_path, resume=False)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    pushed = {r["step"]: r["pushed"] for r in records if "pushed" in r}
    assert pushed[1] == 0 and pushed[2] == 0  # Steps 0 and 1 are inside the warm-up of 2.
    assert all(1 <= pushed[s] <= 3 for s in range(3, 9))


def test_train_refuses_to_overwrite_a_run_without_resume(cfg, tmp_path):
    train(cfg, tmp_path, resume=False)
    with pytest.raises(FileExistsError):
        train(cfg, tmp_path, resume=False)


def test_a_non_finite_loss_skips_the_update(cfg, tmp_path, monkeypatch):
    import sandeha.train.loop as loop

    real = loop.training_step
    calls = []

    def poisoned(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            out["loss"] = out["loss"] * float("nan")
        return out

    monkeypatch.setattr(loop, "training_step", poisoned)
    best = train(cfg, tmp_path, resume=False)
    state = load_checkpoint(best)
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    assert load_checkpoint(tmp_path / "last.pt")["skipped_steps"] == 1


def test_checkpoint_atomic(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"step": 1})

    def interrupted(_state, file):
        file.write(b"half a checkpoint")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(path, {"step": 2})
    monkeypatch.undo()
    assert load_checkpoint(path)["step"] == 1


KILLED_RUN = """
import json, os, signal, sys
from pathlib import Path
import torch
from sandeha.train.loop import train

cfg, run_dir, kill_at, where = json.loads(sys.argv[1])
real_save = torch.save

def on_step(step):
    if where == "between" and step == kill_at:
        os.kill(os.getpid(), signal.SIGKILL)

saves = []

def save(state, file):
    saves.append(1)
    if where == "mid_save" and state.get("step") == kill_at:
        file.write(b"half a checkpoint")
        file.flush()
        os.fsync(file.fileno())
        os.kill(os.getpid(), signal.SIGKILL)
    real_save(state, file)

torch.save = save
train(cfg, Path(run_dir), resume=False, on_step=on_step)
"""


@pytest.mark.parametrize(("where", "kill_at"), [("between", 5), ("mid_save", 6)])
def test_resume_continues(cfg, tmp_path, where, kill_at):
    """A real SIGKILL, then a resume, ends exactly where an unbroken run does."""
    whole = tmp_path / "whole"
    train(cfg, whole, resume=False)

    broken = tmp_path / "broken"
    done = subprocess.run(
        [sys.executable, "-c", KILLED_RUN, json.dumps([cfg, str(broken), kill_at, where])],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert done.returncode == -signal.SIGKILL
    # The last whole checkpoint is from step 3, in both cases.
    assert load_checkpoint(broken / "last.pt")["step"] == 3
    if where == "mid_save":
        assert (broken / "last.pt.tmp").exists()

    seen = []
    train(cfg, broken, resume=True, on_step=seen.append)
    assert seen[0] == 4

    a, b = load_checkpoint(whole / "last.pt"), load_checkpoint(broken / "last.pt")
    assert a["step"] == b["step"] == cfg["total_steps"]
    assert torch.equal(a["rng_cpu"], b["rng_cpu"])
    for name, value in a["model"].items():
        assert torch.equal(value, b["model"][name]), name
    first_a = a["optimizer"]["state"][0]
    first_b = b["optimizer"]["state"][0]
    assert first_a["step"] == first_b["step"]
    assert torch.equal(first_a["exp_avg"], first_b["exp_avg"])
    assert torch.equal(first_a["exp_avg_sq"], first_b["exp_avg_sq"])
    assert a["best_error"] == b["best_error"]


def test_resume_restores_optimizer_and_rng(cfg, tmp_path):
    train(cfg, tmp_path, resume=False)
    state = load_checkpoint(tmp_path / "last.pt")
    assert state["optimizer"]["state"]
    assert state["rng_cpu"].dtype == torch.uint8


def test_resume_of_a_finished_run_only_returns_the_best(cfg, tmp_path):
    train(cfg, tmp_path, resume=False)
    seen = []
    assert train(cfg, tmp_path, resume=True, on_step=seen.append) == tmp_path / "best.pt"
    assert seen == []


def test_resume_with_nothing_saved_starts_fresh(cfg, tmp_path):
    assert train(cfg, tmp_path, resume=True) == tmp_path / "best.pt"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_peak_memory_under_budget():
    """U-Net at batch 16, the deepest training step, with mixed precision.

    The figures count this process only, so a job sharing the GPU doesn't count against the budget.
    The process's limit stays below the budget, and so it can't push that job out of memory.
    """
    budget = 5.5 * 1024**3
    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info(0)
    if free < 2 * 1024**3:
        pytest.skip(f"only {free / 1024**3:.1f} GB free on the GPU")
    torch.cuda.set_per_process_memory_fraction(min(budget, free - 512 * 1024**2) / total, 0)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model({"arch": "unet", "base_width": 32, "levels": 4}).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda")
        normalizer = Normalizer([0.0] * 3, [1.0] * 3, [1.0] * 3, 1.0, 0.0, 1.0, "train", 1)
        batch = {
            "input": torch.randn(16, 6, 128, 128),
            "target_seq": torch.randn(16, 4, 3, 128, 128),
            "obstacle_seq": torch.zeros(16, 4, 1, 128, 128),
            "inflow": torch.zeros(16, 1, 128, 128),
            "buoyancy": torch.zeros(16),
        }
        for _ in range(2):  # The second step holds the optimiser's state too.
            optimizer.zero_grad(set_to_none=True)
            out = training_step(
                model, batch, normalizer, pushed=3, lambda_div=0.1, device=device, amp=True
            )
            scaler.scale(out["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        peak = torch.cuda.max_memory_reserved(0)
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    print(f"peak reserved {peak / 1024**3:.3f} GB")
    assert peak < budget


def test_amp_context_is_off_on_the_cpu():
    with amp_context("cpu", False):
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.parametrize("name", ["train_unet", "train_fno"])
def test_configs_hold_the_planned_values(name):
    config = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
    assert config["lr"] == 3e-4
    assert config["lambda_div"] == 0.1
    assert config["mixed_precision"] is True
    if name == "train_unet":
        assert config["model"] == {"arch": "unet", "base_width": 32, "levels": 4}
    else:
        assert config["model"] == {"arch": "fno", "width": 48, "modes": 24, "layers": 4}
    assert config["rollout_steps"] == 20
    assert build_model(config["model"]) is not None


def test_train_script_runs_and_overrides(cfg, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(config),
         "--run-dir", str(run), "--seed", "9"],
        capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert done.returncode == 0, done.stderr
    assert str(run / "best.pt") in done.stdout
    assert load_checkpoint(run / "best.pt")["config"]["seed"] == 9


def test_throughput_script_writes_a_result(cfg, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "throughput.json"
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "measure_throughput.py"), "--config", str(config),
         "--batch-sizes", "2", "--steps", "2", "--device", "cpu", "--out", str(out)],
        capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result["runs"][0]["arch"] == "unet"
    assert result["runs"][0]["steps_per_second"] > 0
    assert result["runs"][0]["seconds_per_forward"] > 0
    assert result["runs"][0]["div_target_mean_squared"] >= 0
    assert result["git_commit"]
