import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from sandeha.data.dataset import Normalizer
from sandeha.data.store import open_split
from sandeha.models.factory import build_model
from sandeha.train import campaign
from sandeha.train.loop import ERROR_CAP, load_checkpoint, rollout_error, rollout_frame_errors, train

REPO = Path(__file__).resolve().parents[2]
SETTINGS = {"seeds": [0, 1], "max_minutes": 48, "retry_seed_offset": 100, "check_start": 1, "check_steps": 6}


class Stop(Exception):
    pass


def verdict(passed: bool) -> dict:
    return {"passed": passed, "reason": None if passed else "no improvement over persistence"}


@pytest.fixture
def trained(monkeypatch):
    """Records each run ``train`` is asked for, then does it."""
    calls = []

    def recording(cfg, run_dir, resume, **kwargs):
        calls.append((Path(run_dir).name, cfg))
        return train(cfg, run_dir, resume, **kwargs)

    monkeypatch.setattr(campaign, "train", recording)
    return calls


@pytest.fixture
def fails(monkeypatch):
    """Failing run directories' names are added to the returned set. Others pass."""
    failing, calls = set(), []

    def check(best_path, cfg, start, steps):
        name = Path(best_path).parent.name
        calls.append(name)
        return verdict(name not in failing)

    monkeypatch.setattr(campaign, "check_member", check)
    return failing, calls


def test_frame_errors_keep_a_blow_up_and_the_mean_caps_it(train_root):
    store = open_split(train_root, "val")
    normalizer = Normalizer.load(train_root / "normalizer.json")

    class Nan(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x[:, :3], math.nan)

    errors = rollout_frame_errors(Nan(), store, normalizer, 1, 3, batch_size=2)
    assert errors.shape == (store.n_scenes, 3)
    assert errors.isnan().all()
    assert rollout_error(Nan(), store, normalizer, 1, 3, batch_size=2) == pytest.approx(ERROR_CAP)


def test_is_finished_only_when_all_steps_are_done(cfg, tmp_path):
    run = tmp_path / "run"
    assert not campaign.is_finished(run)

    def stop(step):
        if step == 6:
            raise Stop

    with pytest.raises(Stop):
        train(cfg, run, resume=False, on_step=stop)
    assert not campaign.is_finished(run)
    train(cfg, run, resume=True)
    assert campaign.is_finished(run)


def test_run_member_skips_a_finished_run(cfg, tmp_path, trained):
    run = tmp_path / "run"
    best = campaign.run_member(cfg, run)
    assert len(trained) == 1
    assert campaign.run_member(cfg, run) == best
    assert len(trained) == 1


def test_run_member_resumes_a_partial_run(cfg, tmp_path):
    run = tmp_path / "run"

    def stop(step):
        if step == 6:
            raise Stop

    with pytest.raises(Stop):
        train(cfg, run, resume=False, on_step=stop)
    assert load_checkpoint(run / "last.pt")["step"] == 6
    campaign.run_member(cfg, run)
    assert load_checkpoint(run / "last.pt")["step"] == cfg["total_steps"]


def test_check_rejects_a_blown_up_member(cfg, tmp_path):
    best = train(cfg, tmp_path / "run", resume=False)
    saved = load_checkpoint(best)
    saved["model"] = {key: value.float() * 1e12 if value.is_floating_point() else value
                      for key, value in saved["model"].items()}
    torch.save(saved, best)
    result = campaign.check_member(best, cfg, 1, 6)
    assert result["blown_up_frames"] > 0
    assert not result["finite"]
    assert not result["passed"]
    assert "blew up" in result["reason"]


def test_check_rejects_a_member_no_better_than_persistence(cfg, tmp_path):
    best = train(cfg, tmp_path / "run", resume=False)
    saved = load_checkpoint(best)
    saved["model"] = {key: torch.zeros_like(value) for key, value in saved["model"].items()}
    torch.save(saved, best)
    result = campaign.check_member(best, cfg, 1, 6)
    assert result["finite"]
    assert result["error"] == pytest.approx(result["persistence_error"])
    assert not result["improved"]
    assert not result["passed"]
    assert "persistence" in result["reason"]


def test_check_passes_a_finite_member_better_than_persistence(cfg, tmp_path, monkeypatch):
    best = train(cfg, tmp_path / "run", resume=False)
    real = campaign.rollout_frame_errors

    def smaller_for_the_member(model, *args, **kwargs):
        errors = real(model, *args, **kwargs)
        return errors if isinstance(model, campaign.Persistence) else errors.nan_to_num() * 0 + 1e-6

    monkeypatch.setattr(campaign, "rollout_frame_errors", smaller_for_the_member)
    result = campaign.check_member(best, cfg, 1, 6)
    assert result["passed"]
    assert result["reason"] is None
    assert result["error"] < result["persistence_error"]
    assert result["steps"] == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("model", [{"arch": "unet", "base_width": 8, "levels": 2},
                                   {"arch": "fno", "width": 6, "modes": 3, "layers": 2}], ids=["unet", "fno"])
def test_check_runs_on_the_gpu_with_mixed_precision(cfg, tmp_path, model):
    gpu = cfg | {"device": "cuda", "mixed_precision": True, "model": model}
    best = train(gpu, tmp_path / "run", resume=False)
    result = campaign.check_member(best, gpu, 1, 6)
    assert result["error"] > 0 and result["persistence_error"] > 0


def test_ensemble_trains_every_seed_on_its_own_with_the_budget(cfg, tmp_path, trained, fails):
    results = tmp_path / "ensemble.json"
    report = campaign.train_ensemble(cfg, SETTINGS, tmp_path / "runs", results)
    assert [name for name, _ in trained] == ["seed_0", "seed_1"]
    assert [member_cfg["seed"] for _, member_cfg in trained] == [0, 1]
    assert all(member_cfg["max_minutes"] == 48 for _, member_cfg in trained)
    assert report["complete"]
    saved = json.loads(results.read_text())
    assert [m["trained_seed"] for m in saved["members"]] == [0, 1]
    assert all(m["passed"] and m["checkpoint_sha256"] for m in saved["members"])
    first = saved["members"][0]["checkpoint"]
    assert load_checkpoint(first)["config"]["seed"] == 0


def test_a_failing_member_is_retrained_once_with_seed_plus_100(cfg, tmp_path, trained, fails):
    failing, _ = fails
    failing.add("seed_1")
    report = campaign.train_ensemble(cfg, SETTINGS, tmp_path / "runs", tmp_path / "ensemble.json")
    assert [name for name, _ in trained] == ["seed_0", "seed_1", "seed_101"]
    assert trained[-1][1]["seed"] == 101
    member = report["members"][1]
    assert member["seed"] == 1 and member["trained_seed"] == 101 and member["passed"]
    assert "persistence" in member["retrained_because"]
    assert report["complete"]


def test_a_member_failing_twice_is_recorded_and_not_tried_again(cfg, tmp_path, trained, fails):
    failing, _ = fails
    failing |= {"seed_1", "seed_101"}
    report = campaign.train_ensemble(cfg, SETTINGS, tmp_path / "runs", tmp_path / "ensemble.json")
    assert [name for name, _ in trained] == ["seed_0", "seed_1", "seed_101"]
    assert not report["members"][1]["passed"]
    assert not report["complete"]


def test_a_second_run_trains_and_checks_nothing_again(cfg, tmp_path, trained, fails):
    failing, checks = fails
    failing.add("seed_1")
    for _ in range(2):
        report = campaign.train_ensemble(cfg, SETTINGS, tmp_path / "runs", tmp_path / "ensemble.json")
    assert len(trained) == 3
    assert len(checks) == 3
    assert report["members"][1]["trained_seed"] == 101


def test_a_run_stopped_after_a_failed_check_goes_on_to_the_retrain(cfg, tmp_path, trained, fails):
    failing, _ = fails
    failing.add("seed_0")
    settings = dict(SETTINGS, seeds=[0])

    def recording(run_cfg, run_dir, resume, **kwargs):
        if run_cfg["seed"] == 100:
            raise Stop
        return train(run_cfg, run_dir, resume, **kwargs)

    original = campaign.train
    campaign.train = recording
    try:
        with pytest.raises(Stop):
            campaign.train_ensemble(cfg, settings, tmp_path / "runs", tmp_path / "ensemble.json")
    finally:
        campaign.train = original
    report = campaign.train_ensemble(cfg, settings, tmp_path / "runs", tmp_path / "ensemble.json")
    assert [name for name, _ in trained] == ["seed_100"]
    assert report["members"][0]["trained_seed"] == 100


def fno(cfg: dict) -> dict:
    return dict(cfg, model={"arch": "fno", "width": 6, "modes": 3, "layers": 2})


def test_selection_writes_the_winner_on_validation_error(cfg, tmp_path):
    results = tmp_path / "arch_selection.json"
    report = campaign.select_architecture({"unet": cfg, "fno": fno(cfg)}, tmp_path / "runs", results)
    saved = json.loads(results.read_text())
    assert saved == report
    errors = {name: saved["candidates"][name]["val_error"] for name in ("unet", "fno")}
    assert saved["winner"] == min(errors, key=errors.get)
    assert saved["split"] == "val"
    assert saved["persistence_error"] > 0
    assert saved["data_manifest_sha256"]
    for name in ("unet", "fno"):
        best = load_checkpoint(saved["candidates"][name]["checkpoint"])
        assert saved["candidates"][name]["val_error"] == best["val_error"]
        assert best["config"]["model"]["arch"] == name
        assert best["config"]["seed"] == cfg["seed"]


@pytest.mark.parametrize("errors,winner", [((0.2, 0.1), "fno"), ((0.1, 0.2), "unet"), ((0.1, 0.1), "unet")])
def test_selection_takes_the_lower_error_and_a_tie_goes_to_unet(errors, winner):
    assert campaign.pick_winner({"unet": {"val_error": errors[0]}, "fno": {"val_error": errors[1]}}) == winner


def test_selection_continues_with_unet_if_fno_fails(cfg, tmp_path, monkeypatch):
    real = campaign.train

    def fno_fails(run_cfg, run_dir, resume, **kwargs):
        if run_cfg["model"]["arch"] == "fno":
            raise torch.cuda.OutOfMemoryError("out of memory")
        return real(run_cfg, run_dir, resume, **kwargs)

    monkeypatch.setattr(campaign, "train", fno_fails)
    report = campaign.select_architecture({"unet": cfg, "fno": fno(cfg)}, tmp_path / "runs", tmp_path / "a.json")
    assert report["winner"] == "unet"
    assert "out of memory" in report["candidates"]["fno"]["failed"]


def test_selection_stops_if_unet_fails(cfg, tmp_path, monkeypatch):
    def fails(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(campaign, "train", fails)
    results = tmp_path / "a.json"
    with pytest.raises(RuntimeError, match="broken"):
        campaign.select_architecture({"unet": cfg, "fno": fno(cfg)}, tmp_path / "runs", results)
    assert not results.exists()


def test_ablations_change_only_what_they_name(cfg, tmp_path, trained):
    overlays = {
        name: yaml.safe_load((REPO / "configs" / f"ablation_{name}.yaml").read_text())
        for name in ("no_divergence", "no_pushforward")
    }
    paths = campaign.train_ablations(cfg, overlays, tmp_path / "runs")
    assert sorted(paths) == ["no_divergence", "no_pushforward"]
    runs = dict(trained)
    assert runs["no_divergence"]["lambda_div"] == 0
    assert runs["no_divergence"]["pushforward_prob"] == cfg["pushforward_prob"]
    assert runs["no_pushforward"]["pushforward_prob"] == 0
    assert runs["no_pushforward"]["lambda_div"] == cfg["lambda_div"]
    assert all(run["seed"] == 0 and run["max_minutes"] == 25 for run in runs.values())
    assert all(path.exists() for path in paths.values())


def test_no_pushforward_config_never_pushes(cfg, tmp_path):
    run = tmp_path / "run"
    train(dict(cfg, pushforward_prob=0.0, warmup_fraction=0.0), run, resume=False)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert records and all(record["pushed"] == 0 for record in records)


def test_shipped_settings_match_the_plan():
    ensemble = yaml.safe_load((REPO / "configs/ensemble.yaml").read_text())
    data = yaml.safe_load((REPO / "configs/data.yaml").read_text())
    assert ensemble["seeds"] == [0, 1, 2, 3, 4]
    assert ensemble["max_minutes"] == 48
    assert ensemble["retry_seed_offset"] == 100
    assert ensemble["check_steps"] == 100
    assert ensemble["check_start"] + ensemble["check_steps"] < data["frames"]
    for arch in ("unet", "fno"):
        config = yaml.safe_load((REPO / f"configs/train_{arch}.yaml").read_text())
        assert config["max_minutes"] == 40 and config["seed"] == 0 and config["model"]["arch"] == arch
        build_model(config["model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny data comes from the GPU solver")
@pytest.mark.parametrize(
    "script,args,produced",
    [
        ("select_arch.py", [], "arch_selection.json"),
        ("train_ensemble.py", [], "ensemble.json"),
        ("train_ensemble.py", ["--ablations"], "ablation/no_pushforward/best.pt"),
    ],
)
def test_dry_run_leaves_the_repo_alone(tmp_path, script, args, produced):
    # By name, since other jobs may be writing elsewhere in these directories.
    real = ["results/arch_selection.json", "results/ensemble.json", "runs/select", "runs/ensemble", "runs/ablation"]
    before = {name: (REPO / name).exists() for name in real}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--dry-run", "--out", str(tmp_path / "out"), *args],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "out" / produced).exists()
    assert {name: (REPO / name).exists() for name in real} == before
