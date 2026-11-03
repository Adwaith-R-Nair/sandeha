"""Training a member, and the two stages of the evaluation."""

import copy
import json

import pytest
import torch

from sandeha.train.loop import load_checkpoint
from sandeha.well.evaluate import LABEL, evaluate_final, select_on_valid
from sandeha.well.train import train_member, train_members


class _Stop(Exception):
    pass


def _weights(path):
    return load_checkpoint(path)["model"]


def test_training_writes_best_and_resumes_exactly(well_cfg, tmp_path):
    train_member(well_cfg, tmp_path / "whole", resume=False)

    def stop_at_4(step):
        if step == 4:
            raise _Stop

    with pytest.raises(_Stop):
        train_member(well_cfg, tmp_path / "broken", resume=False, on_step=stop_at_4)
    train_member(well_cfg, tmp_path / "broken", resume=True)
    whole, broken = _weights(tmp_path / "whole" / "last.pt"), _weights(tmp_path / "broken" / "last.pt")
    for name in whole:
        assert torch.equal(whole[name], broken[name]), name
    assert load_checkpoint(tmp_path / "whole" / "best.pt")["normalizer"]["fitted_on"] == "train"


def test_training_refuses_an_existing_run(well_cfg, tmp_path):
    train_member(well_cfg, tmp_path, resume=False)
    with pytest.raises(FileExistsError):
        train_member(well_cfg, tmp_path, resume=False)


def test_time_budget_sets_the_total(well_cfg, tmp_path):
    cfg = copy.deepcopy(well_cfg)
    del cfg["total_steps"]
    cfg |= {"max_minutes": 0.02, "timing_steps": 3}
    train_member(cfg, tmp_path, resume=False)
    saved = load_checkpoint(tmp_path / "last.pt")
    assert saved["step"] == saved["total_steps"] >= 3


def test_training_refuses_wrong_counts(well_cfg, tmp_path):
    cfg = copy.deepcopy(well_cfg)
    cfg["expected"]["steps"] = 101
    with pytest.raises(ValueError):
        train_member(cfg, tmp_path, resume=False)


@pytest.fixture
def members(well_cfg, tmp_path):
    return train_members(well_cfg, tmp_path / "runs")


def test_selection_opens_valid_only_and_final_is_labelled(well_cfg, members, tmp_path, monkeypatch):
    import sandeha.well.evaluate as evaluate

    opened = []
    real = evaluate.WellSplit

    def spy(root, split, final_eval=False):
        opened.append(split)
        return real(root, split, final_eval)

    monkeypatch.setattr(evaluate, "WellSplit", spy)
    selection_path = tmp_path / "well_selection.json"
    select_on_valid(well_cfg, members, selection_path)
    assert opened == ["valid"]
    selection = json.loads(selection_path.read_text())
    assert selection["aggregation"] in ("mean", "p95", "p99")
    assert selection["label"] == LABEL
    assert len(selection["doubt_deciles"]) == 9

    result_path = tmp_path / "exp_well.json"
    result = evaluate_final(well_cfg, members, selection_path, result_path)
    assert opened == ["valid", "test"]
    assert json.loads(result_path.read_text()) == result
    assert result["label"] == LABEL
    assert "not measured data" in LABEL
    assert result["split"] == "test"
    assert result["provenance"]["checkpoints_sha256"].keys() == {m.parent.name for m in members}
    assert "git_commit" in result["provenance"] and "data_sha256" in result["provenance"]
    assert result["provenance"]["config"]["seeds"] == [0, 1, 2]
    for window in ("evidence", "secondary"):
        block = result[window]
        for signal in ("doubt", "no_ensemble_divergence"):
            measures = block[signal]
            assert {"within_time_spearman", "pooled_spearman", "within_time_auroc", "pooled_auroc",
                    "calibration", "risk_coverage", "non_finite_frames"} <= measures.keys()
    assert "not applicable" in result["rise_on_unfamiliar"]


def test_final_needs_a_selection(well_cfg, members, tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_final(well_cfg, members, tmp_path / "absent.json", tmp_path / "exp_well.json")


def test_final_needs_a_selection_from_the_same_members(well_cfg, members, tmp_path):
    selection_path = tmp_path / "well_selection.json"
    select_on_valid(well_cfg, members, selection_path)
    with pytest.raises(ValueError):
        evaluate_final(well_cfg, members[:2], selection_path, tmp_path / "exp_well.json")
