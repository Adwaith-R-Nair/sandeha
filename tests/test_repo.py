import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", path],
        capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", ["data/x.npy", "runs/x.log", "checkpoints/x.pt"])
def test_generated_output_is_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize(
    "path",
    ["src/sandeha/data/store.py", "src/sandeha/data/dataset.py", "results/exp_doubt.json"],
)
def test_source_and_results_are_not_ignored(path):
    assert not is_ignored(path)
