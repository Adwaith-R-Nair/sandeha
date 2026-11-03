"""The documented download must fetch the HDF5 files. The hub calls are mocked, so no network."""

import re
import shlex
import sys
from pathlib import Path

import huggingface_hub.cli.download as hf_download
from huggingface_hub.cli import hf
from huggingface_hub.utils import filter_repo_objects

BUILD_LOG = Path(__file__).parents[2] / "docs" / "BUILD_LOG.md"


def documented_commands():
    return re.findall(r"^\s*uv run (hf download .+)$", BUILD_LOG.read_text(), flags=re.MULTILINE)


def test_documented_download_selects_every_split(monkeypatch):
    commands = documented_commands()
    assert commands

    for command in commands:
        snapshots, single_files = [], []
        monkeypatch.setattr(
            hf_download, "snapshot_download", lambda **kw: snapshots.append(kw) or "path"
        )
        monkeypatch.setattr(
            hf_download, "hf_hub_download", lambda **kw: single_files.append(kw) or "path"
        )
        monkeypatch.setattr(sys, "argv", shlex.split(command))
        try:
            hf.main()
        except SystemExit as stop:
            assert not stop.code

        assert not single_files, command
        assert len(snapshots) == 1, command
        assert snapshots[0]["local_dir"] == "data/well"
        repo_files = [f"data/{split}/a.hdf5" for split in ("train", "valid", "test")]
        repo_files.append("visualization.ipynb")
        selected = list(
            filter_repo_objects(repo_files, allow_patterns=snapshots[0]["allow_patterns"])
        )
        assert selected == repo_files[:3], command
