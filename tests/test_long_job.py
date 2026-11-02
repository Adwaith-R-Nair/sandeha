import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_INHIBIT = """#!/usr/bin/env bash
printf '%s\\n' "$*" > "$INHIBIT_ARGS_FILE"
while [[ "${1:-}" == --* ]]; do shift; done
exec "$@"
"""


def make_supply(root: Path, name: str, kind: str, online: str | None) -> None:
    supply = root / name
    supply.mkdir(parents=True)
    (supply / "type").write_text(f"{kind}\n")
    if online is not None:
        (supply / "online").write_text(f"{online}\n")


@pytest.fixture
def job(tmp_path):
    """A copy of the wrapper in a throwaway project, with a recording inhibitor on PATH."""
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    script = project / "scripts" / "long_job.sh"
    shutil.copy(REPO / "scripts" / "long_job.sh", script)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemd-inhibit"
    fake.write_text(FAKE_INHIBIT)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    power = tmp_path / "power"
    power.mkdir()
    args_file = tmp_path / "inhibit_args"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SANDEHA_POWER_SUPPLY_DIR": str(power),
        "INHIBIT_ARGS_FILE": str(args_file),
    }

    class Job:
        def __init__(self):
            self.project = project
            self.power = power
            self.args_file = args_file
            self.marker = tmp_path / "ran"

        def run(self, *args):
            return subprocess.run(
                [str(script), *args], env=env, capture_output=True, text=True, timeout=30
            )

    return Job()


def test_refuses_when_mains_offline(job):
    make_supply(job.power, "ACAD", "Mains", "0")
    make_supply(job.power, "BAT1", "Battery", None)

    result = job.run("demo", "touch", str(job.marker))

    assert result.returncode == 3
    assert "mains" in result.stderr.lower()
    assert not job.marker.exists()
    assert not job.args_file.exists()


def test_refuses_when_mains_state_is_unknown(job):
    make_supply(job.power, "BAT1", "Battery", None)

    result = job.run("demo", "touch", str(job.marker))

    assert result.returncode == 3
    assert not job.marker.exists()


def test_runs_under_inhibitor_when_on_mains(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "touch", str(job.marker))

    assert result.returncode == 0, result.stderr
    assert job.marker.exists()
    assert "--what=sleep:idle:handle-lid-switch" in job.args_file.read_text()


def test_logs_output_and_start_and_end(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "bash", "-c", "echo to-stdout; echo to-stderr >&2")

    assert result.returncode == 0, result.stderr
    job_log = (job.project / "runs" / "demo.log").read_text()
    assert "to-stdout" in job_log
    assert "to-stderr" in job_log
    assert "to-stdout" in result.stdout

    lines = (job.project / "runs" / "jobs.log").read_text().splitlines()
    assert len(lines) == 2
    assert "START" in lines[0] and "demo" in lines[0]
    assert "END" in lines[1] and "demo" in lines[1] and "exit=0" in lines[1]


def test_propagates_exit_code(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "bash", "-c", "exit 7")

    assert result.returncode == 7
    assert "exit=7" in (job.project / "runs" / "jobs.log").read_text()


def test_runs_from_project_root(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "bash", "-c", "pwd")

    assert result.stdout.strip() == str(job.project)


@pytest.mark.parametrize("args", [(), ("only-a-name",), ("../escape", "true")])
def test_rejects_bad_usage(job, args):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run(*args)

    assert result.returncode == 2
    assert not (job.project.parent / "escape.log").exists()
