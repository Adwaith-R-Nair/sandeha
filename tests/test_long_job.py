import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Behaves like the real systemd-inhibit: stays alive as the parent of the command,
# and reports a command killed by a signal as a plain exit 1.
FAKE_INHIBIT = """#!/usr/bin/env bash
echo $$ > "$INHIBIT_PID_FILE"
sleep "${INHIBIT_START_DELAY:-0}"
printf '%s\\n' "$*" > "$INHIBIT_ARGS_FILE"
while [[ "${1:-}" == --* ]]; do shift; done
"$@"
status=$?
((status > 128)) && exit 1
exit "$status"
"""

# Exits without ever starting the command, as when the inhibitor lock is refused.
FAILING_INHIBIT = """#!/usr/bin/env bash
exit 1
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
    pid_file = tmp_path / "inhibit_pid"

    power = tmp_path / "power"
    power.mkdir()
    args_file = tmp_path / "inhibit_args"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SANDEHA_POWER_SUPPLY_DIR": str(power),
        "INHIBIT_ARGS_FILE": str(args_file),
        "INHIBIT_PID_FILE": str(pid_file),
    }

    class Job:
        def __init__(self):
            self.project = project
            self.power = power
            self.args_file = args_file
            self.marker = tmp_path / "ran"
            self.tmp = tmp_path
            self.pid_file = pid_file
            self.fake = fake

        def run(self, *args):
            return subprocess.run(
                [str(script), *args], env=env, capture_output=True, text=True, timeout=30
            )

        def start(self, *args, **extra_env):
            return subprocess.Popen(
                [str(script), *args],
                env={**env, **extra_env},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def jobs_log(self):
            return (self.project / "runs" / "jobs.log").read_text().splitlines()

    return Job()


def wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def process_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while Path(f"/proc/{pid}").exists():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


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
    inhibitor_args = job.args_file.read_text()
    assert "--what=sleep:idle:handle-lid-switch" in inhibitor_args
    assert f"touch {job.marker}" in inhibitor_args


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


def test_reserves_the_name_of_the_shared_job_log(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("jobs", "touch", str(job.marker))

    assert result.returncode == 2
    assert "jobs" in result.stderr
    assert not job.marker.exists()
    assert not (job.project / "runs" / "jobs.log").exists()


@pytest.mark.parametrize(
    ("script", "code"), [("kill -TERM $$", 143), ("kill -KILL $$", 137), ("exit 1", 1)]
)
def test_reports_real_status_of_a_job_killed_by_a_signal(job, script, code):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "bash", "-c", script)

    assert result.returncode == code
    assert f"exit={code} " in job.jobs_log()[-1]


def test_reports_command_not_found(job):
    make_supply(job.power, "ACAD", "Mains", "1")

    result = job.run("demo", "no-such-command-here")

    assert result.returncode == 127
    assert "exit=127 " in job.jobs_log()[-1]


def test_job_never_started_is_not_reported_as_a_job_status(job):
    make_supply(job.power, "ACAD", "Mains", "1")
    job.fake.write_text(FAILING_INHIBIT)

    result = job.run("demo", "touch", str(job.marker))

    assert result.returncode == 4
    assert "no exit status" in result.stderr
    assert not job.marker.exists()
    assert "exit=unknown " in job.jobs_log()[-1]


@pytest.mark.parametrize(
    ("sig", "code"), [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)]
)
def test_signal_to_wrapper_stops_the_job_and_its_children(job, sig, code):
    make_supply(job.power, "ACAD", "Mains", "1")
    child = job.tmp / "child_pid"
    survived = job.tmp / "survived"
    body = f"bash -c 'echo $$ > {child}; exec sleep 30'; touch {survived}"

    wrapper = job.start("demo", "bash", "-c", body)
    wait_for(child)
    time.sleep(0.1)
    wrapper.send_signal(sig)

    assert wrapper.wait(timeout=10) == code
    assert process_gone(int(child.read_text()))
    assert not survived.exists()
    assert f"exit={code} " in job.jobs_log()[-1]


def test_wrapper_waits_for_job_cleanup_and_keeps_the_inhibitor_held(job):
    make_supply(job.power, "ACAD", "Mains", "1")
    held = job.tmp / "held"
    cleanup = f'sleep 0.5; kill -0 "$(<{job.pid_file})" && touch {held}; exit 5'
    body = f"trap '{cleanup}' TERM; touch {job.marker}; while :; do sleep 0.05; done"

    wrapper = job.start("demo", "bash", "-c", body)
    wait_for(job.marker)
    wrapper.send_signal(signal.SIGTERM)

    assert wrapper.wait(timeout=10) == 5
    assert held.exists()
    lines = job.jobs_log()
    assert len(lines) == 2
    assert "END" in lines[1] and "exit=5 " in lines[1]


def test_signal_during_inhibitor_startup_still_stops_the_job(job):
    make_supply(job.power, "ACAD", "Mains", "1")
    survived = job.tmp / "survived"

    wrapper = job.start(
        "demo", "bash", "-c", f"sleep 3; touch {survived}", INHIBIT_START_DELAY="0.5"
    )
    wait_for(job.pid_file)
    time.sleep(0.1)
    wrapper.send_signal(signal.SIGTERM)

    assert wrapper.wait(timeout=10) == 143
    assert not survived.exists()
    assert "exit=143 " in job.jobs_log()[-1]
