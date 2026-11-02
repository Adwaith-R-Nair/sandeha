# Sandeha: Build Log

A dated record of what happened in each phase, including failures and changed decisions. Written so that an interrupted build can pick up where it stopped.

## 2026-11-02: Phase 1, Task 1 (repo, environment, docs skeleton)

### Environment as measured

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 595.99.02, CUDA 13.2 |
| Python | CPython 3.12.13 through `uv` 0.12.1 |
| PyTorch | 2.15.0+cu132, `torch.cuda.is_available()` is `True` |
| NumPy | 2.5.4 |
| Inhibitor | `systemd-inhibit --what=sleep:idle:handle-lid-switch` exits 0 without elevated rights |
| Power | `ACAD`, type `Mains`, `online=1` during this task |

### What was done

- Git repo, identity in repo config, and a local `commit-msg` hook. The hook was tried on sample messages: plain messages exit 0, and messages with a co-author trailer or tool attribution exit 1.
- `uv` project on Python 3.12, with the `sandeha` package installed from `src/`.
- `scripts/long_job.sh`, and tests for it.
- `PRD`, `HLD` (with diagram), `LLD`, `TEST_PLAN`, `DECISIONS` and this log.

### Test runs

| Step | Result |
|---|---|
| `tests/test_env.py` before PyTorch was installed | Collection error: `ModuleNotFoundError: No module named 'torch'` |
| `tests/test_long_job.py` before the wrapper existed | 9 errors: `FileNotFoundError` on `scripts/long_job.sh`. The 6 tests in `tests/test_repo.py` passed in the same run, since `.gitignore` already existed |
| Wrapper and repo tests after implementation | 15 passed |
| `uv run pytest`, full suite after PyTorch and NumPy | 16 passed |

`tests/test_repo.py` was written after `.gitignore`, so it was not seen failing against the final file. Its target was checked separately: in a scratch repo, the plan's `data/` pattern made `git check-ignore` report `src/sandeha/data/store.py` as ignored.

### Hand check of the wrapper

Run with the real inhibitor and real power state: `scripts/long_job.sh task1-smoke bash -c '...'` exited 0, wrote `START` and `END` lines to `runs/jobs.log`, and wrote the output to `runs/task1-smoke.log`. A `systemd-inhibit --list` from inside the job showed the `sandeha` inhibitor held. Run from another directory, the command's working directory was the project root.

Not exercised: the exit 3 path against a real unplugged adapter. It is covered only by tests with a fake power-supply directory.

### Changed decisions

See `DECISIONS.md`: B2 (anchored `.gitignore`), B3 (local tooling excludes), B5 (reworded design text) and B6 (unknown mains state refuses).

### Problems

- The first `uv add` let packages resolve against the PyTorch index and pulled older versions of some. Fixed by marking the index `explicit` and regenerating `uv.lock` from scratch. After that, only `torch` comes from the PyTorch index and the other 35 registry packages come from PyPI.
- An early test run warned `Failed to initialize NumPy`. Adding NumPy, already in the plan's stack, removed it.

### Open items

- **Not done: private GitHub repo and first push.** `gh repo create Adwaith-R-Nair/sandeha --private --source=. --remote=origin --push` was refused by a permission check on the build machine before it ran, and the push was not retried another way. The work exists only in the local repo until the owner runs that command or allows it. The end-of-phase push rule is unmet until then.
- `docs/DESIGN.md` still reads "Status: Awaiting owner approval". The owner has since approved the build, and the status line was left for the owner to change.
- The design asks for a reboot before the overnight run, because one desktop process still had old driver files loaded. This task did not reboot. The GPU test passed without one.

### Next

Task 2: reference solver and its tests.
