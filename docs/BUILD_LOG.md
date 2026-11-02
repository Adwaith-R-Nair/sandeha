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

- The design asks for a reboot before the overnight run, because one desktop process still had old driver files loaded. This task did not reboot. The GPU test passed without one.

### Review fixes

An independent review of Task 1 raised the items below. Each wrapper and hook item was reproduced before it was changed, using a scratch copy of the wrapper and the real `systemd-inhibit`.

| Item | Reproduced | Change |
|---|---|---|
| Test did not prove the job runs under the inhibitor | Yes, by mutant after the fix: running the job outside the inhibitor now fails 3 tests | The test also asserts that the job command is in the inhibitor's arguments |
| Job killed by a signal logged as `exit=1` | Yes: `bash terminated by signal TERM`, wrapper exit 1 | Real status recorded: 143 for SIGTERM, 137 for SIGKILL (B8) |
| Name `jobs` wrote job output into `runs/jobs.log` | Yes: `collide` appeared between `START` and `END` | Name reserved, exit 2 |
| SIGTERM to the wrapper left the job running with no `END` line | Yes: the job's marker file appeared after the wrapper exited | Signal forwarded to the job's process group (B8) |
| Hook accepted "Assisted by" plus an unlisted tool name | Yes: exit 0 | Pattern widened (B9). Attribution samples exit 1. Messages with a conventional subject and no attribution, including one about a brush cursor, exit 0. Subject rules were added to the hook separately by `532ac37`, and earlier commits are not rewritten |
| `docs/DESIGN.md` status line out of date | Fixed by the owner in `8fb4453` before this work reached it | None |
| GitHub repo missing and no push | Was true when the review ran. The repo was created under the owner's account and `main` pushed at 18:27 to 18:28 UTC, as public | `DECISIONS.md` D12 updated to match. This entry's commit is pushed with a plain `git push origin main`, closing Phase 1 |

Test runs for these fixes:

| Step | Result |
|---|---|
| New and changed wrapper tests against the old script | 9 failed, 11 passed |
| After the wrapper change | 1 failed, 19 passed. The SIGINT case timed out |
| After correcting that test | 20 passed, three runs in a row |

The SIGINT failure was in the test job, not the wrapper. The job started `sleep` with `&` from a non-interactive shell, which starts such children with SIGINT ignored (`SigIgn` in `/proc/<pid>/status` confirmed this), so the wrapper correctly kept waiting for it. The test now uses a foreground child, and the limit is documented in `docs/LLD.md`.

Mutants of the new wrapper, run against a scratch copy: signalling only the group leader (3 tests fail), removing the stop-file check (1), allowing the name `jobs` (1), and removing the repeat `wait` (2).

Hand checks with the real inhibitor, in a scratch copy:

- A job killing itself with SIGTERM and with SIGKILL: wrapper exit 143 and 137, with the same codes in `END`.
- SIGTERM to the wrapper, with a job that cleans up and exits 5: wrapper exit 5, and `systemd-inhibit --list` showed the `sandeha` entry during the cleanup.
- SIGTERM to the wrapper, with a plain job: exit 143, and the job's marker file never appeared.
- Ctrl-C in a real pseudo-terminal: wrapper exit 130, `END ... exit=130`, and the job's marker file never appeared.
- SIGKILL to the wrapper: `START` with no `END`, the job ran to completion, and the `sandeha` entry stayed listed until it finished. This is the check that `docs/LLD.md` gives for use before a restart. This path also leaves the wrapper's small state directory behind in the system temp directory.

Not exercised: a real out-of-memory kill, and a real failure of the inhibitor to start a job. The second is covered only by a fake inhibitor.

### Next

Task 2: reference solver and its tests.

## 2026-11-03: Phase 2, Task 2 (reference solver)

### What was done

- `src/sandeha/physics/solver.py`: `SmokeState` and `SmokeSolver`, with a MAC grid, semi-Lagrangian advection, and a multigrid-preconditioned conjugate gradient pressure solve per batch item.
- `tests/physics/test_solver.py`: the plan's six checks, plus further tests listed in `docs/TEST_PLAN.md`.
- `scripts/check_solver.py`: one command that writes `results/solver_check.json` and `results/figures/solver_check.png`.
- `matplotlib` added to the dependencies. It was already in the plan's stack.

### Test runs

| Step | Result |
|---|---|
| Tests with no solver module | Collection error: `ModuleNotFoundError: No module named 'sandeha.physics.solver'` |
| Tests against an interface-only stub | 16 failed, each with `NotImplementedError` |
| First implementation, using MacCormack advection | 1 failed, 15 passed. `test_converges_with_refinement`: `assert 0.6959888935089111 < 0.5129764676094055` |
| Convergence test rewritten (B12) | 17 passed |
| New artefact and mass budget tests, against MacCormack | 2 failed |
| Semi-Lagrangian advection (B10) | Full suite: 46 passed |
| Final code and tests, `uv run pytest` | Full suite: 49 passed, of which 22 are solver tests |

The full suite passed three runs in a row before the last tests were added, and was run again after they were added.

### Mutants

Faults were put into the solver file one at a time, with the solver tests run against each and the file restored afterwards.

First round, 13 mutants. Killed: rebuilding of the pressure setup removed, inflow unmasked, density uncleared in solids, buoyancy sign flipped, projection removed from `step`, asymmetric post-smoother, iteration limit ignored, preconditioner removed (by timeout), density decay removed. Survived: a step size shared across the batch, the velocity mask at the start of `step`, a first-order backtrace, and `v` advected on the wrong lattice.

Each survivor was dealt with:

- The mask at the start of `step` was removed as code with no observable effect, since faces are masked again at the end of the step. `test_smoke_does_not_cross_a_thin_wall` tests the property directly.
- The other three got new or stronger tests.

Second round, 6 mutants, all killed: shared step size, first-order backtrace in `x` and in `y`, `u` and `v` each on the wrong lattice, and faces left unmasked by solids.

### Speed

From `results/solver_check.json`. A plume with one obstacle, `dt=0.01`, timed over 100 steps after 100 warm-up steps. Peak GPU memory across the run: 144 MiB. Batch 16 was measured beyond the plan's 1 and 8, for use in sizing Task 4.

| Grid | Batch | Batch steps/s | Item steps/s | Mean CG iterations | Largest divergence | All converged |
|---|---|---|---|---|---|---|
| 256 | 1 | 20.3 | 20.3 | 8.0 | 0.00098 | yes |
| 256 | 8 | 16.7 | 133.7 | 8.0 | 0.00098 | yes |
| 256 | 16 | 12.0 | 192.0 | 8.0 | 0.00098 | yes |
| 128 | 1 | 25.0 | 25.0 | 7.0 | 0.00055 | yes |
| 128 | 8 | 25.1 | 201.0 | 7.0 | 0.00055 | yes |
| 128 | 16 | 24.3 | 388.9 | 7.0 | 0.00055 | yes |

Time per step changes little with grid size at batch 1. A profile showed low GPU use and most of the time in Python and tensor-operation overhead, so larger batches gain the most. Items in these batches are identical. Real scenes differ, and a batch waits for its slowest item, so Task 4 should measure its own throughput rather than rely on this table.

### Convergence

From `results/solver_check.json`. Error is the relative L2 difference in density against the 256×256 run, averaged down to the coarser grid. Time step is fixed.

| Inflow | Time | 128 vs 256 | 64 vs 256 |
|---|---|---|---|
| soft | 0.25 | 0.0024 | 0.0058 |
| soft | 0.5 | 0.0253 | 0.0680 |
| soft | 0.75 | 0.0929 | 0.2134 |
| soft | 1.0 | 0.1892 | 0.3769 |
| hard | 0.25 | 0.0827 | 0.1384 |
| hard | 0.5 | 0.1232 | 0.2192 |
| hard | 0.75 | 0.3038 | 0.4193 |
| hard | 1.0 | 0.4683 | 0.4317 |

128×128 is closer than 64×64 at every listed time, except for the hard-edged inflow at `t=1.0`. There, pointwise convergence is lost: the plume head has rolled up, and small position differences between fine structures dominate the error. The smooth inflow does not lose it within this window. This limit is expected for chaotic flow (`docs/DESIGN.md` section 6.3), and is not hidden by the test, which asserts each case only inside the window shown here to converge (B12).

### Visual check

`results/figures/solver_check.png` was rendered and inspected. The plume rises, parts around the obstacle without crossing it, rolls up into vortices, spreads along the walls, and stays left-right symmetric as the setup is.

The first render, using MacCormack advection, showed regular hatching along thin sheets. A zoomed comparison at `t=1.75` confirmed cell-scale stair-steps and a blocky spike above the sheet, with both a revert limiter and a clamp limiter, and none with semi-Lagrangian advection (B10). The re-render from the final code shows no hatching.

### Changed decisions

See `DECISIONS.md`: B10 to B14.

### Problems

- A fixed cost of about 50 ms per step at batch 1, regardless of grid size, appeared on the first run. Trimming V-cycle work kept the iteration count but changed the time little, so optimisation stopped there, with data generation throughput judged to be adequate at larger batches.

### Open items

- The independent phase check required by `docs/DESIGN.md` section 8 has not been done. The inspection above was done as part of this task.
- Not exercised: determinism across separate processes, and across restarts. `test_deterministic` compares two runs in one process.
- `SmokeSolver.last_solve` reports non-convergence but does not raise. Task 4 has to check it during generation.
- Semi-Lagrangian advection is dissipative: a test vortex loses a visible part of its velocity within 100 steps at 64×64. This is accepted (B10), and the 256×256 reference is less affected than coarser grids, but it is a limit on how far the reference can be called truth.

### Next

Task 3: scenes. Scene inflows should have smooth edges rather than hard ones (B12), and `dt` has to be chosen for generation. The checks here used `dt=0.01`.
