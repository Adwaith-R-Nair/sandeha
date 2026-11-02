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

- The mask at the start of `step` was removed as code with no observable effect, since faces are masked again at the end of the step. `test_smoke_does_not_cross_a_thin_wall` tests the property directly. **This was wrong.** The mask has an effect on the first step after an obstacle changes, which no test looked at. It was restored after review (see Review fixes below, and B15).
- The other three got new or stronger tests.

Second round, 6 mutants, all killed: shared step size, first-order backtrace in `x` and in `y`, `u` and `v` each on the wrong lattice, and faces left unmasked by solids.

### Speed

From `results/solver_check.json`, regenerated at commit `2b2df8a` with a clean tree. A plume with one obstacle, `dt=0.01`, timed over 100 steps after 100 warm-up steps. Batch 16 was measured beyond the plan's 1 and 8, for use in sizing Task 4. Peak GPU memory across the whole script: 152 MiB. The largest divergence column covers only the timed window, steps 101 to 200. See "Divergence over a longer run" for what happens outside it.

| Grid | Batch | Batch steps/s | Item steps/s | Mean CG iterations | Largest divergence | All converged |
|---|---|---|---|---|---|---|
| 256 | 1 | 19.8 | 19.8 | 8.0 | 0.00098 | yes |
| 256 | 8 | 16.7 | 133.4 | 8.0 | 0.00098 | yes |
| 256 | 16 | 11.9 | 190.2 | 8.0 | 0.00098 | yes |
| 128 | 1 | 24.7 | 24.7 | 7.0 | 0.00055 | yes |
| 128 | 8 | 23.6 | 188.4 | 7.0 | 0.00055 | yes |
| 128 | 16 | 24.6 | 394.2 | 7.0 | 0.00055 | yes |

Time per step changes little with grid size at batch 1. A profile showed low GPU use and most of the time in Python and tensor-operation overhead, so larger batches gain the most. Items in these batches are identical. Real scenes differ, and a batch waits for its slowest item, so Task 4 should measure its own throughput rather than rely on this table.

### Convergence

From `results/solver_check.json`. Error is the relative L2 difference in density against the 256×256 run, averaged down to the coarser grid. Time step is fixed.

| Inflow | Time | 128 vs 256 | 64 vs 256 | Ratio |
|---|---|---|---|---|
| soft | 0.25 | 0.0024 | 0.0058 | 0.414 |
| soft | 0.5 | 0.0253 | 0.0680 | 0.372 |
| soft | 0.75 | 0.0929 | 0.2134 | 0.435 |
| soft | 1 | 0.1892 | 0.3769 | 0.502 |
| soft | 1.25 | 0.3243 | 0.5183 | 0.626 |
| soft | 1.5 | 0.4264 | 0.5632 | 0.757 |
| soft | 1.75 | 0.4826 | 0.6052 | 0.797 |
| soft | 2 | 0.5670 | 0.6465 | 0.877 |
| soft | 2.25 | 0.6639 | 0.6809 | 0.975 |
| soft | 2.5 | 0.7157 | 0.7092 | 1.009 |
| soft | 2.75 | 0.7263 | 0.7212 | 1.007 |
| soft | 3 | 0.7287 | 0.7588 | 0.960 |
| hard | 0.25 | 0.0827 | 0.1384 | 0.597 |
| hard | 0.5 | 0.1232 | 0.2192 | 0.562 |
| hard | 0.75 | 0.3038 | 0.4193 | 0.725 |
| hard | 1 | 0.4683 | 0.4317 | 1.085 |
| hard | 1.25 | 0.5125 | 0.4941 | 1.037 |
| hard | 1.5 | 0.5600 | 0.5650 | 0.991 |
| hard | 1.75 | 0.5821 | 0.6398 | 0.910 |
| hard | 2 | 0.5912 | 0.6548 | 0.903 |
| hard | 2.25 | 0.6488 | 0.6620 | 0.980 |
| hard | 2.5 | 0.6320 | 0.6596 | 0.958 |
| hard | 2.75 | 0.6661 | 0.6751 | 0.987 |
| hard | 3 | 0.6804 | 0.6978 | 0.975 |

128×128 is clearly closer than 64×64 only up to a limit, and the ratio column shows the margin closing:

- **Hard inflow:** the order is reversed at `t=1.0` and `t=1.25` (128 error above 64 error). From `t=1.5` the order returns, with ratios from 0.903 to 0.991, while both errors stay large.
- **Soft inflow:** ordered through `t=2.25`, with the ratio rising from 0.502 at `t=1` to 0.877 at `t=2` and 0.975 at `t=2.25`. The order is reversed at `t=2.5` and `t=2.75`, and it is 0.960 at `t=3`.

Once the plume head has rolled up, small position differences between fine structures dominate the error. This is expected for chaotic flow (`docs/DESIGN.md` section 6.3). The test asserts each case only inside the window shown here to converge (B12).

**Limit on the reference.** The design's starting horizon of 50 neural steps is 200 solver steps, or 2 time units at `dt=0.01`. With the plan's start at frame 20, the window runs from `t=0.8` to `t=2.8`, which is well past the tested window (`t=1` for soft inflow) and into the range where grid order is lost in this plume. By then the 128×128 run differs from the 256×256 run by a relative error above 0.7. The 256×256 run was never compared with a finer grid, so its own pointwise error is unmeasured, and past about `t=1` it should be called a converged reference only in a statistical sense. Task 4 has to take this into account when fixing the horizon and metrics, and scene flows and `dt` may shift the numbers.

### Divergence over a longer run

From `long_run` in `results/solver_check.json`: the same 256×256 plume for 300 steps.

`cg_tol` is relative to the fastest face, in net-flux units, so the limit in physical units is `cg_tol × fastest face × res`, and it grows with grid size and flow speed. On this run it reached 0.00394.

| Measure | Value |
|---|---|
| Largest divergence, whole run | 0.00272 |
| Largest divergence, steps 101 to 200 (the timed window above) | 0.00098 |
| Steps above `1e-3` | 91 of 300 |
| Every solve reported converged | yes |

The plan's `1e-3` limit is set at 64×64, so this breaks no test. It does mean that a converged solve at 256×256 does not guarantee divergence below `1e-3`.



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
- Semi-Lagrangian advection can jump a thin wall when flow is fast enough. In a one-off check on a new one-cell wall at 64×64, with uniform flow through it, no smoke crossed at 3.2 cells per step, and smoke crossed at 6.4 and above. The 256×256 plume above peaks near 3.9 cells per step (`max_physical_limit` divided by `cg_tol`, times `dt`). Static walls and faster flows at this grid size were not measured. Task 3 should check that its obstacles and speeds stay clear of this, and thin demo brush strokes may show it.
- Semi-Lagrangian advection is dissipative: a test vortex loses a visible part of its velocity within 100 steps at 64×64. This is accepted (B10), and the 256×256 reference is less affected than coarser grids, but it is a limit on how far the reference can be called truth.

### Review fixes

An independent review of Task 2 raised the items below. Each was reproduced at `3840702` before anything changed.

| Item | Reproduced | Change |
|---|---|---|
| Density inside a new obstacle on its first step | Yes. 40 plume steps, then a box at `y` 0.1 to 0.3 at 64×64: largest density in solid 0.0887 after step 1, and 0 after step 2. The review measured 0.378 with its own box. The old test ran 2 steps before asserting, which hid it | Faces are masked at the start of `step` (B15). The test asserts after exactly 1 step |
| A NaN or infinite state passed to `step` crashes | Yes. On GPU: `AcceleratorError: device-side assert triggered`, and an unrelated `torch.zeros(3)+1` on the GPU then raised the same error. On CPU: `RuntimeError: index ... out of bounds`. NaN or infinity only in density got through step 1 and crashed on step 2. The old test never called `step` | Lookup indices are kept in range and NaN carries through (B16). Batched `step` tests on GPU and CPU |
| Velocity self-advection untested | Yes. With `new_u = u` and `new_v = v` in a scratch copy: 22 passed in 33.66s | `test_vortex_pair_carries_itself` |
| 256×256 divergence understated | Yes. 91 of 300 steps above `1e-3`, peak 0.00272, all reported converged | Measured by `check_solver.py`, stated in "Divergence over a longer run" and `docs/LLD.md` |
| Convergence limits recorded only up to `t=1` | Yes, as reported, and it goes further: soft inflow also loses grid order at `t=2.5` and `t=2.75`, inside the planned window | `check_solver.py` runs to `t=3`. Table and limit stated above, and B12 updated |
| Cell-centred divergence has a large floor | Yes. 48.6 against 0.00085 on the faces of one 256×256 state, and 36.5 against 0.00012 for a random field at 64×64 | Docstring and `docs/LLD.md` state it, with the figure for 128×128 targets (see below) |

Found while fixing these, and not in the review:

- **An infinite face was reported as converged.** One infinite face also made the tolerance infinite, and `inf <= inf` passed. `SmokeSolver.project` reported `converged=True`. Now reported as not converged (B16).
- **`test_smoke_does_not_cross_a_thin_wall` skipped a row.** Its "above" region started one row too high, because `y > 0.5 + 1.5/64` excludes the row just above the wall. It now covers everything that is neither wall nor below. The solver passes the stronger form.
- **Floor at training resolution.** The 256×256 state's velocity averaged down to 128×128 has a cell-centred divergence of 65.9 at most and 1.26 RMS, on true data. Task 5's penalty works on this kind of field.

Test runs for these fixes:

| Step | Result |
|---|---|
| New late-obstacle and wall tests against the old solver | Both failed: density in solid 0.0887, and density above the wall 0.9191 |
| New infinite-face test against the old solver | Failed: `converged` was True |
| New `step` tests on CPU against the old solver | 6 failed, each `RuntimeError: index ... out of bounds` |
| New `step` test on GPU against the old solver, run alone so as not to disable the GPU for other tests | Failed, with the device-side assert |
| New vortex pair tests against the old solver | Passed, as expected, since the old solver does advect velocity. They were seen failing on mutants, below |
| After the fixes, `uv run pytest tests/physics/test_solver.py` | 44 passed |
| Full suite, `uv run pytest`, at `2b2df8a` | 71 passed |

Mutants, run against a scratch copy of the fixed solver, each failing the tests named:

| Mutant | Tests that fail |
|---|---|
| `new_u = u` and `new_v = v` | Both vortex pair cases, and nothing else |
| `new_v = v` | Both vortex pair cases, and nothing else |
| `new_u = u` | Both vortex pair cases, the thin-sheet test and the mass budget test |
| Start-of-step mask removed | Late-obstacle test and new wall test |
| Mask `new_density` in place of the faces, the review's other option | New wall test only |
| Lookup index left unguarded, run on CPU cases only, since on GPU it disables the device for the rest of the run | All 6 CPU `step` tests |
| NaN coordinates replaced with 0, so a blown-up item partly heals | All 4 `test_bad_velocity_in_a_moving_flow_never_heals` cases |
| Infinite-speed check removed | Infinite-face test |

The vortex pair covers 75% of the point-vortex distance at 64×64 (0.149 against 0.199). Peak vorticity falls from 85.9 to 19.9 over the same 50 steps, and from 95.5 to 25.4 at 128×128. These are from one-off runs and are not saved. Dissipation is heavy here: the core spans about five cells at 128×128 and turns about half a radian a step, at half the peak vorticity. The share due to each was not separated. This adds to the open item on semi-Lagrangian dissipation.

The solver changes do not affect a run with a fixed obstacle. Every convergence value through `t=1` in `results/solver_check.json` matches its value before the fixes, and `results/figures/solver_check.png` regenerated byte-identical, so its earlier inspection stands. Speed changed little: 19.8 steps per second against 20.3 before at 256×256, batch 1, and 24.7 against 25.0 at 128×128, batch 1. Each is one measurement, so the change is not separated from run-to-run spread.

### Next

Task 3: scenes. Scene inflows should have smooth edges rather than hard ones (B12), and `dt` has to be chosen for generation. The checks here used `dt=0.01`.

## 2026-11-03: Phase 2, Task 3 (scenes)

### What was done

- `src/sandeha/physics/scenes.py`: `Scene`, `sample_scene`, and the six shape families.
- `tests/physics/test_scenes.py`: the plan's five tests, plus further tests listed in `docs/TEST_PLAN.md`.
- `scripts/check_scenes.py`, beyond the plan's file list, writes `results/scene_check.json`, `results/figures/scene_check.png` and `results/figures/scene_check_flow.png`.
- `SPLITS`, the range constants, `MIN_THICKNESS` and `LATE_FRAMES` are public additions to the planned interface. `Scene` and `sample_scene` are as planned.

### Test runs

| Step | Result |
|---|---|
| Tests before the module existed | Collection error: `ImportError: cannot import name 'scenes'` |
| Tests against a naive stand-in with the right interface, 91 tests at that point | 63 failed, 28 passed. Those that passed covered properties the stand-in happened to meet, such as a fixed buoyancy inside the training range |
| First real sampler | 86 passed, 5 failed: all `test_obstacles_are_thick_enough_not_to_be_jumped`. See "Tests changed after a failure" |
| After changing the thickness check | 94 passed |
| Area balance test against the first shape sizes | Failed: unseen families covered 0.419 more area than training ones. Passed after rescaling |
| Plume path test against uniform placement | 5 failed, with shares from 0.74 to 0.82 against a limit of 0.85. Passed after the placement change |
| Final, `uv run pytest tests/physics/test_scenes.py` | 116 passed |
| Final full suite, `uv run pytest` | 187 passed |

### Tests changed after a failure

- **Thickness check.** It first required a disc of the minimum thickness to fit around every solid cell, which no square corner can satisfy, and which failed by 2 to 23 cells per scene. A one-off measurement placed the leftover cells at most 3.0 cells from where the disc fits, in every family, with 1.0 on plain circles from pixelation of the disc. The check now allows 3.5 cells. On that change a check wall of 7 rows no longer counted as thin, which showed that the 1.5-cell rasterisation margin in use guarded only about 7 cells. The margin was reduced to 0.5, so the check now guards about 9 of the 10.24 cells, and all real shapes pass it.
- **Late obstacle share.** The first window, 0.28 to 0.39 over 400 scenes, failed at 0.3975 when the placement change altered the random stream. Over 8 seeds of 1000 scenes each, in a one-off run, the pooled share was 0.3399, 1.25 standard deviations from a third, and the rejection step can only lower it. So the window was too tight, and the sampler is not biased. The test now uses 1000 scenes and four binomial standard deviations. A share of one half still fails it.

### Mutants

Run on scratch copies against the final sampler, excluding the obstacle count test: 37 of 40 killed. The 40 cover split rules, physics ranges, source edges and overlap, clearance, the way-up check, thin geometry, wrong shapes, late obstacles, axes, seeding and array types.

| Survivor | Outcome |
|---|---|
| Buoyancy returned as a NumPy float | Equivalent. `Generator.uniform` already returns a Python float |
| Always one obstacle | Killed by `test_a_scene_starts_with_one_to_three_obstacles`, added for this |
| 12 to 14 obstacles | Killed by the same test. An earlier run had killed it only through the solid fraction limit, so that kill depended on the seed |

An earlier run, before the placement change, had 6 survivors, and these led to new tests. Forcing placement onto sources showed that the way-up check already rejects a covered source cell, so the keep-out area's own effect is the clearance gap, and the test asserts that gap.

### Measured scene properties

From `results/scene_check.json`, generated at commit `d870399` with a clean tree. The sampler has not changed since. Peak GPU memory: 80.2 MiB.

Mean share of the box covered by one shape, over 300 draws:

| Family | Mean | Min | Max |
|---|---|---|---|
| circle | 0.0333 | 0.0112 | 0.0613 |
| rectangle | 0.0310 | 0.0192 | 0.0473 |
| wall | 0.0239 | 0.0125 | 0.0410 |
| star | 0.0331 | 0.0200 | 0.0505 |
| letter | 0.0320 | 0.0146 | 0.0599 |
| blob | 0.0306 | 0.0161 | 0.0486 |

Over 200 scenes per split at 256×256. Solid fraction counts late obstacles:

| Split | Mean solid fraction | Largest | Late share | Sampling time per scene (ms) |
|---|---|---|---|---|
| train | 0.066 | 0.149 | 0.295 | 13 |
| val | 0.065 | 0.160 | 0.375 | 14 |
| test_a | 0.071 | 0.150 | 0.405 | 13 |
| test_b | 0.062 | 0.125 | 0 | 61 |
| test_c | 0.054 | 0.115 | 0 | 12 |

The late shares are from 200 draws, where one standard deviation is about 0.033. `test_a` is about 2.2 standard deviations high. At the planned 30 scenes per split, the realised share can differ noticeably from a third.

Eight scenes per group, run at 256×256 with `dt=0.01` for 480 steps, which is 120 frames at 4 steps each. The two corner groups set physics to the extreme of the named range:

| Group | Largest travel (cells/step) | Largest CG iterations | Largest divergence | Unconverged item-steps | Largest density in solid | Finite | Item steps/s |
|---|---|---|---|---|---|---|---|
| train | 4.10 | 14 | 0.00270 | 0 | 0 | yes | 98.0 |
| test_b | 4.37 | 13 | 0.00286 | 0 | 0 | yes | 95.8 |
| train_top_corner | 6.60 | 13 | 0.00353 | 0 | 0 | yes | 100.6 |
| test_c_low_corner | 1.32 | 12 | 0.00125 | 0 | 0 | yes | 107.9 |
| test_c_high_corner | 7.31 | 10 | 0.00397 | 0 | 0 | yes | 110.0 |

First frame at which smoke above density 0.01 touches a cell next to the initial obstacle:

| Group | Frame per scene (-1 is never) |
|---|---|
| train | 32, 49, 19, 42, 47, 20, 36, 43 |
| test_b | 25, 37, 44, 22, 38, 18, 45, 39 |
| train_top_corner | 36, 34, 19, 19, 29, 19, 36, 32 |
| test_c_low_corner | 102, 32, 82, 63, 93, 83, 84, -1 |
| test_c_high_corner | 17, 21, 20, 21, 10, 23, 15, 11 |

### Visual check

Both figures were inspected at `d870399`. In `scene_check.png`, each split shows the intended families, with sources below the shapes. In `scene_check_flow.png`, plumes rise and bend around obstacles in the groups that reach them, and the two `test_c` corners look clearly different from training: at frame 60 the low corner still shows small caps near the sources, and the high corner fills much of the box.

### Problems

- A check on the first placement, with one seed and eight scenes per group, had 3 of 8 `test_b` scenes touching late or never (B19). That run is not saved. The figures above are from the changed sampler, using the same seeds but different draws, so the two are not a paired comparison.
- The mutant run was started twice in one command by mistake, taking over 10 minutes per pass with up to 5 running at once. No result was affected.

### Open items, for Task 4

- **`dt` is still not chosen.** Everything here used `dt=0.01`. Travel scales with `dt`, and the 7.31 largest travel leaves a margin against the 10.24-cell thinnest wall of 1.4 times, or about `dt=0.014` before the two meet. Largest travel comes from eight scenes per group, so it is not a bound.
- **Thin-wall crossing was not measured directly at 256×256.** Travel below thickness is the criterion. It rests on the Task 2 one-off at 64×64, where a one-cell wall let smoke through at between 3.2 and 6.4 cells per step.
- **Low `test_c` scenes are slow to reach obstacles.** At the low corner, 6 of 8 scenes touch only after frame 70, the end of the planned window, or never. Corner values are the extreme, and a quarter of `test_c` scenes are low on both counts. Geometry has little effect on those frames, so `test_c` results should be read per pairing, and `docs/PREREGISTRATION.md` should say so. The ranges have not been changed because of this.
- **Horizon.** In train and `test_b`, the largest first-touch frame is 49, inside the planned window that ends at frame 70, so these eight-scene groups give no reason to extend the horizon. A late obstacle's own contact was not measured.
- **Largest divergence** reached 0.00397 here, above the plan's `1e-3`, which is set at 64×64. It follows the relative stopping rule in `docs/DECISIONS.md` B13.
- **Generation throughput** was 95.8 to 110 item steps per second at batch 8, with real differing scenes. At the planned 360 scenes of 480 steps, that is an estimate of about 30 minutes of solver time, excluding saving and downsampling. Task 4 still has to measure this.
- **Late obstacle share varies at the planned size.** Task 4 should record the realised share per split in the manifest.

### Not done

- The independent phase check required by `docs/DESIGN.md` section 8 has not been done for Phase 2.
- Phase 2 commits are not yet pushed. The plan pushes at the end of each phase.

### Next

The independent Phase 2 check and push, then Task 4: generation and test-set lock.
