# Sandeha: Decisions

Each major decision, the alternatives and the reason. Status is **Default** for a design default open to change, **Owner** for an owner decision, and **Build** for a decision taken during the build.

## Design decisions

| # | Decision | Alternatives | Reason | Status |
|---|---|---|---|---|
| D1 | Reference solver at 256×256, downsampled to 128×128 for targets | Train on a 128×128 solver | A 128×128 solver is itself fast, so beating it on speed proves nothing. A finer grid gives the model something worthwhile to learn | Default |
| D2 | One neural step advances 4 solver steps | One solver step per neural step | Gives the model a real speed advantage | Default |
| D3 | Buoyancy as a constant input channel | Leave it out | It can't be inferred from one frame, so leaving it out would make it a hidden variable and undermine Test C | Default |
| D4 | Ensemble of 5, with 3 as a recorded fallback | Other uncertainty methods | Simple and well studied, and costs are measured | Default |
| D5 | Every member steps from the shared ensemble mean | Independent member rollouts | Independent members drift apart with time regardless of the flow, so doubt could become a disguised clock. Independent rollouts stay as a secondary comparison | Default |
| D6 | Whole-frame hand-off | Per-region hand-off | Per-region is more elegant but much riskier. It is a stretch goal only | Default |
| D7 | Primary metric: normalised error against the reference over a fixed horizon, starting at 50 neural steps. Secondary: divergence, vorticity spectrum, plume statistics. Fixed before Phase 3 | Point-by-point error only | Smoke is chaotic and the solver resumes from a state that is already off, so point error alone may undersell hand-off at long horizons | Default |
| D8 | Threshold frozen on validation. Fixed and random baselines get the budget it actually used on each test set | Compare best sweep points on test | Choosing on test data would invalidate the headline comparison | Default |
| D9 | Demo served locally through a Python server and WebSockets, not publicly hosted | In-browser inference, public hosting | Simpler, the solver needs the GPU anyway, and it must run with no internet | Default |
| D10 | Project-local Python 3.12 through `uv` | System Python 3.14 | System Python has no PyTorch | Default |
| D11 | Public track: `turbulent_radiative_layer_2D` from The Well, 3 members, doubt only | ScalarFlow, `shear_flow`, `rayleigh_benard` | 7.9 GB, CC-BY-4.0, no sign-up. ScalarFlow needs a sign-up form, is almost half a terabyte and 3D. The others are 471 GB and 370 GB | Default |
| D12 | GitHub repo public. Changed from the default on 2026-11-02 (`docs/DESIGN.md` section 3, commit `8fb4453`) | Private until the owner publishes, which was the default and was meant to prevent copying before submission | Owner's choice | Owner |
| D13 | Weights in a GitHub Release, never in git | Git LFS, committing weights | Keeps the repo small while the demo stays reproducible | Default |
| D14 | Freeze on the morning of 2026-11-04 | Work until the deadline | Leaves a full day to learn the project and rehearse | Default |
| D15 | Separate project with no reuse of earlier project code | Reuse | Owner requirement | Owner |

## Build decisions

### B1: PyTorch 2.15.0 with the CUDA 13.2 build (2026-11-02)

- **Decision:** `torch` from the `cu132` index, marked `explicit` in `pyproject.toml` so that only `torch` resolves from it.
- **Alternatives:** the `cu130`, `cu129` or `cu128` builds.
- **Reason:** `nvidia-smi` reports driver 595.99.02 with CUDA 13.2, so `cu132` matches. Without `explicit`, other packages resolved from the PyTorch index and got older versions, for example `packaging` 24.1 and `setuptools` 78.1.0.

### B2: `.gitignore` entries anchored to the repo root (2026-11-02). Changes the plan

- **Decision:** `/data/`, `/runs/`, `/checkpoints/` in place of `data/`, `runs/`, `checkpoints/`. `docs/PLAN.md` is updated to match.
- **Alternative:** the plan's unanchored patterns.
- **Reason:** an unanchored `data/` matches any directory of that name. `git check-ignore` confirmed that it ignores `src/sandeha/data/store.py`, which would have silently left the data package out of the repo. `tests/test_repo.py` guards against this.

### B3: Local tooling paths excluded outside tracked files (2026-11-02). Changes the plan

- **Decision:** local tooling paths are listed in `.git/info/exclude`, not in `.gitignore`. `docs/PLAN.md` is updated to match. A fresh clone needs the same local entries added if those tools are used there.
- **Alternative:** the plan's `.gitignore` entries.
- **Reason:** `.gitignore` is tracked, and tracked files must not name tooling.

### B4: `commit-msg` hook kept in `.git/hooks/` only (2026-11-02)

- **Decision:** the hook is installed locally and is not tracked.
- **Alternative:** a tracked copy under `scripts/`.
- **Reason:** the rejected patterns name specific tools, so a tracked copy would break the same rule as B3. The cost is that a fresh clone has no hook until one is installed. The audit in Task 13 checks the log directly, so that check does not depend on the hook.

### B5: Section 8 of `docs/DESIGN.md` reworded (2026-11-02). Changes the design text

- **Decision:** four lines describing how the build is run were reworded before the first commit: the build approach line, the phase check rule, the results rule and the interruption rule. Requirements are unchanged: each phase still gets an independent check, results still come from the runs, and the build log must still allow a restart.
- **Alternative:** keep the design untracked.
- **Reason:** the design itself requires that documents make no statement about how the code was authored, and it is a tracked deliverable.

### B6: Wrapper refuses when mains state is unknown (2026-11-02)

- **Decision:** `long_job.sh` exits 3 unless a supply of type `Mains` reports `online=1`. Power through a USB-C supply does not count.
- **Alternative:** refuse only on an explicit offline reading, or also accept online `USB` supplies.
- **Reason:** starting a long job on battery is the costly mistake, and this laptop reports its adapter as `ACAD`, type `Mains`. If the laptop is ever charged over USB-C, the wrapper refuses and says so, and this rule gets revisited.

### B7: Tests added beyond the plan's file list (2026-11-02)

- **Decision:** `tests/test_long_job.py` and `tests/test_repo.py` were added.
- **Reason:** the plan requires test-first development and names only `tests/test_env.py` for this task, which leaves the wrapper and ignore rules uncovered.

### B8: Wrapper stops the job on a signal and reports its real status (2026-11-02)

- **Decision:** `long_job.sh` forwards SIGTERM, SIGINT and SIGHUP to the job's process group and waits for it to stop. A helper under the inhibitor records the job's status, so a job killed by signal n reports 128+n. A job that reports no status gives wrapper exit 4 and `exit=unknown`. The name `jobs` is reserved. The job's stdin is `/dev/null`.
- **Alternatives:** document that the job survives a signal to the wrapper, and that signal deaths log as exit 1.
- **Reason:** with the first version, the job kept running after SIGTERM to the wrapper, with no `END` line, so it looked dead. Restarting on that reading could put two jobs on the same checkpoints and GPU, against the design's "safe to restart" rule. An out-of-memory kill also logged the same code as an ordinary failure. The cost is about 25 more lines of shell. SIGKILL cannot be forwarded, so that case is documented with a check to run before restarting.

### B9: Wider attribution patterns in the local `commit-msg` hook (2026-11-02)

- **Decision:** the hook also rejects "assisted by" and "assisted with" phrasing and more tool names.
- **Alternative:** keep the first pattern list.
- **Reason:** a message of the form "Assisted by" plus a tool name not on the list exited 0. One tool name is matched only through such phrasing, because it is also an ordinary word for a demo feature. Under B4, the hook stays untracked.

### B10: Semi-Lagrangian advection, with MacCormack rejected (2026-11-03)

- **Decision:** semi-Lagrangian advection with a midpoint backtrace, for density and both velocity components.
- **Alternative:** MacCormack, which `docs/PLAN.md` also allows and which was built first, with a limiter that either reverts to the semi-Lagrangian value or clamps.
- **Reason:** MacCormack produced cell-scale stair-steps along thin sheets at this time step, which would have become noise in the training targets, and it lost more smoke against the mass budget. `test_thin_sheets_have_no_cell_scale_artefacts` and `test_smoke_matches_the_mass_budget` failed on it. The cost is more numerical dissipation, which is recorded as an open limit in `BUILD_LOG.md`. A smaller time step might also remove the artefact, but was not tried, because it would multiply the data generation cost.

### B11: Multigrid preconditioner (2026-11-03)

- **Decision:** one symmetric V-cycle per conjugate gradient iteration, with red-black smoothing. Coarse levels merge 2×2 blocks and sum the links they replace, so obstacles of any shape are handled exactly. Coarse corrections are scaled up by 1.9, the usual remedy for piecewise-constant transfer.
- **Alternatives:** Jacobi, which barely helps on this nearly constant diagonal, and MIC(0), which is sequential and suits a GPU badly.
- **Reason:** iteration counts stay near 8 from 64×64 to 256×256, as `results/solver_check.json` shows, and a test guards this. A test also confirms the preconditioner is symmetric, as conjugate gradients requires.

### B12: Convergence test uses a smooth inflow to `t=1`, and a hard inflow to `t=0.5` (2026-11-03). Changed after a test failed

- **Decision:** `test_converges_with_refinement` runs two cases, and requires the 128×128 error to be below 0.8 of the 64×64 error, which is stricter than the plan's "smaller than".
- **Alternative:** the first version of the test, a hard-edged disc inflow run to `t=1`, which failed.
- **Reason:** a table of errors over time pinned down the failure. A hard-edged disc is a different source on each grid, and from about `t=0.75` to `t=1`, depending on the advection scheme, its plume loses pointwise convergence. With a smooth source, the same solver converges through the whole window. The test was changed after seeing a failure, so the full table including the non-converging case is kept in `results/solver_check.json` and in `BUILD_LOG.md`. Task 3 should use smooth inflow edges for the same reason.
- **Limit, added after review:** this decision covers the window the test asserts, not the window the project uses. Run to `t=3`, the hard inflow is reversed at `t=1` and `t=1.25`. The soft inflow narrows to a ratio of 0.877 at `t=2`, and is reversed at `t=2.5` and `t=2.75`. The planned primary horizon, 50 neural steps from frame 20, spans `t=0.8` to `t=2.8` at `dt=0.01`. The test is unchanged, since asserting grid order past these points would be asserting something false. Treat the 256×256 run past about `t=1` as one realisation of the flow, not as pointwise truth. `docs/PREREGISTRATION.md` (Task 4) has to allow for this, and the design's physical secondary metrics matter more as a result.

### B13: Pressure solve stops on true divergence relative to flow speed (2026-11-03)

- **Decision:** `cg_tol` means that no cell's net flux may exceed `cg_tol` times the fastest face of that batch item. Each pass re-measures this from the faces, and solves for what is left. An item stops when it reaches the target, when a pass fails to halve its divergence, or at `cg_max_iter`. `SmokeSolver.last_solve` reports iterations, divergence and convergence for each item.
- **Alternatives:** a residual relative to the right-hand side, inside one run of conjugate gradients, or double precision.
- **Reason:** a relative right-hand side lets smooth fields stop with too much divergence relative to their speed, and the recursive residual can drift from the true value in float32. Repeated correction reaches the target in float32. Double precision was not measured.

### B14: Additions to the planned solver interface (2026-11-03)

- **Decision:**
  - `SmokeState` gains an optional `mac` field carrying the face velocities between steps.
  - `SmokeSolver` gains `last_solve` and `preconditioner()`.
  - `DENSITY_DECAY` is a module constant, set to 0.1 per unit time.
  - Rows grow upward, so buoyancy acts along +row and plots need `origin="lower"`.
  - `scripts/check_solver.py` and `results/solver_check.json` were added beyond the plan's file list.
  - Batch 16 was added to the speed measurement.
- **Alternatives:** rebuild the faces from the cell-centred view at each step, and keep the measurements as one-off commands.
- **Reason:** averaging to cell centres and back each step would blur the flow. Saved output and one regenerating command are required by `docs/DESIGN.md` section 6.5. The planned arguments are unchanged.

### B15: Faces masked at the start of `step` (2026-11-03). Reverses an earlier removal

- **Decision:** `step` masks the incoming faces with the current obstacle before advecting anything.
- **Alternatives:** multiply the advected density by the fluid mask, which fixes density in solid cells but still carries smoke across a new wall (`test_new_wall_stops_smoke_on_its_first_step` fails on it). Or project at the start of the step, which also removes the divergence the mask creates, at the cost of a second pressure solve on steps where nothing changed, or of tracking obstacle changes.
- **Reason:** this mask was removed during the build as having no observable effect, which was wrong. It has no effect only while the obstacle is unchanged. Without it, the first step after a change put density inside the new solid, on every `late_obstacle` scene and every demo edit reaching the solver. The rule that density in solid cells is 0 failed on that step. With masked faces, a solid cell traces back to itself, so it stays empty with no separate clearing. The cost: on that one step, advection uses a field with some divergence next to the new solid. The pressure solve at the end of the step removes it.

### B16: A blown-up batch item is carried and reported, not raised or repaired (2026-11-03)

- **Decision:** `_interpolate` keeps its indices in range for any coordinate, and a NaN coordinate yields NaN. A batch item with non-finite input stays non-finite, is reported with `converged=False`, and does not affect the other items. `_project_faces` no longer accepts an item with an infinite face as converged.
- **Alternatives:** replace bad coordinates with 0, the plain reading of "sanitise", which lets a bad velocity stop reaching density (`test_bad_velocity_in_a_moving_flow_never_heals` fails on it). Or raise an error, which would lose a whole generation batch and leaves evaluation and demo to catch it.
- **Reason:** review focus 1 in `docs/PLAN.md` requires a blown-up rollout to count as a failure. A crash takes down the process, and on a GPU disables the device until the process restarts. A repair could make a failure look like a finite result. Infinite coordinates are clamped to the edge, like any large value, and the infinite value still reaches the output through the lookups around it. Callers must check `last_solve.converged` or the fields, as they already had to.

### B17: Each unfamiliar test set changes one thing (2026-11-03)

- **Decision:**
  - `test_b` uses unseen shapes with training physics.
  - `test_c` uses training shapes, with buoyancy and inflow strength both outside the training ranges, with a gap between the ranges. Each fell low or high independently until B20, which ties them.
  - Neither has late obstacles.
  - One inflow strength applies per scene, and the inflow's peak is exactly that value.
- **Alternatives:** a late obstacle in `test_b` and `test_c`, or only one physics quantity out of range per `test_c` scene.
- **Reason:** a rise in doubt on `test_b` or `test_c` should have one explanation. The plan requires late obstacles only in `train`, `val` and `test_a`, and leaves the other two open.
- **Limit:** the values are chosen around the solver checks' buoyancy of 2 and strength of 5. The independent pairing described here was replaced by B20.

### B18: Limits on scene geometry (2026-11-03)

- **Decision:**
  - `MIN_THICKNESS` is 0.04, or 10.24 cells at 256×256.
  - Stars and blobs are polygons grown by half of that, which also rounds their tips.
  - Letters are thick strokes, and only letters with no closed counter are used.
  - Solids keep 0.02 clear of sources and box walls.
  - A scene that seals a source away from the top row is redrawn whole.
  - Source edges follow a raised cosine, which has finite support, rather than a Gaussian.
- **Alternatives:** thin walls limited only by flow speed, a Gaussian source as in the solver checks, or letters such as A and O.
- **Reason:** semi-Lagrangian advection can carry smoke across a wall thinner than one step's travel. A Gaussian has no edge, so "inflow cells are not solid" could not hold exactly. A closed counter is a pocket that smoke can never enter.

### B19: Shape sizes balanced, and the first obstacle placed in a plume's path. Both changed after measuring (2026-11-03)

- **Decision:** star, letter and blob sizes were reduced so that unseen families cover the same mean area as training families. The first obstacle's centre is drawn within 0.08 of a source along x, narrowed to 0.06 by B21. Others stay uniform across the placement range.
- **Alternatives:** the first sizes, and uniform placement of every obstacle.
- **Reason:** unseen families first covered about 42% more area than training ones, so doubt on `test_b` could have risen because more of the box was solid. With uniform placement, smoke in 3 of 8 `test_b` check scenes touched an obstacle only after frame 70, the end of the planned window, or never. Such a scene shows the model nothing of the unseen geometry, which would understate doubt on the set built to measure it. Both changes use geometry alone. No data or model existed at the time.
- **Cost:** with the first obstacle always near a source, scenes where the plume rises freely past all shapes are rarer in every split, including training.

### B20: `test_c` ties buoyancy and inflow strength to the same side (2026-11-03). Changes B17, the plan and the design text

- **Decision:** a `test_c` scene has both quantities low (buoyancy 0.4 to 0.8, strength 1.5 to 2.5) or both high (3.5 to 4.5, 8 to 10), each side with equal chance. A test asserts that their product is outside the training range. `docs/PREREGISTRATION.md` reports `test_c` per side as well as pooled.
- **Alternatives:** keep all four pairings, label the two mixed pairings as amplitude-only, and report per pairing. Or keep independent sides and reject scenes whose product falls in the training range, which shrinks the mixed pairings to their extremes.
- **Reason:** the solver's equations depend on one flow parameter. With density rescaled by `1/strength`, velocity feels `buoyancy * strength` and density advection is linear, so `strength` affects only density's amplitude. In `results/scene_check.json`, buoyancy 0.6 with strength 9 against buoyancy 1.8 with strength 3 differs in velocity by `1.8e-06` at 256×256, and a different product differs by 1.08. The mixed pairings had products from 3.2 to 8 and 5.25 to 11.25, inside the training range of 3 to 21. About half of the old `test_c` was therefore a training flow with rescaled density, against B17's aim and the design's "for physics rather than geometry". The change came before the test set was locked, and no data or model existed. With about 30 scenes planned, four pairings would leave about 7 scenes each, and two of the four would not be new flows. Tying leaves about 15 per side, all with new flows.
- **Limits:**
  - Input amplitude and flow regime leave their ranges together, so a rise in doubt on `test_c` can't be split between them. The set no longer has an amplitude-only group that could separate them. B17's "one explanation" holds for physics against geometry, not within physics.
  - Scenes differ in strength within the training range without any change in flow beyond the product, so training sees one flow parameter, with a separate amplitude, rather than two independent quantities.
  - Train products are sparse near both ends of 3 to 21, so the distance from `test_c` to well-covered training flows is larger than the gaps from 2 to 3 and from 21 to 28 suggest.

### B21: In-path reach narrowed from 0.08 to 0.06, and the check made exact (2026-11-03). Changed after review

- **Decision:** `_IN_PATH_REACH` is 0.06. The test requires every scene, not more than 85%, to have an obstacle above a source, and a second test checks the constant against the narrowest case.
- **Alternatives:** keep 0.08 and state that a rare miss is possible. Or reject poses that miss, which would keep the rule true for any reach and so hide a widened reach from the test.
- **Reason:** the 0.85 limit only separated the current placement from uniform placement, so a partial regression could pass. At 0.08 the measured share was 1.0, but a miss was still possible: an upright wall of half thickness 0.02 over a source of radius 0.05 reaches 0.07 from the centre. At 0.06 at least 0.01 of overlap remains, or 2 columns at 256×256.
- **Limit:** this is about columns. Smoke may still bend away, or arrive late: see the first-touch table in `BUILD_LOG.md`. The overlap can be narrow. Coverage of a source's full width is not guaranteed.

### B22: A placement draws its family once (2026-11-03). Changed after review

- **Decision:** `_place_shape` draws the family before the pose loop, so families are equally likely among placements. Each family is drawn with equal chance regardless of how often its poses fail.
- **Alternative:** keep the redraw and record the skew.
- **Reason:** redrawing on each rejected pose favoured families that fit easily, and made walls, the smallest and most directional family, the rarest. The skew served no purpose and no document stated it.
- **Cost:** a family that fits poorly could exhaust its 200 attempts and raise an error, where before another family could have taken its place. In a one-off run at the edge of the placement range, the lowest acceptance was 0.2175, for walls. That gives a chance of about `5e-22` of exhausting the attempts. Over 13000 scenes, no error occurred.
