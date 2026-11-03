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

### B23: A late obstacle applies from its frame onward, on the step out of that frame (2026-11-03)

- **Decision:** the solid in force from frame `f` to `f+1` includes the late obstacle once `f >= late_frame`. Frame `late_frame` is therefore made under the old solid and is zero only there, and it may hold smoke inside the new solid. The mask in the model's input at that frame is the new one.
- **Alternatives:** apply the obstacle on the way into `late_frame`, so that frame is already clean. The mask in the input one frame earlier would then not show the obstacle, and the target would change for a reason the model can't see. Or clear smoke from the new solid in the stored frame, so that training never sees smoke under a mask.
- **Reason:** this is what the demo and `HybridSimulator.set_obstacle` do when a person draws over smoke: the input has the new mask and the old smoke. It keeps one rule for every transition: a target is zero under the mask in its input, which is what `apply_step` forces. A test compares stored frames with a separate solver loop to the byte.

### B24: A coarse cell is solid if any of its fine cells is (2026-11-03)

- **Decision:** the 128×128 mask is the any-of-four of the 256×256 mask. Fields are 2×2 averages, then zeroed in coarse solid cells. Inflow is a 2×2 average, so its coarse peak can be a little below the scene's strength.
- **Alternatives:** a cell is solid only if most or all of its fine cells are. The averages in partly solid fluid cells would then mix in zeros, and a fluid cell could hold values from next to a wall.
- **Reason:** coarse fluid cells then hold only fluid values, and no wall opens up at the coarse grid. The cost is that solids grow by up to half a coarse cell per side, in the mask only. Fine masks and inflow are kept as well, so that Task 8's hand-off can use the true solid.

### B25: `dt=0.01` for generation (2026-11-03)

- **Decision:** the value all solver and scene checks used.
- **Alternatives:** a larger `dt` for less work, or a smaller one for a margin.
- **Reason:** a larger value gives up margin against the 10.24-cell thinnest wall, and a smaller one multiplies the cost and invalidates the checks.
- **Corrected after review.** This entry first gave a largest travel of 7.32 cells per step, from eight scenes per group, and so a limit near `dt=0.014`, or about 40% of margin. That margin was wrong for late-obstacle scenes. In `results/travel_check.json`, over 62 ordinary late-obstacle train scenes at a seed outside the dataset, scene index 7 reached 9.97 cells per step in the frame out of its `late_frame` of 90, a margin of about 2.6%. Its largest outside the event was 6.24. It was the only such spike: the next largest was 5.64, none passed 10.24, and all solves converged. At that scene's margin, `dt=0.01` is about the limit, not a value well inside it. With about 100 late-obstacle scenes planned in train alone, and more in `val` and `test_a`, a scene above the wall is possible. `dt` is kept because a smaller value would double the cost, and in this scene travel was back to at most 6.24 outside the 3 frames from the event. Whether a spike above the wall leaks smoke has not been measured. See B35.
- This entry also said each scene's largest travel was in the manifest. It held only each split's largest. Each scene's figure and frame are there now.

### B26: What stops a generation run, and what is only recorded (2026-11-03)

- **Decision:** non-finite data as stored stops the run, with nothing from that batch stored. A dirty tree stops it before it starts. An unconverged solve is counted in the manifest and doesn't stop it.
- **Alternatives:** redraw a bad scene, which would drop hard scenes from the set without any record. Or stop on any unconverged solve, which could end an overnight run over one solve at the rounding limit.
- **Reason:** a bad scene must be looked at, not replaced. The check is made on the `float16` form, since a value too large for it only turns infinite on storing. The Task 3 check groups had no unconverged solves. The spot-check has to confirm the count.

### B27: Additions to the planned data interface (2026-11-03)

- **Decision:**
  - `src/sandeha/data/generate.py` holds generation, so that tests can import it, and `scripts/generate_data.py` is a thin command.
  - `Normalizer` also has `encode_delta`, `decode_delta` and `make_input`, with a separate scale for the change over one step.
  - Items also have `obstacle_seq`, `scene` and `frame`.
  - `SplitStore` also has `fine_scene`, and `configs/data.yaml` stores reference-grid states at frame 20.
  - `results/data_manifest.json` is a tracked copy of the manifest. `scripts/estimate_generation.py` and `results/generation_estimate.json` are new. `pyyaml` is a new dependency, for the planned YAML config.
- **Alternatives:** one scale for fields and changes. Or unroll windows that skip over a late obstacle.
- **Reason:** the plan asks for a normalised delta, and one step's change is much smaller than a field's spread, so one scale would give very small targets. Without `obstacle_seq`, an unroll crossing a late obstacle would use a stale mask, and skipping those windows would hide the event from training. `data/` isn't tracked, so without the copy the checksums of the locked sets wouldn't be on record. Regenerating for frame 20 states later would replace the locked sets, and storing them is cheap by comparison.

### B28: Seeds per scene, fixed batches, and resuming only on the same commit (2026-11-03)

- **Decision:** a scene's seed comes from the master seed, split and index only. Batches are fixed index ranges, and a part-finished batch reruns whole. Resuming needs the same config and commit, and finished data is never regenerated.
- **Alternatives:** one stream per split, which rerolls scenes when a split is resized. Or rerun only the missing scenes in a smaller batch.
- **Reason:** the plan allows scaling sizes down, and that mustn't reroll the test sets. `test_batch_matches_single_runs` shows batched items agree with single runs within `1e-4`, not to the byte, so byte-identical resuming needs identical batches. **Cost:** a commit made between stopping and resuming a run blocks the resume, including one that changes only documents.

### B29: Exact figures in the preregistration (2026-11-03)

- **Decision:** `docs/PREREGISTRATION.md` fixes a blow-up cap of 10, a floor of 1% of the training standard deviation on the error's denominator, a bad frame as above validation's 80th percentile, within-time AUROC only with at least 5 bad and 5 good frames, 5 random-baseline seeds, and a budget match within 1 percentage point.
- **Alternatives:** other values were not measured. No model exists, so none could be, and that is the point of fixing them now.
- **Reason:** `docs/PLAN.md` names these rules but not their values. They are the build's choices and are open to the owner's change until a model exists. The horizon stays at 50 steps, and `LATE_FRAMES` stays at 20 to 90, with obstacles appearing after the window treated as stated there.

### B30: The data spot-check opens test splits (2026-11-03). An exception to the plan's constraint

- **Decision:** the spot-check in Task 4 opens test splits with `final_eval=True` to render a sample scene from each and check stats, stated as an exception in `docs/PREREGISTRATION.md`.
- **Alternatives:** rely on the statistics in the manifest for those splits, and render only train and val.
- **Reason:** the plan requires both "test splits are read only by final evaluation" and "render a sample scene from each split". No model exists at that point and nothing gets tuned from it, and an unrendered defect in a test set would cost more. The manifest covers most of the same statistics without opening the splits.

### B31: Which rollouts and frames count as evidence (2026-11-03). Changed after review

- **Decision:** the within-time Spearman, AUROC, calibration and risk-coverage use ensemble-mean rollouts with no hand-off, at frames 21 to 70. Frames 21 to 120 and independent rollouts are reported as secondary and don't count as evidence.
- **Alternatives:** hand-off rollouts, or frames 21 to 120.
- **Reason:** the first version named no rollouts or frames for the evidence measure, so a result could have been chosen after being seen. Solver frames have no doubt score, and hand-off rollouts depend on `tau`, which is chosen later. Frames 21 to 70 match the primary metric.
- **Cost:** no evidence figure covers what doubt does once hand-offs have run, or after frame 70.

### B32: Non-finite doubt ranks highest, and triggers a hand-off under every policy (2026-11-03). Changed after review

- **Decision:** a non-finite score counts as positive infinity in every ranking. It triggers a hand-off in doubt, fixed and random policies alike, starting from the frame before.
- **Alternatives:** drop those frames, which flatters results and goes against the plan's first review focus. Or trigger on doubt only, and let baselines blow up.
- **Reason:** NaN has no rank, and `doubt > tau` is false for NaN, so a blown-up rollout would never have handed off. Giving the rescue only to the doubt policy would credit doubt with a check any system could make. Rule written from the text: Tasks 8 and 9 don't exist yet.
- **Cost:** blown-up frames add agreeing high ranks, which raises the correlation. Their number is reported beside each figure for this reason. Task 8's plan is more exact now: `Policy` needs the forced hand-off and `FixedSchedule` a per-scene offset.

### B33: Hand-off selection in fixed steps, with a candidate grid and no fallback (2026-11-03). Changed after review

- **Decision:** the evidence measure narrows aggregation, comparing each with the leader by a paired bootstrap interval. Then a run for each aggregation left in, `handoff_frames` from `{2, 5, 10}`, and `tau` at 8 percentiles of validation doubt, plus a **never** candidate, is judged by primary error at a share up to 25%. Exact ties have a fixed order.
- **Alternatives:** the first version: first by Spearman, with ties as overlapping intervals. Or a finer grid, or a `tau` search per run.
- **Reason:** the first version could not be applied. Spearman on no-hand-off rollouts can't vary with `handoff_frames`, so all three tied. Overlap isn't transitive, and no tie-break covered that case. The fallback for no eligible `tau` contradicted itself, and no grid was given, so the result depended on an unstated choice. With **never** in the grid, an eligible choice always exists, so the fallback is gone.
- **Cost:** `handoff_frames` is now chosen by error alone. Percentiles come from no-hand-off doubt, and doubt falls after a hand-off, so a percentile doesn't give a known share, and the grid may not get near 25%. At worst this means 72 validation rollout sets, 3 aggregations by 3 values by 8 candidates. No model exists, so the time for this isn't measured, and Task 9 must measure it against the evaluation budget.

### B34: Baselines' schedules, unmatched budgets and what "beats" means (2026-11-03). Changed after review

- **Decision:** schedules are worked out from the budget before any rollout runs, and per-side on `test_c`. The fixed schedule offsets each scene. If a baseline can't land within 1 percentage point, the schedules either side of the budget both run, flagged. Doubt beats a baseline only with a paired interval wholly below zero, against a baseline at or above the budget.
- **Alternatives:** a tolerance widened to fit, which would move with the result. Or dropping an unmatched comparison, which would hide it. Or one budget on `test_c` for both sides, which would compare the sides at unequal shares.
- **Reason:** the first version said each baseline "must land" within 1 point with no rule for when it couldn't. Random with 10-frame blocks and fixed schedules at high budgets are coarse. Forced hand-offs can only add solver frames, so a baseline they push out stays above budget, which is the conservative side. "Beats" had no definition, and the claim rule needed one.

### B35: Each scene's travel in the manifest, and a spot-check rule above the wall (2026-11-03). Changed after review

- **Decision:** the manifest holds each scene's largest travel and its frame, and names the scenes above the thinnest wall. Named scenes are restarted from that frame at `dt=0.01` and `dt=0.005`, and pass at up to twice the largest figure from 3 low-travel controls. A failure means regenerating at `dt=0.005`. Generation doesn't stop for this.
- **Alternatives:** compare whole runs at the two step sizes. Or treat any scene above the wall as a defect.
- **Reason:** nothing compared the recorded travel with the limit. Whole runs at the two step sizes, tried on the sample in a scratch run with no saved result, were already far apart before the spike, so chaos would hide any effect of it. Restarting from one state leaves that out. In `results/travel_check.json`, the 9.97 scene has a figure of 0.093, and the largest of 3 controls is 0.070. Regenerating over any one scene would double the data's cost for what may be a short event with no effect.
- **Limit:** no scene above the wall has been measured, and the factor of 2 is a choice. A whole-field figure can miss a small local leak.

### B36: A torn timing line can't stop generation (2026-11-03). Changed after review

- **Decision:** `_record_timing` syncs each line. A resume cuts anything after the last full line, and the summary skips and counts any line it can't read, as `timing.unreadable_lines`. A missing file gives empty timing.
- **Alternatives:** write each timing entry atomically to its own file, or stop and ask for a hand edit.
- **Reason:** timing isn't part of the data, and the design requires a long job to be safe to restart. A torn last line raised `JSONDecodeError` after all simulation and joining, on every resume. Without the cut, the next entry would have joined the torn one and been lost too.
- **Cost:** a lost line understates that batch's time in the manifest's totals. `by_split` scene counts then fall short of the split's size.

### B37: The travel rule runs on any named scene, from the manifest alone (2026-11-03). Changed after review

- **Decision:** `src/sandeha/data/travel.py` holds the restart and the rule. `scripts/check_travel.py --data-root data` applies it to a finished dataset, reading only the manifest and rerunning scenes from their seeds. Controls exclude named scenes. A split with no control, or a figure that isn't a number, doesn't pass. A restart frame within 5 frames of the scene's end moves back.
- **Alternatives:** a separate script, or leaving the restart in the script and having tests load it by path.
- **Reason:** the preregistration pointed to a tool that raised `TypeError` on any scene with no late obstacle, which is every `test_b` and `test_c` scene, and it accepted only its own seed and late scenes. No command could apply the rule. `max` in Python can pass over a NaN, which a new test showed, so a NaN figure would have read as a pass.
- **Cost:** the manifest's tracked copy is uncommitted during the spot-check, so this mode runs on such a tree and records `git_dirty` instead of refusing.

### B38: The factor of 2 hasn't been judged against its own controls (2026-11-03). Open

- **Decision:** the factor stays at 2 for now. `scripts/check_travel.py --control-sample` measures the rule's controls on 16 scenes per split, of every kind, at a seed outside the dataset, and its figures are logged before generation.
- **Alternatives:** limit controls to late-obstacle scenes, which `test_b` and `test_c` don't have. Or keep the sample and say nothing, which leaves the mismatch hidden.
- **Reason:** B35's figures of 0.034 to 0.070 come from late-obstacle scenes only. That sample's run is at the planned grid, so it was left out of this tiny-data work and hasn't been run. The code path is tested at 64×64.

### B39: Travel is a distance (2026-11-03). Changed after review

- **Decision:** `travel_cells` takes each cell's faster `u` face and faster `v` face, and combines them as a speed. Generation and `scripts/check_travel.py` use it. The manifest's field names are unchanged.
- **Alternatives:** the cell-centred speed, which averages faces and can read low at a spike. Or `sqrt(2)` times the largest component, which is simpler and always at the limit of the overstatement.
- **Reason:** the largest single component was compared with a wall's thickness, which is a distance in any direction, so a diagonal flow could read up to `sqrt(2)` low. An upper bound errs towards naming a scene, and naming one only costs a comparison.
- **Cost:** figures saved before this change use the old measure and read lower: `results/travel_check.json` (including the 9.97 scene, B35) and `results/generation_estimate.json`. They weren't rerun. The 9.97 scene had a 2.6% margin on the old measure, so on this one it may be above the wall, as could real scenes that would have gone unnamed. The stored fields don't change.

### B40: A coarse contact rule, and an oracle rule, before any model (2026-11-03). Changed after review

- **Decision:** `test_c`'s no-contact count uses a stated rule on the stored reference, not the check groups' measurement. The oracle's budget is the baselines' budget on each set, its `tau` comes from a fixed search, and a non-finite output triggers it at any `tau`. Its figures are descriptive.
- **Alternatives:** storing every fine frame to use the fine rule, which multiplies the data's size. For the oracle, a `tau` frozen on validation, whose share on a test set wouldn't match the budget, so it wouldn't bound anything at that budget.
- **Reason:** the two measurements differ in grid, mask and ring, and the preregistration called them the same rule. The oracle had a `tau` in `docs/PLAN.md` and a bound in Task 9 with no rule at all, so both could have been chosen after results were seen.
- **Cost:** the oracle's `tau` is set on a test set's reference, the one exception to thresholds coming from validation. It chooses nothing else. The search can add 9 rollout sets per set or side, which isn't measured. No model exists, so these rules were written from text alone.

### B41: The manifest's tracked copy and the normaliser's checksum are checked (2026-11-03). Changed after review

- **Decision:** `read_manifest` requires `data/manifest.json` to match `results/data_manifest.json`, and is what `open_split` and `Normalizer.load` read. `Normalizer.load` compares a normaliser beside a manifest with the checksum there.
- **Alternatives:** an argument naming the copy, which every caller must remember. Or comparing with the copy in the last commit, which would block the spot-check, since the copy is committed after it.
- **Reason:** a split replaced with its manifest passed the lock, and an edited normaliser went unnoticed. **Limits:** only the default root `data/` is compared with the copy, so `--final` must use it. The comparison reads the working file, so replacing all of them together passes this check, and shows only as a change to a tracked file and in `git_dirty`.

### B42: Generation enforces the controls' measuring (2026-11-03). Changed after review

- **Decision:** `scripts/generate_data.py` calls `check_controls` before it starts. It needs `results/travel_controls.json` from committed code, with a finite figure per split and the same grid, step, frames and solver settings as the config. `--allow-dirty` skips this only outside `data/`.
- **Alternatives:** a check inside `generate()`, which would also bind the estimate and every tiny test that never needs it. Or leaving the gate as a step in the plan.
- **Reason:** B38's order rested on memory alone, and generation could start without it.
- **Limits:** the file isn't compared with the current code's commit, and nothing checks that its figures were read. Calling `generate()` directly doesn't meet the gate.

### B43: The spot-check has a command, and the rule's step 4 has its rendering (2026-11-03). Changed after review

- **Decision:** `scripts/spot_check.py` checks every split and draws each split's scene 0 and first late-obstacle scene, chosen in advance. `scripts/check_travel.py --data-root` draws each named scene at both step sizes, with their difference, reproduced from its seed.
- **Alternatives:** drawing named scenes from stored fields, which the rule reads none of, and which hold only one step size. Or choosing scenes to draw after looking at them.
- **Reason:** step 4 required a rendering that nothing wrote, and the plan's spot-check had no tool.

### B44: `test_c`'s controls may be one side. Rule unchanged, stated, and measured per side (2026-11-03). Open

- **Decision:** the rule's controls stay the lowest travel in the split. Every `test_c` figure states its side, and `--control-sample` also measures each side's lowest-travel controls.
- **Alternatives:** drawing `test_c`'s controls from the named scene's side, which is a new rule made from reasoning alone, and which leaves a side with fewer than 3 scenes short of controls. Or saying nothing.
- **Reason:** low-side controls against a high-side named scene could fail on flow speed alone, and a failure regenerates every split. No figure for this exists yet. The control sample gives one before generation, as with B38.

### B45: The largest sampled scene is above the wall on the current measure (2026-11-03). Measured after review

- **Decision:** no rule changes. The rule was written to cover this. `results/travel_recheck.json` is added, from `scripts/check_travel.py --recheck-largest`.
- **Alternatives:** rerunning all 62 scenes of `results/travel_check.json`, about 5 minutes of the planned grid, left for later since the sample below gives ordinary scenes on the current measure.
- **Reason:** B39 said this scene might be above the wall. It is, at 12.61 cells against 10.24. It was run alone, not in its first batch of 16.
- **Cost:** `results/travel_check.json` stays on the old measure, and is labelled as such in `docs/BUILD_LOG.md`.

### B46: The divergence penalty is on the predicted change, in physical proportion, away from walls (2026-11-03)

- **Decision:** `step_loss` penalises the divergence of the predicted normalised change in `u` and `v`, after undoing their different scales, in grid units. Only cells whose whole stencil is fluid count. `lambda_div` stays 0.1, as planned, and is not yet judged against its floor.
- **Alternatives:** the penalty on the next velocity, which needs the current state in `step_loss`, and pushes each change to cancel the floor already in the field. Or normalised `u` and `v` as they are, where a divergence-free change doesn't score zero.
- **Reason:** `step_loss`'s planned inputs are the change, and the largest cell-centred values are expected at walls (`docs/LLD.md`), where a difference crosses a solid.

### B47: Items hold 4 frames, so the last 3 start frames go unused (2026-11-03)

- **Decision:** the plan's "1 to 3 steps with no gradient, and only the last step gets gradients" is up to 4 steps, and training uses `StepDataset(unroll=4)`.
- **Alternatives:** clamping the pushed depth by frame, which needs a dataset change.
- **Reason:** simple, and frames 118 to 120 are still seen as targets.

### B48: `max_minutes` is spent from a measurement inside the run (2026-11-03)

- **Decision:** the run's first steps give the speed, with one timed validation and timed no-gradient passes. `total_steps` follows, counting pushed passes and validation, and is saved. `scripts/measure_throughput.py` is separate, is for choosing the batch size, and runs when the GPU is free.
- **Alternatives:** a total fixed from the separate measurement, which goes stale if the machine's speed changes.
- **Reason:** the schedule should finish on time on the day. The timing steps run before the warm-up is known, so they have no pushforward.

### B49: How Task 6's runs are recorded and checked (2026-11-03)

- **Decision:** the check runs on each member's `best.pt`, over 100 validation steps from frame 20. "Finite" means no frame blown up in the preregistered sense: not finite, or above 10. "An improvement over persistence" means a strictly lower capped mean error on the same frames. Results go to `results/ensemble.json`, updated after each member, with each verdict kept beside its checkpoint. A member failing twice is recorded and the job exits with 1, with no third try. A selection tie goes to the U-Net, the expected backbone. The ablations are overlays on the winner's config, trained with `scripts/train_ensemble.py --ablations`.
- **Alternatives:** counting only NaN and infinity as not finite, which would pass a rollout that reaches huge values. A separate ablation script, which repeats the same skip and resume code.
- **Reason:** the plan is silent on these. The shared code is in `src/sandeha/train/campaign.py`, which the layout doesn't list.

### B50: What the doubt map's units and edge cases are (2026-11-03)

- **Decision:** "normalised units" means the field normaliser (`encode`), the scale `rollout_frame_errors` reports in, not the change's scale. The variance is the population's, taken relative to the first member, so one member or equal members give exactly zero. A non-finite fluid cell gives a score of positive infinity under every aggregation, as the preregistration requires, since a percentile could otherwise pass over it. A frame with no fluid cells scores 0. The caller passes the mask in force on the way into the frame, `x[:, 3:4]`.
- **Alternatives:** the change's scale, which weights channels by how fast they move and not by their overall spread. The sample variance, which is undefined for one member. Treating a frame with no fluid as non-finite, which would hand off forever on a filled domain.
- **Reason:** the plan and design leave these open. No preregistered rule changed. The population variance means sizes 2, 3 and 5 aren't on the same absolute scale, which matters only if Task 9 compares their doubt values directly rather than their ranks.

### B51: How the public track trains and measures doubt (2026-11-03)

- **Decision:** a member's input is one frame of density, pressure and velocity, and its output is the change to the next. The cooling time isn't an input, so it stays a hidden variable, as in the plan's "input and next frame". Training keeps the main track's pushforward and drops the divergence penalty, since this flow is compressible. The aggregation (the highest within-time figure on valid), the bad-frame threshold and the calibration edges are fixed on the official valid split, written to `results/well_selection.json`, and then used on the test split, for the no-ensemble signal too. Windows are the main track's: frames 21 to 70, with a secondary one ending at frame 100, where this dataset ends.
- **Left out:** the rise on unfamiliar sets, because there's one test set, which the result file states. Independent rollouts, which section 6.2 doesn't list.
- **Alternatives:** the same input with the cooling time as a constant channel. It would differ from the plan, and from how The Well's own benchmark is posed.
- **Reason:** the plan names only `well/data.py`, and the code needs testing, so training, measures and evaluation are in `src/sandeha/well/{train,doubt,evaluate}.py`, with thin scripts. The measures don't use `eval/metrics.py`, because Task 9 isn't built. The rules are those in `docs/PREREGISTRATION.md`.

### B52: What the hybrid simulator and hand-off selection settle (2026-11-03)

- **Decision:** a hand-off blocks inside `step`, returns every frame in `StepInfo.frames`, and keeps the solid it began with, so an edit waits for it to end. Upsampling is bilinear. The ensemble runs on every decision, so `AlwaysSolver` pays for it too. `reset` also takes `frame`, `late_obstacle` and `late_frame`, as a late obstacle can land inside a hand-off, and masks at either grid. After a solver failure, frames are set to NaN, so nothing can make them look finite. The selection's code is `src/sandeha/handoff_selection.py`, which also holds the primary per-frame error, and `scripts/select_handoff.py`. **never** goes first in a tie, as a run tying it gained nothing. If no aggregation has an evidence figure, none is left in and **never** is chosen. An aggregation with no paired interval doesn't stay in. A restart keeps finished rollout sets unless the settings, data or checkpoints changed. It doesn't refuse a new commit: each set records its commit, and the report lists them all.
- **Alternatives:** one solver frame per `step`, which lets an edit land mid-hand-off, against the test plan. Keeping selection for Task 9's `eval/experiments.py`, which leaves nothing to run now. Refusing a restart on any new commit, which would discard hours of rollouts over a docs commit.
- **Reason:** the plan and preregistration leave these open, and no preregistered rule changed. Task 9 should import this per-frame error and these selection steps, and its `test_selection_steps_and_tie_breaks` is covered by `tests/test_handoff_selection.py`.
- **Cost:** `AlwaysSolver` timing overstates a pure solver, so its figure isn't to be used as the timed reference. Selection steps one scene at a time, up to 72 rollout sets of 30 scenes, and the time this takes is still unmeasured. `--dry-run` uses a tiny split solved on the CPU, as `generate` needs a GPU, and its normaliser comes from that split.

### B53: What the demo server settles (2026-11-03)

- **Decision:** a frame is a `uint32` header length, the JSON header, then the three layers, with lengths in domain units and row 0 at the bottom. The header also carries `members`, `weights` and `res`, so the page can show the member count and disclose untrained models. In every mode a non-finite model output hands off, so forced AI mode shows such a frame as a solver frame. A blown-up state restarts from the last finite frame with a hand-off, or from an empty box if that hand-off failed too. `HybridSimulator` gains an optional `on_solver_frame`, so hand-off frames appear as they are made and the slowdown shows as it happens. Presets are fixed scenes: one source, physics in the middle of the training ranges, and one named shape, with `empty` as well. `--random-weights` builds full-size untrained models, labelled in every frame, and `--device cpu` is allowed on request. The benchmark's ladder is 5, 4, 3, 2 members, stopping at the first to reach 15 FPS, and refuses to write `results/demo_bench.json` from random weights.
- **Alternatives:** publishing a hand-off's frames when it ends, which would show only the last. Leaving `AlwaysAI` in forced AI mode, which would leave NaN on screen for good. Stopping the ladder at 3 members, the design's fallback for the ensemble's size, when the demo's rule is only "fewer members, disclosed on screen".
- **Reason:** the plan leaves these open, and no preregistered rule changed. Nothing here affects the experiments: `on_solver_frame` defaults to none.
- **Cost:** with fewer members than 5, the frozen `tau` was chosen at another ensemble size, whose doubt isn't on the same absolute scale (B50), so hand-off behaviour in that demo isn't the measured one. The simulation runs as fast as it can, with no frame-rate cap, so AI-mode smoke moves faster on a faster machine.

### B54: What the experiments settle (2026-11-03)

- **Decision:** usable time is the frames before the per-frame error first passes 1.0 (`configs/eval.yaml`). On a blown-up frame, the secondary metrics have no value and are counted beside their mean, except vorticity correlation, which counts as 0 so that it can't flatter the average, and divergence, which is infinite. "Solver only" is one long hand-off, so that the solver keeps its fine state. The no-ensemble signal's calibration edges are validation's deciles of it, so `--final` also opens `val`, read for that alone. A plain run writes to `runs/experiments/val_results`, and only `--final` writes to `results/`. `spearman`, `auroc`, calibration and risk-coverage are the one copy in `well/doubt.py`, which `eval/metrics.py` re-exports. Sweeps add fixed periods and random rates from the config to validation's candidates, and use one random seed. Pooled `test_c` joins the sides' runs by role, and a side with a budget of 0 joins as its no-hand-off run. If **never** was chosen, step 1's leader stands in for the doubt figures. The reduced trainings run to frame 70, and the oracle's search too, with its kept candidate run again to frame 120.
- **Alternatives:** usable time from the bad-frame threshold, which is a share of validation's errors and not a level of accuracy. Dropping blown-up frames from secondary means, which would flatter a blown-up policy. Leaving calibration out of the no-ensemble comparison, against the preregistration's "the same measures".
- **Reason:** the plan and preregistration leave these open, and no preregistered rule changed. The figures' file names and the `RESULTS.md` tables are also the build's choice.
- **Cost:** a set can take about 45 rollout sets, and `test_c` about twice the baselines, at one scene at a time. That time is still unmeasured, as no model exists. Each rollout set logs its seconds, and `--experiments` runs a subset, so that ablations can go last. `--final` doesn't refuse a second run: `PREREGISTRATION.md` and git hold that rule.
