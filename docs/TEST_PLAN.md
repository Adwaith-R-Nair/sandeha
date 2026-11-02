# Sandeha: Test Plan

## Strategy

- Tests are written first and seen to fail before the implementation is written.
- `uv run pytest` runs the suite from the project root. Test layout mirrors `src/`.
- Tests run on the real GPU where the behaviour depends on it. There is no CPU fallback for the environment check.
- A result counts as validated only when it comes from a saved run. Summaries of a run are not evidence.
- Each phase gets an independent check, separate from its build, before it is committed.

## Environment and repository (Task 1)

| Test | Checks |
|---|---|
| `tests/test_env.py::test_cuda_runs_a_matmul` | The installed PyTorch multiplies two 1024×1024 matrices on the GPU, and the result is finite |
| `tests/test_long_job.py` | Wrapper refuses on offline or unknown mains with exit 3, runs the job under the inhibitor (job command in the inhibitor's arguments), logs output and start and end, passes on the exit code, reports 128+n for a job killed by a signal, exits 4 and logs `exit=unknown` when the job never reports a status, forwards SIGTERM, SIGINT and SIGHUP to the job and its children, waits for job cleanup with the inhibitor held, handles a signal during inhibitor start-up, runs from the project root, rejects bad usage and the reserved name `jobs` with exit 2. The fake inhibitor copies the real one's exit 1 for a job killed by a signal |
| `tests/test_repo.py` | Root `data/`, `runs/` and `checkpoints/` are ignored, while `src/sandeha/data/` and `results/` are not |

The wrapper tests use a fake power-supply directory and a recording `systemd-inhibit`. The real inhibitor and real power state are exercised by hand, as logged in `BUILD_LOG.md`.

## Solver checks (Task 2)

Required to pass before any data is generated, all in `tests/physics/test_solver.py`, on the GPU:

| Test | Checks |
|---|---|
| `test_divergence_known_answer` | `u = x` gives divergence 1 away from walls, independently of `project` |
| `test_projection_removes_divergence` | Random velocity at 64×64, with and without an obstacle: largest divergence in fluid cells below `1e-3`, and the flow is not simply erased |
| `test_no_flow_into_obstacles` | After 50 steps, with inflow partly drawn over the obstacle: zero velocity and density in solid cells, and density non-negative |
| `test_deterministic` | Two runs in one process give identical tensors |
| `test_buoyancy_lifts_smoke` | The density centroid rises, and stays at the inflow with zero buoyancy |
| `test_converges_with_refinement` | 128×128 error below 0.8 of 64×64 error against 256×256, on each grid and on a common grid. Smooth inflow to `t=1`, hard inflow to `t=0.5` (`DECISIONS.md` B12) |
| `test_sealed_cavity_stays_finite` | A closed ring stays finite and converged, and nothing crosses it |
| `test_fully_solid_domain_stays_finite` | All-zero output and no failure of the pressure solve |
| `test_smoke_does_not_cross_a_thin_wall` | A one-cell wall across the box keeps all smoke below it |
| `test_obstacle_change_mid_run_is_respected` | A late obstacle gets a new pressure setup, and after exactly 1 step its cells hold no velocity or density |
| `test_new_wall_stops_smoke_on_its_first_step` | Flow from before an obstacle change does not carry smoke through a new one-cell wall |
| `test_batch_matches_single_runs` | Batched items match single runs, including iteration counts |
| `test_non_finite_item_does_not_stall_or_poison_the_batch` | In `project`, a NaN or infinite item uses no iterations, stays non-finite and is reported, and the other item is unaffected |
| `test_infinite_face_is_not_reported_as_converged` | One infinite face, which also makes the tolerance infinite, is not accepted |
| `test_non_finite_item_survives_step` | On GPU and CPU, over 3 batched steps, NaN or infinity in velocity, faces or density: no crash, item reported as not converged, the other item matches a single run, and the device still works |
| `test_bad_velocity_in_a_moving_flow_never_heals` | NaN or infinity in `u` or `v` of a moving flow reaches density, and stays non-finite for 10 steps |
| `test_huge_item_survives_step` | Velocity of `1e30` does not crash the solver or disturb the other item |
| `test_iteration_limit_is_reported_as_not_converged` | `cg_max_iter` is respected and reported |
| `test_pressure_solve_needs_few_iterations` | Below 60 iterations at both 64×64 and 256×256 |
| `test_preconditioner_is_symmetric` | Required for conjugate gradients to be valid |
| `test_steady_vortex_stays_centred` | A steady solution does not drift. Catches fields advected on the wrong lattice. A frozen velocity field also passes, so it says nothing about momentum advection |
| `test_vortex_pair_carries_itself` | A vortex pair travels 0.6 to 1.1 of the point-vortex distance, along either axis, without drifting sideways. Fails if either velocity component is not advected |
| `test_advection_follows_a_rotation_without_spiralling` | A rotating blob keeps its radius. Catches a first-order backtrace |
| `test_thin_sheets_have_no_cell_scale_artefacts` | Laplacian-to-density ratio on a thin-sheet plume stays below 0.25 (`DECISIONS.md` B10) |
| `test_smoke_matches_the_mass_budget` | Smoke in the box within 5% of inflow minus decay at `t=1.75` |

A rendered 256×256 run, `results/figures/solver_check.png`, is also inspected by eye. `uv run python scripts/check_solver.py` regenerates it, along with the speed, convergence and long-run divergence numbers.

## Data integrity (Tasks 3 and 4)

Shape families and physics ranges match each split. Test splits refuse to open without `final_eval`, tampered files fail the checksum, seeds are disjoint between splits, an interrupted generation resumes identically, and the normaliser uses train only. Sample scenes from each split are rendered and checked for zero NaNs before use.

## Training (Task 5)

Model output shapes, solid cells forced to zero, zero divergence for uniform flow, overfitting one batch by at least 10×, atomic checkpoints under interruption, resume restoring optimiser and RNG state, and peak GPU memory below 5.5 GB for U-Net at batch 16.

## Ensemble and hybrid (Tasks 7 and 8)

Zero doubt from identical members and from a single member, disagreement raising doubt, doubt scores ignoring solid cells, low divergence after hand-off, schedule and budget policies behaving to specification, the threshold policy triggering only above `tau`, the oracle never reading the future, obstacle edits applying on the next frame, and timing including every ensemble member.

## Experiments (Task 9)

Known-answer tests for each metric. A blown-up rollout counts as capped maximum error and stays in the average. Results carry provenance, the renderer refuses results without it, and experiments open only `val` without `--final`.

## Public track (Task 10)

A small synthetic HDF5 file checks shapes and the test-split guard. After download, a further test confirms the real file's grid, steps and trajectories, and is skipped when the data is absent.

## Demo (Tasks 11 and 12)

| Area | Checks |
|---|---|
| Server | Frames stream, brush applies on the next frame, edits during hand-off apply after it, reconnect keeps running, a filled domain stays finite, a blown-up state recovers through the solver, a missing GPU or exhausted memory gives a clear message and non-zero exit |
| Front end | Canvas updates, drawing changes the obstacle layer, indicator switches in solver mode, no external requests, layout fits 1280×720 |
| Performance | Neural-mode frame rate measured for 60 s with 5 members, written to `results/demo_bench.json` |

## Scientific validity rules

- Tuning uses validation only. Test splits are read only by the single final evaluation.
- Rollouts that blow up are failures at maximum error, and are never dropped.
- Any bug fix after the final run is logged with the before and after numbers.
- Every number in a document comes from saved JSON and is never typed by hand.

## Release check (Task 13)

From a clean clone in a temporary directory: install, download Release weights, run tests, start the demo.
