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
| `tests/test_long_job.py` | Wrapper refuses on offline or unknown mains with exit 3, runs under the inhibitor, logs output and start and end, passes on the exit code, runs from the project root, rejects bad usage with exit 2 |
| `tests/test_repo.py` | Root `data/`, `runs/` and `checkpoints/` are ignored, while `src/sandeha/data/` and `results/` are not |

The wrapper tests use a fake power-supply directory and a recording `systemd-inhibit`. The real inhibitor and real power state are exercised by hand, as logged in `BUILD_LOG.md`.

## Solver checks (Task 2)

Required to pass before any data is generated: divergence near zero after projection, no flow or smoke in obstacles, determinism for a given seed, buoyant rise, convergence under grid refinement, and sealed or fully solid domains staying finite. A rendered 256×256 run is also inspected by eye.

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
