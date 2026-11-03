# Sandeha: Product Requirements

**Owner:** Adwaith R Nair
**Source:** `docs/DESIGN.md`. Where this document and the design differ, the design decides.
**Deadline:** submission on 2026-11-05. Code and experiment freeze on the morning of 2026-11-04.

## Problem

Neural simulators are much faster than conventional physics solvers, but they can drift into believable nonsense with no warning. A user has no way to tell a good frame from a bad one while the simulation runs.

Sandeha is a 2D smoke simulator built around one question:

> Does the model's doubt actually predict its errors, and how much speed must be given up for each unit of accuracy recovered?

## Users

| User | Needs |
|---|---|
| Student (project owner) | A system that can be explained and defended in a viva, with every number traceable |
| Professor | Rigorous method, honest reporting, reproducible results, including weak results |
| Demo audience | To understand within 10 seconds what is happening, and to try it by drawing obstacles |

## Functional requirements

| ID | Requirement |
|---|---|
| F1 | A reference solver for 2D incompressible smoke with obstacles, inflow and buoyancy, running at 256×256 on the GPU |
| F2 | Generated datasets with a fixed seed, split by whole simulation, with validation and three locked test sets (A: same distribution, B: unseen shape families, C: inflow and buoyancy outside the training range) |
| F3 | A neural simulator at 128×128 that advances 4 reference solver steps per step. U-Net and FNO are compared on validation, and the winner forms the ensemble |
| F4 | An ensemble of 5 producing a mean prediction, a per-cell doubt map and a per-frame doubt score |
| F5 | A hybrid simulator that hands the whole frame to the reference solver when doubt passes a threshold, then resumes the neural model |
| F6 | Experiments covering simulator accuracy, doubt against error, hand-off against baselines, and ablations, each writing JSON with provenance |
| F7 | A public-dataset track that measures doubt on `turbulent_radiative_layer_2D` from The Well with an ensemble of 3. It is labelled as a published simulation benchmark, not measured data, and has no hand-off |
| F8 | A local browser demo: draw and erase obstacles while running, presets including unseen shapes, reset, doubt threshold control, and forced neural-only or solver-only mode |
| F9 | Smoke and doubt shown side by side, with a clear indicator of who is in control, live frames per second and running solver share |

## Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | Runs on one RTX 3060 Laptop GPU with 6 GB of memory, 14 GB of RAM |
| N2 | Demo targets at least 15 frames per second in neural mode, and runs with no internet |
| N3 | Long jobs resume from checkpoints, and survive a kill during checkpoint writing |
| N4 | Rollouts that blow up count as failures at maximum error and are never dropped |
| N5 | The demo stays up when a user seals off or fills the domain, draws over an inflow, or disconnects and reconnects |
| N6 | Every figure regenerates from a command, and every reported number traces to a saved run |
| N7 | Datasets and checkpoints are never committed. Final weights go in a GitHub Release |

## Success measures

**Must hold regardless of outcome:**

- All solver tests pass, and test sets stayed locked.
- Every figure regenerates from a command, and every number traces to a saved run.
- The speed and accuracy experiments are completed and reported honestly.

| Measure | Outcome on 2026-11-04 |
|---|---|
| Solver tests pass, and test sets stayed locked | **Yes, with one check still to repeat.** The last recorded full run is 321 passed at `dd78b8d`. The tests weren't rerun on the latest commits, because they need the GPU and it's in use. The solver hasn't changed since before that run. `--final` ran once, and the post-hoc work opened no test split |
| Figures regenerate, and numbers trace to a saved run | **Yes, with limits.** Numbers in written text are typed, with their source named beside them. Times in milliseconds are single measurements on one laptop GPU. The limits are listed in `RESULTS.md` |
| Experiments completed and reported honestly | **Yes** for the main track, weak results included. The public track is pending |

**Hoped for, not guaranteed:**

- The neural simulator beats the coarse 128×128 solver on accuracy, for the same compute time.
- The neural simulator is meaningfully faster than the reference solver, ensemble cost included.
- With the threshold frozen on validation, doubt-based hand-off beats fixed and random schedules at the same solver budget on Test B.

| Hope | Outcome on 2026-11-04 | Figures |
|---|---|---|
| Beats the coarse solver on accuracy, for the same compute time | **Partly.** Yes on Test A. Yes on Test B in the primary window, with intervals that touch, and not at 100 frames. No on Test C. "The same compute time" wasn't tested directly | Ensemble against coarse: 0.256 and 0.346 on Test A, 0.308 and 0.369 on Test B, 0.219 and 0.152 on the low side of Test C, 0.526 and 0.516 on the high side (`results/exp_accuracy.json`) |
| Meaningfully faster than the reference solver, ensemble cost included | **Yes** for the ensemble alone, for this project's own untuned solver on one laptop GPU. Hand-off gives much of that back, and on the high side of Test C it is slower than the reference | Ensemble 14.9 to 15.2 ms per frame, reference 192.4 to 262.8, about 13 to 17 times. Doubt-based hand-off 67.1 ms on Test B, and 257.7 ms on the high side of Test C (`results/exp_accuracy.json`, `results/exp_handoff.json`) |
| With a frozen threshold, doubt-based hand-off beats fixed and random schedules on Test B | **No.** It beat neither on any test set, and on Test B it was worse than the fixed schedule | Doubt 0.2881, fixed 0.2652, difference 0.0229 [0.004, 0.041]. Doubt less random 0.011 [-0.005, 0.025] (`results/exp_handoff.json`) |

A hoped-for result that fails is reported as a finding, with analysis of why. That analysis is in `RESULTS.md`. It is post-hoc, and it needs new data to confirm.

**Other requirements with a measured outcome:**

- **N2:** met in neural mode, at 61.3 frames per second with 5 members against a target of 15 (`results/demo_bench.json`).
- **N7:** datasets and checkpoints aren't in git. The release of the weights (`v1.0`, `sandeha-weights.tar.gz`) hadn't been published when this was written.

**PUBLIC TRACK PENDING: F7.** `results/exp_well.json` doesn't exist yet, so F7 has no outcome and this document gives no figure for it. Fill this in from that file once it exists.

## Out of scope

3D simulation, industrial-grade accuracy, real-world prediction of floods, pollution or weather, public hosting of the demo, new scientific claims, and per-region hand-off in the first version.

## Failure conditions

Each of these is a failure even if the system technically works:

- Any invented, unmeasured, cherry-picked or silently re-run number.
- Tuning anything on the test sets.
- A speed claim that leaves out ensemble cost, or that only compares against an artificially slow solver.
- A demo that only works on presets or is secretly pre-rendered.
- Code or theory the owner cannot explain.
