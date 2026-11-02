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

**Hoped for, not guaranteed:**

- The neural simulator beats the coarse 128×128 solver on accuracy, for the same compute time.
- The neural simulator is meaningfully faster than the reference solver, ensemble cost included.
- With the threshold frozen on validation, doubt-based hand-off beats fixed and random schedules at the same solver budget on Test B.

A hoped-for result that fails is reported as a finding, with analysis of why.

## Out of scope

3D simulation, industrial-grade accuracy, real-world prediction of floods, pollution or weather, public hosting of the demo, new scientific claims, and per-region hand-off in the first version.

## Failure conditions

Each of these is a failure even if the system technically works:

- Any invented, unmeasured, cherry-picked or silently re-run number.
- Tuning anything on the test sets.
- A speed claim that leaves out ensemble cost, or that only compares against an artificially slow solver.
- A demo that only works on presets or is secretly pre-rendered.
- Code or theory the owner cannot explain.
