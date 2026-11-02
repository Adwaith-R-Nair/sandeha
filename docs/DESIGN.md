# Sandeha: Design

**Owner:** Adwaith R Nair
**Date:** 2026-11-02
**Status:** Approved by owner, 2026-11-02
**Context:** Deep learning course project, college. Submission on 2026-11-05. The owner is not yet sure whether a real dataset is required, so the project includes a public-dataset track (section 4).

Decisions are marked **[Owner]** when Adwaith specified them and **[Default]** when they are defaults open to change.

---

## 1. Intent

Sandeha (Sanskrit for "doubt") is an AI fluid simulator that knows when not to trust itself.

AI simulators are much faster than conventional physics solvers, but they can drift into believable nonsense with no warning. Sandeha runs an ensemble of neural networks, treats their disagreement as a live doubt signal, and hands control to a conventional physics solver when doubt is too high.

The project answers one question:

> **Does the model's doubt actually predict its errors, and how much speed must be given up for each unit of accuracy recovered?**

### Why it exists

Adwaith wants a deep learning course project that is advanced, unique and memorable. Most course projects stop at "we trained a model and got X%". This one also asks where the model fails, whether it can tell, and what the system should do about it.

The pitch: *"Everyone in this class built an AI that gives answers. I built one that knows when its answers can't be trusted."*

### Honest scope of impact

This is a small 2D simulator. It will not change aircraft design or weather forecasting. Its real value is as a measured small-scale answer to a question that matters in larger settings, and as evidence of Adwaith's ability. Nothing in the project may claim otherwise.

---

## 2. Goals, non-goals, anti-goals

### Goals

1. A trained neural simulator of 2D smoke flowing around arbitrary obstacles, which is faster than the reference solver.
2. A doubt signal, and a rigorous measurement of how well it predicts real error, especially on obstacle shapes never seen in training.
3. A hybrid system that hands off to the solver based on doubt, with a measured speed-versus-accuracy curve compared against baselines.
4. A live browser demo where anyone can draw obstacles and watch the smoke, the doubt heatmap and the hand-offs in real time.
5. Professional documentation **[Owner]**: PRD, HLD, LLD, build log and supporting documents (section 9).
6. A GitHub repo in Adwaith's name, built phase by phase and commit by commit **[Owner]**.
7. Everything Adwaith needs to explain and defend the project in a viva.

### Non-goals

- 3D simulation, or industrial-grade accuracy.
- Real-world prediction of floods, pollution or weather.
- Public hosting of the demo. It runs locally on Adwaith's laptop. **[Default]**
- New scientific claims. Recent research already explores this kind of fallback.
- Per-region hand-off in the first version (see section 5.4).

### Anti-goals: failure even if it technically works

- Any invented, unmeasured, cherry-picked or silently re-run number. A weak result gets reported as a weak result.
- Tuning anything on the test sets.
- A speed claim that leaves out the cost of running the ensemble, or which only compares against an artificially slow solver.
- A demo that only works on presets or is secretly pre-rendered.
- Code or theory Adwaith cannot explain.
- Reuse of MyoGrip. This is a separate project **[Owner]**.

---

## 3. Constraints

| Constraint | Detail |
|---|---|
| GPU | RTX 3060 Laptop, 6 GB memory, driver 595.99.02. Working as of 2026-11-02. |
| CPU, RAM, disk | Ryzen 7 6800H (16 threads), 14 GB RAM, about 190 GB free. |
| OS and Python | Ubuntu 26.04. System Python is 3.14 and has no PyTorch, so use `uv` with a project-local Python 3.12. **[Default]** |
| Time | Submission 2026-11-05. Build runs overnight, and may overrun **[Owner]**. Code and experiments freeze by the morning of 2026-11-04, leaving the owner a full day to learn the project and rehearse **[Default]**. |
| Data | Main experiment data is generated locally. Public data only where it needs no sign-up (section 4). |
| Commit identity | `git config user.name "Adwaith R Nair"`, email `adwaith.r.nair189@gmail.com`, set in this repo. Commits carry no co-author or tool-attribution lines **[Owner]**. |
| Repo | Local at `~/Coding/Projects/sandeha`. GitHub repo `Adwaith-R-Nair/sandeha`, **public** **[Owner]** |
| Size limits | Never commit datasets or checkpoints to git. Final weights go in a GitHub Release, so the demo can be reproduced. **[Default]** |

Before the overnight run Adwaith must: reboot (one desktop process still has old driver files loaded), keep the laptop plugged in **[Owner]**, and set the lid and sleep settings so it stays awake. Long jobs should also run under `systemd-inhibit`.

---

## 4. The physics problem

2D incompressible smoke: a density field carried by a velocity field, with buoyancy, driven by inflow sources and blocked by solid obstacles.

### Reference solver (the source of truth)

- A conventional solver written for this project, on GPU in PyTorch, with semi-Lagrangian or higher-order advection, buoyancy, a solid obstacle mask, and a tightly converged pressure solve.
- It runs at **256×256** and is downsampled to **128×128** to give training targets. **[Default]**

**Why finer than the AI's grid:** a simple solver at 128×128 is itself fast on a GPU, so beating it on speed alone proves nothing. Training on results from a finer, more expensive grid gives the AI something worthwhile to learn: fine-grid accuracy at coarse-grid cost.

### Required solver checks

Before any data is generated, tests must show that: divergence after projection is near zero; smoke behaves sensibly with no flow through obstacles; results converge as the grid is refined; runs are deterministic for a given seed.

### Data

- Random scenes: obstacle shapes, positions, inflow positions and strengths, buoyancy.
- A share of runs where obstacles appear mid-simulation, because people will draw while the demo runs.
- One AI step advances several solver steps (starting point: 4), so the AI gets a real speed advantage.
- Generated with a fixed seed and split by whole simulation, never by frame, so near-identical frames can't leak between training and test.

### Test sets, locked before training

| Set | Contents | Purpose |
|---|---|---|
| Validation | Same distribution as training | All tuning and threshold selection |
| Test A | Same distribution, unseen scenes | Baseline performance |
| Test B | **Unseen shape families**, such as stars, letters and hand-drawn blobs | Does doubt rise where the model is out of its depth? |
| Test C | Inflow and buoyancy outside the training range | Same question, for physics rather than geometry |

Training shapes are limited to simple families such as circles, rectangles and walls, so that Tests B and C are truly unfamiliar.

### Public-dataset track

The owner said a real dataset might be needed **[Owner]**, so the same method also gets tested on data the project didn't generate itself.

- **Dataset: `turbulent_radiative_layer_2D` from The Well** (Polymathic AI). 7.9 GB, CC-BY-4.0, no sign-up. **[Default]**
- **Method:** train a smaller ensemble of 3 on its official training split. Measure whether doubt predicts error on its official test split, using the section 6.2 measures.
- **Limit:** there's no solver for this physics in the project, so this track tests doubt only and has no hand-off.
- **Honest label:** this is a published scientific simulation benchmark, not physical measurement, and must be described that way everywhere. If the course requires measured real-world data, it does not satisfy that.
- **Rejected:** ScalarFlow (real captured smoke) needs a sign-up form, is almost half a terabyte, and is 3D. Other 2D sets in The Well such as `shear_flow` (471 GB) and `rayleigh_benard` (370 GB) are too large.
- The loader must confirm the dataset's grid, step and trajectory counts rather than assume them.

---

## 5. The model and the system

### 5.1 Input and output

Input: velocity (2 channels), density, obstacle mask, inflow map (which carries inflow strength), and buoyancy as a constant channel. **[Default]** Buoyancy can't be inferred from a single frame, so leaving it out would make it a hidden variable and undermine Test C. Output: the change in velocity and density over one AI step. Solid cells are enforced as hard constraints after prediction.

### 5.2 Architectures

- **U-Net:** the expected ensemble backbone.
- **Fourier Neural Operator (FNO):** trained and compared on the same task.

The winner on validation forms the ensemble. If FNO fails for practical reasons, record why in the build log and continue with U-Net.

### 5.3 Training

- Prediction loss plus a physics penalty on divergence.
- Rollout training, so the model stays stable while feeding on its own predictions. Use a pushforward-style method or short unrolled rollouts, whichever validates better.
- Noise or other stability techniques are left to the builder.
- **Ensemble of 5** independently trained models, with different seeds and data order. **[Default]** 3 is an acceptable fallback if time runs short, recorded as a change.

### 5.4 Doubt and hand-off

- **Doubt map:** per-cell standard deviation across the ensemble.
- **Doubt score:** one number per frame aggregated from the map, using a method chosen on validation.
- **Prediction:** the ensemble mean.
- **Rollout:** at each step, every member starts from the shared ensemble mean. **[Default]** With independent rollouts, members drift apart with time regardless of what's happening in the flow, so doubt could become a disguised clock. Independent rollouts are still reported as a secondary comparison.
- **Hand-off:** when the doubt score passes a threshold, the state is upsampled to 256×256 and projected to remove divergence. The reference solver runs for a set number of frames, then its result is downsampled and the AI resumes.
- **Whole-frame hand-off** for the first version. **[Default]** Region-by-region hand-off would be more elegant but much riskier, and is only a stretch goal.

---

## 6. Experiments

These numbers are the heart of the project. All results use Tests A, B and C, across several seeds and scenes, with confidence intervals.

### 6.1 Is the simulator accurate?

Normalised error at several rollout lengths, how long predictions stay usable, and vorticity correlation. Compared with:
- a persistence baseline (nothing changes);
- the **coarse solver at 128×128**, the honest "why not just run a cheap solver?" comparison;
- the reference solver, for timing.

### 6.2 Does doubt predict error?

- Rank correlation between doubt score and true error.
- AUROC for detecting frames whose error passes a threshold fixed on validation.
- Calibration plots and risk-coverage curves.
- Whether doubt rises on Tests B and C relative to A.
- Correlation reported both pooled and **within a fixed rollout time**, since error and doubt can both rise with time alone. The within-time figure is the one that counts as evidence.
- Comparison with a no-ensemble signal: the divergence of a single model's output.

### 6.3 Is doubt-based hand-off worth it? (The main result)

Sweep the threshold and plot accuracy against wall-clock time per simulated frame, ensemble cost included.

**Metrics, fixed before Phase 3 and never changed after seeing results: [Default]**
- **Primary:** normalised error against the reference run, averaged over a fixed rollout horizon, starting at 50 AI steps. The exact horizon is set from validation and recorded before Phase 3 finishes.
- **Secondary:** physical measures at longer horizons: divergence, vorticity spectrum and plume statistics.
- **Known limit:** smoke is chaotic, and the solver resumes from a state that's already off. It can slow error growth but won't return to the reference path, so point-by-point error alone may undersell hand-off at long horizons. That's why the physical measures are reported too.

**Headline comparison: [Default]**
- The threshold is frozen on validation and then applied unchanged to each test set.
- Fixed and random baselines get the same solver budget that frozen threshold actually used on that test set.
- The full sweep curves are also shown on the test sets, for description only. They are never used to choose anything.

Policies:

| Policy | Why it's included |
|---|---|
| Solver only | Upper bound on accuracy |
| Single model, no hand-off | What most AI simulators do |
| Ensemble, no hand-off | Is the ensemble itself useful? |
| **Hand-off on a fixed schedule** | The key baseline. If doubt can't beat this, it isn't earning its keep |
| **Hand-off at random, same solver budget** | Second key baseline |
| **Doubt-based hand-off** | The system under test |
| Oracle using true error | Upper bound on any possible doubt signal |

### 6.4 Ablations

Single seed and shorter training, for comparison only:
- without the physics penalty;
- without rollout training;
- ensemble size 2, 3 and 5, from the existing models.

### 6.5 What counts as success

**Must hold regardless of outcome:**
- All solver tests pass, and test sets stayed locked.
- Every figure regenerates from a command, and every number traces to a saved run.
- The speed and accuracy experiments are completed and reported honestly.

**What we hope to see, not guaranteed:**
- The AI beats the coarse 128×128 solver on accuracy, for the same compute time.
- The AI is meaningfully faster than the reference solver, ensemble cost included.
- With the threshold frozen on validation, doubt-based hand-off beats fixed and random schedules at the same solver budget on Test B.

If a hoped-for result fails, it gets reported as a finding, with analysis of why. Up to one further signal may be tried, chosen on validation only, and disclosed.

---

## 7. The demo

A browser page served locally by a Python server, using WebSockets to the laptop's GPU. **[Default]** This is simpler than in-browser inference, and the solver needs the GPU anyway. It must run with no internet, as classroom Wi-Fi can't be relied on.

**What the user can do:** draw and erase obstacles with the mouse, including while it's running; pick presets, including the unseen shapes; reset; adjust the doubt threshold; force AI-only or solver-only mode.

**What the user sees:**
- smoke and doubt heatmap side by side;
- a clear indicator showing whether AI or the physics solver is in control;
- live frames per second, and running share of frames handled by the solver.

The slowdown during hand-off is shown as it really happens. That's part of the story.

**Quality bar:** it should look like a polished instrument rather than a student dashboard, be understood within 10 seconds, and run with a target of at least 15 frames per second in AI mode. A recorded backup video, clearly a recording of the real system, is included in case of demo trouble.

---

## 8. Build approach

**[Owner]** Built end to end, phase by phase, running overnight, with commits pushed as it goes.

Phases, in order:

1. Project setup, environment, docs skeleton
2. Reference solver and its tests
3. Data generation and test set locking
4. Single-model training, FNO comparison
5. Ensemble training
6. Evaluation, all experiments and figures
7. Demo
8. Final documents, viva guide, report

**Overnight rules:**
- Long jobs must be resumable from checkpoints and safe to restart.
- Each phase gets an independent check, separate from its build, before the phase is committed.
- Watch GPU temperature and memory. A laptop GPU can throttle or run out of memory.
- Results come from the runs, never from a summary of a run.
- No `sudo` is available overnight. Anything needing it is logged for Adwaith in the morning, and work continues elsewhere.
- If a crash or interruption stops the run, the build log must make it possible to pick up where it stopped.

**Rough time budget (estimates, untested):** data about 1.5 h, five ensemble members about 4 h, FNO about 45 min, public-dataset track about 2 h, ablations about 1.5 h, evaluation about 1 h. If time gets short, priority is: ensemble, then main experiment, then demo, then public-dataset track, then ablations. Compute may spill past one night, but the freeze in section 3 holds: anything unfinished by then is cut and recorded as cut.

**Fallback ladder if things go wrong:**

| Problem | Response |
|---|---|
| 128×128 too heavy or too slow | Drop to 96×96 |
| Rollouts unstable | Stronger rollout training, or a larger AI step |
| Ensemble training too slow | Ensemble of 3 |
| Demo under target frame rate | Fewer ensemble members in the demo, disclosed on screen |

Every change gets recorded in the build log with its reason.

---

## 9. Documentation **[Owner]**

In `docs/`, kept current as the build goes on, not written up at the end:

| Document | Contents |
|---|---|
| `PRD.md` | Problem, users (student, professor, demo audience), requirements, success measures |
| `HLD.md` | System architecture, components, data flow, with diagrams |
| `LLD.md` | Module design, interfaces, data formats, model details, training and hand-off logic |
| `BUILD_LOG.md` | Dated, phase-by-phase record of what happened, including failures and changed decisions |
| `DECISIONS.md` | Each major decision, the alternatives and the reason |
| `TEST_PLAN.md` | Test strategy, solver checks, and how the results were validated |
| `RESULTS.md` | All numbers and figures, each traced to a run |
| `MODEL_CARD.md` | What the models are, what they were trained on, and where not to trust them |
| `VIVA_GUIDE.md` | Plain-language explanations of each concept, likely professor questions and honest answers, demo script |
| `REPORT.md` | Course report draft, in standard academic structure, with references |

Plus a `README.md` with results, a demo GIF and one-command setup, and this design document at `docs/DESIGN.md`.

The documents describe the project and its work. They must not make any statement about how the code was authored **[Owner]**.

---

## 10. References to check

To be verified against the original papers before they're cited in the report. So far only a summary of the first has been read.

- *Hybrid Neural World Models*, arXiv 2605.28317. Closest found prior work on surrogate-to-solver fallback.
- Li et al., *Fourier Neural Operator for Parametric PDEs*, 2020.
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, 2017.
- Brandstetter et al., *Message Passing Neural PDE Solvers*, 2022. Pushforward training.
- Kochkov et al., *Machine learning-accelerated computational fluid dynamics*, 2021. Learning from finer grids.
- Stam, *Stable Fluids*, 1999. Solver basis.
- Ohana et al., *The Well*, 2024 (arXiv 2412.00568). Public-dataset track.

---

## 11. Left to the builder

Exact network sizes and hyperparameters, solver numerics within the section 4 requirements, doubt aggregation method (chosen on validation), hand-off duration, dataset size within the time budget, config and experiment-tracking tools, repo layout, front-end stack and visual design details, test framework and CI setup.
