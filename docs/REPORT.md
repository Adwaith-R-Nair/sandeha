# Sandeha: Testing Whether Ensemble Doubt Should Decide When a Neural Smoke Simulator Calls a Solver

**Adwaith R Nair.** Course report draft, 2026-11-04.

Every number here is copied from the file in `results/` named beside it. `docs/RESULTS.md` has the full tables. Lower is better for every score. Brackets are 95% bootstrap intervals over whole scenes.

## Abstract

Neural simulators are fast, but they can drift away from the true flow and give no warning. This project trained five small U-Nets to simulate 2D smoke around obstacles, used their disagreement as a doubt signal, and handed off to a conventional solver for a few frames when doubt passed a threshold. The rules were preregistered, the threshold was frozen on validation, and the test run happened once.

The main idea did not work. Doubt-based hand-off did not beat a fixed or random schedule at equal solver budget on any test set. On unseen shapes, the set it was meant for, it was significantly worse than the fixed schedule, 0.2881 against 0.2652, a difference of 0.0229 [0.004, 0.041]. An oracle that used the true error was no better than doubt on familiar scenes or unseen shapes. Doubt does correlate with error at a given rollout time, but on unseen shapes a divergence signal with no ensemble scored higher, 0.676 [0.537, 0.755] against 0.522 [0.286, 0.673], with overlapping intervals and no paired interval saved. The ensemble is about 14 times faster than this project's own reference solver, and more accurate than a coarse solver on familiar scenes and, in the primary window, on unseen shapes, but not on either side of the out-of-range set. This is a small test on generated 2D data, and it claims no new science.

## 1. Introduction

A neural simulator learns one time step of a flow and then feeds on its own output. Small errors can grow, and the output can still look believable. If a simulator could tell when it was going wrong, it could call a slower, trusted solver only then, and keep most of its speed.

This project tests that idea on a small system. It asks two questions:

1. Does disagreement between ensemble members predict the error of the ensemble's rollout?
2. Does handing off to a solver when disagreement is high beat a schedule that doesn't look at the flow, for the same solver time?

The headline comparison was fixed in advance: the primary metric of doubt-based hand-off against fixed and random baselines at matched solver budget, on the unseen-shapes set, with the threshold frozen on validation (`docs/PREREGISTRATION.md`). A failed hope was to be reported as a finding, and it is one.

## 2. Related work

Falling back to a solver is already studied, and this project doesn't claim otherwise.

- **ANCHOR** (Roy et al. [9]) is the closest prior work found on the mechanism. It tracks a signal during the rollout, and passing a threshold runs the solver for a fixed number of steps before handing back. Its signal is a moving average of the normalised PDE residual, so it needs the governing equation. It isn't an ensemble spread.
- **HyPER** (Srikishan et al. [10]) uses a policy trained by reinforcement learning to choose the surrogate or the simulator at each step, and compares it with a random policy that calls the simulator the same number of times.
- **Hybrid Neural World Models** (Lakshmanan and Chopra [1]) is related, but differs in most ways that matter. It uses step-doubling as its signal, makes one decision per trajectory, and has the solver replace the whole trajectory. It also treats ensembles as a baseline to beat: its Table 3 gives a 3-member deep ensemble a mean AUROC of 0.77 against 0.82 for step-doubling, on its own systems. That says nothing about this system, but it isn't support for the approach here. It is a preprint, marked as under review.
- **Deep ensembles** (Lakshminarayanan et al. [3]) are the source for training networks from different seeds and data order to express uncertainty. The paper trains each network on negative log-likelihood to predict a mean and a variance. The members here use MSE and take the spread of their outputs, which the paper calls "a commonly used heuristic" and finds inferior on a toy regression task. Whether that matters here wasn't measured.
- **Other sources.** The solver's splitting and semi-Lagrangian step come from Stam [6], but its staggered grid, conjugate gradient solver and buoyancy don't. Training a coarse model against a finer simulation follows the idea in Kochkov et al. [5], whose models work inside a coarse solver, while the networks here replace solver steps. Pushforward training is based on Brandstetter et al. [4], with a different unrolling. The U-Net comes from Ronneberger et al. [11], a paper on image segmentation, and the rival architecture from Li et al. [2].

A limited search, a few web queries and the related-work section of [1], found no paper where ensemble spread triggers a solver hand-off in the middle of a neural rollout, with the network then resuming. That doesn't show none exists, so this report doesn't call the approach new.

## 3. Methods

### 3.1 Solver and data

The reference solver is this project's own PyTorch stable-fluids solver at 256×256, with `dt` 0.01. It uses semi-Lagrangian advection on a staggered grid, and pressure projection by conjugate gradient with a multigrid preconditioner. Each frame is averaged down to 128×128 for the networks. One network step covers 4 solver steps.

All main-track data is generated by this solver, so there's no measured smoke anywhere in it (`results/data_manifest.json`, `docs/MODEL_CARD.md`). Each scene has 121 frames and 1 to 3 obstacles, and splits are by whole scene.

| Split | Scenes | What is different |
|---|---|---|
| `train` | 240 | Circles, rectangles and walls, with training ranges of buoyancy and inflow |
| `val` | 30 | Same kinds of scene |
| `test_a` | 30 | New scenes of the same kinds |
| `test_b` | 30 | Unseen shapes: stars, letters and blobs |
| `test_c` low | 10 | Buoyancy and inflow both below the training ranges |
| `test_c` high | 20 | Buoyancy and inflow both above the training ranges |

Test splits are locked in code: `open_split` refuses a test split unless it is called for final evaluation, and checks each file against the manifest.

### 3.2 Models

Each member is a U-Net with widths 32, 64, 128 and 256, and 2,263,107 parameters. It takes velocity, density, solid mask, inflow and buoyancy as input, and predicts the change over one step. The U-Net was chosen over an FNO on validation, on equal 40-minute budgets: 0.0915 against 0.1258, with persistence at 0.6689, one seed each, on a separate 20-step measure (`results/arch_selection.json`). That shows only that this FNO lost this comparison.

The loss is prediction error plus a divergence penalty with weight 0.1, with pushforward training. Five members, seeds 0 to 4, trained for 48 minutes each on one RTX 3060 Laptop GPU (`results/ensemble.json`). At each rollout step, every member starts from the shared ensemble mean.

### 3.3 Doubt and hand-off

The doubt map is the per-cell standard deviation across members, in normalised units, averaged over channels. The doubt score is one number per frame, from that map over fluid cells. Each frame, if the score is above the threshold `tau` or isn't finite, the ensemble's frame is thrown away. The previous frame is upsampled and projected, the solver makes a set number of frames, and the ensemble resumes from the last of them.

Selection used validation only, by steps fixed in advance, and picked the lowest error with a solver share at or below 25%. It chose the 95th percentile of the doubt map, `tau` 0.0198 and 2 frames per hand-off, scoring 0.2362 at a 24.3% solver share against 0.2566 with no hand-off (`results/handoff_selection.json`).

### 3.4 Baselines and oracle

- **Fixed schedule:** hands off every so many frames, with one rollout per scene.
- **Random schedule:** hands off at each frame with a set chance, over 5 seeds.
- **Budget:** both baselines get the solver share that the frozen doubt threshold actually used on that set, and count as matched within 1 percentage point. Doubt "beats" a baseline only if the 95% interval of the paired difference lies wholly below zero, and the baseline's share is matched or above budget.
- **Oracle:** hands off using the true error of the current frame, at the same budget. Nobody could run this in practice. It judges one frame at a time, so it isn't a strict upper bound, and its figures are descriptive.

### 3.5 Measures

The per-frame error is the root mean square difference from the reference over fluid cells, divided by the reference's standard deviation, averaged over `u`, `v` and density. A set score is the mean over frames 21 to 70, with blown-up frames counted as 10 and never dropped. No rollout blew up in the final run.

Evidence for doubt is the within-time Spearman correlation: at one rollout time, across scenes, averaged over the 50 times. It stops a correlation from coming only from error and doubt both growing with time. Intervals come from resampling whole scenes, 2000 times.

### 3.6 Preregistration and its record

`docs/PREREGISTRATION.md` has 6 commits, and `git log` shows all of them. Four come before the data and the models (`a325999`, `6759a1d`, `8ec489c`, `d191984`), and those tighten the rules three times, each listed in its record. `31f57dc` came after the models and added the "Final run" section naming the code commit, before any test split opened. `6b547aa` came after the final run: it marked a stale row and added a dated note, with no rule changed. Nobody outside the project reviewed the file or holds a dated copy. So it's a visible record, not a guarantee.

## 4. Experiments

`scripts/run_experiments.py --final` ran once, from 19:56 on 2026-11-03 to 01:02 on 2026-11-04, and exited with 0, with all code committed. It ran four experiments: accuracy, doubt, hand-off and ablation. The results are stamped `31f57dc`, while the preregistration names `3a16f42`. The later commit changes no code (`docs/DECISIONS.md` B58).

Two things happened to test sets outside that run. A spot-check opened them before any model existed, to render fixed scenes and check their statistics, as the preregistration allows. The oracle reads each test set's reference by design. The test sets have now been seen, so they can't confirm any explanation or any new rule.

## 5. Results

### 5.1 Doubt-based hand-off against the baselines

| Set | Doubt | Solver share | Fixed | Doubt less fixed | Random | Doubt less random | Doubt beats either? |
|---|---|---|---|---|---|---|---|
| `test_a` | 0.245 | 19.9% | 0.251 | -0.006 [-0.019, 0.009] | 0.255 | -0.011 [-0.025, 0.005] | No |
| `test_b` | 0.288 | 13.9% | **0.265** | **0.023 [0.004, 0.041]** | 0.277 | 0.011 [-0.005, 0.025] | No. Worse than fixed |
| `test_c` low | 0.219 | 0.0% | not run | | not run | | No hand-off happened, so nothing to compare |
| `test_c` high | 0.410 | 92.2% | 0.419 at 66.1%, 0.411 at 99.0% | -0.009 [-0.022, 0.006], -0.001 [-0.014, 0.013] | 0.418 | -0.008 [-0.022, 0.007] | No |

Source: `results/exp_handoff.json`.

The hope failed on the headline set, and also failed on every other set. On `test_b`, the whole interval lies above zero, so this isn't only a failure to win. It is one comparison among several with no correction, and each fixed schedule is one rollout per scene, so its size is uncertain.

Validation had already given a warning. In the rehearsal before any test split opened, doubt less fixed was -0.0089 [-0.0230, 0.0049], which includes zero (`results/val_rehearsal/exp_handoff.json`). Nothing was changed because of it.

The threshold doesn't carry across sets. The same `tau` used a solver share of 24.3% on validation, 19.9% on `test_a`, 13.9% on `test_b`, 0.0% on the low side of `test_c` and 92.2% on the high side.

### 5.2 The oracle

| Set | Oracle | Solver share | Doubt less oracle | Fixed |
|---|---|---|---|---|
| `test_a` | 0.2446 | 19.5% | 0.000 [-0.006, 0.007] | 0.2507 |
| `test_b` | 0.2981 | 13.0% | -0.010 [-0.024, 0.002] | 0.2652 |
| `test_c` high | 0.3959 | 92.2% | 0.014 [0.002, 0.027] | 0.4190 at 66.1%, 0.4112 at 99.0% |

Source: `results/exp_handoff.json`. These figures are descriptive.

The oracle was no better than doubt on `test_a` and `test_b`, so the weak link isn't only the doubt signal. On `test_b` it also trailed the fixed schedule. One interpretation of that, not a tested finding, is that on `test_b` handing off when the current frame looks bad was a poor way to spend solver time. It doesn't carry to the other sets: on `test_a` the oracle is nominally ahead of fixed, and on the high side of `test_c` it scored lower than fixed, random (0.4179) and doubt (0.4100).

### 5.3 Does doubt predict error?

| Set | Doubt, within-time Spearman | No-ensemble divergence signal | Chance of higher doubt than `test_a` |
|---|---|---|---|
| `test_a` | 0.671 [0.455, 0.796] | 0.621 [0.378, 0.755] | |
| `test_b` | 0.522 [0.286, 0.673] | 0.676 [0.537, 0.755] | 0.479 [0.335, 0.615] |
| `test_c` low | 0.403 [0.001, 0.709] | 0.669 [0.055, 0.968] | 0.103 [0.033, 0.194] |
| `test_c` high | 0.388 [0.042, 0.633] | 0.174 [-0.105, 0.436] | 0.968 [0.923, 0.996] |

Source: `results/exp_doubt.json`.

Doubt does correlate with error at a given time, though the low-side interval almost touches zero, from 10 scenes. It doesn't warn of unfamiliarity. On unseen shapes the chance of higher doubt than on `test_a` is about a coin toss, even though the ensemble's error is higher there. On the low side of `test_c` the chance falls to 0.103 while the model is weak there, and on the high side it rises.

A divergence signal from one network, with no ensemble, scored higher than doubt on `test_b` and much lower on the high side of `test_c`. The two signals were scored on different rollouts, each against its own error. Their intervals overlap, and no paired interval was saved, so the difference isn't shown to be real. The signal was never used for hand-off.

### 5.4 Speed and accuracy

| Set | Ensemble | Coarse 128×128 solver | Solver only | ms per frame: ensemble, coarse, reference |
|---|---|---|---|---|
| `test_a` | **0.256** [0.220, 0.292] | 0.346 [0.319, 0.373] | 0.097 | 15.2, 178.8, 262.8 |
| `test_b` | **0.308** [0.272, 0.345] | 0.369 [0.343, 0.398] | 0.098 | 14.9, 180.2, 192.4 |
| `test_c` low | 0.219 [0.194, 0.240] | **0.152** [0.139, 0.166] | 0.018 | 14.9, 153.6, 194.4 |
| `test_c` high | 0.526 [0.495, 0.563] | 0.516 [0.488, 0.544] | 0.278 | 14.9, 183.0, 194.4 |

Source: `results/exp_accuracy.json`.

- **Speed.** With all 5 members counted, the ensemble is about 13 to 17 times faster than the reference, and about 13 to 15 times against the longer solver-only runs, so "about 14 times" is a fair single figure. The solver wasn't tuned for speed, and the reference timing comes from only 3 scenes of 20 frames each. A tuned solver could narrow the gap, but none was measured.
- **Accuracy against the coarse solver.** The ensemble is ahead on `test_a`. On `test_b` it is ahead in the primary window with intervals that touch, while at 100 frames the coarse solver leads on the mean, 0.503 against 0.529. On the low side of `test_c` the coarse solver is clearly better, and on the high side the ensemble isn't ahead. The hope was "for the same compute time", and that wasn't tested directly, since no coarse run was limited to 15 ms a frame.
- **Speed isn't free.** The solver alone scores far lower than the ensemble. That solver-only score is against a reference made by the same solver, so it shows the path isn't broken, not that the solver is correct.
- **Hand-off gives speed back.** Doubt took 82.9 ms per frame on `test_a` and 67.1 ms on `test_b`. On the high side of `test_c` it took 257.7 ms, slower than the reference's 194.4 ms, and scored 0.410 where a continuous solver run scored 0.278.

### 5.5 Ablations

Without the divergence penalty or without pushforward, scores were worse than a full member by 0.033 to 0.097 (`results/exp_ablation.json`). Each of those is one seed with a shorter 25-minute training, so this isn't only the effect of what was removed. More members help a little: on `test_b`, sizes 2, 3 and 5 score 0.327, 0.312 and 0.308. No ablation ran hand-off.

## 6. Discussion: why hand-off fell short

**This section is post-hoc and exploratory.** It was chosen after the results were seen, it uses the same test sets, its intervals aren't corrected for the comparisons tried, and confirming it needs new data. It ran no new rollouts and opened no test split. Every number is in `results/posthoc_analysis.json`.

- **Which scenes got the solver is the largest supported cause of the `test_b` loss.** 19 of 30 scenes got no hand-off in frames 21 to 70, and the top 5 scenes took 75% of solver frames. Estimated with doubt's scene shares but even spacing, the gap to fixed is -0.0231 [-0.0416, -0.0048], or -0.0154 [-0.0374, 0.0097] if the curve is extended in a straight line, against an actual gap of -0.0229.
- **Doubt behaves mostly like a property of the scene.** After removing time, the share of log-doubt variance that is between scenes is about 90%, and 83% on the low side of `test_c`. Scene doubt tracks largest flow travel, with Spearman 0.86 on `test_a` and `test_b` and 0.83 on the high side of `test_c`, though not on the low side.
- **Restarting has a real cost.** On the high side of `test_c`, a fixed schedule at 99% solver scored 0.411, and one continuous solver run scored 0.278.
- **Late hand-offs explain little.** This was the first guess after seeing the results, before the post-hoc analysis, and it isn't in the preregistration or the design. Doubt's hand-offs are later, at a mean start frame of 53.0 against 45.5, but start times alone predict +0.0013 [-0.0059, 0.0076] against the actual -0.0229.
- **Doubt's frame choice within a scene is unsettled.** Against nearby frames at lag 1, the gain is -0.0052 [-0.0119, -0.0003] on `test_a` over 19 scenes, a small nominal gain, and -0.0003 [-0.0014, 0.0008] on `test_b` over 14 scenes.
- **The oracle on `test_b` seems to act too late.** It has a median first hand-off at frame 55.5. Even spacing at its own scene shares would gain 0.040 to 0.042, and it actually gained 0.010.

Chaos as a cause of error staying after a hand-off can't be settled, because saved rollouts can't separate chaos from an existing error being carried forward. Whether a per-scene budget, a threshold scaled to the flow, or the divergence signal would beat fixed is unknown. Each needs new validation rollouts and new test scenes.

## 7. Limits

- **The reference is chaotic.** Past about `t=1`, it is one realisation of the flow, not pointwise truth, and the window runs from `t=0.8` to `t=2.8` (`results/solver_check.json`). Error partly measures how far two equally good solvers would drift, so it should be read with the secondary measures in `docs/RESULTS.md`.
- **The solver wasn't checked against experiment,** a published benchmark or a finer grid. It has numerical dissipation.
- **Small test sets.** Each has 30 scenes, and the sides of `test_c` have 10 and 20. The dataset was generated once.
- **Members can share mistakes.** On the low side of `test_c` the ensemble makes 23% too much smoke, yet its largest doubt is 0.009 against `tau` 0.0198, so nothing triggers.
- **2D and simulated.** Nothing here transfers to 3D or to real smoke.
- **One machine.** Times in milliseconds are single measurements on one laptop GPU.
- **Typed numbers.** The text in this report and the analysis in `docs/RESULTS.md` is typed, not generated, so it has to be checked again if a result file changes.
- **Open checks.** The last recorded full test run is 321 passed at commit `dd78b8d`, and later commits haven't had one. Release `v1.0` of the weights isn't published, so the README's demo command doesn't work from a fresh clone, and the clean-clone check hasn't been run.

## 8. Conclusion

Under rules fixed in advance, this threshold rule on this doubt signal didn't earn its solver time on small 2D smoke. It lost to a fixed schedule on unseen shapes and beat nothing on any other set. Doubt correlates with error at a given time, but it doesn't warn of unfamiliar shapes and isn't shown to be better than a simple divergence signal. The ensemble is a fast approximation of this project's own solver, and that holds separately from the hand-off result. The project can't claim that doubt-based hand-off works, or that the post-hoc reasons are proven. It also doesn't show that doubt-based hand-off can never work, since other signals, budgets per scene and other hand-off lengths weren't tested.

## PUBLIC TRACK PENDING

**This section holds no results.** `results/exp_well.json` doesn't exist yet, so nothing in this report says anything about the public dataset. The track will measure doubt, with 3 members, on `turbulent_radiative_layer_2D` from The Well [7], and the dataset card asks for a citation of Fielding et al. [8]. It is a published simulation benchmark, not physical measurement, and it has no hand-off because the project has no solver for that physics. The data licence is CC-BY-4.0, and the dataset is at https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D.

To be filled in from `results/exp_well.json` when it exists: within-time Spearman and AUROC of doubt on the official test split, the same for the no-ensemble signal, and whether doubt predicted error there.

## References

Checks on these sources, and their limits, are in `docs/REFERENCES.md`.

1. Pranav Lakshmanan and Paras Chopra. *Hybrid Neural World Models.* arXiv:2605.28317v1, 2026. Preprint.
2. Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart and Anima Anandkumar. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. arXiv:2010.08895.
3. Balaji Lakshminarayanan, Alexander Pritzel and Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NIPS 2017, pages 6402-6413. arXiv:1612.01474.
4. Johannes Brandstetter, Daniel Worrall and Max Welling. *Message Passing Neural PDE Solvers.* ICLR 2022. arXiv:2202.03376.
5. Dmitrii Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner and Stephan Hoyer. *Machine learning-accelerated computational fluid dynamics.* PNAS 118(21), e2101784118, 2021. DOI 10.1073/pnas.2101784118.
6. Jos Stam. *Stable Fluids.* Proceedings of SIGGRAPH '99, pages 121-128, 1999. DOI 10.1145/311535.311548.
7. Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel and others. *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning.* NeurIPS 2024, Datasets and Benchmarks Track, pages 44989-45037. DOI 10.52202/079017-1430. arXiv:2412.00568.
8. Drummond B. Fielding, Eve C. Ostriker, Greg L. Bryan and Adam S. Jermyn. *Multiphase Gas and the Fractal Nature of Radiative Turbulent Mixing Layers.* The Astrophysical Journal Letters 894(2), L24, 2020. DOI 10.3847/2041-8213/ab8d2c.
9. Rajyasri Roy, Dibyajyoti Nayak and Somdatta Goswami. *ANCHOR: Error-Controlled Adaptive Numerical Correction for Neural Operator Time Marching.* arXiv:2512.19643v2, 2026. Preprint.
10. Bharat Srikishan, Daniel O'Malley, Mohamed Mehana, Nicholas Lubbers and Nikhil Muralidhar. *Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout.* ICLR 2025. arXiv:2503.10048.
11. Olaf Ronneberger, Philipp Fischer and Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, pages 234-241. DOI 10.1007/978-3-319-24574-4_28.