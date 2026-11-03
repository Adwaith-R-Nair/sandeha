# Results

This file has two kinds of text. The block between the `BEGIN GENERATED RESULTS` and `END GENERATED RESULTS` markers is written by `scripts/render_results.py` from `results/exp_*.json`, and nobody edits it by hand. Everything outside the block is written analysis. Its numbers are copied from the file named beside them in `results/`.

Lower is better for every score. A set score is the mean per-frame error over frames 21 to 70, as defined in `PREREGISTRATION.md`. Intervals are 95% bootstrap intervals over whole scenes.

## The short answer

The main idea did not work. **Handing off to the solver when doubt was high did not beat a fixed schedule or a random one on any test set. On `test_b`, the set it was meant for, it was worse than the fixed schedule.**

- Doubt does track error across scenes at a given time, moderately on `test_a` and `test_b` and more weakly on `test_c`. It does not rise on unseen shapes, and it falls on the low side of `test_c`. On `test_b` a divergence signal with no ensemble scored higher, 0.676 [0.537, 0.755] against 0.522 [0.286, 0.673]. Those intervals overlap, and no paired interval was saved.
- The ensemble is about 13 to 17 times faster than the reference solver, with its full cost counted. It is more accurate than the coarse solver on `test_a` and, inside the primary window, on `test_b`. It is not more accurate on either side of `test_c`.
- An oracle that knew the true error was no better than doubt on `test_a` and `test_b`. So the weak link is not only the doubt signal. On `test_b` the oracle also trailed the fixed schedule, 0.2981 against 0.2652. One interpretation of those descriptive figures, not a tested finding, is that on `test_b` handing off when the current frame looks bad was a poor way to spend solver time. It doesn't carry to the other sets. On `test_a` the oracle is nominally ahead of fixed, 0.2446 against 0.2507. On the high side of `test_c` it scored 0.3959, lower than fixed (0.4112 and 0.4190), random (0.4179) and doubt (0.4100).
- The reasons given below come from analysis after the fact, on the same test sets. They are suggestions, not findings that have been confirmed.
- The public-dataset results do not exist yet. See "PUBLIC TRACK PENDING" at the end of this file.

## Design section 6.5, item by item

| Item | Kind | Did it hold? |
|---|---|---|
| All solver tests pass, and test sets stayed locked | Must hold | **Yes, with one check still to repeat.** The tests were not rerun on the last commit |
| Every figure regenerates from a command, and every number traces to a saved run | Must hold | **Yes, with limits** listed below |
| Speed and accuracy experiments completed and reported honestly | Must hold | **Yes** for the main track. The public track is pending |
| The AI beats the coarse 128×128 solver on accuracy, for the same compute time | Hope | **Partly.** Yes on `test_a`. Yes on `test_b` in the primary window, but not at longer range. No on `test_c` |
| The AI is meaningfully faster than the reference solver, ensemble cost included | Hope | **Yes** for the ensemble alone. Hand-off gives much of that back, and on the high side of `test_c` it is slower than the reference |
| With the threshold frozen on validation, doubt-based hand-off beats fixed and random schedules at the same solver budget on Test B | Hope | **No.** It lost to the fixed schedule, and did not beat random |

### Must hold: solver tests and locked test sets

**Solver tests.** The last recorded full run is 321 passed at commit `dd78b8d` (`BUILD_LOG.md`). `src/sandeha/physics/solver.py` has not changed since `2b2df8a`, before that run. The solver tests were not rerun for this write-up, because they need the GPU and the GPU is in use for public-track training. Run on the CPU they stop with "No CUDA GPUs are available", which says nothing about the solver. They must be run once more when the GPU is free.

**Locked test sets.** The hand-off choice was made on validation and committed in `3a16f42`, before any test split opened. `--final` ran once, from 19:56 on 2026-11-03 to 01:02 on 2026-11-04, and exited with 0 (`runs/jobs.log`, which isn't tracked). No rollout was repeated afterwards, and the post-hoc work opened no test split. Three things are the known exceptions or limits:

- The data spot-check opened test splits before any model existed, as the preregistration allows. `results/spot_check.json` and `results/travel_rule.json` both record `git_dirty: true`, so those two checks ran with uncommitted changes in the tree.
- The oracle reads each test set's reference by design. Nothing else is chosen from it.
- The test sets have now been seen. They can't confirm any explanation below, or any new rule.

The results are stamped `31f57dc`, while `PREREGISTRATION.md` names `3a16f42`. The commit between them changes only `docs/BUILD_LOG.md`, `docs/PREREGISTRATION.md` and `results/val_rehearsal/`, so the code is the same. `DECISIONS.md` B58 records this.

### Must hold: regeneration and tracing

These were rerun on the CPU for this write-up, into a scratch folder:

- `scripts/render_results.py` gave a file identical to this one's generated block.
- `scripts/make_figures.py` produced its 8 figures.
- `scripts/posthoc_analysis.py` gave a `posthoc_analysis.json` identical to the committed one, apart from the commit it records.

The limits are these:

- The solver, scene, travel and spot-check figures also come from commands, but those need the GPU or the raw data and weren't rerun for this write-up.
- The written numbers here are typed, not generated. If a result file changes, this text has to be checked again.
- Times in milliseconds are single measurements on one laptop GPU. They would change on another run.

### Must hold: experiments completed

All four experiments finished, and all are in the generated block, weak results included. No rollout blew up in any policy on any set. No hand-off was forced, so every solver frame in the doubt, fixed and random runs was planned by its own rule.

### Hope: the AI beats the coarse solver on accuracy, for the same compute time

**Partly held.**

| Set | Ensemble score | Coarse score | ms per frame, ensemble and coarse | Vorticity correlation, ensemble and coarse | Usable frames, ensemble and coarse | Mean error over 100 frames, ensemble and coarse |
|---|---|---|---|---|---|---|
| `test_a` | **0.256** [0.220, 0.292] | 0.346 [0.319, 0.373] | 15.2, 178.8 | 0.859, 0.806 | 95.2, 98.7 | 0.483, 0.501 |
| `test_b` | **0.308** [0.272, 0.345] | 0.369 [0.343, 0.398] | 14.9, 180.2 | 0.799, 0.776 | 95.1, 100.0 | 0.529, 0.503 |
| `test_c` low | 0.219 [0.194, 0.240] | **0.152** [0.139, 0.166] | 14.9, 153.6 | 0.943, 0.967 | 99.6, 100.0 | 0.384, 0.243 |
| `test_c` high | 0.526 [0.495, 0.563] | 0.516 [0.488, 0.544] | 14.9, 183.0 | 0.581, 0.613 | 81.0, 92.4 | 0.704, 0.631 |

Source: `exp_accuracy.json`.

- **`test_a`:** the ensemble is ahead in the primary window, and the two intervals don't overlap. At 100 frames the two are close, and the intervals overlap.
- **`test_b`:** the ensemble is ahead in the primary window, but the two intervals touch, at 0.343 to 0.345. No paired interval for this comparison was saved. At 100 frames the coarse solver is ahead on the mean, with overlapping intervals, and it has more usable frames. So the lead is real only over the shorter horizon that was fixed in advance.
- **`test_c` low:** the coarse solver is clearly more accurate. The ensemble also makes too much smoke here, with a total smoke ratio of 1.226 against the reference in the primary window. These plumes reach obstacles late, and 1 of the 10 scenes has no contact by frame 70 on the coarse rule, so this side says more about an unfamiliar flow than about flow around obstacles.
- **`test_c` high:** the ensemble's score is slightly worse, and the intervals overlap heavily. This isn't a win for either, and the coarse solver leads on the secondary measures in the table.
- **Secondary measures aren't one-sided.** The coarse solver loses far more of the finest scales: in the highest spectrum band on `test_a`, its log-ratio is -1.299 and the ensemble's is -0.099. But it holds the plume top better there (-0.017 against 0.192) and its RMS divergence is lower (0.00132 against 0.00188). Its largest divergence is slightly higher (0.0433 against 0.0420).
- **"For the same compute time" was not tested directly.** No coarse run was limited to 15 ms a frame, and the coarse solver here takes about 10 to 12 times as long as the ensemble. So on `test_a` and `test_b` the ensemble won with far less time than the hope required, and on `test_c` nothing shows who would win at equal time. A coarse solver with a looser pressure tolerance might be cheaper and only slightly less accurate. That was not tried.
- **What these errors mean.** From about `t=1` the reference is one realisation of a chaotic flow, not pointwise truth (`PREREGISTRATION.md`). An error near the coarse solver's level doesn't by itself mean the prediction is physically wrong, and the coarse score is reported to give that level a figure.

### Hope: the AI is meaningfully faster than the reference solver

**Held for the ensemble alone.**

| Set | Reference ms per frame | Ensemble ms per frame | Times faster | Solver-only run, ms per frame |
|---|---|---|---|---|
| `test_a` | 262.8 | 15.2 | about 17 | 217.6 |
| `test_b` | 192.4 | 14.9 | about 13 | 228.7 |
| `test_c` | 194.4 | 14.9 | about 13 | 189.1 low, 223.5 high |

Source: `exp_accuracy.json`. The ensemble's time includes all 5 members. A single member takes 3.6 ms.

- The reference time comes from only 3 scenes of 20 frames each, and it varies a lot between sets. Against the solver-only runs, which are longer and cover every scene, the ratio is about 13 to 15. So "about 14 times" is a fair single figure, and the table shows why it isn't exact.
- The solver is this project's own PyTorch code with a tight pressure tolerance. It wasn't tuned for speed, and a tuned solver could narrow the gap, but none was measured. The ratio holds only for this solver on this laptop GPU.
- Speed isn't free. The solver-only run scores 0.097 on `test_a` and 0.098 on `test_b`. The ensemble scores 0.256 and 0.308.
- Hand-off gives speed back. The doubt policy took 82.9 ms on `test_a` and 67.1 ms on `test_b`, about 3 times faster than the reference. On the high side of `test_c` it took 257.7 ms, slower than the reference's 194.4 ms, and scored 0.410 where the solver-only run scored 0.278.
- The demo ran in AI mode at 61.3 frames per second with all 5 members, against a target of 15 (`demo_bench.json`). No solver frames were included in that run.

### Hope: doubt-based hand-off beats fixed and random schedules on Test B

**Did not hold.** The rule was that doubt wins only if the whole interval of its difference lies below zero.

| Set | Doubt | Fixed | Doubt less fixed | Random | Doubt less random | Doubt beats either? |
|---|---|---|---|---|---|---|
| `test_a` | 0.245 | 0.251 | -0.006 [-0.019, 0.009] | 0.255 | -0.011 [-0.025, 0.005] | No |
| `test_b` | 0.288 | **0.265** | **0.023 [0.004, 0.041]** | 0.277 | 0.011 [-0.005, 0.025] | No. Worse than fixed |
| `test_c` low | 0.219 | not run | | not run | | No hand-off happened, so there's nothing to compare |
| `test_c` high | 0.410 | 0.419 at 66.1%, 0.411 at 99.0% | -0.009 [-0.022, 0.006], -0.001 [-0.014, 0.013] | 0.418 | -0.008 [-0.022, 0.007] | No |

Source: `exp_handoff.json`. To four places on `test_b`, doubt is 0.2881 and fixed is 0.2652, a difference of 0.0229. Pooled `test_c` figures are in the generated block.

- The whole interval on `test_b` lies above zero, so this isn't only a failure to win. It's one comparison among several with no correction, and each fixed schedule is one rollout per scene, so its size is uncertain. Reading the fixed gain off the other fixed periods gives 0.037 against the 0.0429 measured, which would narrow the gap to about 0.017. Fixed also beat random on `test_b`, by 0.0119 [0.0027, 0.0216].
- **Warning on validation.** Before any test split opened, the rehearsal showed doubt at 0.2362, fixed at 0.2451 and random at 0.2520 (`results/val_rehearsal/exp_handoff.json`). Doubt beat random, but its difference from fixed, -0.0089 [-0.0230, 0.0049], included zero. No code changed because of this.
- **The oracle.** Doubt less oracle is 0.000 [-0.006, 0.007] on `test_a` and -0.010 [-0.024, 0.002] on `test_b`, where the oracle's share was 13.0% against doubt's 13.9%. On the high side of `test_c` the oracle was ahead, by 0.014 [0.002, 0.027]. These are descriptive, and the oracle judges one frame at a time, so a better schedule could exist.
- **The threshold doesn't carry across sets.** The same `tau` used a solver share of 24.3% on validation, 19.9% on `test_a`, 13.9% on `test_b`, 0.0% on the low side of `test_c` and 92.2% on the high side.
- **The further signal the design allows.** The design allows one further signal, chosen on validation only. None was tried. See `DECISIONS.md` B60.

### Speed against accuracy

This is the second half of the project's question. Each row has a real cost.

| Set | Policy | Set score | ms per frame | Solver share |
|---|---|---|---|---|
| `test_a` | ensemble | 0.256 | 15.2 | 0.0% |
| `test_a` | fixed | 0.251 | 62.4 | 19.8% |
| `test_a` | doubt | 0.245 | 82.9 | 19.9% |
| `test_a` | solver only | 0.097 | 217.6 | 100% |
| `test_b` | ensemble | 0.308 | 14.9 | 0.0% |
| `test_b` | fixed | 0.265 | 50.0 | 14.2% |
| `test_b` | doubt | 0.288 | 67.1 | 13.9% |
| `test_b` | solver only | 0.098 | 228.7 | 100% |

Source: `exp_accuracy.json`, `exp_handoff.json`. The milliseconds cover frames 21 to 120, while the score and share cover frames 21 to 70. Item 4 under "Explanations the data supports" shows why that matters for doubt.

On `test_a`, about 20% solver time bought little: a gain of 0.0055 [-0.007, 0.020] under the fixed schedule, which includes zero, and 0.0115 [0.005, 0.018] under doubt. On `test_b`, about 14% bought 0.0429 [0.030, 0.057] under the fixed schedule and 0.0200 [0.007, 0.036] under doubt. Restarts are part of why short hand-offs buy so little (item 3 below). These gains are from `posthoc_analysis.json`.

<!-- BEGIN GENERATED RESULTS -->

## Provenance

| File | Commit | Changed tree | Data manifest | Checkpoints | Final |
|---|---|---|---|---|---|
| `exp_accuracy.json` | `31f57dc3eb2d` | no | `ac363e1f582a` | 5 | yes |
| `exp_doubt.json` | `31f57dc3eb2d` | no | `ac363e1f582a` | 5 | yes |
| `exp_handoff.json` | `31f57dc3eb2d` | no | `ac363e1f582a` | 5 | yes |
| `exp_ablation.json` | `31f57dc3eb2d` | no | `ac363e1f582a` | 5 | yes |

## Accuracy

Reference solver on `test_a`: 262.8 ms per frame.
Reference solver on `test_b`: 192.4 ms per frame.
Reference solver on `test_c`: 194.4 ms per frame.

| Set | Policy | Set score | 95% interval | Blown up | Usable frames | Vorticity corr. | ms/frame |
|---|---|---|---|---|---|---|---|
| test_a | ensemble | 0.256 | [0.220, 0.292] | 0.0% | 95.2 | 0.859 | 15.2 |
| test_a | single_model | 0.292 | [0.257, 0.329] | 0.0% | 97.3 | 0.826 | 3.6 |
| test_a | persistence | 0.946 | [0.935, 0.957] | 0.0% | 21.2 | 0.291 | 0.0 |
| test_a | coarse_solver | 0.346 | [0.319, 0.373] | 0.0% | 98.7 | 0.806 | 178.8 |
| test_a | solver_only | 0.097 | [0.076, 0.120] | 0.0% | 99.9 | 0.974 | 217.6 |
| test_b | ensemble | 0.308 | [0.272, 0.345] | 0.0% | 95.1 | 0.799 | 14.9 |
| test_b | single_model | 0.358 | [0.323, 0.393] | 0.0% | 91.9 | 0.758 | 3.6 |
| test_b | persistence | 0.932 | [0.919, 0.945] | 0.0% | 30.6 | 0.295 | 0.0 |
| test_b | coarse_solver | 0.369 | [0.343, 0.398] | 0.0% | 100.0 | 0.776 | 180.2 |
| test_b | solver_only | 0.098 | [0.079, 0.118] | 0.0% | 100.0 | 0.968 | 228.7 |
| test_c low | ensemble | 0.219 | [0.194, 0.240] | 0.0% | 99.6 | 0.943 | 14.9 |
| test_c low | single_model | 0.343 | [0.286, 0.407] | 0.0% | 95.0 | 0.834 | 3.6 |
| test_c low | persistence | 0.838 | [0.827, 0.849] | 0.0% | 95.0 | 0.414 | 0.0 |
| test_c low | coarse_solver | 0.152 | [0.139, 0.166] | 0.0% | 100.0 | 0.967 | 153.6 |
| test_c low | solver_only | 0.018 | [0.016, 0.021] | 0.0% | 100.0 | 1.000 | 189.1 |
| test_c high | ensemble | 0.526 | [0.495, 0.563] | 0.0% | 81.0 | 0.581 | 14.9 |
| test_c high | single_model | 0.561 | [0.530, 0.594] | 0.0% | 77.5 | 0.558 | 3.6 |
| test_c high | persistence | 0.973 | [0.963, 0.982] | 0.0% | 18.8 | 0.210 | 0.0 |
| test_c high | coarse_solver | 0.516 | [0.488, 0.544] | 0.0% | 92.4 | 0.613 | 183.0 |
| test_c high | solver_only | 0.278 | [0.247, 0.309] | 0.0% | 99.6 | 0.847 | 223.5 |
| test_c pooled | ensemble | 0.424 | [0.366, 0.480] | 0.0% | 87.2 | 0.702 | 14.9 |
| test_c pooled | single_model | 0.489 | [0.439, 0.537] | 0.0% | 83.3 | 0.650 | 3.6 |
| test_c pooled | persistence | 0.928 | [0.903, 0.951] | 0.0% | 44.2 | 0.278 | 0.0 |
| test_c pooled | coarse_solver | 0.395 | [0.329, 0.459] | 0.0% | 94.9 | 0.731 | 173.2 |
| test_c pooled | solver_only | 0.191 | [0.144, 0.239] | 0.0% | 99.7 | 0.898 | 212.0 |

Scenes by when a late obstacle changes the solid:

| Set | Inside the window | After it | None |
|---|---|---|---|
| test_a | 8 | 2 | 20 |
| test_b | 0 | 0 | 30 |
| test_c low | 0 | 0 | 10 |
| test_c high | 0 | 0 | 20 |
| test_c pooled | 0 | 0 | 30 |

Scenes with no contact by the window's end, by the coarse rule on the stored reference:

| Set | No contact | Of |
|---|---|---|
| test_c low | 1 | 10 |
| test_c high | 0 | 20 |
| test_c pooled | 1 | 30 |

## Doubt

Only the within-time Spearman counts as evidence. Pooled figures do not by themselves.

### Frames 21 to 70

| Set | Signal | Within-time Spearman | 95% interval | Undefined times | Non-finite frames | Within-time AUROC | Times excluded | Pooled Spearman |
|---|---|---|---|---|---|---|---|---|
| test_a | mean | 0.667 | [0.435, 0.795] | 0 | 0 | 0.842 | 30 | 0.720 |
| test_a | p95 (chosen) | 0.671 | [0.455, 0.796] | 0 | 0 | 0.860 | 30 | 0.695 |
| test_a | p99 | 0.701 | [0.526, 0.807] | 0 | 0 | 0.885 | 30 | 0.643 |
| test_a | no ensemble | 0.621 | [0.378, 0.755] | 0 | 0 | 0.817 | 27 | 0.901 |
| test_a | independent (secondary) | 0.686 | [0.472, 0.800] | 0 | 0 | 0.834 | 32 | 0.930 |
| test_b | mean | 0.492 | [0.258, 0.645] | 0 | 0 | 0.718 | 26 | 0.660 |
| test_b | p95 (chosen) | 0.522 | [0.286, 0.673] | 0 | 0 | 0.757 | 26 | 0.647 |
| test_b | p99 | 0.603 | [0.396, 0.737] | 0 | 0 | 0.743 | 26 | 0.471 |
| test_b | no ensemble | 0.676 | [0.537, 0.755] | 0 | 0 | 0.867 | 34 | 0.927 |
| test_b | independent (secondary) | 0.598 | [0.384, 0.734] | 0 | 0 | 0.786 | 27 | 0.930 |
| test_c low | mean | 0.458 | [0.123, 0.738] | 0 | 0 | n/a | 50 | 0.753 |
| test_c low | p95 (chosen) | 0.403 | [0.001, 0.709] | 0 | 0 | n/a | 50 | 0.826 |
| test_c low | p99 | 0.445 | [-0.043, 0.790] | 0 | 0 | n/a | 50 | 0.364 |
| test_c low | no ensemble | 0.669 | [0.055, 0.968] | 0 | 0 | 0.840 | 48 | 0.895 |
| test_c low | independent (secondary) | 0.346 | [-0.131, 0.718] | 0 | 0 | n/a | 50 | 0.906 |
| test_c high | mean | 0.351 | [0.029, 0.598] | 0 | 0 | 0.687 | 44 | 0.577 |
| test_c high | p95 (chosen) | 0.388 | [0.042, 0.633] | 0 | 0 | 0.733 | 44 | 0.528 |
| test_c high | p99 | 0.400 | [0.061, 0.644] | 0 | 0 | 0.757 | 44 | 0.464 |
| test_c high | no ensemble | 0.174 | [-0.105, 0.436] | 0 | 0 | 0.780 | 44 | 0.799 |
| test_c high | independent (secondary) | 0.456 | [0.136, 0.669] | 0 | 0 | 0.724 | 40 | 0.870 |
| test_c pooled | mean | 0.759 | [0.570, 0.863] | 0 | 0 | 0.964 | 14 | 0.737 |
| test_c pooled | p95 (chosen) | 0.769 | [0.582, 0.865] | 0 | 0 | 0.970 | 14 | 0.721 |
| test_c pooled | p99 | 0.778 | [0.607, 0.872] | 0 | 0 | 0.977 | 14 | 0.670 |
| test_c pooled | no ensemble | 0.606 | [0.364, 0.768] | 0 | 0 | 0.936 | 24 | 0.797 |
| test_c pooled | independent (secondary) | 0.787 | [0.610, 0.872] | 0 | 0 | 0.955 | 14 | 0.929 |

### Frames 21 to 120 (secondary, not evidence)

| Set | Signal | Within-time Spearman | 95% interval | Undefined times | Non-finite frames | Within-time AUROC | Times excluded | Pooled Spearman |
|---|---|---|---|---|---|---|---|---|
| test_a | mean | 0.533 | [0.303, 0.674] | 0 | 0 | 0.869 | 70 | 0.649 |
| test_a | p95 (chosen) | 0.539 | [0.330, 0.681] | 0 | 0 | 0.882 | 70 | 0.560 |
| test_a | p99 | 0.588 | [0.411, 0.700] | 0 | 0 | 0.905 | 70 | 0.498 |
| test_a | no ensemble | 0.606 | [0.397, 0.723] | 0 | 0 | 0.831 | 74 | 0.887 |
| test_a | independent (secondary) | 0.569 | [0.344, 0.710] | 0 | 0 | 0.833 | 52 | 0.852 |
| test_b | mean | 0.260 | [0.016, 0.446] | 0 | 0 | 0.719 | 74 | 0.580 |
| test_b | p95 (chosen) | 0.271 | [0.037, 0.466] | 0 | 0 | 0.758 | 74 | 0.485 |
| test_b | p99 | 0.344 | [0.122, 0.521] | 0 | 0 | 0.744 | 74 | 0.329 |
| test_b | no ensemble | 0.535 | [0.353, 0.652] | 0 | 0 | 0.867 | 84 | 0.853 |
| test_b | independent (secondary) | 0.424 | [0.196, 0.601] | 0 | 0 | 0.800 | 72 | 0.807 |
| test_c low | mean | 0.089 | [-0.150, 0.399] | 0 | 0 | 0.180 | 96 | 0.761 |
| test_c low | p95 (chosen) | 0.225 | [-0.137, 0.559] | 0 | 0 | 0.310 | 96 | 0.681 |
| test_c low | p99 | 0.301 | [-0.085, 0.645] | 0 | 0 | 0.460 | 96 | 0.339 |
| test_c low | no ensemble | 0.560 | [-0.081, 0.915] | 0 | 0 | 0.840 | 98 | 0.907 |
| test_c low | independent (secondary) | 0.149 | [-0.256, 0.550] | 0 | 0 | 0.280 | 94 | 0.883 |
| test_c high | mean | 0.222 | [-0.004, 0.447] | 0 | 0 | 0.687 | 94 | 0.482 |
| test_c high | p95 (chosen) | 0.179 | [-0.106, 0.438] | 0 | 0 | 0.733 | 94 | 0.316 |
| test_c high | p99 | 0.240 | [-0.055, 0.487] | 0 | 0 | 0.757 | 94 | 0.273 |
| test_c high | no ensemble | 0.116 | [-0.115, 0.348] | 0 | 0 | 0.780 | 94 | 0.730 |
| test_c high | independent (secondary) | 0.519 | [0.305, 0.669] | 0 | 0 | 0.724 | 90 | 0.823 |
| test_c pooled | mean | 0.677 | [0.496, 0.770] | 0 | 0 | 0.947 | 53 | 0.684 |
| test_c pooled | p95 (chosen) | 0.669 | [0.459, 0.783] | 0 | 0 | 0.953 | 53 | 0.613 |
| test_c pooled | p99 | 0.695 | [0.497, 0.801] | 0 | 0 | 0.964 | 53 | 0.572 |
| test_c pooled | no ensemble | 0.553 | [0.300, 0.727] | 0 | 0 | 0.936 | 74 | 0.771 |
| test_c pooled | independent (secondary) | 0.764 | [0.645, 0.827] | 0 | 0 | 0.936 | 49 | 0.882 |

### Rise on unfamiliar sets, against Test A

| Set | Chance of higher doubt | 95% interval |
|---|---|---|
| test_b | 0.479 | [0.335, 0.615] |
| test_c low | 0.103 | [0.033, 0.194] |
| test_c high | 0.968 | [0.923, 0.996] |
| test_c pooled | 0.680 | [0.526, 0.822] |

## Hand-off

Frozen on validation: kind `threshold`, aggregation `p95`, `handoff_frames` 2, `tau` 0.019795792549848555.

| Set | Policy | Set score | Solver share | Match | Forced | Blown up | Doubt less baseline | 95% interval | Doubt beats it |
|---|---|---|---|---|---|---|---|---|---|
| test_a | doubt | 0.245 | 19.9% | budget | 0 | 0.0% |  |  |  |
| test_a | fixed (nearest) | 0.251 | 19.8% | matched | 0 | 0.0% | -0.006 | [-0.019, 0.009] | no |
| test_a | random (nearest) | 0.255 | 19.9% | matched | 0 | 0.0% | -0.011 | [-0.025, 0.005] | no |
| test_b | doubt | 0.288 | 13.9% | budget | 0 | 0.0% |  |  |  |
| test_b | fixed (nearest) | 0.265 | 14.2% | matched | 0 | 0.0% | 0.023 | [0.004, 0.041] | no |
| test_b | random (nearest) | 0.277 | 13.9% | matched | 0 | 0.0% | 0.011 | [-0.005, 0.025] | no |
| test_c low | doubt | 0.219 | 0.0% | budget | 0 | 0.0% |  |  |  |
| test_c high | doubt | 0.410 | 92.2% | budget | 0 | 0.0% |  |  |  |
| test_c high | fixed (below) | 0.419 | 66.1% | below | 0 | 0.0% | -0.009 | [-0.022, 0.006] | no |
| test_c high | fixed (above) | 0.411 | 99.0% | above | 0 | 0.0% | -0.001 | [-0.014, 0.013] | no |
| test_c high | random (nearest) | 0.418 | 92.2% | matched | 0 | 0.0% | -0.008 | [-0.022, 0.007] | no |
| test_c pooled | doubt | 0.346 | 61.5% | budget | 0 | 0.0% |  |  |  |
| test_c pooled | fixed (above) | 0.347 | 66.0% | above | 0 | 0.0% | -0.001 | [-0.010, 0.009] | no |
| test_c pooled | fixed (below) | 0.352 | 44.1% | below | 0 | 0.0% | -0.006 | [-0.016, 0.004] | no |
| test_c pooled | random (nearest) | 0.352 | 83.8% | above | 0 | 0.0% | -0.005 | [-0.015, 0.004] | no |

- `test_c low`: A budget of 0 means there is no hand-off to compare.
- `test_c low`: A budget of 0 means the oracle isn't run.

### Oracle bound (descriptive)

| Set | Set score | Solver share | Doubt less oracle | 95% interval | Unmatched |
|---|---|---|---|---|---|
| test_a | 0.245 | 19.5% | 0.000 | [-0.006, 0.007] | no |
| test_b | 0.298 | 13.0% | -0.010 | [-0.024, 0.002] | no |
| test_c high | 0.396 | 92.2% | 0.014 | [0.002, 0.027] | no |
| test_c pooled | 0.337 | 61.5% | 0.009 | [0.001, 0.019] | no |

## Ablations

Each reduced training is one seed trained for less time than the ensemble's members, and the plan names no baseline with that shorter time, so its comparison with the full member isn't only the effect of what was removed.
Sizes come from the lowest-seed members. Doubt is a population spread, so its absolute scale differs by size, and only rankings are compared.

| Set | Run | Set score | 95% interval | Blown up | Within-time Spearman |
|---|---|---|---|---|---|
| test_a | full member | 0.292 | [0.257, 0.329] | 0.0% |  |
| test_a | no_divergence | 0.336 | [0.303, 0.370] | 0.0% |  |
| test_a | no_pushforward | 0.335 | [0.300, 0.370] | 0.0% |  |
| test_a | 2 members | 0.268 | [0.232, 0.304] | 0.0% | 0.618 |
| test_a | 3 members | 0.259 | [0.223, 0.295] | 0.0% | 0.642 |
| test_a | 5 members | 0.256 | [0.220, 0.292] | 0.0% | 0.671 |
| test_b | full member | 0.358 | [0.323, 0.393] | 0.0% |  |
| test_b | no_divergence | 0.392 | [0.351, 0.437] | 0.0% |  |
| test_b | no_pushforward | 0.408 | [0.369, 0.450] | 0.0% |  |
| test_b | 2 members | 0.327 | [0.295, 0.361] | 0.0% | 0.465 |
| test_b | 3 members | 0.312 | [0.279, 0.344] | 0.0% | 0.511 |
| test_b | 5 members | 0.308 | [0.272, 0.345] | 0.0% | 0.522 |
| test_c low | full member | 0.343 | [0.286, 0.407] | 0.0% |  |
| test_c low | no_divergence | 0.440 | [0.362, 0.533] | 0.0% |  |
| test_c low | no_pushforward | 0.422 | [0.348, 0.501] | 0.0% |  |
| test_c low | 2 members | 0.257 | [0.221, 0.288] | 0.0% | 0.348 |
| test_c low | 3 members | 0.244 | [0.212, 0.274] | 0.0% | 0.292 |
| test_c low | 5 members | 0.219 | [0.194, 0.240] | 0.0% | 0.403 |
| test_c high | full member | 0.561 | [0.530, 0.594] | 0.0% |  |
| test_c high | no_divergence | 0.594 | [0.565, 0.624] | 0.0% |  |
| test_c high | no_pushforward | 0.602 | [0.570, 0.634] | 0.0% |  |
| test_c high | 2 members | 0.530 | [0.502, 0.562] | 0.0% | 0.346 |
| test_c high | 3 members | 0.524 | [0.494, 0.557] | 0.0% | 0.376 |
| test_c high | 5 members | 0.526 | [0.495, 0.563] | 0.0% | 0.388 |
| test_c pooled | full member | 0.489 | [0.439, 0.537] | 0.0% |  |
| test_c pooled | no_divergence | 0.543 | [0.498, 0.587] | 0.0% |  |
| test_c pooled | no_pushforward | 0.542 | [0.494, 0.588] | 0.0% |  |
| test_c pooled | 2 members | 0.439 | [0.384, 0.491] | 0.0% | 0.725 |
| test_c pooled | 3 members | 0.430 | [0.376, 0.484] | 0.0% | 0.733 |
| test_c pooled | 5 members | 0.424 | [0.366, 0.480] | 0.0% | 0.769 |

<!-- END GENERATED RESULTS -->
## Does doubt predict error?

This is the first half of the project's question. The figures are for the chosen signal, the 95th percentile of the doubt map, in frames 21 to 70.

| Set | Within-time Spearman | Within-time AUROC | Times left out of AUROC, of 50 | Chance of higher doubt than `test_a` | No-ensemble Spearman |
|---|---|---|---|---|---|
| `test_a` | 0.671 [0.455, 0.796] | 0.860 | 30 | | 0.621 [0.378, 0.755] |
| `test_b` | 0.522 [0.286, 0.673] | 0.757 | 26 | 0.479 [0.335, 0.615] | 0.676 [0.537, 0.755] |
| `test_c` low | 0.403 [0.001, 0.709] | none | 50 | 0.103 [0.033, 0.194] | 0.669 [0.055, 0.968] |
| `test_c` high | 0.388 [0.042, 0.633] | 0.733 | 44 | 0.968 [0.923, 0.996] | 0.174 [-0.105, 0.436] |

Source: `exp_doubt.json`.

- **Doubt does correlate with error at a given time.** That part held. But the interval on the low side of `test_c` almost touches zero, with only 10 scenes.
- **Doubt did not rise on unseen shapes.** On `test_b` the chance of higher doubt is 0.479, and its interval includes one half. The ensemble's error was higher there, 0.308 against 0.256. So doubt didn't warn of this kind of unfamiliarity.
- **On the low side of `test_c`, doubt fell where the model was weak.** The chance of higher doubt is 0.103, while the ensemble there was less accurate than the coarse solver. **On the high side it rose**, at 0.968. So its level doesn't line up with how unfamiliar a flow is. Post-hoc, it tracks largest flow travel (item 2 below).
- **A simple signal with no ensemble did as well or better on some sets.** On `test_b` the divergence of a single model's output scored 0.676 against doubt's 0.522. It scored much lower on the high side of `test_c`, 0.174 against 0.388. No paired interval between the two signals was saved, and their intervals overlap on every set. It was never used for hand-off, so nothing shows it would do better there.
- **The AUROC figures rest on a part of the window.** Many times had too few bad or too few good frames. On the low side of `test_c`, only 4 times had any of both, and none had enough, so there's no figure.
- **The pooled `test_c` figure of 0.769 isn't evidence about either side.** It's high largely because doubt separates the two sides from each other.

## The ablations and the ensemble

- **The ensemble is more accurate than one member on every set:** 0.256 against 0.292 on `test_a`, 0.308 against 0.358 on `test_b`, 0.219 against 0.343 on the low side of `test_c`, and 0.526 against 0.561 on the high side, for about 4 times the cost.
- **More members help a little.** On `test_b`, sizes 2, 3 and 5 score 0.327, 0.312 and 0.308, with within-time Spearman 0.465, 0.511 and 0.522.
- **Reduced trainings score worse, by 0.033 to 0.097.** Each is one seed with a shorter training than the full member, so this isn't only the effect of what was removed.
- **No ablation ran hand-off,** so they say nothing about why hand-off failed.

## Why hand-off fell short: post-hoc analysis

**POST-HOC AND EXPLORATORY. None of this was preregistered, and all of it was chosen after the results were seen.** It ran no new rollouts, opened no test split and loaded no model. The explanations come from the same test sets that showed the results. The intervals aren't corrected for the number of comparisons tried. Confirming any of it needs new data. No headline result above changes because of it.

Every number here is in `results/posthoc_analysis.json`, from `scripts/posthoc_analysis.py`. "Gain" means the no-hand-off ensemble's score less the policy's score, so higher is better.

**Checks first.** The saved rollouts reproduce every reported set score. Across 1390 rollouts, the error before the first hand-off differs from the no-hand-off ensemble by exactly 0.0, so the ensemble is an exact comparison up to that point. The frozen policy's first hand-off is always the first frame over the threshold.

**Main conclusion.** The first guess after seeing the results, before the post-hoc analysis, was that the solver can't bring a chaotic flow back, and that doubt fires too late to help. No record from before the results states this guess: it isn't in the preregistration or the design. As worded, that is mostly not what the data shows. Its premise, that error rarely comes back, mostly holds. Its proposed cause of the `test_b` loss, late hand-offs, explains little. The largest supported cause is which scenes got the solver.

### Explanations the data supports

1. **Doubt concentrated solver time in a few scenes on `test_b`.** 19 of 30 scenes got no hand-off in frames 21 to 70, and the top 5 scenes took 75% of solver frames. 75% of the fixed schedule's gain came from those 19 scenes. Estimated with doubt's per-scene shares but even spacing, the gap to fixed is -0.0231 [-0.0416, -0.0048]. If shares above the largest schedule are extended in a straight line, it is -0.0154 [-0.0374, 0.0097], which includes zero. The actual gap in gain is -0.0229, so this accounts for roughly two-thirds to all of it. These are estimates from noisy single rollouts.
2. **Doubt behaves mostly like a property of the scene, not a moment-by-moment alarm.** On rollouts with no hand-off, the chance of staying over the threshold is 0.98 to 0.99 on `test_a` and `test_b`, and the chance of crossing it from below is 0.01 to 0.016. After removing time, the share of log-doubt variance that is between scenes is about 90%, and 83% on the low side of `test_c`. Scene doubt tracks largest flow travel, with Spearman 0.86 on `test_a`, 0.86 on `test_b` and 0.83 on the high side of `test_c`, though not on the low side, at -0.079 [-0.765, 0.713] from 10 scenes. In the frozen rollouts, 74 to 76% of the gaps between hand-offs are immediately consecutive.
3. **Restarting the solver has a real cost.** On the high side of `test_c`, the fixed schedule with 99% solver scored 0.411 and one continuous solver run scored 0.278. These are nearly the same solver frames, but the first is restarted from the coarse state every 2 frames. On `test_a`, a hand-off starting at frames 21 to 25 raises error by 0.010 [0.007, 0.013] on its second frame. On `test_a`, fixed at 40% share is nominally worse than no hand-off, -0.012 [-0.033, 0.011]. On validation at about 41% share, gain was 0.039 for 5-frame and 0.044 for 10-frame hand-offs. For 2-frame hand-offs it was 0.024 at 34% share and 0.014 at 43%. At 25% or below, length made little difference. The frozen choice was 2 frames, under the 25% cap the preregistration set.
4. **Doubt's extra milliseconds per frame are a counting effect.** Saved milliseconds cover frames 21 to 120, but the budget covers only frames 21 to 70. On `test_b`, doubt's share over the full range is 21.6% against fixed's 14.3%, and on `test_a` 27.9% against 19.9%, because doubt keeps asking for the solver after frame 70. Counts predict 68.0 ms against the actual 67.1 on `test_b`. In runs that stop at frame 70, doubt takes 49.4 ms and fixed 49.5 ms on `test_b`, and 63.1 and 61.7 on `test_a`. So the "equal budget" was equal only inside the window.
5. **On `test_c`, doubt's level separates the two sides, and one threshold fits neither.** On the low side the largest doubt is 0.009 against `tau` 0.0198, so no frame triggers. Yet that side could have used the solver: fixed schedules gain 0.016 to 0.079, hand-offs lowered error below the frame before in most events (67 to 100% by start bin), and the ensemble makes 23% too much smoke, which doubt doesn't flag. On the high side, 92.6% of frames are over `tau`, and 14 of 20 scenes are over it on every frame. Fixed at 66% scored 0.419 against the policy's 0.410 at 92%, so the extra share bought little.
6. **The oracle lost on `test_b` mainly by acting too late, not by choosing the wrong scenes.** Its threshold (error above 0.596) gives a median first hand-off at frame 55.5, a mean start at frame 61, and 97.7% consecutive hand-offs. Even spacing at the oracle's scene shares would gain 0.040 to 0.042. It actually gained 0.010, a gap of -0.030 [-0.050, -0.013]. Scenes the oracle picked gain more under the fixed schedule, by 0.033 [0.009, 0.058]. So knowing true error does identify where the solver helps, but a threshold on error uses that too late.

### Explanations the data doesn't support

These are exploratory and uncorrected, on the same test sets, so they aren't refuted, only unsupported here.

1. **"Late bunching explains the `test_b` loss."** Doubt's hand-offs are later, with mean start frame 53.0 against 45.5, and only 30 of 208 solver frames fall in frames 21 to 40, against 84 of 213 for fixed. But start times alone predict a difference of +0.0013 [-0.0059, 0.0076], against the actual -0.0229. The same-scene slope on `test_b` gives about -0.0036 for that shift, a small part of the gap.
2. **"Early hand-offs are better," as a general rule.** The same-scene slope per 10 frames later changes direction between sets. On `test_a` it is +0.0069 [0.0026, 0.013], so later is better. On `test_b` it is -0.0048 [-0.0093, -0.0007], so earlier is mildly better. On the low side of `test_c` it is -0.0078 [-0.0268, -0.0004], from only 10 scenes. On the high side it is -0.0184 [-0.039, 0.0015], which includes zero. The earliest hand-offs are the least useful on `test_a` and `test_b`.
3. **"Chaos erases the benefit inside the window."** The gain persists. On `test_b`, hand-offs starting at frames 31 to 40 still show -0.030 [-0.048, -0.016] ten frames later, with no fading inside the window. This says nothing about whether chaos is why error stays after a hand-off, which is under "What this can't settle".

### Partly supported

- **The solver rarely returns to the reference path.** Two frames into a hand-off, error is below the frame before in 0 to 37% of events on `test_a` and 0 to 27% on `test_b`. Hand-offs slow growth rather than reverse it. On `test_b`, starts at frames 31 to 40 reach 0.148 against 0.166 without the hand-off, from 0.141 before. The low side of `test_c` is the exception, as item 5 above shows.
- **Diminishing returns are real on `test_b`.** Gain per 10 points of share falls from 0.030 to 0.015 at 40% share, and the squared term is -0.0007 [-0.0012, -0.0001].

### What this can't settle

- **Chaos as a cause of error staying after a hand-off.** Saved rollouts can't separate chaos from an existing error simply being carried forward. They show only that the gain doesn't fade inside the window.
- **Consecutive hand-offs.** The coefficients in random runs can't be told apart from zero on `test_a`, `test_b` or the high side of `test_c`, so nothing shows that bursts are wasteful in themselves. Random runs have few such pairs.
- **Whether a better rule would beat fixed,** such as a per-scene budget or a threshold scaled to the flow. That needs new rollouts on validation, and new test scenes to check them on.
- **Whether doubt's frame choice helps within a scene.** Against nearby frames in the same scene, at lag 1 only, the gain is small. On `test_a` it is -0.0052 [-0.0119, -0.0003] over 19 scenes, so the interval lies wholly below zero, a small nominal gain for doubt with no correction for the comparisons tried. On `test_b` it is -0.0003 [-0.0014, 0.0008] over 14 scenes. Within a start bin, higher doubt does go with larger reductions (Spearman mostly -0.1 to -0.56, no intervals). Scene mean doubt doesn't predict scene gain, at 0.087 [-0.255, 0.436] on `test_b`.
- **`test_a`.** Doubt's gain of 0.0115 [0.005, 0.018] is above zero and fixed's 0.0055 [-0.007, 0.020] is not, but the paired difference includes zero. This is not a win.
- **Whether doubt's skipped scenes gain more than its chosen scenes** under fixed: 0.022 [-0.003, 0.046] on `test_b`.
- **How much of the 0.0229 is one schedule's luck,** as stated under the third hope.
- **Late obstacles.** Their scene numbers aren't in these files, so they weren't analysed.

### Secondary metrics on `test_b`

| Frames 21 to 70 | Vorticity correlation | Highest spectrum band |
|---|---|---|
| Ensemble | 0.799 | -0.208 |
| Doubt | 0.824 | -0.223 |
| Random | 0.829 | -0.260 |
| Fixed | 0.839 | -0.269 |
| Solver only | 0.968 | -0.043 |

Hand-offs lose more of the finest scales than the ensemble does, which fits the restart cost. They do raise vorticity correlation, and fixed raises it most.

Later frames don't rescue doubt. Over frames 71 to 120 on `test_b`, mean error was 0.711 for doubt at 29.3% solver, 0.647 for fixed at 14.3%, and 0.667 for the oracle at 72.9%. These have no intervals and unequal budgets.

### Figures

| Figure | Shows |
|---|---|
| `results/figures/handoff_comparison.png`, `handoff_sweep.png` | The headline comparison, and the descriptive sweep curves |
| `results/figures/doubt_evidence.png`, `doubt_by_time.png`, `doubt_calibration.png`, `doubt_risk_coverage.png` | Doubt against error |
| `results/figures/accuracy_error_by_time.png`, `ablation.png` | Accuracy by rollout time, and the ablations |
| `results/figures/posthoc_allocation.png`, `posthoc_share_curves.png`, `posthoc_timing.png`, `posthoc_first_handoff.png`, `posthoc_test_c_doubt.png` | The post-hoc analysis above. Exploratory |

## What the project can and can't claim

It can claim that a small neural ensemble runs this 2D smoke about 13 to 17 times faster than this project's reference solver, that it is more accurate than the coarse solver on familiar scenes and in the primary window on unseen shapes, and that its doubt correlates with error at a given time. It can also claim a clean negative result, measured under rules fixed in advance: this threshold rule on this doubt signal didn't earn its solver time.

It can't claim that doubt-based hand-off works, that doubt warns of unfamiliar shapes, or that the post-hoc reasons are proven. None of this is new science. It's a small test on 2D smoke of a question that recent research already explores.

## PUBLIC TRACK PENDING

**This section is a placeholder and holds no results.** `results/exp_well.json` doesn't exist yet, so nothing in this file says anything about the public dataset. Every number above is from the project's own generated data.

To be filled in from `results/exp_well.json` when it exists, with no figure typed from memory:

| Item | Value |
|---|---|
| Dataset | `turbulent_radiative_layer_2D` from The Well (Ohana et al.; Fielding et al. 2020), a published simulation benchmark. It isn't physical measurement |
| Ensemble | 3 members |
| Within-time Spearman, doubt, with interval | PENDING |
| Within-time AUROC | PENDING |
| Calibration and risk-coverage | PENDING |
| No-ensemble signal, same measures | PENDING |
| Did doubt predict error on this dataset? | PENDING |

This track tests doubt only. It has no solver, so it can't say anything about hand-off.

**Dataset and credit.** `turbulent_radiative_layer_2D`, from The Well (Ohana et al., *The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning*, NeurIPS 2024, arXiv:2412.00568). The dataset card asks for this citation: Fielding et al., *Multiphase Gas and the Fractal Nature of Radiative Turbulent Mixing Layers*, The Astrophysical Journal Letters 894(2), L24, 2020, DOI 10.3847/2041-8213/ab8d2c. The data licence is CC-BY-4.0. Dataset link: https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D. Full entries are in `docs/REFERENCES.md`.
