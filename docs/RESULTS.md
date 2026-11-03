# Results

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
