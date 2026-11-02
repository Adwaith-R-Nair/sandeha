# Sandeha: Preregistration

**Fixed on:** 2026-11-03, before the dataset was generated and before any model existed. Git history is the record of this. The file's commit comes before the data manifest and before all model code.

**Changes:** none of the rules below change after a model has been run on any split. If one has to, the change gets its own dated entry in the section at the end, with the reason, and the original figure is reported beside the changed one. `docs/DESIGN.md` sections 6.2, 6.3 and 6.5 are the source for these rules. Where this file is more exact, it decides.

## Terms

- **Frame:** one neural step, or 4 reference solver steps at `dt=0.01`. A scene has frames 0 to 120.
- **Reference:** the 256×256 solver run stored at 128×128 in the data split.
- **Fluid cells:** cells not solid in `obstacle(i, frame - 1)`, the solid in force on the way into that frame.
- **Rollout:** starts from the stored reference at frame 20 and feeds on its own output. The late obstacle joins the mask at `late_frame`, as it does in the data.

## Primary metric

**Per-frame error.** For channels `u`, `v` and density, take the root mean square difference from the reference over fluid cells, and divide by the reference's standard deviation in that channel over the same cells. The per-frame error is the mean of the three channels. Where a channel's standard deviation is below 1% of its training standard deviation in the normaliser, that 1% is used in its place, so that a nearly empty frame can't dominate the mean.

**Blow-ups.** A channel value that isn't finite, or a per-frame error above 10, counts as exactly 10. That frame and all later frames of the rollout stay in every average. Nothing is dropped. The share of blown-up rollouts is reported per policy and test set.

**Scene score.** The mean per-frame error over frames 21 to 70, a horizon of 50 neural steps from start frame 20.

**Set score.** The mean scene score, with a 95% confidence interval from a bootstrap of 2000 resamples of whole scenes.

**Horizon.** It stays at 50 steps. The rule was to extend it only if solver checks showed the plume hasn't reached typical obstacles by then. In `results/scene_check.json`, smoke first touches an obstacle by frame 49 at the latest in the train check group and by frame 58 in `test_b`, inside the window. The `test_c` low side reaches obstacles later, at frames 40 to 79 in its check group, and this doesn't extend the horizon: the set is there to test an unfamiliar flow, and its results are reported per side. Those two check groups have 8 scenes each.

**What this metric can't show.** The window runs from `t=0.8` to `t=2.8`. In `results/solver_check.json`, the 128×128 solver stops being clearly closer to 256×256 than 64×64 is, at about `t=1` on a hard-edged inflow, and at about `t=2` on a smooth one. Grid order is reversed later still in that one plume, and 256×256 was not compared with a finer grid. So past about `t=1`, the reference is one realisation of a chaotic flow, not pointwise truth. The primary metric then measures agreement with that realisation, and an error near the level reached by two equally good solvers doesn't mean the prediction is physically wrong. To give that level a figure, the coarse 128×128 solver's error against the reference is reported on the same axes, and the primary metric is never read without the secondary ones.

## Secondary metrics

Reported per test set at frames 21 to 70 and frames 21 to 120, and over the same rollouts as the primary metric:

- Largest and root mean square divergence in fluid cells, using the same operator on every policy.
- Vorticity correlation with the reference.
- Vorticity spectrum: the log-ratio against the reference, by wavenumber band.
- Plume statistics: total smoke, density centroid height, and the height of the top of the plume, as the ratio or the difference against the reference.

For the longer range, rollouts run on past frame 70 under the same blow-up rule.

## Does doubt predict error?

- **Evidence measure.** Spearman rank correlation between the frame's doubt score and its per-frame error, computed **within each fixed rollout time**, then averaged across rollout times with equal weight. Scenes are resampled whole for the confidence interval. This is the figure that counts as evidence, since error and doubt can both rise with time alone.
- **Pooled correlation** over all frames and rollout times is also reported, and is labelled as not being evidence by itself.
- **No-ensemble comparison.** The same measures for the divergence of a single model's output.
- **Rise on unfamiliar sets.** Doubt on Tests B and C against Test A, compared at matching rollout times.

## Bad-frame threshold

A frame is **bad** when its per-frame error is above the 80th percentile of per-frame errors for frames 21 to 70 of ensemble-mean rollouts with no hand-off, pooled over validation scenes. That value is frozen on validation and written to `results/handoff_selection.json`. It is then used unchanged for AUROC, calibration and risk-coverage on every test set.

Within-time AUROC is reported only at rollout times with at least 5 bad and 5 good frames in the set, and the number of times excluded is stated. If a test set has no bad frames, or all bad frames, at any time, it is reported as such, not imputed.

## Hand-off selection and the frozen threshold

Chosen on validation only, then written to `results/handoff_selection.json` and committed before any test split opens:

- `handoff_frames` from `{2, 5, 10}`, and doubt aggregation from `{mean, p95, p99}`. First by highest within-time Spearman correlation. Ties, meaning overlapping confidence intervals, go to the lower primary error at a solver share of 25%.
- **Threshold `tau`:** the value giving the lowest primary error with a solver share at or below 25% on validation. If no value reaches that share, it is the value with the share closest below 25%, and this is reported.

Solver share is the fraction of frames in frames 21 to 70 produced by the solver.

## Baseline budget

The frozen `tau` runs unchanged on each test set. The solver share it actually uses there, across that set's scenes, sets the budget for the fixed-schedule and random baselines on that set. Each baseline's share must land within 1 percentage point of the budget, and the achieved share is reported.

The random baseline is run with 5 fixed seeds per scene and averaged. Sweep curves on test sets are descriptive and choose nothing. All figures include the cost of every ensemble member.

## Test-set policy

- Tuning, selection, thresholds and every design choice use train and validation only.
- `open_split` refuses a test split without `final_eval=True`, and checks each file against the checksums in `data/manifest.json`. A copy of the manifest is committed to git.
- `--final` is run once, after all code is committed and that commit's hash is added to this file. Any later bug fix is logged in `docs/BUILD_LOG.md` with what changed and the numbers before and after, and the first numbers stay on record.
- **Exception, before any model exists:** the data spot-check after generation opens test splits to render a sample scene from each and to confirm stats, NaN count and manifest. It is an integrity check on the data. Nothing about the data, the ranges or the models is tuned from it. Should it find a defect, the fix and regeneration are logged, and the new data's manifest is committed.

## Reporting `test_c`

Every `test_c` figure is given for the low side, the high side, and pooled. The realised number of scenes per side is in the manifest. A scene's side comes from its buoyancy. Input amplitude and flow regime leave their ranges together in this set, so a rise in doubt can't be divided between them (`docs/DECISIONS.md` B20).

The low side's plumes reach obstacles late or within the window's later part, so its figures say more about an unfamiliar flow than about flow around obstacles. This is stated beside them.

## Late obstacles

`LATE_FRAMES` is unchanged, at frames 20 to 90, and each late-obstacle scene stays in its split. The primary window ends at frame 70.

- **`late_frame` from 20 to 69:** the solid changes inside the window, and the scene counts as it stands, with the obstacle in the mask from `late_frame` on.
- **`late_frame` from 70 to 90:** the change first shows in frame 71 or later, so the primary metric sees these only before the obstacle appears, and treats them as scenes with no late obstacle. They are not dropped or redrawn. The 21 to 120 secondary metrics do include the event.
- **Counts:** the manifest records each split's realised late-obstacle frames, and the report gives the number of scenes in each group.
- **Solid on the event frame:** frame `late_frame` was made under the old solid, so it may hold smoke inside the new one. The first frame with the new solid enforced is `late_frame + 1`. Error at frame `late_frame` uses the old solid's fluid cells, which is what "fluid cells" above says.

## Headline comparison

The result that answers the project's question is the primary metric of doubt-based hand-off against the fixed and random baselines at matched solver budget, on Test B, with the threshold frozen on validation. Tests A and C are reported in the same form. Hoped-for outcomes that fail are reported as findings (`docs/DESIGN.md` section 6.5). Up to one further signal may be tried, chosen on validation only and disclosed.

## Record

| Date | Change | Reason |
|---|---|---|
| 2026-11-03 | First version | |
| | Code commit for `--final`: to be recorded in Task 9 | |
