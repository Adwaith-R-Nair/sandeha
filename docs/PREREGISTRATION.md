# Sandeha: Preregistration

**Fixed on:** 2026-11-03, before the dataset was generated and before any model existed. Git history is the record of this. The file's commits come before the data manifest and before all model code. The rules were tightened three times after review on the same terms, as the record at the end states.

**Changes:** none of the rules below change after a model has been run on any split. If one has to, the change gets its own dated entry in the section at the end, with the reason, and the original figure is reported beside the changed one. `docs/DESIGN.md` sections 6.2, 6.3 and 6.5 are the source for these rules. Where this file is more exact, it decides.

## Terms

- **Frame:** one neural step, or 4 reference solver steps at `dt=0.01`. A scene has frames 0 to 120.
- **Reference:** the 256×256 solver run stored at 128×128 in the data split.
- **Fluid cells:** cells not solid in `obstacle(i, frame - 1)`, the solid in force on the way into that frame.
- **Rollout:** starts from the stored reference at frame 20 and feeds on its own output. The late obstacle joins the mask at `late_frame`, as it does in the data.
- **Rollout time:** a frame's number less 20. Every rollout starts at frame 20, so a rollout time is the same frame in every scene.
- **Doubt score of a frame:** the chosen aggregation of the doubt map from the ensemble step into that frame, over that frame's fluid cells.
- **Evidence rollouts:** ensemble-mean rollouts with no hand-off, every member stepping from the shared mean, over all of a set's scenes. Hand-off rollouts aren't used, because solver frames have no doubt score, and which frames they replace depends on `tau`.
- **Bootstrap:** every interval here is a percentile interval from 2000 resamples of whole scenes, with seed 20261103. A paired comparison applies the same resamples to both sides.

## Primary metric

**Per-frame error.** For channels `u`, `v` and density, take the root mean square difference from the reference over fluid cells, and divide by the reference's standard deviation in that channel over the same cells. The per-frame error is the mean of the three channels. Where a channel's standard deviation is below 1% of its training standard deviation in the normaliser, that 1% is used in its place, so that a nearly empty frame can't dominate the mean.

**Blow-ups.** A channel value that isn't finite, or a per-frame error above 10, counts as exactly 10. That frame and all later frames of the rollout stay in every average. Nothing is dropped. The share of blown-up rollouts is reported per policy and test set.

**Scene score.** The mean per-frame error over frames 21 to 70, a horizon of 50 neural steps from start frame 20.

**Set score.** The mean scene score, with a 95% bootstrap interval.

**Horizon.** It stays at 50 steps. The rule was to extend it only if solver checks showed the plume hasn't reached typical obstacles by then. In `results/scene_check.json`, each check group has 8 scenes. Smoke first touches an obstacle by frame 49 at the latest in the train group and by frame 58 in the `test_b` group, inside the window. The `test_c` low side reaches obstacles later, and has two check groups:

| Group | What it is | First touch | Scenes first touching after frame 70 |
|---|---|---|---|
| `test_c_low_side` | Physics sampled as in the split | Frames 40 to 79 | 2 of 8 (frames 71 and 79) |
| `test_c_low_corner` | Geometry from the split, with buoyancy and strength both set to the bottom of their ranges. A deliberately extreme setting, not a sample of the split | Frames 59 to 92 | 6 of 8 |

A scene first touching after frame 70 has no obstacle contact at all in the primary window. This doesn't extend the horizon: the set is there to test an unfamiliar flow, and its results are reported per side.

**What this metric can't show.** The window runs from `t=0.8` to `t=2.8`. In `results/solver_check.json`, the 128×128 solver stops being clearly closer to 256×256 than 64×64 is, at about `t=1` on a hard-edged inflow, and at about `t=2` on a smooth one. Grid order is reversed later still in that one plume, and 256×256 was not compared with a finer grid. So past about `t=1`, the reference is one realisation of a chaotic flow, not pointwise truth. The primary metric then measures agreement with that realisation, and an error near the level reached by two equally good solvers doesn't mean the prediction is physically wrong. To give that level a figure, the coarse 128×128 solver's error against the reference is reported on the same axes, and the primary metric is never read without the secondary ones.

## Secondary metrics

Reported per test set at frames 21 to 70 and frames 21 to 120, and over the same rollouts as the primary metric:

- Largest and root mean square divergence in fluid cells, using the same operator on every policy.
- Vorticity correlation with the reference.
- Vorticity spectrum: the log-ratio against the reference, by wavenumber band.
- Plume statistics: total smoke, density centroid height, and the height of the top of the plume, as the ratio or the difference against the reference.

For the longer range, rollouts run on past frame 70 under the same blow-up rule.

## Does doubt predict error?

Every measure in this section and the next uses the evidence rollouts at frames 21 to 70, the 50 rollout times of the primary window, unless it says otherwise. The same figures at frames 21 to 120, and the same figures on independent rollouts, are also reported, labelled as secondary, and don't count as evidence.

- **Evidence measure.** At each rollout time, the Spearman rank correlation across a set's scenes between the frame's doubt score and its per-frame error. Tied values share their average rank. The figure is the mean over rollout times, with equal weight, and has a 95% bootstrap interval. This is the figure that counts as evidence, since error and doubt can both rise with time alone. `test_c` has this figure per side and pooled.
- **An undefined time.** Where every doubt score or every error at a rollout time is tied, as when every scene is capped, the correlation is undefined. That time leaves that average, in that resample too, and the number of such times is stated. If none remain, the set is reported as having no figure.
- **Pooled correlation** over all frames and rollout times of the same rollouts is also reported, and is labelled as not being evidence by itself.
- **No-ensemble comparison.** The same measures for a single-model rollout with no hand-off, using the lowest-seed member: the root mean square divergence of its output over fluid cells, against that rollout's own per-frame error.
- **Rise on unfamiliar sets.** At each rollout time, the chance that a frame from Test B or C has higher doubt than one from Test A, counting ties as half. The figure is the mean over rollout times, with scenes resampled within each set. The median doubt at each time is also plotted.

**Non-finite doubt.** A doubt score counts as non-finite if any fluid cell of its doubt map is NaN or infinite, under any aggregation. The same goes for the no-ensemble signal.

- In every ranking, it counts as positive infinity: above any finite score and tied with other non-finite scores. So Spearman, AUROC, calibration and risk-coverage all keep the frame. Nothing is dropped.
- Such a frame also normally has an error of 10 under the blow-up rule, so it adds agreeing high ranks. The number of these frames is given beside each figure, so that a correlation resting on them can be seen.
- In a hand-off policy, it triggers a hand-off, as the hand-off rules below state.

**Calibration.** Ten bins, with edges at the deciles of doubt on validation's evidence rollouts, frozen with the threshold. Non-finite scores go in the top bin, and equal edges merge. Each bin reports its number of frames, its mean per-frame error and its share of bad frames, pooled over rollout times, and is labelled as not being evidence by itself.

**Risk-coverage.** At coverage from 0.1 to 1.0 in steps of 0.1, keep the frames with the lowest doubt, and report their mean per-frame error. Tied frames are kept or left out together, and the achieved coverage is reported. It is computed both pooled and within each rollout time, then averaged across times with equal weight. Neither is evidence by itself.

## Bad-frame threshold

A frame is **bad** when its per-frame error is above the 80th percentile, with linear interpolation, of per-frame errors for frames 21 to 70 of validation's evidence rollouts, pooled over scenes. A frame at the cap of 10 is always bad, even if that percentile reaches 10. That value is frozen on validation and written to `results/handoff_selection.json`. It is then used unchanged for AUROC, calibration and risk-coverage on every test set.

AUROC uses the evidence rollouts, with doubt as the score and non-finite doubt ranked as above. Ties count as half. Within-time AUROC is the mean over rollout times with at least 5 bad and 5 good frames in the set, and the number of times excluded is stated. If a test set has no bad frames, or all bad frames, at any time, it is reported as such, not imputed. Pooled AUROC is also reported, and is labelled as not being evidence by itself.

## Hand-off rules

These apply to doubt-based hand-off and to both baselines, in selection and on test sets.

- **Trigger.** Each frame's doubt score comes from the ensemble step into that frame. A hand-off policy with a finite threshold hands off if the score is above `tau` or non-finite.
- **What the solver starts from.** The ensemble's output for the frame that triggered is thrown away. The solver starts from the frame before, and makes `handoff_frames` frames, starting with the one that triggered. The ensemble then resumes from the solver's last frame. A hand-off therefore never starts from a non-finite state.
- **Forced hand-offs.** A fixed-schedule or random policy whose ensemble output is non-finite hands off in the same way, outside its schedule. A rescue that any system could perform is then not credited to doubt alone. Forced solver frames count towards the solver share, and each policy's number of forced hand-offs is reported.
- **A failing solver.** If the solver's own output isn't finite, that frame and all later ones count as blown up, and no further hand-off is tried in that rollout.
- **Solver share** is the fraction of frames in frames 21 to 70, across all of a set's scenes, produced by the solver. A hand-off running past frame 70 counts only inside the window.

## Hand-off selection and the frozen threshold

Chosen on validation only, then written to `results/handoff_selection.json` and committed before any test split opens. The evidence measure can't vary with `handoff_frames`, because evidence rollouts have no hand-off, so it decides aggregation alone. The steps run in this order:

1. **Aggregations left in.** For `mean`, `p95` and `p99`, compute the evidence measure on validation. The one with the highest value leads. Another stays in if the 95% bootstrap interval of its paired difference from the leader includes zero. Each is compared only with the leader, so the result is a single set however the intervals overlap.
2. **Candidate thresholds.** For each aggregation left in, the 50th, 60th, 70th, 80th, 90th, 95th, 98th and 99th percentiles of its doubt scores on validation's evidence rollouts at frames 21 to 70, pooled. A percentile that isn't finite is left out, and equal values count once. There is one further candidate, **never**: the ensemble with no hand-off, whatever its doubt. This has a solver share of 0, with no forced hand-offs.
3. **Runs.** Each aggregation left in, each `handoff_frames` from `{2, 5, 10}`, and each finite candidate, as hand-off rollouts on all validation scenes, recording the primary set score and the solver share.
4. **Choice.** Among runs with a solver share at or below 25%, the lowest primary set score, compared after rounding to 6 decimal places. Remaining ties go to the lower solver share, then in the order `mean`, `p95`, `p99`, then to the smaller `handoff_frames`.

**never** is always eligible, so a choice always exists. If it wins, the hand-off gave no gain on validation at this budget. That is written down, reported as a finding, and carried to the test sets unchanged.

## Baseline budget

The frozen `tau` runs unchanged on each test set. The solver share it actually uses there, across that set's scenes, sets the budget for the fixed-schedule and random baselines on that set. On `test_c`, each side has its own budget and baselines. Pooled `test_c` figures combine the two sides' runs, so that the pooled budget matches as well.

Both baselines use the frozen `handoff_frames`, and their schedules are worked out from the budget alone, before any rollout runs:

- **Fixed schedule.** A hand-off every `period` frames, with `period` at least `handoff_frames`. The first hand-off in the set's scene number `i` of `n` starts `floor(i * period / n)` frames after frame 21, so that the scenes aren't all handed off on one frame. Of the possible periods, the one with the scheduled share nearest the budget is used, and ties go to the longer period.
- **Random.** At each frame the ensemble would produce, a hand-off starts with chance `rate`, from a stream fixed per scene and seed, with 5 fixed seeds per scene, and the results averaged. `rate` is found by bisection on those streams, to the scheduled share nearest the budget.

**Matching.** A baseline is matched if its achieved share, forced hand-offs included, is within 1 percentage point of the budget. The achieved share is always reported.

**When it can't match.** Long hand-offs make the possible fixed-schedule shares coarse at high budgets. If the nearest scheduled share is more than 1 point away, the nearest schedule below the budget and the nearest above it both run, and both are reported as unmatched. If forced hand-offs push a matched schedule outside the 1 point, it stays as it is and is reported as unmatched. A budget of 0 means there's no hand-off to compare, and this is stated in place of a comparison.

**Comparison.** Doubt-based hand-off against a baseline is the paired difference in scene score over the set, with a 95% bootstrap interval. Doubt beats a baseline only if that interval lies wholly below zero, and only against a baseline whose achieved share is matched or above the budget. One below the budget has had less solver time, so a win against it alone isn't a win.

Sweep curves on test sets are descriptive and choose nothing. All figures include the cost of every ensemble member.

## Oracle bound

The oracle shows what a hand-off could gain from knowing the true error at the current frame. It isn't a system anyone could run, so it reads a test set's reference by design. This is the one exception to choosing thresholds on train and validation only, and nothing else is chosen from it.

- **Trigger.** It hands off if the per-frame error of the ensemble's output for a frame, against the reference at that same frame, is above its `tau`. It reads no later frame. The hand-off rules above then apply as for any other policy: the output is thrown away, the solver starts from the frame before, and a failing solver ends hand-offs in that rollout.
- **Non-finite output.** It counts as an error of 10 and triggers at any `tau`, as a forced hand-off. These are counted, and count towards the solver share.
- **Settings.** It uses the frozen `handoff_frames`, and its budget is the same as the baselines': the solver share the frozen `tau` actually used on that set, and per side on `test_c`. A budget of 0 means it isn't run, and this is stated.
- **Its `tau`, per set and per side.** The first candidate is the `(100 - budget)`th percentile, with the budget in percentage points and linear interpolation, of per-frame errors for frames 21 to 70 of that set's evidence rollouts. If its share isn't within 1 percentage point of the budget, up to 8 further candidates follow by bisection on `[0, 10]`: a share above the budget raises the lower end to that `tau`, and a share below it lowers the upper end. The search stops at the first share within 1 point.
- **What is kept.** The first candidate within 1 point or, failing that, the candidate with the share nearest the budget, with ties going to the lower share and then the lower `tau`. Each candidate's `tau` and share are reported. A kept share more than 1 point away is reported as unmatched, and no further search is made.
- **Reading it.** The set score and the paired difference from doubt-based hand-off, with a 95% bootstrap interval, are reported beside the baselines and are descriptive. They don't enter "beats" above. The oracle judges one frame at a time, so a better schedule could exist, and doubt-based hand-off scoring lower than the oracle would be reported as it stands, not explained away.

Each search runs up to 9 rollout sets per set or side. No model exists, so that time isn't measured, and Task 9 must measure it against the evaluation budget.

## Test-set policy

- Tuning, selection, thresholds and every design choice use train and validation only.
- `open_split` refuses a test split without `final_eval=True`, and checks each file against the checksums in `data/manifest.json`. A copy of the manifest is committed to git as `results/data_manifest.json`, and `data/manifest.json` must match it. `Normalizer.load` checks `data/normalizer.json` against its checksum in the manifest.
- `--final` is run once, after all code is committed and that commit's hash is added to this file. Any later bug fix is logged in `docs/BUILD_LOG.md` with what changed and the numbers before and after, and the first numbers stay on record.
- **Exception, before any model exists:** the data spot-check after generation, `uv run python scripts/spot_check.py --data-root data`, opens test splits to render a sample scene from each and to confirm stats, NaN count and manifest. The scenes drawn are fixed in advance: each split's scene 0 and its first late-obstacle scene. It is an integrity check on the data. Nothing about the data, the ranges or the models is tuned from it. Should it find a defect, the fix and regeneration are logged, and the new data's manifest is committed.

## Travel in the spot-check

A step's largest travel can pass the thinnest wall, 10.24 cells at 256×256, when a late obstacle lands in a fast flow, and semi-Lagrangian advection could then carry smoke across a wall. Travel is a distance, as the wall's thickness is: in each cell, the faster of its two `u` faces and of its two `v` faces, combined as `sqrt(u² + v²)`, times `dt` and the grid size. This is an upper bound on the speed in that cell. `stats.scenes_over_thinnest_wall` in the manifest names any scene whose largest travel is above the wall, per split. The spot-check runs `uv run python scripts/check_travel.py --data-root data`, which reads only the manifest, reruns scenes from their seeds, and writes `results/travel_rule.json`. It states the largest travel in each split and its margin to the wall, and applies this rule:

1. **No scene named:** this item passes.
2. **Any scene named:** the data isn't accepted until each one has been compared. Restart the scene alone from the frame in `stats.travel_frame_by_scene`, and continue for 5 frames at `dt=0.01` and `dt=0.005` from one shared state. Its figure is the largest root mean square density difference over fluid cells, relative to root mean square density. A frame less than 5 frames from the scene's end moves back, so that the comparison stays inside the scene.
3. **Controls:** the same figure for the 3 scenes in that split with the lowest largest travel that aren't named, each from its own frame, whether or not they have a late obstacle.
4. **Pass or fail:** a named scene passes if its figure is at most twice the largest control's. It stays in the set, and its figure and a rendering of the frames around the spike are logged. The command writes that rendering to `results/figures/travel_rule_<split>_<scene>.png`, for any named scene, pass or fail, showing both step sizes and their difference. A split with no control doesn't pass, nor does one with a figure that isn't a number.
5. **On a failure:** this is a defect. Generation is redone for all splits at `dt=0.005`, with 8 solver steps per frame, and is logged under the exception above. No scene is dropped or redrawn over its travel.

Twice is the build's choice, set before any scene above the wall has been seen. It was judged against controls that all had late obstacles (`results/travel_check.json`), whose figures ran from 0.034 to 0.070, while these controls come from every scene in a split, and in `test_b` and `test_c` none have one. So the factor hasn't yet been judged against its own population. `scripts/check_travel.py --control-sample` measures it, at a seed outside the dataset, and writes `results/travel_controls.json`. Its figures are logged before generation. `scripts/generate_data.py` enforces the measuring: it refuses to start unless that file exists, was measured on committed code, has a finite figure for every split, and has the same grid, step, frames and solver settings as the config. It can't check that the figures were read, so logging them stays a step of the build. Should they change the factor, that change gets its own entry in the record below, also before generation. A step-size difference over the whole field can miss a small local leak, so a pass means there's no large effect, not that there's no leak.

**A named scene is likely.** `results/travel_recheck.json` reruns the largest scene of `results/travel_check.json` on the current measure: travel 12.61 cells against its earlier 9.97 on the old measure, 23% above the wall. That is one late-obstacle scene chosen for being the largest, run alone, so it doesn't say how often scenes will be named.

**Sides in `test_c`.** Step 3 takes the lowest travel in the split. In `test_c`, those controls are likely to be low-side scenes, whose plumes are slow, while a named scene is likely to be a high-side one. If step size changes a fast flow more for that reason alone, a high-side scene could fail against low-side controls, and step 5 would then regenerate every split at `dt=0.005`. The rule isn't changed on this reasoning alone. Every `test_c` figure states its side, and `--control-sample` also measures each side's own lowest-travel controls, so the effect has a figure before generation. Any change it leads to goes in the record below, also before generation.

## Reporting `test_c`

Every `test_c` figure is given for the low side, the high side, and pooled. The realised number of scenes per side is in the manifest. A scene's side comes from its buoyancy. Input amplitude and flow regime leave their ranges together in this set, so a rise in doubt can't be divided between them (`docs/DECISIONS.md` B20).

The low side's plumes reach obstacles late in the window, or after it. In the check groups under "Horizon", 2 of 8 sampled scenes and 6 of 8 corner scenes first touch after frame 70, so those have no obstacle contact in the window at all. Its figures therefore say more about an unfamiliar flow than about flow around obstacles, and this is stated beside them. The report gives the number of low-side scenes with no contact by frame 70 in the set itself. These scenes aren't dropped.

**Contact on the stored reference.** A scene has contact by frame 70 if, in any frame `f` from 1 to 70, stored density is above 0.01 in a cell that isn't solid in `obstacle(i, f - 1)` and has a solid cell of that mask in its 3×3 neighbourhood. The box's own walls don't count.

This is a coarse rule, and it isn't the check groups' measurement. Those use 256×256 density in a 3×3 ring around the fine solid. The stored reference holds 2×2 averages, on a dilated coarse mask, and fine states exist only at frame 20, so the fine rule can't be applied to the set. Averaging can hide thin smoke, while the coarse ring reaches further out, so the two can differ in either direction. The set's count therefore isn't compared with the first-touch frames in the table under "Horizon", and is reported with this rule named beside it.

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
| 2026-11-03 | Stated the rollouts and frames of the evidence measure, calibration, risk-coverage and AUROC. Added the rules for non-finite doubt, the trigger, a candidate grid for `tau`, a selection order, baseline schedules, a rule for unmatched budgets, the comparison, and travel above the thinnest wall. Gave both `test_c` low-side check groups | Review before any data or model existed: results could still have been chosen after being seen. `docs/DECISIONS.md` B31 to B35 |
| 2026-11-03 | Stated the coarse contact rule for `test_c`, and added the oracle's rule. Travel became a distance, with its command, the controls' population and what happens with no control. The manifest's tracked copy and the normaliser's checksum are now checked | Second review, before any data or model existed: the oracle's `tau` and the contact count could have been chosen after being seen, and the travel rule couldn't be applied to a scene with no late obstacle. `docs/DECISIONS.md` B37 to B41 |
| 2026-11-03 | Generation now enforces the controls' measuring. The spot-check and the rule's rendering have their commands. Stated the sides of `test_c`'s controls, and that a named scene is likely | Third review, before any data or model existed: the gate had nothing behind it, step 4's rendering had no tool, and the sides' effect wasn't stated. No threshold or factor changed. `docs/DECISIONS.md` B42 to B45 |
| | Code commit for `--final`: to be recorded in Task 9. **Stale, marked 2026-11-04:** this was done, in "Final run" below | |
| 2026-11-04 | Marked the row above as stale. No rule, threshold or commit changed. The results are stamped `31f57dc`, the commit that recorded `3a16f42`, with the same code | Written after the final run, as a note only. `docs/DECISIONS.md` B58 |

## Final run

Code for the single `--final` run is at commit 3a16f42, recorded here before any test split opens.
