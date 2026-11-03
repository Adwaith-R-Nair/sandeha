# Sandeha: Viva Guide

For Adwaith, to explain and defend the project to a professor.

**Status of this guide:** the ensemble's results don't exist yet. Wherever an answer depends on them, this guide says **RESULT PENDING** and names the file that will answer it. When that file exists, read it, and replace the marker with what it says, good or bad.

## Read this first

### The one-paragraph explanation

Neural simulators are fast, but they can drift into believable nonsense and give no warning. Sandeha trains five neural networks to simulate 2D smoke around obstacles. Where they disagree, we call that doubt. When doubt passes a threshold, a conventional physics solver takes over for a few frames, and then the networks resume. The project asks one question: does the doubt actually predict the errors, and how much speed must be given up for each unit of accuracy recovered?

### The sentence not to say yet

`docs/DESIGN.md` has the pitch "I built one that knows when its answers can't be trusted". That is a claim about a result. Until the results exist, say: "We built one that tries to know when its answers can't be trusted, and we measure whether it does."

### Numbers you may quote today

These are the only measured numbers in this guide. None of them is a result of the trained ensemble. The right-hand column is what to say out loud.

**`results/arch_selection.json`.** One model of each type, seed 0, 40 minutes each, on validation. The measure is "mean RMS rollout error in normalised field units, on fluid cells", over 20 steps from frame 20.

| Item | Value in the file | Say |
|---|---|---|
| U-Net error | 0.09147062503461105 | 0.0915 |
| FNO error | 0.12584704872143143 | 0.1258 |
| Persistence error | 0.6689278872931997 | 0.6689 |
| U-Net parameters | 2263107 | 2.26 million |
| FNO parameters | 21248499 | 21.2 million |
| U-Net steps | 26714 total, best at 26000 | |
| FNO steps | 29103 total, best at 27000 | |

**`results/throughput.json`.** Training speed of one model, with mixed precision on.

| Model | Batch | Items per second | Seconds per forward pass | Peak reserved GPU memory (GB) |
|---|---|---|---|---|
| U-Net | 8 | 220.43 | 0.01192 | 0.693 |
| U-Net | 16 | 237.82 | 0.02344 | 1.363 |
| U-Net | 32 | 250.23 | 0.04645 | 2.713 |
| FNO | 8 | 205.66 | 0.00999 | 0.771 |
| FNO | 16 | 249.89 | 0.01867 | 1.176 |
| FNO | 32 | 276.90 | 0.03610 | 1.990 |

A forward pass here is one model on a whole batch. It is not the ensemble's speed, and it can't be turned into one by dividing. The ensemble's speed is **RESULT PENDING** (`results/exp_accuracy.json`, `results/demo_bench.json`).

**`results/spot_check.json`.** The check on the generated data. `passed` is `true`.

| Split | Scenes | Frames per scene | Late-obstacle scenes | Non-finite values | Unconverged solves | Largest density in a solid |
|---|---|---|---|---|---|---|
| train | 240 | 121 | 64 | 0 | 0 | 0.0 |
| val | 30 | 121 | 12 | 0 | 0 | 0.0 |
| test_a | 30 | 121 | 10 | 0 | 0 | 0.0 |
| test_b | 30 | 121 | 0 | 0 | 0 | 0.0 |
| test_c | 30 | 121 | 0 | 0 | 0 | 0.0 |

One scene is named as passing the thinnest wall in one step: scene 12 of val. See question 33.

**`results/solver_check.json`.** The solver on an RTX 3060 Laptop GPU, at `dt` 0.01. Items in a batch here are identical.

| Grid | Batch | Steps per second | Item steps per second | Mean CG iterations | Largest divergence |
|---|---|---|---|---|---|
| 256 | 1 | 19.85 | 19.85 | 8.0 | 0.00098 |
| 256 | 8 | 16.68 | 133.45 | 8.0 | 0.00098 |
| 256 | 16 | 11.89 | 190.20 | 8.0 | 0.00098 |
| 128 | 1 | 24.67 | 24.67 | 7.0 | 0.00055 |
| 128 | 8 | 23.55 | 188.43 | 7.0 | 0.00055 |
| 128 | 16 | 24.64 | 394.20 | 7.0 | 0.00055 |

How far coarser grids are from 256×256, as relative error in density. The ratio is the 128×128 error over the 64×64 error. Below 1 means the finer grid is closer.

| Inflow | Time | 128 against 256 | 64 against 256 | Ratio |
|---|---|---|---|---|
| Soft | 0.25 | 0.0024 | 0.0058 | 0.414 |
| Soft | 1.0 | 0.1892 | 0.3769 | 0.502 |
| Soft | 2.0 | 0.5670 | 0.6465 | 0.877 |
| Soft | 2.5 | 0.7157 | 0.7092 | 1.009 |
| Hard | 0.75 | 0.3038 | 0.4193 | 0.725 |
| Hard | 1.0 | 0.4683 | 0.4317 | 1.085 |

Long run at 256×256, 300 steps: all solves converged, largest divergence 0.00272, and 91 steps above the plan's 0.001. See question 32.

Divergence of a plume at 256×256, the largest value: 0.00085 measured on the faces, and 48.567 measured on the cell-centred view of the same state.

### The files that will hold the results

| File | What it will answer |
|---|---|
| `results/ensemble.json` | Whether all five members trained and passed their check |
| `results/handoff_selection.json` | The chosen aggregation, threshold, hand-off length and bad-frame threshold, all from validation |
| `results/exp_accuracy.json` | Error, usable time and vorticity correlation against persistence and the 128×128 solver. Timing |
| `results/exp_doubt.json` | Whether doubt predicts error, and whether it rises on Tests B and C |
| `results/exp_handoff.json` | The headline comparison, the oracle and the sweep curves |
| `results/exp_ablation.json` | What removing the penalty, removing pushforward and changing ensemble size do |
| `results/exp_well.json` | The doubt measures on the public dataset |
| `results/demo_bench.json` | The demo's frame rate and member count |

`scripts/render_results.py` turns these into `docs/RESULTS.md`, and `scripts/make_figures.py` draws them into `results/figures/`.

---

# Part 1: Every concept in plain language

Each entry says what it is, why we use it and where it is in the code.

## The physics and the solver

### Stable fluids solver

**What it is.** A way to simulate fluid by splitting each time step into simple jobs. Our order is: move everything along the flow, add smoke and buoyancy, then fix the velocity so that the fluid stays incompressible. The "move along the flow" job is semi-Lagrangian: for each cell, trace backwards along the flow to find where its contents came from, and read the value there.

**Why we use it.** It doesn't blow up at large time steps, since a lookup between existing values can't overshoot. It is also simple enough to run on a GPU in PyTorch and explain line by line. Its cost is numerical dissipation: repeated lookups blur fine detail. This solver is the source of truth for the whole project, so we check it first.

**Credit it correctly.** Stam's 1999 paper is the source for the splitting and the semi-Lagrangian idea. It doesn't cover our staggered grid, our conjugate gradient solver or our buoyancy.

**Code.** `src/sandeha/physics/solver.py:SmokeSolver.step` and `src/sandeha/physics/solver.py:SmokeSolver._advect`.

### Pressure projection

**What it is.** Incompressible means that what flows into a cell must flow out. After moving and pushing the fluid, this no longer holds: some cells gain and some lose. That imbalance is called divergence. Projection finds a pressure field whose gradient, subtracted from the velocity, removes the divergence. The velocity that remains is the closest incompressible one.

**Why we use it.** Without it, the flow isn't incompressible and smoke behaves wrongly. It is also the expensive part of a solver step, and that expense is why a network might be worth having. Hand-offs also need it: the network's output isn't exactly incompressible, so it is projected before the solver takes over.

**Code.** `src/sandeha/physics/solver.py:SmokeSolver._project_faces` and `src/sandeha/physics/solver.py:SmokeSolver.project`. The hand-off's use is in `src/sandeha/hybrid.py:HybridSimulator._hand_off`.

### Conjugate gradient

**What it is.** Finding that pressure means solving one large set of linear equations, with one unknown per cell, which is 65,536 at 256×256. Conjugate gradient (CG) solves such a system by repeated improvement, needing only "multiply by the matrix" each round. It needs a symmetric matrix with no negative directions, and ours qualifies. The matrix has one flat direction: adding a constant to the pressure in a connected region of fluid changes nothing, because only the pressure's gradient acts on the flow. That does no harm here, since net flux over a closed region sums to zero.

**Why we use it.** Each cell's equation touches only its neighbours, so the multiply is cheap and suits a GPU. On its own, CG would need many rounds on a large grid, so we add a preconditioner: one multigrid V-cycle per round, which fixes error at all scales by working on coarser copies of the grid. In `results/solver_check.json`, the mean is 8.0 iterations at 256×256 and 7.0 at 128×128, so the count barely grows with the grid.

**Code.** `src/sandeha/physics/solver.py:_conjugate_gradient` and `src/sandeha/physics/solver.py:_v_cycle`.

### Staggered grid

**What it is.** Velocity isn't stored at cell centres. Horizontal velocity `u` lives on the vertical faces between cells, and vertical velocity `v` on the horizontal faces. Density lives at centres. This is also called a MAC grid.

**Why we use it.** A cell's net flow is then just the difference of its own faces, so "what goes in comes out" is exact, and a solid wall is simply a face set to zero. The cell-centred alternative can hide checkerboard patterns that its divergence can't see. The size of the difference shows in `results/solver_check.json`: one projected 256×256 plume has a largest divergence of 0.00085 on its faces, and 48.567 when the same state is viewed at cell centres.

**Watch out.** The network sees the cell-centred view on a 128×128 grid, and so it never sees the faces. That is why its divergence measure has a floor, and why it isn't compared directly with the solver's.

**Code.** `src/sandeha/physics/solver.py:SmokeState`, `src/sandeha/physics/solver.py:SmokeSolver._net_flux` and `src/sandeha/physics/solver.py:SmokeSolver._centre`.

### Why 256 down to 128

**What it is.** The reference solver runs at 256×256. Each frame is averaged over 2×2 blocks down to 128×128, and the network learns on that. A coarse cell counts as solid if any of its four fine cells is, and fields in solid coarse cells are zero.

**Why we use it.** A plain solver at 128×128 is itself cheap, so if we trained on it, being faster than it would prove nothing. Training on results from a finer, costlier grid gives the network something worth learning: fine-grid behaviour at coarse-grid cost. Kochkov et al. support the idea, though their models work inside a coarse solver and ours replaces solver steps.

**Watch out.** This doesn't mean the network will beat the 128×128 solver. That is an open question, tested directly. **RESULT PENDING** (`results/exp_accuracy.json`).

**Code.** `src/sandeha/data/generate.py:_simulate_batch`.

## The network

### Residual prediction

**What it is.** The network doesn't output the next frame. It outputs the change from the current frame, and we add that to the current frame. One network step covers 4 solver steps.

**Why we use it.** Consecutive frames look alike, so most of the next frame is already known. Predicting only the change gives the network a smaller and easier target, and a network that outputs zero is already the persistence baseline. The change has its own scale in the normaliser, because one step's change is much smaller than a field's spread.

**Code.** `src/sandeha/models/factory.py:apply_step`, `src/sandeha/data/dataset.py:Normalizer.encode_delta` and `src/sandeha/train/loop.py:training_step`.

### U-Net

**What it is.** A convolutional network shaped like a U. It shrinks the image step by step to see wide context, then grows it back, and skip connections carry fine detail across from the shrinking side to the growing side. Ours has 4 levels with widths 32, 64, 128 and 256, and 2,263,107 parameters. Its 6 inputs are `u`, `v`, density, obstacle mask, inflow and buoyancy. Its 3 outputs are the changes in `u`, `v` and density.

**Why we use it.** Smoke around obstacles needs sharp local detail at walls as well as wider flow. A U-Net handles both, and it is cheap enough to run five times per frame. It also won the comparison below. Solid cells are forced to zero after each prediction.

**Credit it correctly.** Ronneberger et al. 2015 is a paper about image segmentation. Cite it for the architecture only.

**Code.** `src/sandeha/models/unet.py:UNet`.

### FNO, and why it lost

**What it is.** A Fourier Neural Operator takes the Fourier transform of its features, multiplies the lowest frequencies by learned weights, and transforms back. Each layer sees the whole domain at once. Ours keeps 24 modes in each direction, has 4 layers and 21,248,499 parameters, and adds coordinate channels because our box isn't periodic. Those channels are our own choice, not from the paper.

**Why we use it.** It is the standard alternative for learning PDEs, so choosing the U-Net without testing it would be weak. Both got the same task, the same 40 minutes and the same validation measure.

**What happened.** From `results/arch_selection.json`: U-Net 0.0915, FNO 0.1258, persistence 0.6689. FNO wasn't short of training steps. It made 29,103 steps to the U-Net's 26,714.

**Why it lost.** We know that it lost. We don't know why, because no experiment tested the cause. Three possible reasons, none tested:

- Solid walls make sharp edges, and those need high frequencies that a cut-off at 24 modes throws away.
- The Fourier transform treats the domain as periodic, and our closed box isn't. Coordinate channels only partly deal with that.
- Its local path is a 1×1 convolution, which sees no neighbouring cells.

**Limit.** This is one seed, with no interval. It says that FNO lost on this task, at this size and in this time. It doesn't say that U-Nets beat FNOs in general.

**Code.** `src/sandeha/models/fno.py:FNO2d`, and the comparison in `src/sandeha/train/campaign.py:select_architecture`.

### Divergence penalty

**What it is.** The training loss is the mean squared error of the predicted change on fluid cells, plus 0.1 times the mean squared divergence of the predicted velocity change. The penalty uses central differences, and only on cells whose whole stencil is fluid.

**Why we use it.** The true flow is incompressible, so the network should be nudged the same way. It is a soft nudge: unlike projection, it guarantees nothing about a given output.

**Watch out.** Central differences on cell-centred fields have a floor even on true data, so this can't reach zero. `results/throughput.json` gives that floor on the training targets as 0.00258 at batch 16. The penalty is on the change, not on the next velocity, so it doesn't push each change to cancel the floor already in the field. The weight of 0.1 was kept as planned and hasn't been tuned. Whether the penalty helps is **RESULT PENDING** (`results/exp_ablation.json`).

**Code.** `src/sandeha/train/losses.py:step_loss` and `src/sandeha/train/losses.py:divergence`.

### Pushforward training

**What it is.** In use, the network feeds on its own output, and those frames carry small errors it never saw if it was only trained on perfect inputs. So on half the batches, after the first 10% of steps, we run the model 1 to 3 steps on its own output with no gradient, and then train the next step from there. Its target is the change that gets that imperfect state to the true next frame.

**Why we use it.** It teaches the model to recover from its own kind of error. It costs far less than backpropagating through a whole rollout. Our version is based on Brandstetter et al., and differs from it: they unroll 2 steps, and we push 1 to 3 steps on half the batches.

**Code.** `src/sandeha/train/loop.py:training_step` for the pushing, and `src/sandeha/train/loop.py:train` for when it happens. Whether it helps is **RESULT PENDING** (`results/exp_ablation.json`).

### Mixed precision

**What it is.** Most forward-pass arithmetic runs in 16-bit floats, and the sensitive parts stay in 32-bit. A gradient scaler multiplies the loss up during training, so that small gradients don't round to zero in 16-bit, and it divides them back before the update.

**Why we use it.** To save time and memory on a 6 GB laptop GPU. Some parts had to stay in 32-bit:

- FFTs in the FNO, which have no half-precision form.
- The loss, because squared divergences underflow in 16-bit.
- The state update, because 16-bit is too coarse for the state.

**Limit.** We didn't time the same run with it off, so don't quote a speed-up.

**Code.** `src/sandeha/train/loop.py:amp_context`, the `GradScaler` in `src/sandeha/train/loop.py:train`, and `src/sandeha/models/fno.py:_Spectral.forward`.

## Doubt

### Deep ensembles

**What it is.** Train several networks on the same task, from different random starts and with different data order, then use their mean as the prediction and their disagreement as the uncertainty. We plan 5 members, with seeds 0 to 4.

**Why we use it.** It is simple, well studied, needs no change to the network, and is easy to explain. Members tend to agree on what the data pins down, and may disagree where it doesn't. Its cost is plain: every frame runs every member, and every speed figure must include that.

**Differences from the paper. Know these.** Lakshminarayanan et al. train each network to predict a mean and a variance using negative log-likelihood. Ours are trained on MSE, and we take the spread of their outputs. The paper calls that "a commonly used heuristic" and says "this is inferior to learning the variance by training using NLL", shown on a toy regression task. We have not measured whether this matters here. Stepping every member from a shared mean is also ours.

**Code.** `src/sandeha/ensemble.py:Ensemble.step` and `src/sandeha/train/campaign.py:train_ensemble`.

### Epistemic and aleatoric uncertainty

**What they are.** Epistemic uncertainty is not knowing because the model lacks knowledge, such as on a shape it has never seen. More data could reduce it. Aleatoric uncertainty is randomness in the problem itself, which no amount of data removes.

**Why it matters here.** Ensemble spread mainly aims at epistemic uncertainty. Our solver is deterministic for a given seed, so the data has no sensor noise. But there are two things close to aleatoric:

- A 128×128 frame has lost fine detail, so the same coarse frame can come from different fine states with different futures.
- Smoke is chaotic, so tiny differences grow.

Our members are trained on MSE and don't predict a variance, so nothing in the system separates the two kinds. Don't claim that it does.

**Code.** No function separates them. The nearest is `src/sandeha/ensemble.py:Ensemble.step`.

### Doubt map and doubt score

**What they are.** The doubt map is the spread across members at each cell. For each channel, take the standard deviation across the members, in normaliser units, and then average the three channels. The doubt score is one number per frame, made from that map over fluid cells, by one of three methods: the mean, the 95th percentile or the 99th percentile.

**Why we use them.** The map shows where members disagree, and the demo draws it. A hand-off needs one yes or no per frame, so it needs one number. The method is chosen on validation. **RESULT PENDING** (`results/handoff_selection.json`).

**Details a professor might test.**

- Normalised units stop one channel from swamping the others.
- A non-finite cell gives a score of positive infinity, because a percentile could otherwise pass over it.
- The spread is a population spread, so doubt values at ensemble sizes 2, 3 and 5 aren't on the same absolute scale. Only their rankings are compared.

**Code.** `src/sandeha/ensemble.py:Ensemble.step` and `src/sandeha/ensemble.py:doubt_score`.

## Measuring doubt

**Evidence rollouts.** The measures below use ensemble rollouts with no hand-off, from frame 20, scored on frames 21 to 70. Solver frames have no doubt score, so hand-off rollouts can't be used. The error is the primary per-frame error, in `src/sandeha/handoff_selection.py:frame_error`. A blown-up frame counts as exactly 10 and stays in. Nothing is dropped.

### Spearman

**What it is.** A correlation between ranks. Sort the scenes by doubt, and sort them by error. If the two orders match, it is +1. If they are unrelated, it is about 0. If they are reversed, it is -1.

**Why we use it.** Doubt and error are in different units, and the link between them needn't be a straight line. We only need higher doubt to go with higher error.

**The important part: within time.** Error and doubt can both rise with time alone. A pooled correlation can then look strong even if doubt is only a clock. So the evidence figure works at one rollout time, across scenes, and averages those correlations over the 50 times. The pooled figure is also reported, labelled as not being evidence by itself.

**Limit.** The preregistration sets no pass mark for this figure. Report the value and its interval, and don't invent a mark afterwards.

**Code.** `src/sandeha/well/doubt.py:spearman` and `src/sandeha/well/doubt.py:evidence_measure`. **RESULT PENDING** (`results/exp_doubt.json`).

### AUROC

**What it is.** Take a random bad frame and a random good frame. AUROC is the chance that the bad one has higher doubt, with ties counted as half. 0.5 is a coin toss, and 1.0 is perfect. A frame is bad if its error is above the 80th percentile of validation errors, a value frozen on validation.

**Why we use it.** A hand-off is a yes or no decision, so this asks directly how well doubt separates frames worth handing off. It doesn't depend on any one threshold on doubt.

**Limit.** The within-time figure only uses times with at least 5 bad and 5 good frames, and states how many times were left out. The 80th percentile is a choice made before any model existed, not a measured best.

**Code.** `src/sandeha/well/doubt.py:auroc` and `src/sandeha/well/doubt.py:within_time_auroc`. **RESULT PENDING** (`results/exp_doubt.json`).

### Calibration

**What it is.** Put frames into 10 doubt bins, with edges at validation's deciles, and report each bin's mean error and share of bad frames. If doubt means something, those figures climb from bin to bin.

**Why we use it.** Spearman and AUROC only look at order. Calibration shows the shape: for instance, whether only the top bin carries the signal.

**Watch out.** This isn't calibration in the sense of "a predicted 70% happens 70% of the time". Doubt isn't a probability. It is pooled over time, so it is labelled as not being evidence by itself.

**Code.** `src/sandeha/well/doubt.py:calibration` and `src/sandeha/well/doubt.py:decile_edges`. **RESULT PENDING** (`results/exp_doubt.json`).

### Risk-coverage

**What it is.** Keep only the lowest-doubt frames, and see how large their mean error is. Do this while keeping 10%, 20% and so on up to 100%. Coverage is the share kept, and risk is their error. If doubt works, the error falls as you keep fewer frames.

**Why we use it.** It answers a practical question: if we only trust the frames we are most sure of, how much better are they? It is reported both pooled and within time. Neither is evidence by itself.

**Code.** `src/sandeha/well/doubt.py:risk_coverage`. **RESULT PENDING** (`results/exp_doubt.json`).

### Confidence intervals by bootstrapping scenes

**What it is.** Make a new test set by drawing scenes at random, with repeats, from the real one, and compute the figure on it. Do that 2000 times. The middle 95% of those figures is the interval. The seed is fixed at 20261103.

**Why scenes, not frames.** Frames from one scene are strongly related. Resampling frames would treat them as independent and give an interval that looks narrower than the evidence allows. Whole scenes are the independent unit, which is the same reason the data is split by scene.

**Paired comparisons.** To compare two policies, apply the same resamples to both, and look at the interval of the difference. Both ran on the same scenes, so this removes the differences between easy and hard scenes.

**Limit.** There are only 30 scenes per test set, and about 15 per side on Test C, so intervals may be wide. That is the honest size of the evidence.

**Code.** `src/sandeha/eval/metrics.py:bootstrap_ci` and `src/sandeha/well/doubt.py:evidence_measure`.

## The system and its fair tests

### Hand-off

**What it is.** Each step, the ensemble makes the next frame and its doubt score. If the score is above the threshold `tau`, or isn't finite:

1. The ensemble's frame is thrown away.
2. The previous frame is upsampled to 256×256 with bilinear interpolation, and projected.
3. The solver makes `handoff_frames` frames, at 4 solver steps per frame.
4. Each is averaged back to 128×128, and the ensemble resumes from the last.

**Why we use it.** The aim is to spend solver time where the network is likely to be wrong, and keep the rest cheap. Whether it does that is the main question of the project.

**Limits.**

- It is whole-frame, not per-region.
- The solver starts from a state that is already off, with fine detail missing. It can't be expected to get back to the reference path, only to slow the growth of error. That is why physical secondary measures are reported as well.

**Code.** `src/sandeha/hybrid.py:HybridSimulator.step`, `src/sandeha/hybrid.py:HybridSimulator._hand_off` and `src/sandeha/hybrid.py:DoubtThreshold`.

### Why fixed and random baselines matter

**What they are.** Fixed hands off every so many frames, with the first hand-off offset per scene. Random hands off at each frame with a set chance, averaged over 5 seeds. Both use the same solver budget that the frozen doubt threshold actually used on that test set. A baseline counts as matched if its share is within 1 percentage point of that budget.

**Why they matter.** Any solver time reduces error compared with none. So "hand-off improves accuracy" proves nothing about doubt. The real test is whether choosing when to hand off by doubt beats not choosing, for the same solver time. If it doesn't, doubt isn't earning its keep.

**What "beats" means.** The 95% interval of the paired difference in scene score lies wholly below zero, and the baseline's share is matched or above the budget.

**Fair to both.** A non-finite output forces a hand-off under the baselines too, so a rescue any system could perform isn't credited to doubt.

**Code.** `src/sandeha/hybrid.py:FixedSchedule`, `src/sandeha/hybrid.py:RandomBudget`, `src/sandeha/eval/experiments.py:choose_fixed_schedules`, `src/sandeha/eval/experiments.py:find_random_rate` and `src/sandeha/eval/experiments.py:compare_to_baseline`. **RESULT PENDING** (`results/exp_handoff.json`).

### Oracle

**What it is.** A policy that hands off using the true error of the current frame, which it reads from the reference, at the same solver budget. Nobody could run this in practice. It cheats on purpose.

**Why we use it.** It shows how much could be gained from knowing the current frame's true error. A large gap between it and doubt leaves room for a better signal. A small gap means little was left to gain at that budget.

**Watch out.** `docs/DESIGN.md` calls it an upper bound, but it isn't a strict one. It judges one frame at a time, so a better schedule could exist, and doubt could score lower than it. The preregistration says such a case gets reported as it stands. Its figures are descriptive, and it is the one policy whose threshold comes from a test set.

**Code.** `src/sandeha/hybrid.py:Oracle`, `src/sandeha/eval/experiments.py:oracle_search` and `src/sandeha/eval/rollout.py:oracle_error_of`. **RESULT PENDING** (`results/exp_handoff.json`).

### Ablation

**What it is.** Take a part out and see what changes. We train without the divergence penalty, and without pushforward. We also compare ensembles of 2, 3 and 5 from the existing models.

**Why we use it.** Without it, we can't say that a part does anything. It might only be decoration.

**Limit.** Each reduced training is one seed with 25 minutes of training, which is less than a full member gets. So comparing it with a full member doesn't isolate the effect of what was removed. The result file carries this note.

**Code.** `src/sandeha/train/campaign.py:train_ablations`, `src/sandeha/eval/experiments.py:ablation_experiment`, `configs/ablation_no_divergence.yaml` and `configs/ablation_no_pushforward.yaml`. **RESULT PENDING** (`results/exp_ablation.json`).

### Test-set locking

**What it is.** Test sets are locked in code, not just by promise. `open_split` refuses a test split unless called with `final_eval=True`. It also checks each file against the checksums in the manifest, and the manifest must match its tracked copy in `results/data_manifest.json`.

**Why we use it.** If anything is tuned on a test set, the test set stops measuring what happens on new data. The lock makes an accidental read fail loudly.

**The three test sets.**

- **Test A:** same kinds of scene as training, but new scenes.
- **Test B:** unseen shapes (stars, letters, blobs) with training physics.
- **Test C:** training shapes, with buoyancy and inflow both below or both above the training ranges.

**The one exception.** The spot-check opened test splits before any model existed, to render a sample scene from each and check their statistics. The scenes drawn were fixed in advance, and nothing was tuned from them. This is stated in the preregistration.

**Limit.** The lock stops accidents, not a determined cheat. Someone could change the code. Git history is the record that this wasn't done.

**Code.** `src/sandeha/data/store.py:open_split` and `src/sandeha/data/store.py:read_manifest`.

### Preregistration

**What it is.** Writing down, before seeing any results, exactly what will be measured, how, and what will count as a win. `docs/PREREGISTRATION.md` fixes the metric, the window, the thresholds, the selection steps and the baselines' rules. It was fixed before the data was generated and before any model existed, and git history shows the order. It was tightened three times on those same terms, and its record lists each change.

**Why we use it.** There are many ways to analyse a result. If you choose after looking, you can find a win in noise without meaning to. Fixing the choices first takes that away.

**Limit.** This is our own document in our own repo. Nobody outside reviewed it or holds a copy with a date on it. Some values, such as the blow-up cap of 10 and the 80th percentile, are choices made without measurement, and the point is that they were made first.

**Code.** No function. `src/sandeha/handoff_selection.py:select_handoff` carries out the selection steps, and `src/sandeha/eval/experiments.py:run_experiments` writes the results.

---

# Part 2: The reasoning behind each major design choice

The codes in brackets are entries in `docs/DECISIONS.md`.

**A question about doubt, not only accuracy.** Most course projects report how accurate a model is. This one asks whether the model can tell when it is wrong, since a wrong answer given with confidence is the real danger of a neural simulator. The aim holds even if the answer is no.

**A closed 2D box with smoke and obstacles.** It is small enough for a 6 GB GPU and a deadline, and has walls and chaotic flow. People can also draw on it and see it, so the demo can't be faked with presets.

**Our own solver.** The data, the hand-off and the coarse comparison all need a solver that we can run at both grids, restart from any state, and measure. It must also take hand-offs from a network's state. Existing packages weren't compared, so don't claim that they couldn't do this.

**Semi-Lagrangian over MacCormack (B10).** MacCormack is sharper and was built first. It gave cell-scale stair-steps on thin sheets at this time step, which would have become noise in training targets, and it lost more smoke against the mass budget. The cost is more numerical dissipation, which is an open limit.

**A multigrid preconditioner (B11).** Jacobi barely helps on this nearly constant diagonal, and MIC(0) is sequential, which suits a GPU badly. Red-black multigrid is parallel, and handles obstacles of any shape exactly. A test confirms that it stays symmetric, as CG requires.

**A pressure stopping rule relative to speed (B13).** The rule is that no cell's net flux may exceed `cg_tol` times the fastest face. Each pass re-measures the true divergence from the faces, so float32 drift can't hide. This is also why a fast flow can show a larger absolute divergence (question 32).

**256 for the reference, 128 for the network (D1).** See Part 1. A 128×128 solver is too cheap to be the thing being beaten on speed alone, so the coarse solver is kept as a rival.

**A coarse cell is solid if any fine cell is (B24).** Coarse fluid cells then hold only fluid values, and no wall opens up at the coarse grid. The cost is that solids grow by up to half a coarse cell per side, in the mask.

**One network step is 4 solver steps (D2).** This gives the network a speed advantage per frame to work with, since one forward pass replaces 4 solver steps. 4 is a starting point from the design, and other values weren't compared.

**Buoyancy as an input channel (D3).** One frame can't reveal buoyancy. If it weren't an input, it would be a hidden variable, and Test C wouldn't be a fair test of it.

**Splits by whole scene, with a seed per scene (B28).** Frames of one scene look alike. Splitting by frame would leak near-copies into the test sets. A scene's seed depends only on the master seed, split and index, so resizing a split doesn't reroll the test sets.

**Simple training shapes and two unfamiliar test sets (B17, B20).** Training shapes are circles, rectangles and walls, so stars, letters and blobs are truly new. Each unfamiliar set changes one kind of thing, so that a rise in doubt has one kind of explanation. Test C ties buoyancy and inflow to the same side. The solver's flow depends on their product, and a mixed pairing would be a training flow with rescaled density.

**Balanced shape sizes and an obstacle in the plume's path (B19, B21).** Unseen shapes first covered more area than training shapes, so doubt could have risen simply because more of the box was solid. Sizes were balanced for that reason. The first obstacle sits near a source, since a scene where smoke never reaches a shape shows the model nothing about that shape.

**Late obstacles (B23).** People draw while the demo runs, so the model must see a solid appear in a flow. A share of train, validation and Test A scenes have one. They are absent from Tests B and C, so that a rise in doubt there has one explanation.

**Compare U-Net and FNO, then pick on validation (Architecture selection).** Equal budgets and the same measure. A tie would have gone to the U-Net, as the expected backbone (B49). There was no tie.

**The divergence penalty on the change, away from walls (B46).** Penalising the next velocity would push each change to cancel the floor already in the field. A difference across a wall says nothing about the flow, so only cells with their whole stencil in fluid count.

**Training length set from a speed measured inside the run (B48).** A separate measurement can go stale. So each run measures itself first, and fixes how many steps fit in `max_minutes`.

**Ensemble of 5, from a shared mean (D4, D5).** With independent rollouts, members drift apart over time whatever the flow is doing, and doubt could become a disguised clock. Starting every member from the shared mean at each step makes doubt ask something narrower: from this one state, do they agree on the next step? Independent rollouts are reported as a secondary comparison. The cost is that errors the members make in common go unseen.

**Primary metric and its window (D7).** Normalised error against the reference, from frame 21 to frame 70, a horizon of 50 network steps. Blow-ups count as 10 and stay in, since dropping them would flatter a failing model. Past about `t=1`, the reference is one version of a chaotic flow, not exact truth, so physical secondary measures are always reported beside it.

**Threshold frozen on validation (D8).** Validation chooses the aggregation, then `tau` and `handoff_frames`, by the lowest error with a solver share at or below 25%. Choosing on a test set would invalidate the comparison. "Never hand off" is always a candidate, so a choice always exists. If it wins, that is reported as a finding.

**Baselines get the budget actually used (D8, B34).** The frozen threshold may use more or less solver time on a test set than on validation. Giving the baselines the same share keeps the comparison fair. That reads the frozen policy's behaviour on the test set, but chooses nothing about it.

**Whole-frame hand-off (D6).** Per-region hand-off would be more elegant, but it needs the solver and network to agree along a moving boundary, which is much riskier with this deadline. It is a stretch goal only.

**A hand-off blocks, and an edit waits (B52).** A drawn edit applies after the hand-off ends, so that the solver's solid doesn't change partway through.

**A public-dataset track (D11, B51).** `turbulent_radiative_layer_2D` from The Well tests doubt on data we didn't generate. It has no sign-up and is 7.9 GB. There is no solver for that physics in the project, so it tests doubt only, with no hand-off. It is a published simulation benchmark, not physical measurement, and has to be described that way.

**A local demo over a WebSocket (D9, B53).** The solver needs the GPU anyway, and it must run with no classroom Wi-Fi. A non-finite output hands off in every mode, including forced AI mode, and shows on screen as a solver frame.

**Every number traces to a saved run.** Each result file records its commit, and checksums for data and checkpoints. A kept rollout set is rerun if the code it depended on has changed (B56). A weak number gets reported as a weak number.

---

# Part 3: Likely questions and honest answers

## The hard ones

**1. Why not just run the solver at 128?**
That is the real rival, and it is in the experiment for that reason. From `results/solver_check.json`, at batch 1 the 128×128 solver makes 24.67 steps per second, and 256×256 makes 19.85, so it is only about 1.24 times faster. The log traces this to overhead per step, which is much the same at either grid. At batch 16 it is 394.20 against 190.20 item steps per second, about 2.07 times. Those were identical items, so they are a rough guide. What a network has to do is beat the 128×128 solver on accuracy, for similar compute. Whether it does is **RESULT PENDING** (`results/exp_accuracy.json`). If it doesn't, then on this problem, you should just run the coarse solver for speed and accuracy alone, and we would report that. A coarse solver does have no doubt signal, but that isn't a reason to prefer a less accurate model.

**2. Is this new science?**
No, and we don't claim it is. Recent work already looks at falling back to a solver. ANCHOR (Roy et al.) watches a signal during a rollout and runs the solver for a fixed number of steps, as we do, but its signal is the PDE residual. HyPER (Srikishan et al.) uses a policy trained by reinforcement learning. Hybrid Neural World Models (Lakshmanan and Chopra) uses step-doubling and hands off whole trajectories. A limited search found no paper using ensemble spread to trigger a mid-rollout hand-off, but a few searches don't show that none exists, so we don't call it new. This is a course project, with a measured small-scale answer to a question that matters at larger scale.

**3. Isn't simulated data easy?**
In some ways, yes. There is no sensor noise and no missing data, and the flow follows our solver's equations exactly. So nothing here says the model would work on real smoke. What stays hard is that the flow is chaotic, the model feeds on its own errors, and Tests B and C are built to be unfamiliar. The question isn't "can it match the data" but "can it tell when it is off", and simulation lets us check that, because the true answer exists for every frame. Real captured smoke (ScalarFlow) is 3D and about 450 GB, so it was out of reach here.

**4. Why is it only 2D?**
Cost. A 256×256 grid has 65,536 cells, and 256×256×256 has about 16.8 million, which is 256 times as many, on a 6 GB GPU with a three-day deadline. The price is real: 2D flow behaves differently from 3D flow, since vortices in 2D tend to merge into larger ones. So no result here transfers to 3D, and we don't claim that it does.

**5. Does doubt actually predict error?**
**RESULT PENDING** (`results/exp_doubt.json`). The figure to read is the within-time Spearman on Test B, with its interval. Then the same for Tests A and C, within-time AUROC, and the comparison with the no-ensemble signal.

**6. Does doubt-based hand-off beat fixed and random schedules?**
**RESULT PENDING** (`results/exp_handoff.json`). The headline is Test B, at matched solver budget, with the threshold frozen on validation. "Beats" means the paired interval lies wholly below zero, against a baseline that is matched or above budget.

**7. What happens if doubt doesn't predict error, or doesn't beat the baselines?**
It gets reported as a finding, with an analysis of why. The design says so before any result. Up to one further signal may be tried, chosen on validation only and disclosed. "A cheap ensemble spread didn't help here" is still an answer to the question.

**8. Is your network faster than the solver?**
**RESULT PENDING** (`results/exp_accuracy.json` for the timing, `results/demo_bench.json` for the demo). The figure has to include the cost of all five members. A single forward pass in `results/throughput.json` can't answer this, because it is one model on a batch.

**9. How accurate is it?**
**RESULT PENDING** (`results/exp_accuracy.json`). All we have now is the architecture selection: one U-Net on validation over 20 steps, 0.0915 against 0.6689 for persistence. That is a single model on a short rollout, using a different measure from the primary one. It isn't the ensemble's accuracy, so don't offer it as that.

**10. Isn't an ensemble of five just five times the cost? Doesn't that eat the speed gain?**
Yes, it is five forward passes per frame, and possibly it does. That is why the design forbids a speed claim that leaves out the ensemble's cost. Whether anything is left is **RESULT PENDING** (`results/exp_accuracy.json`). Ablation also looks at sizes 2 and 3 (`results/exp_ablation.json`).

**11. Can't all five members be wrong in the same way?**
Yes. They share an architecture, training data and loss. They can all make the same mistake with high confidence, and spread would stay low. Stepping from a shared mean makes this easier to miss, since a shared error is passed on to all of them. Doubt can only flag disagreement, so this is a real limit of the method. The oracle comparison and the calibration show how much is missed. **RESULT PENDING** (`results/exp_handoff.json`, `results/exp_doubt.json`).

**12. Why not MC dropout, or a Bayesian network, or a network that predicts its own variance?**
Other uncertainty methods were alternatives, and the design chose ensembles because they are simple and well studied, and their costs can be measured. We didn't compare them here. The Deep Ensembles paper itself prefers variance training through NLL to what we do, on a toy task. We didn't test that either. That is a fair criticism and a good next experiment.

**13. The paper you cite for ensembles says your version is inferior. Why use it?**
It says so on a toy regression task, and we state that in `docs/REFERENCES.md`. We cite it for seeds and shuffling and for the general idea, and we say that ours is the MSE variant. We kept MSE because it matches our loss and rollouts, and it adds no extra output. Whether the difference matters here hasn't been measured. Lakshmanan and Chopra also report ensembles losing to step-doubling on their own systems, which says nothing about ours. We mention it, and don't cite that paper only as support.

**14. How do you know your solver is right?**
We check it, within limits. Tests in `tests/physics/test_solver.py` show that projection removes divergence, no flow enters obstacles, runs are deterministic, known vortex behaviour holds, and results move toward the fine grid with refinement over a stated window. `results/solver_check.json` holds the convergence table. It wasn't compared with a physical experiment or a published benchmark, and 256×256 wasn't compared with a finer grid. So "right" means right for what those tests check. It also has numerical dissipation.

**15. Your reference is chaotic. What does an error against it even mean?**
This is the most serious limit. In `results/solver_check.json`, the 128×128 grid stops being clearly closer to 256×256 than 64×64 is at about `t=1` on a hard-edged inflow, and about `t=2` on a smooth one. The ratios are 1.085 at `t=1` (hard) and 0.877 at `t=2` (soft), and the order reverses at `t=2.5` on the soft inflow, at 1.009. The primary window runs from `t=0.8` to `t=2.8`. So past about `t=1`, the reference is one version of the flow, not exact truth, and error partly measures how far two equally good solvers would drift. That is why the 128×128 solver's error is reported on the same axes to give that level, and why the primary metric is never read without the physical measures. This limit is in the preregistration, and it was written before any model existed.

**16. Can a hand-off actually fix anything if the solver starts from a wrong state?**
It can't return to the reference path. It can remove divergence, restore physically consistent behaviour and slow the growth of error. Point-by-point error may therefore undersell it at long horizons, which is why vorticity, spectrum and plume statistics are also reported. Whether it helps at all is **RESULT PENDING** (`results/exp_handoff.json`).

**17. Did you look at the test sets?**
Once, before any model existed: the spot-check rendered fixed scenes and checked statistics, which is the exception written in the preregistration. `--final` is meant to run once, after all code is committed. Any later fix is logged with the numbers before and after, and the first numbers stay on record. Selection and tuning use validation. `results/spot_check.json` shows `git_dirty: true`, because the manifest's tracked copy and the spot-check's own results are committed after the check runs (`docs/DECISIONS.md`, B37). The file also records `manifest_git_dirty: false`, for the commit that generated the data.

**18. Can't you edit a preregistration afterwards? Who checks it?**
Yes, in principle. It is our own repo. Git history shows its commits come before the data and the models, and the repo is public, so changes can be seen, but no outside body holds a dated copy. So it is good discipline and a visible record. It isn't a guarantee. The rules were tightened three times, each before any data or model existed, and each change is listed in its record.

**19. The FNO is the fashionable choice. Are you sure it should lose?**
Sure of this one comparison only: seed 0, 40 minutes each, validation, 0.1258 against 0.0915. There is no interval, as there was one seed. FNO has about nine times the parameters, and made more steps, so it wasn't short of either. We haven't tested why it lost. A wider or longer FNO might do better.

**20. Thirty test scenes is not many. Can you conclude anything?**
Only what the intervals allow, and they may be wide. On a single rollout time, with 30 scenes, a correlation as large as about 0.36 can arise by chance alone. Averaging over times helps less than it seems, since those times are related. The bootstrap resamples scenes for this reason, so that the interval shows the real amount of evidence. The dataset size was a choice within the time budget. If the interval includes zero, that is what we report.

**21. Only 240 training scenes. Isn't that small?**
Yes, for this kind of model. It suits a laptop GPU, and each scene gives many training steps, though neighbouring frames aren't independent. A small training set also makes unfamiliar scenes more likely to look unfamiliar, which suits the doubt question. It doesn't suit the accuracy question, and we wouldn't claim otherwise.

**22. Why does Test C mix up low and high sides? Can you separate input size from flow regime?**
No. In Test C, input amplitude and flow regime leave their ranges together, so a rise in doubt can't be divided between them (B20). We chose that because the flow depends on the product of buoyancy and inflow strength. Without tying them, about half the set would have been a training flow with rescaled density. All figures are per side and pooled. The low side's plumes often reach obstacles late, so its figures say more about an unfamiliar flow than about flow around obstacles.

**23. What would actually be useful about this outside your project?**
Honestly, little directly. It is a small 2D simulator, and it won't change aircraft design or weather forecasting. The value is as a measured small-scale answer to a question that matters in larger settings.

## Method

**24. Why correlate within a rollout time?**
Error grows with time, and doubt may too. Across all times, they would correlate even if doubt were only a clock. Fixing the time removes that, so the question becomes: among scenes at the same moment, does higher doubt mean higher error?

**25. Why Spearman and AUROC both?**
Spearman asks about order across all frames. AUROC asks about the decision a hand-off actually makes, bad or good. A signal could order the middle well and still miss the bad frames, or the reverse.

**26. Why bootstrap whole scenes?**
Frames in a scene are strongly related, so treating them as independent makes intervals too narrow. Scenes are the independent unit.

**27. Why cap errors at 10 and keep blow-ups?**
A blown-up frame has no meaningful error, but dropping it would flatter a failing model. Counting it as 10 keeps it in every average. The share of blown-up rollouts is also reported. The value 10 is a choice, fixed before any model existed.

**28. How was the threshold chosen, and could it be overfit?**
On validation only, by steps fixed in advance: narrow the aggregations by the evidence measure, try 8 percentiles of doubt with hand-off lengths of 2, 5 and 10, and pick the lowest error with a solver share at or below 25%. With up to 72 runs on 30 scenes, it could overfit validation somewhat. The test sets are there to show this. The choice itself is **RESULT PENDING** (`results/handoff_selection.json`).

**29. Why 25% as the solver budget cap?**
It is a choice, to keep the system mostly neural. With a much larger share, the comparison would mostly measure the solver. It wasn't derived from anything, and the sweep curves show the other shares for description. **RESULT PENDING** (`results/exp_handoff.json`).

**30. Giving baselines the budget from a test set: isn't that peeking?**
It reads how much solver time the frozen policy used, after the policy is fixed, and it chooses nothing about that policy. It only makes the comparison fair. The baselines' schedules are then worked out from the budget alone, before any rollout runs.

**31. Why isn't the oracle a true upper bound, as the design says?**
It decides one frame at a time from that frame's error. A schedule that looked ahead could do better, and handing off later might sometimes help more than handing off now. So doubt could score lower than the oracle. The preregistration says we would report that as it stands.

## Physics and engineering

**32. Your long-run divergence went above the plan's limit. Isn't that a failure?**
In `results/solver_check.json`, 91 of 300 steps at 256×256 were above the plan's 0.001, with a peak of 0.00272, and all were reported as converged. The plan's figure was set at 64×64. Our stopping rule is relative to flow speed, so faster flow is allowed a larger absolute figure (B13). The file records the limit that follows from that rule, 0.00394. So this follows from the rule. It is recorded and not hidden, though it does mean that "near zero" is relative to speed.

**33. `results/spot_check.json` names a scene passing the thinnest wall. Doesn't smoke leak through?**
It might, by a little. Semi-Lagrangian advection can jump a wall thinner than one step's travel. Scene 12 of val is named. A rule fixed in advance compares it at two step sizes against controls, and regenerates all the data on a failure. Read `results/travel_rule.json` and `results/figures/travel_rule_val_00012.png` before the viva, so that you can state the outcome. The rule has a known weakness: a whole-field comparison can miss a small local leak, so a pass means no large effect, not no leak. The factor of 2 is a choice.

**34. Why is your cell-centred divergence 48.567 when the solver says it is near zero?**
They are different measures on the same state. The solver makes the faces divergence-free, and on faces the plume's largest value is 0.00085. Averaging to centres and taking central differences uses a different operator, which doesn't preserve the projection. So it stays large on a correct field, especially by walls. It isn't a measure of how good the solve was.

**35. Is the network's output divergence-free?**
Not guaranteed. The penalty is soft. Solid cells are forced to zero, but nothing projects an output unless a hand-off follows. The secondary metrics measure divergence on every policy with the same operator. **RESULT PENDING** (`results/exp_accuracy.json`).

**36. Walk me through one solver step.**
Mask faces touching solids. Advect density and both velocity components by tracing back with a midpoint rule. Add smoke from the inflow, apply a light decay, and add buoyancy to `v`, using density averaged onto the faces. Then project, by measuring net flux, solving for the correction with preconditioned CG, and subtracting its gradient from the faces. Read `src/sandeha/physics/solver.py:SmokeSolver.step` before the viva, and be able to point to each line.

**37. Why batch size 16?**
`results/throughput.json` shows U-Net items per second going 220.43, 237.82, 250.23 at batches 8, 16, 32, with memory going 0.693, 1.363, 2.713 GB. Doubling from 16 to 32 adds little speed and doubles the memory. It is a reasonable middle, and batch size wasn't tuned for accuracy.

**38. What does the public dataset add? Is it real data?**
It tests the doubt measures on data we didn't generate, with 3 members. It is a published scientific simulation, not physical measurement. If the course requires measured real-world data, it doesn't meet that. It has no hand-off, as there is no solver for that physics in the project. The dataset card asks us to cite Fielding et al. 2020, so we cite that as well as The Well. **RESULT PENDING** (`results/exp_well.json`).

**39. Why would disagreement rise on shapes the network hasn't seen?**
The hope is that members, trained from different starts, agree where the data pins them down and vary where it doesn't. That is a hope, and Test B is what tests it. **RESULT PENDING** (`results/exp_doubt.json`, under `rise_on_unfamiliar`).

**40. Why use a signal that needs five networks, when divergence from one network is free?**
That is one of the comparisons. The no-ensemble signal is the RMS divergence of a single model's output, scored by the same measures. If it does as well, the ensemble isn't earning its cost as a doubt signal. **RESULT PENDING** (`results/exp_doubt.json`).

**41. What are the project's weakest points?**
State them plainly:

- Chaos limits what error against one reference means past about `t=1`.
- There are only 30 scenes per test set.
- Members can share mistakes.
- The solver wasn't checked against experiment.
- The world is 2D and simulated.
- Several thresholds are choices, not measurements.
- Each ablation also trains for less time, so it isn't a clean comparison.

**42. What would you do next?**
Train networks to predict their own variance, and compare that with spread. Compare with a residual signal, as ANCHOR uses. Try per-region hand-off. Use a finer reference, to reduce the chaos limit. Run more test scenes and more seeds.

**43. Why the name?**
Sandeha is Sanskrit for "doubt".

---

# Part 4: Demo script

About five minutes. The demo is a real model running live on the laptop's GPU, with no internet needed.

## Before the viva

| When | Do this |
|---|---|
| The day before | Run `uv run python scripts/bench_demo.py`, with the GPU otherwise idle. It writes `results/demo_bench.json` and picks the member count |
| The day before | Record the backup video from the real system, and say at its start that it is a recording. It doesn't exist yet |
| The day before | Run `uv run python scripts/make_figures.py`, and open each figure. Fill in the RESULT PENDING markers below from the result files |
| The day before | Read `docs/RESULTS.md`, and rehearse the weak numbers as well as the good ones |
| An hour before | Plug in the charger, and close anything using the GPU. `nvidia-smi` should show no other job |
| Ten minutes before | Start the server, open the page, and try each preset once |

Start it with:

```bash
uv run python scripts/run_demo.py
```

It prints the address and member count, normally `http://127.0.0.1:8000`. Open it in a browser.

## The script

### 1. The opening (30 seconds)

Say: "Neural simulators are fast, but they can go wrong without warning. This is a 2D smoke simulator run by an ensemble of neural networks. When they disagree too much, a conventional solver takes over. What we measure is whether their disagreement really predicts their errors."

### 2. What is on the screen (30 seconds)

Point to each:

- **Smoke**, on the left, and **Doubt**, on the right, showing where the members disagree.
- **The AI and PHYSICS lamps**, showing which is in control of this frame.
- **The doubt strip**, showing doubt over time, with the threshold line. The slider sets that line, and its top means never.
- **Frames per second** and **Frames by solver**.

Say how many members are running, which the page shows. Current value: RESULT PENDING (`results/demo_bench.json`). If it is below 5, say so, and say that the threshold was frozen at another ensemble size, whose doubt isn't on the same scale.

### 3. A shape from training (45 seconds)

Choose **Who runs the smoke: By doubt**, and press **Circle**.

Say: "This shape family is in the training data. The source and physics are in the middle of the training ranges."

Read out the frames per second from the screen. Target: at least 15 in AI mode. Measured: RESULT PENDING (`results/demo_bench.json`).

### 4. Shapes it has never seen (60 seconds)

Press **Star**, then **Letter**, then **Blob**.

Say: "These families were never in training. If doubt works, it should tend to rise on them. One live run can't show that, so we look at what happens, and I'll show the measured figures afterwards."

Point to the doubt map and the lamps, and describe what is actually on screen. Don't predict it in advance.

### 5. Draw something (45 seconds)

Choose **Brush** at **Medium** or **Wide**, and draw a shape across the plume while it runs. If the professor is willing, let them draw.

Say: "Nothing here is pre-rendered. The solid applies from the next step. If a hand-off is under way, the edit waits until it ends."

### 6. The threshold, and the cost of the solver (45 seconds)

Move the threshold slider down, then up to its top.

Say: "Lower means more hand-offs, more solver frames and lower frames per second. The slowdown you see is the real cost. The top means it never hands off. The slider is for showing the idea. The measured results used one threshold, frozen on validation."

Then choose **AI only**, and read the frame rate. Choose **Physics only**, and read the frame rate. Return to **By doubt**.

- AI only: RESULT PENDING frames per second (`results/demo_bench.json`).
- Physics only: read it from the screen, and say that it is a live reading, not a benchmark. The ensemble still runs on each decision in this mode, so it overstates a pure solver's cost.

### 7. The measured results (60 seconds)

Open the figures in `results/figures/`.

| Figure | Say | Number |
|---|---|---|
| `doubt_evidence.png` | "Within a fixed rollout time, does doubt rank frames by error?" | Within-time Spearman on Test B: RESULT PENDING, interval RESULT PENDING (`results/exp_doubt.json`) |
| `handoff_comparison.png` | "At the same solver budget on Test B, does doubt beat fixed and random schedules?" | Difference from fixed: RESULT PENDING. From random: RESULT PENDING. Intervals: RESULT PENDING (`results/exp_handoff.json`) |
| `accuracy_error_by_time.png` | "How does the error grow, against persistence and the 128×128 solver?" | Set score: RESULT PENDING. Coarse solver: RESULT PENDING (`results/exp_accuracy.json`) |
| `handoff_sweep.png` | "The whole speed and accuracy trade-off, for description only." | Solver share at the frozen threshold: RESULT PENDING (`results/exp_handoff.json`) |

Say the result as it is. If doubt didn't beat a baseline, say so here, and say why that is still an answer.

### 8. The close (15 seconds)

Say: "So the question was whether a neural simulator's doubt predicts its errors, and what hand-off costs. On Test B, the answer we measured was RESULT PENDING (`results/exp_handoff.json`). It's a small 2D system, and these results apply to it, not to larger ones."

## If something fails

Stay calm and say what happened. A failure explained well is better than one hidden.

| What happens | What it means | What to do |
|---|---|---|
| `results/ensemble.json is missing` or `says the ensemble isn't complete` | Training hasn't finished, or the file isn't there | Use `--random-weights`. The page then says "Untrained models: this smoke and doubt are not results". Read that out, show only the mechanism, and use the figures for results |
| `results/handoff_selection.json is missing` | Selection hasn't run | The same as above, or run `scripts/select_handoff.py` beforehand. Don't run it during the viva |
| A checkpoint `is missing or doesn't match its checksum` | The weights aren't the measured ones | Don't work around it. Restore them from the GitHub Release, or use the backup video |
| `No GPU found` | The driver or CUDA isn't available | Run with `--device cpu`, which is slow on purpose, and say so. Otherwise use the backup video |
| `The demo ran out of GPU memory` | Something else is using the GPU | Close it. If that fails, use `--members 3`, and say that the page now runs fewer members, and that the frozen threshold came from another ensemble size |
| Address already in use | Another server is on port 8000 | Stop it, or run with `--port 8001` |
| The page says "The demo server isn't answering" | The server stopped | The page retries every second. Start `scripts/run_demo.py` again, and press **Reset** |
| The page is blank or looks broken | A browser problem | Try another browser. The page needs no internet, as its fonts are local |
| Frame rate is under the target | Battery power, heat or a busy GPU | Check that the charger is in, and close other jobs. Then state the number on screen. Fewer members is the planned fallback, and has to be disclosed |
| Doubt never crosses the line | The scene isn't raising doubt above the frozen threshold | Say so. Press another shape, or lower the slider and say you have moved it off the frozen value. Don't imply that this was the measured setting |
| Hand-offs never stop | Doubt is above the line for this scene | Say so, and raise the slider if needed, as above. Point to the frame rate as the cost |
| Doubt doesn't rise on a star | One run shows nothing either way | Say: "One scene can't settle this. The measured figure is over 30 scenes", and go to the figure |
| Smoke passes through a thin brush stroke | A known limit: semi-Lagrangian advection can jump thin walls | Say so, and use a wider brush. Don't call it a model error |
| Smoke goes wild for a moment, then a solver frame shows | A non-finite output: the server goes back to the last finite frame with a hand-off, or to an empty box if that fails too | Say what it is: the network blew up, and the system recovered. Then press **Reset** |
| The laptop or the whole demo fails | | Play the backup video, and say it is a recording. Then use the figures in `results/figures/` and `docs/RESULTS.md` |
| A question you can't answer | | Say "I don't know", and say what you would check and where. Don't guess a number |

## Rules while speaking

- Say "measured", and name the set, for every figure.
- Don't turn a live demo into evidence. The figures are the evidence.
- Don't call any threshold "tuned on the test set", because none was, apart from the oracle's, which is descriptive.
- Don't say "real data" for the public track. Say "a published simulation benchmark".
- Don't call the approach new. Say "we found no paper doing exactly this, in a limited search".
