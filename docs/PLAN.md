# Sandeha Implementation Plan

**Goal:** Build, measure and demonstrate an ensemble neural smoke simulator that hands off to a physics solver when its own doubt is high.

**Architecture:** A GPU PyTorch solver generates fine-grid truth. Neural models learn one coarse step, and an ensemble turns their disagreement into a doubt signal. A hybrid simulator uses that signal to choose between AI and solver. Experiments write JSON that figures and `RESULTS.md` are generated from, and a local WebSocket server drives the browser demo.

**Tech stack:** Python 3.12 via `uv`, PyTorch (CUDA), NumPy, SciPy, h5py, huggingface_hub, FastAPI, uvicorn, matplotlib, pytest, Playwright (dev only). Front end: plain HTML, CSS, JS and canvas, with all assets served locally.

**Spec:** `docs/DESIGN.md`. Read it first. Where this plan is silent, the design decides.

## Global Constraints

- Git author `Adwaith R Nair`, email `adwaith.r.nair189@gmail.com`, set in repo config. Commit messages carry no co-author or tool-attribution lines, enforced by a `commit-msg` hook.
- Docs state nothing about how the code was authored. Tooling config directories and files stay untracked.
- GitHub repo `Adwaith-R-Nair/sandeha`, public. Push at the end of every phase.
- Never commit datasets or checkpoints. Weights go in a GitHub Release.
- GPU memory 6 GB. Reference grid 256×256, AI grid 128×128, AI step 4 reference solver steps, ensemble of 5, public-track ensemble of 3.
- Test splits are read only by final evaluation, and tuning uses validation only.
- Every number in a document comes from a saved JSON result. Never type a result by hand.
- Code and experiment freeze: morning of 2026-11-04. Priority if time runs short: ensemble, main experiment, demo, public track, ablations.
- Long jobs run under `scripts/long_job.sh`, which holds a sleep and lid inhibitor, refuses to start without mains power, and appends start and end times to `runs/jobs.log`.
- Every changed decision goes in `docs/DECISIONS.md` and `docs/BUILD_LOG.md`.

## Review Focus

1. **Rollouts that blow up (NaN or huge values):** must count as failures with maximum error, never be dropped, as dropping them would flatter the results. Tested in Task 9.
2. **A user seals off or fills the domain, or draws over an inflow:** the solver stays finite, and the AI and demo keep running. Tested in Tasks 2 and 11.
3. **A training job is killed mid-checkpoint:** it resumes from the last good checkpoint with no corrupted file. Tested in Task 5.
4. **Edits arrive during a slow solver hand-off, or the browser disconnects and reconnects:** edits apply on the next frame, and the server survives. Tested in Task 11.
5. **GPU memory is exhausted or the GPU is missing at demo start:** a clear message and clean exit, not a hang. Tested in Task 11.

---

## Layout

```
configs/                 data.yaml  train_unet.yaml  train_fno.yaml  eval.yaml  well.yaml
scripts/                 long_job.sh  generate_data.py  train.py  select_arch.py  run_experiments.py
                         make_figures.py  render_results.py  train_well.py  eval_well.py  run_demo.py  bench_demo.py
src/sandeha/
  physics/solver.py      SmokeSolver, SmokeState
  physics/scenes.py      Scene, sample_scene, shape families
  data/store.py          on-disk format, split guard
  data/dataset.py        StepDataset, Normalizer
  models/unet.py         UNet
  models/fno.py          FNO2d
  models/factory.py      build_model
  train/losses.py        step_loss, divergence
  train/loop.py          train(), checkpoints
  ensemble.py            Ensemble, doubt
  hybrid.py              HybridSimulator, policies
  eval/metrics.py        errors, ranks, AUROC, bootstrap
  eval/rollout.py        rollouts and timing
  eval/experiments.py    accuracy, doubt, handoff, ablation experiments
  well/data.py           public-dataset loader
  demo/server.py         FastAPI app and simulation loop
  demo/static/           index.html  app.js  style.css  fonts/
tests/                   mirrors src/
results/                 JSON results and figures (tracked)
data/ runs/ checkpoints/ (untracked)
docs/                    DESIGN PLAN PRD HLD LLD BUILD_LOG DECISIONS TEST_PLAN RESULTS MODEL_CARD VIVA_GUIDE REPORT
```

---

## Phase 1: Foundation

### Task 1: Repo, environment, docs skeleton

**Files:** `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `scripts/long_job.sh`, `tests/test_env.py`, `docs/{PRD,HLD,LLD,BUILD_LOG,DECISIONS,TEST_PLAN}.md`

**Produces:** an installed `sandeha` package, a `uv run pytest` command, and the long-job wrapper.

- [ ] `git init -b main`. Set `user.name` and `user.email`. Install the `commit-msg` hook that rejects `Co-Authored-By` and tool attribution.
- [ ] `.gitignore`: `/data/ /runs/ /checkpoints/ .venv/ __pycache__/`, anchored to the root so `src/sandeha/data/` stays tracked. Local tooling paths go in `.git/info/exclude`.
- [ ] `uv` project on Python 3.12, with PyTorch's CUDA build matching driver 595.
- [ ] Write `tests/test_env.py::test_cuda_runs_a_matmul`: on `cuda`, multiply two 1024×1024 matrices and assert the result is finite. Run it and see it pass.
- [ ] `scripts/long_job.sh <name> <cmd...>`: exit 3 with a message if mains power is offline, otherwise run under `systemd-inhibit --what=sleep:idle:handle-lid-switch` and log to `runs/<name>.log` and `runs/jobs.log`.
- [ ] Write the PRD, HLD and LLD from the design, including an HLD diagram, and `TEST_PLAN.md`. Start `BUILD_LOG.md` and `DECISIONS.md`, with the design's defaults recorded as decisions.
- [ ] Commit. Create the public GitHub repo and push.

---

## Phase 2: Physics

### Task 2: Reference solver

**Files:** `src/sandeha/physics/solver.py`, `tests/physics/test_solver.py`

**Produces:**
```python
@dataclass
class SmokeState:
    velocity: Tensor   # (B,2,H,W), cell-centred view for ML; solver may use MAC internally
    density: Tensor    # (B,1,H,W)

class SmokeSolver:
    def __init__(self, res: int, device, dt: float, cg_tol: float = 1e-5, cg_max_iter: int = 2000): ...
    def step(self, state: SmokeState, obstacle: Tensor, inflow: Tensor, buoyancy: Tensor) -> SmokeState: ...
    def project(self, state: SmokeState, obstacle: Tensor) -> SmokeState: ...
    def divergence(self, state: SmokeState, obstacle: Tensor) -> Tensor: ...
```
Closed box walls. A staggered (MAC) grid inside, semi-Lagrangian or MacCormack advection, a preconditioned conjugate gradient pressure solve per batch item, and light density decay so that smoke doesn't saturate the closed box. Batched, so data generation stays fast. Tensors `obstacle` and `inflow` are `(B,1,H,W)`, `buoyancy` is `(B,)`.

- [ ] Write the failing tests:
  - `test_projection_removes_divergence`: random velocity at 64², max absolute divergence in fluid cells below `1e-3` after `project`.
  - `test_no_flow_into_obstacles`: after 50 steps, velocity in solid cells is 0 and density in solid cells is 0.
  - `test_deterministic`: two runs give identical tensors.
  - `test_buoyancy_lifts_smoke`: density centroid rises over 100 steps from a low inflow.
  - `test_converges_with_refinement`: after a fixed simulated time, the 128² vs 256² downsampled density error is smaller than the 64² vs 256² error.
  - `test_sealed_cavity_stays_finite`: a closed ring of obstacle with velocity inside, and a fully solid domain, both stay finite for 20 steps.
- [ ] See them fail, implement, and see them pass.
- [ ] Measure and record 256² and 128² steps per second, with batch sizes 1 and 8, in `BUILD_LOG.md`.
- [ ] Render a 256² test run to an image strip in `results/figures/solver_check.png`, and have it inspected to confirm that the plume rises and flows around an obstacle.
- [ ] Commit.

### Task 3: Scenes

**Files:** `src/sandeha/physics/scenes.py`, `tests/physics/test_scenes.py`

**Produces:**
```python
@dataclass
class Scene:
    obstacle: np.ndarray        # (res,res) bool at reference grid
    inflow: np.ndarray          # (res,res) float rate
    buoyancy: float
    late_obstacle: np.ndarray | None   # obstacle added mid-run
    late_frame: int | None

TRAIN_SHAPES = ("circle", "rectangle", "wall")
UNSEEN_SHAPES = ("star", "letter", "blob")
def sample_scene(rng: np.random.Generator, res: int, split: str) -> Scene: ...
```
`split` is one of `train val test_a test_b test_c`. Ranges: training buoyancy and inflow strength span the middle of the physical range, and `test_c` samples from outside it on both sides. A third of `train`, `val` and `test_a` scenes have a `late_obstacle`.

- [ ] Write failing tests: `test_train_never_uses_unseen_shapes`, `test_test_b_only_uses_unseen_shapes`, `test_test_c_outside_train_ranges`, `test_inflow_not_blocked` (inflow cells aren't solid at the start), `test_sampling_is_seeded`.
- [ ] Implement, see tests pass, and commit.

---

## Phase 3: Data

### Task 4: Generation and test-set lock

**Files:** `src/sandeha/data/store.py`, `src/sandeha/data/dataset.py`, `scripts/generate_data.py`, `configs/data.yaml`, `docs/PREREGISTRATION.md`, tests

**Produces:**
```python
def open_split(root: Path, split: str, final_eval: bool = False) -> SplitStore: ...
# SplitStore: .fields (memmap float16: scene,frame,3,128,128), .scene(i) -> Scene downsampled,
#             .obstacle(i, frame), .inflow(i), .buoyancy(i), .fine_state(i, frame) only if stored

class Normalizer:  # means and stds from train only, saved in data/normalizer.json
    def encode(x: Tensor) -> Tensor; def decode(x: Tensor) -> Tensor

class StepDataset(Dataset):
    def __init__(self, store, normalizer, unroll: int = 1): ...
    # item: dict(input=(6,128,128), target_seq=(unroll,3,128,128), obstacle, inflow, buoyancy)
```
Input channels, in this order: `u, v, density, obstacle, inflow, buoyancy`.

Starting sizes, scaled down if the measured solver speed would exceed the 1.5 h budget: train 240, val 30, each test 30 scenes, 121 frames. Scenes are split at scene level.

- [ ] Write failing tests:
  - `test_open_split_refuses_test_without_final_eval`: `open_split(root, "test_b")` raises `PermissionError`.
  - `test_manifest_detects_tampering`: a changed test file fails the checksum check in `open_split`.
  - `test_no_scene_shared_between_splits`: seeds are disjoint.
  - `test_resume_generation`: an interrupted run continues and matches an uninterrupted one.
  - `test_normalizer_uses_train_only`.
  - `test_item_shapes`: channels and shapes as above.
- [ ] Implement. Writes are atomic per scene. `data/manifest.json` records seeds, config, git commit and SHA-256 of every split.
- [ ] Before generation, write `docs/PREREGISTRATION.md`. It fixes: primary metric, horizon 50 AI steps from start frame 20 (extended only if solver checks show the plume hasn't reached typical obstacles by then), secondary metrics, within-time correlation as the evidence measure, bad-frame threshold rule, threshold rule (Task 8), baseline budget rule, and test-set policy. Commit it **before** any model exists.
- [ ] Run `scripts/long_job.sh data uv run scripts/generate_data.py --config configs/data.yaml`.
- [ ] Spot-check: render a sample scene from each split, and check stats, NaN count (must be 0) and the manifest. Log timings. Commit.

---

## Phase 4: Models

### Task 5: Models, losses, resumable training

**Files:** `src/sandeha/models/{unet,fno,factory}.py`, `src/sandeha/train/{losses,loop}.py`, `scripts/train.py`, `configs/train_{unet,fno}.yaml`, tests

**Produces:**
```python
def build_model(cfg: dict) -> nn.Module           # forward: (B,6,H,W) -> (B,3,H,W), the normalised delta
def apply_step(model, x: Tensor, normalizer) -> Tensor   # next (B,3,H,W), with solid cells forced to zero
def divergence(velocity: Tensor, obstacle: Tensor) -> Tensor    # central differences, fluid cells only
def step_loss(pred, target, obstacle, lambda_div: float) -> dict
def train(cfg: dict, run_dir: Path, resume: bool) -> Path       # returns best checkpoint path
```
U-Net: base width 32, 4 levels, GroupNorm, SiLU. FNO: width 48, 24 modes, 4 layers, with coordinate channels because the box isn't periodic. Loss: MSE on normalised delta plus `lambda_div=0.1` × mean squared divergence. Pushforward stability: after the warm-up, sometimes unroll 1 to 3 steps with no gradient, and only the last step gets gradients. AdamW, lr `3e-4`, cosine schedule, mixed precision. `max_minutes` config sets the total steps from a measured throughput, so the schedule finishes on time. Best checkpoint is chosen by validation rollout error over 20 steps.

- [ ] Write failing tests:
  - `test_model_output_shape` for both.
  - `test_solid_cells_zero_after_apply_step`.
  - `test_divergence_of_uniform_flow_is_zero`.
  - `test_overfits_one_batch`: loss drops by at least 10× in 200 steps.
  - `test_checkpoint_atomic`: an interrupt during save leaves the previous checkpoint loadable.
  - `test_resume_continues`: killing at step N and resuming reaches the same step count, and restores optimizer and RNG state.
  - `test_peak_memory_under_budget`: U-Net at batch 16, peak memory below 5.5 GB.
- [ ] Implement, see tests pass, and commit.
- [ ] Measure throughput for both models, and pick the batch size. Log it.

### Task 6: Architecture choice and ensemble

**Files:** `scripts/select_arch.py`, `results/arch_selection.json`

- [ ] Train U-Net and FNO for 40 minutes each, seed 0. Compare rollout error on validation only. Write `results/arch_selection.json`. Record the winner in `DECISIONS.md`.
- [ ] Train the ensemble under `long_job.sh`: seeds 0 to 4, 48 minutes each, with the winning architecture. Reuse nothing but the architecture. Log each finished member. If time runs short, stop at 3 and record this as a change.
- [ ] Check each member: finite rollouts over 100 validation steps and an improvement over persistence. A failing member gets retrained once with its seed + 100, with the reason logged.
- [ ] Ablations, last in the compute queue: 25 minutes each, seed 0, without divergence loss, and without pushforward.
- [ ] Commit configs and selection JSON.

---

## Phase 5: Doubt and hand-off

### Task 7: Ensemble and doubt

**Files:** `src/sandeha/ensemble.py`, `tests/test_ensemble.py`

**Produces:**
```python
class Ensemble:
    def __init__(self, models: list[nn.Module], normalizer): ...
    def step(self, x: Tensor) -> EnsembleOut   # .mean (B,3,H,W), .doubt_map (B,1,H,W), .members
def doubt_score(doubt_map, obstacle, method: Literal["mean","p95","p99"]) -> Tensor  # (B,), fluid cells only
```
`doubt_map` is the square root of the per-channel member variance, in normalised units, averaged across channels. Every member steps from the shared mean (the design's choice). `independent_rollout` in Task 9 provides the secondary comparison.

- [ ] Failing tests: `test_identical_members_zero_doubt`, `test_disagreement_raises_doubt`, `test_mean_matches_manual_mean`, `test_score_ignores_solid_cells`, `test_one_member_doubt_is_zero` (with `size 1`, the single-model policy).
- [ ] Implement, see tests pass, and commit.

### Task 8: Hybrid simulator and policies

**Files:** `src/sandeha/hybrid.py`, `tests/test_hybrid.py`

**Produces:**
```python
class Policy:
    def decide(self, frame: int, doubt: float, ctx: dict) -> bool: ...  # True = hand off now
class AlwaysAI(Policy); class AlwaysSolver(Policy); class FixedSchedule(Policy)  # period
class RandomBudget(Policy)  # rate, seed
class DoubtThreshold(Policy)  # tau
class Oracle(Policy)  # uses true error from ctx, tau

class HybridSimulator:
    def __init__(self, ensemble, solver, policy, handoff_frames: int, substeps: int = 4): ...
    def reset(self, state, obstacle, inflow, buoyancy): ...
    def set_obstacle(self, obstacle): ...
    def step(self) -> StepInfo   # .state, .doubt_map, .doubt, .source ("ai"|"solver"), .ms, .frames_advanced
```
Hand-off: upsample the 128² state to 256², `project`, run `4 × handoff_frames` solver steps, emit each frame, downsample, then AI resumes. Solver and AI both step in the same physical time units. Timing calls `torch.cuda.synchronize()`.

Selection on validation only, and recorded in `results/handoff_selection.json`:
- `handoff_frames` from `{2, 5, 10}` and doubt aggregation from `{mean, p95, p99}`, by best rank correlation and error at equal budget.
- **Frozen threshold `tau`:** the value giving lowest primary error with solver share at or below 25% on validation.

- [ ] Failing tests: `test_handoff_returns_low_divergence_state`, `test_fixed_schedule_period`, `test_random_budget_rate_within_tolerance`, `test_threshold_triggers_only_above_tau`, `test_oracle_never_reads_future`, `test_obstacle_change_applies_next_frame`, `test_time_accounting_includes_every_member`.
- [ ] Implement, see tests pass, and commit.

---

## Phase 6: Experiments

### Task 9: Metrics, rollouts and experiments

**Files:** `src/sandeha/eval/{metrics,rollout,experiments}.py`, `scripts/run_experiments.py`, `scripts/make_figures.py`, `scripts/render_results.py`, `configs/eval.yaml`, `results/*`, `docs/RESULTS.md`, tests

**Produces:** `results/exp_accuracy.json`, `exp_doubt.json`, `exp_handoff.json`, `exp_ablation.json`, `results/figures/*.png`, each JSON carrying git commit, data manifest hash, checkpoint hashes and config.

```python
def nrmse(pred, ref, obstacle) -> float        # normalised by reference std, fluid cells
def bootstrap_ci(values_by_scene, n=2000) -> tuple[float,float]   # resampling scenes
def spearman(x, y) -> float; def auroc(score, label) -> float
def rollout(sim, scene, ref, start: int, horizon: int) -> RolloutTrace
```

Experiments, per design sections 6.1 to 6.4:
- **Accuracy:** every rollout length, usable-time, vorticity correlation; against persistence, coarse 128² solver, and timed reference.
- **Doubt:** pooled and per-rollout-time Spearman, AUROC, calibration, risk-coverage, Tests A vs B vs C, divergence-of-single-model comparison, independent rollouts as secondary.
- **Hand-off:** the frozen `tau` on each test set, with fixed and random baselines matched to the solver share it actually used on that set. Also sweeps for all policies, marked descriptive. Oracle bound.
- **Ablations:** the 2 reduced trainings, and ensemble sizes 2, 3, 5 from existing members.

- [ ] Failing tests:
  - `test_nrmse_zero_for_identical`, `test_spearman_known`, `test_auroc_known`, `test_bootstrap_contains_mean`.
  - `test_blown_up_rollout_counts_as_max_error`: NaN state gives capped maximum error, and the frame stays in the average.
  - `test_results_carry_provenance`.
  - `test_render_results_refuses_missing_provenance`.
  - `test_experiments_use_val_unless_final`: without `--final`, only `val` is opened.
- [ ] Implement, see tests pass, and commit.
- [ ] Dry-run everything on validation with 3 scenes. Fix bugs, then run hand-off selection on validation and commit the selection.
- [ ] Commit the code. Record the commit hash in `PREREGISTRATION.md`, then run once with `--final` under `long_job.sh`.
- [ ] **Any bug fix after `--final` runs gets logged in `BUILD_LOG.md`, with what changed and the before and after numbers.**
- [ ] Generate figures and the numbers in `RESULTS.md`. Write the analysis, including all weak or failed hopes from the design's section 6.5. Commit and push.

### Task 10: Public-dataset track

**Files:** `src/sandeha/well/data.py`, `scripts/train_well.py`, `scripts/eval_well.py`, `configs/well.yaml`, `results/exp_well.json`, tests

```python
class WellStepDataset(Dataset):  # h5py reader on official splits; input and next frame, normalised from train
    def __init__(self, root: Path, split: Literal["train","valid","test"], final_eval: bool = False): ...
```
Download `polymathic-ai/turbulent_radiative_layer_2D` from Hugging Face to `data/well/`, read with h5py, and don't execute anything in the download. Train 3 U-Nets at 40 minutes each.

- [ ] Failing tests with a small synthetic HDF5 in the same layout: `test_reads_expected_shapes`, `test_test_split_guarded`. After download, `test_real_file_shapes` confirms grid, steps and trajectories, skipped when data is absent.
- [ ] Implement, download, train under `long_job.sh`.
- [ ] Evaluate doubt with the section 6.2 measures on the official test split. Label it everywhere as a published simulation benchmark, not measured data.
- [ ] Commit.

---

## Phase 7: Demo

### Task 11: Demo server

**Files:** `src/sandeha/demo/server.py`, `scripts/run_demo.py`, `scripts/bench_demo.py`, tests

WebSocket `/ws`. Server to client: a binary frame with a JSON header (frame, source, doubt, tau, fps, solver_share, mode) and `uint8` density, doubt and obstacle at 128². Client to server: JSON messages `brush{x,y,r,erase}`, `preset{name}`, `reset`, `threshold{tau}`, `mode{auto|ai|solver}`. Presets include unseen shapes. Simulation runs in one thread, with the latest frame sent to each client and older frames dropped.

- [ ] Failing tests, using the FastAPI test client:
  - `test_ws_streams_frames`.
  - `test_brush_changes_obstacle_next_frame`.
  - `test_edit_during_handoff_applies_after`.
  - `test_reconnect_keeps_running`.
  - `test_fills_domain_stays_finite`: brush covers everything, and no crash.
  - `test_blown_up_state_recovers`: NaN in state triggers a solver hand-off and the indicator shows it.
  - `test_no_gpu_exits_with_message`, and `test_oom_exits_with_message`: both print a clear message and exit non-zero.
- [ ] Implement, see tests pass, and commit.
- [ ] `scripts/bench_demo.py`: measure AI-mode FPS with 5 members for 60 s. Write `results/demo_bench.json`. If under 15 FPS, follow the fallback ladder, and show member count on screen.

### Task 12: Demo front end

**Files:** `src/sandeha/demo/static/{index.html,app.js,style.css,fonts/*}`, `tests/demo/test_ui.py`

Smoke and doubt side by side at equal size, with a very clear AI or PHYSICS indicator, and live FPS and solver share. Tools: brush, erase, presets, threshold slider, mode control, reset. Must read as a precise instrument rather than a dashboard template, and be understood within 10 seconds. Also usable with touch, keyboard focus, and at 1280×720 for projectors. No network requests except to localhost.

- [ ] Failing Playwright tests against a running server: `test_canvas_updates_over_time`, `test_drawing_changes_obstacle_layer`, `test_indicator_switches_in_solver_mode`, `test_no_external_requests`, `test_layout_fits_1280x720`.
- [ ] Implement, see tests pass, and inspect screenshots in light of the design's quality bar. Iterate on anything that looks generic or is hard to read.
- [ ] Record the real system with Playwright video, save to a tracked `docs/assets/`, keep it under 50 MB, and make a README GIF. Commit and push.

---

## Phase 8: Hand-over

### Task 13: Final documents and release

**Files:** all of `docs/`, `README.md`

- [ ] Verify all references against the original papers. Remove anything unverified.
- [ ] Complete `HLD`, `LLD`, `DECISIONS`, `BUILD_LOG`, `TEST_PLAN`, `RESULTS`, `MODEL_CARD`.
- [ ] Write `VIVA_GUIDE.md`: each concept in plain language; line-of-reasoning for each design choice; 30 or more likely questions with honest answers, including hard ones about weak results; demo script with what to do if something fails.
- [ ] Write `REPORT.md`, with standard structure: abstract, intro, related work, methods, experiments, results, discussion, limits, conclusion, references.
- [ ] README: pitch, GIF, results, one-command setup and demo.
- [ ] From a clean clone in a temp directory: install, download Release weights, run tests, start the demo. Fix what fails.
- [ ] Audit: `git log --format='%an <%ae>%n%B'` has only the owner's identity and no attribution lines. A grep of tracked files finds no mention of the tooling used. Each number in `REPORT.md` and `README.md` appears in `results/`.
- [ ] Upload weights to a GitHub Release. Commit and push.
