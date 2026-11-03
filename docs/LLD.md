# Sandeha: Low-Level Design

**Status:** matches the code as built. Files marked "written when run" don't exist yet, because no experiment has run. Experiment rules are in `PREREGISTRATION.md`, which decides where it is more exact than this file.

## Conventions

- Reference grid 256×256, neural grid 128×128. `dt=0.01`. One frame is one neural step, or 4 solver steps. A scene has frames 0 to 120.
- Field tensors are `(B,C,H,W)`. Index `(row, column)` is `(y, x)`, rows grow upward, and plots need `origin="lower"`.
- The solid in force from frame `f` to `f+1` is `obstacle(i, f)`. A frame's fields are zero in the solid on the way into it.
- Saved writes are atomic: a temporary file, then a rename.

## Input and output channels

`Normalizer.make_input(fields, obstacle, inflow, buoyancy)` builds the input, `(B,6,H,W)`, in this order.

| Index | Channel | Encoding |
|---|---|---|
| 0, 1, 2 | `u`, `v`, density | `(x - field_mean) / field_std`, per channel |
| 3 | obstacle | 0 or 1, unscaled, the solid in force from this frame to the next |
| 4 | inflow | `inflow / inflow_scale`, so zero stays zero |
| 5 | buoyancy | `(b - buoyancy_mean) / buoyancy_std`, repeated over the grid |

The model's output is `(B,3,H,W)`: the change in `u`, `v` and density over one frame, divided by `delta_std`, which is the channel's RMS change around zero. `apply_step` turns it into the next frame:

```text
next = decode(x[:, :3]) + decode_delta(model(x))
next = 0 in cells where x[:, 3] is nonzero
```

Solid cells are filled, not multiplied, so a NaN can't stay inside a solid. The output runs through `.float()` first, because half precision is too coarse for the state.

The normaliser is fitted on fluid cells of `train` only, and `fit_normalizer` refuses any other split. `Normalizer.load` checks the file against its checksum in the manifest.

## Physics

### `physics/solver.py`

```python
@dataclass
class SmokeState:
    velocity: Tensor   # (B,2,H,W), cell-centred view of the faces in mac
    density: Tensor    # (B,1,H,W)
    mac: tuple[Tensor, Tensor] | None = None   # u (B,1,H,W+1), v (B,1,H+1,W)

@dataclass
class PressureReport:
    iterations: Tensor; residual: Tensor; converged: Tensor   # each (B,)

class SmokeSolver:
    def __init__(self, res: int, device, dt: float, cg_tol: float = 1e-05, cg_max_iter: int = 2000)
    def step(self, state: SmokeState, obstacle: Tensor, inflow: Tensor, buoyancy: Tensor) -> SmokeState
    def project(self, state: SmokeState, obstacle: Tensor) -> SmokeState
    def divergence(self, state: SmokeState, obstacle: Tensor) -> Tensor   # (B,1,H,W), zero in solids
    def preconditioner(self, obstacle: Tensor) -> Callable[[Tensor], Tensor]
    last_solve: PressureReport | None
```

| Item | Detail |
|---|---|
| Units | Unit-square box, cell width `1/res`, velocity in domain lengths per unit time |
| Walls | Closed box. Faces on walls or touching a solid cell are zero |
| Step order | Mask faces with the current obstacle; advect density, `u` and `v`; add `dt × inflow` in fluid cells; apply decay; add `dt × buoyancy × density` to `v`; project |
| Advection | Semi-Lagrangian with a midpoint backtrace and bilinear lookup. Density can't go negative |
| Decay | `DENSITY_DECAY = 0.1` per unit time |
| Pressure | Conjugate gradients with one symmetric multigrid V-cycle as preconditioner. Each batch item stops on its own |
| Convergence | No cell's net flux above `cg_tol` × the item's fastest face, measured from the faces. In physical units the limit grows with grid size and speed, so a converged 256×256 solve doesn't guarantee divergence below `1e-3` |
| Stopping otherwise | A pass that fails to halve the divergence, `cg_max_iter`, or a non-finite input |
| Non-finite input | The item stays non-finite with `converged=False`, and other items are unaffected |
| Non-convergence | Reported in `last_solve`, not raised |
| Obstacle change | Pressure setup is rebuilt when the obstacle differs, and applies from that step. Faces are masked before advection, and density in new solid cells is removed |
| State with `mac=None` | Faces are averaged from `velocity` and projected first |

From `results/solver_check.json`, on the 256×256 grid at batch 1: `steps_per_second` `19.84501788396093`, `mean_cg_iterations` `8.0`. In the 300-step long run, `max_divergence` is `0.00272369384765625` against a `plan_limit` of `0.001`, `steps_above_plan_limit` is `91`, and `all_converged` is `true`. `uv run python scripts/check_solver.py` regenerates the file and `results/figures/solver_check.png`.

### `physics/scenes.py`

```python
@dataclass
class Scene:
    obstacle: np.ndarray               # (res,res) bool
    inflow: np.ndarray                 # (res,res) float rate
    buoyancy: float
    late_obstacle: np.ndarray | None   # joins the solid at late_frame
    late_frame: int | None

def sample_scene(rng: np.random.Generator, res: int, split: str) -> Scene
```

| Split | Shape families | Buoyancy and inflow strength | Late obstacle |
|---|---|---|---|
| `train`, `val`, `test_a` | circle, rectangle, wall | 1 to 3, and 3 to 7 | In a third of scenes, at frame 20 to 90 |
| `test_b` | star, letter, blob | 1 to 3, and 3 to 7 | None |
| `test_c` | circle, rectangle, wall | Both outside, on the same side: 0.4 to 0.8 with 1.5 to 2.5, or 3.5 to 4.5 with 8 to 10 | None |

In every split, a scene has 1 or 2 sources with raised-cosine edges, and 1 to 3 obstacles. The first obstacle sits over a source, within 0.06 along x. No solid part is thinner than `MIN_THICKNESS`, 0.04. Solids keep 0.02 clear of sources and box walls. A scene where a source can't reach the top row through fluid, with the late obstacle in place, is redrawn. `scripts/check_scenes.py` measures the sampler.

## Data

```python
# data/store.py
def open_split(root: Path, split: str, final_eval: bool = False) -> SplitStore
def read_manifest(root: Path) -> dict

class SplitStore:
    fields: np.memmap   # float16, read-only, (scene, frame, 3, 128, 128): u, v, density
    n_scenes: int; n_frames: int; split: str
    def scene(self, i: int) -> Scene            # at the neural grid
    def fine_scene(self, i: int) -> Scene       # at the reference grid
    def obstacle(self, i: int, frame: int) -> np.ndarray   # bool (H,W), in force from frame to frame+1
    def inflow(self, i: int) -> np.ndarray
    def buoyancy(self, i: int) -> float
    def fine_state(self, i: int, frame: int) -> SmokeState  # KeyError unless the frame was stored

# data/dataset.py
def fit_normalizer(store) -> Normalizer
class StepDataset(Dataset):
    def __init__(self, store, normalizer: Normalizer, unroll: int = 1)
    # item: input (6,H,W), target_seq (unroll,3,H,W), obstacle (1,H,W), obstacle_seq (unroll,1,H,W),
    #       inflow (1,H,W), buoyancy (), scene, frame

# data/generate.py
def generate(config: dict, root: Path, *, allow_dirty: bool = False,
             on_scene: Callable[[str, int], None] | None = None, log: Callable[[str], None] = print) -> dict
```

Files under `data/`, none tracked:

| File | Format |
|---|---|
| `manifest.json` | Seeds, config, commit, SHA-256 of each file and split, realised late obstacles, `test_c` sides, solver health, rounding, timing. Written last, and with no manifest, no split opens. Must match the tracked copy `results/data_manifest.json` |
| `normalizer.json` | `field_mean`, `field_std`, `delta_std`, `inflow_scale`, `buoyancy_mean`, `buoyancy_std`, `fitted_on`, `n_scenes` |
| `generation.json` | The config and commit a run started with. Guards resuming |
| `timing.jsonl` | Seconds per batch, across restarts. Not checksummed |
| `<split>/fields.npy` | `float16`, `(scene, frame, 3, 128, 128)` |
| `<split>/statics.npz` | Per scene: `obstacle`, `late_obstacle`, `inflow`, their `fine_` forms, `buoyancy`, `late_frame` (-1 if none), `seed` |
| `<split>/fine.npz` | `frames`, and float32 `states` at the reference grid. Cell-centred, so a solver started from one projects first |
| `<split>/scenes/` | One file per finished scene, during generation only |

| Rule | Detail |
|---|---|
| Late obstacle | Frame `late_frame` is the last made under the old solid, and may hold smoke inside the new one |
| Coarse grid | Fields and inflow are 2×2 averages. A coarse cell is solid if any of its fine cells is |
| Lock | `open_split` checks the lock before it touches any file, then every checksum |
| Resuming | A resumed run does the same arithmetic and gives the same bytes. Same config and commit required |
| Stops | Non-finite stored data stops the run. A dirty tree stops it unless `--allow-dirty`, which the manifest records. Unconverged solves are counted, not fatal |
| Finished data | Never regenerated. Replacing a locked set means deleting `data/` by hand |
| Controls gate | `scripts/generate_data.py` calls `check_controls` first, and refuses without `results/travel_controls.json` for this physics, from committed code |

`scripts/spot_check.py --data-root data` checks each split and draws its scene 0 and first late-obstacle scene, writing `results/spot_check.json` and `results/figures/spot_check.png`. `scripts/check_travel.py` applies the travel rule in `data/travel.py`.

## Models

```python
# models/factory.py
def build_model(cfg: dict) -> nn.Module     # cfg["arch"] is unet or fno; the rest goes to the constructor
def apply_step(model: nn.Module, x: Tensor, normalizer) -> Tensor

# models/unet.py
class UNet(nn.Module):
    def __init__(self, base_width: int = 32, levels: int = 4, groups: int = 8, inputs: int = 6, outputs: int = 3)

# models/fno.py
class FNO2d(nn.Module):
    def __init__(self, width: int = 48, modes: int = 24, layers: int = 4, inputs: int = 6, outputs: int = 3)
```

| Model | Structure | `parameters` |
|---|---|---|
| U-Net, the winner | Widths 32, 64, 128, 256. Each level has a residual block of GroupNorm, SiLU and 3×3 convolution, twice. Stride-2 convolutions go down, and bilinear upsampling with skip connections goes up | `2263107` |
| FNO | Two coordinate channels added, because the box isn't periodic. A 1×1 lift, then 4 layers of a spectral product on the lowest 24 modes plus a 1×1 convolution, with SiLU, then a 1×1 projection. FFTs run in full precision | `21248499` |

## Training

```python
# train/losses.py
def divergence(velocity: Tensor, obstacle: Tensor) -> Tensor
def step_loss(pred: Tensor, target: Tensor, obstacle: Tensor, lambda_div: float,
              velocity_scale: tuple[float, float] = (1.0, 1.0)) -> dict   # loss, mse, div, div_target

# train/loop.py
def train(cfg: dict, run_dir: Path, resume: bool, *, on_step=None) -> Path   # the best checkpoint
def rollout_error(model, store, normalizer, start: int, steps: int, batch_size: int,
                  device: str = "cpu", amp: bool = False) -> float

# train/campaign.py
def select_architecture(configs: dict[str, dict], run_root: Path, results_path: Path) -> dict
def train_ensemble(cfg: dict, settings: dict, run_root: Path, results_path: Path) -> dict
def check_member(best_path: Path, cfg: dict, start: int, steps: int) -> dict
def train_ablations(cfg: dict, overlays: dict[str, dict], run_root: Path) -> dict[str, Path]
```

| Item | Detail |
|---|---|
| Loss | MSE of the normalised change on fluid cells, plus `lambda_div=0.1` × mean squared divergence of the predicted change |
| Divergence | Central differences in grid units, with `u` and `v` rescaled to physical proportion, over cells whose whole stencil is fluid. `div_target` is the same measure on the target |
| Pushforward | After the first 10% of steps, half the batches are first pushed 1 to 3 steps with no gradient. The dataset has `unroll=4`, so an item can't start in the last 4 frames of a scene |
| Optimiser | AdamW, lr `3e-4`, weight decay `0.01`, cosine schedule, mixed precision. A non-finite loss skips the update and is counted |
| Duration | `max_minutes`. The first `timing_steps` are ordinary steps that give the speed. A timed validation and timed forward passes then fix `total_steps` |
| Best checkpoint | Lowest validation rollout over 20 steps from frame 20: RMS error in normalised field units on fluid cells. A frame that blows up counts as `ERROR_CAP`, 10 |
| Selection | One seed-0 run per architecture, 40 minutes each. The lower error wins, and a tie goes to the U-Net |
| Ensemble | Seeds 0 to 4, 48 minutes each. A member must have no blown-up frame over 100 steps from frame 20, and beat persistence on those frames. Failing that, it is retrained once with 100 added to its seed |
| Ablations | One seed, 25 minutes, with `lambda_div: 0.0` or `pushforward_prob: 0.0` laid over the winner's config |

| File | Format |
|---|---|
| `runs/<run>/last.pt` | `model`, `optimizer`, `scaler`, `step`, `skipped_steps`, `best_error`, `total_steps`, `seconds_per_step`, `config`, `rng_cpu`, and `rng_cuda` if there is a GPU |
| `runs/<run>/best.pt` | `model`, `step`, `val_error`, `config`, `normalizer` |
| `runs/<run>/metrics.jsonl` | `step`, `lr`, `pushed`, `skipped_steps`, `loss`, `mse`, `div`, `div_target`, and `val_error` when measured |
| `runs/ensemble/seed_<n>/check.json` | The member's check, kept so a restart doesn't repeat it |
| `results/ensemble.json` (written when run) | `arch`, `seeds`, `members` with checkpoint path and SHA-256, `complete`, provenance |

Scripts: `scripts/train.py` for one model, `select_arch.py`, and `train_ensemble.py`, which takes `--ablations`. The last two have `--dry-run` on tiny data.

From `results/throughput.json`, at batch 16, the size in the configs:

| Model | `steps_per_second` | `seconds_per_forward` | `peak_reserved_gb` |
|---|---|---|---|
| U-Net | `14.86350045939139` | `0.023438122600782662` | `1.36328125` |
| FNO | `15.618378366425217` | `0.018669968799804336` | `1.17578125` |

These are training figures. They say nothing about the demo's frame rate.

**The divergence floor.** The penalty uses central differences on cell-centred velocity. That has a large nonzero value even on true data, so it can't reach zero. Its floor on the training targets is `div_target_mean_squared` in `results/throughput.json`, `0.0025816269568167626` at batch 16, and `metrics.jsonl` reports `div` beside `div_target`. From `long_run.view_divergence` in `results/solver_check.json`, in physical units:

| Field | `max` | `rms` |
|---|---|---|
| `plume_256_faces` | `0.0008544921875` | `1.6283853256027214e-05` |
| `plume_256_cell_centred` | `48.56700134277344` | `0.7967968583106995` |
| `plume_128_downsampled_cell_centred` | `65.86290740966797` | `1.257594347000122` |
| `random_64_faces` | `0.00012302398681640625` | `2.8387958082021214e-05` |
| `random_64_cell_centred` | `36.53493118286133` | `8.911713600158691` |

## Ensemble and doubt

```python
# ensemble.py
@dataclass
class EnsembleOut:
    mean: Tensor        # (B,3,H,W), physical units
    doubt_map: Tensor   # (B,1,H,W), in field standard deviations
    members: Tensor     # (K,B,3,H,W), physical units

class Ensemble:
    def __init__(self, models: list[nn.Module], normalizer)
    def step(self, x: Tensor) -> EnsembleOut

def doubt_score(doubt_map: Tensor, obstacle: Tensor, method: Literal["mean", "p95", "p99"]) -> Tensor  # (B,)
```

- **Doubt map.** Encode each member's next frame with the field normaliser. Take the population standard deviation across members per channel, then average the 3 channels. Equal members, or a single member, give exactly zero.
- **Doubt score.** The chosen aggregation over fluid cells, using the mask on the way into the frame. One non-finite fluid cell makes the score infinite under every method. A frame with no fluid cells scores zero.

## Hand-off logic

```python
# hybrid.py
class Policy:
    rescues = True
    def decide(self, frame: int, doubt: float, ctx: dict) -> bool

class HybridSimulator:
    def __init__(self, ensemble, solver, policy: Policy, handoff_frames: int, substeps: int = 4, *,
                 doubt_method: str = "mean", amp: bool = False,
                 error_of: Callable[[int, Tensor, Tensor], float] | None = None,
                 on_solver_frame: Callable[[int, Tensor, Tensor, float], None] | None = None)
    def reset(self, state: Tensor, obstacle: Tensor, inflow: Tensor, buoyancy: float, *, frame: int = 0,
              late_obstacle: Tensor | None = None, late_frame: int | None = None) -> None
    def set_obstacle(self, obstacle: Tensor) -> None
    def step(self) -> StepInfo
```

`StepInfo` holds `state`, `doubt_map`, `doubt`, `source` (`"ai"` or `"solver"`), `ms`, `frames_advanced`, `frame`, `frames`, `forced` and `solver_failed`. `ms` covers the whole step, with every member and any solver steps, between `torch.cuda.synchronize()` calls.

| Policy | Hands off when | Rescues a non-finite output |
|---|---|---|
| `AlwaysAI` | Never | No |
| `AlwaysSolver` | Always | Yes |
| `FixedSchedule(period, offset=0, start=21)` | Every `period` frames from `start + offset` | Yes, counted as forced |
| `RandomBudget(rate, seed, scene=0)` | With chance `rate`, from a stream fixed per scene and seed | Yes, counted as forced |
| `DoubtThreshold(tau)` | Doubt above `tau`, or non-finite | Yes, except `tau=inf`, the **never** candidate |
| `Oracle(tau)` | True error of this frame above `tau`, read through `ctx["error"]` | Yes, counted as forced |

One `step`:

1. Fix the solid for this step. An edit arriving later waits for the next step.
2. Run the ensemble from the current frame. This happens under every policy, including `AlwaysSolver`.
3. Score the doubt map with `doubt_method`, and ask the policy. If the output isn't finite and the policy said no, a policy with `rescues` hands off anyway, with `forced=True`.
4. No hand-off: the next frame is the ensemble mean, with `source="ai"`.
5. Hand-off: the mean is thrown away. The frame before it is upsampled bilinearly to 256×256 and projected with the fine solid. For each of `handoff_frames` frames, the solver takes `substeps` steps, and the result is averaged 2×2 and zeroed in the coarse solid, as the data is. A late obstacle joins at its frame inside a hand-off. The ensemble resumes from the last frame.
6. Solver failure: a non-finite solver frame ends the hand-off. That frame is set to NaN, `solver_failed` stays set for the rollout, and no further hand-off is tried, so later frames count as blown up.

### Selection

```python
# handoff_selection.py
def select_handoff(ensemble: Ensemble, solver, store, *, progress_path: Path, window: Window = Window(),
                   percentiles=PERCENTILES, handoff_frames=HANDOFF_FRAMES, substeps: int = 4, amp: bool = False,
                   n_resamples: int = BOOTSTRAP_RESAMPLES, provenance: dict | None = None, log=print) -> dict
def frame_error(pred: np.ndarray, ref: np.ndarray, solid: np.ndarray, floors) -> float
```

`select_handoff` refuses any split but `val`. It runs in this order:

1. Evidence rollouts with no hand-off, frames 21 to 70, giving each aggregation's evidence measure.
2. `aggregations_left_in`: the leader, and any other whose paired bootstrap interval from it includes zero.
3. `candidate_thresholds`: the 50th, 60th, 70th, 80th, 90th, 95th, 98th and 99th percentiles of doubt.
4. A hand-off rollout set for each aggregation left in, each `handoff_frames` in `{2, 5, 10}`, and each threshold.
5. `choose_run`: among runs with a solver share at or below 25%, the lowest score after rounding to 6 decimals. **never** is always eligible and goes first in a tie.

`results/handoff_selection.json` (written when run) has `runs`, `candidates`, `aggregation`, `bad_frame_threshold`, `calibration_edges`, provenance and `choice`, which holds `kind`, `gain_on_validation`, `aggregation`, `handoff_frames`, `tau`, `set_score`, `solver_share`, `forced_handoffs` and `never_set_score`. A `tau` of null means never. Finished rollout sets are kept in `runs/handoff_selection/progress.json`.

## Evaluation

```python
# eval/rollout.py
def rollout(sim, scene: int, ref, start: int, horizon: int, *, floors, scores_by=(), device=None) -> RolloutTrace
class PersistenceSim(_Baseline):  ...
class CoarseSolverSim(_Baseline): ...
class IndependentSim(_Baseline):  ...
def timed_reference(solver, store, scenes, start: int, frames: int, substeps: int) -> float

# eval/metrics.py
def nrmse(pred, ref, obstacle, floors=_SMALLEST_FLOOR) -> float
def bootstrap_ci(values_by_scene, n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]
def rise_on_unfamiliar(other, familiar, n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict
# spearman, auroc, calibration, risk_coverage and evidence_measure are shared from well/doubt.py

# eval/experiments.py
def open_eval_sets(root: Path, final: bool) -> tuple[dict, object]
def run_experiments(ev: Evaluation, stores: dict, reference, names, out: Path, final: bool) -> dict[str, Path]
```

**Per-frame error.** For each of the 3 channels, the RMS difference from the reference over fluid cells, divided by the reference's standard deviation there, with a floor of 1% of the channel's training standard deviation. The error is the mean of the 3. A non-finite value or an error above 10 counts as 10, and later frames stay in the average.

Settings are in `configs/eval.yaml`: rollouts start at frame 20, the primary window is frames 21 to 70, and the secondary one runs to 120.

| Result file (written when run) | Contents |
|---|---|
| `exp_accuracy.json` | Ensemble, single model, persistence, coarse solver and solver only, with per-frame error, secondary metrics and the reference's time per frame |
| `exp_doubt.json` | Evidence measure, AUROC, calibration and risk-coverage, the no-ensemble signal, independent rollouts, and rise on unfamiliar sets |
| `exp_handoff.json` | Frozen threshold against fixed and random baselines at the same budget, the oracle, and descriptive sweeps |
| `exp_ablation.json` | Reduced trainings against the full member, and ensemble sizes 2, 3 and 5 |

Each file has `experiment`, `final`, `sets` and provenance, with `git_commit`, `data_manifest_sha256`, `checkpoint_sha256` and `config` required. `test_c` reports the low side, the high side and pooled. A plain run opens only `val` and writes to `runs/experiments/val_results`. `--final` opens the test splits and writes to `results/`. Rollout sets are kept in `runs/experiments`, and rerun if `code_sha256` of the code that makes them has changed.

`scripts/make_figures.py` draws `results/figures/` from these files. `scripts/render_results.py` replaces the generated block in `docs/RESULTS.md`, and refuses a file without provenance.

## Public-dataset track

The Well's `turbulent_radiative_layer_2D`, a published simulation benchmark and not measured data. It measures doubt only, with no hand-off.

```python
# well/data.py
class WellSplit:
    def __init__(self, root: Path, split: str, final_eval: bool = False)
    def read(self, trajectory: int, start: int, stop: int) -> np.ndarray
class WellStepDataset(Dataset):
    def __init__(self, root: Path, split: Literal["train", "valid", "test"], final_eval: bool = False, unroll: int = 1)
def check_counts(split: WellSplit, expected: dict) -> None

# well/train.py
def train_members(cfg: dict, run_root: Path) -> list[Path]

# well/evaluate.py
def select_on_valid(cfg: dict, paths: list[Path], selection_path: Path) -> dict
def evaluate_final(cfg: dict, paths: list[Path], selection_path: Path, result_path: Path) -> dict
```

| Item | Detail |
|---|---|
| Channels | Input and output have 4: density, pressure, `velocity_x`, `velocity_y`. There's no mask, inflow or buoyancy channel |
| Grid and counts | Compared with `expected` in `configs/well.yaml` by `check_counts`, not assumed |
| Model | U-Net with `inputs: 4`, `outputs: 4`, 3 seeds, in `runs/well/member_<seed>/`. MSE of the normalised change, with pushforward and no divergence penalty, because this flow is compressible |
| Choices | The aggregation, bad-frame threshold and calibration edges are fixed on `valid`, in `results/well_selection.json` |
| Final | `scripts/eval_well.py --final` opens `test`, refuses other checkpoints, and writes `results/exp_well.json` |

Files in `data/well/data/<split>/*.hdf5` hold `t0_fields/density`, `t0_fields/pressure` and `t1_fields/velocity`.

## Demo

```python
# demo/server.py
class DemoPolicy(Policy):
    def __init__(self, mode: str, tau: float)

class DemoSimulation:
    def __init__(self, ensemble: Ensemble, solver: SmokeSolver, *, handoff_frames: int, substeps: int, coarse_res: int,
                 tau: float, doubt_method: str = "mean", amp: bool = False, mode: str = "auto",
                 weights: str = "trained", preset: str = "circle", doubt_full_scale: float = 1.0)
    def submit(self, message) -> None   # checks and queues; ValueError if it can't be applied
    def start(self) -> None
    def stop(self) -> None
    def advance(self) -> None           # simulation thread only
    def preflight(self) -> None
    latest: tuple[int, bytes]

def preset_scene(name: str, res: int) -> Scene
def create_app(sim: DemoSimulation, run: bool = True) -> FastAPI
def build_simulation(args: argparse.Namespace, *, need_selection: bool = True) -> DemoSimulation
def main(argv: list[str] | None = None) -> int
```

**Start-up.** `uv run python scripts/run_demo.py` takes `--random-weights`, `--members`, `--device`, `--data-root`, `--ensemble`, `--selection`, `--host`, `--port` and `--preset`. It serves `http://127.0.0.1:8000` by default. It reads physics from the manifest, or `configs/data.yaml` without one. It loads `results/ensemble.json`, checks each checkpoint's SHA-256, and takes `tau`, aggregation and hand-off length from `results/handoff_selection.json`. The member count is `--members`, or `chosen_members` in `results/demo_bench.json`, or all of them. `preflight` runs one hand-off so that memory runs out at start-up if it is going to.

A missing GPU, missing files or exhausted memory print a message and exit with status 1.

**Protocol.** WebSocket at `/ws`. Lengths are in domain units, with row 0 at the bottom.

| Direction | Content |
|---|---|
| Server to client | Binary: `uint32` little-endian header length, a JSON header, then `res × res` bytes each of density, doubt and obstacle |
| Header | `frame, source, doubt, tau, fps, solver_share, mode, members, weights, res, doubt_scale, preset`. `doubt` and `tau` are null when infinite or NaN |
| Client to server | JSON: `{"type": "brush", "x", "y", "r", "erase"}`, `{"type": "preset", "name"}`, `{"type": "reset"}`, `{"type": "threshold", "tau"}` with null for never, `{"type": "mode", "mode"}` with `auto`, `ai` or `solver` |

| Layer | Byte value |
|---|---|
| Density | `density / (mean + 3 × std)`, clamped to 0 to 1, × 255 |
| Doubt | `doubt / doubt_scale`, clamped, × 255. `doubt_scale` is twice `tau`, or 1 if `tau` is never or zero. A non-finite cell shows full scale |
| Obstacle | 0 or 255, on the coarse grid |

| Behaviour | Detail |
|---|---|
| Threads | One simulation thread calls `advance`. Client sockets poll `latest` every 5 ms and send the newest frame, so a slow client loses older ones |
| Edits | Queued, and applied before the next step. An edit during a hand-off applies after it. The brush paints on the fine grid and updates both masks |
| Bad messages | Logged and ignored. The connection stays open |
| Modes | `DemoPolicy` uses `DoubtThreshold(tau)` in `auto`, never hands off in `ai`, and always hands off in `solver`. It rescues in every mode, so a non-finite frame in `ai` mode shows as a solver frame |
| Solver frames | Published one by one as they are made, with the doubt from the ensemble step into that hand-off |
| Recovery | On a non-finite state or a failed solver, `_recover` returns to the last finite frame and hands off on the next step. If that also fails, it starts from zero fields, keeping the solids |
| `fps`, `solver_share` | `fps` is frames published over the last 30. `solver_share` counts since the last preset or reset |
| Presets | `empty`, `circle`, `rectangle`, `wall`, `star`, `letter`, `blob`. One source at (0.5, 0.12), radius 0.065, strength 5, buoyancy 2, with the shape centred at (0.5, 0.55) |
| `--random-weights` | Untrained U-Nets, `tau=0.0225` so that both sources show, and `weights: "random"` in the header, which makes the page show a warning. The value isn't a result |

**Page.** `demo/static/index.html`, `app.js` and `style.css`, with local fonts. The page holds no simulation state. It draws the newest frame only, flipping rows and showing solids in grey. Pointer, touch and keyboard all draw. The brush radii are 0.02, 0.045 and 0.09. The threshold slider runs from 0 to 100, where 100 sends null and any other value `v` sends `v / 99 × doubt_scale`. The page reconnects every second if the socket drops.

**Benchmark.** `scripts/bench_demo.py` runs the demo's own simulation in AI mode for 60 s per rung, trying 5, then 4, 3 and 2 members until one reaches 15 frames per second. It writes `results/demo_bench.json` (written when run), with `target_fps`, `target_met`, `chosen_members`, `rungs`, `weights`, `device` and the commit. `--random-weights` needs its own `--out`.

## Long-job wrapper

`scripts/long_job.sh <name> <command...>`

| Behaviour | Detail |
|---|---|
| Exit 2 | Fewer than two arguments, a bad name, or the reserved name `jobs` |
| Exit 3 | No power supply of type `Mains` reports `online=1`. An unknown state also refuses |
| Exit 4 | The job reported no exit status. Logged as `exit=unknown` |
| Exit code otherwise | The job's own, or 128+n when signal n killed it |
| Inhibitor | `systemd-inhibit --what=sleep:idle:handle-lid-switch` |
| Logs | Output to `runs/<name>.log` and the terminal. `START` and `END` lines to `runs/jobs.log` |
| Signals to the wrapper | Sent to the job's whole process group. The wrapper waits for the job, then writes `END` |
| `START` with no `END` | The wrapper was killed or the machine died, and the job may still be running. Check `systemd-inhibit --list` before restarting |
| Test override | `SANDEHA_POWER_SUPPLY_DIR` replaces `/sys/class/power_supply` |
