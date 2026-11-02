# Sandeha: Low-Level Design

**Status:** interfaces as planned in `docs/PLAN.md`. Built so far: `scripts/long_job.sh`, the package skeleton and the reference solver. Each section is updated to match the code when its component is built.

## Conventions

- Grids: reference 256×256, neural 128×128. One neural step is 4 reference solver steps.
- Field tensors are `(B,C,H,W)`. `obstacle` and `inflow` are `(B,1,H,W)`, and `buoyancy` is `(B,)`.
- Neural input channels, in order: `u, v, density, obstacle, inflow, buoyancy`.
- Neural output channels: the normalised change in `u, v, density`. Solid cells are forced to zero after prediction.

## Long-job wrapper (built)

`scripts/long_job.sh <name> <command...>`

| Behaviour | Detail |
|---|---|
| Exit 2 | Fewer than two arguments, a name with characters outside letters, digits, `.`, `_`, `-`, or the reserved name `jobs`, since `runs/jobs.log` is the shared log |
| Exit 3 | No power supply of type `Mains` reports `online=1`. An unknown state also refuses, so that a job never starts on a guess |
| Exit 4 | The job reported no exit status, for example because the inhibitor could not start it. Logged as `exit=unknown` |
| Inhibitor | `systemd-inhibit --what=sleep:idle:handle-lid-switch`, with the job as its descendant |
| Working directory | Project root, wherever the wrapper is called from |
| Logs | Combined output appended to `runs/<name>.log` and shown on the terminal. `START` and `END` lines with exit code and elapsed seconds appended to `runs/jobs.log` |
| Exit code | The job's own, or 128+n when signal n killed it. A helper under the inhibitor records this, because `systemd-inhibit` reports any job killed by a signal as exit 1 |
| Standard input | `/dev/null`: the job runs in a background process group |
| SIGTERM, SIGINT, SIGHUP to the wrapper | Sent to the job's whole process group, including its children. The inhibitor stays held and the wrapper waits for the job to stop, then writes `END`. A second signal is forwarded the same way. A child started with `&` from a non-interactive shell ignores SIGINT, and the wrapper keeps waiting for it; SIGTERM stops it |
| `START` with no `END` | The wrapper got SIGKILL or the machine died. **The job may still be running.** Before restarting, check that `systemd-inhibit --list` shows no `sandeha` entry for the job, so that two jobs never share checkpoints or the GPU |
| Test override | `SANDEHA_POWER_SUPPLY_DIR` replaces `/sys/class/power_supply` |

## Physics

### Reference solver (built)

```python
@dataclass
class SmokeState:
    velocity: Tensor   # (B,2,H,W), cell-centred view of the faces in mac
    density: Tensor    # (B,1,H,W)
    mac: tuple[Tensor, Tensor] | None = None   # u (B,1,H,W+1), v (B,1,H+1,W)

@dataclass
class PressureReport:
    iterations: Tensor   # (B,)
    residual: Tensor     # (B,) largest absolute divergence in a cell, physical units
    converged: Tensor    # (B,) bool

class SmokeSolver:
    def __init__(self, res: int, device, dt: float, cg_tol: float = 1e-5, cg_max_iter: int = 2000): ...
    def step(self, state, obstacle, inflow, buoyancy) -> SmokeState: ...
    def project(self, state, obstacle) -> SmokeState: ...
    def divergence(self, state, obstacle) -> Tensor: ...     # (B,1,H,W), physical units, zero in solids
    def preconditioner(self, obstacle) -> Callable: ...      # exposed for checking
    last_solve: PressureReport | None                        # for the most recent pressure solve
```

| Item | Detail |
|---|---|
| Units | Unit-square domain, cell width `1/res`, velocity in domain lengths per unit time |
| Direction | Index `(row, column)` is `(y, x)`. Rows grow upward, buoyancy acts along +row, plots need `origin="lower"` |
| Walls | Closed box. Faces on walls or touching a solid cell are zero |
| Step order | Mask faces with the current obstacle; advect density, `u` and `v`; add `dt × inflow` in fluid cells; apply decay; add `dt × buoyancy × density` to `v`; project |
| Advection | Semi-Lagrangian with a midpoint backtrace and bilinear lookup. Density cannot go negative |
| Decay | `DENSITY_DECAY = 0.1` per unit time |
| Pressure | Conjugate gradients with a step size per batch item, and one symmetric multigrid V-cycle as preconditioner. Each item stops on its own |
| Convergence | No cell's net flux above `cg_tol` × the item's fastest face, measured from the faces. Repeated correction passes reach this in float32. In physical units the limit is `cg_tol × fastest face × res`, so it grows with grid size and speed: a converged 256×256 solve does not guarantee divergence below `1e-3` (`BUILD_LOG.md`) |
| Stopping otherwise | A pass that fails to halve the divergence, `cg_max_iter`, or a non-finite input, which uses no iterations |
| Non-finite input | Applies to both `project` and `step`. The item stays non-finite with `converged=False`, and other items are unaffected. A NaN coordinate in advection gives NaN at a valid index, so a bad item neither crashes the device nor heals (`DECISIONS.md` B16) |
| Non-convergence | Reported in `last_solve`, not raised. Callers that need a guarantee must check it |
| Obstacle change | Pressure setup is compared against the last obstacle, and rebuilt when it differs. It applies from the first step: faces are masked before advection and density in new solid cells is removed (`DECISIONS.md` B15) |
| States from elsewhere | With `mac=None`, faces are averaged from `velocity` and projected first. To change `velocity`, build a new state without `mac` |
| `divergence` with `mac=None` | Faces averaged from `velocity`, which in the interior equals central differences. Averaging does not preserve the projection, so this stays large on a projected field and says nothing about the quality of a solve. See the warning under "Models and training" |

Measured speed and convergence are in `results/solver_check.json`, regenerated by `uv run python scripts/check_solver.py`. Inflows should have smooth edges: a hard-edged source is a different problem on each grid (`DECISIONS.md` B12).

### Scenes

```python
@dataclass
class Scene:
    obstacle: np.ndarray               # (res,res) bool at reference grid
    inflow: np.ndarray                 # (res,res) float rate
    buoyancy: float
    late_obstacle: np.ndarray | None   # obstacle added mid-run
    late_frame: int | None

def sample_scene(rng: np.random.Generator, res: int, split: str) -> Scene: ...
```

`split` is one of `train val test_a test_b test_c`.

## Data

```python
def open_split(root: Path, split: str, final_eval: bool = False) -> SplitStore: ...

class Normalizer:   # fitted on train only, saved in data/normalizer.json
    def encode(self, x: Tensor) -> Tensor: ...
    def decode(self, x: Tensor) -> Tensor: ...

class StepDataset(Dataset):
    def __init__(self, store, normalizer, unroll: int = 1): ...
    # item: input (6,128,128), target_seq (unroll,3,128,128), obstacle, inflow, buoyancy
```

`SplitStore.fields` is a `float16` memmap of `(scene, frame, 3, 128, 128)`. Writes are atomic per scene. `data/manifest.json` records seeds, config, git commit and the SHA-256 of each split. `open_split` raises `PermissionError` on a test split unless `final_eval=True`, and fails on a checksum mismatch.

## Models and training

```python
def build_model(cfg: dict) -> nn.Module                 # (B,6,H,W) -> (B,3,H,W), the normalised delta
def apply_step(model, x: Tensor, normalizer) -> Tensor  # next (B,3,H,W), solid cells forced to zero
def divergence(velocity: Tensor, obstacle: Tensor) -> Tensor
def step_loss(pred, target, obstacle, lambda_div: float) -> dict
def train(cfg: dict, run_dir: Path, resume: bool) -> Path   # returns best checkpoint path
```

| Item | Detail |
|---|---|
| U-Net | Base width 32, 4 levels, GroupNorm, SiLU |
| FNO | Width 48, 24 modes, 4 layers, coordinate channels because the box is not periodic |
| Loss | MSE on normalised delta, plus `lambda_div=0.1` × mean squared divergence |
| Stability | After warm-up, sometimes unroll 1 to 3 steps without gradient; only the last step gets gradients |
| Optimiser | AdamW, lr `3e-4`, cosine schedule, mixed precision |
| Duration | `max_minutes` sets total steps from measured throughput |
| Best checkpoint | Lowest validation rollout error over 20 steps |

**Warning for the physics penalty.** The planned `divergence(velocity, obstacle)` uses central differences on cell-centred velocity, the same in the interior as `SmokeSolver.divergence` with `mac=None`. That has a large nonzero floor on true data, so the penalty cannot reach zero, and pushing it towards zero works against the targets. From `long_run.view_divergence` in `results/solver_check.json`:

| Field | Largest | RMS |
|---|---|---|
| 256×256 plume, on the faces | 0.00085449 | 1.63e-05 |
| 256×256 plume, cell-centred | 48.567 | 0.797 |
| Same state averaged to 128×128, cell-centred | 65.863 | 1.26 |
| Random field at 64×64, projected, on the faces | 0.00012302 | 2.84e-05 |
| Same field, cell-centred | 36.535 | 8.91 |

Task 5 should measure this floor on its training targets before fixing `lambda_div`, and report the penalty relative to it. These figures are in physical units, before normalisation, and the largest values were not separated by distance from the obstacle.

## Ensemble and doubt

```python
class Ensemble:
    def __init__(self, models: list[nn.Module], normalizer): ...
    def step(self, x: Tensor) -> EnsembleOut   # .mean (B,3,H,W), .doubt_map (B,1,H,W), .members

def doubt_score(doubt_map, obstacle, method: Literal["mean","p95","p99"]) -> Tensor  # (B,), fluid cells only
```

`doubt_map` is the square root of per-channel member variance in normalised units, averaged across channels.

## Hybrid simulator

```python
class Policy:
    def decide(self, frame: int, doubt: float, ctx: dict) -> bool: ...   # True = hand off now
# AlwaysAI, AlwaysSolver, FixedSchedule(period), RandomBudget(rate, seed), DoubtThreshold(tau), Oracle(tau)

class HybridSimulator:
    def __init__(self, ensemble, solver, policy, handoff_frames: int, substeps: int = 4): ...
    def reset(self, state, obstacle, inflow, buoyancy): ...
    def set_obstacle(self, obstacle): ...
    def step(self) -> StepInfo   # .state, .doubt_map, .doubt, .source, .ms, .frames_advanced
```

Hand-off upsamples the state to 256×256, projects it, runs `4 × handoff_frames` solver steps, emits each frame, and downsamples. Timing calls `torch.cuda.synchronize()`.

Chosen on validation only, and recorded in `results/handoff_selection.json`: `handoff_frames` from `{2, 5, 10}`, doubt aggregation from `{mean, p95, p99}`, and the frozen threshold `tau`, which gives the lowest primary error with solver share at or below 25%.

## Evaluation

```python
def nrmse(pred, ref, obstacle) -> float            # normalised by reference std, fluid cells
def bootstrap_ci(values_by_scene, n=2000) -> tuple[float, float]   # resampling scenes
def spearman(x, y) -> float
def auroc(score, label) -> float
def rollout(sim, scene, ref, start: int, horizon: int) -> RolloutTrace
```

A blown-up rollout gets a capped maximum error and stays in the average. Result files are `results/exp_accuracy.json`, `exp_doubt.json`, `exp_handoff.json` and `exp_ablation.json`, each carrying git commit, data manifest hash, checkpoint hashes and config.

## Public-dataset track

```python
class WellStepDataset(Dataset):   # h5py reader on official splits; input and next frame, normalised from train
    def __init__(self, root: Path, split: Literal["train","valid","test"], final_eval: bool = False): ...
```

The loader confirms grid, step and trajectory counts rather than assuming them.

## Demo protocol

WebSocket at `/ws`.

| Direction | Content |
|---|---|
| Server to client | Binary frame: JSON header (`frame, source, doubt, tau, fps, solver_share, mode`), then `uint8` density, doubt and obstacle at 128×128 |
| Client to server | JSON: `brush{x,y,r,erase}`, `preset{name}`, `reset`, `threshold{tau}`, `mode{auto,ai,solver}` |

The simulation runs in one thread. Each client gets the latest frame, and older frames are dropped.
