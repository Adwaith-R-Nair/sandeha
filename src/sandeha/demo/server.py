"""The demo's server: one simulation thread, and a WebSocket that streams its latest frame.

    uv run python scripts/run_demo.py                    # Needs the trained ensemble and a GPU.
    uv run python scripts/run_demo.py --random-weights   # Untrained models, before training ends.

Protocol at ``/ws``. Lengths are in domain units, the unit square, with row 0 at the bottom.

- Server to client, binary: a little-endian ``uint32`` header length, a JSON header, then ``res * res``
  bytes each of density, doubt and obstacle (0 or 255). The header holds ``frame, source, doubt, tau,
  fps, solver_share, mode, members, weights, res``. ``doubt`` and ``tau`` are null when infinite or
  not a number. ``weights`` is ``"random"`` for untrained models, which the page must disclose.
- Client to server, JSON: ``{"type": "brush", "x", "y", "r", "erase"}``, ``{"type": "preset", "name"}``,
  ``{"type": "reset"}``, ``{"type": "threshold", "tau"}`` with null for never, and
  ``{"type": "mode", "mode"}`` with ``auto``, ``ai`` or ``solver``.

Every edit waits in a queue until the step under way has ended, so one arriving during a hand-off
applies after it. Each client gets the newest frame when it is ready for one, and older ones are lost.
"""

import argparse
import asyncio
import collections
import json
import logging
import math
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from torch import Tensor

from sandeha.data.dataset import Normalizer
from sandeha.data.store import read_manifest, sha256_file
from sandeha.ensemble import Ensemble
from sandeha.hybrid import DoubtThreshold, HybridSimulator, Policy
from sandeha.models.factory import build_model
from sandeha.physics.scenes import _BUILDERS, _grid, TRAIN_SHAPES, UNSEEN_SHAPES, Scene
from sandeha.physics.solver import SmokeSolver
from sandeha.train.loop import load_checkpoint

log = logging.getLogger("sandeha.demo")

STATIC = Path(__file__).parent / "static"
MODES = ("auto", "ai", "solver")
PRESETS = ("empty", *TRAIN_SHAPES, *UNSEEN_SHAPES)
BENCH_PATH = Path("results/demo_bench.json")
TRAIN_CONFIG = Path("configs/train_unet.yaml")
DATA_CONFIG = Path("configs/data.yaml")

OOM_MESSAGE = "The demo ran out of GPU memory. Close whatever else is using the GPU, or try fewer --members."
NO_GPU_MESSAGE = "No GPU found. The demo needs CUDA for its frame rate. To run it slowly anyway, pass --device cpu."

_PRESET_SOURCE = (0.5, 0.12, 0.065)  # Centre and radius.
_PRESET_STRENGTH = 5.0  # The middle of the training ranges.
_PRESET_BUOYANCY = 2.0
_PRESET_CENTRE = (0.5, 0.55)  # In the plume's path.
_MAX_BRUSH_RADIUS = 2.0  # Past the corners of the box, from anywhere in it.
_FPS_WINDOW = 30  # Frames.
_SEND_POLL_SECONDS = 0.005
_FALLBACK_HANDOFF_FRAMES = 5
_DEFAULT_DOUBT_FULL_SCALE = 1.0  # Doubt is in field standard deviations.
_RANDOM_TAU = 0.0225  # Near where untrained models settle, so that both sources show. Not a result.
_RANDOM_MEMBERS = 5
# data/normalizer.json's figures, for untrained models on a machine without the dataset.
_PLACEHOLDER_NORMALIZER = {
    "field_mean": [0.0, 0.0, 0.062],
    "field_std": [0.154, 0.179, 0.167],
    "delta_std": [0.0217, 0.0235, 0.0427],
    "inflow_scale": 6.93,
    "buoyancy_mean": 1.99,
    "buoyancy_std": 0.595,
    "fitted_on": "placeholder",
    "n_scenes": 0,
}


class StartupError(Exception):
    """The demo can't start, for a reason its message explains."""


def preset_scene(name: str, res: int) -> Scene:
    """A fixed scene: one source at the bottom, and the named shape in its plume."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}, expected one of {PRESETS}")
    x, y = _grid(res)
    source_x, source_y, radius = _PRESET_SOURCE
    distance = np.hypot(x - source_x, y - source_y)
    bump = np.where(distance < radius, np.cos(0.5 * np.pi * distance / radius) ** 2, 0.0)
    inflow = (bump / bump.max() * _PRESET_STRENGTH).astype(np.float32)
    obstacle = np.zeros((res, res), dtype=bool)
    if name != "empty":
        rng = np.random.default_rng(PRESETS.index(name))
        obstacle = _BUILDERS[name](rng, x - _PRESET_CENTRE[0], y - _PRESET_CENTRE[1])
    return Scene(obstacle, inflow, _PRESET_BUOYANCY, None, None)


class DemoPolicy(Policy):
    """The mode and threshold in force, either of which the user can change between steps.

    It rescues in every mode, unlike ``AlwaysAI``: forcing AI on the page doesn't mean a NaN stays on
    screen. Such a frame shows as a solver frame, which is what it is.
    """

    rescues = True

    def __init__(self, mode: str, tau: float):
        self.mode = mode
        self.tau = tau
        self.force_next = False

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        if self.force_next:
            self.force_next = False
            return True
        if self.mode == "ai":
            return False
        if self.mode == "solver":
            return True
        return DoubtThreshold(self.tau).decide(frame, doubt, ctx)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _checked(message) -> dict:
    """The message if it is one the simulation can apply. It raises ``ValueError`` if not."""
    if not isinstance(message, dict):
        raise ValueError(f"a message must be an object, got {message!r}")
    kind = message.get("type")
    if kind == "brush":
        for name in ("x", "y"):
            _number(message.get(name), name)
        if not 0 < _number(message.get("r"), "r") <= _MAX_BRUSH_RADIUS:
            raise ValueError(f"r must be above 0 and at most {_MAX_BRUSH_RADIUS}, got {message['r']}")
    elif kind == "preset":
        if message.get("name") not in PRESETS:
            raise ValueError(f"unknown preset {message.get('name')!r}, expected one of {PRESETS}")
    elif kind == "threshold":
        if "tau" not in message:
            raise ValueError("a threshold message must have tau, with null for never")
        if message["tau"] is not None and _number(message["tau"], "tau") < 0:
            raise ValueError(f"tau can't be negative, got {message['tau']}")
    elif kind == "mode":
        if message.get("mode") not in MODES:
            raise ValueError(f"unknown mode {message.get('mode')!r}, expected one of {MODES}")
    elif kind != "reset":
        raise ValueError(f"unknown message type {kind!r}")
    return message


def _bytes_of(field: Tensor) -> bytes:
    """``(H,W)`` in the range 0 to 1 as ``uint8``, row 0 first."""
    return (field.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy().tobytes()


class DemoSimulation:
    """A hybrid simulator with the user's edits, a blow-up's recovery, and the latest encoded frame.

    Only one thread may call ``advance``. ``submit`` and ``latest`` can be used from any thread.
    """

    def __init__(
        self,
        ensemble: Ensemble,
        solver: SmokeSolver,
        *,
        handoff_frames: int,
        substeps: int,
        coarse_res: int,
        tau: float,
        doubt_method: str = "mean",
        amp: bool = False,
        mode: str = "auto",
        weights: str = "trained",
        preset: str = "circle",
        doubt_full_scale: float = _DEFAULT_DOUBT_FULL_SCALE,
    ):
        self.ensemble = ensemble
        self.solver = solver
        self.coarse_res = coarse_res
        self.weights = weights
        self.doubt_full_scale = doubt_full_scale
        normalizer = ensemble.normalizer
        self.density_full_scale = normalizer.field_mean[2] + 3.0 * normalizer.field_std[2]
        self.policy = DemoPolicy(mode, tau)
        self.hybrid = HybridSimulator(
            ensemble,
            solver,
            self.policy,
            handoff_frames,
            substeps,
            doubt_method=doubt_method,
            amp=amp,
            on_solver_frame=self._on_solver_frame,
        )
        self.latest: tuple[int, bytes] = (0, b"")
        self.fatal: str | None = None
        self.on_fatal = None
        self.checkpoint_sha256: list[str] = []
        self._messages: collections.deque = collections.deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._times: collections.deque = collections.deque(maxlen=_FPS_WINDOW)
        x, y = _grid(solver.res)
        self._x = torch.from_numpy(x).float().to(solver.device)
        self._y = torch.from_numpy(y).float().to(solver.device)
        self._load(preset)

    # What other threads use

    def submit(self, message) -> None:
        """Queue a client's message for the next step. A message it can't apply raises ``ValueError``."""
        self._messages.append(_checked(message))

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="simulation", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    # The simulation thread

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.advance()
        except torch.cuda.OutOfMemoryError:
            self.fatal = OOM_MESSAGE
        except Exception as error:
            log.exception("the simulation stopped")
            self.fatal = f"The simulation stopped: {error}"
        if self.fatal and self.on_fatal is not None:
            self.on_fatal()

    def advance(self) -> None:
        """Apply what is queued, recover from any blow-up, and make the next frame or frames."""
        while self._messages:
            self._apply(self._messages.popleft())
        hybrid = self.hybrid
        if hybrid.solver_failed or not hybrid.fields.isfinite().all():
            self._recover()
        info = hybrid.step()
        finite = bool(info.state.isfinite().all())
        if finite:
            self._last_finite, self._failures = info.state, 0
        if info.source == "ai" and finite:
            self._publish(info.frame, info.state, info.doubt_map, info.doubt, "ai")

    def preflight(self) -> None:
        """One hand-off, so that memory runs out now if it is going to, and then a fresh start."""
        self.policy.force_next = True
        self.advance()
        self._load(self.preset)

    def _apply(self, message: dict) -> None:
        kind = message["type"]
        if kind == "brush":
            self._brush(message["x"], message["y"], message["r"], bool(message.get("erase", False)))
        elif kind == "preset":
            self._load(message["name"])
        elif kind == "reset":
            self._load(self.preset)
        elif kind == "threshold":
            self.policy.tau = math.inf if message["tau"] is None else float(message["tau"])
        elif kind == "mode":
            self.policy.mode = message["mode"]

    def _load(self, name: str) -> None:
        self.preset = name
        scene = preset_scene(name, self.solver.res)
        device = self.solver.device
        self._fine_obstacle = torch.from_numpy(scene.obstacle)[None, None].to(device)
        self._coarse_obstacle = self._pooled(self._fine_obstacle)
        inflow = torch.from_numpy(scene.inflow)[None, None].to(device)
        state = torch.zeros(1, 3, self.coarse_res, self.coarse_res, device=device)
        self.hybrid.reset(state, self._fine_obstacle, inflow, scene.buoyancy)
        self.policy.force_next = False
        self._last_finite, self._failures = state, 0
        self._frames_made, self._solver_frames = 0, 0
        self._times.clear()

    def _pooled(self, fine: Tensor) -> Tensor:
        return F.max_pool2d(fine.float(), self.solver.res // self.coarse_res) > 0

    def _brush(self, x: float, y: float, r: float, erase: bool) -> None:
        inside = (self._x - x) ** 2 + (self._y - y) ** 2 < r * r
        self._fine_obstacle = self._fine_obstacle & ~inside if erase else self._fine_obstacle | inside
        self._coarse_obstacle = self._pooled(self._fine_obstacle)
        self.hybrid.set_obstacle(self._fine_obstacle)

    def _recover(self) -> None:
        """Back to the last finite frame, with a hand-off, or to an empty box if that just failed."""
        hybrid = self.hybrid
        self._failures += 1
        log.warning("blown-up state at frame %d, attempt %d to recover", hybrid.frame, self._failures)
        hybrid.fields = self._last_finite if self._failures < 2 else torch.zeros_like(self._last_finite)
        hybrid.solver_failed = False
        self.policy.force_next = True

    def _on_solver_frame(self, frame: int, fields: Tensor, doubt_map: Tensor, doubt: float) -> None:
        if fields.isfinite().all():
            self._publish(frame, fields, doubt_map, doubt, "solver")

    def _publish(self, frame: int, fields: Tensor, doubt_map: Tensor, doubt: float, source: str) -> None:
        self._times.append(time.perf_counter())
        self._frames_made += 1
        self._solver_frames += source == "solver"
        span = self._times[-1] - self._times[0]
        tau = self.policy.tau
        header = {
            "frame": frame,
            "source": source,
            "doubt": doubt if math.isfinite(doubt) else None,
            "tau": tau if math.isfinite(tau) else None,
            "fps": (len(self._times) - 1) / span if span > 0 else 0.0,
            "solver_share": self._solver_frames / self._frames_made,
            "mode": self.policy.mode,
            "members": len(self.ensemble.models),
            "weights": self.weights,
            "res": self.coarse_res,
        }
        encoded = json.dumps(header, allow_nan=False).encode()
        doubt_layer = torch.nan_to_num(doubt_map[0, 0] / self.doubt_full_scale, nan=1.0, posinf=1.0)
        message = b"".join(
            [
                struct.pack("<I", len(encoded)),
                encoded,
                _bytes_of(fields[0, 2] / self.density_full_scale),
                _bytes_of(doubt_layer),
                _bytes_of(self._coarse_obstacle[0, 0].float()),
            ]
        )
        self.latest = (self.latest[0] + 1, message)


async def _send_frames(websocket: WebSocket, sim: DemoSimulation) -> None:
    sent = 0
    try:
        while True:
            number, message = sim.latest
            if number > sent:
                await websocket.send_bytes(message)
                sent = number
            else:
                await asyncio.sleep(_SEND_POLL_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        pass  # The client has gone. The receiving side ends the connection.


def create_app(sim: DemoSimulation, run: bool = True) -> FastAPI:
    """The app. With ``run``, its lifespan runs the simulation thread, whether or not anyone is connected."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run:
            sim.start()
        yield
        if run:
            sim.stop()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        sender = asyncio.create_task(_send_frames(websocket, sim))
        try:
            while True:
                try:
                    sim.submit(json.loads(await websocket.receive_text()))
                except (ValueError, KeyError) as error:  # KeyError: a binary message has no text.
                    log.warning("ignored a message: %s", error)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()

    if (STATIC / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """What the demo and its benchmark both take."""
    parser.add_argument("--random-weights", action="store_true", help="Untrained models. Nothing they show is a result.")
    parser.add_argument("--members", type=int, help="Ensemble size. Default: the benchmark's choice, or all of them.")
    parser.add_argument("--device", default="cuda", help="cuda, or cpu to run slowly on purpose.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ensemble", type=Path, default=Path("results/ensemble.json"))
    parser.add_argument("--selection", type=Path, default=Path("results/handoff_selection.json"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the demo server.")
    add_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preset", default="circle", choices=PRESETS)
    return parser.parse_args(argv)


def _physics(data_root: Path) -> dict:
    if (data_root / "manifest.json").exists():
        return read_manifest(data_root)["config"]
    return yaml.safe_load(DATA_CONFIG.read_text())


def _chosen_members(args: argparse.Namespace) -> int | None:
    if args.members is not None:
        return args.members
    if not args.random_weights and BENCH_PATH.exists():
        return json.loads(BENCH_PATH.read_text()).get("chosen_members")
    return None


def _random_models(count: int, device: str) -> list:
    config = yaml.safe_load(TRAIN_CONFIG.read_text())["model"]
    models = []
    for seed in range(count):
        torch.manual_seed(seed)
        models.append(build_model(config).to(device))
    return models


def _trained_models(path: Path, count: int | None, device: str) -> tuple[list, list[str], bool]:
    """The ensemble in the training report, each checkpoint checked against it, or the first ``count``."""
    if not path.exists():
        raise StartupError(f"{path} is missing: train the ensemble first, or pass --random-weights")
    report = json.loads(path.read_text())
    if not report["complete"]:
        raise StartupError(f"{path} says the ensemble isn't complete: finish it, or pass --random-weights")
    members = report["members"][:count]
    if count is not None and len(members) < count:
        raise StartupError(f"{path} has {len(members)} members, and {count} were asked for")
    models, hashes, amp = [], [], False
    for member in members:
        checkpoint = Path(member["checkpoint"])
        if not checkpoint.exists() or sha256_file(checkpoint) != member["checkpoint_sha256"]:
            raise StartupError(f"{checkpoint} is missing or doesn't match its checksum in {path}")
        saved = load_checkpoint(checkpoint, map_location=device)
        model = build_model(saved["config"]["model"]).to(device)
        model.load_state_dict(saved["model"])
        models.append(model)
        hashes.append(member["checkpoint_sha256"])
        amp = bool(saved["config"]["mixed_precision"]) and device.startswith("cuda")
    return models, hashes, amp


def build_simulation(args: argparse.Namespace, *, need_selection: bool = True) -> DemoSimulation:
    """The simulation the arguments ask for, on the data's grids and time step."""
    count = _chosen_members(args)
    if count is not None and count < 1:
        raise StartupError(f"an ensemble needs at least one member, and --members is {count}")
    physics = _physics(args.data_root)
    solver = SmokeSolver(
        physics["res"],
        args.device,
        physics["dt"],
        cg_tol=physics.get("cg_tol", 1e-5),
        cg_max_iter=physics.get("cg_max_iter", 2000),
    )
    common = {"substeps": physics["steps_per_frame"], "coarse_res": physics["coarse_res"], "preset": getattr(args, "preset", "circle")}
    normalizer_path = args.data_root / "normalizer.json"

    if args.random_weights:
        normalizer = Normalizer.load(normalizer_path) if normalizer_path.exists() else Normalizer(**_PLACEHOLDER_NORMALIZER)
        models = _random_models(count or _RANDOM_MEMBERS, args.device)
        return DemoSimulation(
            Ensemble(models, normalizer),
            solver,
            handoff_frames=_FALLBACK_HANDOFF_FRAMES,
            tau=_RANDOM_TAU,
            weights="random",
            **common,
        )

    if not normalizer_path.exists():
        raise StartupError(f"{normalizer_path} is missing: it comes with the data, or with the released weights")
    choice = {}
    if args.selection.exists():
        choice = json.loads(args.selection.read_text())["choice"]
    elif need_selection:
        raise StartupError(f"{args.selection} is missing: run scripts/select_handoff.py, or pass --random-weights")
    models, hashes, amp = _trained_models(args.ensemble, count, args.device)
    tau = math.inf if choice.get("tau") is None else float(choice["tau"])
    sim = DemoSimulation(
        Ensemble(models, Normalizer.load(normalizer_path)),
        solver,
        handoff_frames=choice.get("handoff_frames") or _FALLBACK_HANDOFF_FRAMES,
        tau=tau,
        doubt_method=choice.get("aggregation") or "mean",
        amp=amp,
        doubt_full_scale=2.0 * tau if math.isfinite(tau) and tau > 0 else _DEFAULT_DOUBT_FULL_SCALE,
        **common,
    )
    sim.checkpoint_sha256 = hashes
    return sim


def check_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise StartupError(NO_GPU_MESSAGE)


def serve(sim: DemoSimulation, host: str, port: int) -> int:
    """Serve until stopped, or until the simulation fails. Returns the process's exit code."""
    server = uvicorn.Server(
        uvicorn.Config(create_app(sim), host=host, port=port, log_level="warning", timeout_graceful_shutdown=2)
    )
    sim.on_fatal = lambda: setattr(server, "should_exit", True)
    server.run()
    if sim.fatal:
        print(sim.fatal, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        check_device(args.device)
        sim = build_simulation(args)
        sim.preflight()
    except torch.cuda.OutOfMemoryError:
        print(OOM_MESSAGE, file=sys.stderr)
        return 1
    except StartupError as error:
        print(error, file=sys.stderr)
        return 1
    if args.random_weights:
        print("Random weights: the smoke and doubt on screen are not those of a trained model.")
    print(f"http://{args.host}:{args.port} on {args.device}, {len(sim.ensemble.models)} members")
    return serve(sim, args.host, args.port)
