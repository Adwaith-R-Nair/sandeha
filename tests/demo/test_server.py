"""The demo server: its protocol, edits, recovery from a blow-up, and how it refuses to start."""

import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from sandeha.data.dataset import Normalizer
from sandeha.demo import server
from sandeha.demo.server import PRESETS, DemoSimulation, create_app, preset_scene, serve
from sandeha.ensemble import Ensemble
from sandeha.models.factory import build_model
from sandeha.physics.scenes import UNSEEN_SHAPES, _has_way_up
from sandeha.physics.solver import SmokeSolver

ROOT = Path(__file__).resolve().parents[2]
FINE, COARSE, DT, SUBSTEPS, HANDOFF = 32, 16, 0.01, 2, 2
WAIT = 20.0


class Member(nn.Module):
    """A member predicting a constant change, or NaN while ``bad``, or running ``fail``."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = value
        self.bad = False
        self.fail = None

    def forward(self, x):
        if self.fail is not None:
            raise self.fail
        return torch.full((x.shape[0], 3, *x.shape[2:]), math.nan if self.bad else self.value)


class HeldSolver(SmokeSolver):
    """A solver whose first step after ``hold`` waits until the test lets it go."""

    def __init__(self):
        super().__init__(FINE, "cpu", DT)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.holding = False

    def step(self, *args):
        if self.holding:
            self.holding = False
            self.entered.set()
            assert self.release.wait(WAIT)
        return super().step(*args)


def normalizer() -> Normalizer:
    return Normalizer(
        field_mean=[0.0, 0.0, 0.0],
        field_std=[1.0, 1.0, 1.0],
        delta_std=[1.0, 1.0, 1.0],
        inflow_scale=5.0,
        buoyancy_mean=2.0,
        buoyancy_std=1.0,
        fitted_on="train",
        n_scenes=1,
    )


def make_sim(members=None, tau=math.inf, mode="auto", solver=None) -> DemoSimulation:
    members = members if members is not None else [Member(0.0), Member(0.01)]
    return DemoSimulation(
        Ensemble(members, normalizer()),
        solver or SmokeSolver(FINE, "cpu", DT),
        handoff_frames=HANDOFF,
        substeps=SUBSTEPS,
        coarse_res=COARSE,
        tau=tau,
        mode=mode,
        weights="random",
    )


def decode(message: bytes) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    length = struct.unpack_from("<I", message)[0]
    header = json.loads(message[4 : 4 + length])
    layers = np.frombuffer(message, np.uint8, offset=4 + length).reshape(3, header["res"], header["res"])
    return header, layers[0], layers[1], layers[2]


def latest(sim: DemoSimulation):
    return decode(sim.latest[1])


def record(sim: DemoSimulation) -> list:
    """Every frame the simulation makes, whether or not a client would get it."""
    made = []
    publish = sim._publish

    def recording(*args):
        publish(*args)
        made.append(latest(sim))

    sim._publish = recording
    return made


def frames_until(ws, condition, seconds: float = WAIT):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = decode(ws.receive_bytes())
        if condition(frame):
            return frame
    raise AssertionError("no frame met the condition in time")


def test_ws_streams_frames():
    sim = make_sim(tau=1e9)
    with TestClient(create_app(sim)) as client, client.websocket_connect("/ws") as ws:
        first = decode(ws.receive_bytes())
        later = frames_until(ws, lambda f: f[0]["frame"] > first[0]["frame"])
    header, density, doubt, obstacle = later
    assert {"frame", "source", "doubt", "tau", "fps", "solver_share", "mode", "members", "weights"} <= set(header)
    assert header["source"] == "ai" and header["mode"] == "auto" and header["tau"] == 1e9
    assert header["members"] == 2 and header["weights"] == "random"
    assert density.shape == doubt.shape == obstacle.shape == (COARSE, COARSE)
    assert doubt.max() > 0  # The stub members disagree everywhere.


def test_header_gives_the_page_its_doubt_scale_and_preset():
    sim = make_sim(tau=1e9)
    sim.doubt_full_scale = 0.5
    with TestClient(create_app(sim)) as client, client.websocket_connect("/ws") as ws:
        header = decode(ws.receive_bytes())[0]
    assert header["doubt_scale"] == 0.5 and header["preset"] == sim.preset


def test_random_weights_doubt_scale_spans_twice_tau(monkeypatch):
    monkeypatch.setattr(server, "_random_models", lambda count, device: [Member(0.0), Member(1.0)])
    args = server.parse_args(["--random-weights", "--device", "cpu", "--data-root", "no-such-directory"])
    sim = server.build_simulation(args)
    assert sim.doubt_full_scale == 2.0 * sim.policy.tau


def test_clients_get_the_latest_frame_and_older_ones_are_dropped():
    sim = make_sim()
    for _ in range(3):
        sim.advance()
    with TestClient(create_app(sim, run=False)) as client, client.websocket_connect("/ws") as ws:
        assert decode(ws.receive_bytes())[0]["frame"] == 3


def test_brush_changes_obstacle_next_frame():
    sim = make_sim()
    sim.submit({"type": "preset", "name": "empty"})
    sim.advance()
    frame = latest(sim)[0]["frame"]
    assert latest(sim)[3].max() == 0
    # x along columns and y along rows, in domain units, with row 0 at the bottom.
    sim.submit({"type": "brush", "x": 0.25, "y": 0.75, "r": 0.1, "erase": False})
    sim.advance()
    header, _, _, obstacle = latest(sim)
    assert header["frame"] == frame + 1
    assert obstacle[12, 4] == 255 and obstacle[4, 12] == 0
    sim.submit({"type": "brush", "x": 0.25, "y": 0.75, "r": 0.1, "erase": True})
    sim.advance()
    assert latest(sim)[3].max() == 0


def test_brush_over_the_socket_reaches_the_obstacle():
    sim = make_sim()
    with TestClient(create_app(sim)) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "preset", "name": "empty"}))
        frames_until(ws, lambda f: f[3].max() == 0)
        ws.send_text(json.dumps({"type": "brush", "x": 0.5, "y": 0.5, "r": 0.2, "erase": False}))
        frames_until(ws, lambda f: f[3].max() == 255)


def test_edit_during_handoff_applies_after():
    solver = HeldSolver()
    sim = make_sim(mode="solver", solver=solver)
    sim.submit({"type": "preset", "name": "empty"})
    sim.advance()
    made = record(sim)
    solver.holding = True
    with TestClient(create_app(sim)) as client, client.websocket_connect("/ws") as ws:
        assert solver.entered.wait(WAIT)
        began = sim.hybrid.frame
        ws.send_text(json.dumps({"type": "brush", "x": 0.5, "y": 0.5, "r": 0.2, "erase": False}))
        ws.send_text(json.dumps({"type": "threshold", "tau": 3.0}))
        deadline = time.monotonic() + WAIT
        while len(sim._messages) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        solver.release.set()
        frames_until(ws, lambda f: f[3].max() == 255 and f[0]["tau"] == 3.0)
    edited = next(header["frame"] for header, _, _, obstacle in made if obstacle.max() == 255)
    assert edited == began + HANDOFF + 1
    assert all(header["source"] == "solver" for header, *_ in made)


def test_handoff_frames_are_published_as_they_are_made():
    solver = HeldSolver()
    sim = make_sim(mode="solver", solver=solver)
    sim.advance()
    made = record(sim)
    sim.advance()
    assert [header["frame"] for header, *_ in made] == [HANDOFF + 1, HANDOFF + 2]
    assert sim.latest[0] == len(made) + HANDOFF


def test_reconnect_keeps_running():
    sim = make_sim()
    with TestClient(create_app(sim)) as client:
        with client.websocket_connect("/ws") as ws:
            before = decode(ws.receive_bytes())[0]["frame"]
        deadline = time.monotonic() + WAIT
        while sim.hybrid.frame <= before + 3:
            assert time.monotonic() < deadline, "the simulation stopped when the client left"
            time.sleep(0.005)
        with client.websocket_connect("/ws") as ws:
            frames_until(ws, lambda f: f[0]["frame"] > before + 3)
    assert sim.fatal is None


def test_fills_domain_stays_finite():
    sim = make_sim(tau=0.0)  # Every frame with any doubt hands off. A filled domain has none.
    sim.submit({"type": "brush", "x": 0.5, "y": 0.5, "r": 2.0, "erase": False})
    sim.advance()
    header, density, _, obstacle = latest(sim)
    assert obstacle.min() == 255 and density.max() == 0
    sim.submit({"type": "mode", "mode": "solver"})
    for _ in range(2):
        sim.advance()
    header, density, _, obstacle = latest(sim)
    assert header["source"] == "solver" and obstacle.min() == 255
    assert sim.hybrid.fields.isfinite().all()
    sim.submit({"type": "mode", "mode": "ai"})
    sim.advance()
    assert latest(sim)[0]["source"] == "ai"
    sim.submit({"type": "reset"})
    sim.advance()
    assert latest(sim)[3].min() == 0


def test_brush_seals_off_and_covers_the_inflow_and_keeps_running():
    sim = make_sim(mode="solver")
    sim.submit({"type": "preset", "name": "empty"})
    sim.submit({"type": "brush", "x": 0.5, "y": 0.12, "r": 0.2, "erase": False})
    for x in np.linspace(0.0, 1.0, 12):
        sim.submit({"type": "brush", "x": float(x), "y": 0.5, "r": 0.1, "erase": False})
    for _ in range(3):
        sim.advance()
    assert sim.hybrid.fields.isfinite().all()


def test_blown_up_state_recovers():
    sim = make_sim(tau=1e9)
    for _ in range(2):
        sim.advance()
    assert latest(sim)[0]["source"] == "ai"
    good = sim.hybrid.fields
    sim.hybrid.fields = torch.full_like(good, math.nan)
    sim.advance()
    header, density, *_ = latest(sim)
    assert header["source"] == "solver"
    assert sim.hybrid.fields.isfinite().all() and not sim.hybrid.solver_failed
    sim.advance()
    assert latest(sim)[0]["source"] == "ai"


@pytest.mark.parametrize("mode", ["auto", "ai"])
def test_a_model_going_non_finite_hands_off_in_every_mode_but_solver(mode):
    members = [Member(0.0), Member(0.01)]
    sim = make_sim(members, tau=1e9, mode=mode)
    sim.advance()
    members[0].bad = True
    sim.advance()
    header, density, doubt, _ = latest(sim)
    assert header["source"] == "solver" and header["doubt"] is None
    json.dumps(header, allow_nan=False)
    assert sim.hybrid.fields.isfinite().all()
    members[0].bad = False
    sim.advance()
    assert latest(sim)[0]["source"] == "ai"


def test_a_failing_solver_falls_back_and_the_simulation_keeps_going():
    class FailingOnce(SmokeSolver):
        fail = 0

        def step(self, state, *args):
            out = super().step(state, *args)
            if self.fail:
                self.fail -= 1
                out.density = torch.full_like(out.density, math.nan)
            return out

    solver = FailingOnce(FINE, "cpu", DT)
    sim = make_sim(mode="solver", solver=solver)
    sim.advance()
    solver.fail = 1
    sim.advance()
    assert sim.hybrid.solver_failed
    sim.advance()
    assert not sim.hybrid.solver_failed and sim.hybrid.fields.isfinite().all()


def test_threshold_and_mode_control_the_source():
    sim = make_sim(tau=1e9)
    sim.advance()
    assert latest(sim)[0]["source"] == "ai"
    sim.submit({"type": "threshold", "tau": 0.0})
    sim.advance()
    header = latest(sim)[0]
    assert header["source"] == "solver" and header["tau"] == 0.0 and header["solver_share"] > 0
    sim.submit({"type": "mode", "mode": "ai"})
    sim.advance()
    assert latest(sim)[0]["source"] == "ai" and latest(sim)[0]["mode"] == "ai"
    sim.submit({"type": "threshold", "tau": None})
    sim.submit({"type": "mode", "mode": "solver"})
    sim.advance()
    header = latest(sim)[0]
    assert header["source"] == "solver" and header["tau"] is None
    sim.submit({"type": "reset"})
    sim.submit({"type": "mode", "mode": "auto"})
    sim.advance()
    assert latest(sim)[0]["solver_share"] == 0


@pytest.mark.parametrize(
    "message",
    [
        {"type": "nonsense"},
        {"type": "brush", "x": math.nan, "y": 0.5, "r": 0.1, "erase": False},
        {"type": "brush", "x": 0.5, "y": 0.5, "r": -1.0, "erase": False},
        {"type": "brush", "x": "left", "y": 0.5, "r": 0.1},
        {"type": "preset", "name": "no such"},
        {"type": "threshold", "tau": math.nan},
        {"type": "threshold", "tau": -1.0},
        {"type": "threshold"},
        {"type": "mode", "mode": "turbo"},
        ["not", "a", "dict"],
    ],
)
def test_a_bad_message_is_refused(message):
    with pytest.raises(ValueError):
        make_sim().submit(message)


def test_bad_messages_do_not_end_the_connection():
    sim = make_sim()
    with TestClient(create_app(sim)) as client, client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        ws.send_bytes(b"\x00")
        ws.send_text(json.dumps({"type": "nonsense"}))
        ws.send_text(json.dumps({"type": "threshold"}))
        ws.send_text(json.dumps({"type": "mode", "mode": "solver"}))
        frames_until(ws, lambda f: f[0]["mode"] == "solver")
    assert sim.fatal is None


def test_presets_include_unseen_shapes_and_leave_every_source_open():
    assert set(UNSEEN_SHAPES) <= set(PRESETS) and "empty" in PRESETS
    for name in PRESETS:
        scene = preset_scene(name, 256)
        assert scene.obstacle.shape == (256, 256)
        assert (name == "empty") == (not scene.obstacle.any())
        assert not (scene.obstacle & (scene.inflow > 0)).any()
        assert _has_way_up(scene.obstacle, scene.inflow)
    assert np.array_equal(preset_scene("star", 64).obstacle, preset_scene("star", 64).obstacle)


def test_small_random_unets_run_end_to_end():
    torch.manual_seed(0)
    models = [build_model({"arch": "unet", "base_width": 8, "levels": 2}) for _ in range(2)]
    sim = make_sim(models, tau=1e9)
    sim.submit({"type": "preset", "name": "star"})
    sim.advance()
    sim.submit({"type": "mode", "mode": "solver"})
    sim.advance()
    assert latest(sim)[0]["source"] == "solver"
    assert latest(sim)[3].max() == 255


def test_no_gpu_exits_with_message():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_demo.py"), "--random-weights"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0
    assert "No GPU" in done.stderr and "Traceback" not in done.stderr


def test_oom_exits_with_message(monkeypatch, capsys):
    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(server, "build_simulation", out_of_memory)
    assert server.main(["--random-weights", "--device", "cpu"]) != 0
    assert "out of GPU memory" in capsys.readouterr().err


def test_oom_while_running_exits_with_message(capsys):
    member = Member(0.0)
    sim = make_sim([member])
    member.fail = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert serve(sim, "127.0.0.1", 0) != 0
    assert "out of GPU memory" in capsys.readouterr().err


def test_missing_ensemble_exits_with_message(tmp_path, capsys):
    (tmp_path / "normalizer.json").write_text(json.dumps(normalizer().to_dict()))
    code = server.main(
        ["--device", "cpu", "--data-root", str(tmp_path), "--ensemble", str(tmp_path / "none.json")]
    )
    assert code != 0
    assert "--random-weights" in capsys.readouterr().err


def test_random_weights_build_says_so_and_runs(capsys):
    args = server.parse_args(["--random-weights", "--device", "cpu", "--members", "2"])
    sim = server.build_simulation(args)
    assert sim.weights == "random" and len(sim.ensemble.models) == 2
    assert sim.hybrid.coarse_res == 128 and sim.solver.res == 256


def test_bench_refuses_to_write_results_from_random_weights():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bench_demo.py"), "--random-weights", "--device", "cpu"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0
    assert "--out" in done.stderr
