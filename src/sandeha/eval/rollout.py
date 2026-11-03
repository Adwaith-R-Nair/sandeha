"""Rollouts of one scene against its stored reference, for every simulator and baseline.

``rollout`` works on anything with the ``HybridSimulator`` interface, so the persistence, coarse
solver and independent baselines are subclasses that replace ``step``.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from sandeha.eval.metrics import frame_secondary
from sandeha.handoff_selection import Window, _scene_inputs, frame_error
from sandeha.ensemble import doubt_score
from sandeha.hybrid import HybridSimulator, StepInfo
from sandeha.models.factory import apply_step
from sandeha.physics.solver import SmokeState
from sandeha.train.loop import ERROR_CAP, amp_context


@dataclass
class RolloutTrace:
    """One scene's frames from ``start + 1`` on, in order.

    ``errors`` are per-frame errors, with a blown-up frame at the cap. Nothing is ever left out, so
    each list runs the whole horizon. ``ms`` and ``frames_made`` cover whole steps, so a hand-off
    running past the horizon is timed over all the frames it made.
    """

    errors: list[float] = field(default_factory=list)
    solver_flags: list[bool] = field(default_factory=list)
    forced: int = 0
    ms: float = 0.0
    frames_made: int = 0
    scores: dict[str, list[float]] = field(default_factory=dict)
    secondary: list[dict] = field(default_factory=list)

    @property
    def blown_up(self) -> bool:
        return max(self.errors) >= ERROR_CAP

    def to_dict(self) -> dict:
        return asdict(self)


def rollout(
    sim, scene: int, ref, start: int, horizon: int, *, floors, scores_by=(), device=None
) -> RolloutTrace:
    """Roll scene number ``scene`` of the split ``ref`` from its stored frame ``start``.

    ``scores_by`` is for rollouts with no hand-off, where every frame has a doubt score of its own.
    """
    device = device or sim.solver.device
    end = start + horizon
    sim.reset(**_scene_inputs(ref, scene, Window(start, end), device))
    trace = RolloutTrace(scores={name: [] for name in scores_by})
    while sim.frame < end:
        first = sim.frame + 1
        info = sim.step()
        if info.doubt_map is not None:
            mask = torch.from_numpy(ref.obstacle(scene, first - 1))[None, None].to(info.doubt_map.device)
            for name in scores_by:
                trace.scores[name].append(float(doubt_score(info.doubt_map, mask, name)[0]))
        trace.forced += int(info.forced)
        trace.ms += info.ms
        trace.frames_made += len(info.frames)
        for k, fields in enumerate(info.frames):
            frame = first + k
            if frame > end:
                break
            pred = fields[0].detach().cpu().numpy()
            stored = np.array(ref.fields[scene, frame], dtype=np.float32)
            solid = ref.obstacle(scene, frame - 1)
            trace.errors.append(frame_error(pred, stored, solid, floors))
            trace.solver_flags.append(info.source == "solver")
            trace.secondary.append(frame_secondary(pred, stored, solid))
    return trace


def oracle_error_of(ref, scene: int, floors):
    """The ``error_of`` of an oracle: the output's true error at that same frame, and no other."""

    def error_of(frame: int, fields, solid) -> float:
        stored = np.array(ref.fields[scene, frame], dtype=np.float32)
        return frame_error(fields[0].detach().cpu().numpy(), stored, solid[0, 0].cpu().numpy(), floors)

    return error_of


class _Baseline(HybridSimulator):
    """A simulator's masks and scene, with a ``step`` of its own and no hand-off."""

    def _info(self, fields, doubt_map, began: float) -> StepInfo:
        self.fields = fields
        self.frame += 1
        self._sync()
        return StepInfo(
            state=fields,
            doubt_map=doubt_map,
            doubt=float("nan"),
            source="ai",
            ms=1000 * (time.perf_counter() - began),
            frames_advanced=1,
            frame=self.frame,
            frames=[fields],
        )


class PersistenceSim(_Baseline):
    """Nothing changes."""

    def __init__(self, solver):
        super().__init__(None, solver, None, 1)

    def step(self) -> StepInfo:
        self._sync()
        return self._info(self.fields, None, time.perf_counter())


class CoarseSolverSim(_Baseline):
    """The solver on the neural grid, with the reference's frame length, from the stored frame."""

    def __init__(self, fine_solver, coarse_solver, substeps: int):
        super().__init__(None, fine_solver, None, 1, substeps)
        self.coarse_solver = coarse_solver
        self._state = None

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._state = SmokeState(self.fields[:, :2], self.fields[:, 2:])

    def step(self) -> StepInfo:
        self._sync()
        began = time.perf_counter()
        solid = self._solid_at(self._solid, self.frame)[1]
        state = self._state
        for _ in range(self.substeps):
            state = self.coarse_solver.step(state, solid, self.coarse_inflow, self.lift)
        self._state = state
        return self._info(torch.cat([state.velocity, state.density], dim=1) * ~solid, None, began)


class IndependentSim(_Baseline):
    """Every member feeds on its own output, the secondary comparison. The frame is their mean."""

    def __init__(self, ensemble, solver, substeps: int, amp: bool = False):
        super().__init__(ensemble, solver, None, 1, substeps, amp=amp)
        self._states = None

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._states = [self.fields] * len(self.ensemble.models)

    @torch.no_grad()
    def step(self) -> StepInfo:
        self._sync()
        began = time.perf_counter()
        solid = self._solid_at(self._solid, self.frame)[1]
        normalizer = self.ensemble.normalizer
        outputs = []
        with amp_context(self.solver.device.type, self.amp):
            for model, state in zip(self.ensemble.models, self._states, strict=True):
                x = normalizer.make_input(state, solid, self.coarse_inflow, self.lift)
                outputs.append(apply_step(model, x, normalizer))
        self._states = outputs
        members = torch.stack(outputs)
        # The ensemble's own doubt map, as the spread of these separate rollouts.
        encoded = normalizer.encode(members)
        doubt_map = (encoded - encoded[:1]).var(dim=0, unbiased=False).sqrt().mean(dim=1, keepdim=True)
        return self._info(members.mean(dim=0), doubt_map, began)


def timed_reference(solver, store, scenes, start: int, frames: int, substeps: int) -> float:
    """The reference solver's milliseconds per frame, from the stored fine state where there is one."""
    device = solver.device
    total, made = 0.0, 0
    for scene in scenes:
        inputs = store.fine_scene(scene)
        solid = torch.from_numpy(inputs.obstacle)[None, None].to(device)
        inflow = torch.from_numpy(inputs.inflow)[None, None].float().to(device)
        lift = torch.tensor([inputs.buoyancy], dtype=torch.float32, device=device)
        try:
            state = store.fine_state(scene, start)
            state = SmokeState(state.velocity.to(device), state.density.to(device))
        except (KeyError, FileNotFoundError):  # A tiny split keeps no fine state, and only times here.
            coarse = torch.from_numpy(np.array(store.fields[scene, start], dtype=np.float32))[None].to(device)
            factor = solver.res // coarse.shape[-1]
            fine = coarse.repeat_interleave(factor, -1).repeat_interleave(factor, -2)
            state = SmokeState(fine[:, :2], fine[:, 2:])
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(frames * substeps):
            state = solver.step(state, solid, inflow, lift)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total += time.perf_counter() - began
        made += frames
    return 1000 * total / made
