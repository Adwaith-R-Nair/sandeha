"""The hand-off policies, and the simulator that steps with the ensemble or the solver.

The rules are in docs/PREREGISTRATION.md, "Hand-off rules". Each step runs the ensemble into the next
frame and asks the policy. On a hand-off the ensemble's frame is thrown away, the solver starts from
the frame before and makes ``handoff_frames`` frames, starting with the one that triggered, and the
ensemble resumes from the solver's last frame.

A solid in force from frame f to frame f + 1 is applied as it is in the data (``data/store.py``).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sandeha.ensemble import doubt_score
from sandeha.physics.solver import SmokeState
from sandeha.train.loop import ERROR_CAP, amp_context

_FIRST_WINDOW_FRAME = 21  # Rollouts start from frame 20, so their first own frame is 21.


class Policy:
    """Decides, for a frame the ensemble has just made, whether to throw it away and hand off.

    ``rescues`` says a non-finite output hands off under this policy even if ``decide`` says no. Only
    the no-hand-off policies leave it off, since they are the record of what happens with no rescue.
    """

    rescues = True

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        """True to hand off now. ``ctx`` holds ``output_finite``, and ``error`` if one was given."""
        raise NotImplementedError


class AlwaysAI(Policy):
    """No hand-off at all, whatever happens."""

    rescues = False

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        return False


class AlwaysSolver(Policy):
    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        return True


class FixedSchedule(Policy):
    """A hand-off every ``period`` frames, the first ``offset`` frames after ``start``.

    The schedule depends on the frame alone, so a scheduled frame inside another hand-off, such as a
    forced one, is passed over, and the ones after it stay where they were.
    """

    def __init__(self, period: int, offset: int = 0, start: int = _FIRST_WINDOW_FRAME):
        if period < 1:
            raise ValueError(f"a period of {period} frames isn't a schedule")
        self.period = period
        self.offset = offset
        self.start = start

    @classmethod
    def for_scene(cls, period: int, scene: int, n_scenes: int, start: int = _FIRST_WINDOW_FRAME) -> "FixedSchedule":
        """The set's scene number ``scene`` of ``n_scenes``, so that scenes don't all hand off together."""
        return cls(period, scene * period // n_scenes, start)

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        since = frame - self.start - self.offset
        return since >= 0 and since % self.period == 0


class RandomBudget(Policy):
    """A hand-off starts with chance ``rate`` at each frame the ensemble makes.

    A frame's draw depends only on the scene, the seed and the frame number, so changing ``rate``
    moves the same stream's threshold, and a frame passed over by a hand-off doesn't shift the rest.
    """

    def __init__(self, rate: float, seed: int, scene: int = 0):
        self.rate = rate
        self._generator = np.random.default_rng([seed, scene])
        self._draws = np.empty(0)

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        if frame >= len(self._draws):
            extra = max(256, frame + 1 - len(self._draws))
            self._draws = np.concatenate([self._draws, self._generator.random(extra)])
        return bool(self._draws[frame] < self.rate)


class DoubtThreshold(Policy):
    """Hands off above ``tau`` or on non-finite doubt. ``tau = inf`` is the **never** candidate."""

    def __init__(self, tau: float):
        if math.isnan(tau):
            raise ValueError("tau can't be NaN")
        self.tau = tau
        self.rescues = not math.isinf(tau)

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        if math.isinf(self.tau):
            return False
        return not math.isfinite(doubt) or doubt > self.tau


class Oracle(Policy):
    """Hands off above ``tau`` on the current frame's true error, in ``ctx["error"]``.

    A non-finite output answers no here, because the simulator forces that hand-off and counts it.
    """

    def __init__(self, tau: float):
        self.tau = tau

    def decide(self, frame: int, doubt: float, ctx: dict) -> bool:
        if not ctx["output_finite"]:
            return False
        error = ctx["error"]
        return (error if math.isfinite(error) else ERROR_CAP) > self.tau


@dataclass
class StepInfo:
    """What one ``step`` did. It advances one frame, or several in a hand-off.

    - ``state`` ``(1,3,H,W)``: the last frame made, in physical units.
    - ``frames``: each frame made, in order, of which ``frame`` is the number of the last.
    - ``doubt_map``, ``doubt``: from the ensemble step into the first of them. A solver frame has no
      doubt of its own.
    - ``source``: ``"ai"`` or ``"solver"``.
    - ``ms``: the whole step, with every member and any solver steps.
    - ``forced``: a hand-off because of a non-finite output, which the policy didn't ask for.
    - ``solver_failed``: the solver has failed in this rollout, and no hand-off will be tried again.
    """

    state: Tensor
    doubt_map: Tensor
    doubt: float
    source: str
    ms: float
    frames_advanced: int
    frame: int
    frames: list[Tensor]
    forced: bool = False
    solver_failed: bool = False


class HybridSimulator:
    """One scene, stepped by the ensemble, with whole-frame hand-offs to the solver.

    ``error_of(frame, fields, solid)`` gives the true error of the ensemble's output for ``frame``,
    with ``solid`` the mask on the way into it. It is called only for that frame, and only for ``Oracle``.
    ``on_solver_frame(frame, fields, doubt_map, doubt)`` is called for each hand-off frame as it is made,
    with the doubt that came from the ensemble step into the hand-off.
    """

    def __init__(
        self,
        ensemble,
        solver,
        policy: Policy,
        handoff_frames: int,
        substeps: int = 4,
        *,
        doubt_method: str = "mean",
        amp: bool = False,
        error_of: Callable[[int, Tensor, Tensor], float] | None = None,
        on_solver_frame: Callable[[int, Tensor, Tensor, float], None] | None = None,
    ):
        self.ensemble = ensemble
        self.solver = solver
        self.policy = policy
        self.handoff_frames = handoff_frames
        self.substeps = substeps
        self.doubt_method = doubt_method
        self.amp = amp
        self.error_of = error_of
        self.on_solver_frame = on_solver_frame
        self.frame = 0
        self.solver_failed = False
        self.fine_state: SmokeState | None = None  # The last hand-off's state at the reference grid.

    def reset(
        self,
        state: Tensor,
        obstacle: Tensor,
        inflow: Tensor,
        buoyancy: float,
        *,
        frame: int = 0,
        late_obstacle: Tensor | None = None,
        late_frame: int | None = None,
    ) -> None:
        """Start from physical fields ``(1,3,H,W)``, at frame number ``frame``.

        ``obstacle``, ``late_obstacle`` and ``inflow`` are ``(1,1,H,W)``, at the reference grid when
        known, or at the neural grid, in which case the reference grid's are made by repeating cells.
        """
        if state.dim() != 4 or state.shape[0] != 1:
            raise ValueError(f"a simulator holds one scene, got fields of shape {state.shape}")
        device = self.solver.device
        self.fields = state.to(device)
        self.coarse_res = state.shape[-1]
        self.factor = self.solver.res // self.coarse_res
        self.frame = frame
        self.solver_failed = False
        self.fine_state = None
        self.fine_inflow, self.coarse_inflow = self._two_grids(inflow.float(), F.avg_pool2d)
        self.lift = torch.tensor([float(buoyancy)], dtype=torch.float32, device=device)
        self._late = None
        if late_obstacle is not None and late_frame is not None:
            self._late = (*self._masks(late_obstacle), late_frame)
        self.set_obstacle(obstacle)

    def set_obstacle(self, obstacle: Tensor) -> None:
        """Replace the solid from the next step on. A hand-off under way keeps the solid it began with."""
        self._solid = self._masks(obstacle)

    def _two_grids(self, field: Tensor, pool) -> tuple[Tensor, Tensor]:
        field = field.to(self.solver.device)
        if field.shape[-1] == self.solver.res:
            return field, pool(field, self.factor)
        if field.shape[-1] == self.coarse_res:
            return field.repeat_interleave(self.factor, -1).repeat_interleave(self.factor, -2), field
        raise ValueError(f"a mask of shape {field.shape} is on neither grid")

    def _masks(self, obstacle: Tensor) -> tuple[Tensor, Tensor]:
        fine, coarse = self._two_grids(obstacle.float(), F.max_pool2d)
        return fine > 0, coarse > 0

    def _solid_at(self, solid: tuple[Tensor, Tensor], frame: int) -> tuple[Tensor, Tensor]:
        """The fine and coarse solids in force from ``frame`` to ``frame + 1``."""
        if self._late is not None and self._late[2] <= frame:
            return solid[0] | self._late[0], solid[1] | self._late[1]
        return solid

    def _sync(self) -> None:
        if self.solver.device.type == "cuda":
            torch.cuda.synchronize()

    def step(self) -> StepInfo:
        self._sync()
        began = time.perf_counter()
        solid = self._solid  # Fixed for this step, so that an edit arriving during it waits.
        before, frame = self.frame, self.frame + 1
        coarse_solid = self._solid_at(solid, before)[1]

        x = self.ensemble.normalizer.make_input(self.fields, coarse_solid, self.coarse_inflow, self.lift)
        with amp_context(self.solver.device.type, self.amp):
            out = self.ensemble.step(x)
        doubt = float(doubt_score(out.doubt_map, coarse_solid, self.doubt_method)[0])

        hand_off, forced = False, False
        if not self.solver_failed:
            finite = bool(out.mean.isfinite().all())
            ctx = {"output_finite": finite}
            if self.error_of is not None:
                ctx["error"] = self.error_of(frame, out.mean, coarse_solid)
            hand_off = self.policy.decide(frame, doubt, ctx)
            forced = not hand_off and not finite and self.policy.rescues
            hand_off = hand_off or forced

        frames = self._hand_off(solid, before, out.doubt_map, doubt) if hand_off else [out.mean]
        if self.solver_failed:
            # That frame and all later ones count as blown up, whatever a model makes of them.
            frames[-1] = torch.full_like(frames[-1], torch.nan)
        self.fields = frames[-1]
        self.frame = before + len(frames)
        self._sync()
        return StepInfo(
            state=self.fields,
            doubt_map=out.doubt_map,
            doubt=doubt,
            source="solver" if hand_off else "ai",
            ms=1000 * (time.perf_counter() - began),
            frames_advanced=len(frames),
            frame=self.frame,
            frames=frames,
            forced=forced,
            solver_failed=self.solver_failed,
        )

    def _hand_off(self, solid: tuple[Tensor, Tensor], before: int, doubt_map: Tensor, doubt: float) -> list[Tensor]:
        """Up to ``handoff_frames`` frames from the kept frame. A failing solver stops at the failure."""
        fine = F.interpolate(self.fields, scale_factor=self.factor, mode="bilinear", align_corners=False)
        state = SmokeState(fine[:, :2], fine[:, 2:])
        state = self.solver.project(state, self._solid_at(solid, before)[0])
        frames = []
        for k in range(self.handoff_frames):
            fine_solid, coarse_solid = self._solid_at(solid, before + k)
            for _ in range(self.substeps):
                state = self.solver.step(state, fine_solid, self.fine_inflow, self.lift)
            # The same 2x2 average, zero in solid cells, that the data stores.
            pooled = F.avg_pool2d(torch.cat([state.velocity, state.density], dim=1), self.factor)
            frames.append(pooled * ~coarse_solid)
            if self.on_solver_frame is not None:
                self.on_solver_frame(before + k + 1, frames[-1], doubt_map, doubt)
            if not frames[-1].isfinite().all():
                self.solver_failed = True
                break
        self.fine_state = state
        return frames
