"""The hand-off policies, and the simulator that follows the preregistration's hand-off rules."""

import math
import time

import pytest
import torch
from torch import nn

from sandeha.data.dataset import Normalizer
from sandeha.ensemble import Ensemble
from sandeha.hybrid import (
    AlwaysAI,
    AlwaysSolver,
    DoubtThreshold,
    FixedSchedule,
    HybridSimulator,
    Oracle,
    RandomBudget,
)
from sandeha.physics.solver import SmokeSolver, SmokeState

FINE, COARSE, DT, SUBSTEPS = 32, 16, 0.01, 4


@pytest.fixture
def normalizer() -> Normalizer:
    return Normalizer(
        field_mean=[0.0, 0.0, 0.0],
        field_std=[1.0, 1.0, 1.0],
        delta_std=[1.0, 1.0, 1.0],
        inflow_scale=5.0,
        buoyancy_mean=1.0,
        buoyancy_std=1.0,
        fitted_on="train",
        n_scenes=1,
    )


class Member(nn.Module):
    """A member predicting a constant change, or NaN while ``bad``. It records its inputs."""

    def __init__(self, value: float = 0.0, seconds: float = 0.0):
        super().__init__()
        self.value = value
        self.seconds = seconds
        self.bad = False
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        if self.seconds:
            time.sleep(self.seconds)
        return torch.full((x.shape[0], 3, *x.shape[2:]), math.nan if self.bad else self.value)


class CountingSolver(SmokeSolver):
    """Counts steps, keeps the masks it was given, and can run a hook or fail."""

    def __init__(self, fail: bool = False, hook=None):
        super().__init__(FINE, "cpu", DT)
        self.steps = 0
        self.masks = []
        self.fail = fail
        self.hook = hook

    def step(self, state, obstacle, inflow, buoyancy):
        self.steps += 1
        self.masks.append(obstacle.clone())
        if self.hook:
            self.hook(self.steps)
        if self.fail:
            nan = torch.full_like(state.velocity, math.nan)
            return SmokeState(nan, torch.full_like(state.density, math.nan), None)
        return super().step(state, obstacle, inflow, buoyancy)


def start_fields(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.1 * torch.rand(1, 3, COARSE, COARSE, generator=generator)


def fine_mask() -> torch.Tensor:
    solid = torch.zeros(1, 1, FINE, FINE, dtype=torch.bool)
    solid[..., 12:16, 12:16] = True
    return solid


def fine_inflow() -> torch.Tensor:
    inflow = torch.zeros(1, 1, FINE, FINE)
    inflow[..., 2:6, 12:20] = 4.0
    return inflow


def make(normalizer, policy, members=None, solver=None, handoff_frames=2, frame=20, error_of=None, **reset):
    members = members if members is not None else [Member(0.0), Member(0.0)]
    solver = solver or CountingSolver()
    sim = HybridSimulator(
        Ensemble(members, normalizer), solver, policy, handoff_frames, SUBSTEPS, error_of=error_of
    )
    sim.reset(start_fields(), fine_mask(), fine_inflow(), 2.0, frame=frame, **reset)
    return sim, members, solver


def solver_frames(fields, masks, handoff_frames):
    """A separate run of what a hand-off should do, from ``fields``. ``masks[k]`` is for frame k."""
    solver = SmokeSolver(FINE, "cpu", DT)
    fine = torch.nn.functional.interpolate(fields, scale_factor=2, mode="bilinear", align_corners=False)
    state = solver.project(SmokeState(fine[:, :2], fine[:, 2:]), masks[0])
    frames = []
    for k in range(handoff_frames):
        for _ in range(SUBSTEPS):
            state = solver.step(state, masks[k], fine_inflow(), torch.tensor([2.0]))
        pooled = torch.nn.functional.avg_pool2d(torch.cat([state.velocity, state.density], dim=1), 2)
        frames.append(pooled * ~(torch.nn.functional.max_pool2d(masks[k].float(), 2) > 0))
    return frames


# The simulator's hand-off


def test_handoff_returns_low_divergence_state(normalizer):
    sim, _, solver = make(normalizer, AlwaysSolver())
    info = sim.step()
    assert info.source == "solver"
    speed = max(float(sim.fine_state.mac[0].abs().max()), float(sim.fine_state.mac[1].abs().max()))
    assert speed > 0
    divergence = solver.divergence(sim.fine_state, fine_mask()).abs().max()
    assert divergence <= 1e-3 * speed * FINE


def test_handoff_starts_from_the_frame_before_and_emits_each_frame(normalizer):
    sim, members, solver = make(normalizer, AlwaysSolver(), handoff_frames=3)
    info = sim.step()
    assert (info.frame, info.frames_advanced, len(info.frames)) == (23, 3, 3)
    assert solver.steps == 3 * SUBSTEPS
    for got, want in zip(info.frames, solver_frames(start_fields(), [fine_mask()] * 3, 3), strict=True):
        assert torch.allclose(got, want, atol=1e-6)
    assert torch.equal(info.state, info.frames[-1])

    sim.policy = AlwaysAI()
    resumed = sim.step()
    assert resumed.source == "ai"
    assert resumed.frame == 24
    assert torch.equal(members[0].inputs[-1][:, :3], normalizer.encode(info.state))


def test_late_obstacle_joins_inside_a_handoff_as_in_the_data(normalizer):
    late = torch.zeros(1, 1, FINE, FINE, dtype=torch.bool)
    late[..., 24:28, 4:8] = True
    # Frame 22 is the first with the late solid in force on the way out of it: frame 21 is the trigger.
    sim, _, solver = make(normalizer, AlwaysSolver(), handoff_frames=3, late_obstacle=late, late_frame=21)
    sim.step()
    assert not solver.masks[SUBSTEPS - 1][late].any()  # Into frame 21, from frame 20.
    assert solver.masks[SUBSTEPS][late].all()  # Into frame 22, from frame 21.

    sim.policy = AlwaysAI()
    members_x = sim.ensemble.models[0]
    info = sim.step()
    assert info.frame == 24
    coarse_late = torch.nn.functional.max_pool2d(late.float(), 2) > 0
    assert (members_x.inputs[-1][:, 3:4][coarse_late] == 1).all()


def test_late_obstacle_uses_the_old_solid_before_its_frame(normalizer):
    late = torch.zeros(1, 1, FINE, FINE, dtype=torch.bool)
    late[..., 24:28, 4:8] = True
    sim, members, _ = make(normalizer, AlwaysAI(), late_obstacle=late, late_frame=21)
    coarse_late = torch.nn.functional.max_pool2d(late.float(), 2) > 0
    sim.step()  # From frame 20, under the old solid.
    assert (members[0].inputs[-1][:, 3:4][coarse_late] == 0).all()
    sim.step()  # From frame 21, the event frame, under the new solid.
    assert (members[0].inputs[-1][:, 3:4][coarse_late] == 1).all()


def test_failing_solver_ends_handoffs_and_later_frames_stay_blown_up(normalizer):
    solver = CountingSolver(fail=True)
    sim, _, solver = make(normalizer, AlwaysSolver(), solver=solver, handoff_frames=5)
    info = sim.step()
    assert info.solver_failed
    assert info.frames_advanced == 1  # The hand-off stops at the frame that failed.
    assert not info.state.isfinite().all()
    assert solver.steps == SUBSTEPS

    later = sim.step()
    assert later.source == "ai"
    assert later.solver_failed
    assert later.state.isnan().all()
    assert solver.steps == SUBSTEPS


# Policies


def test_fixed_schedule_period():
    policy = FixedSchedule(period=5, offset=2, start=21)
    triggered = [f for f in range(21, 40) if policy.decide(f, 0.0, {})]
    assert triggered == [23, 28, 33, 38]


def test_fixed_schedule_offset_per_scene():
    # floor(i * period / n) frames after frame 21.
    offsets = [FixedSchedule.for_scene(period=7, scene=i, n_scenes=30).offset for i in (0, 1, 4, 5, 29)]
    assert offsets == [0, 0, 0, 1, 6]
    assert FixedSchedule.for_scene(period=7, scene=5, n_scenes=30).decide(22, 0.0, {})


def test_scheduled_frame_inside_a_forced_handoff_is_skipped(normalizer):
    members = [Member(0.0)]
    sim, _, _ = make(normalizer, FixedSchedule(period=3, offset=0, start=21), members=members, handoff_frames=2)
    sources = [sim.step() for _ in range(3)]
    assert [(i.frame, i.source) for i in sources] == [(22, "solver"), (23, "ai"), (25, "solver")]


def test_random_budget_rate_within_tolerance():
    policy = RandomBudget(rate=0.2, seed=3, scene=1)
    draws = [policy.decide(f, 0.0, {}) for f in range(20000)]
    assert abs(sum(draws) / len(draws) - 0.2) < 0.02


def test_random_budget_stream_is_fixed_per_scene_and_seed():
    frames = range(21, 200)
    first = [RandomBudget(0.3, seed=1, scene=2).decide(f, 0.0, {}) for f in frames]
    again = RandomBudget(0.3, seed=1, scene=2)
    # A frame's draw doesn't depend on which frames were read before it.
    assert [again.decide(f, 0.0, {}) for f in reversed(frames)] == first[::-1]
    assert first != [RandomBudget(0.3, seed=1, scene=3).decide(f, 0.0, {}) for f in frames]
    assert first != [RandomBudget(0.3, seed=2, scene=2).decide(f, 0.0, {}) for f in frames]


def test_threshold_triggers_only_above_tau(normalizer):
    policy = DoubtThreshold(0.5)
    assert not policy.decide(21, 0.4, {})
    assert not policy.decide(21, 0.5, {})
    assert policy.decide(21, 0.6, {})
    assert policy.decide(21, math.inf, {})
    assert policy.decide(21, math.nan, {})

    calm = make(normalizer, DoubtThreshold(0.5), members=[Member(0.0), Member(0.2)])[0].step()
    assert calm.source == "ai"
    assert calm.doubt == pytest.approx(0.1)
    wild = make(normalizer, DoubtThreshold(0.05), members=[Member(0.0), Member(0.2)])[0].step()
    assert wild.source == "solver"
    assert not wild.forced


def test_doubt_score_uses_the_chosen_aggregation(normalizer):
    members = [Member(0.0), Member(0.2)]
    sim = HybridSimulator(
        Ensemble(members, normalizer), CountingSolver(), AlwaysAI(), 2, SUBSTEPS, doubt_method="p99"
    )
    sim.reset(start_fields(), fine_mask(), fine_inflow(), 2.0)
    assert sim.step().doubt == pytest.approx(0.1)


def test_non_finite_doubt_triggers_and_starts_from_the_last_finite_frame(normalizer):
    members = [Member(0.0), Member(0.0)]
    sim, members, _ = make(normalizer, DoubtThreshold(1.0), members=members, handoff_frames=2)
    first = sim.step()
    assert first.source == "ai"
    members[1].bad = True
    info = sim.step()
    assert info.source == "solver"
    assert not info.forced  # A doubt policy's own trigger, not a rescue.
    assert math.isinf(info.doubt)
    assert info.frame == 23
    assert all(frame.isfinite().all() for frame in info.frames)
    for got, want in zip(info.frames, solver_frames(first.state, [fine_mask()] * 2, 2), strict=True):
        assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize(
    "policy",
    [
        FixedSchedule(period=1000, offset=500),
        RandomBudget(rate=0.0, seed=1),
        Oracle(tau=100.0),
    ],
    ids=["fixed", "random", "oracle"],
)
def test_non_finite_output_forces_a_handoff_under_every_policy(normalizer, policy):
    members = [Member(0.0), Member(0.0)]
    sim, members, _ = make(normalizer, policy, members=members, error_of=lambda *_: 0.0)
    assert sim.step().source == "ai"
    for member in members:
        member.bad = True
    info = sim.step()
    assert info.source == "solver"
    assert info.forced
    assert all(frame.isfinite().all() for frame in info.frames)


def test_scheduled_handoff_on_a_non_finite_output_is_not_counted_as_forced(normalizer):
    members = [Member(0.0)]
    members[0].bad = True
    sim, _, _ = make(normalizer, FixedSchedule(period=5, offset=0, start=21), members=members)
    info = sim.step()
    assert info.source == "solver"
    assert not info.forced


@pytest.mark.parametrize("policy", [DoubtThreshold(math.inf), AlwaysAI()], ids=["never", "always_ai"])
def test_never_candidate_does_not_hand_off(normalizer, policy):
    members = [Member(0.0), Member(0.0)]
    sim, members, solver = make(normalizer, policy, members=members)
    for member in members:
        member.bad = True
    for _ in range(3):
        info = sim.step()
        assert info.source == "ai"
        assert not info.forced
        assert not math.isfinite(info.doubt)
    assert solver.steps == 0


def test_oracle_triggers_above_its_tau_only():
    policy = Oracle(tau=0.5)
    assert not policy.decide(21, 0.0, {"output_finite": True, "error": 0.5})
    assert policy.decide(21, 0.0, {"output_finite": True, "error": 0.51})
    assert policy.decide(21, 0.0, {"output_finite": True, "error": math.nan})  # Counts as 10.


def test_oracle_never_reads_future(normalizer):
    reads = []

    def error_of(frame, fields, solid):
        reads.append(frame)
        return 1.0 if frame == 23 else 0.0

    sim, _, _ = make(normalizer, Oracle(tau=0.5), handoff_frames=3, error_of=error_of)
    steps = [sim.step() for _ in range(4)]
    assert [(s.frame, s.source) for s in steps] == [
        (21, "ai"),
        (22, "ai"),
        (25, "solver"),
        (26, "ai"),
    ]
    # It reads only the frame being decided. Frames the solver made aren't read at all.
    assert reads == [21, 22, 23, 26]


def test_oracle_reads_the_ensemble_output_for_that_frame(normalizer):
    seen = []
    members = [Member(0.1), Member(0.3)]

    def error_of(frame, fields, solid):
        seen.append((fields.clone(), solid.clone()))
        return 0.0

    sim, _, _ = make(normalizer, Oracle(tau=0.5), members=members, error_of=error_of)
    info = sim.step()
    assert torch.equal(seen[0][0], info.state)
    assert seen[0][1].shape == (1, 1, COARSE, COARSE)


def test_obstacle_change_applies_next_frame(normalizer):
    sim, members, solver = make(normalizer, AlwaysAI())
    sim.step()
    assert members[0].inputs[-1][0, 3, 0, 0] == 0
    new = torch.zeros(1, 1, FINE, FINE, dtype=torch.bool)
    new[..., :4, :4] = True
    sim.set_obstacle(new)
    info = sim.step()
    assert members[0].inputs[-1][0, 3, 0, 0] == 1
    assert (members[0].inputs[-1][0, 3, 6, 6] == 0)  # The old solid has gone.
    assert (info.state[..., :2, :2] == 0).all()
    assert solver.steps == 0

    # A mask on the neural grid does too, and reaches the solver at the reference grid.
    coarse = torch.zeros(1, 1, COARSE, COARSE, dtype=torch.bool)
    coarse[..., 8:10, 8:10] = True
    sim.set_obstacle(coarse)
    sim.policy = AlwaysSolver()
    sim.step()
    assert solver.masks[0][..., 16:20, 16:20].all()
    assert solver.masks[0].sum() == 16


def test_obstacle_change_during_a_handoff_applies_after_it(normalizer):
    new = torch.zeros(1, 1, FINE, FINE, dtype=torch.bool)
    new[..., :4, :4] = True
    holder = {}

    def edit(steps):
        if steps == 2:
            holder["sim"].set_obstacle(new)

    solver = CountingSolver(hook=edit)
    sim, members, _ = make(normalizer, AlwaysSolver(), solver=solver, handoff_frames=2)
    holder["sim"] = sim
    sim.step()
    assert all(torch.equal(mask, fine_mask()) for mask in solver.masks)
    sim.policy = AlwaysAI()
    sim.step()
    assert members[0].inputs[-1][0, 3, 0, 0] == 1


def test_time_accounting_includes_every_member(normalizer):
    wait = 0.01
    one = make(normalizer, AlwaysAI(), members=[Member(0.0, wait)])[0].step()
    three = make(normalizer, AlwaysAI(), members=[Member(0.0, wait) for _ in range(3)])[0].step()
    assert one.ms >= 1000 * wait
    assert three.ms >= 3000 * wait
    # A hand-off's time also holds the ensemble step it threw away, and the solver's.
    handoff = make(normalizer, AlwaysSolver(), members=[Member(0.0, wait) for _ in range(3)])[0]
    assert handoff.step().ms >= 3000 * wait


def test_one_scene_only(normalizer):
    sim = HybridSimulator(Ensemble([Member(0.0)], normalizer), CountingSolver(), AlwaysAI(), 2, SUBSTEPS)
    with pytest.raises(ValueError, match="one scene"):
        sim.reset(torch.zeros(2, 3, COARSE, COARSE), fine_mask(), fine_inflow(), 2.0)
