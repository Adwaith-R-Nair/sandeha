"""Checks the reference solver must pass before any data is generated.

The domain is the unit square. Rows grow upward, so "up" is +row and +v.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sandeha.physics.solver import DENSITY_DECAY, SmokeSolver, SmokeState

DEVICE = torch.device("cuda")
DT = 0.01


def coords(res):
    centres = (torch.arange(res, device=DEVICE, dtype=torch.float32) + 0.5) / res
    y, x = torch.meshgrid(centres, centres, indexing="ij")
    return x, y


def disc(res, cx, cy, r):
    x, y = coords(res)
    return (((x - cx) ** 2 + (y - cy) ** 2) < r**2)[None, None]


def box(res, x0, x1, y0, y1):
    x, y = coords(res)
    return ((x > x0) & (x < x1) & (y > y0) & (y < y1))[None, None]


def still(res, batch=1):
    return SmokeState(
        velocity=torch.zeros(batch, 2, res, res, device=DEVICE),
        density=torch.zeros(batch, 1, res, res, device=DEVICE),
    )


def buoyancy(value, batch=1):
    return torch.full((batch,), float(value), device=DEVICE)


def run(solver, state, obstacle, inflow, lift, steps):
    for _ in range(steps):
        state = solver.step(state, obstacle, inflow, lift)
    return state


def low_plume(res):
    """A low inflow and no obstacles."""
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    inflow = 5.0 * disc(res, 0.5, 0.15, 0.08).float()
    return obstacle, inflow


def test_divergence_known_answer():
    """u = x gives divergence 1 away from the walls, in physical units."""
    res = 32
    x, _ = coords(res)
    velocity = torch.stack([x, torch.zeros_like(x)])[None]
    state = SmokeState(velocity=velocity, density=torch.zeros(1, 1, res, res, device=DEVICE))
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)

    div = SmokeSolver(res, DEVICE, DT).divergence(state, obstacle)

    assert div.shape == (1, 1, res, res)
    assert div[..., 2:-2, 2:-2].cpu().numpy() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("with_obstacle", [False, True])
def test_projection_removes_divergence(with_obstacle):
    res = 64
    torch.manual_seed(0)
    state = still(res, batch=2)
    state.velocity = 2 * torch.rand(2, 2, res, res, device=DEVICE) - 1
    obstacle = torch.zeros(2, 1, res, res, dtype=torch.bool, device=DEVICE)
    if with_obstacle:
        obstacle = obstacle | disc(res, 0.5, 0.5, 0.2)
    solver = SmokeSolver(res, DEVICE, DT)

    before = solver.divergence(state, obstacle)
    after_state = solver.project(state, obstacle)
    after = solver.divergence(after_state, obstacle)

    assert before[~obstacle].abs().max() > 1.0  # The check starts from a truly divergent field.
    assert after[~obstacle].abs().max() < 1e-3
    assert after_state.velocity.std() > 0.05  # Projection did not just erase the flow.
    assert solver.last_solve.converged.all()


def test_no_flow_into_obstacles():
    res = 64
    obstacle = disc(res, 0.5, 0.5, 0.15)
    # The inflow is partly drawn over the obstacle, as a user can do in the demo.
    inflow = 5.0 * (disc(res, 0.5, 0.25, 0.1) | disc(res, 0.5, 0.45, 0.1)).float()
    state = still(res)
    state.velocity = torch.ones(1, 2, res, res, device=DEVICE) * 0.2
    solver = SmokeSolver(res, DEVICE, DT)

    state = run(solver, state, obstacle, inflow, buoyancy(2.0), 50)

    assert state.density[~obstacle].max() > 0  # There is smoke to keep out.
    assert state.velocity.abs().amax(dim=1, keepdim=True)[~obstacle].max() > 0
    assert state.velocity[obstacle.expand(-1, 2, -1, -1)].abs().max() == 0
    assert state.density[obstacle].abs().max() == 0
    assert state.density.min() >= 0


def test_deterministic():
    res = 64
    obstacle = disc(res, 0.5, 0.5, 0.12)
    _, inflow = low_plume(res)

    first = run(SmokeSolver(res, DEVICE, DT), still(res), obstacle, inflow, buoyancy(2.0), 30)
    second = run(SmokeSolver(res, DEVICE, DT), still(res), obstacle, inflow, buoyancy(2.0), 30)

    assert torch.equal(first.velocity, second.velocity)
    assert torch.equal(first.density, second.density)


def test_buoyancy_lifts_smoke():
    res = 64
    obstacle, inflow = low_plume(res)
    solver = SmokeSolver(res, DEVICE, DT)
    rows = (torch.arange(res, device=DEVICE, dtype=torch.float32) + 0.5) / res

    def centroid(state):
        return float((state.density.sum(dim=-1).squeeze() * rows).sum() / state.density.sum())

    state = run(solver, still(res), obstacle, inflow, buoyancy(2.0), 10)
    early = centroid(state)
    state = run(solver, state, obstacle, inflow, buoyancy(2.0), 90)
    late = centroid(state)

    assert late > early + 0.05

    # No buoyancy means no motion, so the centroid stays at the inflow.
    calm = run(SmokeSolver(res, DEVICE, DT), still(res), obstacle, inflow, buoyancy(0.0), 100)
    assert centroid(calm) == pytest.approx(0.15, abs=1e-2)


def test_steady_vortex_stays_centred():
    """A vortex is a steady solution, so it must not drift. Faces on the wrong lattice do."""
    res = 64
    x, y = coords(res)
    swirl = 10 * torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.1**2)
    start = SmokeState(
        velocity=torch.stack([-(y - 0.5) * swirl, (x - 0.5) * swirl])[None],
        density=torch.zeros(1, 1, res, res, device=DEVICE),
    )
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)
    no_inflow = torch.zeros(1, 1, res, res, device=DEVICE)

    state = run(solver, solver.project(start, obstacle), obstacle, no_inflow, buoyancy(0.0), 100)

    energy = state.velocity[0].pow(2).sum(dim=0)
    assert energy.max() > 0.01  # The vortex is still there to measure.
    centre_x = float((energy * x).sum() / energy.sum())
    centre_y = float((energy * y).sum() / energy.sum())
    assert centre_x == pytest.approx(0.5, abs=1e-3)
    assert centre_y == pytest.approx(0.5, abs=1e-3)


def lamb_oseen(res, cx, cy, circulation, core):
    x, y = coords(res)
    dx, dy = x - cx, y - cy
    r2 = dx**2 + dy**2 + 1e-12
    swirl = circulation / (2 * math.pi * r2) * (1 - torch.exp(-r2 / core**2))
    return torch.stack([-dy * swirl, dx * swirl])


def positive_vortex_centre(velocity):
    """Centroid of the part of the anticlockwise vorticity above half its peak."""
    res = velocity.shape[-1]
    x, y = coords(res)
    u, v = velocity[0]
    vorticity = torch.zeros_like(u)
    vorticity[1:-1, 1:-1] = (
        (v[1:-1, 2:] - v[1:-1, :-2]) - (u[2:, 1:-1] - u[:-2, 1:-1])
    ) * res / 2
    core = torch.where(vorticity > 0.5 * vorticity.max(), vorticity, torch.zeros_like(vorticity))
    return float((core * x).sum() / core.sum()), float((core * y).sum() / core.sum())


@pytest.mark.parametrize("axis", ["up", "right"])
def test_vortex_pair_carries_itself(axis):
    """Velocity must carry itself. Each vortex moves its partner, so the pair travels.

    Point vortices give speed circulation / (2 pi separation). The bound is loose on the
    low side, because walls and numerical dissipation slow the pair. With either velocity
    component left out of the advection, the pair covers at most a quarter of the distance.
    """
    res = 64
    circulation, separation, core, steps = 0.5, 0.2, 0.04, 50
    half = separation / 2
    if axis == "up":
        velocity = lamb_oseen(res, 0.5 - half, 0.25, circulation, core)
        velocity = velocity + lamb_oseen(res, 0.5 + half, 0.25, -circulation, core)
    else:
        velocity = lamb_oseen(res, 0.25, 0.5 + half, circulation, core)
        velocity = velocity + lamb_oseen(res, 0.25, 0.5 - half, -circulation, core)
    solver = SmokeSolver(res, DEVICE, DT)
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    no_inflow = torch.zeros(1, 1, res, res, device=DEVICE)
    start = solver.project(
        SmokeState(velocity[None], torch.zeros(1, 1, res, res, device=DEVICE)), obstacle
    )

    end = run(solver, start, obstacle, no_inflow, buoyancy(0.0), steps)

    x0, y0 = positive_vortex_centre(start.velocity)
    x1, y1 = positive_vortex_centre(end.velocity)
    along, across = (y1 - y0, x1 - x0) if axis == "up" else (x1 - x0, y1 - y0)
    point_vortex_distance = circulation / (2 * math.pi * separation) * steps * DT
    assert 0.6 * point_vortex_distance < along < 1.1 * point_vortex_distance
    assert abs(across) < 0.01


def test_advection_follows_a_rotation_without_spiralling():
    """A first-order trace moves a rotating blob outward a little on every step."""
    res = 128
    omega = 2 * torch.pi
    solver = SmokeSolver(res, DEVICE, DT)
    centres = (torch.arange(res, device=DEVICE, dtype=torch.float32) + 0.5) / res
    edges = torch.arange(res + 1, device=DEVICE, dtype=torch.float32) / res
    y_u, _ = torch.meshgrid(centres, edges, indexing="ij")
    _, x_v = torch.meshgrid(edges, centres, indexing="ij")
    u = (-omega * (y_u - 0.5))[None, None]
    v = (omega * (x_v - 0.5))[None, None]
    x, y = coords(res)
    blob = torch.exp(-((x - 0.75) ** 2 + (y - 0.5) ** 2) / 0.05**2)[None, None]

    def radius(field):
        cx = (field[0, 0] * x).sum() / field.sum()
        cy = (field[0, 0] * y).sum() / field.sum()
        return float(((cx - 0.5) ** 2 + (cy - 0.5) ** 2).sqrt())

    field = blob
    for _ in range(100):  # One full turn.
        field = solver._advect(field, 0.5, 0.5, u, v)

    assert radius(field) == pytest.approx(radius(blob), rel=0.02)


def soft_inflow(res):
    x, y = coords(res)
    return 5.0 * torch.exp(-((x - 0.5) ** 2 + (y - 0.15) ** 2) / 0.08**2)[None, None]


def hard_inflow(res):
    return 5.0 * disc(res, 0.5, 0.15, 0.08).float()


def simulate_to_fixed_time(res, inflow, steps):
    obstacle = disc(res, 0.5, 0.55, 0.1)
    solver = SmokeSolver(res, DEVICE, DT)
    return run(solver, still(res), obstacle, inflow(res), buoyancy(2.0), steps).density


def relative_error(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize(
    "inflow, steps",
    [
        # A smooth source is the same problem on every grid, so it can run further.
        (soft_inflow, 100),
        # A hard-edged source differs by grid, and stops converging pointwise as the
        # plume head becomes chaotic. results/solver_check.json records how that goes.
        (hard_inflow, 50),
    ],
    ids=["soft_inflow_t1", "hard_inflow_t0.5"],
)
def test_converges_with_refinement(inflow, steps):
    fine = simulate_to_fixed_time(256, inflow, steps)
    d128 = simulate_to_fixed_time(128, inflow, steps)
    d64 = simulate_to_fixed_time(64, inflow, steps)

    error_128 = relative_error(d128, F.avg_pool2d(fine, 2))
    error_64 = relative_error(d64, F.avg_pool2d(fine, 4))
    assert error_128 < 0.8 * error_64

    # The same ordering holds when both are compared on one common grid.
    common = F.avg_pool2d(fine, 4)
    assert relative_error(F.avg_pool2d(d128, 2), common) < 0.8 * relative_error(d64, common)


def test_sealed_cavity_stays_finite():
    res = 64
    ring = disc(res, 0.5, 0.5, 0.3) & ~disc(res, 0.5, 0.5, 0.2)
    inside = disc(res, 0.5, 0.5, 0.2)
    torch.manual_seed(1)
    state = still(res)
    state.velocity = (2 * torch.rand(1, 2, res, res, device=DEVICE) - 1) * inside
    inflow = 5.0 * disc(res, 0.5, 0.45, 0.05).float()
    solver = SmokeSolver(res, DEVICE, DT)

    state = run(solver, state, ring, inflow, buoyancy(2.0), 20)

    assert torch.isfinite(state.velocity).all()
    assert torch.isfinite(state.density).all()
    assert solver.last_solve.converged.all()
    assert solver.divergence(state, ring)[~ring].abs().max() < 1e-3
    # Nothing crosses the ring.
    outside = ~disc(res, 0.5, 0.5, 0.3)
    assert state.density[outside].abs().max() == 0


def test_smoke_does_not_cross_a_thin_wall():
    res = 64
    wall = box(res, -1.0, 2.0, 0.5, 0.5 + 1.5 / res)  # Full width, one cell thick.
    assert wall.sum() == res
    _, inflow = low_plume(res)
    solver = SmokeSolver(res, DEVICE, DT)

    state = run(solver, still(res), wall, inflow, buoyancy(2.0), 100)

    below = box(res, -1.0, 2.0, -1.0, 0.5)
    above = ~(wall | below)
    assert state.density[below].max() > 0
    assert state.density[above].abs().max() == 0


def test_fully_solid_domain_stays_finite():
    res = 64
    obstacle = torch.ones(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    inflow = torch.ones(1, 1, res, res, device=DEVICE)
    state = still(res)
    state.velocity = torch.ones(1, 2, res, res, device=DEVICE)
    state.density = torch.ones(1, 1, res, res, device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)

    state = run(solver, state, obstacle, inflow, buoyancy(2.0), 20)

    assert state.velocity.abs().max() == 0
    assert state.density.abs().max() == 0
    assert solver.last_solve.converged.all()


def test_obstacle_change_mid_run_is_respected():
    """A late obstacle applies from its first step, with a new pressure setup."""
    res = 64
    obstacle, inflow = low_plume(res)
    solver = SmokeSolver(res, DEVICE, DT)
    state = run(solver, still(res), obstacle, inflow, buoyancy(2.0), 40)

    late = box(res, 0.3, 0.7, 0.1, 0.3)
    assert state.density[late].max() > 0.1  # There is smoke for the new solid to clear.
    state = solver.step(state, late, inflow, buoyancy(2.0))

    assert state.velocity[late.expand(-1, 2, -1, -1)].abs().max() == 0
    assert state.density[late].abs().max() == 0
    assert solver.divergence(state, late)[~late].abs().max() < 1e-2
    assert solver.last_solve.converged.all()


def test_new_wall_stops_smoke_on_its_first_step():
    """Flow from before the change must not carry smoke through the new wall."""
    res = 64
    wall = box(res, -1.0, 2.0, 0.5, 0.5 + 1.5 / res)  # Full width, one cell thick.
    assert wall.sum() == res
    below = box(res, -1.0, 2.0, -1.0, 0.5)
    above = ~(wall | below)
    u = torch.zeros(1, 1, res, res + 1, device=DEVICE)
    v = torch.full((1, 1, res + 1, res), 3.0, device=DEVICE)  # Two cells a step, straight up.
    v[..., 0, :] = 0
    v[..., -1, :] = 0
    state = SmokeState(
        velocity=SmokeSolver._centre(u, v),
        density=below.float(),
        mac=(u, v),
    )
    no_inflow = torch.zeros(1, 1, res, res, device=DEVICE)

    state = SmokeSolver(res, DEVICE, DT).step(state, wall, no_inflow, buoyancy(0.0))

    assert state.density[below].max() > 0
    assert state.density[above].abs().max() == 0


def test_batch_matches_single_runs():
    """Each batch item converges on its own and does not affect the others."""
    res = 64
    obstacles = torch.cat([disc(res, 0.5, 0.5, 0.15), box(res, 0.1, 0.4, 0.4, 0.6)])
    inflow = 5.0 * disc(res, 0.5, 0.15, 0.08).float().expand(2, -1, -1, -1)
    lift = torch.tensor([1.0, 3.0], device=DEVICE)

    solver = SmokeSolver(res, DEVICE, DT)
    batched = run(solver, still(res, 2), obstacles, inflow, lift, 20)
    batched_iterations = solver.last_solve.iterations.clone()

    for i in range(2):
        single_solver = SmokeSolver(res, DEVICE, DT)
        single = run(
            single_solver,
            still(res),
            obstacles[i : i + 1],
            inflow[i : i + 1],
            lift[i : i + 1],
            20,
        )
        assert torch.allclose(batched.density[i], single.density[0], atol=1e-4)
        assert torch.allclose(batched.velocity[i], single.velocity[0], atol=1e-4)
        assert batched_iterations[i] == single_solver.last_solve.iterations[0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_item_does_not_stall_or_poison_the_batch(bad):
    res = 64
    obstacle = torch.zeros(2, 1, res, res, dtype=torch.bool, device=DEVICE)
    state = still(res, 2)
    state.velocity[1] = 2 * torch.rand(2, res, res, device=DEVICE) - 1
    state.velocity[0, 0, 10, 10] = bad
    solver = SmokeSolver(res, DEVICE, DT)

    result = solver.project(state, obstacle)

    assert solver.last_solve.iterations[0] == 0
    assert not solver.last_solve.converged[0]
    assert solver.last_solve.converged[1]
    assert not torch.isfinite(result.velocity[0]).all()
    assert torch.isfinite(result.velocity[1]).all()
    assert solver.divergence(result, obstacle)[1].abs().max() < 1e-3


def test_infinite_face_is_not_reported_as_converged():
    """One infinite face makes the tolerance infinite too, which must not count as a pass."""
    res = 64
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    u = torch.zeros(1, 1, res, res + 1, device=DEVICE)
    v = torch.zeros(1, 1, res + 1, res, device=DEVICE)
    u[0, 0, 10, 10] = float("inf")
    state = SmokeState(SmokeSolver._centre(u, v), torch.zeros(1, 1, res, res, device=DEVICE), (u, v))
    solver = SmokeSolver(res, DEVICE, DT)

    solver.project(state, obstacle)

    assert not solver.last_solve.converged[0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("component", [0, 1], ids=["u", "v"])
def test_bad_velocity_in_a_moving_flow_never_heals(component, bad):
    """Task 9 counts a blown-up rollout as a failure, so a step must not make it look finite."""
    res = 64
    torch.manual_seed(5)
    solver = SmokeSolver(res, DEVICE, DT)
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    state = solver.project(
        SmokeState(
            2 * torch.rand(1, 2, res, res, device=DEVICE) - 1,
            torch.rand(1, 1, res, res, device=DEVICE),
        ),
        obstacle,
    )
    faces = [face.clone() for face in state.mac]
    faces[component][0, 0, 20, 20] = bad
    state = SmokeState(state.velocity, state.density, tuple(faces))
    no_inflow = torch.zeros(1, 1, res, res, device=DEVICE)

    for _ in range(10):
        state = solver.step(state, obstacle, no_inflow, buoyancy(0.0))

        assert not solver.last_solve.converged[0]
        assert not torch.isfinite(state.velocity).all()
        assert not torch.isfinite(state.density).all()


def plume_with_fault(fault, device):
    """Two items of the same plume start, with a fault put into item 0 only."""
    res = 64
    state = SmokeState(
        velocity=torch.zeros(2, 2, res, res, device=device),
        density=torch.zeros(2, 1, res, res, device=device),
    )
    where, bad = fault.split("_")
    value = float(bad)
    if where == "velocity":
        state.velocity[0, 0, 10, 10] = value
    elif where == "density":
        state.density[0, 0, 10, 10] = value
    else:
        u = torch.zeros(2, 1, res, res + 1, device=device)
        v = torch.zeros(2, 1, res + 1, res, device=device)
        u[0, 0, 10, 10] = value
        state.mac = (u, v)
    return state


def plume_inputs(res, batch, device):
    obstacle = torch.zeros(batch, 1, res, res, dtype=torch.bool, device=device)
    inflow = hard_inflow(res).to(device).expand(batch, -1, -1, -1)
    return obstacle, inflow, torch.full((batch,), 2.0, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "fault", ["velocity_nan", "velocity_inf", "faces_nan", "faces_inf", "density_nan", "density_inf"]
)
def test_non_finite_item_survives_step(fault, device):
    """A blown-up item stays visibly blown up, and the rest of the batch carries on."""
    res = 64
    state = plume_with_fault(fault, device)
    solver = SmokeSolver(res, device, DT)
    obstacle, inflow, lift = plume_inputs(res, 2, device)
    single_solver = SmokeSolver(res, device, DT)
    single = SmokeState(state.velocity[1:] * 0, state.density[1:] * 0)
    single_inputs = plume_inputs(res, 1, device)

    for _ in range(3):
        state = solver.step(state, obstacle, inflow, lift)
        single = single_solver.step(single, *single_inputs)

        assert solver.last_solve.converged.tolist() == [False, True]
        assert not torch.isfinite(state.velocity[0]).all()
        assert torch.allclose(state.velocity[1], single.velocity[0], atol=1e-4)
        assert torch.allclose(state.density[1], single.density[0], atol=1e-4)

    # The device still works for unrelated tensors.
    assert float((torch.zeros(3, device=device) + 1).sum()) == 3


def test_huge_item_survives_step():
    """Huge but finite values must not crash the solver or disturb the other item."""
    res = 64
    state = plume_with_fault("velocity_1e30", DEVICE)
    solver = SmokeSolver(res, DEVICE, DT)
    obstacle, inflow, lift = plume_inputs(res, 2, DEVICE)
    single_solver = SmokeSolver(res, DEVICE, DT)
    single = SmokeState(state.velocity[1:] * 0, state.density[1:] * 0)
    single_inputs = plume_inputs(res, 1, DEVICE)

    for _ in range(3):
        state = solver.step(state, obstacle, inflow, lift)
        single = single_solver.step(single, *single_inputs)

        assert solver.last_solve.converged[1]
        assert torch.allclose(state.velocity[1], single.velocity[0], atol=1e-4)


def test_iteration_limit_is_reported_as_not_converged():
    res = 64
    torch.manual_seed(2)
    state = still(res)
    state.velocity = 2 * torch.rand(1, 2, res, res, device=DEVICE) - 1
    obstacle = torch.zeros(1, 1, res, res, dtype=torch.bool, device=DEVICE)
    solver = SmokeSolver(res, DEVICE, DT, cg_max_iter=1)

    solver.project(state, obstacle)

    assert solver.last_solve.iterations[0] == 1
    assert not solver.last_solve.converged[0]
    assert solver.last_solve.residual[0] > 1e-3


@pytest.mark.parametrize("res", [64, 256])
def test_pressure_solve_needs_few_iterations(res):
    """The multigrid preconditioner keeps the count low as the grid grows."""
    torch.manual_seed(3)
    state = still(res)
    state.velocity = 2 * torch.rand(1, 2, res, res, device=DEVICE) - 1
    solver = SmokeSolver(res, DEVICE, DT)

    solver.project(state, disc(res, 0.5, 0.5, 0.2))

    assert solver.last_solve.converged.all()
    assert solver.last_solve.iterations[0] < 60


def test_preconditioner_is_symmetric():
    """Conjugate gradients is only valid with a symmetric preconditioner."""
    res = 64
    solver = SmokeSolver(res, DEVICE, DT)
    obstacle = disc(res, 0.5, 0.5, 0.2)
    torch.manual_seed(4)
    a = torch.randn(1, 1, res, res, device=DEVICE) * ~obstacle
    b = torch.randn(1, 1, res, res, device=DEVICE) * ~obstacle
    a = a - a.sum() / (~obstacle).sum() * ~obstacle
    b = b - b.sum() / (~obstacle).sum() * ~obstacle

    apply = solver.preconditioner(obstacle)

    left = float((apply(a) * b).sum())
    right = float((a * apply(b)).sum())
    assert left == pytest.approx(right, rel=1e-3)
    assert float((apply(a) * a).sum()) > 0


@pytest.fixture(scope="module")
def plume_past_obstacle():
    """A fine plume at t = 1.75, split into thin sheets by an obstacle."""
    res = 256
    state = run(
        SmokeSolver(res, DEVICE, DT),
        still(res),
        disc(res, 0.5, 0.55, 0.1),
        soft_inflow(res),
        buoyancy(2.0),
        175,
    )
    return state.density[0, 0]


def test_thin_sheets_have_no_cell_scale_artefacts(plume_past_obstacle):
    """Stair-steps along sheets would become noise in the training targets."""
    d = plume_past_obstacle
    laplacian = d[1:-1, 2:] + d[1:-1, :-2] + d[2:, 1:-1] + d[:-2, 1:-1] - 4 * d[1:-1, 1:-1]
    assert float(laplacian.pow(2).mean().sqrt() / d.pow(2).mean().sqrt()) < 0.25


def test_smoke_matches_the_mass_budget(plume_past_obstacle):
    """Density added minus density decayed, against what is in the box."""
    res = 256
    rate = float(soft_inflow(res)[0, 0][~disc(res, 0.5, 0.55, 0.1)[0, 0]].sum()) / res**2
    steps = 175
    # Each step adds rate * dt, then applies decay to everything including what it just added.
    keep = float(torch.exp(torch.tensor(-DENSITY_DECAY * DT)))
    expected = rate * DT * sum(keep ** (i + 1) for i in range(steps))

    assert float(plume_past_obstacle.mean()) == pytest.approx(expected, rel=0.05)
