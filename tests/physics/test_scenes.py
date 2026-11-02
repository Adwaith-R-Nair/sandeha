"""Checks on the scene sampler: which shapes and physics each split may contain.

The domain is the unit square. Rows grow upward, so "up" is +row.
"""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sandeha.physics import scenes
from sandeha.physics.scenes import (
    MIN_THICKNESS,
    SPLITS,
    TEST_C_BUOYANCY,
    TEST_C_INFLOW,
    TRAIN_BUOYANCY,
    TRAIN_INFLOW,
    TRAIN_SHAPES,
    UNSEEN_SHAPES,
    sample_scene,
)
from sandeha.physics.solver import SmokeSolver, SmokeState

COUNT = 150
LATE_SPLITS = ("train", "val", "test_a")
CORNER_ALLOWANCE = 3.5


def sample_many(split, count=COUNT, res=64, seed=0):
    rng = np.random.default_rng(seed)
    return [sample_scene(rng, res, split) for _ in range(count)]


def solid_ever(scene):
    if scene.late_obstacle is None:
        return scene.obstacle
    return scene.obstacle | scene.late_obstacle


def families_drawn(monkeypatch, split, count=COUNT):
    """Every family the sampler really draws, including attempts it goes on to reject."""
    drawn = []

    def spy(name, builder):
        def wrapped(*args, **kwargs):
            drawn.append(name)
            return builder(*args, **kwargs)

        return wrapped

    spies = {name: spy(name, builder) for name, builder in scenes._BUILDERS.items()}
    monkeypatch.setattr(scenes, "_BUILDERS", spies)
    sample_many(split, count)
    return set(drawn)


def within(value, bounds):
    return bounds[0] <= value <= bounds[1]


def disc_kernel(radius):
    r = int(np.ceil(radius))
    rows, columns = np.mgrid[-r : r + 1, -r : r + 1]
    kernel = torch.from_numpy((rows**2 + columns**2 <= radius**2).astype(np.float32))
    return kernel[None, None]


def thin_cells(mask, res):
    """Solid cells far from anywhere a disc of the minimum thickness fits.

    A disc can't reach into a square corner, and is itself pixelated, so cells within
    ``CORNER_ALLOWANCE`` of a place it fits are not thin parts. On real shapes these leftovers
    reach at most 3 cells, which is short against the minimum thickness of about 10 cells.
    """
    kernel = disc_kernel(0.5 * MIN_THICKNESS * res - 0.5)  # Less rasterisation error.
    pad = kernel.shape[-1] // 2
    solid = torch.from_numpy(mask.astype(np.float32))[None, None]
    eroded = F.conv2d(F.pad(1 - solid, (pad,) * 4, value=1.0), kernel) == 0
    fits = F.conv2d(eroded.float(), kernel, padding=pad) > 0
    grow = disc_kernel(CORNER_ALLOWANCE)
    near_fit = F.conv2d(fits.float(), grow, padding=grow.shape[-1] // 2) > 0
    return mask & ~near_fit[0, 0].numpy()


# Shape families


@pytest.mark.parametrize("split", ["train", "val", "test_a", "test_c"])
def test_train_never_uses_unseen_shapes(monkeypatch, split):
    assert families_drawn(monkeypatch, split) == set(TRAIN_SHAPES)


def test_test_b_only_uses_unseen_shapes(monkeypatch):
    assert families_drawn(monkeypatch, "test_b") == set(UNSEEN_SHAPES)


def test_families_do_not_overlap():
    assert not set(TRAIN_SHAPES) & set(UNSEEN_SHAPES)
    assert set(scenes._BUILDERS) == set(TRAIN_SHAPES) | set(UNSEEN_SHAPES)


def disc_overlap(mask, x, y):
    """Intersection over union of a mask with the disc of equal area at its centroid."""
    radius2 = mask.sum() * (x[0, 1] - x[0, 0]) ** 2 / np.pi
    disc = (x - x[mask].mean()) ** 2 + (y - y[mask].mean()) ** 2 < radius2
    return (mask & disc).sum() / (mask | disc).sum()


def test_families_really_are_different_shapes():
    """A sampler could call the star builder and still draw a circle."""
    x, y = scenes._grid(256)
    x, y = x - 0.5, y - 0.5
    rng = np.random.default_rng(0)
    overlap = {
        name: [disc_overlap(scenes._BUILDERS[name](rng, x, y), x, y) for _ in range(40)]
        for name in TRAIN_SHAPES + UNSEEN_SHAPES
    }
    assert min(overlap["circle"]) > 0.97
    assert max(overlap["star"]) < 0.9
    assert max(overlap["letter"]) < 0.7
    assert max(overlap["wall"]) < 0.7
    assert np.mean(overlap["blob"]) < 0.93
    assert max(overlap["rectangle"]) < 0.95


def test_unseen_families_are_not_simply_larger():
    """Test B must differ in form. If its shapes were also bigger, doubt could rise for that."""
    x, y = scenes._grid(128)
    x, y = x - 0.5, y - 0.5
    rng = np.random.default_rng(0)

    def mean_area(name):
        return np.mean([scenes._BUILDERS[name](rng, x, y).mean() for _ in range(200)])

    areas = {name: mean_area(name) for name in TRAIN_SHAPES + UNSEEN_SHAPES}
    seen = np.mean([areas[name] for name in TRAIN_SHAPES])
    unseen = np.mean([areas[name] for name in UNSEEN_SHAPES])
    assert abs(unseen / seen - 1) < 0.1
    assert all(0.65 * seen < areas[name] < 1.35 * seen for name in areas)


# Physics ranges


@pytest.mark.parametrize("split", ["train", "val", "test_a", "test_b"])
def test_other_splits_stay_inside_train_ranges(split):
    for scene in sample_many(split):
        assert within(scene.buoyancy, TRAIN_BUOYANCY)
        assert within(float(scene.inflow.max()), TRAIN_INFLOW)


def test_test_c_outside_train_ranges():
    for scene in sample_many("test_c"):
        strength = float(scene.inflow.max())
        assert not within(scene.buoyancy, TRAIN_BUOYANCY)
        assert not within(strength, TRAIN_INFLOW)
        assert any(within(scene.buoyancy, side) for side in TEST_C_BUOYANCY)
        assert any(within(strength, side) for side in TEST_C_INFLOW)


def test_test_c_covers_both_sides_of_both_ranges_in_every_pairing():
    pairs = set()
    for scene in sample_many("test_c"):
        pairs.add((scene.buoyancy > TRAIN_BUOYANCY[1], float(scene.inflow.max()) > TRAIN_INFLOW[1]))
    assert len(pairs) == 4


@pytest.mark.parametrize("split", ["train", "test_c"])
def test_train_range_is_covered(split):
    """A sampler stuck near one value would pass the range checks above."""
    values = [s.buoyancy for s in sample_many(split)]
    assert max(values) - min(values) > 0.8 * (TRAIN_BUOYANCY[1] - TRAIN_BUOYANCY[0])


# Inflow


@pytest.mark.parametrize("split", SPLITS)
def test_inflow_not_blocked(split):
    for scene in sample_many(split):
        assert (scene.inflow > 0).any()
        assert not (solid_ever(scene) & (scene.inflow > 0)).any()


@pytest.mark.parametrize("split", SPLITS)
def test_obstacles_keep_off_sources_even_when_placed_right_on_them(monkeypatch, split):
    """Normal placement rarely reaches a source, so this forces the attempt.

    The way-up check alone would reject a covered source cell. This also asserts the gap kept
    around each source, which is at least one cell at this grid size.
    """
    monkeypatch.setattr(scenes, "_OBSTACLE_Y", (0.1, 0.4))
    for scene in sample_many(split, count=40):
        near = scene.inflow > 0
        near[1:] |= scene.inflow[:-1] > 0
        near[:-1] |= scene.inflow[1:] > 0
        near[:, 1:] |= scene.inflow[:, :-1] > 0
        near[:, :-1] |= scene.inflow[:, 1:] > 0
        assert not (solid_ever(scene) & near).any()


@pytest.mark.parametrize("split", SPLITS)
def test_inflow_edges_are_smooth(split):
    """Hard-edged sources lose grid convergence early: see DECISIONS.md B12."""
    for scene in sample_many(split, count=40, res=256):
        step = max(np.abs(np.diff(scene.inflow, axis=axis)).max() for axis in (0, 1))
        assert step < 0.15 * scene.inflow.max()


@pytest.mark.parametrize("split", SPLITS)
def test_inflow_sits_below_obstacles(split):
    """Smoke rises along +row, so a flipped axis would put the source on top of the shapes."""
    rows = None
    for scene in sample_many(split):
        rows = np.arange(scene.inflow.shape[0])[:, None]
        inflow_row = (rows * scene.inflow).sum() / scene.inflow.sum()
        assert inflow_row < 0.3 * len(rows)
        assert (rows * scene.obstacle).sum() / scene.obstacle.sum() > inflow_row


# Obstacles


@pytest.mark.parametrize("split", SPLITS)
def test_most_scenes_put_an_obstacle_in_a_plume_path(split):
    """A plume that misses every shape makes a scene easy, whichever family the shapes are."""
    hits = []
    for scene in sample_many(split):
        columns = (scene.inflow > 0).any(axis=0)
        hits.append(scene.obstacle[:, columns].any())
    assert np.mean(hits) > 0.85


@pytest.mark.parametrize("split", SPLITS)
def test_obstacles_are_thick_enough_not_to_be_jumped(split):
    """Fast smoke can jump a thin wall, so no solid part may be thinner than the limit."""
    res = 256
    for scene in sample_many(split, count=40, res=res):
        assert not thin_cells(solid_ever(scene), res).any()


@pytest.mark.parametrize("rows, thin", [(4, True), (8, True), (11, False)])
def test_thickness_check_tells_thin_walls_from_thick_ones(rows, thin):
    res = 256
    mask = np.zeros((res, res), dtype=bool)
    mask[100 : 100 + rows, 50:200] = True
    assert thin_cells(mask, res).any() == thin


def test_thickness_check_detects_a_thin_spike_on_a_thick_shape():
    res = 256
    mask = np.zeros((res, res), dtype=bool)
    mask[100:140, 100:140] = True
    mask[140:150, 118:121] = True
    assert thin_cells(mask, res).any()


@pytest.mark.parametrize("split", SPLITS)
def test_obstacles_stay_clear_of_the_box_walls(split):
    for scene in sample_many(split):
        solid = solid_ever(scene)
        assert not (solid[0].any() or solid[-1].any() or solid[:, 0].any() or solid[:, -1].any())


def reaches_top(fluid, start):
    reached = start & fluid
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= fluid
        if (grown == reached).all():
            return reached[-1].any(), reached
        reached = grown


@pytest.mark.parametrize("split", SPLITS)
def test_smoke_always_has_a_way_up(split):
    for scene in sample_many(split):
        solid = solid_ever(scene)
        top_reached, reached = reaches_top(~solid, scene.inflow > 0)
        assert top_reached
        assert reached[scene.inflow > 0].all()


def test_way_up_check_detects_a_sealed_source():
    fluid = np.ones((64, 64), dtype=bool)
    fluid[30:32] = False
    start = np.zeros((64, 64), dtype=bool)
    start[5, 5] = True
    assert not reaches_top(fluid, start)[0]


def test_sampler_rejects_a_sealed_scene():
    fluid = np.ones((64, 64), dtype=bool)
    fluid[30:32] = False
    inflow = np.zeros((64, 64))
    inflow[5, 5] = 1.0
    assert not scenes._has_way_up(~fluid, inflow)
    fluid[30:32, 40:44] = True
    assert scenes._has_way_up(~fluid, inflow)


def test_way_up_requires_every_source_cell_to_get_out():
    solid = np.zeros((64, 64), dtype=bool)
    solid[8:13, 8:13] = True
    solid[10, 10] = False  # A sealed pocket, holding one cell of the source.
    inflow = np.zeros((64, 64))
    inflow[5, 5] = 1.0
    assert scenes._has_way_up(solid, inflow)
    inflow[10, 10] = 1.0
    assert not scenes._has_way_up(solid, inflow)


def test_sealed_draws_are_thrown_away(monkeypatch):
    real = scenes._has_way_up
    answers = iter([False, False, False])
    calls = []

    def sealed_at_first(solid, inflow):
        calls.append(solid)
        return next(answers, real(solid, inflow))

    monkeypatch.setattr(scenes, "_has_way_up", sealed_at_first)
    scene = sample_scene(np.random.default_rng(0), 64, "train")
    assert len(calls) == 4
    assert np.array_equal(calls[-1], solid_ever(scene))


def test_sampler_gives_up_rather_than_return_a_sealed_scene(monkeypatch):
    monkeypatch.setattr(scenes, "_has_way_up", lambda solid, inflow: False)
    with pytest.raises(RuntimeError, match="open scene"):
        sample_scene(np.random.default_rng(0), 64, "train")


@pytest.mark.parametrize("split", LATE_SPLITS)
def test_way_up_is_judged_with_the_late_obstacle_in_place(monkeypatch, split):
    real = scenes._has_way_up
    seen = []

    def spy(solid, inflow):
        seen.append(solid)
        return real(solid, inflow)

    monkeypatch.setattr(scenes, "_has_way_up", spy)
    rng = np.random.default_rng(0)
    for _ in range(30):
        scene = sample_scene(rng, 64, split)
        assert np.array_equal(seen[-1], solid_ever(scene))


@pytest.mark.parametrize("split", SPLITS)
def test_most_of_the_box_stays_fluid(split):
    fractions = [solid_ever(s).mean() for s in sample_many(split)]
    assert 0 < min(fractions)
    assert max(fractions) < 0.4


# Late obstacles


@pytest.mark.parametrize("split", LATE_SPLITS)
def test_a_third_have_a_late_obstacle(split):
    count = 1000
    share = np.mean([s.late_obstacle is not None for s in sample_many(split, count=count)])
    # Four standard deviations of the binomial, so that a change of seed can't fail it by chance.
    assert abs(share - 1 / 3) < 4 * np.sqrt(1 / 3 * 2 / 3 / count)


@pytest.mark.parametrize("split", ["test_b", "test_c"])
def test_no_late_obstacle_in_the_unfamiliar_splits(split):
    for scene in sample_many(split):
        assert scene.late_obstacle is None and scene.late_frame is None


@pytest.mark.parametrize("split", LATE_SPLITS)
def test_late_obstacle_is_valid_when_present(split):
    frames = []
    for scene in sample_many(split):
        assert (scene.late_obstacle is None) == (scene.late_frame is None)
        if scene.late_obstacle is not None:
            assert scene.late_obstacle.shape == scene.obstacle.shape
            assert scene.late_obstacle.dtype == bool
            new_cells = (scene.late_obstacle & ~scene.obstacle).sum()
            assert new_cells >= 0.5 * scene.late_obstacle.sum()
            assert isinstance(scene.late_frame, int)
            assert scenes.LATE_FRAMES[0] <= scene.late_frame <= scenes.LATE_FRAMES[1]
            frames.append(scene.late_frame)
    assert len(set(frames)) > 5


# Form


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("res", [64, 128, 256])
def test_arrays_have_the_promised_form(split, res):
    scene = sample_scene(np.random.default_rng(1), res, split)
    assert scene.obstacle.shape == (res, res) and scene.obstacle.dtype == bool
    assert scene.inflow.shape == (res, res) and scene.inflow.dtype == np.float32
    assert (scene.inflow >= 0).all()
    assert type(scene.buoyancy) is float


@pytest.mark.parametrize("split", ["", "test", "TRAIN", "test_d"])
def test_unknown_split_is_refused(split):
    with pytest.raises(ValueError, match="split"):
        sample_scene(np.random.default_rng(0), 64, split)


# Seeding


def scene_digest(scene):
    digest = hashlib.sha256()
    digest.update(scene.obstacle.tobytes())
    digest.update(scene.inflow.tobytes())
    digest.update(repr((scene.buoyancy, scene.late_frame)).encode())
    if scene.late_obstacle is not None:
        digest.update(scene.late_obstacle.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("split", SPLITS)
def test_sampling_is_seeded(split):
    first = [scene_digest(s) for s in sample_many(split, count=20, seed=7)]
    second = [scene_digest(s) for s in sample_many(split, count=20, seed=7)]
    other = [scene_digest(s) for s in sample_many(split, count=20, seed=8)]
    assert first == second
    assert len(set(first)) == 20
    assert not set(first) & set(other)


def test_sampling_is_the_same_in_a_new_process():
    code = (
        "import numpy as np, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from test_scenes import sample_many, scene_digest\n"
        "print([scene_digest(s) for s in sample_many('test_b', count=5)])\n"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code, os.path.dirname(__file__)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.add(result.stdout)
    assert outputs == {str([scene_digest(s) for s in sample_many("test_b", count=5)]) + "\n"}


# With the solver


@pytest.mark.parametrize("split", SPLITS)
def test_scenes_run_through_the_solver(split):
    """Batched like generation, using a late obstacle when a scene has one."""
    device = torch.device("cuda")
    res = 64
    batch = sample_many(split, count=4, res=res, seed=3)
    obstacle = torch.from_numpy(np.stack([s.obstacle for s in batch]))[:, None].to(device)
    ever = torch.from_numpy(np.stack([solid_ever(s) for s in batch]))[:, None].to(device)
    inflow = torch.from_numpy(np.stack([s.inflow for s in batch]))[:, None].to(device)
    buoyancy = torch.tensor([s.buoyancy for s in batch], dtype=torch.float32, device=device)
    state = SmokeState(
        velocity=torch.zeros(len(batch), 2, res, res, device=device),
        density=torch.zeros(len(batch), 1, res, res, device=device),
    )
    solver = SmokeSolver(res, device, dt=0.01)
    for step in range(30):
        state = solver.step(state, ever if step >= 15 else obstacle, inflow, buoyancy)
        assert solver.last_solve.converged.all()
    assert state.density.isfinite().all() and state.velocity.isfinite().all()
    assert state.density[ever].abs().max() == 0
    assert (state.density.sum(dim=(1, 2, 3)) > 0).all()
