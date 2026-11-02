"""Random smoke scenes: where the obstacles and inflows go, and how strong the physics is.

Conventions match ``solver.py``: the domain is the unit square and rows grow upward, so
index ``(row, column)`` is ``(y, x)``. Lengths here are in domain units, so a scene looks the
same at any grid size, up to rasterisation.

What separates the splits:

- ``train``, ``val``, ``test_a``: simple shapes, physics inside the training ranges, and a
  third of scenes have an obstacle that appears mid-run.
- ``test_b``: unseen shape families, with physics inside the training ranges. Only the
  geometry is unfamiliar.
- ``test_c``: simple shapes, with buoyancy and inflow strength both outside the training
  ranges, on either side. Only the physics is unfamiliar.

``test_b`` and ``test_c`` have no late obstacles, so that each changes one thing.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TRAIN_SHAPES = ("circle", "rectangle", "wall")
UNSEEN_SHAPES = ("star", "letter", "blob")
SPLITS = ("train", "val", "test_a", "test_b", "test_c")

# Gaps between the training ranges and test_c, so that test_c is outside by a clear margin.
TRAIN_BUOYANCY = (1.0, 3.0)
TRAIN_INFLOW = (3.0, 7.0)  # Peak rate of a source.
TEST_C_BUOYANCY = ((0.4, 0.8), (3.5, 4.5))
TEST_C_INFLOW = ((1.5, 2.5), (8.0, 10.0))

# No solid part is thinner than this. Semi-Lagrangian advection can carry smoke across a wall
# thinner than one step's travel. results/scene_check.json measures that travel.
MIN_THICKNESS = 0.04

LATE_SHARE = 1 / 3
LATE_FRAMES = (20, 90)  # Inclusive. Frames are neural steps of the planned 121-frame scenes.

_OBSTACLE_COUNT = (1, 3)  # Inclusive.
_OBSTACLE_X = (0.15, 0.85)
_OBSTACLE_Y = (0.3, 0.85)  # Centres. Above the sources.
_IN_PATH_REACH = 0.08  # The first obstacle's centre is this close to a source, along x.
_SOURCE_COUNT = (1, 2)  # Inclusive.
_SOURCE_X = (0.12, 0.88)
_SOURCE_Y = (0.08, 0.16)
_SOURCE_RADIUS = (0.05, 0.08)
_CLEARANCE = 0.02  # Empty gap kept around sources, and between shapes and the box walls.
_POSE_ATTEMPTS = 200
_SCENE_ATTEMPTS = 200

# Stroke endpoints in a box of height 1 centred on the origin. Only letters with no closed
# counter, since an A or an O would seal off a pocket no smoke can enter.
_LETTERS = {
    "E": [(-.3, -.5, -.3, .5), (-.3, .5, .3, .5), (-.3, 0, .2, 0), (-.3, -.5, .3, -.5)],
    "F": [(-.3, -.5, -.3, .5), (-.3, .5, .3, .5), (-.3, 0, .2, 0)],
    "H": [(-.3, -.5, -.3, .5), (.3, -.5, .3, .5), (-.3, 0, .3, 0)],
    "K": [(-.3, -.5, -.3, .5), (-.3, 0, .3, .5), (-.3, 0, .3, -.5)],
    "L": [(-.3, -.5, -.3, .5), (-.3, -.5, .3, -.5)],
    "M": [(-.4, -.5, -.4, .5), (-.4, .5, 0, -.1), (0, -.1, .4, .5), (.4, .5, .4, -.5)],
    "N": [(-.3, -.5, -.3, .5), (-.3, .5, .3, -.5), (.3, -.5, .3, .5)],
    "T": [(-.35, .5, .35, .5), (0, .5, 0, -.5)],
    "V": [(-.35, .5, 0, -.5), (0, -.5, .35, .5)],
    "W": [(-.45, .5, -.2, -.5), (-.2, -.5, 0, .2), (0, .2, .2, -.5), (.2, -.5, .45, .5)],
    "X": [(-.33, -.5, .33, .5), (-.33, .5, .33, -.5)],
    "Y": [(-.33, .5, 0, 0), (.33, .5, 0, 0), (0, 0, 0, -.5)],
    "Z": [(-.3, .5, .3, .5), (.3, .5, -.3, -.5), (-.3, -.5, .3, -.5)],
}
_LETTER_NAMES = tuple(_LETTERS)


@dataclass
class Scene:
    """One scene. From ``late_frame`` on, the solid is ``obstacle | late_obstacle``."""

    obstacle: np.ndarray  # (res,res) bool at reference grid
    inflow: np.ndarray  # (res,res) float32 rate. Its peak is the scene's inflow strength
    buoyancy: float
    late_obstacle: np.ndarray | None  # obstacle added mid-run
    late_frame: int | None


def _grid(res: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres as ``(x, y)``, each ``(res,res)``, with x along columns and y along rows."""
    centres = (np.arange(res) + 0.5) / res
    x, y = np.meshgrid(centres, centres, indexing="xy")
    return x, y


def _turned(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in a frame turned by a random angle."""
    angle = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    return c * x + s * y, c * y - s * x


def _segment_distance(x: np.ndarray, y: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    dx, dy = x1 - x0, y1 - y0
    along = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(x - x0 - along * dx, y - y0 - along * dy)


def _rounded_polygon(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """A polygon grown by half the minimum thickness, so that no tip is thinner than the limit."""
    inside = np.zeros_like(x, dtype=bool)
    near = np.zeros_like(x, dtype=bool)
    for x0, y0, x1, y1 in zip(px, py, np.roll(px, -1), np.roll(py, -1)):
        if y0 != y1:
            inside ^= ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        near |= _segment_distance(x, y, x0, y0, x1, y1) < 0.5 * MIN_THICKNESS
    return inside | near


# Each builder takes coordinates relative to the shape's centre, and returns its mask.
# Sizes are set so that all the families cover similar areas: results/scene_check.json.


def _circle(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * x + y * y < rng.uniform(0.06, 0.14) ** 2


def _rectangle(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = _turned(rng, x, y)
    # Unequal sides, so that a rectangle is told from a square turned at random.
    half_long, half_short = rng.uniform(0.09, 0.15), rng.uniform(0.05, 0.08)
    return (np.abs(x) < half_long) & (np.abs(y) < half_short)


def _wall(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = _turned(rng, x, y)
    half_length, half_thickness = rng.uniform(0.15, 0.3), rng.uniform(0.5 * MIN_THICKNESS, 0.035)
    return (np.abs(x) < half_length) & (np.abs(y) < half_thickness)


def _star(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = int(rng.integers(5, 8))
    outer = rng.uniform(0.08, 0.14)
    inner = outer * rng.uniform(0.3, 0.45)
    angles = rng.uniform(0.0, 2.0 * np.pi) + np.arange(2 * points) * np.pi / points
    radii = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return _rounded_polygon(x, y, radii * np.cos(angles), radii * np.sin(angles))


def _letter(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    strokes = _LETTERS[_LETTER_NAMES[int(rng.integers(len(_LETTER_NAMES)))]]
    height = rng.uniform(0.2, 0.32)
    half_stroke = rng.uniform(0.5 * MIN_THICKNESS, 0.03)
    tilt = rng.uniform(-0.4, 0.4)
    c, s = np.cos(tilt), np.sin(tilt)
    x, y = c * x + s * y, c * y - s * x
    mask = np.zeros_like(x, dtype=bool)
    for x0, y0, x1, y1 in strokes:
        mask |= _segment_distance(x, y, x0 * height, y0 * height, x1 * height, y1 * height) < half_stroke
    return mask


def _blob(rng: np.random.Generator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth closed outline, of the kind a person draws: a radius with random harmonics."""
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    orders = np.arange(2, 6)
    amplitudes = rng.uniform(0.0, 1.0, orders.size) / orders
    amplitudes *= rng.uniform(0.3, 0.6) / amplitudes.sum()  # Keeps the radius positive.
    phases = rng.uniform(0.0, 2.0 * np.pi, orders.size)
    bumps = (amplitudes[:, None] * np.cos(orders[:, None] * angles + phases[:, None])).sum(axis=0)
    radii = rng.uniform(0.05, 0.1) * (1.0 + bumps)
    return _rounded_polygon(x, y, radii * np.cos(angles), radii * np.sin(angles))


_BUILDERS: dict[str, Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]] = {
    "circle": _circle,
    "rectangle": _rectangle,
    "wall": _wall,
    "star": _star,
    "letter": _letter,
    "blob": _blob,
}


def _sample_sources(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, strength: float
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Smooth sources that don't overlap, peaking at exactly ``strength``.

    Returns the inflow, the area obstacles must keep out of, and each source's x. The edges
    follow a raised cosine rather than a step, because a hard edge is a different source on
    each grid (``docs/DECISIONS.md`` B12). The support is finite, so a source can be kept
    wholly clear.
    """
    count = int(rng.integers(_SOURCE_COUNT[0], _SOURCE_COUNT[1] + 1))
    placed: list[tuple[float, float, float]] = []
    while len(placed) < count:
        cx, cy = rng.uniform(*_SOURCE_X), rng.uniform(*_SOURCE_Y)
        radius = rng.uniform(*_SOURCE_RADIUS)
        if all(np.hypot(cx - a, cy - b) > radius + r + _CLEARANCE for a, b, r in placed):
            placed.append((cx, cy, radius))

    inflow = np.zeros_like(x, dtype=np.float32)
    keep_out = np.zeros_like(x, dtype=bool)
    for cx, cy, radius in placed:
        distance = np.hypot(x - cx, y - cy)
        bump = np.where(distance < radius, np.cos(0.5 * np.pi * distance / radius) ** 2, 0.0)
        # The peak falls between cells, so scale it to reach the strength exactly.
        inflow += (bump / bump.max() * strength).astype(np.float32)
        keep_out |= distance < radius + _CLEARANCE
    return inflow, keep_out, [cx for cx, _, _ in placed]


def _place_shape(
    rng: np.random.Generator,
    families: tuple[str, ...],
    x: np.ndarray,
    y: np.ndarray,
    keep_out: np.ndarray,
    existing: np.ndarray | None = None,
    x_range: tuple[float, float] = _OBSTACLE_X,
) -> np.ndarray:
    """A random shape clear of ``keep_out``, and mostly new if ``existing`` is given."""
    for _ in range(_POSE_ATTEMPTS):
        family = families[int(rng.integers(len(families)))]
        cx, cy = rng.uniform(*x_range), rng.uniform(*_OBSTACLE_Y)
        shape = _BUILDERS[family](rng, x - cx, y - cy)
        if not shape.any() or (shape & keep_out).any():
            continue
        if existing is not None and (shape & ~existing).sum() < 0.5 * shape.sum():
            continue
        return shape
    raise RuntimeError(f"no clear place found for any of {families}")


def _has_way_up(solid: np.ndarray, inflow: np.ndarray) -> bool:
    """True if smoke from every source cell can reach the top row through fluid."""
    fluid = ~solid
    source = inflow > 0
    reached = fluid.copy()
    reached[:-1] = False  # Spread down from the top row, since it is one region to fill.
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= fluid
        if np.array_equal(grown, reached):
            return bool(reached[source].all())
        reached = grown


def _box_margin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    edge = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return edge < _CLEARANCE


def _sample_physics(rng: np.random.Generator, split: str) -> tuple[float, float]:
    """Buoyancy and inflow strength. In test_c, each independently falls low or high."""
    if split != "test_c":
        return float(rng.uniform(*TRAIN_BUOYANCY)), float(rng.uniform(*TRAIN_INFLOW))
    buoyancy = TEST_C_BUOYANCY[int(rng.integers(2))]
    strength = TEST_C_INFLOW[int(rng.integers(2))]
    return float(rng.uniform(*buoyancy)), float(rng.uniform(*strength))


def sample_scene(rng: np.random.Generator, res: int, split: str) -> Scene:
    """Draw one scene for ``split``. The result depends only on ``rng``'s state and the arguments."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    families = UNSEEN_SHAPES if split == "test_b" else TRAIN_SHAPES
    has_late = split in ("train", "val", "test_a")
    x, y = _grid(res)
    box_margin = _box_margin(x, y)

    # A scene that seals its own plume in is redrawn whole, so that the draw isn't biased
    # towards particular late obstacles.
    for _ in range(_SCENE_ATTEMPTS):
        buoyancy, strength = _sample_physics(rng, split)
        inflow, keep_out, source_xs = _sample_sources(rng, x, y, strength)
        keep_out |= box_margin

        # The first shape goes in a plume's path. A plume that misses every shape makes a
        # scene easy, and in test_b would hide the unseen geometry from the model.
        source_x = source_xs[int(rng.integers(len(source_xs)))]
        in_path = (source_x - _IN_PATH_REACH, source_x + _IN_PATH_REACH)
        obstacle = _place_shape(rng, families, x, y, keep_out, x_range=in_path)
        for _ in range(int(rng.integers(_OBSTACLE_COUNT[0], _OBSTACLE_COUNT[1] + 1)) - 1):
            obstacle |= _place_shape(rng, families, x, y, keep_out)

        late_obstacle = None
        late_frame = None
        if has_late and rng.uniform() < LATE_SHARE:
            late_obstacle = _place_shape(rng, families, x, y, keep_out, existing=obstacle)
            late_frame = int(rng.integers(LATE_FRAMES[0], LATE_FRAMES[1] + 1))

        solid = obstacle if late_obstacle is None else obstacle | late_obstacle
        if _has_way_up(solid, inflow):
            return Scene(obstacle, inflow, buoyancy, late_obstacle, late_frame)
    raise RuntimeError(f"no open scene found for split {split!r}")
