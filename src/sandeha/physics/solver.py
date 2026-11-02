"""Reference smoke solver: incompressible 2D flow on a staggered grid, in PyTorch.

Conventions:
- The domain is the unit square, so a cell is ``1 / res`` wide, and velocity is in
  domain lengths per unit time.
- Tensor index ``(row, column)`` maps to ``(y, x)``. Rows grow upward, so up is +row,
  and buoyancy pushes along +v. Plots of these fields need ``origin="lower"``.
- Inside, ``u`` lives on vertical cell faces ``(B,1,H,W+1)`` and ``v`` on horizontal
  cell faces ``(B,1,H+1,W)``. Those are the fields actually made divergence-free.

Pressure is solved in units where the spacing and time step drop out. For a correction
``phi``, faces change by ``-grad(phi)``, and ``A phi = -b``, where ``b`` is the net flux
out of each cell and ``A`` is the graph Laplacian of connected fluid cells.

All arithmetic is float32. Its rounding limit is handled by repeatedly re-measuring the
true divergence from the faces and solving for what is left. See ``_project_faces``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

DENSITY_DECAY = 0.1  # Per unit time. Light, so smoke does not saturate the closed box.

_COARSEST = 4  # Largest grid side at which multigrid stops coarsening.
_COARSEST_SWEEPS = 2  # Symmetric sweep pairs on that grid.
_OVER_CORRECTION = 1.9  # Piecewise-constant coarse corrections are too small by about 2.
_INNER_REDUCTION = 1e-3  # Residual reduction asked of one run of conjugate gradients.
_MIN_PROGRESS = 0.5  # A run must at least halve the true divergence, or the item stops.


@dataclass
class SmokeState:
    """A smoke state. ``velocity`` is the cell-centred view of the faces in ``mac``.

    ``mac`` is what the solver carries from step to step, because averaging to cell
    centres and back would act as a blur every step. Leave it as None for a state built
    from cell-centred fields. To change ``velocity``, build a new state without ``mac``.
    """

    velocity: Tensor  # (B,2,H,W)
    density: Tensor  # (B,1,H,W)
    mac: tuple[Tensor, Tensor] | None = field(default=None, repr=False)


@dataclass
class PressureReport:
    """How the most recent pressure solve ended, per batch item."""

    iterations: Tensor  # (B,) conjugate gradient iterations
    residual: Tensor  # (B,) largest absolute divergence in a cell, in physical units
    converged: Tensor  # (B,) bool. False if the limit was hit or the input was not finite


@dataclass
class _Level:
    gx: Tensor  # Links between horizontal neighbours, (B,1,H,W-1)
    gy: Tensor  # Links between vertical neighbours, (B,1,H-1,W)
    inv_diag: Tensor  # Zero for cells with no links
    red: Tensor
    black: Tensor


def _laplacian(level: _Level, x: Tensor) -> Tensor:
    fx = level.gx * (x[..., 1:] - x[..., :-1])
    fy = level.gy * (x[..., 1:, :] - x[..., :-1, :])
    return (
        F.pad(fx, (1, 0)) - F.pad(fx, (0, 1)) + F.pad(fy, (0, 0, 1, 0)) - F.pad(fy, (0, 0, 0, 1))
    )


def _make_level(gx: Tensor, gy: Tensor) -> _Level:
    diag = F.pad(gx, (1, 0)) + F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 1, 0)) + F.pad(gy, (0, 0, 0, 1))
    inv_diag = torch.where(diag > 0, 1.0 / diag.clamp(min=1.0), torch.zeros_like(diag))
    h, w = diag.shape[-2:]
    rows = torch.arange(h, device=diag.device)[:, None]
    columns = torch.arange(w, device=diag.device)[None, :]
    red = ((rows + columns) % 2 == 0).to(diag.dtype)
    return _Level(gx, gy, inv_diag, red, 1.0 - red)


def _build_hierarchy(fluid: Tensor) -> list[_Level]:
    """Build Galerkin levels by merging 2x2 blocks. A link's weight is the fine links it replaces.

    This handles any obstacle exactly and keeps each coarse operator symmetric.
    """
    dtype = torch.float32
    gx = (fluid[..., 1:] & fluid[..., :-1]).to(dtype)
    gy = (fluid[..., 1:, :] & fluid[..., :-1, :]).to(dtype)
    levels = [_make_level(gx, gy)]
    h, w = fluid.shape[-2:]
    while h > _COARSEST and w > _COARSEST and h % 2 == 0 and w % 2 == 0:
        gx = gx[..., 1::2]
        gx = gx[..., 0::2, :] + gx[..., 1::2, :]
        gy = gy[..., 1::2, :]
        gy = gy[..., 0::2] + gy[..., 1::2]
        h, w = h // 2, w // 2
        levels.append(_make_level(gx, gy))
    return levels


def _smooth(level: _Level, x: Tensor, rhs: Tensor, first: Tensor, second: Tensor) -> Tensor:
    x = x + first * level.inv_diag * (rhs - _laplacian(level, x))
    return x + second * level.inv_diag * (rhs - _laplacian(level, x))


def _v_cycle(levels: list[_Level], rhs: Tensor, depth: int = 0) -> Tensor:
    """One symmetric V-cycle. The post-smoother is the exact reverse of the pre-smoother."""
    level = levels[depth]
    # The first half-sweep from zero needs no stencil, which saves one in five on each level.
    x = level.red * level.inv_diag * rhs
    x = x + level.black * level.inv_diag * (rhs - _laplacian(level, x))
    if depth == len(levels) - 1:
        for _ in range(_COARSEST_SWEEPS - 1):
            x = _smooth(level, x, rhs, level.red, level.black)
        for _ in range(_COARSEST_SWEEPS):
            x = _smooth(level, x, rhs, level.black, level.red)
        return x

    residual = rhs - _laplacian(level, x)
    coarse = _v_cycle(levels, F.avg_pool2d(residual, 2) * 4.0, depth + 1)
    x = x + _OVER_CORRECTION * coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    return _smooth(level, x, rhs, level.black, level.red)


def _conjugate_gradient(
    levels: list[_Level], rhs: Tensor, target: Tensor, budget: Tensor
) -> tuple[Tensor, Tensor]:
    """Solve ``A x = rhs`` with one step size per batch item, stopping each item on its own.

    Items whose budget is zero, or whose residual max-norm is at or below ``target``, stay
    frozen. An item also stops if the search direction loses positive curvature, which
    happens at the rounding limit. Returns the solution and the iterations used per item.
    """
    level = levels[0]
    batch = rhs.shape[0]
    shape = (batch, 1, 1, 1)
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    active = budget > 0
    z = _v_cycle(levels, residual)
    direction = z
    rho = (residual * z).sum(dim=(1, 2, 3))
    iterations = torch.zeros(batch, dtype=torch.long, device=rhs.device)

    while True:
        active = active & (residual.abs().amax(dim=(1, 2, 3)) > target) & (iterations < budget)
        if not active.any():
            return x, iterations
        a_direction = _laplacian(level, direction)
        curvature = (direction * a_direction).sum(dim=(1, 2, 3))
        active = active & (curvature > 0) & (rho > 0)
        if not active.any():
            return x, iterations
        alpha = torch.where(active, rho / curvature, torch.zeros_like(rho)).view(shape)
        x = x + alpha * direction
        residual = residual - alpha * a_direction
        iterations = iterations + active.long()

        z = _v_cycle(levels, residual)
        rho_next = (residual * z).sum(dim=(1, 2, 3))
        beta = torch.where(active, rho_next / rho, torch.zeros_like(rho)).view(shape)
        direction = z + beta * direction
        rho = rho_next


def _interpolate(q: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Bilinear lookup at lattice coordinates, clamped to the lattice."""
    batch, _, height, width = q.shape
    x = x.clamp(min=0.0, max=width - 1.0)
    y = y.clamp(min=0.0, max=height - 1.0)
    i = x.floor().long().clamp(max=width - 2)
    j = y.floor().long().clamp(max=height - 2)
    fx = x - i
    fy = y - j
    flat = q.view(batch, -1)
    index = (j * width + i).view(batch, -1)

    def corner(offset: int) -> Tensor:
        return flat.gather(1, index + offset).view_as(x)

    bottom = corner(0) * (1 - fx) + corner(1) * fx
    top = corner(width) * (1 - fx) + corner(width + 1) * fx
    return bottom * (1 - fy) + top * fy


class SmokeSolver:
    """Closed-box smoke solver. Batched, with one pressure solve per batch item."""

    def __init__(self, res: int, device, dt: float, cg_tol: float = 1e-5, cg_max_iter: int = 2000):
        self.res = res
        self.device = torch.device(device)
        self.dt = dt
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.last_solve: PressureReport | None = None
        self._cached_fluid: Tensor | None = None
        self._levels: list[_Level] = []

    # Public interface

    def step(
        self, state: SmokeState, obstacle: Tensor, inflow: Tensor, buoyancy: Tensor
    ) -> SmokeState:
        """Advance one step: advect, add smoke and buoyancy, then project."""
        fluid = obstacle == 0
        levels = self._hierarchy(fluid)
        if state.mac is None:
            state = self.project(state, obstacle)
        u, v = state.mac
        density = state.density * fluid

        new_density = self._advect(density, 0.5, 0.5, u, v)
        new_u = self._advect(u, 0.0, 0.5, u, v)
        new_v = self._advect(v, 0.5, 0.0, u, v)

        new_density = new_density + self.dt * inflow.clamp(min=0.0) * fluid
        new_density = new_density * torch.exp(torch.tensor(-DENSITY_DECAY * self.dt))
        lift = self.dt * buoyancy.view(-1, 1, 1, 1)
        new_v = torch.cat(
            [
                new_v[..., :1, :],
                new_v[..., 1:-1, :] + lift * 0.5 * (new_density[..., 1:, :] + new_density[..., :-1, :]),
                new_v[..., -1:, :],
            ],
            dim=-2,
        )

        new_u, new_v = self._project_faces(new_u, new_v, levels)
        return SmokeState(self._centre(new_u, new_v), new_density, (new_u, new_v))

    def project(self, state: SmokeState, obstacle: Tensor) -> SmokeState:
        """Remove divergence, and clear velocity and density from solid cells."""
        fluid = obstacle == 0
        levels = self._hierarchy(fluid)
        u, v = state.mac if state.mac is not None else self._faces_from_centres(state.velocity)
        u, v = self._project_faces(u, v, levels)
        return SmokeState(self._centre(u, v), state.density * fluid, (u, v))

    def divergence(self, state: SmokeState, obstacle: Tensor) -> Tensor:
        """Divergence per cell ``(B,1,H,W)`` in physical units, and zero in solid cells.

        It is measured on the faces when the state has them, since those are what the
        solver makes divergence-free. The cell-centred view has a larger divergence.
        """
        fluid = obstacle == 0
        levels = self._hierarchy(fluid)
        u, v = state.mac if state.mac is not None else self._faces_from_centres(state.velocity)
        return self._net_flux(*self._mask_faces(u, v, levels)) * fluid * self.res

    def preconditioner(self, obstacle: Tensor) -> Callable[[Tensor], Tensor]:
        """The multigrid preconditioner for this obstacle, exposed so it can be checked."""
        levels = self._hierarchy(obstacle == 0)
        return lambda rhs: _v_cycle(levels, rhs)

    # Pressure

    def _hierarchy(self, fluid: Tensor) -> list[_Level]:
        cached = self._cached_fluid
        if cached is None or cached.shape != fluid.shape or not torch.equal(cached, fluid):
            self._levels = _build_hierarchy(fluid)
            self._cached_fluid = fluid.clone()
        return self._levels

    def _project_faces(self, u: Tensor, v: Tensor, levels: list[_Level]) -> tuple[Tensor, Tensor]:
        """Make the faces divergence-free, per batch item.

        The target is that no cell's net flux exceeds ``cg_tol`` times the item's fastest
        face. Each pass re-measures the true flux from the faces and solves for what is
        left, so the recursive residual inside conjugate gradients can't drift unseen, and
        each correction is small enough for float32 to represent accurately.
        """
        u, v = self._mask_faces(u, v, levels)
        batch = u.shape[0]
        speed = torch.maximum(u.abs().amax(dim=(1, 2, 3)), v.abs().amax(dim=(1, 2, 3)))
        tolerance = self.cg_tol * speed
        used = torch.zeros(batch, dtype=torch.long, device=u.device)
        previous = torch.full((batch,), torch.inf, device=u.device)
        level = levels[0]

        while True:
            flux = self._net_flux(u, v) * (level.inv_diag > 0)
            measured = flux.abs().amax(dim=(1, 2, 3))
            # Comparisons with NaN are false, so a non-finite item never starts or converges.
            converged = measured <= tolerance
            active = ~converged & (measured < _MIN_PROGRESS * previous) & (used < self.cg_max_iter)
            if not active.any():
                self.last_solve = PressureReport(used, measured * self.res, converged)
                return u, v

            view = active.view(-1, 1, 1, 1)
            rhs = torch.where(view, -flux, torch.zeros_like(flux))
            target = torch.maximum(_INNER_REDUCTION * measured, 0.5 * tolerance)
            budget = torch.where(active, self.cg_max_iter - used, torch.zeros_like(used))
            phi, iterations = _conjugate_gradient(levels, rhs, target, budget)

            u = torch.cat(
                [u[..., :1], u[..., 1:-1] - level.gx * (phi[..., 1:] - phi[..., :-1]), u[..., -1:]],
                dim=-1,
            )
            v = torch.cat(
                [
                    v[..., :1, :],
                    v[..., 1:-1, :] - level.gy * (phi[..., 1:, :] - phi[..., :-1, :]),
                    v[..., -1:, :],
                ],
                dim=-2,
            )
            used = used + iterations
            previous = torch.where(active, measured, previous)

    @staticmethod
    def _mask_faces(u: Tensor, v: Tensor, levels: list[_Level]) -> tuple[Tensor, Tensor]:
        """Zero every wall face and every face that touches a solid cell."""
        level = levels[0]
        return F.pad(u[..., 1:-1] * level.gx, (1, 1)), F.pad(
            v[..., 1:-1, :] * level.gy, (0, 0, 1, 1)
        )

    @staticmethod
    def _net_flux(u: Tensor, v: Tensor) -> Tensor:
        return u[..., 1:] - u[..., :-1] + v[..., 1:, :] - v[..., :-1, :]

    @staticmethod
    def _faces_from_centres(velocity: Tensor) -> tuple[Tensor, Tensor]:
        cu, cv = velocity[:, :1], velocity[:, 1:]
        u = F.pad(0.5 * (cu[..., 1:] + cu[..., :-1]), (1, 1))
        v = F.pad(0.5 * (cv[..., 1:, :] + cv[..., :-1, :]), (0, 0, 1, 1))
        return u, v

    @staticmethod
    def _centre(u: Tensor, v: Tensor) -> Tensor:
        return torch.cat(
            [0.5 * (u[..., 1:] + u[..., :-1]), 0.5 * (v[..., 1:, :] + v[..., :-1, :])], dim=1
        )

    # Advection

    def _advect(self, q: Tensor, offset_x: float, offset_y: float, u: Tensor, v: Tensor) -> Tensor:
        """Semi-Lagrangian advection, tracing back with a midpoint rule.

        A lattice node ``(j, i)`` of ``q`` sits at ``(i + offset_x, j + offset_y)`` in cells.
        Bilinear lookup can't overshoot, so density stays non-negative.
        """
        batch, _, height, width = q.shape
        rows = torch.arange(height, device=q.device, dtype=q.dtype) + offset_y
        columns = torch.arange(width, device=q.device, dtype=q.dtype) + offset_x
        y, x = torch.meshgrid(rows, columns, indexing="ij")
        x = x.expand(batch, 1, -1, -1)
        y = y.expand(batch, 1, -1, -1)

        reach = self.dt * self.res
        mid_x = x - 0.5 * reach * _interpolate(u, x, y - 0.5)
        mid_y = y - 0.5 * reach * _interpolate(v, x - 0.5, y)
        from_x = x - reach * _interpolate(u, mid_x, mid_y - 0.5)
        from_y = y - reach * _interpolate(v, mid_x - 0.5, mid_y)
        return _interpolate(q, from_x - offset_x, from_y - offset_y)
