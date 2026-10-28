"""Null geodesics (light rays) in Schwarzschild spacetime.

Units: G = c = 1 and the Schwarzschild radius r_s = 2M = 1, so the black
hole has mass M = 1/2, the event horizon sits at r = 1 and the photon sphere
at r = 3/2.

The trick
---------
A photon's orbit in Schwarzschild spacetime obeys (u = 1/r, phi = orbital angle)

    d^2u/dphi^2 + u = 3 M u^2 = (3/2) u^2.

Exactly the same orbit shape comes out of a *Newtonian* particle moving in
flat 3D space under the central force

    a = -(3/2) h^2 x / |x|^5,      h = |x x v|  (conserved)

(apply Binet's formula: u'' + u = -F / (h^2 u^2) = (3/2) u^2). So we can
integrate plain Cartesian ODEs, fully vectorised over every pixel's ray,
and still trace the exact Schwarzschild light paths. The parameter along
the ray is not affine, but rays only need their *shape*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M = 0.5  # black hole mass in units of r_s
HORIZON = 1.0
PHOTON_SPHERE = 1.5
ISCO = 3.0  # innermost stable circular orbit, 6M
B_CRIT = 1.5 * np.sqrt(3.0)  # critical impact parameter 3*sqrt(3) M

ESCAPED, CAPTURED, DISK = 0, 1, 2


def _norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def acceleration(x: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Effective central acceleration -(3/2) h^2 x / r^5 for an (N, 3) batch."""
    r2 = np.einsum("ij,ij->i", x, x)
    return -1.5 * (h2 / r2**2.5)[:, None] * x


def rk4_step(x, v, h2, dt):
    """One classical Runge-Kutta 4 step with a per-ray step size ``dt``."""
    dt = dt[:, None]
    k1x, k1v = v, acceleration(x, h2)
    k2x, k2v = v + 0.5 * dt * k1v, acceleration(x + 0.5 * dt * k1x, h2)
    k3x, k3v = v + 0.5 * dt * k2v, acceleration(x + 0.5 * dt * k2x, h2)
    k4x, k4v = v + dt * k3v, acceleration(x + dt * k3x, h2)
    x_new = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_new = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_new, v_new


def impact_parameter(x0: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """Exact impact parameter b = L/E of rays launched from x0 along d0.

    Matching the first integral of the orbit equation,
    (du/dphi)^2 = 1/b^2 - u^2 + u^3, to the launch geometry gives
    1/b^2 = 1/h^2 - 1/r^3 with h = |x0 x d0| for unit d0.
    """
    x0 = np.broadcast_to(np.atleast_2d(x0), d0.shape)
    h = _norm(np.cross(x0, d0))
    r = _norm(x0)
    return 1.0 / np.sqrt(1.0 / h**2 - 1.0 / r**3)


@dataclass
class TraceResult:
    status: np.ndarray  # ESCAPED / CAPTURED / DISK per ray
    position: np.ndarray  # disk hit point, or last position
    direction: np.ndarray  # unit direction at the end of the ray
    min_radius: np.ndarray  # closest approach to the hole
    path: np.ndarray | None = None  # (steps, N, 3) if record=True


def trace(
    x0,
    d0,
    disk: tuple[float, float] | None = (ISCO, 15.0),
    r_escape: float | None = None,
    step: float = 0.04,
    max_step: float = np.inf,
    max_steps: int = 20000,
    record: bool = False,
) -> TraceResult:
    """Trace a batch of light rays backwards from the camera.

    Parameters
    ----------
    x0 : (3,) or (N, 3) start position(s)
    d0 : (N, 3) initial directions (normalised internally)
    disk : (r_in, r_out) of a thin opaque disk in the z = 0 plane, or None
    r_escape : rays moving outward beyond this radius count as escaped
    step : step size as a fraction of the current radius (dt = step * r)
    record : keep the full path (only sensible for a handful of rays)
    """
    d0 = np.atleast_2d(np.asarray(d0, dtype=float))
    d0 = d0 / _norm(d0)[:, None]
    n = len(d0)
    x = np.array(np.broadcast_to(np.atleast_2d(np.asarray(x0, float)), (n, 3)))
    v = d0.copy()
    h2 = np.einsum("ij,ij->i", c := np.cross(x, v), c)

    if r_escape is None:
        r_escape = max(2.0 * _norm(x).max(), 50.0)

    status = np.full(n, CAPTURED, dtype=np.int8)  # leftovers: trapped near photon sphere
    pos = x.copy()
    vel = v.copy()
    min_r = _norm(x)
    active = np.arange(n)
    path = [x.copy()] if record else None

    for _ in range(max_steps):
        if active.size == 0:
            break
        xa, va = x[active], v[active]
        ra = _norm(xa)
        dt = np.minimum(step * ra, max_step)
        xn, vn = rk4_step(xa, va, h2[active], dt)
        rn = _norm(xn)
        min_r[active] = np.minimum(min_r[active], rn)

        hit = np.zeros(active.size, dtype=bool)
        if disk is not None:
            z0, z1 = xa[:, 2], xn[:, 2]
            cross = (z0 * z1 < 0) | ((z1 == 0) & (z0 != 0))
            if cross.any():
                t = np.where(cross, z0 / np.where(cross, z0 - z1, 1.0), 0.0)
                p = xa + t[:, None] * (xn - xa)
                rp = _norm(p)
                hit = cross & (rp >= disk[0]) & (rp <= disk[1])
                idx = active[hit]
                status[idx] = DISK
                pos[idx] = p[hit]
                vel[idx] = vn[hit]

        cap = (rn < HORIZON) & ~hit
        esc = (rn > r_escape) & (np.einsum("ij,ij->i", xn, vn) > 0) & ~hit
        status[active[cap]] = CAPTURED
        status[active[esc]] = ESCAPED
        done = hit | cap | esc
        keep = ~hit
        x[active[keep]] = xn[keep]
        v[active[keep]] = vn[keep]
        pos[active[cap | esc]] = xn[cap | esc]
        vel[active[cap | esc]] = vn[cap | esc]
        if record:
            frame = np.full_like(x, np.nan)
            frame[active] = np.where(hit[:, None], pos[active], xn)
            path.append(frame)
        active = active[~done]

    direction = vel / _norm(vel)[:, None]
    return TraceResult(
        status=status,
        position=pos,
        direction=direction,
        min_radius=min_r,
        path=np.array(path) if record else None,
    )
