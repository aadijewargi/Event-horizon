"""Thin accretion disk: temperature profile, relativistic redshift, texture."""
from __future__ import annotations

import numpy as np

from .geodesic import M


def temperature_profile(r: np.ndarray, r_in: float) -> np.ndarray:
    """Relative temperature of a thin (Shakura-Sunyaev-like) disk.

    T(r) ∝ r^(-3/4) (1 - sqrt(r_in / r))^(1/4), from the zero-torque inner
    boundary condition. Normalised so the peak (at r = 49/36 r_in) is 1.
    """
    r = np.maximum(r, r_in)
    f = r**-0.75 * (1.0 - np.sqrt(r_in / r)) ** 0.25
    rp = 49.0 / 36.0 * r_in
    fmax = rp**-0.75 * (1.0 - np.sqrt(r_in / rp)) ** 0.25
    return f / fmax


def keplerian_omega(r: np.ndarray) -> np.ndarray:
    """Coordinate angular velocity dphi/dt of a circular geodesic: sqrt(M / r^3)."""
    return np.sqrt(M / r**3)


def redshift_g(r: np.ndarray, lz: np.ndarray, observer_r: float = np.inf) -> np.ndarray:
    """Frequency ratio g = nu_observed / nu_emitted for disk light.

    The gas moves on prograde circular orbits with 4-velocity
    u = u^t (1, 0, 0, Omega), u^t = 1 / sqrt(1 - 3M/r). A photon with
    conserved energy E and angular momentum L_z about the disk axis has

        1 + z = (p·u)_emit / (p·u)_obs = u^t (1 - Omega L_z / E) * sqrt(1 - 2M/r_obs)

    for a static observer at r_obs. This one formula contains gravitational
    redshift, the transverse Doppler effect and Doppler beaming asymmetry.
    """
    omega = keplerian_omega(r)
    ut = 1.0 / np.sqrt(1.0 - 3.0 * M / r)
    obs = np.sqrt(1.0 - 2.0 * M / observer_r) if np.isfinite(observer_r) else 1.0
    return 1.0 / (ut * (1.0 - omega * lz) * obs)


def disk_texture(r: np.ndarray, phi: np.ndarray, t: float = 0.0, seed: int = 7) -> np.ndarray:
    """Streaky turbulent-looking brightness modulation, co-rotating with the gas.

    Each feature is advected at its local Keplerian rate, so in an animation
    the inner disk visibly laps the outer disk (differential rotation).
    """
    rng = np.random.default_rng(seed)
    phase = phi - keplerian_omega(r) * t
    lr = np.log(r)
    tex = np.zeros_like(r)
    for m in (2, 3, 5, 8, 13, 21, 34):
        k = rng.uniform(-6, 6)  # radial wavenumber -> spiral pitch
        a = rng.uniform(0.5, 1.0) / np.sqrt(m)
        tex += a * np.sin(m * phase + k * lr * m**0.5 + rng.uniform(0, 2 * np.pi))
        tex += 0.5 * a * np.sin(rng.uniform(25, 60) * lr + rng.uniform(0, 2 * np.pi))
    tex /= np.abs(tex).max() + 1e-12
    return np.clip(1.0 + 0.45 * tex, 0.2, None)
