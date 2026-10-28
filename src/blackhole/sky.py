"""Procedural celestial spheres for the background."""
from __future__ import annotations

import numpy as np

from .color import BlackbodyLUT

_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        _LUT = BlackbodyLUT()
    return _LUT


def _hash(a: np.ndarray, b: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministic integer hash -> uniform floats in [0, 1)."""
    with np.errstate(over="ignore"):
        h = (a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (
            b.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        )
        h ^= np.uint64(k * 0x165667B19E3779F9 + seed)
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xC4CEB9FE1A85EC53)
        h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def starfield(dirs: np.ndarray, pixel_angle: float, seed: int = 1, rows: int = 260, density: float = 0.22) -> np.ndarray:
    """Stars + a faint galactic band, in linear RGB, for unit directions (N, 3).

    The sphere is cut into equal-area cells (uniform in z and phi); each cell
    may hold one star with a random position, power-law brightness and a
    blackbody colour. Gravitational lensing then smears and duplicates them.
    """
    z = np.clip(dirs[:, 2], -1, 1)
    phi = np.arctan2(dirs[:, 1], dirs[:, 0]) + np.pi
    cols = 2 * rows
    iz = np.minimum(((z + 1) / 2 * rows).astype(np.int64), rows - 1)
    ip = np.minimum((phi / (2 * np.pi) * cols).astype(np.int64), cols - 1)

    sz = (iz + 0.2 + 0.6 * _hash(iz, ip, 1, seed)) / rows * 2 - 1
    sp = (ip + 0.2 + 0.6 * _hash(iz, ip, 2, seed)) / cols * 2 * np.pi - np.pi
    s_xy = np.sqrt(1 - sz**2)
    star = np.stack([s_xy * np.cos(sp), s_xy * np.sin(sp), sz], axis=-1)
    dist2 = np.einsum("ij,ij->i", dirs - star, dirs - star)

    present = _hash(iz, ip, 3, seed) < density
    mag = _hash(iz, ip, 4, seed)
    bright = 0.012 / (mag + 0.02) ** 1.3 * present
    temp = 2800 + 12000 * _hash(iz, ip, 5, seed) ** 2
    sigma = 0.6 * pixel_angle
    flux = bright * np.exp(-dist2 / (2 * sigma**2))
    rgb = flux[:, None] * _lut()(temp)

    # faint, mottled galactic band on a tilted great circle
    n = np.array([0.35, -0.25, 0.9])
    n /= np.linalg.norm(n)
    lat = dirs @ n
    mottle = 0.6 + 0.4 * np.sin(9 * dirs[:, 0] + 3) * np.sin(7 * dirs[:, 1] - 1) * np.sin(11 * dirs[:, 2] + 2)
    band = 0.035 * np.exp(-((lat / 0.18) ** 2)) * mottle
    rgb += band[:, None] * np.array([0.75, 0.8, 1.0])
    return rgb


def checkerboard(dirs: np.ndarray, n: int = 18) -> np.ndarray:
    """Checkerboard sky: makes the lensing distortion easy to read."""
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    a = np.floor(theta / np.pi * n).astype(int)
    b = np.floor((phi + np.pi) / (2 * np.pi) * 2 * n).astype(int)
    c1, c2 = np.array([0.95, 0.55, 0.15]), np.array([0.1, 0.3, 0.7])
    return np.where(((a + b) % 2 == 0)[:, None], c1, c2) * 0.6
