"""Physically based blackbody colours.

The colour of a blackbody at temperature T is found by integrating Planck's
law against the CIE 1931 colour-matching functions (using the analytic
multi-lobe Gaussian fit of Wyman, Sloan & Shirley 2013), then converting
CIE XYZ to linear sRGB. Results are cached in a lookup table.
"""
from __future__ import annotations

import numpy as np

_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23

_XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_cmf(lam_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2-degree colour matching functions, shape (len(lam), 3)."""
    x = 1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def planck(lam_m: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Spectral radiance B_lambda(T) (arbitrary overall scale)."""
    a = _H * _C / (lam_m * _KB * T)
    return 1.0 / (lam_m**5 * np.expm1(np.minimum(a, 700.0)))


def blackbody_rgb(T: np.ndarray | float) -> np.ndarray:
    """Chromaticity of a blackbody as linear sRGB, normalised to max channel = 1."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    lam = np.linspace(380, 780, 201)
    cmf = cie_cmf(lam)
    spec = planck(lam[None, :] * 1e-9, T[:, None])
    xyz = spec @ cmf
    rgb = xyz @ _XYZ_TO_SRGB.T
    rgb = np.clip(rgb, 0.0, None)
    return rgb / rgb.max(axis=1, keepdims=True)


class BlackbodyLUT:
    """Fast log-spaced lookup table of blackbody colours."""

    def __init__(self, t_min=500.0, t_max=1e5, n=2048):
        self.logt = np.linspace(np.log(t_min), np.log(t_max), n)
        self.rgb = blackbody_rgb(np.exp(self.logt))

    def __call__(self, T: np.ndarray) -> np.ndarray:
        lt = np.clip(np.log(np.maximum(T, 1.0)), self.logt[0], self.logt[-1])
        return np.stack([np.interp(lt, self.logt, self.rgb[:, k]) for k in range(3)], axis=-1)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear [0,1] -> sRGB gamma-encoded [0,1]."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
