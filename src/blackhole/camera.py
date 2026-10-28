"""Pinhole camera that generates one ray per pixel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int = 960
    height: int = 540
    fov: float = 40.0  # vertical field of view, degrees
    distance: float = 30.0  # in Schwarzschild radii
    inclination: float = 84.0  # angle from the disk axis; 90 = edge-on
    azimuth: float = 0.0
    roll: float = 0.0

    @property
    def position(self) -> np.ndarray:
        i, a = np.radians(self.inclination), np.radians(self.azimuth)
        return self.distance * np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])

    @property
    def pixel_angle(self) -> float:
        return np.radians(self.fov) / self.height

    def basis(self):
        pos = self.position
        fwd = -pos / np.linalg.norm(pos)
        up0 = np.array([0.0, 0.0, 1.0])
        if abs(fwd @ up0) > 0.999:
            up0 = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up0)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        r = np.radians(self.roll)
        right, up = np.cos(r) * right + np.sin(r) * up, -np.sin(r) * right + np.cos(r) * up
        return fwd, right, up

    def rays(self) -> np.ndarray:
        """Unit ray directions, shape (height * width, 3), row-major from top-left."""
        fwd, right, up = self.basis()
        t = np.tan(np.radians(self.fov) / 2)
        xs = ((np.arange(self.width) + 0.5) / self.width * 2 - 1) * t * self.width / self.height
        ys = (1 - (np.arange(self.height) + 0.5) / self.height * 2) * t
        X, Y = np.meshgrid(xs, ys)
        d = fwd[None, :] + X.reshape(-1, 1) * right[None, :] + Y.reshape(-1, 1) * up[None, :]
        return d / np.linalg.norm(d, axis=1, keepdims=True)
