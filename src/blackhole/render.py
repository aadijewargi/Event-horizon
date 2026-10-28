"""Turn traced rays into images."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import geodesic as geo
from .camera import Camera
from .color import BlackbodyLUT, srgb_encode
from .disk import disk_texture, redshift_g, temperature_profile
from .sky import checkerboard, starfield


@dataclass
class Scene:
    camera: Camera = field(default_factory=Camera)
    disk_inner: float = geo.ISCO
    disk_outer: float = 12.0
    disk_temperature: float = 4800.0  # peak emitted temperature, kelvin
    sky: str = "stars"  # "stars", "checker" or "none"
    relativistic_shading: bool = True  # apply redshift / beaming
    show_disk: bool = True
    supersample: int = 2
    exposure: float = 0.9
    contrast: float = 1.6  # display-only power curve on luminance (1 = linear)
    saturation: float = 1.5  # display-only colour saturation boost
    bloom: float = 0.35
    step: float = 0.04

    _trace: geo.TraceResult | None = field(default=None, init=False, repr=False)

    def _ss_camera(self) -> Camera:
        c = self.camera
        s = self.supersample
        return Camera(c.width * s, c.height * s, c.fov, c.distance, c.inclination, c.azimuth, c.roll)

    def trace(self) -> geo.TraceResult:
        """Trace every pixel's light ray once; shading can then be redone cheaply."""
        if self._trace is None:
            cam = self._ss_camera()
            self._dirs0 = cam.rays()
            disk = (self.disk_inner, self.disk_outer) if self.show_disk else None
            self._trace = geo.trace(cam.position, self._dirs0, disk=disk, step=self.step, max_step=2.0)
        return self._trace

    def hdr(self, t: float = 0.0) -> np.ndarray:
        """Linear-light HDR image at supersampled resolution, shape (H, W, 3)."""
        res = self.trace()
        cam = self._ss_camera()
        img = np.zeros((len(res.status), 3))

        esc = res.status == geo.ESCAPED
        if self.sky == "stars":
            img[esc] = starfield(res.direction[esc], pixel_angle=cam.pixel_angle)
        elif self.sky == "checker":
            img[esc] = checkerboard(res.direction[esc])

        hit = res.status == geo.DISK
        if hit.any():
            p = res.position[hit]
            r = np.linalg.norm(p, axis=1)
            phi = np.arctan2(p[:, 1], p[:, 0])
            T = temperature_profile(r, self.disk_inner)
            if self.relativistic_shading:
                # conserved L_z/E of the photon travelling *towards* the camera
                d0 = self._dirs0[hit]
                b = geo.impact_parameter(cam.position, d0)
                h = np.linalg.norm(np.cross(cam.position[None, :], d0), axis=1)
                lz = -np.cross(cam.position[None, :], d0)[:, 2] * (b / h)
                g = redshift_g(r, lz, observer_r=cam.distance)
            else:
                g = np.ones_like(r)
            lut = _lut()
            color = lut(self.disk_temperature * T * g)
            intensity = g**4 * T**4 * disk_texture(r, phi, t)  # I_obs = g^4 I_emit
            img[hit] = color * intensity[:, None]

        return img.reshape(cam.height, cam.width, 3)

    def image(self, t: float = 0.0) -> np.ndarray:
        """Final 8-bit sRGB image (H, W, 3)."""
        hdr = self.hdr(t)
        s = self.supersample
        if s > 1:
            H, W = hdr.shape[0] // s, hdr.shape[1] // s
            hdr = hdr.reshape(H, s, W, s, 3).mean(axis=(1, 3))
        if self.bloom > 0:
            glow = np.maximum(hdr - 0.6, 0)
            sig = hdr.shape[0] / 90
            glow = sum(gaussian_filter(glow, sigma=(sig * k, sig * k, 0)) / k for k in (1, 3, 8))
            hdr = hdr + self.bloom * glow
        if self.contrast != 1.0:
            lum = hdr.sum(axis=-1, keepdims=True) / 3 + 1e-9
            hdr = hdr * lum ** (self.contrast - 1.0)
        if self.saturation != 1.0:
            lum = hdr.mean(axis=-1, keepdims=True)
            hdr = np.maximum(lum + self.saturation * (hdr - lum), 0.0)
        mapped = 1.0 - np.exp(-self.exposure * hdr)
        return (srgb_encode(mapped) * 255 + 0.5).astype(np.uint8)


_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        _LUT = BlackbodyLUT()
    return _LUT
