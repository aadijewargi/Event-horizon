"""Validate the ray tracer against analytic general relativity."""
import numpy as np
import pytest

from blackhole import geodesic as geo
from blackhole.camera import Camera
from blackhole.color import blackbody_rgb
from blackhole.disk import redshift_g, temperature_profile

FAR = 2000.0


def _shoot(b, step=0.01, **kw):
    """Rays coming in from x = -FAR travelling +x with impact parameter(s) b."""
    b = np.atleast_1d(b).astype(float)
    x0 = np.stack([np.full_like(b, -FAR), b, np.zeros_like(b)], axis=1)
    d0 = np.tile([1.0, 0.0, 0.0], (len(b), 1))
    # trace() takes one start point; shoot each ray separately (only a few)
    return [geo.trace(x0[i], d0[i : i + 1], disk=None, step=step, r_escape=2.5 * FAR, **kw) for i in range(len(b))]


@pytest.mark.parametrize("eps", [1e-3, 1e-2])
def test_photon_sphere_capture_threshold(eps):
    """Rays just inside b_c = 3*sqrt(3) M fall in; just outside escape."""
    inside, outside = _shoot([geo.B_CRIT * (1 - eps), geo.B_CRIT * (1 + eps)])
    assert inside.status[0] == geo.CAPTURED
    assert outside.status[0] == geo.ESCAPED
    # the escaping ray skims the photon sphere (r = 3M = 1.5 r_s); near b_c the
    # turning point obeys b = r0 / sqrt(1 - 1/r0) with r0 -> 1.5 as eps -> 0
    r0 = outside.min_radius[0]
    assert geo.PHOTON_SPHERE < r0 < geo.PHOTON_SPHERE * (1 + 3 * np.sqrt(eps))
    assert r0 / np.sqrt(1 - 1 / r0) == pytest.approx(geo.B_CRIT * (1 + eps), rel=1e-3)


@pytest.mark.parametrize("b", [2.8, 4.0, 8.0, 20.0])
def test_closest_approach(b):
    """Turning point r0 satisfies b = r0 / sqrt(1 - r_s / r0)."""
    (res,) = _shoot(b)
    r0 = res.min_radius[0]
    assert r0 / np.sqrt(1 - 1 / r0) == pytest.approx(b, rel=2e-3)


@pytest.mark.parametrize("b", [50.0, 100.0, 200.0])
def test_weak_field_deflection(b):
    """Light bending: alpha = 4M/b + (15 pi / 4) M^2 / b^2 + ... (Einstein's 4GM/c^2 b)."""
    (res,) = _shoot(b, step=0.02)
    d = res.direction[0]
    alpha = np.arccos(np.clip(d[0], -1, 1))
    M = geo.M
    expected = 4 * M / b + 15 * np.pi / 4 * M**2 / b**2
    assert alpha == pytest.approx(expected, rel=5e-3)
    assert d[1] < 0  # bent *towards* the hole


def test_conserved_quantities_along_ray():
    """h = |x × v| and E = v²/2 - h²/(2 r³) are constants of the effective motion,
    so their drift measures the RK4 integration error."""
    x = np.array([[-60.0, 3.0, 0.0]])
    v = np.array([[1.0, 0.0, 0.0]])
    h2 = np.sum(np.cross(x, v) ** 2, axis=1)

    def energy(x, v):
        return 0.5 * np.sum(v**2) - 0.5 * h2[0] / np.linalg.norm(x) ** 3

    E0, min_r = energy(x, v), np.inf
    while np.linalg.norm(x) < 200:
        r = np.linalg.norm(x)
        min_r = min(min_r, r)
        x, v = geo.rk4_step(x, v, h2, np.array([0.02 * r]))
    assert min_r < 3.0  # the ray really did pass through the strong field
    assert abs(x[0, 2]) < 1e-12  # motion stays in the initial plane
    assert np.linalg.norm(np.cross(x, v)) == pytest.approx(np.sqrt(h2[0]), rel=1e-6)
    assert energy(x, v) == pytest.approx(E0, rel=1e-6)


def test_shadow_size_in_image():
    """The black hole's shadow edge sits where b = b_c, seen from distance D."""
    cam = Camera(width=601, height=1, fov=40.0, distance=30.0, inclination=90.0)
    dirs = cam.rays()
    res = geo.trace(cam.position, dirs, disk=None, step=0.02)
    captured = res.status == geo.CAPTURED
    # angle of each pixel from the optical axis
    fwd = -cam.position / np.linalg.norm(cam.position)
    ang = np.arccos(np.clip(dirs @ fwd, -1, 1))
    measured = ang[captured].max()
    # 1/b^2 = 1/(D sin a)^2 - 1/D^3   =>   sin a = b / (D sqrt(1 + b^2 / D^3))
    D, b = cam.distance, geo.B_CRIT
    expected = np.arcsin(b / (D * np.sqrt(1 + b**2 / D**3)))
    spacing = np.diff(ang[300:302])[0]
    assert measured == pytest.approx(expected, abs=1.5 * spacing)


def test_impact_parameter_far_limit():
    x0 = np.array([-1e5, 0.0, 0.0])
    d = np.array([[1.0, 1e-4, 0.0]])
    d /= np.linalg.norm(d)
    assert geo.impact_parameter(x0, d)[0] == pytest.approx(10.0, rel=1e-6)


def test_redshift_face_on_is_pure_gravitational():
    """With L_z = 0 only gravity + time dilation remain: g = sqrt(1 - 3M/r)."""
    r = np.array([3.0, 6.0, 20.0])
    g = redshift_g(r, np.zeros_like(r))
    assert np.allclose(g, np.sqrt(1 - 1.5 / r))


def test_redshift_doppler_asymmetry():
    r = np.array([6.0, 6.0])
    g = redshift_g(r, np.array([+6.0, -6.0]))  # approaching vs receding gas
    assert g[0] > 1.0 > g[1]


def test_disk_temperature_peak():
    r = np.linspace(3.0, 30.0, 20001)
    T = temperature_profile(r, 3.0)
    assert r[np.argmax(T)] == pytest.approx(49 / 36 * 3.0, rel=1e-3)
    assert T.max() == pytest.approx(1.0)
    assert T[0] == 0.0


def test_blackbody_colours():
    cool, sun, hot = blackbody_rgb([2000.0, 5800.0, 20000.0])
    assert cool[0] > cool[2]  # red-orange
    assert hot[2] > hot[0]  # blue-white
    assert np.ptp(sun) < 0.25  # roughly white
