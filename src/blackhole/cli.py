"""Command line interface:  python -m blackhole render|animate ..."""
from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from .camera import Camera
from .render import Scene


def _scene(a) -> Scene:
    cam = Camera(a.width, a.height, a.fov, a.distance, a.inclination, a.azimuth, a.roll)
    return Scene(
        camera=cam,
        disk_inner=a.disk_inner,
        disk_outer=a.disk_outer,
        disk_temperature=a.temperature,
        sky=a.sky,
        relativistic_shading=not a.no_doppler,
        show_disk=not a.no_disk,
        supersample=a.ss,
        exposure=a.exposure,
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="blackhole", description="Schwarzschild black hole ray tracer")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("render", "animate"):
        s = sub.add_parser(name)
        s.add_argument("-o", "--output", default=f"blackhole.{'png' if name == 'render' else 'gif'}")
        s.add_argument("--width", type=int, default=960)
        s.add_argument("--height", type=int, default=540)
        s.add_argument("--fov", type=float, default=40.0, help="vertical field of view (deg)")
        s.add_argument("--distance", type=float, default=30.0, help="camera distance (r_s)")
        s.add_argument("--inclination", type=float, default=82.0, help="deg from disk axis; 90 = edge-on")
        s.add_argument("--azimuth", type=float, default=0.0)
        s.add_argument("--roll", type=float, default=0.0)
        s.add_argument("--disk-inner", type=float, default=3.0)
        s.add_argument("--disk-outer", type=float, default=12.0)
        s.add_argument("--temperature", type=float, default=4800.0, help="peak disk temperature (K)")
        s.add_argument("--sky", choices=["stars", "checker", "none"], default="stars")
        s.add_argument("--no-doppler", action="store_true", help="disable redshift/beaming")
        s.add_argument("--no-disk", action="store_true")
        s.add_argument("--ss", type=int, default=2, help="supersampling factor")
        s.add_argument("--exposure", type=float, default=0.9)
        if name == "animate":
            s.add_argument("--frames", type=int, default=60)
            s.add_argument("--duration", type=float, default=150.0, help="simulated time span (r_s/c)")
            s.add_argument("--fps", type=int, default=20)
    a = p.parse_args(argv)

    scene = _scene(a)
    t0 = time.time()
    scene.trace()
    print(f"traced {a.width * a.height * a.ss**2:,} rays in {time.time() - t0:.1f}s")
    if a.cmd == "render":
        Image.fromarray(scene.image()).save(a.output)
    else:
        frames = [Image.fromarray(scene.image(t)) for t in np.linspace(0, a.duration, a.frames, endpoint=False)]
        frames[0].save(a.output, save_all=True, append_images=frames[1:], duration=1000 // a.fps, loop=0, optimize=True)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
