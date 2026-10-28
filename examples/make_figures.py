"""Regenerate every image in the README:  python examples/make_figures.py [--quick]"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from blackhole import Camera, Scene
from blackhole import geodesic as geo

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")


def save(img, name):
    path = os.path.join(OUT, name)
    Image.fromarray(img).save(path)
    print("wrote", path)


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((10, 8), text, fill=(235, 235, 235))
    return np.asarray(im)


def hero(scale):
    s = Scene(camera=Camera(int(1600 * scale), int(900 * scale), inclination=82, fov=38))
    save(s.image(), "hero.png")


def spin_gif(scale, frames=60):
    s = Scene(camera=Camera(int(640 * scale), int(360 * scale), inclination=80, fov=40))
    s.trace()
    ims = [Image.fromarray(s.image(t)) for t in np.linspace(0, 160, frames, endpoint=False)]
    path = os.path.join(OUT, "spin.gif")
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=60, loop=0, optimize=True)
    print("wrote", path)


def anatomy(scale):
    w, h = int(480 * scale), int(270 * scale)
    cam = dict(width=w, height=h, inclination=82)
    a = Scene(camera=Camera(**cam), show_disk=False, sky="checker", bloom=0, contrast=1, saturation=1, exposure=2.5)
    b = Scene(camera=Camera(**cam), relativistic_shading=False)
    c = Scene(camera=Camera(**cam))
    row = np.concatenate(
        [
            label(a.image(), "1. lensing only (checkerboard sky)"),
            label(b.image(), "2. + disk, no redshift"),
            label(c.image(), "3. + gravitational & Doppler shift"),
        ],
        axis=1,
    )
    save(row, "anatomy.png")


def inclinations(scale):
    w, h = int(400 * scale), int(225 * scale)
    tiles = [label(Scene(camera=Camera(w, h, inclination=i, fov=42)).image(), f"i = {i} deg") for i in (20, 60, 80, 89)]
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    save(grid, "inclinations.png")


def trajectories():
    bs = np.concatenate([np.linspace(1.0, geo.B_CRIT - 0.02, 6), [geo.B_CRIT * 1.0005], np.linspace(2.75, 7, 7)])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0b0d12")
    ax.set_facecolor("#0b0d12")
    cmap = plt.get_cmap("plasma")
    for i, b in enumerate(bs):
        res = geo.trace(np.array([-12.0, b, 0.0]), np.array([[1.0, 0.0, 0.0]]), disk=None, step=0.01, r_escape=14, record=True)
        p = res.path[:, 0, :]
        crit = abs(b - geo.B_CRIT) < 0.01
        ax.plot(p[:, 0], p[:, 1], lw=2.2 if crit else 1.1, color="#ffffff" if crit else cmap(0.15 + 0.75 * i / len(bs)))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.fill(np.cos(t), np.sin(t), color="black", zorder=3)
    ax.plot(np.cos(t), np.sin(t), color="#666", lw=1, zorder=4)
    ax.plot(1.5 * np.cos(t), 1.5 * np.sin(t), "--", color="#8ab4ff", lw=1, label="photon sphere  r = 1.5 r$_s$")
    ax.plot(3 * np.cos(t), 3 * np.sin(t), ":", color="#ffb86b", lw=1, label="ISCO  r = 3 r$_s$")
    ax.set_xlim(-12, 8)
    ax.set_ylim(-6.5, 8)
    ax.set_aspect("equal")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.set_xlabel("x / r$_s$", color="#ccc")
    ax.set_ylabel("y / r$_s$", color="#ccc")
    ax.set_title("Light rays near a Schwarzschild black hole (white: b ≈ 3√3 M)", color="#eee")
    leg = ax.legend(facecolor="#14171f", edgecolor="#333", loc="upper right")
    for txt in leg.get_texts():
        txt.set_color("#ddd")
    fig.tight_layout()
    path = os.path.join(OUT, "trajectories.png")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quarter-size images for a fast check")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    scale = 0.5 if args.quick else 1.0
    jobs = {"trajectories": trajectories, "anatomy": lambda: anatomy(scale), "inclinations": lambda: inclinations(scale),
            "spin": lambda: spin_gif(scale), "hero": lambda: hero(scale)}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        t = time.time()
        fn()
        print(f"  {name}: {time.time() - t:.0f}s")
