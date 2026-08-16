"""Record a demo video/GIF headlessly using the real visualizer (no GUI needed).

  .venv/bin/python tools/record_demo.py --out figures/demo
Produces figures/demo.mp4 and figures/demo.gif (requires ffmpeg).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

os.environ["SDL_VIDEODRIVER"] = "dummy"  # force headless even if the session sets wayland
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

from src.visualizer import App, main as _m  # noqa: E402,F401
import src.visualizer as V  # noqa: E402


def make_args(**kw):
    ap_defaults = dict(scenario="mission", fleet=4, seed=42, policy="P3", policy_b="P2", comm="degraded",
                       episode=None, width=1600, height=960, screenshot=None, sim_time=0, frames=0,
                       braess_moves=0, marginal=False, cycle=False, help_overlay=False, zoom=1.0, focus_boat=-1,
                       world="study", beliefs=-1)
    ap_defaults.update(kw)
    return argparse.Namespace(**ap_defaults)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/demo")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mission-seconds", type=float, default=24.0, help="video seconds of mission")
    ap.add_argument("--speed", type=int, default=1, help="speed index: 1=x1 2=x4 3=x16")
    ap.add_argument("--start", type=float, default=20.0, help="sim time to start recording mission")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    k = 0
    dt = 1.0 / a.fps

    def grab(app):
        nonlocal k
        pygame.image.save(app.screen, os.path.join(tmp, f"f{k:05d}.png"))
        k += 1

    # --- part 1: exact game, 5 best-response moves
    app = App(make_args(scenario="braess"))
    for move in range(4):
        for f in range(int(2.2 * a.fps)):
            if f == 0:
                app.braess.step()
            app.draw(dt, 0); grab(app)
    # --- part 2: mission
    app = App(make_args(scenario="mission", seed=a.seed))
    sim = app.views[0].sim
    while sim.t < a.start:
        sim.step()
        for g in app.views[0].ghosts:
            g[0].step()
    app.views[0].ev_idx = len(sim.events)
    app.speed_idx = a.speed
    app.views[0].follow_lerp = 0.02
    for f in range(int(a.mission_seconds * a.fps)):
        d = app.step_sims(dt)
        app.draw(dt, d); grab(app)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png",
                    "-vf", "scale=1280:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    f"{a.out}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{a.out}.mp4", "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer",
                    f"{a.out}.gif"], check=True)
    shutil.rmtree(tmp)
    print("wrote", f"{a.out}.mp4", f"{a.out}.gif", k, "frames")


if __name__ == "__main__":
    main()
