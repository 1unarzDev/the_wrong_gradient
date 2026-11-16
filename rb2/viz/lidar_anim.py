"""Animated lidar preview: a sweeping beam whose returns appear on real surfaces and fade.

  .venv/bin/python -m rb2.viz.lidar_anim     # writes figures/v2_preview/lidar_sweep.{mp4,gif}
"""
import math
import os
import shutil
import subprocess
import tempfile

os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from ..generate import GenConfig, generate_course  # noqa: E402
from . import style as S  # noqa: E402
from .gallery import CAM_FOV, View, _obstacles, draw_course  # noqa: E402

OUT = "figures/v2_preview"
REV_S = 1.6           # seconds per sweep (slowed down from a real 10 Hz unit so the eye can follow it)
FADE_S = 1.3          # how long a return stays visible
RMAX = 20.0
FPS = 30


def ray(circ, segs, bx, by, a, rmax=RMAX):
    dx, dy = math.cos(a), math.sin(a)
    best = None
    for (ox, oy, r, o) in circ:
        fx, fy = ox - bx, oy - by
        t = fx * dx + fy * dy
        if t <= 0 or (best is not None and t - r > best):
            continue
        d2 = fx * fx + fy * fy - t * t
        if d2 <= r * r:
            tt = t - math.sqrt(r * r - d2)
            if 0 < tt < rmax and (best is None or tt < best):
                best = tt
    for (p, q, _o) in segs:
        ex, ey = q[0] - p[0], q[1] - p[1]
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        wx, wy = p[0] - bx, p[1] - by
        t = (wx * ey - wy * ex) / den
        u = (wx * dy - wy * dx) / den
        if 0 < t < rmax and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return best


def main(seconds=4.8):
    pygame.init()
    c = generate_course(3, GenConfig(difficulty="hard"))
    m = c.module("T3")[0]
    bay = m.bays[len(m.bays) // 2]
    W, H = 1100, 700
    surf = pygame.Surface((W, H))
    rect = pygame.Rect(0, 0, W, H)
    ux, uy = math.cos(m.heading), math.sin(m.heading)            # out of the dock, into the basin
    cx, cy = m.x + ux * 7.5, m.y + uy * 7.5
    v = View(rect, cx - 13.0, cy - 8.3, cx + 13.0, cy + 8.3)
    circ, segs = _obstacles(c)
    rng = np.random.default_rng(1)
    points = []                  # (x, y, t_birth)
    tmp = tempfile.mkdtemp()
    n_frames = int(seconds * FPS)
    dang = 2 * math.pi / (REV_S * FPS)
    res = math.radians(0.35)
    sweep = 0.0
    for f in range(n_frames):
        t = f / FPS
        # boat creeps toward the bay entry (into the dock)
        d = 9.0 - 4.0 * t / seconds
        bx, by = bay.entry[0] + ux * (d - 2.6), bay.entry[1] + uy * (d - 2.6)
        psi = math.atan2(-uy, -ux)
        # cast the rays covered by this frame's slice of the sweep
        a = sweep
        while a < sweep + dang:
            r = ray(circ, segs, bx, by, a)
            if r is not None:
                rr = r + rng.normal(0, 0.03)
                points.append((bx + rr * math.cos(a), by + rr * math.sin(a), t))
            a += res
        sweep = (sweep + dang) % (2 * math.pi)
        points = [q for q in points if t - q[2] < FADE_S]
        # draw
        surf.fill(S.PAPER)
        draw_course(surf, v, c)
        lay = pygame.Surface((W, H), pygame.SRCALPHA)
        C = v.p(bx, by)
        S.camera_cone(lay, C, psi, 18 * v.s, CAM_FOV)
        arc = [(C[0] + 18 * v.s * math.cos(q), C[1] - 18 * v.s * math.sin(q)) for q in np.linspace(psi - CAM_FOV / 2, psi + CAM_FOV / 2, 40)]
        # trailing beam: a thin wedge that fades behind the sweep line
        R = RMAX * v.s
        for k in range(16):
            a1 = sweep - k * math.radians(2.0)
            al = int(46 * (1 - k / 16) ** 2)
            S.sector(lay, (*S.SENSE, al), C, 0, R, a1 - math.radians(2.0), a1, 3)
        e = (C[0] + R * math.cos(sweep), C[1] - R * math.sin(sweep))
        S.aa_line(lay, (*S.SENSE, 150), C, e, 1)
        for (x, y, tb) in points:
            age = (t - tb) / FADE_S
            al = int(255 * (1 - age) ** 1.5)
            X, Y = v.p(x, y)
            S.aa_circle(lay, (*S.SENSE, al), (X, Y), 2.2 if age < 0.12 else 1.5)
        surf.blit(lay, (0, 0))
        S.boat(surf, C, psi, 0, 44, focus=True)
        cap = S.font(13).render("Lidar: a sweeping beam; each return appears where it touches a surface, then fades. "
                                "Positions only, no colour; colour comes from the camera.", True, S.INK2)
        bg = pygame.Surface((cap.get_width() + 16, cap.get_height() + 8), pygame.SRCALPHA); bg.fill((*S.CHIP_BG, 235))
        surf.blit(bg, (16, H - cap.get_height() - 24)); surf.blit(cap, (24, H - cap.get_height() - 20))
        pygame.image.save(surf, os.path.join(tmp, f"f{f:04d}.png"))
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", f"{OUT}/lidar_sweep.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/lidar_sweep.mp4", "-vf",
                    "fps=20,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                    f"{OUT}/lidar_sweep.gif"], check=True)
    shutil.rmtree(tmp)
    print("wrote lidar_sweep.mp4/gif")


if __name__ == "__main__":
    main()
