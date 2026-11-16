"""Three reserved lidar treatments on the same scene, real 10 Hz timing, no beam.

  .venv/bin/python -m rb2.viz.lidar_variants   # figures/v2_preview/lidar_variants.mp4
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
from .lidar_anim import ray  # noqa: E402

OUT = "figures/v2_preview"
FPS, HZ = 30, 10.0
VARIANTS = [("A  live scan", dict(fade=0.33, r=1.3, alpha=200, contour=False)),
            ("B  short memory", dict(fade=1.5, r=1.1, alpha=130, contour=False)),
            ("C  contour", dict(fade=0.0, r=0.0, alpha=0, contour=True))]


def main(seconds=4.0):
    pygame.init()
    c = generate_course(3, GenConfig(difficulty="hard"))
    m = c.module("T3")[0]
    bay = m.bays[len(m.bays) // 2]
    ux, uy = math.cos(m.heading), math.sin(m.heading)
    cx, cy = m.x + ux * 7.0, m.y + uy * 7.0
    PW, PH, GAP = 600, 560, 12
    W, H = PW * 3 + GAP * 4, PH + 70
    surf = pygame.Surface((W, H))
    circ, segs = _obstacles(c)
    rng = np.random.default_rng(2)
    clouds = [[] for _ in VARIANTS]
    tmp = tempfile.mkdtemp()
    res = math.radians(0.5)
    rot_per_frame = 2 * math.pi * HZ / FPS            # a third of a turn per frame at 10 Hz / 30 fps
    sweep = 0.0
    for f in range(int(seconds * FPS)):
        t = f / FPS
        d = 8.5 - 3.5 * t / seconds
        bx, by = bay.entry[0] + ux * (d - 2.6), bay.entry[1] + uy * (d - 2.6)
        psi = math.atan2(-uy, -ux)
        fresh, full = [], []
        a = sweep
        while a < sweep + rot_per_frame:
            r = ray(circ, segs, bx, by, a)
            if r is not None:
                rr = r + rng.normal(0, 0.03)
                fresh.append((bx + rr * math.cos(a), by + rr * math.sin(a), t))
            a += res
        sweep = (sweep + rot_per_frame) % (2 * math.pi)
        surf.fill(S.PAPER)
        for k, (name, P) in enumerate(VARIANTS):
            rect = pygame.Rect(GAP + k * (PW + GAP), 50, PW, PH)
            v = View(rect, cx - 9.5, cy - 8.9, cx + 9.5, cy + 8.9)
            surf.set_clip(rect)
            draw_course(surf, v, c)
            lay = pygame.Surface((W, H), pygame.SRCALPHA)
            C = v.p(bx, by)
            S.camera_cone(lay, C, psi, 18 * v.s, CAM_FOV)
            if P["contour"]:
                n = 240
                pts = []
                for j in range(n):
                    aa = 2 * math.pi * j / n
                    r = ray(circ, segs, bx, by, aa) or 20.0
                    pts.append(v.p(bx + r * math.cos(aa), by + r * math.sin(aa)))
                S.aa_polyline(lay, (*S.SENSE, 120), S.chaikin(pts, 1, closed=True), 1.0, closed=True)
            else:
                clouds[k] = [q for q in clouds[k] + fresh if t - q[2] < P["fade"]]
                for (x, y, tb) in clouds[k]:
                    al = int(P["alpha"] * (1 - (t - tb) / P["fade"]) ** 1.2)
                    S.aa_circle(lay, (*S.SENSE, al), v.p(x, y), P["r"])
            surf.blit(lay, (0, 0))
            S.boat(surf, C, psi, 0, 40, focus=True)
            surf.set_clip(None)
            pygame.draw.rect(surf, S.FRAME, rect, 1)
            surf.blit(S.font(15, True).render(name, True, S.INK1), (rect.x, 18))
        surf.blit(S.font(12).render("10 Hz scan, no beam drawn. Pick the treatment that feels right.", True, S.INK3),
                  (GAP, H - 16 - 12))
        pygame.image.save(surf, os.path.join(tmp, f"f{f:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", f"{OUT}/lidar_variants.mp4"], check=True)
    shutil.rmtree(tmp)
    print("wrote lidar_variants.mp4")


if __name__ == "__main__":
    main()
