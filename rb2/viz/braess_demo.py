"""Braess return-rush replay: the same four boats with the shortcut closed (left) and open (right), each on the
equilibrium routes of the calibrated game, simulated with the full boat dynamics.

  .venv/bin/python -m rb2.viz.braess_demo --course queue      # single-file channels: the paradox vanishes
  .venv/bin/python -m rb2.viz.braess_demo --course slowzone   # slow zones: the paradox appears
  add --record figures/braess_replay to write an mp4 (headless)
Keys: Space pause, 1/2/3 speed, R restart, Esc quit.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shutil
import subprocess
import tempfile

import pygame

from ..braess2 import RushGeom, get_course, run_scenario
from . import style as S
from .gallery import View, draw_course

GEOMS = {
    "queue": (RushGeom(lane_len=6.0, lane_len_e=10.0, v_open=0.4), ["west", "west", "east", "east"],
              ["shortcut"] * 4, "single-file channels"),
    "slowzone": (RushGeom(congestion="slow_zone", alpha=1.0, v_open=0.3), ["west", "west", "east", "east"],
                 ["east", "shortcut", "shortcut", "shortcut"], "slow zones"),
}
ROUTE_LABEL = {"west": "west channel", "east": "east channel", "shortcut": "shortcut"}


class Replay:
    def __init__(self, geom, routes, seed, shortcut_open):
        self.rc = get_course(geom)
        self.routes = routes
        self.open = shortcut_open
        r = run_scenario(geom, routes, seed, shortcut_open, record=True, t_max=600.0)
        self.traj = r["traj"]
        self.times = r["times"]
        self.t_end = max(t for t in self.times if math.isfinite(t)) + 3.0

    def at(self, t):
        """Interpolated positions and headings at time t."""
        tr = self.traj
        k = min(range(len(tr)), key=lambda q: abs(tr[q][0] - t))
        k = max(1, min(k, len(tr) - 1))
        (t0, x0, y0, a0), (t1, x1, y1, a1) = tr[k - 1], tr[k]
        f = 0.0 if t1 == t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        out = []
        for i in range(len(x0)):
            x, y = x0[i] + f * (x1[i] - x0[i]), y0[i] + f * (y1[i] - y0[i])
            out.append((x, y, math.atan2(y1[i] - y0[i], x1[i] - x0[i]) if (x1[i], y1[i]) != (x0[i], y0[i]) else None))
        return out


class Demo:
    def __init__(self, a):
        self.a = a
        geom, closed, opened, self.label = GEOMS[a.course]
        self.left = Replay(geom, closed, a.seed, False)
        self.right = Replay(geom, opened, a.seed, True)
        pygame.init()
        self.screen = pygame.display.set_mode((a.width, a.height))
        pygame.display.set_caption("Braess return rush")
        self.t, self.speed, self.paused = 0.0, 4.0, False
        self.psi = [[math.pi / 2] * 4, [math.pi / 2] * 4]

    def panel(self, rect, rp, side, title):
        rc, c = rp.rc, rp.rc.course
        c.shortcut_open = rp.open
        xs = [p[0] for p in rc.nodes.values()] + [rc.x0, rc.x1]
        ys = [p[1] for p in rc.nodes.values()] + [rc.y0, rc.y1]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs) + 14, (max(ys) - min(ys) + 14) * rect.w / rect.h)
        hw, hh = span / 2, span / 2 * rect.h / rect.w
        v = View(rect, cx - hw, cy - hh, cx + hw, cy + hh)
        self.screen.set_clip(rect)
        draw_course(self.screen, v, c)
        lay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for name, (boxes, alpha) in rc.czones.items():          # slow zones
            for (xa, xb, ya, yb) in boxes:
                p0, p1 = v.p(xa, yb), v.p(xb, ya)
                pygame.draw.rect(lay, (214, 168, 74, 30), pygame.Rect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]))
        for name, (xa, xb, ya, yb) in rc.boxes.items():          # single-file channels
            p0, p1 = v.p(xa, yb), v.p(xb, ya)
            pygame.draw.rect(lay, (214, 168, 74, 30), pygame.Rect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]))
        self.screen.blit(lay, (0, 0))
        for n_, (x, y) in rc.nodes.items():
            P = v.p(x, y)
            S.aa_circle(self.screen, S.PAPER, P, 9, S.INK2)
            img = S.font(12, True).render(n_, True, S.INK1)
            self.screen.blit(img, (P[0] - img.get_width() / 2, P[1] - img.get_height() / 2))
        for i, (x, y, h) in enumerate(rp.at(self.t)):
            if h is not None:
                self.psi[side][i] = h
            done = math.isfinite(rp.times[i]) and self.t >= rp.times[i]
            S.boat(self.screen, v.p(x, y), self.psi[side][i], i, max(20, 1.8 * v.s * 1.2), focus=not done)
        self.screen.set_clip(None)
        f = S.font
        self.screen.blit(f(17, True).render(title, True, S.INK1), (rect.x + 14, rect.y - 56))
        fleet = sum(min(self.t, tt) if math.isfinite(tt) else self.t for tt in rp.times)
        home = sum(1 for tt in rp.times if math.isfinite(tt) and self.t >= tt)
        txt = f"fleet time {fleet:6.1f} s     {home}/4 home"
        self.screen.blit(f(14).render(txt, True, S.INK2), (rect.x + 14, rect.y - 30))
        routes = ", ".join(f"{sum(r == q for r in rp.routes)} {ROUTE_LABEL[q]}" for q in ("west", "east", "shortcut")
                           if any(r == q for r in rp.routes))
        self.screen.blit(f(12).render(routes, True, S.INK3), (rect.x + 14, rect.bottom + 8))

    def draw(self):
        w, h = self.screen.get_size()
        self.screen.fill(S.PAPER)
        f = S.font
        self.screen.blit(f(16, True).render(f"Return rush with {self.label}: does opening the shortcut slow the fleet?",
                                            True, S.INK1), (24, 16))
        self.screen.blit(f(13).render("Same boats, same seed, full boat dynamics. Routes are the equilibrium of the "
                                      "calibrated congestion game. Shaded: congested legs.", True, S.INK3), (24, 42))
        pw = (w - 72) // 2
        top = 140
        self.panel(pygame.Rect(24, top, pw, h - top - 70), self.left, 0, "Shortcut closed")
        self.panel(pygame.Rect(48 + pw, top, pw, h - top - 70), self.right, 1, "Shortcut open")
        if self.t >= max(self.left.t_end, self.right.t_end) - 3:
            a = sum(self.left.times); b = sum(self.right.times)
            msg = f"Opening the shortcut changed fleet time by {b - a:+.1f} s"
            img = f(16, True).render(msg, True, S.INK1)
            self.screen.blit(img, (w // 2 - img.get_width() // 2, h - 36))

    def step(self, dt):
        if not self.paused:
            self.t = min(self.t + self.speed * dt, max(self.left.t_end, self.right.t_end))

    def run(self):
        if self.a.record:
            return self.record()
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(60) / 1000
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit(); return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE: self.paused = not self.paused
                    elif ev.key == pygame.K_1: self.speed = 1.0
                    elif ev.key == pygame.K_2: self.speed = 4.0
                    elif ev.key == pygame.K_3: self.speed = 10.0
                    elif ev.key == pygame.K_r: self.t = 0.0
            self.step(dt)
            self.draw()
            pygame.display.flip()

    def record(self):
        tmp, fps = tempfile.mkdtemp(), 30
        end = max(self.left.t_end, self.right.t_end)
        n = int(end / self.speed * fps) + 2 * fps
        for k in range(n):
            self.step(1 / fps); self.draw()
            pygame.image.save(self.screen, os.path.join(tmp, f"f{k:05d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%05d.png",
                        "-vf", "scale=1440:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        f"{self.a.record}.mp4"], check=True)
        shutil.rmtree(tmp)
        print("wrote", f"{self.a.record}.mp4")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", choices=list(GEOMS), default="slowzone")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--record", default=None)
    a = ap.parse_args(argv)
    if a.record:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    Demo(a).run()


if __name__ == "__main__":
    main()
