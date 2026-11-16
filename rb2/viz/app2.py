"""Course v2 game front end (restrained 'learning simulation' style).

  .venv/bin/python -m rb2.viz.app2 --seed 11 --policy P3
  .venv/bin/python -m rb2.viz.app2 --variant return_rush --shortcut
Keys: Space pause, 1/2/3 speed x1/x2/x4, Tab focus next boat, G cycle the phantom fleet shown, K phantoms on/off,
      P planned paths, L lidar, C camera cone, D detections, F follow camera / whole course, R new course.
Headless video: .venv/bin/python -m rb2.viz.app2 --record figures/v2_demo --seconds 60
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pygame

from ..config2 import V2Config
from ..course import TASK_NAMES
from ..sim2 import Simulation2
from . import style as S
from .gallery import CAM_FOV, View, draw_course, raycast, _obstacles
from .lidar_anim import ray

import json

PANEL_W, TOP_H = 440, 48
PHANTOMS = ["P1", "P2", "P3", "P4"]          # every other strategy runs live on the same course (same seed)
NAME = {"P0": "Greedy", "P1": "Selfish", "P2": "Team-aware, simple ETA", "A1": "Team-aware, age-aware",
        "A2": "Team-aware, dynamics ETA", "P3": "Team-aware", "P4": "Central planner", "P5": "Leader on radio",
        "R3": "Reservations", "L3": "Heartbeat lease"}
HOW = {"P0": "Each boat drives to the nearest task. No coordination.",
       "P1": "Each boat takes the task that is best for itself (points minus travel time) and ignores teammates.",
       "P2": "Each boat takes the task that adds most to the team, given its teammates' radio claims.",
       "A1": "Team-aware bids; discounts stale radio messages.",
       "A2": "Team-aware bids with travel times from a boat-dynamics model.",
       "P3": "Each boat bids for the task that adds most to the team, using radio claims (stale ones discounted) "
             "and travel times from a boat-dynamics model.",
       "P4": "One planner sees everything and assigns every boat at once (an optimal assignment, re-solved every 2 s).",
       "P5": "A leader boat runs the planner on what it hears and radios the assignments.",
       "R3": "Team-aware bids; a teammate's claim is trusted until its announced arrival, then expires.",
       "L3": "Team-aware bids; a teammate's claim is trusted while the teammate keeps broadcasting it."}
TASK_MAX = {"T1": 200, "T2": 650, "T3": 400, "T4": 650, "T5": 0, "T6": 100}


_FRIENDLY = [("T3:look", "the dock banner scan"), ("T3:bay0", "dock bay 1"), ("T3:bay1", "dock bay 2"),
             ("T3:bay2", "dock bay 3"), ("T3:bay3", "dock bay 4"), ("T2", "the path"), ("T4", "the speed challenge"),
             ("T6", "the trip home"), ("T1", "the channel")]


def friendly(txt):
    import re
    txt = re.sub(r"T5:\d+", "a rescue vessel", txt)
    for k, v in _FRIENDLY:
        txt = txt.replace(k + ":", v + ":").replace(k + " ", v + " ").replace(k + ",", v + ",")
        if txt.endswith(" " + k):
            txt = txt[: -len(k)] + v
    return txt


def carrot(g, x, y, look):
    """The LOS controller's lookahead point (same geometry as src.navigation.los_command, read-only)."""
    wps = g.wps
    if len(wps) < 2:
        return None
    k = min(g.seg, len(wps) - 2)
    ax, ay = wps[k]; bx, by = wps[k + 1]
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy) + 1e-9
    t = max(0.0, min(1.0, ((x - ax) * sx + (y - ay) * sy) / (L * L)))
    rem, cur = look, (1 - t) * L
    px, py = ax + t * sx, ay + t * sy
    while rem > cur and k < len(wps) - 2:
        rem -= cur; k += 1
        px, py = wps[k]
        cur = math.hypot(wps[k + 1][0] - px, wps[k + 1][1] - py)
    ex, ey = wps[k + 1]
    LL = math.hypot(ex - px, ey - py) + 1e-9
    d = min(rem, LL)
    return (px + d * (ex - px) / LL, py + d * (ey - py) / LL), (ax + t * sx, ay + t * sy)


class Cam:
    """World->screen with zoom and smooth follow."""

    def __init__(self, rect, course):
        self.rect, self.c = pygame.Rect(rect), course
        self.fit = min((self.rect.w - 30) / course.width, (self.rect.h - 30) / course.height)
        self.zoom, self.cx, self.cy = 1.0, course.width / 2, course.height / 2

    @property
    def s(self):
        return self.fit * self.zoom

    def view(self):
        hw, hh = self.rect.w / 2 / self.s, self.rect.h / 2 / self.s
        return View(self.rect, self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)

    def follow(self, sim, zoom=1.35, lerp=0.03, focus=None):
        if focus is not None:
            x, y = sim.pos(focus)
            xs, ys = [x], [y]
            cl = sim.claim[focus]
            if cl is not None:
                xs.append(cl.xy[0]); ys.append(cl.xy[1])
            for j in range(sim.n):                      # keep nearby teammates in frame
                if math.hypot(sim.S[0, j] - x, sim.S[1, j] - y) < 22:
                    xs.append(sim.S[0, j]); ys.append(sim.S[1, j])
        else:
            xs, ys = list(sim.S[0]), list(sim.S[1])
        tx, ty = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        bw, bh = max(xs) - min(xs) + 24, max(ys) - min(ys) + 24
        z = min(zoom, max(1.0, min(self.rect.w / (bw * self.fit), self.rect.h / (bh * self.fit))))
        self.zoom += (z - self.zoom) * lerp
        self.cx += (tx - self.cx) * lerp; self.cy += (ty - self.cy) * lerp
        W, H = self.c.width, self.c.height
        self.zoom = max(self.zoom, max(self.rect.w / (W * self.fit), self.rect.h / (H * self.fit)) * 1.0)
        hw, hh = self.rect.w / 2 / self.s, self.rect.h / 2 / self.s
        self.cx = min(max(self.cx, hw), W - hw) if 2 * hw < W else W / 2
        self.cy = min(max(self.cy, hh), H - hh) if 2 * hh < H else H / 2


class Game:
    def __init__(self, args):
        self.args = args
        pygame.init()
        flags = 0 if args.record else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((args.width, args.height), flags)
        pygame.display.set_caption("RoboBoat learning simulation")
        self.clock = pygame.time.Clock()
        self.speed, self.paused = 1.0, False
        self.focus, self.show_phantoms, self.show_lidar, self.show_cones, self.show_det = 0, True, True, True, True
        self.follow, self.show_paths, self.ghost_sel = True, True, 0
        try:
            self.board = json.load(open("data/v2_replication/leaderboard.json"))
        except Exception:
            self.board = {}
        self.seed = args.seed
        self.build()

    def cfg(self, policy):
        return V2Config(n_boats=self.args.fleet, policy=policy, comm_mode=self.args.comm, variant=self.args.variant,
                        shortcut_open=self.args.shortcut, perception=self.args.perception, episode_s=self.args.episode)

    def build(self):
        self.sim = Simulation2(self.cfg(self.args.policy), self.seed)
        self.ghosts = [(Simulation2(self.cfg(p), self.seed), p) for p in PHANTOMS if p != self.args.policy]
        self.ghost_sel = min(self.ghost_sel, max(0, len(self.ghosts) - 1))
        if getattr(self.args, "ghost", None):
            self.ghost_sel = next((k for k, (_, p) in enumerate(self.ghosts) if p == self.args.ghost), self.ghost_sel)
        self.sev_idx = 0
        w, h = self.screen.get_size()
        self.cam = Cam(pygame.Rect(0, TOP_H, w - PANEL_W, h - TOP_H), self.sim.course)
        self.lidar_pts = []
        self.ev_idx = 0
        self.popups = []
        self.sweep = 0.0
        self.obst = _obstacles(self.sim.course)

    # ---------------------------------------------------------------------------------------
    def step(self, real_dt):
        if self.paused:
            return 0.0
        self.acc = getattr(self, "acc", 0.0) + self.speed * real_dt
        before = self.sim.t
        while self.acc >= self.sim.dt - 1e-9 and not self.sim.done_flag:
            self.sim.step()
            for g, _ in self.ghosts:
                g.step()
            self.acc -= self.sim.dt
        return self.sim.t - before

    def update_lidar(self, sim_dt):
        """Live scan (variant A): 10 Hz rotation, returns fade in ~1/3 s."""
        if not self.show_lidar or sim_dt <= 0:
            self.lidar_pts = [q for q in self.lidar_pts if self.sim.t - q[2] < 0.33]
            return
        circ, segs = self.obst
        bx, by = self.sim.pos(self.focus)
        rot = 2 * math.pi * 10.0 * sim_dt
        a = self.sweep
        res = math.radians(0.8)
        while a < self.sweep + rot:
            r = ray(circ, segs, bx, by, a)
            if r is not None:
                self.lidar_pts.append((bx + r * math.cos(a), by + r * math.sin(a), self.sim.t))
            a += res
        self.sweep = (self.sweep + rot) % (2 * math.pi)
        self.lidar_pts = [q for q in self.lidar_pts if self.sim.t - q[2] < 0.33]

    def consume_events(self):
        evs = self.sim.ref.events
        for e in evs[self.ev_idx:]:
            if e.boat < 0:
                continue
            self.popups.append([e.x, e.y, f"{TASK_NAMES.get(e.task, e.task)}  +{e.points:.0f}" if e.points else
                                f"{TASK_NAMES.get(e.task, e.task)}: {e.note}", self.sim.t, e.boat, e.note if e.points else ""])
        self.ev_idx = len(evs)
        sev = self.sim.events
        for e in sev[self.sev_idx:]:
            if e[1] == "found":
                xy = self.task_xy(e[3])
                if xy:
                    lab = {"T2": "Found: follow the path", "T3dock": "Found: docking", "T3": "Read: dock banner",
                           "T4": "Found: speed challenge"}.get(e[3], "Found: rescue vessel")
                    self.popups.append([xy[0], xy[1], lab, self.sim.t, e[2], f"boat {e[2] + 1}, camera + lidar"])
        self.sev_idx = len(sev)
        ce = [e for e in self.sim.events if e[1] == "contact"]
        for e in ce[getattr(self, "cev_idx", 0):]:
            x, y = self.sim.pos(e[2])
            self.popups.append([x, y, "contact", self.sim.t, e[2], ""])
        self.cev_idx = len(ce)

    def task_xy(self, key):
        c = self.sim.course
        if key == "T2":
            return self.sim.gate_mid(c.module("T2")[0], 0)
        if key in ("T3", "T3dock"):
            m = c.module("T3")[0]; return (m.x, m.y)
        if key == "T4":
            return c.module("T4")[0].path[0]
        if key.startswith("T5:"):
            m = c.modules[int(key[3:])]; o = c.obj(m.objects[0]); return (o.x, o.y)
        return None

    # ---------------------------------------------------------------------------------------
    def draw(self):
        scr = self.screen
        scr.fill(S.PAPER)
        if self.follow:
            self.cam.follow(self.sim, focus=self.focus)
        v = self.cam.view()
        scr.set_clip(self.cam.rect)
        green = self.sim.ref.light_green(self.sim.t)
        def under(surf):
            lay0 = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            if self.show_cones:
                x, y, psi, _ = self.sim.state(self.focus)
                S.camera_cone(lay0, v.p(x, y), psi, 18 * v.s, CAM_FOV)
            if self.show_paths:
                self.draw_paths(lay0, v)
            surf.blit(lay0, (0, 0))
        draw_course(scr, v, self.sim.course, light_green=green, underlay=under)
        self.draw_pennants(scr, v)
        lay = pygame.Surface(scr.get_size(), pygame.SRCALPHA)
        if self.show_phantoms and self.ghosts:
            g, pol = self.ghosts[self.ghost_sel]
            taken = [pygame.Rect(0, 0, 30, 30).move(v.p(*self.sim.pos(j))[0] - 15, v.p(*self.sim.pos(j))[1] - 15) for j in range(self.sim.n)]
            for i in range(g.n):
                P = v.p(*g.pos(i))
                self.ghost_boat(lay, P, float(g.S[2, i]), i, max(22, 1.8 * v.s * 1.2))
                img = S.font(10).render(NAME[pol], True, (*S.INK3, 255))
                lr = img.get_rect(topleft=(P[0] + 14, P[1] + 8))
                if lr.collidelist(taken) < 0:
                    img.set_alpha(170)
                    lay.blit(img, lr)
                    taken.append(lr)
        for (px, py, tb) in self.lidar_pts:
            age = (self.sim.t - tb) / 0.33
            S.aa_circle(lay, (*S.SENSE, int(190 * (1 - age) ** 1.2)), v.p(px, py), 1.5)
        scr.blit(lay, (0, 0))
        if self.show_det and hasattr(self.sim.know, "maps"):
            tr = self.sim.know.maps[self.focus].tracker
            fx, fy = self.sim.pos(self.focus)
            near = sorted((f for f in tr.confirmed() if self.sim.t - f.last_t <= 0.6 and f.source == "own"),
                          key=lambda f: math.hypot(f.x - fx, f.y - fy))
            labelled = {id(f) for f in near[:4]}
            for f in tr.confirmed():
                if self.sim.t - f.last_t > 0.6 or f.source != "own":
                    continue
                cls = f.best_class(); sem = S.CLASS_SEM.get(cls)
                if not sem:
                    continue
                X, Y = v.p(f.x, f.y)
                size = (1.9 * v.s) if cls.startswith("vessel") else max(12, 0.7 * v.s)
                r = pygame.Rect(0, 0, size + 6, size + 6); r.center = (X, Y)
                S.brackets(scr, r, S.SEM[sem][1], 255)
                if id(f) in labelled:
                    S.detection_label(scr, r, S.CLASS_LABEL.get(cls, cls), f"{f.confidence():.2f}")
        for i in range(self.sim.n):                       # intent: a small ring at each boat's current target
            cl = self.sim.claim[i]
            if cl is None:
                continue
            X, Y = v.p(*cl.xy)
            acc = S.FLEET[i % 8]
            pygame.draw.circle(scr, acc, (int(X), int(Y)), 7, 1)
            pygame.draw.circle(scr, acc, (int(X), int(Y)), 2)
        for i in range(self.sim.n):
            x, y, psi, _ = self.sim.state(i)
            S.boat(scr, v.p(x, y), psi, i, max(22, 1.8 * v.s * 1.2), focus=(i == self.focus))
        self.draw_popups(scr, v)
        if self.show_phantoms and self.ghosts:
            g, pol = self.ghosts[self.ghost_sel]
            txt = f"Outline boats: the {NAME[pol]} fleet replaying this same course ({g.metrics()['points']:.0f} points)"
            img = S.font(11).render(txt, True, S.INK2)
            vr = self.cam.rect
            bg = pygame.Surface((img.get_width() + 16, img.get_height() + 8), pygame.SRCALPHA)
            pygame.draw.rect(bg, (*S.CHIP_BG, 230), bg.get_rect(), border_radius=4)
            pygame.draw.rect(bg, (*S.CHIP_BORDER, 255), bg.get_rect(), 1, border_radius=4)
            scr.blit(bg, (vr.x + 12, vr.bottom - bg.get_height() - 12)); scr.blit(img, (vr.x + 20, vr.bottom - bg.get_height() - 8))
        scr.set_clip(None)
        self.draw_top(scr)
        self.draw_panel(scr)

    def draw_paths(self, lay, v):
        """Planned path of every boat (what its executor will follow), thicker for the focused boat, plus the
        focused boat's line-of-sight lookahead point (the path-following controller's 'carrot')."""
        sim = self.sim
        for i in range(sim.n):
            ex = sim.exec[i]
            if ex is None or ex.current() is None:
                continue
            x, y = sim.pos(i)
            pts = [(x, y)]
            if ex.g is not None:
                pts += list(ex.g.wps[ex.g.seg + 1:])
            for ph in ex.phases[ex.k + 1:]:
                if ph.kind == "route":
                    pts.append(ph.target)
                elif ph.kind == "track":
                    pts += list(ph.wps[1:])
            if len(pts) < 2:
                continue
            foc = i == self.focus
            acc = S.FLEET[i % 8]
            S.aa_polyline(lay, (*acc, 235 if foc else 170), [v.p(*q) for q in pts], 2.0 if foc else 1.3)
            if foc and ex.g is not None:
                look = sim.dcfg.los_lookahead_m
                if ex.g.tail_segments and ex.g.seg >= len(ex.g.wps) - 1 - ex.g.tail_segments:
                    look = ex.g.tail_lookahead
                cr = carrot(ex.g, x, y, look)
                if cr:
                    (cx, cy), (px, py) = cr
                    S.aa_line(lay, (*S.INK2, 150), v.p(x, y), v.p(cx, cy), 1.0)
                    S.aa_circle(lay, (*S.PAPER, 255), v.p(cx, cy), 4.0, (*S.INK1, 230))
                    S.aa_circle(lay, (*S.INK2, 200), v.p(px, py), 1.8)
                    if v.s > 7:
                        lay.blit(S.font(10).render("lookahead", True, S.INK3), (v.p(cx, cy)[0] + 7, v.p(cx, cy)[1] - 6))

    def ghost_boat(self, lay, pos, psi, idx, L):
        acc = S.FLEET[idx % 8]
        B = L * 0.62; pw = B * 0.26
        c, s = math.cos(psi), math.sin(psi)

        def P(a, b):
            return (pos[0] + a * c - b * s, pos[1] - (a * s + b * c))
        for side in (-1, 1):
            yc = side * (B / 2 - pw / 2)
            hull = [P(-0.5 * L, yc - pw / 2), P(0.25 * L, yc - pw / 2), P(0.5 * L, yc), P(0.25 * L, yc + pw / 2), P(-0.5 * L, yc + pw / 2)]
            pygame.draw.aalines(lay, (*acc, 170), True, hull)
        deck = [P(-0.25 * L, -B / 2 + pw), P(0.25 * L, -B / 2 + pw), P(0.25 * L, B / 2 - pw), P(-0.25 * L, B / 2 - pw)]
        pygame.draw.aalines(lay, (*acc, 170), True, deck)

    def draw_pennants(self, scr, v):
        for e in self.sim.ref.events:
            if e.points <= 0 or e.task == "T1":
                continue
            X, Y = v.p(e.x, e.y)
            acc = S.FLEET[e.boat % 8]
            S.aa_line(scr, S.INK2, (X + 12, Y - 4), (X + 12, Y - 20), 1.5)
            S.aa_poly(scr, acc, [(X + 13, Y - 20), (X + 24, Y - 16), (X + 13, Y - 12)])

    def draw_popups(self, scr, v):
        keep = []
        for p in self.popups:
            age = self.sim.t - p[3]
            if age > 3.0:
                continue
            keep.append(p)
            X, Y = v.p(p[0], p[1])
            img = S.font(13, True).render(p[2], True, S.INK1)
            sub = S.font(11).render(p[5], True, S.INK3) if len(p) > 5 and p[5] else None
            a = int(255 * min(1.0, 3.0 - age))
            wbg = max(img.get_width(), sub.get_width() if sub else 0) + 18
            hbg = img.get_height() + (sub.get_height() + 1 if sub else 0) + 8
            bg = pygame.Surface((wbg, hbg), pygame.SRCALPHA)
            pygame.draw.rect(bg, (*S.CHIP_BG, int(0.94 * a)), bg.get_rect(), border_radius=4)
            pygame.draw.rect(bg, (*S.CHIP_BORDER, a), bg.get_rect(), 1, border_radius=4)
            pygame.draw.rect(bg, (*S.FLEET[p[4] % 8], a), (0, 3, 3, hbg - 6), border_radius=1)
            img.set_alpha(a)
            yoff = -34 - hbg + 18 - 10 * min(1.0, age)
            vr = self.cam.rect
            bx = min(max(X - wbg / 2, vr.x + 6), vr.right - wbg - 6)
            by = min(max(Y + yoff, vr.y + 6), vr.bottom - hbg - 6)
            scr.blit(bg, (bx, by)); scr.blit(img, (bx + 10, by + 4))
            if sub:
                sub.set_alpha(a); scr.blit(sub, (bx + 10, by + 5 + img.get_height()))
        self.popups = keep

    def draw_top(self, scr):
        w = scr.get_width()
        pygame.draw.rect(scr, S.PAPER, (0, 0, w, TOP_H))
        pygame.draw.line(scr, S.CHIP_BORDER, (0, TOP_H - 1), (w, TOP_H - 1))
        scr.blit(S.font(16, True).render("RoboBoat learning simulation", True, S.INK1), (20, 14))
        sim = self.sim
        t, T = sim.t, sim.cfg.episode_s
        mm = lambda z: f"{int(z // 60)}:{int(z % 60):02d}"
        txt = f"{mm(t)} elapsed    {mm(max(0.0, T - t))} left" + ("    paused" if self.paused else ("" if self.speed == 1 else f"    x{self.speed:g}"))
        img = S.font(14).render(txt, True, S.INK2)
        xr = w - PANEL_W - 20
        scr.blit(img, (xr - img.get_width(), 9))
        bw = 220                                            # thin time bar
        pygame.draw.rect(scr, S.CHIP_BORDER, (xr - bw, 32, bw, 3), border_radius=1)
        pygame.draw.rect(scr, S.INK2, (xr - bw, 32, int(bw * min(1.0, t / T)), 3), border_radius=1)

    @staticmethod
    def _wrap(text, f, width):
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if f.size(cur + " " + wd)[0] > width and cur:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        return lines + ([cur] if cur else [])

    def task_rows(self, sim):
        """(task, short status, points) using plain words."""
        R, c, n = sim.ref, sim.course, sim.n
        pts = {k: 0.0 for k in ("T1", "T2", "T3", "T4", "T5", "T6")}
        bonus = 0.0
        for e in R.events:
            if e.note.startswith("bonus"):
                bonus += e.points
            else:
                pts[e.task] += e.points
        def todo(kind):
            if any(cl is not None and cl.kind == kind for cl in sim.claim):
                return "in progress"
            return "found" if kind in sim.found_t else "not found yet"
        rows = [("T1", f"{sum(p.t1_done for p in R.prog)}/{n} boats through", pts["T1"])]
        if "T2" in R.done:
            note = next((e.note for e in R.events if e.task == "T2"), "")
            rows.append(("T2", "done, count right" if "right" in note else "done, count wrong", pts["T2"]))
        else:
            rows.append(("T2", todo("T2"), 0))
        if R.t3_attempted:
            rows.append(("T3", "correct bay" if "T3" in R.done else "wrong bay", pts["T3"]))
        else:
            busy = any(cl is not None and cl.kind in ("T3", "T3look") for cl in sim.claim)
            rows.append(("T3", "in progress" if busy else ("found" if "T3dock" in sim.found_t else "not found yet"), 0))
        if "T4" in R.done:
            note = next((e.note for e in R.events if e.task == "T4"), "")
            rows.append(("T4", "done" if pts["T4"] > 0 else note, pts["T4"]))
        else:
            st4 = [p.t4_state for p in R.prog]
            rows.append(("T4", "racing" if "circling" in st4 else ("waiting for green" if "holding" in st4 else todo("T4")), 0))
        nv = len(c.module("T5")); k5 = sum(1 for k in R.done if k.startswith("T5"))
        rows.append(("T5", f"{k5} of {nv} delivered", pts["T5"]))
        home = sum(p.home for p in R.prog)
        rows.append(("T6", (f"all home, bonus" if bonus else f"{home}/{n} home"), pts["T6"] + bonus))
        return rows

    def draw_panel(self, scr):
        w, h = scr.get_size()
        r = pygame.Rect(w - PANEL_W, 0, PANEL_W, h)
        pygame.draw.rect(scr, (250, 251, 252), r)
        pygame.draw.line(scr, S.CHIP_BORDER, r.topleft, r.bottomleft)
        x0, x1 = r.x + 26, r.right - 26
        sim = self.sim
        f12, f13, f14 = S.font(12), S.font(13), S.font(14)
        rule = lambda yy: pygame.draw.line(scr, S.CHIP_BORDER, (x0, yy), (x1, yy))
        head = lambda text, yy: scr.blit(f12.render(text, True, S.INK3), (x0, yy))
        y = 20

        # fleet + score
        scr.blit(S.font(18, True).render(f"{NAME.get(sim.cfg.policy, sim.cfg.policy)} fleet", True, S.INK1), (x0, y)); y += 30
        pts = sim.ref.points
        big = S.font(34, True)
        scr.blit(big.render(f"{pts:.0f}", True, S.INK1), (x0, y))
        scr.blit(f13.render("points", True, S.INK3), (x0 + big.size(f"{pts:.0f}")[0] + 8, y + 18))
        y += 56

        # boats: just the task
        rule(y); y += 12; head("Boats", y); y += 22
        for i in range(sim.n):
            S.boat(scr, (x0 + 11, y + 8), 0.0, i, 20, focus=(i == self.focus))
            cl = sim.claim[i]
            if sim.ref.prog[i].home and cl is None:
                job = "home"
            elif cl is None:
                job = "waiting"
            elif cl.kind == "explore":
                job = "searching"
            elif cl.kind == "T3look":
                job = "reading dock banners"
            else:
                job = TASK_NAMES.get(cl.kind, cl.kind)
                ex = sim.exec[i]; ph = ex.current() if ex else None
                if ph is not None and ph.kind == "hold":
                    job += ", waiting for green"
            scr.blit(f14.render(job, True, S.INK1 if cl else S.INK3), (x0 + 32, y)); y += 26
        y += 6

        # tasks
        rule(y); y += 12; head("Tasks", y); y += 22
        for k, st, p in self.task_rows(sim):
            ok = p > 0
            S.aa_circle(scr, S.INK2 if ok else S.HULL, (x0 + 4, y + 8), 4, S.INK3)
            scr.blit(f13.render(TASK_NAMES[k], True, S.INK1 if ok else S.INK2), (x0 + 16, y))
            scr.blit(f12.render(st, True, S.INK3), (x0 + 168, y + 1))
            pv = f"{p:.0f}" if ok else ""
            scr.blit(f13.render(pv, True, S.INK1), (x1 - f13.size(pv)[0], y)); y += 22
        y += 6

        # strategies
        rule(y); y += 12; head("Strategies on this course", y)
        board = self.board.get(sim.cfg.comm_mode, {})
        if board:
            lab = "average"
            scr.blit(f12.render(lab, True, S.INK4), (x1 - f12.size(lab)[0], y))
        y += 22
        rows = [(pts, sim.cfg.policy, None)] + [(g.ref.points, p, gi) for gi, (g, p) in enumerate(self.ghosts)]
        for pv_, p, gi in sorted(rows, key=lambda q: -q[0]):
            me = gi is None
            sel = gi is not None and gi == self.ghost_sel and self.show_phantoms
            col = S.INK1 if me else S.INK2
            fnt = S.font(13, True) if me else f13
            scr.blit(fnt.render(NAME.get(p, p), True, col), (x0, y))
            if sel:
                scr.blit(f12.render("outline boats", True, S.INK4), (x0 + fnt.size(NAME.get(p, p))[0] + 8, y + 1))
            v_ = f"{pv_:.0f}"
            scr.blit(fnt.render(v_, True, col), (x1 - 70 - fnt.size(v_)[0], y))
            if p in board:
                av = f"{board[p]['points']:.0f}"
                scr.blit(f12.render(av, True, S.INK4), (x1 - f12.size(av)[0], y + 1))
            y += 22
        y += 8
        self.draw_teammap(scr, pygame.Rect(x0, y, x1 - x0, h - y - 30))
        scr.blit(S.font(11).render("Tab next boat    G other outline fleet    F whole course", True, S.INK4), (x0, h - 22))

    def draw_teammap(self, scr, area):
        """What the focused boat's team map knows: confirmed objects (own = filled, told by radio = hollow),
        its recon sector, and where the selected phantom fleet is."""
        know = self.sim.know
        if not hasattr(know, "maps") or area.h < 90:
            return
        c = self.sim.course
        scr.blit(S.font(12).render("Team map", True, S.INK3), (area.x, area.y))
        k = min(area.w / c.width, (area.h - 22) / c.height)
        r = pygame.Rect(area.x, area.y + 20, int(c.width * k), int(c.height * k))
        pygame.draw.rect(scr, S.WATER, r); pygame.draw.rect(scr, S.CHIP_BORDER, r, 1)
        P = lambda x, y: (r.x + x * k, r.bottom - y * k)
        M = know.maps[self.focus]
        if hasattr(know, "sector"):
            lo, hi = know.sector(self.focus)
            sec = pygame.Surface((int((hi - lo) * k), r.h), pygame.SRCALPHA)
            sec.fill((*S.FLEET[self.focus % 8], 22))
            scr.blit(sec, (r.x + lo * k, r.y))
        for f in M.fused(0.5):
            sem = S.CLASS_SEM.get(f.best_class() or "")
            if not sem:
                continue
            col = S.SEM[sem][0]
            if f.source == "own":
                S.aa_circle(scr, col, P(f.x, f.y), 2)
            else:
                pygame.draw.circle(scr, col, [int(q) for q in P(f.x, f.y)], 2, 1)
        if self.show_phantoms and self.ghosts:
            g, _ = self.ghosts[self.ghost_sel]
            for i in range(g.n):
                pygame.draw.circle(scr, S.FLEET[i % 8], [int(q) for q in P(*g.pos(i))], 3, 1)
        for i in range(self.sim.n):
            S.aa_circle(scr, S.FLEET[i % 8], P(*self.sim.pos(i)), 3, S.INK1 if i == self.focus else None)


    # ---------------------------------------------------------------------------------------
    def run(self):
        if self.args.record:
            return self.record()
        running = True
        while running:
            dt = min(0.05, self.clock.tick(60) / 1000)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_SPACE: self.paused = not self.paused
                    elif k == pygame.K_1: self.speed = 1.0
                    elif k == pygame.K_2: self.speed = 2.0
                    elif k == pygame.K_3: self.speed = 4.0
                    elif k == pygame.K_TAB: self.focus = (self.focus + 1) % self.sim.n
                    elif k == pygame.K_k: self.show_phantoms = not self.show_phantoms
                    elif k == pygame.K_g and self.ghosts: self.ghost_sel = (self.ghost_sel + 1) % len(self.ghosts)
                    elif k == pygame.K_p: self.show_paths = not self.show_paths
                    elif k == pygame.K_l: self.show_lidar = not self.show_lidar
                    elif k == pygame.K_c: self.show_cones = not self.show_cones
                    elif k == pygame.K_d: self.show_det = not self.show_det
                    elif k == pygame.K_f: self.follow = not self.follow
                    elif k == pygame.K_r: self.seed += 1; self.build()
            sdt = self.step(dt)
            self.update_lidar(sdt)
            self.consume_events()
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def record(self):
        a = self.args
        tmp = tempfile.mkdtemp()
        fps = 30
        for f in range(int(a.seconds * fps)):
            sdt = self.step(1 / fps)
            self.update_lidar(sdt)
            self.consume_events()
            self.draw()
            pygame.image.save(self.screen, os.path.join(tmp, f"f{f:05d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/f%05d.png",
                        "-vf", "scale=1440:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", f"{a.record}.mp4"], check=True)
        shutil.rmtree(tmp)
        print("wrote", f"{a.record}.mp4")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--policy", default="P3")
    ap.add_argument("--ghost", default=None, help="phantom fleet shown in the world first (e.g. P4)")
    ap.add_argument("--fleet", type=int, default=4)
    ap.add_argument("--comm", default="degraded")
    ap.add_argument("--variant", default="standard")
    ap.add_argument("--shortcut", action="store_true")
    ap.add_argument("--perception", default="sensors")
    ap.add_argument("--episode", type=float, default=300.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args(argv)
    if a.record:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    Game(a).run()


if __name__ == "__main__":
    main()
