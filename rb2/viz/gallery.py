"""Preview renders (checkpoint A, revision 3: restrained visual system from the two design reviews).

  .venv/bin/python -m rb2.viz.gallery && .venv/bin/python -m rb2.viz.gallery_html
"""
from __future__ import annotations

import math
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from ..course import TASK_NAMES  # noqa: E402
from ..geom import obstacle_list, raycast  # noqa: E402,F401
from ..generate import GenConfig, generate_course  # noqa: E402
from . import style as S  # noqa: E402

OUT = "figures/v2_preview"
CAM_FOV, CAM_RANGE, LIDAR_RANGE = math.radians(90), 18.0, 20.0


class View:
    def __init__(self, rect, x0, y0, x1, y1):
        self.rect = pygame.Rect(rect)
        self.s = min(self.rect.w / (x1 - x0), self.rect.h / (y1 - y0))
        self.x0, self.y1 = x0, y1
        self.ox = self.rect.x + (self.rect.w - (x1 - x0) * self.s) / 2
        self.oy = self.rect.y + (self.rect.h - (y1 - y0) * self.s) / 2

    def p(self, x, y):
        return (self.ox + (x - self.x0) * self.s, self.oy + (self.y1 - y) * self.s)


def _buoy_px(o, v):
    tall = o.cls.endswith("_tall") or o.cls == "blue"
    base = 11 if tall else 6
    return max(base, 2 * o.r * v.s * (1.25 if tall else 1.6))


def draw_course(surf, v: View, c, light_green=False, labels=False, underlay=None):
    W, H = c.width, c.height
    x0, y0 = v.p(0, H)
    rect = pygame.Rect(x0, y0, W * v.s, H * v.s)
    pygame.draw.rect(surf, S.WATER_EDGE, rect)
    pygame.draw.rect(surf, S.WATER, rect.inflate(-12, -12))
    pygame.draw.rect(surf, S.FRAME, rect, 1)
    if underlay is not None:              # drawn on the water, beneath every course object (e.g. the camera cone)
        underlay(surf)
    sx, sy, _ = c.start
    pier = pygame.Rect(0, 0, 5.0 * v.s, 1.4 * v.s); pier.midbottom = v.p(sx, 0.05)
    pygame.draw.rect(surf, S.DOCK, pier)
    cubes = [q for m in c.modules if m.kind == "T3" for q in m.cubes] + list(c.piers)
    for (cx, cy, cs, hd) in cubes:
        h = cs / 2 + 0.02
        co, sn = math.cos(hd), math.sin(hd)
        S.aa_poly(surf, S.DOCK, [v.p(cx + a * co - b * sn, cy + a * sn + b * co) for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))])
    for m in c.modules:
        if m.kind == "T3":
            for b in m.bays:
                S.banner(surf, v.p(*b.banner), b.shape, b.color, 14 if v.s < 12 else 18)
    for line in c.booms + ([] if c.shortcut_open else c.boom_gates):
        P = S.chaikin([v.p(*q) for q in line[::3] + [line[-1]]], 2)
        S.aa_polyline(surf, S.BOOM, P, 2.5)
        acc, nxt = 0.0, 8.0
        for a, b in zip(P, P[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            while acc + L >= nxt:
                f = (nxt - acc) / max(L, 1e-9)
                S.aa_circle(surf, S.BOOM_FLOAT, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), 2)
                nxt += 16.0
            acc += L
    for o in c.objects:
        X, Y = v.p(o.x, o.y)
        if o.cls in ("vessel_orange", "vessel_black"):
            S.vessel(surf, (X, Y), o.heading, o.cls, o.mark, max(20, o.length * v.s))
        elif o.cls == "moored":
            S.moored(surf, (X, Y), o.heading, max(18, o.length * v.s))
        elif o.cls == "light_panel":
            S.light_panel(surf, (X, Y), light_green, max(1.0, min(1.8, v.s / 12)))
        elif o.cls == "marker":
            if o.r >= 0.3:
                S.aa_circle(surf, S.INK4, (X, Y), 3)
            else:
                S.aa_circle(surf, S.HULL, (X, Y), 3, S.INK3)
        else:
            S.buoy(surf, (X, Y), o.cls, _buoy_px(o, v))
    if labels:
        placed = []
        objs = [(v.p(o.x, o.y)) for o in c.objects]
        done = set()
        for m in c.modules:
            if m.kind in done:
                continue
            done.add(m.kind)
            img = S.font(11).render(TASK_NAMES[m.kind], True, S.INK3)
            w, h = img.get_width(), img.get_height()
            mx, my = v.p(m.x, m.y)
            cands = [(mx - w / 2, my - m.radius * v.s * 0.9 - h), (mx - w / 2, my + m.radius * v.s * 0.75),
                     (mx + m.radius * v.s * 0.6, my - h / 2), (mx - m.radius * v.s * 0.6 - w, my - h / 2),
                     (mx - w / 2, my - h - 14), (mx - w / 2, my + 14)]
            if m.kind == "T3":       # dock sits on an edge: label goes into the water
                ux, uy = math.cos(m.heading), -math.sin(m.heading)
                cands = [(mx + ux * 70 - w / 2, my + uy * 70 - h / 2)] + cands
            best = None
            for (x, y) in cands:
                r = pygame.Rect(x - 3, y - 2, w + 6, h + 4)
                if not v.rect.contains(r):
                    continue
                hit = sum(r.collidepoint(px, py) for px, py in objs) + sum(r.colliderect(q) for q in placed) * 5
                if best is None or hit < best[0]:
                    best = (hit, x, y, r)
            if best:
                _, x, y, r = best
                placed.append(r)
                surf.blit(img, (x, y))


# ---- sensing (ray-cast lidar; camera FOV): geometry lives in rb2.geom ---------------------------
_obstacles = obstacle_list


def draw_sensors(surf, v: View, c, bx, by, psi, idx=0, lidar="hits", labels=True, cam_range=CAM_RANGE, max_labels=6):
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    C = v.p(bx, by)
    S.camera_cone(layer, C, psi, cam_range * v.s, CAM_FOV)
    ranges, hits = raycast(c, bx, by)
    if lidar == "outline":
        n = len(ranges)
        pts = [v.p(bx + r * math.cos(2 * math.pi * k / n), by + r * math.sin(2 * math.pi * k / n)) for k, r in enumerate(ranges)]
        S.lidar_outline(layer, C, pts)
    if lidar in ("hits", "outline"):
        for o in hits.values():
            if o.cls == "marker" and o.r >= 0.3:
                continue
            X, Y = v.p(o.x, o.y)
            rim = (max(10, o.length * v.s) * 0.3) if o.cls.startswith(("vessel", "moored")) else _buoy_px(o, v) / 2
            S.lidar_tick(layer, (X, Y), rim, math.atan2(by - o.y, bx - o.x))
    surf.blit(layer, (0, 0))
    det = []
    for o in c.objects:
        d = math.hypot(o.x - bx, o.y - by)
        off = abs((math.atan2(o.y - by, o.x - bx) - psi + math.pi) % (2 * math.pi) - math.pi)
        if d > cam_range or off > CAM_FOV / 2 or o.cls not in S.CLASS_SEM:
            continue
        conf = int(round(100 * max(0.30, 0.97 - 0.022 * d - 0.25 * off / (CAM_FOV / 2))))
        det.append((conf, o))
    det.sort(key=lambda t: -t[0])
    for k, (conf, o) in enumerate(det):
        X, Y = v.p(o.x, o.y)
        size = (max(20, o.length * v.s) * 0.75) if o.cls.startswith("vessel") else _buoy_px(o, v)
        r = pygame.Rect(0, 0, size + 8, size + 8); r.center = (X, Y)
        rim = S.SEM[S.CLASS_SEM[o.cls]][1]
        S.brackets(surf, r, rim, 128 if conf < 50 else 255)
        if labels and k < max_labels:
            S.detection_label(surf, r, S.CLASS_LABEL[o.cls], conf, 10 if v.s < 15 else 11)
    S.boat(surf, C, psi, idx, max(22, min(44, 1.8 * v.s * 1.25)), focus=True)


def title(surf, text, x, y, size=16, color=None):
    surf.blit(S.font(size, True).render(text, True, color or S.INK1), (x, y))


def caption(surf, text, x, y, size=12, color=None):
    surf.blit(S.font(size).render(text, True, color or S.INK3), (x, y))


def task_gallery():
    Wd, Ht = 1800, 1180
    surf = pygame.Surface((Wd, Ht)); surf.fill(S.PAPER)
    title(surf, "The six RoboBoat tasks, and what the boat senses", 30, 22, 20)
    caption(surf, "Camera field of view fades with distance. Brackets mark camera detections in the object's colour, with "
                  "confidence. Short dark arcs are lidar returns on the faces the laser hits.", 30, 50, 13)
    rules = {"T1": "Pass both red/green gates first; required before anything else.",
             "T2": "Stay inside the gates, avoid yellow buoys and vessels, count the yellows.",
             "T3": "Find the banner of the day and enter an empty matching bay.",
             "T4": "Wait for green, go through the gate, circle the blue buoy, come back.",
             "T5": "Spray the triangle on an orange vessel; drop a ball on a black vessel's plus.",
             "T6": "Return through the black home gate."}
    tw, th = 570, 500
    c = generate_course(3, GenConfig(difficulty="hard"))
    for k, kind in enumerate(["T1", "T2", "T3", "T4", "T5", "T6"]):
        ms = c.module(kind); m = ms[0]
        rx, ry = 30 + (k % 3) * (tw + 15), 86 + (k // 3) * (th + 46)
        rect = pygame.Rect(rx, ry + 44, tw, th - 44)
        title(surf, TASK_NAMES[kind], rx, ry + 2, 16)
        caption(surf, rules[kind], rx, ry + 24)
        if kind == "T5":
            vs = [c.obj(mm.objects[0]) for mm in ms]
            pair = [next(o for o in vs if o.cls == "vessel_orange"), next(o for o in vs if o.cls == "vessel_black")]
            for j, o in enumerate(pair):
                sub = pygame.Rect(rect.x + j * rect.w // 2, rect.y, rect.w // 2 - 3, rect.h)
                vv = View(sub, o.x - 6, o.y - 7.5, o.x + 6, o.y + 7.5)
                surf.set_clip(sub); draw_course(surf, vv, c)
                bx, by = o.x - 3.4 * math.cos(o.heading + math.pi / 2), o.y - 3.4 * math.sin(o.heading + math.pi / 2)
                psi = math.atan2(o.y - by, o.x - bx)
                A0, A1 = vv.p(bx, by), vv.p(o.x, o.y)
                if o.cls == "vessel_orange":
                    pts = [(A0[0] + (A1[0] - A0[0]) * f, A0[1] + (A1[1] - A0[1]) * f - 16 * math.sin(math.pi * f)) for f in np.linspace(0, 1, 20)]
                    lay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                    S.aa_polyline(lay, (*S.SEM["blue"][0], 153), pts, 1.5); surf.blit(lay, (0, 0))
                else:
                    S.aa_circle(surf, S.MARK, (A1[0] + 1, A1[1]), 4, S.INK2)
                draw_sensors(surf, vv, c, bx, by, psi, labels=False, cam_range=9)
                surf.set_clip(None)
                pygame.draw.rect(surf, S.FRAME, sub, 1)
            continue
        R = {"T1": 13, "T2": 16, "T3": 10, "T4": 14, "T6": 10}[kind]
        v = View(rect, m.x - R * 1.15, m.y - R, m.x + R * 1.15, m.y + R)
        surf.set_clip(rect); draw_course(surf, v, c, light_green=(kind == "T4"))
        if kind == "T1":
            draw_sensors(surf, v, c, m.x, m.y - 9.5, math.pi / 2)
        elif kind == "T2":
            px, py = m.path[2]; draw_sensors(surf, v, c, px, py, math.atan2(m.path[3][1] - py, m.path[3][0] - px))
        elif kind == "T3":
            b = m.bays[0]
            draw_sensors(surf, v, c, b.entry[0] + 2.5 * math.cos(b.heading + math.pi / 2), b.entry[1] + 2.5 * math.sin(b.heading + math.pi / 2), b.heading)
        elif kind == "T4":
            hx, hy = m.path[0]; draw_sensors(surf, v, c, hx, hy, m.heading)
        else:
            draw_sensors(surf, v, c, m.x, m.y + 8, -math.pi / 2, cam_range=12)
        surf.set_clip(None)
        pygame.draw.rect(surf, S.FRAME, rect, 1)
        if kind == "T3":
            S.banner(surf, (rect.x + 22, rect.y + 20), c.day_shape, c.day_color, 18)
            caption(surf, f"banner of the day: {c.day_color} {c.day_shape}", rect.x + 38, rect.y + 13, 12, S.INK2)
    pygame.image.save(surf, f"{OUT}/tasks_gallery.png")


def courses_sheet():
    Wd, Ht = 1800, 990
    surf = pygame.Surface((Wd, Ht)); surf.fill(S.PAPER)
    title(surf, "Procedural courses", 30, 20, 20)
    caption(surf, "Three seeds at two difficulties: Poisson-disc module placement, spline path, cube dock, validated reachability.", 30, 48, 13)
    tw, th = 570, 410
    for k, (seed, diff) in enumerate([(11, "standard"), (12, "standard"), (13, "standard"), (11, "hard"), (12, "hard"), (13, "hard")]):
        c = generate_course(seed, GenConfig(difficulty=diff))
        rx, ry = 30 + (k % 3) * (tw + 15), 84 + (k // 3) * (th + 50)
        rect = pygame.Rect(rx, ry + 24, tw, th)
        draw_course(surf, View(rect, -1, -1, c.width + 1, c.height + 1), c, labels=True)
        caption(surf, f"seed {seed}, {diff}; banner of the day {c.day_color} {c.day_shape}", rx, ry + 2, 12, S.INK2)
    pygame.image.save(surf, f"{OUT}/courses.png")


def return_rush_sheet(seed=7):
    Wd, Ht = 1800, 1190
    surf = pygame.Surface((Wd, Ht)); surf.fill(S.PAPER)
    title(surf, "Return rush: the Braess paradox as a real course", 30, 18, 20)
    caption(surf, "Marinas narrow the upper-west and lower-east corridors to one boat width; the oil-spill boom blocks the middle. "
                  "The containment crew can open a channel through the boom.", 30, 46, 13)
    c = generate_course(seed, GenConfig(variant="return_rush"))
    N = c.rush_nodes; Sx, A, B, T_ = N["S"], N["A"], N["B"], N["T"]
    west = [Sx, (A[0], Sx[1] - 2), A, (A[0], T_[1] + 2), T_]
    east = [Sx, (B[0], Sx[1] - 2), B, (B[0], T_[1] + 2), T_]
    short = [Sx, (A[0], Sx[1] - 2), A, B, (B[0], T_[1] + 2), T_]
    for k, opened in enumerate((False, True)):
        c.shortcut_open = opened
        rect = pygame.Rect(30 + k * 885, 80, 860, 1040)
        v = View(rect, -1, -1, c.width + 1, c.height + 1)
        draw_course(surf, v, c)
        lay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        if not opened:
            for path in (west, east):
                P = [v.p(*q) for q in path]; S.aa_polyline(lay, S.INK2, P, 2.0); S.chevrons(lay, S.INK2, P)
            for q in (A, B):
                X, Y = v.p(*q); S.aa_line(lay, S.INK4, (X, Y - 6), (X, Y + 6), 1.5)
            cap = "Shortcut closed: the fleet splits; each route has one single-file channel."
        else:
            for path in (west, east):
                S.aa_polyline(lay, S.INK4, [v.p(*q) for q in path], 1.5)
            P = [v.p(*q) for q in short]
            S.aa_polyline(lay, S.INK2, P, 2.0); S.chevrons(lay, S.INK2, P)
            S.aa_line(lay, S.FLEET[0], v.p(*A), v.p(*B), 2.5)
            cap = "Shortcut open: every boat takes it, so every boat uses both single-file channels."
        surf.blit(lay, (0, 0))
        for nm, q, dx in (("task-area exit", Sx, 12), ("A", A, -18), ("B", B, 12), ("home gate", T_, 12)):
            X, Y = v.p(*q)
            S.aa_circle(surf, S.HULL, (X, Y), 5, S.INK2)
            img = S.font(12).render(nm, True, S.INK2)
            surf.blit(img, (X + dx if dx > 0 else X + dx - img.get_width() + 6, Y - img.get_height() / 2))
        for i in range(4):
            S.boat(surf, v.p(Sx[0] - 3 + 2 * i, Sx[1] + 2.5), -math.pi / 2, i, 22)
        caption(surf, cap, rect.x, rect.bottom + 10, 14, S.INK2)
    c.shortcut_open = False
    pygame.image.save(surf, f"{OUT}/return_rush.png")


def boats_sheet():
    Wd, Ht = 1800, 360
    surf = pygame.Surface((Wd, Ht)); surf.fill(S.PAPER)
    title(surf, "Fleet", 30, 18, 20)
    caption(surf, "Each boat: hull in a light tint of its own muted colour, deck in the full colour (boats 9 to 16 use a split deck). No numbers.", 30, 46, 13)
    rect = pygame.Rect(30, 84, Wd - 60, 250)
    pygame.draw.rect(surf, S.WATER, rect, border_radius=6)
    for i in range(16):
        x = rect.x + 60 + (i % 8) * 212
        y = rect.y + (60 if i < 8 else 170)
        S.boat(surf, (x, y), 0.35, i, 44, focus=(i == 0))
        S.boat(surf, (x + 80, y), math.pi / 2, i, 22)
    pygame.image.save(surf, f"{OUT}/boats.png")


def main():
    pygame.init()
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    task_gallery(); courses_sheet(); return_rush_sheet(); boats_sheet()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
