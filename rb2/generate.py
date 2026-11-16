"""Procedural RoboBoat course generation (v2).

Techniques
 1. Basin: open rectangle, corner markers, launch on the south edge (scaled with fleet size).
 2. Module placement: Poisson-disc ("dart throwing" with per-module exclusion radii); T1 straight ahead of the
    launch, T6 beside it; the dock against an edge; everything else in the open basin.
 3. T2 path: Catmull-Rom spline through jittered control points, rejected if curvature exceeds a limit; gates every
    s metres of arc length (width jitter); yellow buoys by constrained sampling INSIDE the corridor that always leaves a
    passable side; stationary vessels just outside the corridor.
 4. T3 dock: spine + tines built from the 20-in cube grid; banners sampled so >= 1 bay matches the day's
    shape/colour and is EMPTY (plus, sometimes, a matching decoy bay that is occupied).
 5. T4: holding bay with light panel, gate 1.8-3.0 m, blue buoy 12-30 m beyond on the gate axis, black hazards
    sampled off the loop corridor.
 6. T5: orange (triangle) and black (plus) vessels by Poisson disc in free water.
 7. Validation: compile to a planning grid (src.world.World with no land) and require every module entry to be
    reachable from the launch; otherwise resample. Deterministic for a given seed.
 8. Variant "return_rush": a containment boom with an optional gap and two single-file lanes (Braess layout).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.config import SimConfig
from src.world import CurrentField, World

from .course import BUOY_RADIUS, COLORS, SHAPES, Bay, Course, Module, Obj
from .rng2 import stream

VALUES = {"T1": 10, "T2": 20, "T3": 20, "T4": 20, "T5": 10, "T6": 10}   # fictional; T5 per vessel
DIFFICULTY = {
    "standard": dict(n_yellow=(3, 4), n_black4=(2, 3), n_orange=(1, 2), n_blackv=(1, 2), n_gates2=(6, 7),
                     curv_max=1 / 7.0, lateral=4.0, bays=3, decoy=0.4, path_vessels=1),
    "hard": dict(n_yellow=(5, 6), n_black4=(4, 5), n_orange=(2, 3), n_blackv=(2, 3), n_gates2=(7, 9),
                 curv_max=1 / 5.0, lateral=6.0, bays=4, decoy=0.8, path_vessels=2),
}


@dataclass
class GenConfig:
    n_boats: int = 4
    difficulty: str = "standard"
    variant: str = "standard"
    base_w: float = 80.0
    base_h: float = 56.0
    current_strength: float = 0.12
    grid_res: float = 0.5
    hull_radius: float = 0.9
    plan_margin: float = 0.35


class _Builder:
    def __init__(self, seed, gc: GenConfig):
        self.seed, self.gc = seed, gc
        self.rng = stream(seed, "course")
        self.objs: List[Obj] = []
        self.mods: List[Module] = []
        self.d = DIFFICULTY[gc.difficulty]
        s = math.sqrt(gc.n_boats / 4.0)
        self.W, self.H = gc.base_w * s, gc.base_h * s
        self.task_floor = 0.0
        self.booms, self.gates_rr, self.piers, self.nodes = [], [], [], {}
        if gc.variant == "return_rush":
            self.H = gc.base_h * s * 1.75

    def add(self, cls, x, y, module=-1, r=None, **kw) -> int:
        o = Obj(len(self.objs), cls, float(x), float(y), BUOY_RADIUS.get(cls, 0.8) if r is None else r, module, **kw)
        self.objs.append(o)
        if module >= 0:
            self.mods[module].objects.append(o.id)
        return o.id

    def module(self, kind, x, y, heading, radius) -> Module:
        m = Module(len(self.mods), kind, float(x), float(y), float(heading), radius, VALUES[kind])
        self.mods.append(m)
        return m

    def free(self, x, y, r, margin=1.0) -> bool:
        if not (r + 3 < x < self.W - r - 3 and r + 3 < y < self.H - r - 3):
            return False
        return all(math.hypot(x - m.x, y - m.y) > r + m.radius + margin for m in self.mods)

    # --- modules ------------------------------------------------------------------
    def gate(self, m, cx, cy, heading, width, small=False):
        nx, ny = -math.sin(heading), math.cos(heading)        # left of travel direction
        red = self.add("red" if small else "red_tall", cx + nx * width / 2, cy + ny * width / 2, m.id)
        grn = self.add("green" if small else "green_tall", cx - nx * width / 2, cy - ny * width / 2, m.id)
        m.gates.append((red, grn))

    def t1_t6(self, sx, sy):
        m1 = self.module("T1", sx, sy + 11, math.pi / 2, 8)
        for dy in (6.0, 13.0):
            self.gate(m1, sx, sy + dy, math.pi / 2, 4.0)
        hx = sx + 11.0
        m6 = self.module("T6", hx, sy + 4, -math.pi / 2, 5)
        self.add("black", hx - 1.6, sy + 4, m6.id, r=0.10)
        self.add("black", hx + 1.6, sy + 4, m6.id, r=0.10)
        m6.gates.append((m6.objects[0], m6.objects[1]))

    def place(self, radius, ymin, tries=400, edge=False):
        ymin = max(ymin, self.task_floor)
        if ymin + radius > self.H - radius - 3:
            raise RuntimeError("no room above task floor")
        for _ in range(tries):
            if edge:
                side = self.rng.integers(0, 3)          # 0 east, 1 north, 2 west
                if side == 0:
                    x, y, hd = self.W - 2.5, self.rng.uniform(ymin + radius, self.H - radius - 3), math.pi
                elif side == 1:
                    x, y, hd = self.rng.uniform(radius + 3, self.W - radius - 3), self.H - 2.5, -math.pi / 2
                else:
                    x, y, hd = 2.5, self.rng.uniform(ymin + radius, self.H - radius - 3), 0.0
                if all(math.hypot(x - m.x, y - m.y) > radius + m.radius + 2 for m in self.mods):
                    return x, y, hd
            else:
                x, y = self.rng.uniform(radius + 3, self.W - radius - 3), self.rng.uniform(ymin + radius, self.H - radius - 3)
                if self.free(x, y, radius, 2.0):
                    return x, y, float(self.rng.uniform(-math.pi, math.pi))
        raise RuntimeError("placement failed")

    def t2_path(self):
        d = self.d
        for _ in range(60):
            x, y, hd = self.place(13.5, 20.0)
            n_ctrl = 5
            L = 26.0
            ctrl = []
            for k in range(n_ctrl):
                s = -L / 2 + L * k / (n_ctrl - 1)
                lat = self.rng.uniform(-d["lateral"], d["lateral"]) if 0 < k < n_ctrl - 1 else 0.0
                ctrl.append((x + s * math.cos(hd) - lat * math.sin(hd), y + s * math.sin(hd) + lat * math.cos(hd)))
            pts = catmull_rom(ctrl, 24)
            if max_curvature(pts) <= d["curv_max"] and all(3 < px < self.W - 3 and 3 < py < self.H - 3 for px, py in pts):
                break
        else:
            raise RuntimeError("T2 spline failed")
        m = self.module("T2", x, y, hd, 13.5)
        m.path = pts
        s_tot = arc_length(pts)
        n_g = int(self.rng.integers(d["n_gates2"][0], d["n_gates2"][1] + 1))
        gate_s = np.linspace(1.5, s_tot - 1.5, n_g)
        for s in gate_s:
            (gx, gy), th = point_at(pts, s)
            self.gate(m, gx, gy, th, float(self.rng.uniform(2.8, 3.3)), small=True)
        # yellow buoys inside the corridor, between gates, leaving a passable side
        n_y = int(self.rng.integers(d["n_yellow"][0], d["n_yellow"][1] + 1))
        mids = [(a + b) / 2 for a, b in zip(gate_s, gate_s[1:])]
        picks = self.rng.choice(len(mids), size=min(n_y, len(mids)), replace=False)
        for k in sorted(picks):
            (px, py), th = point_at(pts, mids[k])
            off = self.rng.choice([-1, 1]) * self.rng.uniform(0.55, 0.85)
            self.add("yellow", px - off * math.sin(th), py + off * math.cos(th), m.id)
        m.data["n_yellow"] = len(picks)
        # stationary vessels just outside the corridor
        for _ in range(d["path_vessels"]):
            s = float(self.rng.uniform(0.25, 0.75) * s_tot)
            (px, py), th = point_at(pts, s)
            side = self.rng.choice([-1, 1])
            cls = "vessel_orange" if self.rng.random() < 0.5 else "vessel_black"
            self.add(cls, px - side * 3.4 * math.sin(th), py + side * 3.4 * math.cos(th), m.id, r=0.8,
                     heading=float(th + self.rng.uniform(-0.4, 0.4)), length=1.6)

    def t3_dock(self, day_shape, day_color):
        d = self.d
        nb = d["bays"]
        bay_w, tine_len, cube = 2.6, 2.4, 0.508
        span = nb * bay_w + (nb + 1) * cube
        x, y, hd = self.place(span / 2 + 3, 20.0, edge=True)
        m = self.module("T3", x, y, hd, span / 2 + 3)
        ux, uy = math.cos(hd), math.sin(hd)              # into the basin
        vx, vy = -uy, ux                                  # along the spine

        def W(a, b):                                      # local (along, out) -> world
            return (x + a * vx + b * ux, y + a * vy + b * uy)
        # spine: 2 cubes deep along the full span; tines: 1 cube wide, tine_len long
        a = -span / 2
        while a < span / 2 - 1e-6:
            for b in (0.0, cube):
                m.cubes.append((*W(a + cube / 2, b + cube / 2), cube, hd))
            a += cube
        tines = [-span / 2 + k * (bay_w + cube) for k in range(nb + 1)]
        for t0 in tines:
            b = 2 * cube
            while b < 2 * cube + tine_len - 1e-6:
                m.cubes.append((*W(t0 + cube / 2, b + cube / 2), cube, hd))
                b += cube
        # banners: one empty matching bay guaranteed; optional occupied matching decoy
        match = int(self.rng.integers(0, nb))
        combos = [(s, c) for s in SHAPES for c in COLORS if (s, c) != (day_shape, day_color)]
        decoy = int(self.rng.integers(0, nb)) if self.rng.random() < d["decoy"] else -1
        if decoy == match:
            decoy = -1
        for k in range(nb):
            center = tines[k] + cube + bay_w / 2
            if k == match or k == decoy:
                shape, color = day_shape, day_color
            else:
                shape, color = combos[int(self.rng.integers(0, len(combos)))]
            occupied = (k == decoy) or (k != match and self.rng.random() < 0.35)
            bay = Bay(entry=W(center, 2 * cube + tine_len + 2.6), inner=W(center, 2 * cube + 1.0), heading=hd + math.pi,
                      shape=shape, color=color, occupied=occupied, banner=W(center, cube * 0.5))
            m.bays.append(bay)
            if occupied:
                vx_, vy_ = W(center, 2 * cube + 1.1)
                self.add("vessel_black" if self.rng.random() < 0.5 else "vessel_orange", vx_, vy_, m.id, r=0.8,
                         heading=hd, length=1.6)
        m.data.update(match=match, decoy=decoy)

    def t4_speed(self):
        d = self.d
        x, y, hd = self.place(11.0, 20.0)
        m = self.module("T4", x, y, hd, 11.0)
        ux, uy = math.cos(hd), math.sin(hd)
        gx, gy = x - 5.0 * ux, y - 5.0 * uy                       # gate
        width = float(self.rng.uniform(2.4, 3.0))                # 8-10 ft (within the 6-10 ft rule)
        self.gate(m, gx, gy, hd, width, small=True)
        dist = float(self.rng.uniform(12.2, 18.0))               # 40-100 ft rule, kept compact
        bx, by = gx + dist * ux, gy + dist * uy
        self.add("blue", bx, by, m.id)
        hx, hy = gx - 4.5 * ux, gy - 4.5 * uy                     # holding bay behind the gate
        for side in (-1, 1):
            self.add("marker", hx - side * 2.4 * uy, hy + side * 2.4 * ux, m.id, r=0.12)
        self.add("light_panel", hx - 2.0 * uy + 3.0 * ux, hy + 2.0 * ux + 3.0 * uy, m.id, r=0.25)   # visible from the hold
        loop = [(hx, hy), (gx, gy)] + [(bx + 3.0 * math.cos(a), by + 3.0 * math.sin(a))
                                       for a in np.linspace(hd + math.pi / 2, hd + math.pi / 2 - 2 * math.pi, 25)] + [(gx, gy)]
        m.path = [(float(p), float(q)) for p, q in loop]
        n_b = int(self.rng.integers(d["n_black4"][0], d["n_black4"][1] + 1))
        placed = 0
        for _ in range(300):
            if placed >= n_b:
                break
            t = self.rng.uniform(0.15, 1.0)
            lat = self.rng.choice([-1, 1]) * self.rng.uniform(2.4, 5.5)
            px, py = gx + t * dist * ux - lat * uy, gy + t * dist * uy + lat * ux
            if min(math.hypot(px - o.x, py - o.y) for o in self.objs if o.module == m.id) > 2.0:
                self.add("black", px, py, m.id); placed += 1
        m.data.update(n_black=placed, gate_width=width, buoy_dist=dist,
                      light_delay=float(stream(self.seed, "light").uniform(3.0, 12.0)))

    def t5_vessels(self):
        d = self.d
        n_o = int(self.rng.integers(d["n_orange"][0], d["n_orange"][1] + 1))
        n_b = int(self.rng.integers(d["n_blackv"][0], d["n_blackv"][1] + 1))
        for cls, mark, n in (("vessel_orange", "triangle", n_o), ("vessel_black", "plus", n_b)):
            for _ in range(n):
                x, y, hd = self.place(3.0, 18.0)
                m = self.module("T5", x, y, hd, 3.0)
                self.add(cls, x, y, m.id, r=0.8, heading=hd, length=1.6, mark=mark)

    def return_rush(self, sx):
        """Braess network built from real course elements (see module docstring):
        S (task-area exit, north of the spill) -> west corridor -> A -> T (home gate) and S -> east corridor -> B -> T.
        West-upper and east-lower corridor halves are single-file channels squeezed between the spill boom and a
        marina pier with moored boats; the other halves are open water. A channel through the middle of the spill
        (closable boom gates) is the shortcut A -> B. Positions are jittered procedurally."""
        r = self.rng
        W, H = self.W, self.H
        x0, x1 = W * 0.5 - r.uniform(12.5, 15.0), W * 0.5 + r.uniform(12.5, 15.0)   # spill zone x-extent
        y0, y1 = r.uniform(16.0, 19.0), r.uniform(46.0, 50.0)                        # spill zone y-extent
        ym = (y0 + y1) / 2 + r.uniform(-1.5, 1.5)                                    # shortcut channel centre
        half = r.uniform(1.8, 2.2)                                                   # shortcut half-width
        self.task_floor = y1 + 6.0

        def noisy(pts, amp=0.35, step=1.2):
            out = []
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                L = math.hypot(bx - ax, by - ay); n = max(1, int(L / step))
                for k in range(n):
                    f = k / n
                    nx_, ny_ = -(by - ay) / L, (bx - ax) / L
                    j = r.uniform(-amp, amp)
                    out.append((ax + f * (bx - ax) + j * nx_, ay + f * (by - ay) + j * ny_))
            out.append(pts[-1])
            return out
        # two spill lobes (north and south of the shortcut channel), each enclosed by boom
        north = [(x0, ym + half), (x0, y1), (x1, y1), (x1, ym + half)]
        south = [(x0, ym - half), (x0, y0), (x1, y0), (x1, ym - half)]
        self.booms = [noisy(north, 0.12, 3.0), noisy(south, 0.12, 3.0), noisy([(x0, ym + half), (x1, ym + half)], 0.1, 3.0),
                      noisy([(x0, ym - half), (x1, ym - half)], 0.1, 3.0)]
        self.gates_rr = [[(x0, ym - half), (x0, ym + half)], [(x1, ym - half), (x1, ym + half)]]   # closable ends
        # black spill buoys inside the lobes
        for (ya, yb) in ((ym + half + 1.5, y1 - 1.5), (y0 + 1.5, ym - half - 1.5)):
            for _ in range(int(r.integers(5, 9))):
                self.add("black", r.uniform(x0 + 1.5, x1 - 1.5), r.uniform(ya, yb), r=0.10)
        # marinas: piers from the shore leaving a single-file channel next to the boom
        lane = r.uniform(2.9, 3.3)
        cube = 0.508

        def marina(side, y):
            xa, xb = (0.0, x0 - lane) if side == "W" else (x1 + lane, W)
            k = 0
            xx = xa
            while xx < xb - 1e-6:
                for dy in (0.0, cube):
                    self.piers.append((xx + cube / 2, y + dy, cube, 0.0))
                xx += cube
            # moored boats on the open-water side of the pier, random count and spacing
            for _ in range(int(r.integers(3, 6))):
                mx = r.uniform(xa + 2.0, xb - 2.0)
                self.add("moored", mx, y + (2.2 if side == "W" else -1.7), r=0.8,
                         heading=math.pi / 2 + r.uniform(-0.2, 0.2), length=1.7)
            # channel marker gate at the pier head (red/green, jittered)
            gx = (x0 - lane / 2) if side == "W" else (x1 + lane / 2)
            self.add("red", gx - lane / 2 + 0.25 + r.uniform(-0.1, 0.1), y + 0.5 + r.uniform(-0.3, 0.3))
            self.add("green", gx + lane / 2 - 0.25 + r.uniform(-0.1, 0.1), y + 0.5 + r.uniform(-0.3, 0.3))
        marina("W", (ym + y1) / 2 + r.uniform(-2, 2))
        marina("E", (y0 + ym) / 2 + r.uniform(-2, 2))
        self.nodes = dict(S=(W * 0.5, y1 + 3.0), A=(x0 - lane / 2, ym), B=(x1 + lane / 2, ym), T=(W * 0.5, y0 - 4.0))


def catmull_rom(ctrl, per_seg=20):
    P = [ctrl[0]] + list(ctrl) + [ctrl[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = map(np.array, (P[i - 1], P[i], P[i + 1], P[i + 2]))
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(q[0]), float(q[1])))
    out.append(tuple(map(float, ctrl[-1])))
    return out


def arc_length(pts):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])))


def point_at(pts, s):
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= s:
            f = (s - acc) / max(L, 1e-9)
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), math.atan2(b[1] - a[1], b[0] - a[0])
        acc += L
    a, b = pts[-2], pts[-1]
    return b, math.atan2(b[1] - a[1], b[0] - a[0])


def max_curvature(pts):
    P = np.array(pts)
    d1 = np.gradient(P, axis=0); d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
    return float(np.max(num / den))


def compile_chart(course: Course, gc: GenConfig) -> World:
    """Planning grid with no land: buoys/vessels as point obstacles, dock cubes as polygons."""
    cfg = SimConfig(grid_res_m=gc.grid_res, hull_radius_m=gc.hull_radius, inflate_margin_m=gc.plan_margin)
    piers = []
    for (cx, cy, s, hd) in course.module("T3")[0].cubes if course.module("T3") else []:
        c, sn = math.cos(hd), math.sin(hd)
        h = s / 2
        piers.append(np.array([[cx + (a * c - b * sn), cy + (a * sn + b * c)] for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))]))
    for (cx, cy, s_, hd) in course.piers:
        h = s_ / 2
        piers.append(np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]))
    segs = list(course.booms) + ([] if course.shortcut_open else list(course.boom_gates))
    for line in segs:
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            L = math.hypot(bx - ax, by - ay) + 1e-9; nx_, ny_ = -(by - ay) / L * 0.2, (bx - ax) / L * 0.2
            piers.append(np.array([[ax + nx_, ay + ny_], [bx + nx_, by + ny_], [bx - nx_, by - ny_], [ax - nx_, ay - ny_]]))
    w = World(cfg, course.seed, course.width, course.height, [], [], piers, [], 0.0, course.start[:2], course.current)
    w.point_obstacles = [(o.x, o.y, o.r) for o in course.objects]
    w.build_grid()
    dist, _ = w.field_to(*course.start[:2])
    w._launch_reach = np.isfinite(dist)
    return w


def entry_points(m: Module):
    if m.kind == "T3":
        return [b.entry for b in m.bays]
    if m.kind in ("T1", "T6", "T2", "T4"):
        return [m.path[0]] if m.path else [(m.x, m.y)]
    return [(m.x + 2.5 * math.cos(m.heading + math.pi / 2), m.y + 2.5 * math.sin(m.heading + math.pi / 2))]


def generate_course(seed: int, gc: Optional[GenConfig] = None) -> Course:
    gc = gc or GenConfig()
    for attempt in range(60):
        b = _Builder(seed * 1000 + attempt, gc)
        b.seed = seed
        sx, sy = b.W * 0.32, 3.0
        crng = stream(seed, "current")
        day_shape = str(SHAPES[int(b.rng.integers(0, len(SHAPES)))])
        day_color = str(COLORS[int(b.rng.integers(0, len(COLORS)))])
        try:
            b.t1_t6(sx, sy)
            if gc.variant == "return_rush":
                b.return_rush(sx)
            b.t3_dock(day_shape, day_color)
            b.t2_path()
            b.t4_speed()
            b.t5_vessels()
        except RuntimeError:
            continue
        for (x, y) in ((1.0, 1.0), (b.W - 1.0, 1.0), (1.0, b.H - 1.0), (b.W - 1.0, b.H - 1.0)):
            b.add("marker", x, y, r=0.3)
        ang = crng.uniform(0, 2 * math.pi)
        cur = CurrentField(np.array([math.cos(ang), math.sin(ang)]) * gc.current_strength * 0.5,
                           np.array([[crng.uniform(0.2, 0.8) * b.W, crng.uniform(0.3, 0.8) * b.H, 22.0,
                                      crng.choice([-1, 1]) * gc.current_strength]]),
                           np.zeros((0, 5)), unk_amp=0.3 * gc.current_strength,
                           unk_period=float(crng.uniform(70, 130)), unk_phase=float(crng.uniform(0, 6.28)))
        course = Course(seed, b.W, b.H, (sx, sy, math.pi / 2), b.objs, b.mods, day_shape, day_color, cur,
                        gc.difficulty, gc.variant)
        course.booms, course.boom_gates, course.piers, course.rush_nodes = b.booms, b.gates_rr, b.piers, b.nodes
        chart = compile_chart(course, gc)
        if all(chart.reachable(*p) for m in course.modules for p in entry_points(m)):
            course.data_attempt = attempt
            return course
    raise RuntimeError(f"no valid course for seed {seed}")
