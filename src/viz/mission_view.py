"""Renders a Simulation (read-only) as a quiet chart that foregrounds planning:
committed routes with predicted-time ticks, the options each boat weighed at its last
decision, claims, message age, and (optionally) what one boat believes about the others."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pygame

from ..policies import POLICY_INFO, make_policy
from ..tasks import RADIUS
from . import fx
from . import theme as T

TICK_S = 10.0        # predicted-time tick spacing along routes
FLEET_NAME = {"P0": "Greedy fleet", "P1": "Selfish fleet", "P2": "Graph-ETA fleet", "A1": "Graph-ETA fleet, age-aware",
              "A2": "Control-aware fleet, naive", "P3": "Control-aware fleet", "P4": "Centralized fleet"}
SHORT = {"P0": "Greedy nearest", "P1": "Selfish race", "P2": "Graph ETA", "A1": "Graph ETA, age-aware",
         "A2": "Control-aware, naive", "P3": "Control-aware", "P4": "Centralized"}


class Camera:
    def __init__(self, rect: pygame.Rect, world, zoom=1.0):
        self.rect = pygame.Rect(rect)
        self.world = world
        self.fit = min((self.rect.w - 40) / world.width, (self.rect.h - 40) / world.height)
        self.zoom = zoom
        self.cx, self.cy = world.width / 2, world.height / 2

    @property
    def scale(self):
        return self.fit * self.zoom

    @property
    def vx(self): return self.rect.x

    @property
    def vy(self): return self.rect.y

    @property
    def vw(self): return self.rect.w

    @property
    def vh(self): return self.rect.h

    def to_screen(self, x, y):
        s = self.scale
        return (int(self.rect.x + self.rect.w / 2 + (x - self.cx) * s),
                int(self.rect.y + self.rect.h / 2 - (y - self.cy) * s))

    def to_world(self, X, Y):
        s = self.scale
        return (self.cx + (X - self.rect.x - self.rect.w / 2) / s,
                self.cy - (Y - self.rect.y - self.rect.h / 2) / s)


def text(screen, s, pos, size=13, color=T.INK, bold=False, anchor="tl"):
    img = T.font(size, bold).render(str(s), True, color)
    r = img.get_rect()
    setattr(r, {"tl": "topleft", "tr": "topright", "c": "center", "ml": "midleft", "mr": "midright",
                "bl": "bottomleft", "mb": "midbottom"}[anchor], pos)
    screen.blit(img, r)
    return r


def label_box(screen, s, pos, size=11, color=T.INK, anchor="ml"):
    """Small label on a paper chip so it stays legible over routes."""
    img = T.font(size).render(s, True, color)
    r = img.get_rect().inflate(8, 4)
    setattr(r, {"ml": "midleft", "mr": "midright", "c": "center", "mb": "midbottom"}[anchor], pos)
    pygame.draw.rect(screen, T.PAPER, r, border_radius=4)
    pygame.draw.rect(screen, T.RULE, r, 1, border_radius=4)
    screen.blit(img, (r.x + 4, r.y + 2))
    return r


def card(screen, rect, fill=T.PAPER):
    pygame.draw.rect(screen, fill, rect, border_radius=8)
    pygame.draw.rect(screen, T.RULE, rect, 1, border_radius=8)


def sparkline(screen, rect, series, color, zero=True):
    if len(series) < 2:
        return
    ys = np.asarray(series, float)
    lo, hi = min(0.0 if zero else ys.min(), ys.min()), max(1.0, ys.max())
    pts = [(rect.x + rect.w * k / (len(ys) - 1), rect.bottom - rect.h * (v - lo) / (hi - lo)) for k, v in enumerate(ys)]
    if lo < 0 < hi:
        y0 = rect.bottom - rect.h * (0 - lo) / (hi - lo)
        pygame.draw.line(screen, T.RULE, (rect.x, y0), (rect.right, y0), 1)
    pygame.draw.aalines(screen, color, False, pts)
    pygame.draw.circle(screen, color, pts[-1], 3)


class PlanProbe:
    """Re-evaluates a boat's OWN policy on its OWN observation (read-only) to show which
    options it weighed. Same code path as the decision; nothing is fed back to the sim."""

    def __init__(self, sim):
        self.sim = sim
        self.enabled = not make_policy(sim.cfg.policy).centralized
        self.last: Dict[int, dict] = {}
        self.next_ok: Dict[int, float] = {}

    def probe(self, i, now_vis):
        if not self.enabled or now_vis < self.next_ok.get(i, 0):
            return
        self.next_ok[i] = now_vis + 1.0
        pol = make_policy(self.sim.cfg.policy)
        got = {}
        orig = pol._choose

        def capture(obs, utils, etas, cache):
            got.update(utils)
            return orig(obs, utils, etas, cache)
        pol._choose = capture
        d = pol.decide(self.sim.observe(i))
        if self.sim.cfg.policy == "P0":
            got = {j: -(1000.0 - u) for j, u in got.items()}       # greedy: show −ETA
        self.last[i] = dict(t_vis=now_vis, chosen=d.task, utils=got)


class MissionView:
    def __init__(self, sim, rect: pygame.Rect, label: str = "", label_right: bool = False):
        self.sim = sim
        self.cam = Camera(rect, sim.world)
        self.label, self.label_right = label, label_right
        self.static: Optional[pygame.Surface] = None
        self.static_key = None
        self.fx = fx.Effects()
        self.probe = PlanProbe(sim)
        self.ev_idx = 0
        self.layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        self.history: List[tuple] = []
        self.trails: List[List[tuple]] = [[] for _ in range(sim.n)]
        self.last_trail_t = -1.0
        self.last_t = -1.0
        self.vis_t = 0.0
        self.show_sensors = False
        self.show_graph = False         # planning overlay: routes, time ticks, ETA labels (key G)
        self.show_options = False       # options weighed at the last decision (key G)
        self.ghosts = []                # [(sim, policy, colour)] phantom fleets on the same world
        self.gone_vis = {}
        self.follow = False
        self.follow_zoom = 1.45
        self.follow_lerp = 0.02
        self.minimap = None
        self.show_flow = True           # kept for key compatibility
        self.belief_of: Optional[int] = None
        self._pending_probe = set()

    def resize(self, rect):
        self.cam.rect = pygame.Rect(rect)
        self.cam.fit = min((rect.w - 40) / self.sim.world.width, (rect.h - 40) / self.sim.world.height)
        self.layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        self.static = None

    # ------------------------------------------------------------------------------
    def consume_events(self):
        sim = self.sim
        for e in sim.events[self.ev_idx:]:
            if e.kind == "complete":
                col = T.BOAT_COLORS[e.boat % 16]
                self.fx.ring(e.x, e.y, self.vis_t, col, r_max=7.0, life=1.0)
                self.fx.label(e.x, e.y, self.vis_t, f"+{int(e.value)}", T.shade(col, 0.8))
            elif e.kind in ("collision", "obstacle"):
                self.fx.ring(e.x, e.y, self.vis_t, T.BAD, r_max=3.0, life=0.8)
            elif e.kind == "redundant":
                self.fx.label(e.x, e.y, self.vis_t, "already done", T.INK2, life=1.4)
            elif e.kind == "expire":
                self.fx.label(e.x, e.y, self.vis_t, "expired", T.BAD, life=1.4)
            elif e.kind == "decide" and e.boat >= 0:
                self._pending_probe.add(e.boat)
        self.ev_idx = len(sim.events)

    def follow_step(self):
        """Adaptive follow camera: frame the padded box of the real boats and their targets.
        Zoom eases within [1.10, 1.45]; a 40 px dead-zone prevents drift; clamped to the course."""
        sim, cam = self.sim, self.cam
        xs = list(map(float, sim.S[0])); ys = list(map(float, sim.S[1]))
        for m in sim.minds:
            if m.claim is not None:
                tk = sim.tasks[m.claim]; xs.append(tk.x); ys.append(tk.y)
        pad = 80.0
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        zx = (cam.rect.w - 2 * pad) / max(1e-6, bw * cam.fit)
        zy = (cam.rect.h - 2 * pad) / max(1e-6, bh * cam.fit)
        z_t = min(1.45, max(1.10, min(zx, zy)))
        cam.zoom += (z_t - cam.zoom) * self.follow_lerp * 2
        tx, ty = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if math.hypot((tx - cam.cx) * cam.scale, (ty - cam.cy) * cam.scale) > 40 or getattr(self, "_moving", False):
            self._moving = math.hypot((tx - cam.cx) * cam.scale, (ty - cam.cy) * cam.scale) > 4
            cam.cx += (tx - cam.cx) * self.follow_lerp
            cam.cy += (ty - cam.cy) * self.follow_lerp
        W, H = sim.world.width, sim.world.height
        m = 24.0 / cam.scale
        hw, hh = cam.rect.w / 2 / cam.scale, cam.rect.h / 2 / cam.scale
        cam.cx = min(max(cam.cx, hw - m), W - hw + m) if 2 * hw < W + 2 * m else W / 2
        cam.cy = min(max(cam.cy, hh - m), H - hh + m) if 2 * hh < H + 2 * m else H / 2

    def update(self, real_dt, sim_dt):
        self.vis_t += real_dt
        if self.follow:
            self.follow_step()
        self.consume_events()
        if self.show_options:
            for i in sorted(self._pending_probe):
                self.probe.probe(i, self.vis_t)
        self._pending_probe = set()
        sim = self.sim
        if sim.t - self.last_trail_t >= 0.25:
            for i in range(sim.n):
                self.trails[i].append((float(sim.S[0, i]), float(sim.S[1, i])))
                if len(self.trails[i]) > 120:
                    self.trails[i].pop(0)
            self.last_trail_t = sim.t
        if sim.t - self.last_t >= 1.0 or not self.history:
            self.history.append((sim.t, self.live_metrics()["welfare"]))
            self.last_t = sim.t

    def live_metrics(self):
        return MissionView.metrics_of(self.sim)

    @staticmethod
    def metrics_of(sim):
        cfg = sim.cfg
        pts = sum(tk.value for tk in sim.tasks if tk.completed_by is not None)
        coll = sum(1 for e in sim.events if e.kind in ("collision", "obstacle"))
        en = float(sim.energy.sum())
        return dict(points=pts, collisions=coll, energy=en,
                    welfare=pts - cfg.w_energy * en - cfg.w_collision * coll,
                    completed=sum(1 for tk in sim.tasks if tk.completed_by is not None),
                    expired=sum(1 for tk in sim.tasks if tk.completed_by is None and sim.t > tk.deadline),
                    redundant=sum(1 for e in sim.events if e.kind == "redundant"))

    # ------------------------------------------------------------------------------
    def draw(self, screen):
        sim, cam = self.sim, self.cam
        vp = cam.rect
        screen.set_clip(vp)
        screen.fill(T.PAPER, vp)
        key = (round(cam.scale, 3),)
        if self.static is None or self.static_key != key:
            self.static = fx.render_static(sim.world, cam.scale)
            self.static_key = key
        ox, oy = cam.to_screen(0, sim.world.height)
        screen.fill(T.OUTER_LAND, vp)
        ar = pygame.Rect(ox, oy, self.static.get_width(), self.static.get_height())
        pygame.draw.rect(screen, T.LAND_EDGE, ar.inflate(int(3.2 * cam.scale), int(3.2 * cam.scale)), border_radius=int(3 * cam.scale))
        pygame.draw.rect(screen, T.SAND, ar.inflate(int(2.4 * cam.scale), int(2.4 * cam.scale)), border_radius=int(3 * cam.scale))
        screen.blit(self.static, (ox, oy))
        L = self.layer
        L.fill((0, 0, 0, 0))
        loc = Camera(pygame.Rect(0, 0, vp.w, vp.h), sim.world)
        loc.cx, loc.cy, loc.zoom, loc.fit = cam.cx, cam.cy, cam.zoom, cam.fit
        self._draw_channels(L, loc)
        if self.show_graph:
            self._draw_comms(L, loc)
        self._draw_ghosts(L, loc)
        self._draw_options(L, loc)
        if self.show_graph:
            self._draw_routes(L, loc)
        self._draw_tasks(L, loc)
        self._draw_beliefs(L, loc)
        self._draw_boats(L, loc)
        self.fx.draw(L, loc, self.vis_t)
        self._draw_route_labels(L, loc)
        screen.blit(L, vp.topleft)
        if self.label:
            img = T.font(15, True).render(self.label, True, T.INK)
            x = vp.x + 16 if not self.label_right else vp.right - img.get_width() - 16
            screen.blit(img, (x, vp.y + 12))
        screen.set_clip(None)

    def draw_minimap(self, screen, r):
        """Course overview for the side panel (real boats and tasks only)."""
        sim, cam = self.sim, self.cam
        W, H = sim.world.width, sim.world.height
        if self.minimap is None or self.minimap.get_size() != r.size:
            self.minimap = pygame.transform.smoothscale(fx.render_static(sim.world, 2.0), r.size)
        screen.blit(self.minimap, r.topleft)
        pygame.draw.rect(screen, T.LAND_EDGE, r, 1)
        sx, sy = r.w / W, r.h / H
        for tk in sim.tasks:
            if tk.release <= sim.t and (tk.completed_by is not None or tk.active(sim.t)):
                c = T.BOAT_COLORS[tk.completed_by % 16] if tk.completed_by is not None else (255, 255, 255)
                pygame.draw.circle(screen, c, (r.x + tk.x * sx, r.y + (H - tk.y) * sy), 2)
        for i in range(sim.n):
            p = (r.x + sim.S[0, i] * sx, r.y + (H - sim.S[1, i]) * sy)
            pygame.draw.circle(screen, (12, 22, 40), p, 4)
            pygame.draw.circle(screen, T.BOAT_COLORS[i % 16], p, 3)
        if self.follow:
            vw, vh = cam.rect.w / cam.scale, cam.rect.h / cam.scale
            vr = pygame.Rect(r.x + (cam.cx - vw / 2) * sx, r.y + (H - cam.cy - vh / 2) * sy, vw * sx, vh * sy).clip(r)
            pygame.draw.rect(screen, (200, 210, 220), vr, 1)

    def _boat_len(self, cam):
        return max(38.0, 1.8 * cam.scale)

    def _draw_channels(self, L, cam):
        sim = self.sim
        for ch in sim.world.channels:
            occ = any(ch.contains(sim.S[0, i], sim.S[1, i], pad=0.5) for i in range(sim.n))
            c = cam.to_screen(ch.x1 + 2.2, ch.center[1])
            pygame.draw.circle(L, T.BAD if occ else T.GOOD, (c[0], c[1]), 4)
            img = T.font(10).render("one lane", True, T.INK2)
            L.blit(img, (c[0] + 7, c[1] - img.get_height() / 2))

    def _draw_trails(self, L, cam):
        for i, tr in enumerate(self.trails):
            if len(tr) < 2:
                continue
            pts = [cam.to_screen(x, y) for x, y in tr[-40:]]
            n = len(pts)
            for k in range(1, n):
                pygame.draw.line(L, (*T.WAKE, int(110 * k / n)), pts[k - 1], pts[k], 2 if k > n - 8 else 1)

    def _route_pts(self, i, cam):
        sim = self.sim
        g = sim.minds[i].guidance
        if g is None:
            return None, None
        w = [(float(sim.S[0, i]), float(sim.S[1, i]))] + [tuple(p) for p in g.wps[g.seg + 1:]]
        return w, [cam.to_screen(*p) for p in w]

    def _draw_routes(self, L, cam):
        sim = self.sim
        v = sim.tables.v_ss
        for i, m in enumerate(sim.minds):
            w, pts = self._route_pts(i, cam)
            if not pts or len(pts) < 2:
                continue
            col = T.BOAT_COLORS[i % 16]
            if m.claim is None:
                for a, b in zip(pts, pts[1:]):
                    fx.dashed_line(L, (*col, 120), a, b, 4, 5)
                continue
            fx.thick_polyline(L, (*col, 200), pts, 3)
            for p in pts[1:-1]:
                pygame.draw.circle(L, T.PAPER, p, 3)
                pygame.draw.circle(L, col, p, 3, 1)
            # predicted-time ticks: where the boat should be every TICK_S seconds
            acc, nxt = 0.0, TICK_S * v
            for (a, b), (A, B) in zip(zip(w, w[1:]), zip(pts, pts[1:])):
                seg = math.hypot(b[0] - a[0], b[1] - a[1])
                while seg > 0 and acc + seg >= nxt:
                    f = (nxt - acc) / seg
                    px, py = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
                    dx, dy = B[0] - A[0], B[1] - A[1]
                    ln = math.hypot(dx, dy) or 1
                    nx, ny = -dy / ln * 5, dx / ln * 5
                    pygame.draw.line(L, T.shade(col, 0.7), (px - nx, py - ny), (px + nx, py + ny), 2)
                    nxt += TICK_S * v
                acc += seg
            end = pts[-1]
            pygame.draw.circle(L, T.PAPER, end, 6)
            pygame.draw.circle(L, col, end, 6, 2)

    def _draw_route_labels(self, L, cam):
        if not self.show_graph:
            return
        sim = self.sim
        for i, m in enumerate(sim.minds):
            if m.claim is None or not m.pred:
                continue
            w, pts = self._route_pts(i, cam)
            if not pts:
                continue
            rem = max(0.0, m.pred["eta_control"] - (sim.t - m.pred["t"]))
            label_box(L, f"{rem:.0f} s", (pts[-1][0] + 10, pts[-1][1] + 12), 11, T.shade(T.BOAT_COLORS[i % 16], 0.75))

    def _draw_options(self, L, cam):
        """Runner-up options from the last decision: dashed rays labelled with utility."""
        if not self.show_options or not self.probe.enabled:
            return
        sim = self.sim
        for i, p in self.probe.last.items():
            a = max(0.0, 1.0 - (self.vis_t - p["t_vis"]) / 1.8)
            if a <= 0 or not p["utils"]:
                continue
            col = T.BOAT_COLORS[i % 16]
            ranked = sorted(p["utils"].items(), key=lambda kv: -kv[1])
            alts = [(j, u) for j, u in ranked if j != p["chosen"] and u > 0][:2]
            bx, by = cam.to_screen(sim.S[0, i], sim.S[1, i])
            for j, u in alts:
                tk = sim.tasks[j]
                q = cam.to_screen(tk.x, tk.y)
                fx.dashed_line(L, (*col, int(170 * a)), (bx, by), q, 3, 6)
                mx, my = bx + (q[0] - bx) * 0.62, by + (q[1] - by) * 0.62
                img = T.font(10).render(f"{u:.1f}", True, T.shade(col, 0.7))
                img.set_alpha(int(255 * a))
                L.blit(img, (mx + 4, my - 12))

    def _draw_tasks(self, L, cam):
        """Course objects: buoy gates, inspection rings, docks, sampling buoys. A navy pill
        shows the points (orange when < 15 s remain); boat-coloured dots mark who is going.
        Completed and expired tasks fade out."""
        sim = self.sim
        t = sim.t
        for tk in sim.tasks:
            if tk.release > t:
                continue
            done = tk.completed_by is not None
            expired = (not done) and t > tk.deadline
            if done or expired:
                t0 = self.gone_vis.setdefault(tk.id, self.vis_t)
                fade = max(0.0, 1.0 - (self.vis_t - t0) / 1.5)
                if fade <= 0:
                    if tk.kind == "gate":       # buoys stay on the course
                        for p_, c_ in zip((cam.to_screen(*b) for b in tk.buoys()), ((150, 90, 85), (80, 140, 100))):
                            pygame.draw.circle(L, c_, p_, 4)
                    self._pennants(L, cam, tk)
                    continue
            else:
                fade = 1.0
            A = int(255 * fade)
            anchor = cam.to_screen(tk.x, tk.y)
            if tk.kind == "gate":
                (rx, ry), (gx, gy) = tk.buoys()
                a, b = cam.to_screen(rx, ry), cam.to_screen(gx, gy)
                fx.dashed_line(L, (255, 255, 255, int(110 * fade)), a, b, 3, 4)
                for p_, c_ in ((a, T.RED_BUOY), (b, T.GREEN_BUOY)):
                    pygame.draw.circle(L, (255, 255, 255, A), p_, 7)
                    pygame.draw.circle(L, (*c_, A), p_, 5)
            elif tk.kind == "inspect":
                pygame.draw.circle(L, (*T.WAKE, A), anchor, 12, 2)
                pygame.draw.circle(L, (*T.WAKE, A), anchor, 2)
            elif tk.kind == "dock":
                rr = pygame.Rect(0, 0, 14, 14); rr.center = anchor
                pygame.draw.rect(L, (*T.WAKE, A), rr, 2, border_radius=2)
            else:
                yel = (255, 200, 60)
                pygame.draw.circle(L, (255, 255, 255, A), anchor, 7)
                pygame.draw.circle(L, (*yel, A), anchor, 5)
            if done:
                continue
            if expired:
                continue
            self._pennants(L, cam, tk)
            left = tk.deadline - t
            img = T.font(11, True).render(f"{tk.value}", True, T.INK)
            claim = [i for i, m in enumerate(sim.minds) if m.claim == tk.id]
            pw_ = max(28, img.get_width() + 14 + 8 * len(claim))
            pill = pygame.Rect(0, 0, pw_, img.get_height() + 4)
            pill.midleft = (anchor[0] + 15, anchor[1] - 13)
            pygame.draw.rect(L, (*(T.WARN if left < 15 else T.PAPER), 235), pill, border_radius=6)
            L.blit(img, (pill.x + 7, pill.y + 2))
            for k, i in enumerate(claim):
                pygame.draw.rect(L, T.BOAT_COLORS[i % 16], pygame.Rect(pill.x + 9 + img.get_width() + 8 * k, pill.centery - 3, 6, 6))

    def _pennants(self, L, cam, tk):
        """Flags: solid for the playing fleet, hollow for phantom fleets that completed it."""
        b0 = cam.to_screen(tk.x, tk.y)
        base = (b0[0] + 18, b0[1] - 6)
        k0 = 0
        if tk.completed_by is not None:
            fx.pennant(L, base, T.BOAT_COLORS[tk.completed_by % 16]); k0 = 1
        for k, (gsim, pol, col) in enumerate(self.ghosts):
            gt = gsim.tasks[tk.id]
            if gt.completed_by is not None:
                fx.pennant(L, base, col, hollow=True, dx=11 * k0); k0 += 1

    def _draw_comms(self, L, cam):
        sim = self.sim
        t = sim.t
        for i in range(sim.n):
            for j in range(i + 1, sim.n):
                mi, mj = sim.minds[i].beliefs.get(j), sim.minds[j].beliefs.get(i)
                ages = [t - m.t_sent for m in (mi, mj) if m is not None]
                if not ages:
                    continue
                a = math.exp(-min(ages) / 4.0)
                if a < 0.08:
                    continue
                p = cam.to_screen(sim.S[0, i], sim.S[1, i]); q = cam.to_screen(sim.S[0, j], sim.S[1, j])
                pygame.draw.aaline(L, (*T.WAKE, int(70 * a)), p, q)
        for deliver_t, r, msg in sim.net.in_flight:
            f = min(1.0, max(0.0, (t - msg.t_sent) / max(1e-3, deliver_t - msg.t_sent)))
            a = cam.to_screen(sim.S[0, msg.sender], sim.S[1, msg.sender])
            b = cam.to_screen(sim.S[0, r], sim.S[1, r])
            pygame.draw.circle(L, T.BOAT_COLORS[msg.sender % 16], (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), 2)
        if len(sim.net.dropped_events) > 4000:
            del sim.net.dropped_events[:2000]

    def _draw_beliefs(self, L, cam):
        """What boat `belief_of` believes: teammates where their last message put them."""
        i = self.belief_of
        sim = self.sim
        if i is None or i >= sim.n:
            return
        for j, msg in sim.minds[i].beliefs.items():
            p = cam.to_screen(msg.x, msg.y)
            q = cam.to_screen(sim.S[0, j], sim.S[1, j])
            fx.dashed_line(L, (*T.INK2, 200), p, q, 2, 4)
            pygame.draw.aalines(L, T.BOAT_COLORS[j % 16], True, fx.hull_points(p, msg.psi, self._boat_len(cam)))
            label_box(L, f"{sim.t - msg.t_sent:.1f} s old", (p[0], p[1] - 14), 10, T.INK2, "mb")
        b = cam.to_screen(sim.S[0, i], sim.S[1, i])
        pygame.draw.circle(L, T.BOAT_COLORS[i % 16], b, int(self._boat_len(cam)), 1)

    def _draw_ghosts(self, L, cam):
        """Phantom fleets: the same world played by other strategies, in lockstep."""
        Lpx = self._boat_len(cam)
        for gsim, pol, col in self.ghosts:
            for i in range(gsim.n):
                p = cam.to_screen(float(gsim.S[0, i]), float(gsim.S[1, i]))
                fx.ghost_boat(L, p, float(gsim.S[2, i]), col, Lpx, dot=T.BOAT_COLORS[i % 16])

    def _draw_boats(self, L, cam):
        sim = self.sim
        Lpx = self._boat_len(cam)
        for i in range(sim.n):
            x, y, psi, v, r = sim.S[:, i]
            col = T.BOAT_COLORS[i % 16]
            p = cam.to_screen(x, y)
            if self.show_sensors:
                pygame.draw.circle(L, (*col, 60), p, int(sim.cfg.sensor_range_m * cam.scale), 1)
            if abs(v) > 0.3 * sim.cfg.v_max:
                fx.v_wake(L, p, psi, Lpx)
            fx.draw_boat(L, p, psi, col, Lpx)
            m = sim.minds[i]
            tag = f"{i + 1}"
            fnt = T.font(11, True)
            q = (p[0] + Lpx * 0.55, p[1] + Lpx * 0.2)
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                L.blit(fnt.render(tag, True, (12, 22, 40)), (q[0] + ox, q[1] + oy))
            L.blit(fnt.render(tag, True, (240, 244, 248)), q)


# ---------------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------------
def draw_panel(screen, rect, view: MissionView, extra_lines=()):
    """Restrained panel: who is planning what, and how the team is doing. Press D in the
    app for the diagnostic detail (ETA models, message ages)."""
    sim = view.sim
    cfg = sim.cfg
    pygame.draw.rect(screen, T.PAPER, rect)
    pygame.draw.line(screen, T.RULE, rect.topleft, rect.bottomleft, 1)
    x0, x1 = rect.x + 28, rect.right - 28
    y = rect.y + 26
    text(screen, FLEET_NAME.get(cfg.policy, cfg.policy), (x0, y), 20, T.INK, True)
    y += 30
    text(screen, f"strategy {cfg.policy}   comms {cfg.comm_mode}", (x0, y), 11, (120, 135, 160))
    y += 36
    m = view.live_metrics()
    text(screen, "Team welfare", (x0, y), 12, T.INK2)
    y += 18
    text(screen, f"{m['welfare']:+.1f}", (x0, y), 30, T.INK, True)
    sparkline(screen, pygame.Rect(x0 + 120, y + 6, x1 - x0 - 120, 30), [h[1] for h in view.history], T.ACCENT)
    y += 46
    tot = sum(tk.value for tk in sim.tasks)
    text(screen, f"{m['points']} of {tot} points   {m['completed']} done   {m['expired']} expired"
                 + (f"   {m['collisions']} contacts" if m["collisions"] else ""), (x0, y), 12, T.INK2)
    y += 38
    pygame.draw.line(screen, T.RULE, (x0, y), (x1, y))
    y += 22
    detail = getattr(view, "show_detail", False)
    rh = 64 if detail else 46
    rh = min(rh, max(26, (rect.bottom - y - 60) // max(1, sim.n)))
    for i in range(sim.n):
        if y + rh > rect.bottom - 18 * len(extra_lines) - 12:
            break
        mi = sim.minds[i]
        col = T.BOAT_COLORS[i % 16]
        fx.draw_boat(screen, (x0 + 8, y + 8), 0.0, col, 15)
        text(screen, f"{i + 1}", (x0 + 22, y), 13, T.INK, True)
        if mi.claim is not None:
            tk = sim.tasks[mi.claim]
            vimg = T.font(11, True).render(str(tk.value), True, T.INK)
            pr = pygame.Rect(x0 + 44, y + 1, max(22, vimg.get_width() + 10), 16)
            pygame.draw.rect(screen, (12, 22, 40), pr, border_radius=5)
            pygame.draw.rect(screen, T.RULE, pr, 1, border_radius=5)
            screen.blit(vimg, (pr.centerx - vimg.get_width() / 2, pr.y + 1))
            text(screen, T.TASK_LABEL[tk.kind], (pr.right + 8, y), 13, T.INK)
        else:
            text(screen, "repositioning", (x0 + 44, y), 13, T.INK2)
        p = mi.pred if mi.claim is not None else None
        if mi.holding:
            st = "waiting"
        elif mi.avoiding:
            st = "giving way"
        elif p:
            st = f"{max(0.0, p['eta_control'] - (sim.t - p['t'])):.0f} s"
        else:
            st = ""
        text(screen, st, (x1, y), 12, T.WARN if (mi.holding or mi.avoiding) else T.INK2, False, "tr")
        if rh >= 40 and p:
            frac = min(1.0, (sim.t - p["t"]) / max(1e-6, p["eta_control"]))
            pygame.draw.line(screen, T.RULE, (x0 + 44, y + 24), (x1, y + 24), 2)
            pygame.draw.line(screen, col, (x0 + 44, y + 24), (x0 + 44 + (x1 - x0 - 44) * frac, y + 24), 2)
        if detail and rh >= 60:
            ages = []
            for j in range(sim.n):
                if j != i:
                    b = mi.beliefs.get(j)
                    ages.append(f"{j + 1}: {sim.t - b.t_sent:.0f}s" if b else f"{j + 1}: –")
            line = (f"graph {p['eta_graph']:.0f}s, control {p['eta_control']:.0f}s   " if p else "") + "heard " + "  ".join(ages[:4])
            text(screen, line, (x0 + 44, y + 34), 11, T.INK2)
        y += rh
    if view.ghosts:
        y += 10
        pygame.draw.line(screen, T.RULE, (x0, y), (x1, y))
        y += 16
        text(screen, "Same course, other strategies (Δ vs playing)", (x0, y), 12, T.INK2)
        y += 24
        rows = [(cfg.policy, m["welfare"], None)]
        for gsim, pol, gcol in view.ghosts:
            gm = MissionView.metrics_of(gsim)
            rows.append((pol, gm["welfare"], gcol))
        for pol, w, gcol in rows:
            if gcol is None:
                for q in range(4):
                    pygame.draw.rect(screen, T.BOAT_COLORS[q], (x0 + 1 + 4 * q, y + 5, 4, 6))
                name = f"{SHORT.get(pol, pol)} (playing)"
            else:
                gs = pygame.Surface((24, 18), pygame.SRCALPHA)
                fx.ghost_boat(gs, (12, 9), 0.0, gcol, 16)
                screen.blit(gs, (x0 - 4, y - 1))
                name = f"{SHORT.get(pol, pol)} (phantom)"
            text(screen, f"{pol}  {name}", (x0 + 24, y), 12, T.INK if gcol is None else T.INK2)
            if gcol is None:
                text(screen, f"{w:+.1f}", (x1, y), 12, T.INK, True, "tr")
            else:
                dd = w - m["welfare"]
                dcol = (120, 135, 160) if abs(dd) < 0.05 else ((240, 110, 100) if dd > 0 else (120, 220, 140))
                text(screen, "tied" if abs(dd) < 0.05 else f"{dd:+.1f}", (x1, y), 12, dcol, True, "tr")
            y += 22
    # course overview fills the bottom of the panel
    mw = x1 - x0
    mh = int(mw * sim.world.height / sim.world.width)
    mr = pygame.Rect(x0, max(y + 30, rect.bottom - mh - 30), mw, mh)
    if mr.y > y + 20:
        text(screen, "Course", (x0, mr.y - 18), 11, (120, 135, 160))
        view.draw_minimap(screen, mr)
