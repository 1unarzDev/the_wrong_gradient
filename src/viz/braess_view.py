"""Mode A: the exact Braess / potential-game scene, drawn as a quiet diagram.
Every number shown is computed live by src.game_exact with exact fractions."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pygame

from ..game_exact import analyze, braess_game, four_cycle_sums, matching_pennies_cost
from . import fx
from . import theme as T
from .mission_view import text

NODE_POS = {"S": (0.12, 0.52), "A": (0.50, 0.20), "B": (0.50, 0.84), "T": (0.88, 0.52)}
NODE_NAME = {"S": "Dock", "A": "Buoy A", "B": "Buoy B", "T": "Inspection area"}


def PRETTY(route: str) -> str:
    names = {"S": "Dock", "A": "A", "B": "B", "T": "Insp."}
    return " → ".join(names[c] for c in route)


def fr(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


class BraessScene:
    def __init__(self):
        self.shortcut = True
        self.marginal = False
        self.autoplay = False
        self.show_cycle = False
        self.t = 0.0
        self.auto_timer = 0.0
        self.caption = None          # (line1, line2, colour, t0)
        self.reset()

    def reset(self):
        self.game = braess_game(self.shortcut, 4)
        g = self.game
        sat = g.routes.index((("S", "A"), ("A", "T")))
        sbt = g.routes.index((("S", "B"), ("B", "T")))
        self.profile = [sat, sat, sbt, sbt]
        self.ptr = 0
        self.history = [(self.phi(), g.team_cost(self.profile))]
        self.last = None
        self.at_ne = False
        self.analysis = analyze(g, g.marginal_cost if self.marginal else None)
        self.opt = analyze(g).optimum_cost
        self.progress = [k * 0.23 for k in range(4)]
        self.caption = None

    def ecost(self):
        return self.game.marginal_cost if self.marginal else None

    def phi(self):
        return self.game.potential(self.profile, self.ecost())

    def step(self):
        g = self.game
        for k in range(4):
            i = (self.ptr + k) % 4
            old = g.private_cost(self.profile, i, self.ecost())
            r, new = g.best_response(self.profile, i, self.ecost())
            if new < old:
                C0, P0 = g.team_cost(self.profile), self.phi()
                old_r = self.profile[i]
                self.profile[i] = r
                C1, P1 = g.team_cost(self.profile), self.phi()
                self.history.append((P1, C1))
                self.last = dict(i=i, old=old_r, new=r, c_old=old, c_new=new, C0=C0, C1=C1, P0=P0, P1=P1)
                self.ptr = (i + 1) % 4
                self.progress[i] = 0.0
                if C1 > C0:
                    self.caption = (f"Boat {i + 1} saves {fr(old - new)} min for itself.",
                                    f"The fleet loses {fr(C1 - C0)} min.", T.BAD, self.t)
                else:
                    self.caption = (f"Boat {i + 1} improves by {fr(old - new)}.",
                                    f"The fleet gains {fr(C0 - C1)} min.", T.GOOD, self.t)
                return True
        self.at_ne = True
        self.caption = ("Nash equilibrium.", "No boat can improve by changing route alone.", T.INK2, self.t)
        return False

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.step()
        elif key == pygame.K_a:
            self.autoplay = not self.autoplay
        elif key == pygame.K_m:
            self.marginal = not self.marginal
            self.analysis = analyze(self.game, self.game.marginal_cost if self.marginal else None)
            self.history = [(self.phi(), self.game.team_cost(self.profile))]
            self.at_ne = False; self.last = None
            self.caption = (("Marginal-cost pricing on." if self.marginal else "Selfish costs."),
                            ("Each boat pays m(k) = k c(k) − (k−1) c(k−1)." if self.marginal
                             else "Each boat pays its own travel time."), T.INK2, self.t)
        elif key == pygame.K_o:
            self.shortcut = not self.shortcut
            self.reset()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_4:
            self.show_cycle = not self.show_cycle

    def update(self, dt):
        self.t += dt
        if self.autoplay and not self.at_ne:
            self.auto_timer += dt
            if self.auto_timer > 2.2:
                self.auto_timer = 0
                self.step()
        g = self.game
        loads = g.loads(self.profile)
        for i in range(4):
            total = float(sum(g.c(e, loads[e]) for e in g.routes[self.profile[i]]))
            self.progress[i] = (self.progress[i] + dt / (0.25 * total)) % 1.0

    # ------------------------------------------------------------------------------
    def _pos(self, rect, n):
        fx_, fy = NODE_POS[n]
        return np.array((rect.x + rect.w * fx_, rect.y + rect.h * fy))

    def _edge_pts(self, rect, e, k=60):
        a, b = self._pos(rect, e[0]), self._pos(rect, e[1])
        if e == ("A", "B"):
            return np.linspace(a, b, k)
        mid, d = (a + b) / 2, b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        c0 = np.array([rect.centerx, rect.centery])
        if np.dot(nrm, mid - c0) < 0:
            nrm = -nrm
        ctrl = mid + nrm * 0.12 * np.linalg.norm(d)
        ts = np.linspace(0, 1, k)[:, None]
        return (1 - ts) ** 2 * a + 2 * (1 - ts) * ts * ctrl + ts ** 2 * b

    def draw(self, screen, rect, panel):
        g = self.game
        screen.set_clip(rect)
        screen.fill(T.OUTER_LAND, rect)
        loads = g.loads(self.profile)
        ec = self.ecost() or g.c
        mover_edges = set(g.routes[self.last["new"]]) if self.last else set()
        mover_col = T.BOAT_COLORS[self.last["i"]] if self.last else None
        W_CANAL = 34
        # canals: sand bank, surf line, water (wider for open-lake legs)
        layers = ((T.LAND_EDGE, 12), (T.SHALLOW, 6), (T.WATER, 0))
        for col, extra in layers:
            for e in g.edges:
                w = W_CANAL + (14 if e.label == "10" else 0) + extra
                fx.ribbon(screen, col, self._edge_pts(rect, e.key), w)
            for nname in NODE_POS:
                pygame.draw.circle(screen, col, self._pos(rect, nname), 38 + extra // 2)
        for e in g.edges:
            pts = self._edge_pts(rect, e.key)
            k = loads[e.key]
            if e.key in mover_edges and self.caption and self.t - self.caption[3] < 3.0:
                fx.ribbon(screen, mover_col, pts, 3)
            m = len(pts) // 2
            p, q = pts[m], pts[m + 1]
            d = (q - p) / (np.linalg.norm(q - p) + 1e-9); nn = np.array([-d[1], d[0]])
            tip = p + d * 6
            pygame.draw.lines(screen, (150, 190, 220), False, [tuple(p - d * 4 + nn * 5), tuple(tip), tuple(p - d * 4 - nn * 5)], 2)
            mid = pts[m]
            off = {("S", "A"): (-82, -34), ("A", "T"): (82, -34), ("S", "B"): (-82, 34),
                   ("B", "T"): (82, 34), ("A", "B"): (0, -150)}[e.key]
            formula = ({"2k": "4k − 2", "10": "10", "1": "1"}[e.label] if self.marginal else e.label)
            cost_now = ec(e.key, k) if k else ec(e.key, 1)
            c = (mid[0] + off[0], mid[1] + off[1])
            chip = pygame.Rect(0, 0, 128, 42); chip.center = c
            pygame.draw.rect(screen, T.PAPER, chip, border_radius=8)
            name = {("S", "A"): "narrow canal", ("B", "T"): "narrow canal", ("A", "B"): "shortcut",
                    ("A", "T"): "open lake", ("S", "B"): "open lake"}[e.key]
            text(screen, name + ("  (tolled)" if self.marginal else ""), (c[0], c[1] - 9), 12, T.INK, True, "c")
            heavy = e.label == "2k" and k >= 3
            text(screen, f"{k} boat{'s' if k != 1 else ''}: {fr(cost_now)} min", (c[0], c[1] + 9), 11,
                 T.WARN if heavy else T.INK2, False, "c")
        # nodes: dock, buoys, inspection ring
        S_, A_, B_, T_ = (self._pos(rect, n) for n in "SABT")
        pier = pygame.Rect(0, 0, 44, 12); pier.midright = (S_[0] - 6, S_[1])
        pygame.draw.rect(screen, T.PIER_DARK, pier.inflate(4, 4)); pygame.draw.rect(screen, T.PIER, pier)
        for p_, c_ in ((A_, T.RED_BUOY), (B_, T.GREEN_BUOY)):
            pygame.draw.circle(screen, T.shade(c_, 0.55), p_, 10); pygame.draw.circle(screen, c_, p_, 9)
            pygame.draw.circle(screen, (255, 255, 255), (p_[0] - 3, p_[1] - 3), 2)
        pygame.draw.circle(screen, T.CONTOUR, T_, 16, 3)
        pygame.draw.circle(screen, T.CONTOUR, T_, 3)
        for nname, pos in zip("SABT", (S_, A_, B_, T_)):
            lab = {"S": "Dock S", "A": "Buoy A", "B": "Buoy B", "T": "Inspection T"}[nname]
            img = T.font(12, True).render(lab, True, T.INK)
            r_ = img.get_rect().inflate(12, 6)
            r_.center = (pos[0] + {"S": -62, "T": 72, "A": 0, "B": 0}[nname], pos[1] + {"S": 26, "T": 26, "A": -52, "B": 52}[nname])
            pygame.draw.rect(screen, T.PAPER, r_, border_radius=6)
            screen.blit(img, (r_.x + 6, r_.y + 3))
        if not self.shortcut:
            text(screen, "Shortcut closed", (A_ + B_) / 2, 13, T.BAD, True, "c")
        for i in range(4):
            route = g.routes[self.profile[i]]
            costs = [float(g.c(e, loads[e])) for e in route]
            s = self.progress[i] * sum(costs)
            for e, ce in zip(route, costs):
                if s <= ce:
                    f = s / ce
                    break
                s -= ce
            pts = self._edge_pts(rect, e)
            idx = min(len(pts) - 2, int(f * (len(pts) - 1)))
            d = pts[idx + 1] - pts[idx]
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            p = pts[idx] + nrm * (i - 1.5) * 4
            fx.draw_boat(screen, p, math.atan2(-d[1], d[0]), T.BOAT_COLORS[i], 44)
            if self.last and self.last["i"] == i and self.caption and self.t - self.caption[3] < 4.0:
                pygame.draw.circle(screen, T.BOAT_COLORS[i], p, 30, 2)
            tag = T.font(12, True).render(str(i + 1), True, T.PAPER)
            tr = tag.get_rect().inflate(8, 2); tr.center = (p[0], p[1] - 26)
            pygame.draw.rect(screen, T.BOAT_COLORS[i], tr, border_radius=tr.h // 2)
            screen.blit(tag, (tr.x + 4, tr.y + 1))
        if self.caption:
            age = self.t - self.caption[3]
            a = max(0.0, min(1.0, age * 5) * min(1.0, (5.0 - age)))
            if a > 0:
                l1, l2, col, _ = self.caption
                i1 = T.font(24, True).render(l1, True, T.INK); i1.set_alpha(int(255 * a))
                i2 = T.font(24, True).render(l2, True, col); i2.set_alpha(int(255 * a))
                bg = pygame.Surface((max(i1.get_width(), i2.get_width()) + 56, 80), pygame.SRCALPHA)
                pygame.draw.rect(bg, (*T.PAPER, int(220 * a)), bg.get_rect(), border_radius=10)
                screen.blit(bg, (rect.x + 20, rect.y + 18))
                if self.last:
                    sw = pygame.Rect(rect.x + 36, rect.y + 33, 14, 14)
                    pygame.draw.rect(screen, T.BOAT_COLORS[self.last["i"]], sw, border_radius=3)
                screen.blit(i1, (rect.x + 60, rect.y + 26)); screen.blit(i2, (rect.x + 60, rect.y + 58))
        screen.set_clip(None)
        self.draw_panel(screen, panel)

    def draw_panel(self, screen, rect):
        g = self.game
        pygame.draw.rect(screen, T.PAPER, rect)
        pygame.draw.line(screen, T.RULE, rect.topleft, rect.bottomleft, 1)
        x0, x1 = rect.x + 24, rect.right - 24
        y = rect.y + 22
        text(screen, "Braess paradox at the harbour", (x0, y), 20, T.INK, True); y += 30
        text(screen, "Four boats, dock to inspection area. Exact game.", (x0, y), 12, T.INK2); y += 18
        mode = "marginal-cost pricing" if self.marginal else "selfish travel time"
        text(screen, f"Costs: {mode}.  Shortcut {'open' if self.shortcut else 'closed'}.", (x0, y), 12, T.INK2); y += 30
        for i in range(4):
            fx.draw_boat(screen, (x0 + 7, y + 8), 0.0, T.BOAT_COLORS[i], 14)
            text(screen, f"Boat {i + 1}", (x0 + 22, y), 13, T.INK, True)
            text(screen, PRETTY(g.route_name(self.profile[i])), (x0 + 84, y), 13, T.INK)
            s = f"{fr(g.private_cost(self.profile, i))} min"
            if self.marginal:
                s += f", pays {fr(g.private_cost(self.profile, i, g.marginal_cost))}"
            text(screen, s, (x1, y), 13, T.INK, False, "tr")
            y += 24
        y += 10
        pygame.draw.line(screen, T.RULE, (x0, y), (x1, y)); y += 14
        P, C = self.phi(), g.team_cost(self.profile)
        BLUE, ORANGE = T.ACCENT, (255, 140, 60)
        text(screen, "Potential Φ" if not self.marginal else "Potential Φm", (x0, y), 12, T.INK2)
        text(screen, "Team cost C", (x0 + 150, y), 12, T.INK2); y += 16
        text(screen, fr(P), (x0, y), 30, BLUE, True)
        text(screen, fr(C), (x0 + 150, y), 30, ORANGE, True)
        text(screen, f"optimum {fr(self.opt)}", (x1, y + 14), 12, T.INK2, False, "tr"); y += 48
        cr = pygame.Rect(x0, y, x1 - x0, 130)
        hist = self.history
        allv = [float(v) for h in hist for v in h] + [float(self.opt)]
        lo, hi = min(allv) - 3, max(allv) + 3
        n = max(2, len(hist))

        def Pt(k, v):
            return (cr.x + 8 + (cr.w - 16) * k / max(1, n - 1), cr.bottom - (cr.h - 10) * (float(v) - lo) / (hi - lo))
        yo = Pt(0, self.opt)[1]
        fx.dashed_line(screen, T.MUTED, (cr.x, yo), (cr.right, yo), 3, 4)
        for idx, colr in ((0, BLUE), (1, ORANGE)):
            pts = [Pt(k, h[idx]) for k, h in enumerate(hist)]
            if len(pts) >= 2:
                pygame.draw.lines(screen, colr, False, pts, 2)
            for p in pts:
                pygame.draw.circle(screen, colr, p, 3)
            if pts:
                text(screen, "Φ" if idx == 0 else "C", (pts[-1][0] - 4, pts[-1][1] + (10 if idx == 0 else -10)), 12, colr, True, "mr")
        pygame.draw.line(screen, T.RULE, (cr.x, cr.bottom + 2), (cr.right, cr.bottom + 2))
        text(screen, "Each move lowers Φ. C need not follow.", (x0, cr.bottom + 8), 13, T.INK)
        y = cr.bottom + 34
        if self.last:
            d = self.last
            dphi, dci = d["P1"] - d["P0"], d["c_new"] - d["c_old"]
            text(screen, f"Boat {d['i'] + 1}: {PRETTY(g.route_name(d['old']))}  to  {PRETTY(g.route_name(d['new']))}", (x0, y), 13, T.INK, True); y += 20
            text(screen, f"own cost {fr(d['c_old'])} → {fr(d['c_new'])}   (Δc = {fr(dci)})", (x0, y), 12, T.INK); y += 18
            text(screen, f"team cost {fr(d['C0'])} → {fr(d['C1'])}   (ΔC = {fr(d['C1'] - d['C0'])})", (x0, y), 12,
                 T.BAD if d["C1"] > d["C0"] else T.GOOD); y += 18
            text(screen, f"ΔΦ = {fr(dphi)} = Δc, checked exactly" if dphi == dci else "identity FAILED", (x0, y), 11, (120, 135, 160)); y += 26
        else:
            text(screen, "Press space: one boat switches to its best route.", (x0, y), 12, T.INK2); y += 26
        A = self.analysis
        text(screen, f"Worst equilibrium {fr(A.worst_eq_cost)}, optimum {fr(A.optimum_cost)}, "
                     f"price of anarchy {fr(A.price_of_anarchy)}.", (x0, y), 12, T.INK); y += 28
        if self.show_cycle:
            nA = [len(g.routes)] * 4
            Imax = max(abs(I) for _, I in four_cycle_sums(nA, lambda a, i: g.private_cost(a, i, self.ecost())))
            mp = max(abs(I) for _, I in four_cycle_sums([2, 2], matching_pennies_cost))
            pygame.draw.line(screen, T.RULE, (x0, y), (x1, y)); y += 12
            text(screen, "Four-cycle test (Monderer and Shapley)", (x0, y), 13, T.INK, True); y += 20
            text(screen, f"This game: largest circulation {fr(Imax)}, so a potential exists.", (x0, y), 12, T.INK); y += 18
            text(screen, f"Matching pennies: circulation ±{fr(mp)}, so none exists.", (x0, y), 12, T.INK); y += 18
            text(screen, "Compare: a closed line integral of a conservative field is zero.", (x0, y), 11, T.INK2)
        text(screen, "H shows the controls", (x0, rect.bottom - 26), 11, (90, 104, 130))
