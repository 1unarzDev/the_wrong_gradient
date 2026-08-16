"""Path planning, line-of-sight guidance, and the two travel-time (ETA) models.

* Planner: Dijkstra distance field on an inflated 1 m 8-grid -> pointer descent ->
  greedy line-of-sight shortcutting.  (Graph = grid cells, edges = 8-neighbours.)
* Guidance: LOS / pure-pursuit with lookahead, heading P + yaw-rate feedback,
  speed scheduling by heading error and by the sharpness of the upcoming corner.
* ETA models
    graph ETA   = shortest grid distance / v_graph            (+ service time)
    control ETA = sum_k L_k / (along-track speed incl. charted current)
                  + sum_corners  Pi_turn(theta_k)
                  + Pi_init(heading error, current speed) + Pi_stop  (+ service)
  where the Pi tables are measured ONCE by rolling out the real controller+physics on
  canonical manoeuvres (so they are "control-aware", derived, not hand-tuned).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import SimConfig
from .physics import allocate_thrust, rk4_step
from .world import NBRS, World


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------
def grid_path(world: World, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
    dist, nxt = world.field_to(*goal_xy)
    i, j = world.snap_free(*start_xy)
    if not np.isfinite(dist[i, j]):
        return None
    pts = [world.cell_center(i, j)]
    for _ in range(dist.size):
        k = nxt[i, j]
        if k < 0:
            break
        di, dj = NBRS[k]
        i, j = i + di, j + dj
        pts.append(world.cell_center(i, j))
    return pts


def smooth(world: World, pts: List[Tuple[float, float]], start_xy) -> List[Tuple[float, float]]:
    """Greedy LOS shortcut; anchor at the boat's true position."""
    clear = world.cfg.hull_radius_m + 0.75 * world.cfg.inflate_margin_m
    out = [tuple(start_xy)]
    anchor = tuple(start_xy)
    i, n = 0, len(pts)
    while i < n - 1:
        # largest jump index visible from anchor (coarse-to-fine)
        best = i + 1
        step = 2
        while best + step <= n - 1 and world.segment_free(anchor, pts[best + step], clear):
            best += step
            step *= 2
        lo, hi = best, min(best + step, n - 1)        # pts[lo] visible; search (lo, hi]
        while hi > lo:
            mid = (lo + hi + 1) // 2
            if world.segment_free(anchor, pts[mid], clear):
                lo = mid
            else:
                hi = mid - 1
        best = lo
        out.append(pts[best])
        anchor = pts[best]
        i = best
    return out


def plan_route(world: World, start_xy, task_points: Sequence[Tuple[float, float]]):
    """Waypoints from the boat to the task's first point, then through the remaining
    task points (e.g. gate entry -> centre -> exit). Returns (waypoints, grid_dist)."""
    raw = grid_path(world, start_xy, task_points[0])
    if raw is None:
        return None, math.inf
    dist, _ = world.field_to(*task_points[0])
    i, j = world.snap_free(*start_xy)
    cx, cy = world.cell_center(i, j)
    gdist = dist[i, j] + math.hypot(cx - start_xy[0], cy - start_xy[1])
    wps = smooth(world, raw, start_xy)
    wps[-1] = tuple(task_points[0])
    for p in task_points[1:]:
        gdist += math.hypot(p[0] - wps[-1][0], p[1] - wps[-1][1])
        wps.append(tuple(p))
    return wps, gdist


def path_length(wps) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:]))


def corner_angles(wps) -> List[float]:
    out = []
    for a, b, c in zip(wps, wps[1:], wps[2:]):
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        out.append(abs(wrap(h2 - h1)))
    return out


# ---------------------------------------------------------------------------------
# Guidance (LOS + speed scheduling)
# ---------------------------------------------------------------------------------
@dataclass
class GuidanceState:
    wps: List[Tuple[float, float]]
    seg: int = 0
    stop_at_end: bool = True
    tail_segments: int = 0          # final segments needing precise tracking (gate run-in)
    tail_lookahead: float = 2.5


def los_command(gs: GuidanceState, x, y, psi, cfg: SimConfig, v_cruise=None):
    """Returns (psi_des, v_des, dist_to_end). Advances gs.seg in place."""
    v_cruise = cfg.cruise_frac * cfg.v_max if v_cruise is None else v_cruise
    wps = gs.wps
    if len(wps) < 2:
        return psi, 0.0, 0.0
    # advance segment when past its end (projection) or close to the waypoint
    while gs.seg < len(wps) - 2:
        ax, ay = wps[gs.seg]; bx, by = wps[gs.seg + 1]
        sx, sy = bx - ax, by - ay
        L2 = sx * sx + sy * sy + 1e-9
        tproj = ((x - ax) * sx + (y - ay) * sy) / L2
        if tproj >= 1.0 or math.hypot(bx - x, by - y) < 0.6 * cfg.los_lookahead_m:
            gs.seg += 1
        else:
            break
    ax, ay = wps[gs.seg]; bx, by = wps[gs.seg + 1]
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy) + 1e-9
    tproj = max(0.0, min(1.0, ((x - ax) * sx + (y - ay) * sy) / (L * L)))
    # lookahead target: walk forward along the path
    look = cfg.los_lookahead_m
    if gs.tail_segments and gs.seg >= len(wps) - 1 - gs.tail_segments:
        look = gs.tail_lookahead
    px, py = ax + tproj * sx, ay + tproj * sy
    remain_seg = (1 - tproj) * L
    k = gs.seg
    tx, ty = px, py
    rem = look
    cur_rem = remain_seg
    while True:
        if rem <= cur_rem or k >= len(wps) - 2:
            ex, ey = wps[k + 1]
            sxx, syy = ex - wps[k][0], ey - wps[k][1]
            LL = math.hypot(sxx, syy) + 1e-9
            d = min(rem, cur_rem)
            tx = ex - (cur_rem - d) * sxx / LL
            ty = ey - (cur_rem - d) * syy / LL
            break
        rem -= cur_rem
        k += 1
        cur_rem = math.hypot(wps[k + 1][0] - wps[k][0], wps[k + 1][1] - wps[k][1])
    psi_des = math.atan2(ty - y, tx - x)
    err = abs(wrap(psi_des - psi))
    # distance to end of path
    d_end = remain_seg + sum(math.hypot(wps[m + 1][0] - wps[m][0], wps[m + 1][1] - wps[m][1])
                             for m in range(gs.seg + 1, len(wps) - 1))
    v = v_cruise * max(0.12, 1.0 - err / cfg.turn_slow_rad)
    # slow ahead of a sharp corner
    if gs.seg < len(wps) - 2:
        a, b, c = wps[gs.seg], wps[gs.seg + 1], wps[gs.seg + 2]
        th = abs(wrap(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0])))
        v_corner = v_cruise * max(0.3, 1.0 - th / math.pi * 1.1)
        v = min(v, v_corner + 0.45 * remain_seg)
    if gs.stop_at_end:
        v = min(v, max(0.0, 0.5 * (d_end - 0.4)))
    return psi_des, v, d_end


def heading_controller(psi_des, psi, cfg: SimConfig):
    return float(np.clip(cfg.k_heading * wrap(psi_des - psi), -cfg.r_max, cfg.r_max))


# ---------------------------------------------------------------------------------
# Control-aware penalty tables (measured from the real closed loop)
# ---------------------------------------------------------------------------------
def _zero_current(x, y, t):
    return 0.0 * x, 0.0 * y


def rollout(wps, x0, y0, psi0, v0, cfg: SimConfig, stop=False, tmax=200.0,
            current=_zero_current, dt=None):
    """Closed-loop rollout of guidance+controller+physics (no disturbances, no
    other boats). Returns time to reach the end of the path."""
    dt = dt or cfg.dt * cfg.control_every
    S = np.array([[x0], [y0], [psi0], [v0], [0.0]], float)
    gs = GuidanceState(list(wps), 0, stop)
    t = 0.0
    end = wps[-1]
    while t < tmax:
        x, y, psi, v, r = S[:, 0]
        psi_d, v_d, d_end = los_command(gs, x, y, psi, cfg)
        if gs.seg >= len(wps) - 2:
            ax, ay = wps[-2]
            sx, sy = end[0] - ax, end[1] - ay
            past = ((x - end[0]) * sx + (y - end[1]) * sy) >= 0
            if (stop and math.hypot(x - end[0], y - end[1]) < 1.0 and abs(v) < 0.35) or \
               (not stop and (past or math.hypot(x - end[0], y - end[1]) < 0.5)):
                return t
        r_d = heading_controller(psi_d, psi, cfg)
        TL, TR = allocate_thrust(np.array([v_d]), np.array([r_d]), np.array([v]), np.array([r]), cfg)
        S = rk4_step(S, TL, TR, current, t, dt, cfg)
        t += dt
    return tmax


def steady_speed(cfg: SimConfig) -> float:
    """Straight-line equilibrium of v' = (v_cmd - v)/tau_v - k v^2 with cruise command."""
    vc = cfg.cruise_frac * cfg.v_max
    # allocate_thrust adds 0.25*(v_des - v)/v_max to the command: solve numerically
    v = vc
    for _ in range(200):
        cmd = min(1.0, vc / cfg.v_max + 0.25 * (vc - v) / cfg.v_max) * cfg.v_max
        v = v + 0.05 * ((cmd - v) / cfg.tau_v - cfg.k_drag * v * v) * 10
    return v


@dataclass
class PenaltyTables:
    v_ss: float
    turn_th: np.ndarray       # corner angles
    turn_pen: np.ndarray      # seconds lost at a corner of that angle
    init_th: np.ndarray
    init_v: np.ndarray
    init_pen: np.ndarray      # (len(init_th), len(init_v)) seconds lost starting misaligned/slow
    stop_pen: float           # seconds lost decelerating to a stop at the end

    def turn(self, th):
        return float(np.interp(th, self.turn_th, self.turn_pen))

    def init(self, th, v):
        th = min(abs(th), math.pi); v = min(max(v, 0.0), self.init_v[-1])
        i = np.interp(th, self.init_th, np.arange(len(self.init_th)))
        j = np.interp(v, self.init_v, np.arange(len(self.init_v)))
        i0, j0 = int(min(i, len(self.init_th) - 2)), int(min(j, len(self.init_v) - 2))
        fi, fj = i - i0, j - j0
        P = self.init_pen
        return float((1 - fi) * (1 - fj) * P[i0, j0] + fi * (1 - fj) * P[i0 + 1, j0]
                     + (1 - fi) * fj * P[i0, j0 + 1] + fi * fj * P[i0 + 1, j0 + 1])


@lru_cache(maxsize=8)
def _tables_cached(key):
    cfg = SimConfig(**dict(key))
    v_ss = steady_speed(cfg)
    L = 40.0
    # straight reference (start at cruise speed, aligned, no stop)
    t_ref = rollout([(0, 0), (L, 0), (2 * L, 0)], 0, 0, 0, v_ss, cfg)
    base = t_ref - 2 * L / v_ss
    turn_th = np.linspace(0, math.pi * 0.95, 12)
    turn_pen = []
    for th in turn_th:
        wps = [(0, 0), (L, 0), (L + L * math.cos(th), L * math.sin(th))]
        turn_pen.append(rollout(wps, 0, 0, 0, v_ss, cfg) - 2 * L / v_ss - base)
    init_th = np.linspace(0, math.pi, 9)
    init_v = np.linspace(0, v_ss, 4)
    init_pen = np.zeros((len(init_th), len(init_v)))
    for a, th in enumerate(init_th):
        for b, v0 in enumerate(init_v):
            init_pen[a, b] = rollout([(0, 0), (L, 0), (2 * L, 0)], 0, 0, th, v0, cfg) \
                - 2 * L / v_ss - base
    stop_pen = rollout([(0, 0), (L, 0), (2 * L, 0)], 0, 0, 0, v_ss, cfg, stop=True) - t_ref
    return PenaltyTables(v_ss, turn_th, np.maximum(np.array(turn_pen), 0), init_th, init_v,
                         np.maximum(init_pen, 0), max(stop_pen, 0.0))


DYN_KEYS = ("v_max", "r_max", "tau_v", "tau_r", "k_drag", "cruise_frac", "k_heading",
            "k_rate", "turn_slow_rad", "los_lookahead_m", "dt", "control_every")


def penalty_tables(cfg: SimConfig) -> PenaltyTables:
    return _tables_cached(tuple((k, getattr(cfg, k)) for k in DYN_KEYS))


# ---------------------------------------------------------------------------------
# ETA models
# ---------------------------------------------------------------------------------
def graph_eta(world: World, xy, task, cfg: SimConfig) -> float:
    dist, _ = world.field_to(*task.points[0])
    i, j = world.snap_free(*xy)
    d = dist[i, j]
    if not np.isfinite(d):
        return math.inf
    cx, cy = world.cell_center(i, j)
    d += math.hypot(cx - xy[0], cy - xy[1]) + task.extra_len
    return d / cfg.graph_speed + task.service_s


def control_eta(world: World, state, task, cfg: SimConfig, tables: PenaltyTables,
                wps=None) -> Tuple[float, list]:
    """state = (x, y, psi, v). Uses only the CHARTED mean current."""
    x, y, psi, v = state
    if wps is None:
        wps, gd = plan_route(world, (x, y), task.points)
        if wps is None:
            return math.inf, None
    t = 0.0
    for a, b in zip(wps, wps[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6:
            continue
        tx, ty = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        cu, cv = world.current.mean(mx, my)
        cpar = float(cu * tx + cv * ty)
        cperp = float(-cu * ty + cv * tx)
        vs = tables.v_ss
        sp = math.sqrt(max(vs * vs - cperp * cperp, 0.04)) + cpar
        t += L / max(sp, 0.25)
    for th in corner_angles(wps):
        t += tables.turn(th)
    if len(wps) >= 2:
        h0 = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
        t += tables.init(abs(wrap(h0 - psi)), v)
    if task.stop:
        t += tables.stop_pen
    return t + task.service_s, wps
