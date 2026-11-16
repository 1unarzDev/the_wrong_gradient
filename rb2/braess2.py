"""Return rush v2: a physical scenario runner for testing Braess's paradox on the RoboBoat course.

What this module does
---------------------
* Builds a *copy* of a ``GenConfig(variant="return_rush")`` course and applies course-level knobs
  (``RushGeom``) to it: the single-file channels are rebuilt as a marina pier from the shore plus a pier
  wall running parallel to the spill boom (``lane_len`` = channel length, ``lane_width`` = clear width),
  and an optional no-wake speed limit ``v_open`` is imposed in the open-water corridor halves. Nothing in
  ``rb2/generate.py`` is changed; the chart comes from ``compile_chart``.
* Runs boats on fixed routes (west S-A-T, east S-B-T, shortcut S-A-B-T) with the v1 physics
  (``src.physics.rk4_step``) and guidance (``src.navigation.los_command`` / ``heading_controller`` /
  ``plan_route``), boat-boat separation, simple car following, and single-file channel discipline:
  a channel admits one boat at a time per direction, everybody else holds in a queue at its mouth (physical order along the approach).
* Measures each boat's S->T travel time and every node passage time (so edge times can be read off).
* Calibrates edge costs c_e(k) from isolated runs (k boats on one route, nobody else) and turns them into an
  exact ``src.game_exact.RouteGame`` with Fraction costs.

Why a speed knob is needed at all
---------------------------------
With a single boat speed everywhere, free-flow edge times obey the triangle inequality:
c_AT(1) <= c_AB(1) + c_BT(1). A Braess equilibrium in which everybody takes S-A-B-T needs
c_AT(n_AT) >= c_AB + c_BT(n) > c_AB + c_BT(1), which contradicts it for a fixed-cost A->T edge.
So the open-water halves must be *slower per metre* than the channels + shortcut (here: a marina
no-wake zone, a legitimate course rule), otherwise the paradox is geometrically impossible.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import SimConfig
from src.game_exact import Edge, RouteGame, analyze, best_response_dynamics
from src.navigation import GuidanceState, heading_controller, los_command, plan_route, wrap
from src.physics import allocate_thrust, disturbance_tape, rk4_step

from .generate import GenConfig, compile_chart, generate_course

CUBE = 0.508
ROUTES = ("west", "east", "shortcut")
ROUTE_NODES = {"west": ("S", "A", "T"), "east": ("S", "B", "T"), "shortcut": ("S", "A", "B", "T")}
EDGES = (("S", "A"), ("A", "T"), ("S", "B"), ("B", "T"), ("A", "B"))
CONGESTIBLE = {("S", "A"), ("B", "T")}
CHANNEL_START = {"west": "S", "east": "B"}   # start node of each channel edge
LANE_SPACING = 2.2          # lateral spacing of open-water lanes (m)


# ------------------------------------------------------------------------------------------------
# Geometry knobs and course construction
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RushGeom:
    course_seed: int = 7
    lane_len: float = 10.0            # length of the west single-file channel S->A (m)
    lane_len_e: Optional[float] = None  # length of the east channel B->T (m); None = same as lane_len
    lane_width: float = 3.1           # width between pier wall and boom centreline (m)
    east_top: float = 1.5             # east channel mouth height above the bottom of the south spill lobe (m)
    v_open: Optional[float] = 0.8     # no-wake speed limit in the open-water corridor halves (m/s); None = none
    n_boats_gen: int = 4              # GenConfig.n_boats used to generate the basin
    grid_res: float = 0.25            # chart resolution (finer than the 0.5 m default: tight lanes)
    plan_margin: float = 1.0          # planner clearance beyond the hull radius (keeps planned legs off boom corners)
    # congestion model of the two congestible legs (S->A west-upper, B->T east-lower):
    #   "single_file": pier-lined lanes, one boat at a time, queue at the mouth (experiment 1)
    #   "slow_zone":   open water, no walls, no single-file rule; inside the zone every boat's speed limit is
    #                  v0 / (1 + alpha (k - 1)) with k = boats currently in the zone (road-like congestion, experiment 2)
    congestion: str = "single_file"
    alpha: float = 0.5                # slow-zone congestion coefficient (west zone, S->A)
    alpha_e: Optional[float] = None   # east zone (B->T); None = same as alpha
    lane_speed_e: Optional[float] = None  # speed limit inside the east single-file channel (m/s); None = none

    def key(self):
        return tuple(sorted(asdict(self).items()))


@dataclass
class RushCourse:
    geom: RushGeom
    course: object
    gc: GenConfig
    x0: float
    x1: float
    y0: float
    y1: float
    ym: float
    half: float
    nodes: Dict[str, Tuple[float, float]]
    keys: Dict[str, Tuple[float, float]]         # route key points (channel mouths/exits etc.)
    channels: Dict[str, Tuple[str, str]]         # name -> (mouth key, exit key)
    zones: List[Tuple[float, float, float, float]]   # no-wake boxes (xmin, xmax, ymin, ymax)
    boxes: Dict[str, Tuple[float, float, float, float]]  # channel boxes (for the occupancy audit)
    czones: Dict[str, Tuple[List[Tuple[float, float, float, float]], float]] = field(default_factory=dict)
    # slow-zone congestion zones: name -> ([(xmin, xmax, ymin, ymax), ...], alpha)


def _cubes_row(xa, xb, y_top):
    """Row of 2-cube-deep pier cubes from xa to xb whose top edge is at y_top."""
    out = []
    n = max(1, int(round((xb - xa) / CUBE)))
    s = (xb - xa) / n
    for k in range(n):
        for dy in (0.5, 1.5):
            out.append((xa + (k + 0.5) * s, y_top - dy * CUBE, s, 0.0))
    return out


def _cubes_col(x_inner, side, ya, yb):
    """Column (1 cube wide) whose inner edge is at x_inner; side=-1 extends west, +1 east."""
    out = []
    n = max(1, int(round((yb - ya) / CUBE)))
    s = (yb - ya) / n
    for k in range(n):
        out.append((x_inner + side * CUBE / 2, ya + (k + 0.5) * s, s, 0.0))
    return out


def build_course(geom: RushGeom) -> RushCourse:
    base = generate_course(geom.course_seed, GenConfig(n_boats=geom.n_boats_gen, variant="return_rush"))
    gc = GenConfig(n_boats=geom.n_boats_gen, variant="return_rush", grid_res=geom.grid_res,
                   plan_margin=geom.plan_margin)
    c = copy.deepcopy(base)
    x0 = c.boom_gates[0][0][0]
    x1 = c.boom_gates[1][0][0]
    ylo, yhi = c.boom_gates[0][0][1], c.boom_gates[0][1][1]
    ym, half = (ylo + yhi) / 2, (yhi - ylo) / 2
    y1 = max(p[1] for p in c.booms[0])
    y0 = min(p[1] for p in c.booms[1])
    L, w = geom.lane_len, geom.lane_width
    Le = geom.lane_len if geom.lane_len_e is None else geom.lane_len_e
    if L > (y1 - (ym + half)) - 2.0 or Le > y0 + geom.east_top - 5.0 or min(L, Le) < 2.0:
        raise ValueError(f"lane lengths {L}/{Le} do not fit beside the spill lobes")
    xw, xe = x0 - w, x1 + w                      # pier-wall inner faces
    cw, ce = x0 - w / 2, x1 + w / 2              # channel centrelines
    # west channel: upper-west half, right at the S end (between the north spill lobe and a pier wall).
    # east channel: lower-east half at the T end; its mouth sits `east_top` m above the bottom of the south lobe so
    # that a queue of waiting boats fits between B and the mouth; below the lobe a second pier wall continues it.
    wy_top, wy_bot = y1, y1 - L
    ey_top, ey_bot = y0 + geom.east_top, y0 + geom.east_top - Le
    piers = []
    piers += _cubes_row(0.0, xw, wy_bot)                       # west marina pier (channel exit)
    piers += _cubes_col(xw, -1, wy_bot, wy_top)                # west pier wall along the boom
    piers += _cubes_row(xe, c.width, ey_top + 2 * CUBE)        # east marina pier (channel entry)
    piers += _cubes_col(xe, +1, ey_bot, ey_top)                # east pier wall along the boom
    if ey_bot < y0:
        piers += _cubes_col(x1 + 0.2, -1, ey_bot, y0 + 0.2)    # inner wall where the boom ends
    c.piers = piers
    # move the moored boats into the marina basins (off the channel) and the channel markers to the mouths
    rng = np.random.default_rng(geom.course_seed + 4242)
    moored = [o for o in c.objects if o.cls == "moored"]
    for k, o in enumerate(moored):
        if k % 2 == 0:
            o.x, o.y = float(rng.uniform(2.5, xw - 3.0)), wy_bot + 2.2
        else:
            o.x, o.y = float(rng.uniform(xe + 3.0, c.width - 2.5)), ey_top - 1.7
    marks = [o for o in c.objects if o.cls in ("red", "green") and o.module < 0]
    spots = [(xw - 0.3, wy_top + 0.3), (x0 + 0.3, wy_top + 0.3), (x1 + 0.3 if False else xe + 0.3, ey_top + 1.4),
             (x1 - 0.3, ey_top + 1.4)]
    for o, (px, py) in zip(marks, spots):
        o.x, o.y = px, py
    S = (c.width * 0.5, y1 + 3.0)
    T = (c.width * 0.5, y0 - 4.0)
    A = (cw, ym)
    B = (ce, ym)
    c.rush_nodes = dict(S=S, A=A, B=B, T=T)
    keys = dict(Pw=(cw, wy_top + 4.0), Mw=(cw, wy_top), Xw=(cw, wy_bot),
                Pe=(ce, min(ey_top + 2 * CUBE + 3.5, ym - half - 0.3)), Me=(ce, ey_top + 2 * CUBE), Xe=(ce, ey_bot), Qe=(ce, ey_bot - 2.5),
                Qw=(cw, y0 - 1.5))
    # slow (no-wake) zones on the STRAIGHT parts of the open-water halves only (west-lower: A->T, east-upper: S->B);
    # corners are excluded so that boats running abreast in parallel lanes do not pay for outer-lane arcs at low speed
    zones = [(0.0, x0, y0 + 1.0, ym - half - 0.5), (x0 + 3.0, T[0] - 2.0, -1.0, y0 - 0.5),
             (x1, c.width, ym + half + 0.5, y1 - 1.0), (S[0] + 2.0, x1 - 3.0, y1 + 0.5, c.height)]
    if geom.congestion == "slow_zone":
        return _slow_zone_variant(geom, c, gc, x0, x1, y0, y1, ym, half, S, T, zones)
    if geom.congestion != "single_file":
        raise ValueError(geom.congestion)
    boxes = {"west": (x0 - w, x0, wy_bot, wy_top), "east": (x1, x1 + w, ey_bot, ey_top)}
    return RushCourse(geom, c, gc, x0, x1, y0, y1, ym, half, dict(S=S, A=A, B=B, T=T), keys,
                      dict(west=("Mw", "Xw"), east=("Me", "Xe")), zones, boxes)


def _slow_zone_variant(geom, c, gc, x0, x1, y0, y1, ym, half, S, T, zones) -> RushCourse:
    """Experiment 2: no marina piers or lane walls; the congestible legs are open-water slow zones beside the spill.
    West zone = the whole corridor west of the boom above the shortcut (S->A); east zone = the corridor east of the
    boom below the shortcut (B->T). A and B sit 4.5 m out from the boom so boats can run abreast up to them."""
    c.piers = []
    for k, o in enumerate(o for o in c.objects if o.cls == "moored"):   # moor them along the shores, out of the way
        o.x, o.y = (1.6, 4.0 + 2.2 * k) if k % 2 == 0 else (c.width - 1.6, c.height - 30.0 - 2.2 * k)
    for k, o in enumerate(o for o in c.objects if o.cls in ("red", "green") and o.module < 0):
        o.x, o.y = c.width * 0.5 + (k - 1.5) * 1.0, 1.0
    A, B = (x0 - 4.5, ym), (x1 + 4.5, ym)
    c.rush_nodes = dict(S=S, A=A, B=B, T=T)
    ae = geom.alpha if geom.alpha_e is None else geom.alpha_e
    # each zone is a union of boxes: the corridor beside the spill plus the strip that leads into (west) or out of
    # (east) it, mirroring the open-water no-wake zones of the other two halves
    cz = {"west": ([(-1.0, x0, ym + half + 1.0, c.height + 1.0), (-1.0, S[0] - 2.0, y1 + 0.5, c.height + 1.0)],
                   geom.alpha),
          "east": ([(x1, c.width + 1.0, -1.0, ym - half - 1.0), (T[0] + 2.0, c.width + 1.0, -1.0, y0 - 0.5)], ae)}
    return RushCourse(geom, c, gc, x0, x1, y0, y1, ym, half, dict(S=S, A=A, B=B, T=T), {}, {}, zones, {}, cz)


def offset_path(chart, wps, idx, off, step=3.0, taper=8.0, min_clear=1.9):
    """Whole-route lateral offset (slow-zone experiment): resample the route every `step` m keeping the timing nodes as
    exact samples, shift every interior sample by `off` (left of travel), tapered only at the start and the finish,
    shrunk where the shifted point would come within `min_clear` of an obstacle (e.g. in the 4 m shortcut channel)."""
    if abs(off) < 1e-9:
        return wps, idx
    P = np.asarray(wps, float)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    L = d[-1]
    node_s = {n: d[i] for n, i in idx.items()}
    ss = np.unique(np.concatenate([np.linspace(0, L, max(3, int(L / step) + 1)), list(node_s.values())]))
    Q = np.column_stack([np.interp(ss, d, P[:, 0]), np.interp(ss, d, P[:, 1])])
    out = [tuple(Q[0])]
    for k in range(1, len(Q) - 1):
        tx, ty = Q[min(k + 1, len(Q) - 1)] - Q[k - 1]
        nrm = math.hypot(tx, ty) + 1e-9
        nx, ny = -ty / nrm, tx / nrm
        w = min(1.0, ss[k] / taper, (L - ss[k]) / taper)
        for f in (1.0, 0.66, 0.33, 0.0):
            qx, qy = Q[k][0] + f * w * off * nx, Q[k][1] + f * w * off * ny
            if f == 0.0 or (chart.clearance_at(qx, qy) >= min_clear
                            and chart.segment_free(out[-1], (qx, qy), 1.2)
                            and chart.segment_free(tuple(Q[k]), (qx, qy), 1.2)):
                break
        out.append((float(qx), float(qy)))
    out.append(tuple(Q[-1]))
    new_idx = {n: int(np.argmin(np.abs(ss - s_))) for n, s_ in node_s.items()}
    return out, new_idx


_RC_CACHE: Dict = {}
_CHART_CACHE: Dict = {}


def get_course(geom: RushGeom) -> RushCourse:
    k = geom.key()
    if k not in _RC_CACHE:
        _RC_CACHE[k] = build_course(geom)
    return _RC_CACHE[k]


def get_chart(geom: RushGeom, shortcut_open: bool):
    k = (geom.key(), bool(shortcut_open))
    if k not in _CHART_CACHE:
        rc = get_course(geom)
        c = copy.copy(rc.course)
        c.shortcut_open = bool(shortcut_open)
        _CHART_CACHE[k] = compile_chart(c, rc.gc)
    return _CHART_CACHE[k]


# ------------------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------------------
def route_keys(rc: RushCourse, route: str):
    """Ordered key points after the start: (name, xy, leg) with leg in {"plan", "line"} = how the leg that ENDS at
    this point is built. Node names S/A/B/T are timing nodes; channel mouths/exits are Mw/Xw/Me/Xe."""
    K, N = rc.keys, rc.nodes
    if rc.geom.congestion == "slow_zone":
        return {"west": [("A", N["A"], "plan"), ("T", N["T"], "plan")],
                "east": [("B", N["B"], "plan"), ("T", N["T"], "plan")],
                "shortcut": [("A", N["A"], "plan"), ("B", N["B"], "line"), ("T", N["T"], "plan")]}[route]
    west_in = [("Pw", K["Pw"], "line"), ("Mw", K["Mw"], "line"), ("Xw", K["Xw"], "line"), ("A", N["A"], "line")]
    east_in = [("Pe", K["Pe"], "plan"), ("Me", K["Me"], "line"), ("Xe", K["Xe"], "line"), ("Qe", K["Qe"], "line"),
               ("T", N["T"], "plan")]
    if route == "west":
        return west_in + [("T", N["T"], "plan")]
    if route == "east":
        return [("B", N["B"], "plan")] + east_in
    if route == "shortcut":
        return west_in + [("B", N["B"], "line")] + east_in
    raise KeyError(route)


def offset_leg(chart, pts, off, step=3.0, taper=10.0, min_clear=1.9):
    """Resample a planned open-water leg and shift its interior laterally by `off` metres (left of travel, tapered
    to zero at both ends, shrunk where the shifted point would be too close to an obstacle). Lets a bunch of boats
    run abreast in open water instead of collapsing onto one racing line."""
    if abs(off) < 1e-9 or len(pts) < 2:
        return list(pts)
    P = np.asarray(pts, float)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    L = d[-1]
    if L < 2 * step:
        return list(pts)
    ss = np.linspace(0.0, L, max(3, int(L / step) + 1))
    Q = np.column_stack([np.interp(ss, d, P[:, 0]), np.interp(ss, d, P[:, 1])])
    out = [tuple(Q[0])]
    for k in range(1, len(Q) - 1):
        tx, ty = Q[k + 1] - Q[k - 1]
        nrm = math.hypot(tx, ty) + 1e-9
        nx, ny = -ty / nrm, tx / nrm
        w = min(1.0, ss[k] / taper, (L - ss[k]) / taper)
        for f in (1.0, 0.66, 0.33, 0.0):
            qx, qy = Q[k][0] + f * w * off * nx, Q[k][1] + f * w * off * ny
            if f == 0.0 or chart.clearance_at(qx, qy) >= min_clear:
                break
        out.append((float(qx), float(qy)))
    out.append(tuple(Q[-1]))
    return out


def build_waypoints(rc: RushCourse, chart, start_xy, keys, lane_off: float = 0.0):
    """Concatenate legs into one waypoint list. Returns (wps, idx) with idx[name] = waypoint index."""
    wps = [tuple(map(float, start_xy))]
    idx = {"S": 0}
    for name, p, leg in keys:
        if leg == "plan":
            seg, _ = plan_route(chart, wps[-1], [p])
            if seg is None:
                raise RuntimeError(f"no path to {name}")
            seg = offset_leg(chart, seg, lane_off)
            wps.extend(tuple(map(float, q)) for q in seg[1:])
        else:
            wps.append(tuple(map(float, p)))
        wps[-1] = tuple(map(float, p))
        idx[name] = len(wps) - 1
    # drop zero-length segments (keep indices consistent)
    out, remap = [wps[0]], {0: 0}
    for k in range(1, len(wps)):
        if math.hypot(wps[k][0] - out[-1][0], wps[k][1] - out[-1][1]) > 0.05:
            out.append(wps[k])
        remap[k] = len(out) - 1
    return out, {n: remap[i] for n, i in idx.items()}


def start_slots(rc: RushCourse, n: int, seed: int, spacing: float = 3.4, jitter: float = 0.25):
    """Clustered start just north of S (task-area exit): 2 rows, ceil(n/2) columns, small seeded jitter.
    The jitter for slot k depends only on (seed, k), so it is common across conditions and route choices."""
    Sx, Sy = rc.nodes["S"]
    ncol = int(math.ceil(n / 2))
    out = []
    for k in range(n):
        col, row = k % ncol, k // ncol
        r = np.random.default_rng([int(seed), 7919, k])
        jx, jy = np.clip(r.normal(0, jitter, 2), -2 * jitter, 2 * jitter)
        jh = float(np.clip(r.normal(0, 0.1), -0.2, 0.2))
        out.append((Sx + (col - (ncol - 1) / 2) * spacing + jx, Sy + 1.6 + row * spacing + jy, -math.pi / 2 + jh))
    return out


def assign_routes(counts: Dict[str, int], slots) -> List[str]:
    """Deterministic boat->route assignment for a profile given as route counts: slots sorted west to east get
    west, then shortcut, then east (the shortcut starts through the west channel)."""
    order = sorted(range(len(slots)), key=lambda k: (slots[k][0], slots[k][1]))
    seq = ["west"] * counts.get("west", 0) + ["shortcut"] * counts.get("shortcut", 0) + ["east"] * counts.get("east", 0)
    if len(seq) != len(slots):
        raise ValueError("counts do not match number of boats")
    out = [None] * len(slots)
    for k, r in zip(order, seq):
        out[k] = r
    return out


# ------------------------------------------------------------------------------------------------
# Scenario runner
# ------------------------------------------------------------------------------------------------
@dataclass
class BoatPlan:
    wps: list
    idx: Dict[str, int]
    cum: np.ndarray
    channels: List[Tuple[str, float, float]]    # (channel name, s_mouth, s_exit)


def _plan_for(rc, chart, start_xy, route, lane_off=0.0):
    keys = route_keys(rc, route)
    if rc.geom.congestion == "slow_zone":
        wps, idx = build_waypoints(rc, chart, start_xy, keys, 0.0)
        wps, idx = offset_path(chart, wps, idx, lane_off)
    else:
        wps, idx = build_waypoints(rc, chart, start_xy, keys, lane_off)
    cum = np.concatenate([[0.0], np.cumsum([math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:])])])
    chans = []
    for name, (m, x) in rc.channels.items():
        if m in idx:
            chans.append((name, float(cum[idx[m]]), float(cum[idx[x]])))
    return BoatPlan(wps, idx, cum, chans)


def run_scenario(geom: RushGeom, routes: Sequence[str], seed: int, shortcut_open: bool, n_slots: Optional[int] = None,
                 t_max: float = 400.0, record: bool = False, cfg: Optional[SimConfig] = None,
                 discipline: bool = True, starts: Optional[Sequence[Tuple[float, float, float, float]]] = None,
                 lanes: bool = True, release: Optional[Sequence[float]] = None) -> dict:
    """Simulate len(routes) boats from the clustered start near S to T.

    `starts` (optional) overrides the clustered start with explicit (x, y, heading, speed) per boat (used to inject
    arrivals with controlled spacing); `lanes=False` disables the open-water lane offsets.

    Returns dict(times=[S->T per boat], node_times=[{node: t}], contacts=..., channel_log=[(t, ch, event, boat)],
    max_in_channel={ch: max simultaneous occupancy measured geometrically}, traj=(optional))."""
    rc = get_course(geom)
    chart = get_chart(geom, shortcut_open)
    cfg = cfg or SimConfig()
    cfg_tight = dataclasses.replace(cfg, los_lookahead_m=2.0)     # precise tracking into/through the lanes
    n = len(routes)
    if any(r == "shortcut" for r in routes) and not shortcut_open:
        raise ValueError("shortcut route with the shortcut closed")
    slots = start_slots(rc, n_slots or n, seed)[:n] if starts is None else [tuple(st[:3]) for st in starts]
    # open-water lanes: boats on the same route, ranked west to east by start slot, get offsets 0, +L, -L, +2L, ...
    lane_of = [0.0] * n
    for r in (set(routes) if lanes else ()):
        mem = sorted([i for i in range(n) if routes[i] == r], key=lambda i: slots[i][0])
        for rank, i in enumerate(mem):
            lane_of[i] = LANE_SPACING * ((rank + 1) // 2) * (1 if rank % 2 else -1) if rank else 0.0
    plans = [_plan_for(rc, chart, s[:2], r, lane_of[i]) for i, (s, r) in enumerate(zip(slots, routes))]
    # a boat joins a channel queue 22 m before the mouth, but not before the channel edge's start node (B for the
    # east channel): upstream of the node it is still in open water (S->B or the shortcut A->B)
    s_req_of = {}
    for i, p in enumerate(plans):
        for (ch, s_m, s_x) in p.channels:
            nd = CHANNEL_START[ch]
            s_req_of[(ch, i)] = max(s_m - 22.0, p.cum[p.idx[nd]] - 0.3 if nd in p.idx else -1e9)
    dt = cfg.dt
    n_steps = int(round(t_max / dt))
    tape = disturbance_tape(seed, 8, n_steps + 1, cfg)[:, :, :n] if n <= 8 else \
        disturbance_tape(seed, n, n_steps + 1, cfg)
    v_init = np.zeros(n) if starts is None else np.array([float(st[3]) for st in starts])
    S = np.array([[s[0] for s in slots], [s[1] for s in slots], [s[2] for s in slots], v_init, np.zeros(n)], float)
    TL, TR = np.zeros(n), np.zeros(n)
    gs = [GuidanceState(list(p.wps), 0, stop_at_end=False) for p in plans]
    seg = [0] * n                       # own monotone progress segment
    prog = np.zeros(n)
    active = np.ones(n, bool)
    pending = set()                     # boats not yet released (frozen and invisible until their release time)
    if release is not None:
        pending = {i for i in range(n) if release[i] > 0}
        for i in pending:
            active[i] = False
    finish = [math.nan] * n
    node_t = [{"S": 0.0} for _ in range(n)]
    queues: Dict[str, List[int]] = {ch: [] for ch in rc.channels}
    occupant: Dict[str, Optional[int]] = {ch: None for ch in rc.channels}
    req_t: Dict[Tuple[str, int], float] = {}
    hold_start: Dict[Tuple[str, int], float] = {}
    done_ch = set()
    chan_log = []
    contacts = dict(boat=0, structure=0, object=0)
    cset = set()
    max_in = {ch: 0 for ch in rc.channels}
    max_zone = {zn: 0 for zn in rc.czones}
    cur = rc.course.current
    zones = rc.zones if geom.v_open is not None else []
    v_cruise = cfg.cruise_frac * cfg.v_max
    traj = [] if record else None
    # geometric channel boxes for the occupancy audit
    boxes = rc.boxes
    t = 0.0
    for k in range(n_steps):
        t = k * dt
        if k % cfg.control_every == 0:
            for i in [i for i in pending if release[i] <= k * dt + 1e-9]:
                pending.discard(i); active[i] = True
                node_t[i]["S"] = k * dt
            X, Y, PSI, V = S[0], S[1], S[2], S[3]
            # --- progress along own route (monotone) --------------------------------------
            for i in range(n):
                if not active[i]:
                    continue
                p = plans[i]
                while seg[i] < len(p.wps) - 2:
                    ax, ay = p.wps[seg[i]]; bx, by = p.wps[seg[i] + 1]; cx, cy = p.wps[seg[i] + 2]
                    sx, sy = bx - ax, by - ay
                    tp = ((X[i] - ax) * sx + (Y[i] - ay) * sy) / (sx * sx + sy * sy + 1e-9)
                    ux, uy = cx - bx, cy - by
                    tq = ((X[i] - bx) * ux + (Y[i] - by) * uy) / (ux * ux + uy * uy + 1e-9)
                    d1 = _seg_dist(X[i], Y[i], ax, ay, bx, by)
                    d2 = _seg_dist(X[i], Y[i], bx, by, cx, cy)
                    if tp >= 1.0 or (tq > 0 and d2 < d1):
                        seg[i] += 1
                    else:
                        break
                ax, ay = p.wps[seg[i]]; bx, by = p.wps[seg[i] + 1]
                sx, sy = bx - ax, by - ay
                Ls = math.hypot(sx, sy) + 1e-9
                tp = ((X[i] - ax) * sx + (Y[i] - ay) * sy) / (Ls * Ls)
                prog[i] = max(prog[i], p.cum[seg[i]] + min(max(tp, 0.0), 1.0) * Ls)
                for name, wi in p.idx.items():
                    if name not in node_t[i] and prog[i] >= p.cum[wi] - 0.3:
                        node_t[i][name] = t
                if math.hypot(X[i] - p.wps[-1][0], Y[i] - p.wps[-1][1]) < 2.0 or prog[i] >= p.cum[-1] - 0.3:
                    finish[i] = t
                    node_t[i].setdefault("T", t)
                    active[i] = False
                    for ch in rc.channels:
                        if occupant[ch] == i:
                            occupant[ch] = None
                        if i in queues[ch]:
                            queues[ch].remove(i)
            if not active.any() and not pending:
                break
            # --- channel discipline: requests, releases, grants ---------------------------
            for i in range(n):
                if not active[i]:
                    continue
                for (ch, s_m, s_x) in plans[i].channels:
                    key = (ch, i)
                    if key in done_ch:
                        continue
                    if occupant[ch] == i and prog[i] >= s_x + 1.2:
                        occupant[ch] = None
                        done_ch.add(key)
                        chan_log.append((t, ch, "exit", i))
                    elif key not in req_t and prog[i] >= s_req_of[(ch, i)]:
                        req_t[key] = t
                        queues[ch].append(i)
            for ch in rc.channels:
                if not discipline:
                    for i in list(queues[ch]):
                        queues[ch].remove(i); chan_log.append((t, ch, "grant", i))
                        done_ch.add((ch, i))
                    continue
                # queue order = physical order along the approach (closest to the mouth first)
                queues[ch].sort(key=lambda j: ([c[1] for c in plans[j].channels if c[0] == ch][0] - prog[j], j))
                o = occupant[ch]
                if o is not None and queues[ch]:         # a grant is revocable until the boat reaches the mouth
                    so = [c[1] for c in plans[o].channels if c[0] == ch][0] - prog[o]
                    h = queues[ch][0]
                    sh = [c[1] for c in plans[h].channels if c[0] == ch][0] - prog[h]
                    if so > 0.5 and sh < so - 1.0:
                        queues[ch].append(o)
                        occupant[ch] = None
                        chan_log.append((t, ch, "revoke", o))
                        queues[ch].sort(key=lambda j: ([c[1] for c in plans[j].channels if c[0] == ch][0] - prog[j], j))
                if occupant[ch] is None and queues[ch]:
                    j = queues[ch].pop(0)
                    occupant[ch] = j
                    chan_log.append((t, ch, "grant", j))
            # --- control -------------------------------------------------------------------
            v_des, r_des = np.zeros(n), np.zeros(n)
            zk = {zn: sum(1 for j in range(n) if active[j] and _in_boxes(bxs, X[j], Y[j]))
                  for zn, (bxs, _) in rc.czones.items()}
            for zn, kk in zk.items():
                max_zone[zn] = max(max_zone[zn], kk)
            for i in range(n):
                if not active[i]:
                    continue
                x, y, psi, v = X[i], Y[i], PSI[i], V[i]
                near = any(s_m - 9.0 <= prog[i] <= s_x for (_, s_m, s_x) in plans[i].channels) \
                    or chart.clearance_at(x, y) < 3.0
                psi_d, vd, _ = los_command(gs[i], x, y, psi, cfg_tight if near else cfg, v_cruise)
                v_los = vd
                # no-wake zones
                for (xa, xb, ya, yb) in zones:
                    if xa <= x <= xb and ya <= y <= yb:
                        vd = min(vd, geom.v_open)
                # optional speed limit inside the east single-file channel (coupling study: sets its service interval)
                if geom.lane_speed_e is not None and "east" in rc.boxes:
                    xa, xb, ya, yb = rc.boxes["east"]
                    if xa <= x <= xb and ya - 1.0 <= y <= yb + 1.0:
                        vd = min(vd, geom.lane_speed_e)
                # slow-zone congestion (experiment 2): v_lim = v0 / (1 + alpha (k - 1)), k = boats in the zone
                for zn, (bxs, al) in rc.czones.items():
                    if _in_boxes(bxs, x, y):
                        vd = min(vd, v_cruise / (1.0 + al * (zk[zn] - 1)))
                v_ref = vd
                # hold in the queue at the channel mouth (single file along the approach)
                for (ch, s_m, s_x) in plans[i].channels:
                    if discipline and i in queues[ch]:
                        pos = queues[ch].index(i)
                        s_hold = s_m - 4.5 - 3.6 * pos
                        vd = min(vd, max(0.0, 0.6 * (s_hold - prog[i])))
                # car following: slow behind a boat that is ahead of us and has us behind it. Single-file spacing
                # applies in queues and channels; in open water only a short collision-avoidance gap (boats may
                # pass or run abreast there, so a bunch does not turn into a slow platoon)
                single = any(i in queues[ch] or occupant[ch] == i for ch in rc.channels)
                d_look, gap, cone = (7.0, 2.6, 0.6) if single else (3.4, 1.9, 0.35)
                for j in range(n):
                    if j == i or not active[j]:
                        continue
                    dx_, dy_ = X[j] - x, Y[j] - y
                    d = math.hypot(dx_, dy_)
                    if d > d_look:
                        continue
                    beta = abs(wrap(math.atan2(dy_, dx_) - psi))
                    back = abs(wrap(math.atan2(-dy_, -dx_) - PSI[j]))
                    if beta < cone and back > math.pi / 2:
                        vd = min(vd, max(0.0, 0.55 * (d - gap)))
                for ch in rc.channels:                   # first moment the queue for ch slows this boat down
                    if i in queues[ch] and (ch, i) not in hold_start and vd < 0.8 * v_ref:
                        hold_start[(ch, i)] = (t, float(prog[i]), float(v_ref))
                gx, gy = vd * math.cos(psi_d), vd * math.sin(psi_d)
                for j in range(n):                       # separation (as in rb2.sim2)
                    if j == i or not active[j]:
                        continue
                    dx_, dy_ = x - X[j], y - Y[j]
                    d_ = math.hypot(dx_, dy_)
                    if 1e-6 < d_ < 2.2:
                        push = 0.3 * (2.2 - d_)
                        gx += push * dx_ / d_; gy += push * dy_ / d_
                clr = chart.clearance_at(x, y)
                if clr < 1.1:                            # slide along structures instead of pushing into them
                    ex, ey = chart.clearance_grad(x, y)
                    into = gx * ex + gy * ey
                    if into < 0:
                        gx -= into * ex; gy -= into * ey
                    gx += 0.8 * (1.1 - clr) * ex; gy += 0.8 * (1.1 - clr) * ey
                cu, cv = cur.mean(x, y)
                wx_, wy_ = gx - float(cu), gy - float(cv)
                wn = math.hypot(wx_, wy_)
                if wn > 0.05:
                    psi_d = math.atan2(wy_, wx_)
                    vd = wn * max(0.0, math.cos(wrap(psi_d - psi)))
                else:
                    vd = 0.0
                r_des[i] = heading_controller(psi_d, psi, cfg)
                v_des[i] = min(vd, v_cruise)
            TL, TR = allocate_thrust(v_des, r_des, S[3], S[4], cfg)
            TL = np.where(active, TL, 0.0); TR = np.where(active, TR, 0.0)
        prev = S.copy()
        S = rk4_step(S, TL, TR, cur.total, t, dt, cfg, tape[k, 0] * active, tape[k, 1] * active)
        S[:, ~active] = prev[:, ~active]
        # contacts (structures via the chart, buoys/moored as points, boat-boat)
        X, Y = S[0], S[1]
        for i in range(n):
            if not active[i]:
                continue
            if chart.is_blocked(X[i], Y[i]) or chart.clearance_at(X[i], Y[i]) < 0.4:
                # contact: push the hull back out along the clearance gradient and bleed speed
                ex, ey = chart.clearance_grad(prev[0, i], prev[1, i])
                X[i], Y[i] = prev[0, i], prev[1, i]
                for _ in range(8):
                    if chart.clearance_at(X[i], Y[i]) >= 0.45:
                        break
                    X[i] += 0.05 * ex; Y[i] += 0.05 * ey
                S[3, i] *= 0.6
                if (i, "s") not in cset:
                    cset.add((i, "s")); contacts["structure"] += 1
            elif chart.clearance_at(X[i], Y[i]) > 1.0:
                cset.discard((i, "s"))
        for i in range(n):
            for j in range(i + 1, n):
                if not (active[i] and active[j]):
                    continue
                dx, dy = X[j] - X[i], Y[j] - Y[i]
                d = math.hypot(dx, dy)
                if d < 1.5:
                    if ("b", i, j) not in cset:
                        cset.add(("b", i, j)); contacts["boat"] += 1
                    ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
                    push = (1.5 - d) / 2 + 0.01
                    X[i] -= ux * push; Y[i] -= uy * push; X[j] += ux * push; Y[j] += uy * push
                elif d > 2.2:
                    cset.discard(("b", i, j))
        for ch, (xa, xb, ya, yb) in boxes.items():
            m = int(np.sum(active & (X > xa) & (X < xb) & (Y > ya + 0.9) & (Y < yb - 0.9)))
            max_in[ch] = max(max_in[ch], m)
        if record and k % 10 == 0:
            traj.append((t, S[0].copy(), S[1].copy(), active.copy()))
    hold = [{ch: hold_start[(ch, i)] for ch in rc.channels if (ch, i) in hold_start} for i in range(n)]
    node_arc = [{nm: float(p.cum[wi]) for nm, wi in p.idx.items()} for p in plans]
    return dict(times=finish, node_times=node_t, hold_start=hold, node_arc=node_arc, max_in_zone=max_zone, routes=list(routes), contacts=contacts, channel_log=chan_log,
                max_in_channel=max_in, traj=traj, seed=seed, shortcut_open=shortcut_open,
                all_finished=bool(np.all(np.isfinite(finish))))


def _in_boxes(bxs, x, y):
    return any(xa <= x <= xb and ya <= y <= yb for (xa, xb, ya, yb) in bxs)


def _seg_dist(px, py, ax, ay, bx, by):
    sx, sy = bx - ax, by - ay
    t = ((px - ax) * sx + (py - ay) * sy) / (sx * sx + sy * sy + 1e-9)
    t = min(max(t, 0.0), 1.0)
    return math.hypot(px - ax - t * sx, py - ay - t * sy)


# ------------------------------------------------------------------------------------------------
# Calibration and exact game
# ------------------------------------------------------------------------------------------------
def edge_times(res: dict) -> List[Dict[Tuple[str, str], float]]:
    """Per boat, time on each edge of its route (from node passage times)."""
    out = []
    for i, (r, nt) in enumerate(zip(res["routes"], res["node_times"])):
        nt = dict(nt)
        hs = res.get("hold_start", [{}] * len(res["routes"]))[i]
        cums = res.get("node_arc", [{}] * len(res["routes"]))[i]
        # Delay attribution: waiting in a channel queue is charged to that channel's edge. If a boat starts waiting
        # for the east channel before it reaches B (the queue spilled back over B), the upstream edge (S->B or A->B)
        # gets the boat's time to the point where waiting began plus the free-flow time from there to B.
        if "B" in nt and "east" in hs and "B" in cums:
            th, sh, vr = hs["east"]
            if th < nt["B"] and sh < cums["B"]:
                nt["B"] = min(nt["B"], th + (cums["B"] - sh) / max(vr, 0.3))
        nodes = ROUTE_NODES[r]
        out.append({(u, v): nt[v] - nt[u] for u, v in zip(nodes, nodes[1:]) if u in nt and v in nt})
    return out


def calib_job(args):
    """One isolated calibration run: k boats all on `route`, nobody else. Returns per-edge mean time."""
    geom, route, k, seed = args
    res = run_scenario(geom, [route] * k, seed, shortcut_open=(route == "shortcut"))
    et = edge_times(res)
    out = {}
    for e in {e for d in et for e in d}:
        out[e] = float(np.mean([d[e] for d in et if e in d]))
    return dict(route=route, k=k, seed=seed, edges=out, finished=res["all_finished"], contacts=res["contacts"])


# which isolated run defines which edge's cost c_e(k)
EDGE_SOURCE = {("S", "A"): "west", ("A", "T"): "west", ("S", "B"): "east", ("B", "T"): "east", ("A", "B"): "shortcut"}


def aggregate_calibration(rows, kmax):
    """rows: outputs of calib_job. Returns costs[edge] = [c(1)..c(kmax)] (mean over seeds) and the shortcut-run
    B->T times (the same edge reached through the metering first channel) for the separability diagnostic."""
    costs, sds, metered_bt = {}, {}, []
    for e, src in EDGE_SOURCE.items():
        cs, ss = [], []
        for k in range(1, kmax + 1):
            v = [r["edges"][e] for r in rows if r["route"] == src and r["k"] == k and e in r["edges"]]
            cs.append(float(np.mean(v))); ss.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        costs["".join(e)] = cs; sds["".join(e)] = ss
    for k in range(1, kmax + 1):
        v = [r["edges"][("B", "T")] for r in rows if r["route"] == "shortcut" and r["k"] == k]
        metered_bt.append(float(np.mean(v)))
    return costs, sds, metered_bt


def to_fraction(x: float, q: int = 10) -> Fraction:
    return Fraction(int(round(x * q)), q)


def make_game(costs: Dict[str, List[float]], with_shortcut: bool, n_agents: int, q: int = 10) -> RouteGame:
    """Exact RouteGame from measured costs (rounded to 1/q s). Costs beyond the measured k are linearly
    extrapolated from the last two points (only used if n_agents exceeds the calibrated range)."""
    def fn(vals):
        fr = [to_fraction(v, q) for v in vals]

        def c(k):
            if k <= len(fr):
                return fr[k - 1]
            return fr[-1] + (k - len(fr)) * (fr[-1] - fr[-2])
        return c
    edges = [Edge(e[0], e[1], fn(costs[e]), e) for e in ("SA", "AT", "SB", "BT")]
    if with_shortcut:
        edges.append(Edge("A", "B", fn(costs["AB"]), "AB"))
    return RouteGame(edges, "S", "T", n_agents)


def route_label(game: RouteGame, r: int) -> str:
    return {"SAT": "west", "SBT": "east", "SABT": "shortcut"}[game.route_name(r)]


def counts_of(game: RouteGame, profile) -> Dict[str, int]:
    out = {"west": 0, "east": 0, "shortcut": 0}
    for r in profile:
        out[route_label(game, r)] += 1
    return out


def predict(costs: Dict[str, List[float]], n: int = 4, q: int = 10) -> dict:
    """Exact prediction: NE/OPT/PoA without and with the shortcut, the tolled (marginal-cost) NE, and the
    sequential best-response profiles used for the measured conditions."""
    g0, g1 = make_game(costs, False, n, q), make_game(costs, True, n, q)
    A0, A1 = analyze(g0), analyze(g1)
    A1m = analyze(g1, g1.marginal_cost)
    start0 = [0] * n
    p_closed = best_response_dynamics(g0, start0)[-1].profile
    lab = {route_label(g0, r): r for r in range(len(g0.routes))}
    lab1 = {route_label(g1, r): r for r in range(len(g1.routes))}
    start1 = [lab1[route_label(g0, r)] for r in p_closed]
    p_open = best_response_dynamics(g1, start1)[-1].profile
    p_toll = best_response_dynamics(g1, p_open, g1.marginal_cost)[-1].profile

    def cset(g, A):
        return sorted({tuple(sorted(counts_of(g, a).items())) for a in A})
    eq_counts0 = sorted({tuple(counts_of(g0, a)[r] for r in ROUTES) for a in A0.equilibria})
    eq_counts1 = sorted({tuple(counts_of(g1, a)[r] for r in ROUTES) for a in A1.equilibria})
    eq_counts1m = sorted({tuple(counts_of(g1, a)[r] for r in ROUTES) for a in A1m.equilibria})
    opt_counts1 = sorted({tuple(counts_of(g1, a)[r] for r in ROUTES) for a in A1.optima})
    paradox = (A1.worst_eq_cost > A1.optimum_cost) and (A1.worst_eq_cost > A0.worst_eq_cost)
    strong = A1.best_eq_cost > A0.worst_eq_cost
    return dict(
        n=n, costs_fraction={e: [str(to_fraction(v, q)) for v in vals] for e, vals in costs.items()},
        closed=dict(ne_counts=eq_counts0, worst_ne=A0.worst_eq_cost, best_ne=A0.best_eq_cost,
                    opt=A0.optimum_cost, poa=A0.price_of_anarchy),
        open=dict(ne_counts=eq_counts1, worst_ne=A1.worst_eq_cost, best_ne=A1.best_eq_cost, opt=A1.optimum_cost,
                  opt_counts=opt_counts1, poa=A1.price_of_anarchy),
        tolled=dict(ne_counts=eq_counts1m, worst_ne_true_cost=A1m.worst_eq_cost, best_ne_true_cost=A1m.best_eq_cost),
        profiles=dict(closed=counts_of(g0, p_closed), open=counts_of(g1, p_open), tolled=counts_of(g1, p_toll)),
        team_cost=dict(closed=g0.team_cost(p_closed), open=g1.team_cost(p_open), tolled=g1.team_cost(p_toll)),
        paradox=bool(paradox), strong_paradox=bool(strong), order=("west", "east", "shortcut"))


def counts_team_cost(costs, counts: Dict[str, int], q: int = 10) -> Fraction:
    n = sum(counts.values())
    g = make_game(costs, True, n, q)
    lab = {route_label(g, r): r for r in range(len(g.routes))}
    prof = [lab[r] for r in ROUTES for _ in range(counts.get(r, 0))]
    return g.team_cost(prof)


# ------------------------------------------------------------------------------------------------
# Measurement helpers (profiles given as route counts) and the profile-level ("true") game
# ------------------------------------------------------------------------------------------------
def all_count_profiles(n: int, with_shortcut: bool = True):
    out = []
    for w in range(n + 1):
        for s in range(n + 1 - w if with_shortcut else 1):
            out.append({"west": w, "east": n - w - s, "shortcut": s})
    return out


def measure_job(args):
    """(geom, counts, seed, shortcut_open) -> summary of one full scenario with that route profile."""
    geom, counts, seed, shortcut_open = args
    n = sum(counts.values())
    rc = get_course(geom)
    routes = assign_routes(counts, start_slots(rc, n, seed))
    res = run_scenario(geom, routes, seed, shortcut_open)
    t = np.array(res["times"], float)
    per_route = {r: float(np.mean(t[[i for i in range(n) if routes[i] == r]])) for r in ROUTES if r in routes}
    return dict(seed=seed, counts=dict(counts), shortcut_open=shortcut_open, routes=routes,
                times=[float(x) for x in t], total=float(np.sum(t)), makespan=float(np.max(t)),
                per_route=per_route, finished=res["all_finished"], contacts=res["contacts"],
                max_in_channel=res["max_in_channel"], max_in_zone=res.get("max_in_zone", {}),
                edges=[{"".join(e): v for e, v in d.items()} for d in edge_times(res)])


def ckey(c) -> Tuple[int, int, int]:
    return (c.get("west", 0), c.get("east", 0), c.get("shortcut", 0))


def profile_game(table: Dict[Tuple[int, int, int], Dict[str, float]], n: int, with_shortcut: bool) -> dict:
    """Pure NE / optimum of the measured profile-level game. table[(w, e, s)][route] = mean measured S->T time of
    the boats on `route` in that profile (route-symmetric costs, averaged over seeds). A profile is an NE if no boat
    can lower its own measured time by switching route (the deviation lands in another measured profile)."""
    profs = [ckey(c) for c in all_count_profiles(n, with_shortcut)]
    idx = {"west": 0, "east": 1, "shortcut": 2}
    allowed = ROUTES if with_shortcut else ("west", "east")

    def team(p):
        return sum(p[idx[r]] * table[p][r] for r in ROUTES if p[idx[r]] > 0)
    eqs = []
    for p in profs:
        ok = True
        for r in allowed:
            if p[idx[r]] == 0:
                continue
            for r2 in allowed:
                if r2 == r:
                    continue
                q = list(p); q[idx[r]] -= 1; q[idx[r2]] += 1; q = tuple(q)
                if table[q][r2] < table[p][r] - 1e-9:
                    ok = False
        if ok:
            eqs.append(p)
    costs = {p: team(p) for p in profs}
    opt = min(costs.values())
    return dict(equilibria=eqs, eq_costs=[costs[p] for p in eqs], optimum=opt,
                optima=[p for p in profs if abs(costs[p] - opt) < 1e-9], costs=costs,
                worst_eq=max(costs[p] for p in eqs) if eqs else math.nan,
                best_eq=min(costs[p] for p in eqs) if eqs else math.nan)


# ------------------------------------------------------------------------------------------------
# Recorded calibration of the primary geometry (written by analysis/return_rush.py, data/return_rush/results.json;
# 16 seeds x k = 1..4, mean per-boat edge time in seconds). The exact game built from these predicts the paradox:
# closed NE 2-2 split C = 349.4 s; open NE all-shortcut C = 363.6 s; OPT (1 west, 1 east, 2 shortcut) 338.5 s.
CALIBRATED_GEOM = dict(lane_len=6.0, lane_len_e=10.0, v_open=0.4)
CALIBRATED = {
    "SA": [24.112500000000004, 28.890625, 32.10208333333333, 36.50625000000001],
    "AT": [56.75, 58.34375, 58.85208333333334, 58.684375],
    "SB": [56.625, 59.484375, 62.150000000000006, 64.1921875],
    "BT": [25.4, 28.000000000000004, 30.42916666666667, 35.6140625],
    "AB": [18.843750000000004, 18.803125, 18.843750000000004, 18.840625000000003],
}
