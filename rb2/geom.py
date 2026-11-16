"""Ray-casting geometry for v2 sensing (no pygame dependency).

`obstacle_list` / `raycast` are the reference (pure-Python) implementations previously in rb2/viz/gallery.py
(gallery re-exports them).  `Obstacles` holds the same geometry as numpy arrays, precomputed once per course,
and `cast` intersects many rays with all obstacles at once (vectorised rays x obstacles).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

LIDAR_RANGE = 20.0
NONE = -2            # owner code: ray hit nothing
STATIC = -1          # owner code: static structure (dock cubes, piers, booms)
STATIC_CLASSES = ("moored", "boom_float")   # treated as static structure by the sensors (points only)


def hull_polygon(o):
    """Vessel hull outline (pointed bow) for vessel/moored objects."""
    L, B = o.length, o.length * 0.4
    co, sn = math.cos(o.heading), math.sin(o.heading)
    hull = [(-0.5 * L, -B / 2), (0.15 * L, -B / 2), (0.5 * L, 0), (0.15 * L, B / 2), (-0.5 * L, B / 2)]
    return [(o.x + a * co - b * sn, o.y + a * sn + b * co) for a, b in hull]


def obstacle_list(c):
    """Circles for buoys; true polygon outlines (as segments) for vessels, moored boats, dock cubes and booms.
    Returns (circ, segs): circ = [(x, y, r, obj)], segs = [((x0, y0), (x1, y1), obj_or_None)]."""
    circ, segs = [], []
    for o in c.objects:
        if o.cls in ("vessel_orange", "vessel_black", "moored"):
            P = hull_polygon(o)
            segs += [(P[i], P[(i + 1) % len(P)], o) for i in range(len(P))]
        else:
            circ.append((o.x, o.y, max(o.r, 0.12), o))
    for (cx, cy, cs, hd) in [q for m in c.modules if m.kind == "T3" for q in m.cubes] + list(c.piers):
        h = cs / 2; co, sn = math.cos(hd), math.sin(hd)
        P = [(cx + a * co - b * sn, cy + a * sn + b * co) for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))]
        segs += [(P[i], P[(i + 1) % 4], None) for i in range(4)]
    for line in c.booms + ([] if c.shortcut_open else c.boom_gates):
        segs += [(p, q, None) for p, q in zip(line, line[1:])]
    return circ, segs


def raycast(c, bx, by, n=180, rmax=LIDAR_RANGE):
    """Reference scalar ray caster: n rays over 360 deg from (bx, by). Returns (ranges, {obj id: obj hit})."""
    circ, segs = obstacle_list(c)
    hits, ranges = {}, []
    for k in range(n):
        a = 2 * math.pi * k / n
        dx, dy = math.cos(a), math.sin(a)
        best, who = rmax, None
        for (ox, oy, r, o) in circ:
            fx, fy = ox - bx, oy - by
            t = fx * dx + fy * dy
            if t <= 0 or t - r > best:
                continue
            d2 = fx * fx + fy * fy - t * t
            if d2 <= r * r:
                tt = t - math.sqrt(r * r - d2)
                if 0 < tt < best:
                    best, who = tt, o
        for (p, q, so) in segs:
            ex, ey = q[0] - p[0], q[1] - p[1]
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            wx, wy = p[0] - bx, p[1] - by
            t = (wx * ey - wy * ex) / den
            u = (wx * dy - wy * dx) / den
            if 0 < t < best and 0 <= u <= 1:
                best, who = t, so
        ranges.append(best)
        if who is not None:
            hits[who.id] = who
    return ranges, hits


class Obstacles:
    """Course obstacles as numpy arrays (built once per course).

    circ (N,3) x, y, r and circ_owner (N,) object id; seg (M,4) x0, y0, x1, y1 and seg_owner (M,) object id or
    STATIC.  Moored boats / boom floats get owner STATIC (static structure).  ext (n_obj,) is each object's
    bounding radius (hull half-length for vessels)."""

    def __init__(self, course):
        circ, segs = obstacle_list(course)
        self.n_obj = len(course.objects)
        self.circ = np.array([(x, y, r) for x, y, r, _ in circ], float).reshape(-1, 3)
        self.circ_owner = np.array([STATIC if o.cls in STATIC_CLASSES else o.id for *_, o in circ], int)
        self.seg = np.array([(p[0], p[1], q[0], q[1]) for p, q, _ in segs], float).reshape(-1, 4)
        self.seg_owner = np.array([STATIC if (o is None or o.cls in STATIC_CLASSES) else o.id for *_, o in segs], int)
        self.ext = np.array([max(o.r, 0.12) if not o.cls.startswith(("vessel", "moored")) else 0.5 * o.length
                             for o in course.objects], float)
        self.xy = np.array([(o.x, o.y) for o in course.objects], float).reshape(-1, 2)
        self._sd = self.seg[:, 2:] - self.seg[:, :2]

    def cull(self, ox, oy, rmax):
        """Indices of circles / segments that can be hit within rmax of (ox, oy)."""
        f = self.circ[:, :2] - (ox, oy)
        ci = np.nonzero(np.hypot(f[:, 0], f[:, 1]) - self.circ[:, 2] <= rmax)[0]
        w = (ox, oy) - self.seg[:, :2]
        L2 = np.einsum("ij,ij->i", self._sd, self._sd) + 1e-12
        u = np.clip(np.einsum("ij,ij->i", w, self._sd) / L2, 0, 1)
        q = self.seg[:, :2] + u[:, None] * self._sd - (ox, oy)
        si = np.nonzero(np.hypot(q[:, 0], q[:, 1]) <= rmax)[0]
        return ci, si


def cast(obs: Obstacles, ox: float, oy: float, ang: np.ndarray, rmax: float,
         exclude: Optional[np.ndarray] = None, culled=None) -> Tuple[np.ndarray, np.ndarray]:
    """Cast rays from (ox, oy) at absolute angles `ang` (K,). Returns (range (K,), owner (K,)); range = rmax and
    owner = NONE where nothing is hit.  `exclude` (K,) optionally gives one owner id per ray to ignore."""
    ang = np.asarray(ang, float)
    K = ang.shape[0]
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ci, si = culled if culled is not None else obs.cull(ox, oy, rmax)
    best = np.full(K, rmax)
    who = np.full(K, NONE)
    if K == 0:
        return best, who
    if len(ci):
        C = obs.circ[ci]
        fx, fy, r = C[:, 0] - ox, C[:, 1] - oy, C[:, 2]
        t = dx * fx + dy * fy                                   # (K,N)
        d2 = (fx * fx + fy * fy) - t * t
        disc = r * r - d2
        with np.errstate(invalid="ignore"):
            tt = np.where((disc >= 0) & (t > 0), t - np.sqrt(np.maximum(disc, 0)), np.inf)
        tt[tt <= 0] = np.inf
        own = obs.circ_owner[ci]
        if exclude is not None:
            tt[own[None, :] == exclude[:, None]] = np.inf
        j = np.argmin(tt, axis=1)
        tmin = tt[np.arange(K), j]
        m = tmin < best
        best[m] = tmin[m]; who[m] = own[j[m]]
    if len(si):
        S = obs.seg[si]
        ex, ey = S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]
        wx, wy = S[:, 0] - ox, S[:, 1] - oy
        den = dx * ey - dy * ex                                 # (K,M)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / den
            t = (wx * ey - wy * ex) * inv
            u = (wx * dy - wy * dx) * inv
        ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf)
        own = obs.seg_owner[si]
        if exclude is not None:
            t[own[None, :] == exclude[:, None]] = np.inf
        j = np.argmin(t, axis=1)
        tmin = t[np.arange(K), j]
        m = tmin < best
        best[m] = tmin[m]; who[m] = own[j[m]]
    return best, who
