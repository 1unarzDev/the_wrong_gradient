"""Procedurally generated RoboBoat-style course (fictional; not an official layout).

World frame: x right, y up (rendering flips y), metres, heading psi measured from +x.
Contents: shoreline boundary, launch pier, a rocky barrier with narrow channels
(capacity-limited bottlenecks), islands, a smooth current field, a planning grid with
obstacle inflation, and per-goal shortest-path distance fields (Dijkstra on 8-grid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .config import SimConfig
from .rng import stream

SQ2 = math.sqrt(2.0)
# 8-neighbour offsets (dy, dx) and step lengths
NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NBR_LEN = [SQ2, 1, SQ2, 1, 1, SQ2, 1, SQ2]


@dataclass
class Channel:
    """Narrow passage through the barrier; capacity 1 boat (reservation rule)."""
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def center(self):
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, x, y, pad=0.0):
        return (self.x0 - pad <= x <= self.x1 + pad) and (self.y0 - pad <= y <= self.y1 + pad)


@dataclass
class CurrentField:
    """Charted mean current (known to all policies) + unknown time-varying part.
    All parameters are drawn once from the 'currents' stream: identical across policies."""
    drift: np.ndarray                    # (2,)
    vortices: np.ndarray                 # (k, 4): x, y, radius, circulation scale
    jets: np.ndarray                     # (m, 5): x, y, half_len, half_width, signed speed
    unk_amp: float
    unk_period: float
    unk_phase: float

    def mean(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        u = np.full_like(x, self.drift[0]); v = np.full_like(y, self.drift[1])
        for cx, cy, R, s in self.vortices:
            dx, dy = x - cx, y - cy
            g = s * np.exp(-(dx * dx + dy * dy) / (R * R))
            u += -dy / R * g
            v += dx / R * g
        for cx, cy, hl, hw, s in self.jets:
            g = s * np.exp(-((x - cx) / hl) ** 2 - ((y - cy) / hw) ** 2)
            u += g
        return u, v

    def unknown(self, x, y, t):
        """Unmodelled, time-varying component (never available to any policy)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        ang = 2 * math.pi * t / self.unk_period + self.unk_phase
        a = self.unk_amp
        u = a * math.cos(ang) * (0.6 + 0.4 * np.sin(y / 23.0 + 0.5 * ang))
        v = a * math.sin(ang) * (0.6 + 0.4 * np.cos(x / 31.0 - 0.3 * ang))
        return u, v

    def total(self, x, y, t):
        mu, mv = self.mean(x, y)
        uu, uv = self.unknown(x, y, t)
        return mu + uu, mv + uv


@dataclass
class World:
    cfg: SimConfig
    seed: int
    width: float
    height: float
    islands: List[np.ndarray]            # polygons (k,2)
    rocks: List[np.ndarray]              # barrier rock polygons
    piers: List[np.ndarray]              # dock pier polygons (obstacles)
    channels: List[Channel]
    barrier_x: float
    launch: Tuple[float, float]
    current: CurrentField
    blocked: np.ndarray = None           # (ny, nx) True = land/rock
    clearance: np.ndarray = None         # metres to nearest blocked cell
    free_plan: np.ndarray = None         # plannable cells (inflated)
    dock_sites: List[Tuple[float, float, float]] = field(default_factory=list)  # x,y,heading
    _fields: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # --- grid helpers -------------------------------------------------------------
    @property
    def res(self):
        return self.cfg.grid_res_m

    @property
    def shape(self):
        return self.blocked.shape

    def to_cell(self, x, y) -> Tuple[int, int]:
        ny, nx = self.shape
        j = int(min(max(x / self.res, 0), nx - 1))
        i = int(min(max(y / self.res, 0), ny - 1))
        return i, j

    def cell_center(self, i, j):
        return ((j + 0.5) * self.res, (i + 0.5) * self.res)

    def clearance_at(self, x, y) -> float:
        i, j = self.to_cell(x, y)
        return float(self.clearance[i, j])

    def clearance_grad(self, x, y):
        """Unit vector pointing away from the nearest obstacle (central differences)."""
        h = self.res
        gx = self.clearance_at(x + h, y) - self.clearance_at(x - h, y)
        gy = self.clearance_at(x, y + h) - self.clearance_at(x, y - h)
        n = math.hypot(gx, gy)
        return (gx / n, gy / n) if n > 1e-9 else (0.0, 0.0)

    def is_blocked(self, x, y) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        i, j = self.to_cell(x, y)
        return bool(self.blocked[i, j])

    def snap_free(self, x, y) -> Tuple[int, int]:
        """Nearest plannable cell to (x, y)."""
        i, j = self.to_cell(x, y)
        if self.free_plan[i, j]:
            return i, j
        return int(self._near_i[i, j]), int(self._near_j[i, j])

    def segment_free(self, p, q, min_clear=None) -> bool:
        """Is the straight segment p->q at least `min_clear` from obstacles?"""
        min_clear = self.cfg.hull_radius_m + self.cfg.inflate_margin_m if min_clear is None else min_clear
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(L / (0.5 * self.res)) + 1)
        xs = np.linspace(p[0], q[0], n); ys = np.linspace(p[1], q[1], n)
        ny, nx = self.shape
        jj = np.clip((xs / self.res).astype(int), 0, nx - 1)
        ii = np.clip((ys / self.res).astype(int), 0, ny - 1)
        return bool(np.all(self.clearance[ii, jj] >= min_clear))

    # --- distance fields ------------------------------------------------------------
    def field_to(self, x, y):
        """(dist, nextdir) arrays for shortest paths TO the plannable cell nearest (x,y).
        dist in metres (inf where unreachable); nextdir in 0..7 (index into NBRS), -1 at goal."""
        key = self.snap_free(x, y)
        if key not in self._fields:
            self._fields[key] = self._compute_field(key)
        return self._fields[key]

    def _compute_field(self, goal):
        gi, gj = goal
        node = self._node_index[gi, gj]
        d = dijkstra(self._graph, directed=False, indices=int(node))
        ny, nx = self.shape
        dist = np.full((ny, nx), np.inf)
        dist[self._free_ii, self._free_jj] = d * self.res
        # next-step pointer: neighbour minimising (step + dist[nbr])
        pad = np.pad(dist, 1, constant_values=np.inf)
        cand = np.empty((8, ny, nx))
        for k, ((di, dj), L) in enumerate(zip(NBRS, NBR_LEN)):
            cand[k] = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx] + L * self.res
        nextdir = np.argmin(cand, axis=0).astype(np.int8)
        nextdir[gi, gj] = -1
        nextdir[~np.isfinite(dist)] = -1
        return dist, nextdir

    def set_point_obstacles(self, pts, radius=0.45):
        """Small circular obstacles (gate buoys). Rebuilds grid and clears fields."""
        self.point_obstacles = [(float(x), float(y), radius) for x, y in pts]
        reach = getattr(self, "_launch_reach", None)
        self.build_grid()
        self._fields.clear()
        if reach is not None:
            dist, _ = self.field_to(*self.launch)
            self._launch_reach = np.isfinite(dist)

    def build_grid(self):
        cfg = self.cfg
        nx, ny = int(math.ceil(self.width / self.res)), int(math.ceil(self.height / self.res))
        xs = (np.arange(nx) + 0.5) * self.res
        ys = (np.arange(ny) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        blocked = np.zeros(ny * nx, bool)
        for poly in self.islands + self.rocks + self.piers:
            lo, hi = poly.min(0) - 1, poly.max(0) + 1
            m = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
            if m.any():
                blocked[np.where(m)[0][Path(poly).contains_points(pts[m])]] = True
        blocked = blocked.reshape(ny, nx)
        for (bx, by, br) in getattr(self, "point_obstacles", []):
            m = (X - bx) ** 2 + (Y - by) ** 2 <= (br + 0.5 * self.res) ** 2
            blocked |= m
        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True
        self.blocked = blocked
        self.clearance = ndimage.distance_transform_edt(~blocked) * self.res
        self.free_plan = self.clearance >= cfg.hull_radius_m + cfg.inflate_margin_m
        _, (ni, nj) = ndimage.distance_transform_edt(~self.free_plan, return_indices=True)
        self._near_i, self._near_j = ni, nj
        # 8-connected graph on plannable cells
        ii, jj = np.nonzero(self.free_plan)
        self._free_ii, self._free_jj = ii, jj
        idx = -np.ones((ny, nx), np.int64)
        idx[ii, jj] = np.arange(len(ii))
        self._node_index = idx
        rows, cols, w = [], [], []
        for (di, dj), L in zip(NBRS[4:], NBR_LEN[4:]):     # half the neighbours (undirected)
            i2, j2 = ii + di, jj + dj
            ok = (i2 >= 0) & (i2 < ny) & (j2 >= 0) & (j2 < nx)
            ok[ok] &= self.free_plan[i2[ok], j2[ok]]
            rows.append(idx[ii[ok], jj[ok]]); cols.append(idx[i2[ok], j2[ok]])
            w.append(np.full(ok.sum(), L))
        # (1,-1) neighbour is in NBRS[5]; NBRS[4:] = (0,1),(1,-1),(1,0),(1,1) -> all 4 forward dirs
        n = len(ii)
        self._graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n, n)).tocsr()
        # launch region must be connected to every channel (sanity)
        return self

    def random_water_point(self, rng, min_clear=4.0, x_range=None, tries=500):
        xr = x_range or (8.0, self.width - 8.0)
        for _ in range(tries):
            x = rng.uniform(*xr); y = rng.uniform(8.0, self.height - 8.0)
            if self.clearance_at(x, y) >= min_clear and self.reachable(x, y):
                return x, y
        raise RuntimeError("could not place point on water")

    def reachable(self, x, y) -> bool:
        i, j = self.snap_free(x, y)
        return bool(self._launch_reach[i, j])


# --------------------------------------------------------------------------------------
def _blob(rng, cx, cy, r, k=14, rough=0.28):
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    # smooth radial noise using a few harmonics
    a = rng.normal(0, rough, 3); ph = rng.uniform(0, 2 * np.pi, 3)
    rad = r * (1 + sum(a[m] * np.cos((m + 2) * ang + ph[m]) for m in range(3)) / 2)
    rad = np.clip(rad, 0.55 * r, 1.45 * r)
    return np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])


def _rect(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([cx, cy])


def generate_world(cfg: SimConfig, seed: int) -> World:
    rng = stream(seed, "map")
    scale = math.sqrt(cfg.n_boats / 4.0) if cfg.scale_arena_with_fleet else 1.0
    W, H = cfg.base_width_m * scale, cfg.base_height_m * scale
    launch = (9.0, H / 2)

    # barrier spine with channels (bottlenecks)
    rocks, channels = [], []
    bx = W * rng.uniform(0.42, 0.52)
    if cfg.barrier:
        n_ch = max(1, int(round(cfg.n_channels * scale)))
        cw = cfg.channel_width_m
        for _ in range(200):
            cy = np.sort(rng.uniform(14, H - 14, n_ch))
            if n_ch == 1 or np.min(np.diff(cy)) > 22:
                break
        bw = 7.0                                  # barrier thickness
        edges = [0.0]
        for c in cy:
            edges += [c - cw / 2, c + cw / 2]
            channels.append(Channel(bx - bw / 2, bx + bw / 2, c - cw / 2, c + cw / 2))
        edges.append(H)
        wob = rng.uniform(0, 2 * np.pi)
        for a, b in zip(edges[0::2], edges[1::2]):
            y = a
            while y < b - 0.5:
                seg = min(rng.uniform(5, 9), b - y)
                yc = y + seg / 2
                xoff = 1.2 * math.sin(yc / 9 + wob)
                # rectangular core keeps channel walls straight, noisy cap adds texture
                core = np.array([[bx - bw / 2 + xoff, y], [bx + bw / 2 + xoff, y],
                                 [bx + bw / 2 + xoff, y + seg], [bx - bw / 2 + xoff, y + seg]])
                rocks.append(core)
                if seg > 3:
                    rocks.append(_blob(rng, bx + xoff + rng.uniform(-1, 1), yc,
                                       min(bw * 0.62, seg * 0.6), k=10, rough=0.2))
                y += seg
        # clip blobs that intrude into channels: drop any rock vertex inside a channel
        rocks = [r for r in rocks if not any(
            Path(r).contains_point(ch.center) for ch in channels)]

    # islands
    islands = []
    n_isl = int(round(cfg.n_islands * scale ** 2))
    for _ in range(n_isl * 30):
        if len(islands) >= n_isl:
            break
        r = rng.uniform(4.5, 10.0)
        cx, cy = rng.uniform(30, W - 12), rng.uniform(12, H - 12)
        if abs(cx - bx) < r + 14:
            continue
        if math.hypot(cx - launch[0], cy - launch[1]) < 35:
            continue
        if any(math.hypot(cx - p[:, 0].mean(), cy - p[:, 1].mean()) < r + 16 + 10 for p in islands):
            continue
        islands.append(_blob(rng, cx, cy, r))

    # launch pier on the left shore
    piers = [_rect(3.0, launch[1] + 6.5, 6.0, 1.6, 0.0), _rect(3.0, launch[1] - 6.5, 6.0, 1.6, 0.0)]

    # currents
    crng = stream(seed, "currents")
    ang = crng.uniform(0, 2 * np.pi)
    s = cfg.current_strength
    drift = 0.3 * s * np.array([math.cos(ang), math.sin(ang)])
    vort = []
    for _ in range(int(round(3 * scale))):
        vort.append([crng.uniform(0.15 * W, 0.95 * W), crng.uniform(0.1 * H, 0.9 * H),
                     crng.uniform(18, 34), crng.choice([-1, 1]) * crng.uniform(0.6, 1.0) * s * 1.6])
    jets = [[ch.center[0], ch.center[1], 14.0, cfg.channel_width_m * 0.45,
             crng.choice([-1, 1]) * crng.uniform(0.5, 1.0) * s] for ch in channels]
    current = CurrentField(drift, np.array(vort).reshape(-1, 4), np.array(jets).reshape(-1, 5),
                           unk_amp=cfg.current_unknown_frac * s,
                           unk_period=crng.uniform(70, 130), unk_phase=crng.uniform(0, 2 * np.pi))

    w = World(cfg, seed, W, H, islands, rocks, piers, channels, bx, launch, current)
    w.build_grid()
    # reachability from launch
    dist, _ = w.field_to(*launch)
    w._launch_reach = np.isfinite(dist)

    # dock sites: berths beside islands (pier drawn, berth point in water)
    for isl in islands:
        c = isl.mean(0)
        for _ in range(12):
            a = rng.uniform(0, 2 * np.pi)
            d = np.array([math.cos(a), math.sin(a)])
            # march outward until water with clearance
            for rr in np.arange(3, 20, 0.5):
                p = c + d * rr
                if w.clearance_at(*p) >= 1.5:
                    break
            berth = p + d * 4.2
            if (w.clearance_at(*berth) >= 2.6 and w.reachable(*berth)
                    and all(math.hypot(berth[0] - x, berth[1] - y) > 15 for x, y, _ in w.dock_sites)):
                w.dock_sites.append((float(berth[0]), float(berth[1]), float(a + math.pi)))
                pc = p + d * 1.2
                w.piers.append(_rect(pc[0], pc[1], 3.2, 1.2, a))
                break
    if w.dock_sites:
        w.build_grid()                          # piers are obstacles too
        w._fields.clear()
        dist, _ = w.field_to(*launch)
        w._launch_reach = np.isfinite(dist)
    return w
