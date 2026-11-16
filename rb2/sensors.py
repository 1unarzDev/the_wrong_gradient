"""Perception sensors for course v2: a forward camera (classes + attributes) and a 360-degree lidar (geometry).

Units metres / seconds; boat pose (x, y, psi), psi measured from +x, y up.

Determinism / common random numbers
-----------------------------------
Every random number comes from a counter-based generator keyed by (seed, tick, boat, STREAM_KEY):
``np.random.default_rng(np.random.SeedSequence([seed, tick, boat, key, sub]))`` where key is the crc32 of the
named rb2.rng2 stream ("v2:sensors_cam" / "v2:sensors_lidar").  Each (tick, boat) draws a FIXED-size block whose
rows are indexed by target (object id, then bay index for banners), so what a boat sees at a tick does not depend on
what it (or any other boat, or any policy) saw before, nor on which objects happen to be visible.

`Detection.truth` is the true object id (dock banners: BANNER_TRUTH_BASE + bay index).  It exists ONLY for
evaluation logs; policies, trackers and team maps must never read it.
"""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .course import COLORS, SHAPES
from .geom import STATIC_CLASSES, Obstacles, cast

# ---- class vocabulary ------------------------------------------------------------------------------------------
CAM_CLASSES = ("red", "green", "yellow", "black", "blue", "vessel_orange", "vessel_black", "light_panel", "marker")
TRUTH_TO_CAM = {"red_tall": "red", "green_tall": "green", **{c: c for c in CAM_CLASSES}}
MARKS = ("triangle", "plus")
BANNERS = tuple((s, c) for s in SHAPES for c in COLORS)       # 12 (shape, colour) combinations
LIGHTS = ("red", "green")
BANNER_TRUTH_BASE = 100000

CAM_FOV = math.radians(90.0)
CAM_RANGE = 18.0
CAM_HZ = 5.0
MARK_RANGE = 9.0
BANNER_RANGE = 7.0
BANNER_MAX_INCIDENCE = math.radians(70.0)   # banner must be viewed from its front side
LIGHT_RANGE = 15.0
LIDAR_RANGE = 20.0
LIDAR_HZ = 10.0
LIDAR_RES_DEG = 0.5

CAM_KEY = zlib.crc32(b"v2:sensors_cam")
LIDAR_KEY = zlib.crc32(b"v2:sensors_lidar")


def counter_rng(seed: int, tick: int, boat: int, key: int, sub: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(tick), int(boat), int(key), int(sub)]))


def due(t: float, dt: float, hz: float) -> bool:
    """True when a sensor running at `hz` fires in the step (t - dt, t] (use with the sim tick)."""
    return math.floor(t * hz + 1e-6) != math.floor((t - dt) * hz + 1e-6)


@dataclass
class Detection:
    t: float
    boat: int
    sensor: str                         # 'cam' | 'lidar'
    x: float
    y: float
    sigma: float                        # position std (m)
    probs: Optional[Dict[str, float]]   # camera: class -> prob (incl. 'banner'); lidar: None
    attrs: Dict = field(default_factory=dict)   # 'mark' | 'banner' | 'light' -> prob dicts
    truth: int = -1                     # EVALUATION ONLY - policies must never read this


# ---- confusion models ------------------------------------------------------------------------------------------
_CONF_NEAR = {
    "red": {"yellow": 1.0},
    "yellow": {"red": 1.0},
    "green": {"blue": 1.0},
    "blue": {"green": 0.5, "black": 0.5},
    "black": {"blue": 0.8, "vessel_black": 0.2},
    "vessel_orange": {"vessel_black": 1.0},
    "vessel_black": {"black": 0.5, "vessel_orange": 0.5},
    "light_panel": {"marker": 1.0},
    "marker": {"black": 0.5, "light_panel": 0.5},
}
_CONF_FAR = {**_CONF_NEAR,                                  # long range: orange vessel <-> red buoy
             "red": {"yellow": 0.7, "vessel_orange": 0.3},
             "vessel_orange": {"red": 0.7, "vessel_black": 0.3}}
_SHAPE_CONF = {"circle": {"square": 0.6, "triangle": 0.2, "plus": 0.2},
               "triangle": {"plus": 0.5, "circle": 0.25, "square": 0.25},
               "square": {"circle": 0.6, "triangle": 0.2, "plus": 0.2},
               "plus": {"triangle": 0.5, "circle": 0.25, "square": 0.25}}
_COLOR_CONF = {"blue": {"green": 0.8, "red": 0.2}, "green": {"blue": 0.8, "red": 0.2}, "red": {"green": 0.5, "blue": 0.5}}


def _rows(names, conf, floor=0.1):
    """Off-diagonal confusion rows (each sums to 1): (1-floor) on the listed confusions, floor uniform elsewhere."""
    n = len(names)
    Q = np.zeros((n, n))
    for i, a in enumerate(names):
        tot = sum(conf[a].values())
        for b, w in conf[a].items():
            Q[i, names.index(b)] += (1 - floor) * w / tot
        Q[i] += floor / (n - 1)
        Q[i, i] = 0.0
        Q[i] /= Q[i].sum()
    return Q


_Q_NEAR = _rows(CAM_CLASSES, _CONF_NEAR)
_Q_FAR = _rows(CAM_CLASSES, _CONF_FAR)
_Q_SHAPE = _rows(SHAPES, _SHAPE_CONF, 0.0)
_Q_COLOR = _rows(COLORS, _COLOR_CONF, 0.0)


def class_correct_prob(d):
    return max(0.55, 0.96 - 0.018 * d)


def confusion_matrix(d: float) -> np.ndarray:
    """C[true, reported] for the camera classifier at range d."""
    s = min(1.0, max(0.0, (d - 9.0) / 6.0))
    pc = class_correct_prob(d)
    return pc * np.eye(len(CAM_CLASSES)) + (1 - pc) * ((1 - s) * _Q_NEAR + s * _Q_FAR)


def p_detect(d, off):
    """Camera detection probability at range d (m) and off-axis angle off (rad); vectorised."""
    d, off = np.asarray(d, float), np.abs(np.asarray(off, float))
    p = 0.97 * np.exp(-(d / 14.0) ** 2) * (1 - 0.4 * (off / (CAM_FOV / 2)) ** 2)
    return np.where((d <= CAM_RANGE) & (off <= CAM_FOV / 2), p, 0.0)


def mark_correct_prob(d):
    return 0.98 - 0.23 * (d / MARK_RANGE) ** 2


def banner_correct_probs(d):
    """(shape, colour) correct-read probabilities; colour is more robust than shape."""
    q = (d / BANNER_RANGE) ** 2
    return 0.96 - 0.30 * q, 0.98 - 0.12 * q


def light_correct_prob(d):
    return 0.99 - 0.12 * (d / LIGHT_RANGE) ** 2


def _sample(row, u):
    return min(int(np.searchsorted(np.cumsum(row), u * row.sum(), side="right")), len(row) - 1)


def _posterior(C, rep):
    col = C[:, rep]
    return col / col.sum()


# ---- per-course precomputation -----------------------------------------------------------------------------------
class _Prep:
    def __init__(self, course, obstacles=None):
        self.course = course
        self.obs = obstacles if obstacles is not None else Obstacles(course)
        objs = course.objects
        self.n_obj = len(objs)
        self.cam_cls = np.array([CAM_CLASSES.index(TRUTH_TO_CAM[o.cls]) if o.cls in TRUTH_TO_CAM else -1
                                 for o in objs], int)
        self.lidar_ok = np.array([o.cls not in STATIC_CLASSES for o in objs], bool)
        self.mark = [MARKS.index(o.mark) if o.mark in MARKS else -1 for o in objs]
        self.bays = [b for m in course.modules if m.kind == "T3" for b in m.bays]
        self.bxy = np.array([b.banner for b in self.bays], float).reshape(-1, 2)
        self.bout = np.array([b.heading + math.pi for b in self.bays], float)   # banner faces out of the bay
        self.bshape = [SHAPES.index(b.shape) for b in self.bays]
        self.bcolor = [COLORS.index(b.color) for b in self.bays]
        self.n_targets = self.n_obj + len(self.bays)


class _PrepCache:
    def __init__(self):
        self._key, self._prep = None, None

    def get(self, course, obstacles):
        key = (id(course), id(obstacles))
        if key != self._key:
            self._key, self._prep = key, _Prep(course, obstacles)
        return self._prep


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- camera ------------------------------------------------------------------------------------------------------
class Camera:
    """Forward camera: FOV 90 deg, range 18 m, 5 Hz (caller schedules it, see `due`)."""
    hz = CAM_HZ

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._cache = _PrepCache()

    def sense(self, t: float, tick: int, boat_idx: int, pose, course, obstacles: Optional[Obstacles] = None,
              light: str = "red") -> List[Detection]:
        """Detections for one boat at one tick.  `light` is the TRUE light-panel state ('red'|'green')."""
        P = self._cache.get(course, obstacles)
        seed = course.seed if self.seed is None else self.seed
        rng = counter_rng(seed, tick, boat_idx, CAM_KEY)
        U = rng.random((P.n_targets, 6))            # detect, class, mark, shape, colour, light
        N = rng.standard_normal((P.n_targets, 2))   # position noise
        bx, by, psi = float(pose[0]), float(pose[1]), float(pose[2])
        out: List[Detection] = []
        obs = P.obs
        # objects
        if P.n_obj:
            dxy = obs.xy - (bx, by)
            d = np.hypot(dxy[:, 0], dxy[:, 1])
            bear = np.arctan2(dxy[:, 1], dxy[:, 0])
            off = _wrap(bear - psi)
            pd = p_detect(d, off) * (P.cam_cls >= 0)
            cand = np.nonzero(U[:P.n_obj, 0] < pd)[0]
            if len(cand):
                rr, _ = cast(obs, bx, by, bear[cand], CAM_RANGE + 1.0, exclude=cand)
                vis = cand[rr >= d[cand] - obs.ext[cand] - 0.05]
                for i in vis:
                    out.append(self._object_det(t, boat_idx, P, int(i), float(d[i]), U[i], N[i], light))
        # dock banners
        if len(P.bays):
            dxy = P.bxy - (bx, by)
            d = np.hypot(dxy[:, 0], dxy[:, 1])
            bear = np.arctan2(dxy[:, 1], dxy[:, 0])
            off = _wrap(bear - psi)
            inc = np.abs(_wrap(bear + np.pi - P.bout))
            pd = p_detect(d, off) * (d <= BANNER_RANGE) * (inc <= BANNER_MAX_INCIDENCE)
            Ub, Nb = U[P.n_obj:], N[P.n_obj:]
            cand = np.nonzero(Ub[:, 0] < pd)[0]
            if len(cand):
                rr, _ = cast(obs, bx, by, bear[cand], BANNER_RANGE + 1.0)
                for k in cand[rr >= d[cand] - 0.9]:          # banner sits ~0.76 m behind the spine face
                    out.append(self._banner_det(t, boat_idx, P, int(k), float(d[k]), Ub[k], Nb[k]))
        return out

    @staticmethod
    def _object_det(t, boat, P, i, d, u, n, light):
        o = P.course.objects[i]
        C = confusion_matrix(d)
        true = P.cam_cls[i]
        rep = _sample(C[true], u[1])
        post = _posterior(C, rep)
        probs = {c: float(p) for c, p in zip(CAM_CLASSES, post)}
        sig = 0.05 + 0.02 * d
        attrs = {}
        if P.mark[i] >= 0 and d <= MARK_RANGE:
            pm = mark_correct_prob(d)
            r = P.mark[i] if u[2] < pm else 1 - P.mark[i]
            attrs["mark"] = {MARKS[r]: pm, MARKS[1 - r]: 1 - pm}
        if o.cls == "light_panel" and d <= LIGHT_RANGE:
            pl = light_correct_prob(d)
            li = LIGHTS.index(light)
            r = li if u[5] < pl else 1 - li
            attrs["light"] = {LIGHTS[r]: pl, LIGHTS[1 - r]: 1 - pl}
        return Detection(t, boat, "cam", o.x + sig * n[0], o.y + sig * n[1], sig, probs, attrs, o.id)

    @staticmethod
    def _banner_det(t, boat, P, k, d, u, n):
        ps, pc = banner_correct_probs(d)
        Cs = ps * np.eye(len(SHAPES)) + (1 - ps) * _Q_SHAPE
        Cc = pc * np.eye(len(COLORS)) + (1 - pc) * _Q_COLOR
        rs, rc = _sample(Cs[P.bshape[k]], u[3]), _sample(Cc[P.bcolor[k]], u[4])
        joint = np.outer(_posterior(Cs, rs), _posterior(Cc, rc)).ravel()
        sig = 0.05 + 0.02 * d
        x, y = P.bxy[k]
        return Detection(t, boat, "cam", x + sig * n[0], y + sig * n[1], sig, {"banner": 1.0},
                         {"banner": {bc: float(p) for bc, p in zip(BANNERS, joint)}}, BANNER_TRUTH_BASE + k)


# ---- lidar -------------------------------------------------------------------------------------------------------
class Lidar:
    """360-degree lidar: range 20 m, 0.5 deg resolution, 10 Hz.  Positions only."""
    hz = LIDAR_HZ

    def __init__(self, seed: Optional[int] = None, res_deg: float = LIDAR_RES_DEG, rmax: float = LIDAR_RANGE,
                 sigma_pt: float = 0.03, sigma_obj: float = 0.06):
        self.seed, self.rmax, self.sigma_pt, self.sigma_obj = seed, rmax, sigma_pt, sigma_obj
        self.res = math.radians(res_deg)
        self.n_rays = int(round(2 * math.pi / self.res))
        self._cache = _PrepCache()

    def _seed(self, course):
        return course.seed if self.seed is None else self.seed

    def scan(self, pose, course, obstacles: Optional[Obstacles] = None, tick: Optional[int] = None, boat: int = 0):
        """Full sweep. Returns (angles (R,), ranges (R,), owner (R,)); ranges carry noise when tick is given;
        owner is the object id hit, geom.STATIC for structure, geom.NONE for no return."""
        P = self._cache.get(course, obstacles)
        bx, by, psi = float(pose[0]), float(pose[1]), float(pose[2])
        ang = psi + self.res * np.arange(self.n_rays)
        rng_, who = cast(P.obs, bx, by, ang, self.rmax)
        if tick is not None:
            noise = counter_rng(self._seed(course), tick, boat, LIDAR_KEY, 1).standard_normal(self.n_rays)
            rng_ = np.where(who != -2, rng_ + self.sigma_pt * noise, rng_)
        return ang, rng_, who

    def scan_points(self, pose, course, obstacles: Optional[Obstacles] = None, tick: Optional[int] = None,
                    boat: int = 0) -> np.ndarray:
        """(P, 2) array of hit points (rays with a return only)."""
        ang, r, who = self.scan(pose, course, obstacles, tick, boat)
        m = who != -2
        return np.column_stack((pose[0] + r[m] * np.cos(ang[m]), pose[1] + r[m] * np.sin(ang[m])))

    def sense(self, t: float, tick: int, boat_idx: int, pose, course,
              obstacles: Optional[Obstacles] = None) -> List[Detection]:
        """One Detection per (non-structure) object whose surface is hit by >= 1 ray of the sweep."""
        P = self._cache.get(course, obstacles)
        N = counter_rng(self._seed(course), tick, boat_idx, LIDAR_KEY, 0).standard_normal((P.n_obj, 2))
        if not P.n_obj:
            return []
        obs = P.obs
        bx, by, psi = float(pose[0]), float(pose[1]), float(pose[2])
        dxy = obs.xy - (bx, by)
        d = np.hypot(dxy[:, 0], dxy[:, 1])
        cand = np.nonzero(P.lidar_ok & (d - obs.ext <= self.rmax))[0]
        if not len(cand):
            return []
        bear = np.arctan2(dxy[cand, 1], dxy[cand, 0])
        w = np.arcsin(np.minimum(1.0, obs.ext[cand] / np.maximum(d[cand], 1e-6)))
        rel = (bear - psi) % (2 * np.pi)
        lo = np.ceil((rel - w) / self.res).astype(int)
        hi = np.floor((rel + w) / self.res).astype(int)
        ks = np.unique(np.concatenate([np.arange(a, b + 1) for a, b in zip(lo, hi)] + [np.zeros(0, int)])
                       % self.n_rays)
        if not len(ks):
            return []
        _, who = cast(obs, bx, by, psi + self.res * ks, self.rmax, culled=obs.cull(bx, by, self.rmax))
        hit = set(who[who >= 0].tolist())
        s = self.sigma_obj
        return [Detection(t, boat_idx, "lidar", float(obs.xy[i, 0] + s * N[i, 0]), float(obs.xy[i, 1] + s * N[i, 1]),
                          s, None, {}, int(i)) for i in cand if int(i) in hit]


# ---- demo --------------------------------------------------------------------------------------------------------
def demo_stats(seed: int = 3, n_ticks: int = 400):
    """Print camera/lidar detection and confusion statistics for one generated course, plus sensing cost."""
    from .generate import generate_course
    c = generate_course(seed)
    obs = Obstacles(c)
    cam, lid = Camera(), Lidar()
    rng = np.random.default_rng(0)
    poses = [(rng.uniform(5, c.width - 5), rng.uniform(5, c.height - 5), rng.uniform(-math.pi, math.pi))
             for _ in range(n_ticks)]
    conf = np.zeros((len(CAM_CLASSES), len(CAM_CLASSES)), int)
    n_cam = n_lid = n_ban = ban_ok = 0
    t0 = time.perf_counter()
    for k, p in enumerate(poses):
        for det in cam.sense(k * 0.2, k, 0, p, c, obs):
            if det.truth >= BANNER_TRUTH_BASE:
                n_ban += 1
                b = [b for m in c.modules if m.kind == "T3" for b in m.bays][det.truth - BANNER_TRUTH_BASE]
                ban_ok += max(det.attrs["banner"], key=det.attrs["banner"].get) == (b.shape, b.color)
                continue
            n_cam += 1
            tr = CAM_CLASSES.index(TRUTH_TO_CAM[c.objects[det.truth].cls])
            conf[tr, CAM_CLASSES.index(max(det.probs, key=det.probs.get))] += 1
    t_cam = (time.perf_counter() - t0) / n_ticks
    t0 = time.perf_counter()
    for k, p in enumerate(poses):
        n_lid += len(lid.sense(k * 0.1, k, 0, p, c, obs))
    t_lid = (time.perf_counter() - t0) / n_ticks
    t0 = time.perf_counter()
    for k, p in enumerate(poses[:100]):
        lid.scan_points(p, c, obs, k, 0)
    t_scan = (time.perf_counter() - t0) / 100
    print(f"course seed {seed}: {len(c.objects)} objects, {n_ticks} random poses")
    print(f"camera: {n_cam / n_ticks:.2f} object dets/tick, banners read {n_ban} (argmax correct {ban_ok})")
    print(f"lidar : {n_lid / n_ticks:.2f} object dets/tick")
    print("confusion (rows true, cols argmax reported):")
    print("              " + " ".join(f"{c_[:6]:>6}" for c_ in CAM_CLASSES))
    for i, a in enumerate(CAM_CLASSES):
        if conf[i].sum():
            print(f"{a:>13} " + " ".join(f"{v:6d}" for v in conf[i]) + f"   acc {conf[i, i] / conf[i].sum():.2f}")
    print(f"cost per boat per tick: camera {1e3 * t_cam:.3f} ms, lidar objects {1e3 * t_lid:.3f} ms, "
          f"lidar scan_points {1e3 * t_scan:.3f} ms")
    return dict(conf=conf, t_cam=t_cam, t_lid=t_lid, t_scan=t_scan)


if __name__ == "__main__":
    demo_stats()
