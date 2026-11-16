"""Per-boat object tracker for course v2 (static objects).

Association: gated nearest neighbour (gate max(1.2 m, 3 sigma)), greedy one-to-one per batch, class-compatibility
check.  Position: Kalman update for a static point (isotropic variance P).  Identity: two parallel summaries per
track -
  * class_counts - Dirichlet pseudo-counts (prior 0.1 per class) incremented by the camera's soft probs; dirichlet_mean()
    is the long-run frequency view;
  * class_logl   - tempered naive-Bayes log evidence (sum of tau*log probs, tau = 0.6 for correlated errors); this is
    what best_class()/confidence() use, because repeated independent views should raise confidence above the
    single-view accuracy (Dirichlet means plateau at the per-view accuracy).
Confirmation: two class-consistent camera hits at most CONFIRM_WINDOW s apart plus lidar support (>= 1 lidar hit;
banners, which lidar does not report, are exempt; otherwise three consistent camera hits within the window), or
>= 3 hits of which >= 1 is lidar.  The lidar/time-window conditions reject coincident camera clutter.  Attributes (vessel mark, banner (shape, colour), light state) use the same two summaries; the light summary decays
(half-life 1 s) so a switch of the light is followed.  Lidar updates position / existence only.

Track.truth_votes is filled from Detection.truth for EVALUATION ONLY; nothing in this module reads it for decisions.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .sensors import BANNERS, CAM_CLASSES, LIGHTS, MARKS, Detection

TRACK_CLASSES = CAM_CLASSES + ("banner",)
_CI = {c: i for i, c in enumerate(TRACK_CLASSES)}
BANNER = _CI["banner"]
PRIOR = 0.1
TAU = 0.6
LIGHT_HALF_LIFE = 1.0
P_FLOOR = 0.02 ** 2
CONFIRM_WINDOW = 2.0      # two class-consistent camera hits must be at most this far apart (s)


def _vec(d: dict, names) -> np.ndarray:
    return np.array([d.get(n, 0.0) for n in names], float)


def _softmax(l):
    e = np.exp(l - l.max())
    return e / e.sum()


@dataclass
class Track:
    tid: int
    x: float
    y: float
    P: float                                   # position variance (m^2)
    class_counts: np.ndarray = field(default_factory=lambda: np.full(len(TRACK_CLASSES), PRIOR))
    class_logl: np.ndarray = field(default_factory=lambda: np.zeros(len(TRACK_CLASSES)))
    mark_counts: np.ndarray = field(default_factory=lambda: np.full(len(MARKS), PRIOR))
    mark_logl: np.ndarray = field(default_factory=lambda: np.zeros(len(MARKS)))
    banner_counts: np.ndarray = field(default_factory=lambda: np.full(len(BANNERS), PRIOR))
    banner_logl: np.ndarray = field(default_factory=lambda: np.zeros(len(BANNERS)))
    light_counts: np.ndarray = field(default_factory=lambda: np.full(len(LIGHTS), PRIOR))
    light_logl: np.ndarray = field(default_factory=lambda: np.zeros(len(LIGHTS)))
    votes: np.ndarray = field(default_factory=lambda: np.zeros(len(TRACK_CLASSES), int))   # camera argmax votes
    hits: int = 0
    cam_hits: int = 0
    lidar_hits: int = 0
    mark_hits: int = 0
    banner_hits: int = 0
    light_hits: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    light_t: float = -math.inf
    confirmed: bool = False
    source: Union[str, int] = "own"            # 'own' or teammate boat id
    t_sent: Optional[float] = None             # teammate tracks: when the digest was sent
    recent: list = field(default_factory=list)  # last camera hits as (t, argmax class index)
    truth_votes: Counter = field(default_factory=Counter)   # EVALUATION ONLY

    # --- identity helpers ---
    def class_post(self) -> np.ndarray:
        return _softmax(self.class_logl)

    def best_class(self) -> Optional[str]:
        """Most probable class, or None for an existence-only (lidar) track."""
        if not self.cam_hits:
            return None
        return TRACK_CLASSES[int(np.argmax(self.class_logl))]

    def confidence(self) -> float:
        return float(self.class_post().max()) if self.cam_hits else 0.0

    def dirichlet_mean(self) -> np.ndarray:
        return self.class_counts / self.class_counts.sum()

    def mark(self) -> Optional[Tuple[str, float]]:
        if not self.mark_hits:
            return None
        p = _softmax(self.mark_logl)
        return MARKS[int(np.argmax(p))], float(p.max())

    def banner_probs(self) -> Optional[dict]:
        if not self.banner_hits:
            return None
        return {bc: float(p) for bc, p in zip(BANNERS, _softmax(self.banner_logl))}

    def light(self) -> Optional[Tuple[str, float, float]]:
        """(state, prob, time of last light reading)."""
        if not self.light_hits:
            return None
        p = _softmax(self.light_logl)
        return LIGHTS[int(np.argmax(p))], float(p.max()), self.light_t

    def compatible(self, probs: Optional[dict]) -> bool:
        """Can a detection with these class probs be the same object as this track?"""
        if probs is None:                       # lidar: anything but a banner
            return not (self.cam_hits and self.best_class() == "banner")
        if not self.cam_hits:
            return "banner" not in probs        # lidar-only tracks never are banners
        mine = self.best_class()
        if (mine == "banner") != ("banner" in probs):
            return False
        return not (self.confidence() >= 0.8 and probs.get(mine, 0.0) < 0.02)

    # --- updates ---
    def absorb_position(self, x, y, var):
        k = self.P / (self.P + var)
        self.x += k * (x - self.x)
        self.y += k * (y - self.y)
        self.P = max(P_FLOOR, (1 - k) * self.P)

    def absorb(self, det: Detection):
        self.absorb_position(det.x, det.y, det.sigma ** 2)
        self.hits += 1
        self.last_t = max(self.last_t, det.t)
        if det.truth >= 0:
            self.truth_votes[det.truth] += 1
        if det.probs is None:
            self.lidar_hits += 1
            return
        self.cam_hits += 1
        v = _vec(det.probs, TRACK_CLASSES)
        self.class_counts += v
        self.class_logl += TAU * np.log(np.maximum(v, 1e-4))
        self.votes[int(np.argmax(v))] += 1
        self.recent = (self.recent + [(det.t, int(np.argmax(v)))])[-6:]
        a = det.attrs
        if "mark" in a:
            v = _vec(a["mark"], MARKS); self.mark_counts += v; self.mark_hits += 1
            self.mark_logl += TAU * np.log(np.maximum(v, 1e-4))
        if "banner" in a:
            v = _vec(a["banner"], BANNERS); self.banner_counts += v; self.banner_hits += 1
            self.banner_logl += TAU * np.log(np.maximum(v, 1e-4))
        if "light" in a:
            v = _vec(a["light"], LIGHTS)
            decay = 0.5 ** (max(0.0, det.t - self.light_t) / LIGHT_HALF_LIFE) if self.light_hits else 1.0
            self.light_counts = PRIOR + decay * (self.light_counts - PRIOR) + v
            self.light_logl = decay * self.light_logl + TAU * np.log(np.maximum(v, 1e-4))
            self.light_t = max(self.light_t, det.t); self.light_hits += 1

    def check_confirm(self):
        if not self.confirmed:
            b = int(np.argmax(self.class_logl))
            consistent = [tt for tt, a in self.recent if a == b]
            pair = len(consistent) >= 2 and consistent[-1] - consistent[-2] <= CONFIRM_WINDOW
            triple = len(consistent) >= 3 and consistent[-1] - consistent[-3] <= CONFIRM_WINDOW
            if (pair and (self.lidar_hits >= 1 or b == BANNER or triple)) or (self.hits >= 3 and self.lidar_hits >= 1):
                self.confirmed = True

    def merge_from(self, o: "Track"):
        """Fold duplicate track o into self (independent evidence: positions fused, evidence summed)."""
        self.absorb_position(o.x, o.y, o.P)
        for f in ("class_counts", "mark_counts", "banner_counts"):
            setattr(self, f, getattr(self, f) + getattr(o, f) - PRIOR)
        for f in ("class_logl", "mark_logl", "banner_logl", "votes"):
            setattr(self, f, getattr(self, f) + getattr(o, f))
        if o.light_hits and o.light_t > self.light_t:
            self.light_counts, self.light_logl, self.light_t = o.light_counts.copy(), o.light_logl.copy(), o.light_t
        for f in ("hits", "cam_hits", "lidar_hits", "mark_hits", "banner_hits", "light_hits"):
            setattr(self, f, getattr(self, f) + getattr(o, f))
        self.first_t = min(self.first_t, o.first_t)
        self.last_t = max(self.last_t, o.last_t)
        self.confirmed = self.confirmed or o.confirmed
        self.recent = sorted(self.recent + o.recent)[-6:]
        self.truth_votes.update(o.truth_votes)


Region = Union[None, Sequence[Tuple[float, float]], Callable[[float, float], bool]]


def point_in_polygon(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def in_region(x, y, region: Region) -> bool:
    if region is None:
        return True
    if callable(region):
        return bool(region(x, y))
    return point_in_polygon(x, y, region)


def count_tracks(tracks: Iterable[Track], cls: str, region: Region = None, min_conf: float = 0.5,
                 dedup_dist: float = 0.6) -> int:
    """Number of distinct confirmed objects of class cls (confidence >= min_conf) inside region.  Tracks of the same
    class closer than dedup_dist are counted once."""
    pts: List[Tuple[float, float]] = []
    for tr in tracks:
        if tr.confirmed and tr.best_class() == cls and tr.confidence() >= min_conf and in_region(tr.x, tr.y, region):
            if all(math.hypot(tr.x - a, tr.y - b) >= dedup_dist for a, b in pts):
                pts.append((tr.x, tr.y))
    return len(pts)


class Tracker:
    """One per boat. Feed it the boat's own camera + lidar detections with update()."""

    def __init__(self, gate: float = 1.2, merge_dist: float = 0.6, max_age_unconfirmed: float = 4.0):
        self.gate, self.merge_dist, self.max_age = gate, merge_dist, max_age_unconfirmed
        self.tracks: List[Track] = []
        self._next = 0
        self.t = 0.0
        self.dropped: List[int] = []          # confirmed track ids merged away (for team-map tombstones)

    def get(self, tid: int) -> Optional[Track]:
        for tr in self.tracks:
            if tr.tid == tid:
                return tr
        return None

    def confirmed(self) -> List[Track]:
        return [tr for tr in self.tracks if tr.confirmed]

    def update(self, detections: Sequence[Detection], t: Optional[float] = None) -> List[int]:
        """Associate, fuse, spawn, merge, age. Returns ids of tracks touched by this batch."""
        if t is None:
            t = max((d.t for d in detections), default=self.t)
        self.t = max(self.t, t)
        touched = set()
        if detections:
            n, m = len(detections), len(self.tracks)
            pairs = []
            if m:
                TX = np.array([[tr.x, tr.y, tr.P] for tr in self.tracks])
                for i, det in enumerate(detections):
                    d2 = (TX[:, 0] - det.x) ** 2 + (TX[:, 1] - det.y) ** 2
                    var = TX[:, 2] + det.sigma ** 2
                    gate2 = np.maximum(self.gate ** 2, 9.0 * var)
                    for j in np.nonzero(d2 <= gate2)[0]:
                        if self.tracks[j].compatible(det.probs):
                            pairs.append((d2[j] / var[j], i, int(j)))
            pairs.sort()
            used_d, used_t = set(), set()
            for _, i, j in pairs:
                if i in used_d or (detections[i].sensor, j) in used_t:
                    continue
                used_d.add(i); used_t.add((detections[i].sensor, j))  # one-to-one per sensor
                self.tracks[j].absorb(detections[i])
                touched.add(self.tracks[j].tid)
            for i, det in enumerate(detections):
                if i not in used_d:
                    tr = Track(self._next, det.x, det.y, det.sigma ** 2, first_t=det.t, last_t=det.t)
                    self._next += 1
                    tr.P = det.sigma ** 2 * 1e6        # absorb() then sets P ~= sigma^2
                    tr.x, tr.y = det.x, det.y
                    tr.absorb(det)
                    self.tracks.append(tr)
                    touched.add(tr.tid)
            for tr in self.tracks:
                if tr.tid in touched:
                    tr.check_confirm()
            self._merge(touched)
        self.tracks = [tr for tr in self.tracks if tr.confirmed or self.t - tr.last_t <= self.max_age]
        return sorted(tid for tid in touched if self.get(tid) is not None)

    def _merge(self, touched):
        if len(self.tracks) < 2:
            return
        alive = {tr.tid: tr for tr in self.tracks}
        for tid in sorted(touched):
            a = alive.get(tid)
            if a is None:
                continue
            for b in list(alive.values()):
                if b is a or math.hypot(a.x - b.x, a.y - b.y) >= self.merge_dist:
                    continue
                ca, cb = a.best_class(), b.best_class()
                if ca is not None and cb is not None and ca != cb and min(a.confidence(), b.confidence()) >= 0.8:
                    continue
                if (ca == "banner") != (cb == "banner") and ca is not None and cb is not None:
                    continue
                keep, drop = (a, b) if a.tid < b.tid else (b, a)
                if drop.confirmed:
                    self.dropped.append(drop.tid)
                keep.merge_from(drop)
                keep.check_confirm()
                del alive[drop.tid]
                if drop is a:
                    break
                a = keep
        self.tracks = [tr for tr in self.tracks if tr.tid in alive]

    def count(self, cls: str, region: Region = None, min_conf: float = 0.5) -> int:
        return count_tracks(self.tracks, cls, region, min_conf)
