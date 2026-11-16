"""Team map: a boat's own tracker plus what teammates told it, with a camera-coverage grid for exploration.

Digest wire format (all plain tuples, bounded):
  digest = (tracks, tiles, gone)
  tracks = ((tid, x, y, P, last_t, cam_hits, class_post(10), attr), ...)   at most k, most recently updated first
           attr = () | ('mark', post(2)) | ('banner', post(12)) | ('light', post(2), light_t)
  tiles  = (flat tile index, ...)   tiles newly covered by this boat's camera since its previous digest
  gone   = (tid, ...)                 sender's tracks merged away since its previous digest (receivers drop them)

Teammate tracks are stored per (sender, sender's tid) and replaced by newer digests; the fused view associates them
with the boat's own tracks and with each other by position (<= 1.2 m) + class, so one physical object reported by
several boats is one object.  The fused class posterior is the member with the most camera evidence (no
multiplication of possibly repeated evidence); positions are inverse-variance fused.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .sensors import CAM_FOV, CAM_RANGE
from .tracking import Region, Track, Tracker, _softmax, count_tracks

ASSOC_DIST = 1.2
MAX_TILE_DELTA = 64


@dataclass(frozen=True)
class Message2:
    """Status broadcast for v2; compatible with src.comms.Network (uses .sender and .t_sent)."""
    sender: int
    t_sent: float
    x: float
    y: float
    psi: float
    v: float
    claim: Optional[int]
    claim_arrival: float
    completed: FrozenSet[int]
    digest: tuple = ()


def same_object(a: Track, b: Track) -> bool:
    """Class compatibility for association (position is checked by the caller)."""
    ca, cb = a.best_class(), b.best_class()
    if (ca == "banner") != (cb == "banner"):
        return False
    if ca is None or cb is None or ca == cb:
        return True
    if min(a.confidence(), b.confidence()) < 0.8:
        return True
    lo, hi = sorted((a.cam_hits, b.cam_hits))       # a few-view track beside a well-observed one: misclassified copy
    return lo <= 5 and hi >= 5 * lo


def _r(v, nd=3):
    return tuple(round(float(a), nd) for a in v)


def _log(p):
    return np.log(np.maximum(np.asarray(p, float), 1e-6))


class TeamMap:
    def __init__(self, boat: int, width: float, height: float, tile: float = 4.0, tracker: Optional[Tracker] = None):
        self.boat = boat
        self.tracker = tracker or Tracker()
        self.tile = tile
        self.nx, self.ny = int(math.ceil(width / tile)), int(math.ceil(height / tile))
        cx = (np.arange(self.nx) + 0.5) * tile
        cy = (np.arange(self.ny) + 0.5) * tile
        self._cx, self._cy = np.meshgrid(cx, cy, indexing="ij")          # (nx, ny)
        self.seen = np.zeros((self.nx, self.ny), bool)                    # own camera coverage
        self.told = np.zeros((self.nx, self.ny), bool)                    # coverage reported by teammates
        self._delta: set = set()
        self.remote: Dict[Tuple[int, int], Track] = {}
        self.t = 0.0

    # --- own sensing ---
    def observe(self, t: float, pose, detections: Sequence, camera_ran: bool = True) -> List[int]:
        """Feed this boat's detections (camera and/or lidar); mark camera coverage when the camera ran."""
        self.t = max(self.t, t)
        if camera_ran:
            self.mark_coverage(pose)
        return self.tracker.update(detections, t)

    def mark_coverage(self, pose):
        x, y, psi = pose[0], pose[1], pose[2]
        dx, dy = self._cx - x, self._cy - y
        off = np.abs((np.arctan2(dy, dx) - psi + np.pi) % (2 * np.pi) - np.pi)
        m = (np.hypot(dx, dy) <= CAM_RANGE) & ((off <= CAM_FOV / 2) | (np.hypot(dx, dy) <= self.tile * 0.75))
        new = m & ~self.seen
        if new.any():
            self._delta.update((np.ravel_multi_index(np.nonzero(new), self.seen.shape)).tolist())
        self.seen |= m

    # --- comms ---
    def digest(self, t: float, k: int = 12) -> tuple:
        trs = sorted(self.tracker.confirmed(), key=lambda tr: (-tr.last_t, tr.tid))[:k]
        out = []
        for tr in trs:
            if tr.light_hits:
                attr = ("light", _r(_softmax(tr.light_logl)), round(tr.light_t, 3))
            elif tr.banner_hits:
                attr = ("banner", _r(list(tr.banner_probs().values()), 4))
            elif tr.mark_hits:
                attr = ("mark", _r(_softmax(tr.mark_logl)))
            else:
                attr = ()
            out.append((tr.tid, round(tr.x, 3), round(tr.y, 3), round(tr.P, 5), round(tr.last_t, 3), tr.cam_hits,
                        _r(tr.class_post()) if tr.cam_hits else (), attr))
        tiles = tuple(sorted(self._delta)[:MAX_TILE_DELTA])
        self._delta.difference_update(tiles)
        gone = tuple(self.tracker.dropped[:MAX_TILE_DELTA])
        del self.tracker.dropped[:MAX_TILE_DELTA]
        return (tuple(out), tiles, gone)

    def merge(self, digest: tuple, sender: int, t_sent: float):
        if not digest:
            return
        tracks, tiles, gone = digest
        for tid in gone:                                  # sender merged these tracks away
            self.remote.pop((sender, tid), None)
        for (tid, x, y, P, last_t, cam_hits, post, attr) in tracks:
            tr = Track(tid, x, y, max(P, 1e-4), last_t=last_t, first_t=last_t, confirmed=True, source=sender, t_sent=t_sent)
            tr.cam_hits = tr.hits = cam_hits
            if cam_hits and post:
                tr.class_logl = _log(post)
                tr.class_counts = tr.class_counts + cam_hits * np.asarray(post)
            if attr:
                if attr[0] == "light":
                    tr.light_logl, tr.light_t, tr.light_hits = _log(attr[1]), attr[2], 1
                elif attr[0] == "banner":
                    tr.banner_logl, tr.banner_hits = _log(attr[1]), 1
                elif attr[0] == "mark":
                    tr.mark_logl, tr.mark_hits = _log(attr[1]), 1
            old = self.remote.get((sender, tid))
            if old is None or old.t_sent <= t_sent:
                self.remote[(sender, tid)] = tr
        for f in tiles:
            self.told[np.unravel_index(int(f), self.told.shape)] = True

    # --- fused view ---
    def fused(self, min_conf: float = 0.0) -> List[Track]:
        """Own confirmed tracks + teammate tracks, one entry per physical object (copies; safe to inspect)."""
        groups: List[List[Track]] = []
        cands = self.tracker.confirmed() + sorted(self.remote.values(), key=lambda tr: (-tr.last_t, str(tr.source)))
        for tr in cands:
            best, bd = None, ASSOC_DIST
            for g in groups:
                d = math.hypot(tr.x - g[0].x, tr.y - g[0].y)
                if d <= bd and same_object(tr, g[0]):
                    best, bd = g, d
            if best is None:
                groups.append([tr])
            else:
                best.append(tr)
        out = []
        for g in groups:
            ident = max(g, key=lambda tr: (tr.cam_hits, tr.source == "own"))
            f = copy.deepcopy(ident)
            w = np.array([1.0 / tr.P for tr in g])
            f.x = float(np.dot(w, [tr.x for tr in g]) / w.sum())
            f.y = float(np.dot(w, [tr.y for tr in g]) / w.sum())
            f.P = float(1.0 / w.sum())
            f.last_t = max(tr.last_t for tr in g)
            for attr in ("banner", "mark"):
                if not getattr(f, attr + "_hits"):
                    src = [tr for tr in g if getattr(tr, attr + "_hits")]
                    if src:
                        s = max(src, key=lambda tr: getattr(tr, attr + "_hits"))
                        setattr(f, attr + "_logl", getattr(s, attr + "_logl").copy())
                        setattr(f, attr + "_hits", getattr(s, attr + "_hits"))
            lit = [tr for tr in g if tr.light_hits]
            if lit:
                s = max(lit, key=lambda tr: tr.light_t)
                f.light_logl, f.light_t, f.light_hits = s.light_logl.copy(), s.light_t, s.light_hits
            f.members = [(tr.source, tr.tid) for tr in g]
            if f.confidence() >= min_conf or min_conf <= 0:
                out.append(f)
        return out

    def objects(self, cls: Optional[str] = None, min_conf: float = 0.5) -> List[Track]:
        return [f for f in self.fused() if (cls is None or f.best_class() == cls) and f.confidence() >= min_conf]

    def banner_bays(self, min_conf: float = 0.0) -> List[Tuple[float, float, dict]]:
        """[(x, y, {(shape, colour): prob})] for every detected dock banner."""
        return [(f.x, f.y, f.banner_probs()) for f in self.fused()
                if f.best_class() == "banner" and f.banner_hits and f.confidence() >= min_conf]

    def light_state(self) -> Optional[Tuple[str, float, float]]:
        """(state, prob, time of the reading) from the most recent light-panel reading (own or teammate)."""
        lit = [tr for tr in self.tracker.tracks + list(self.remote.values()) if tr.light_hits]
        if not lit:
            return None
        return max(lit, key=lambda tr: tr.light_t).light()

    def count(self, cls: str, region: Region = None, min_conf: float = 0.5) -> int:
        return count_tracks(self.fused(), cls, region, min_conf)

    def remote_age(self, now: float) -> Dict[Tuple[int, int], float]:
        return {k: now - tr.t_sent for k, tr in self.remote.items()}

    def unexplored_tiles(self) -> List[Tuple[float, float]]:
        """Centres of 4 m tiles neither this boat's camera nor any teammate has covered."""
        m = ~(self.seen | self.told)
        return [(float(self._cx[i, j]), float(self._cy[i, j])) for i, j in zip(*np.nonzero(m))]
