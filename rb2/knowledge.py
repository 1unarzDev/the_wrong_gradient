"""Sensor-based knowledge: camera (5 Hz) + lidar (10 Hz) -> per-boat tracker -> TeamMap, shared over the lossy
radio as bounded digests. Everything a policy or executor knows comes from here (no truth access), except the
documented geometric lookups that map a DETECTED object to the module geometry used for manoeuvring
(e.g. which bay a detected banner belongs to)."""
from __future__ import annotations

import math

from .geom import Obstacles
from .sensors import Camera, Lidar, due
from .teammap import Message2, TeamMap

SENSE_DT_CAM, SENSE_DT_LIDAR = 0.2, 0.1


class SensorKnowledge:
    def __init__(self, sim):
        self.sim = sim
        c = sim.course
        self.obs = Obstacles(c)
        self.cam = Camera(seed=sim.seed)
        self.lidar = Lidar(seed=sim.seed)
        self.maps = [TeamMap(i, c.width, c.height) for i in range(sim.n)]
        self.tick = 0
        self.scan = [None] * sim.n           # latest lidar points per boat (for display)
        self.dock_seen = [False] * sim.n     # lidar has had the dock structure in range (shared over the radio)
        m3 = c.module("T3")[0]
        self.dock_xy = (m3.x, m3.y)

    # --- sensing --------------------------------------------------------------------------
    def update(self, t):
        sim = self.sim
        if not due(t, sim.dt, 1.0 / SENSE_DT_LIDAR):
            return
        self.tick += 1
        cam_now = due(t, sim.dt, 1.0 / SENSE_DT_CAM)
        light = "green" if sim.ref.light_green(t) else "red"
        for i in range(sim.n):
            x, y, psi, _ = sim.state(i)
            pose = (x, y, psi)
            dets = self.lidar.sense(t, self.tick, i, pose, sim.course, self.obs)
            if cam_now:
                dets = dets + self.cam.sense(t, self.tick, i, pose, sim.course, self.obs, light=light)
            self.maps[i].observe(t, pose, dets, camera_ran=cam_now)
            if not self.dock_seen[i] and math.hypot(x - self.dock_xy[0], y - self.dock_xy[1]) < 18.0:
                self.dock_seen[i] = True        # a 20 m lidar would return the dock structure here

    def make_message(self, i, t, state, claim, arrival):
        x, y, psi, v = state
        done = frozenset({-3}) if self.dock_seen[i] else frozenset()
        return Message2(i, t, x, y, psi, v, claim, arrival, done, self.maps[i].digest(t))

    def on_message(self, r, msg):
        if -3 in getattr(msg, "completed", ()):
            self.dock_seen[r] = True
        if getattr(msg, "digest", None):
            self.maps[r].merge(msg.digest, msg.sender, msg.t_sent)

    # --- queries -------------------------------------------------------------------------------
    def found(self, i, kind):
        M = self.maps[i]
        if kind == "T2":
            return bool(M.objects("yellow", 0.6))
        if kind == "T3":
            return bool(M.banner_bays())
        if kind == "T3dock":
            return self.dock_seen[i]
        if kind == "T4":
            return bool(M.objects("blue", 0.6) or M.objects("light_panel", 0.6))
        return True

    def known_near(self, i, xy, cls, radius):
        best, bd = None, radius
        for f in self.maps[i].objects(cls, 0.5):
            d = math.hypot(f.x - xy[0], f.y - xy[1])
            if d < bd:
                best, bd = (f.x, f.y), d
        return best

    def believes_green(self, i):
        ls = self.maps[i].light_state()
        return bool(ls and ls[0] == "green" and ls[1] > 0.7 and self.sim.t - ls[2] < 2.0)

    def count(self, i, task):
        M = self.maps[i]
        if task == "T2":
            return M.count("yellow", None, 0.5)
        anchors = M.objects("blue", 0.5) + M.objects("light_panel", 0.5)
        if not anchors:
            return 0
        a = anchors[0]
        return M.count("black", lambda x, y: math.hypot(x - a.x, y - a.y) < 22.0, 0.5)

    def bay_choices(self, i):
        """(bay index, P(match day banner) x P(empty)) for every bay whose banner this boat knows about."""
        sim, M = self.sim, self.maps[i]
        m3 = sim.course.module("T3")[0]
        day = (sim.course.day_shape, sim.course.day_color)
        vessels = M.objects("vessel_black", 0.4) + M.objects("vessel_orange", 0.4)
        out = {}
        for (bx, by, probs) in M.banner_bays():
            if not probs:
                continue
            k = min(range(len(m3.bays)), key=lambda j: math.hypot(m3.bays[j].banner[0] - bx, m3.bays[j].banner[1] - by))
            if math.hypot(m3.bays[k].banner[0] - bx, m3.bays[k].banner[1] - by) > 2.0:
                continue
            inner = m3.bays[k].inner
            occupied = any(math.hypot(v.x - inner[0], v.y - inner[1]) < 1.6 for v in vessels)
            p = probs.get(day, 0.0) * (0.05 if occupied else 0.9)
            out[k] = max(out.get(k, 0.0), p)
        # elimination: the course always has at least one EMPTY bay showing the day's banner. If every bay but one
        # has been read and ruled out, the unread bay is the match even though its banner was never seen.
        if not self.sim.cfg.coord_dock_infer:
            return sorted((k, p) for k, p in out.items())
        for k, b in enumerate(m3.bays):                # a vessel seen inside a bay rules it out (and hides its banner)
            if k not in out and any(math.hypot(v.x - b.inner[0], v.y - b.inner[1]) < 1.6 for v in vessels):
                out[k] = 0.0
        unread = [k for k in range(len(m3.bays)) if k not in out]
        if len(unread) == 1 and out and max(out.values()) < 0.2:
            k = unread[0]
            inner = m3.bays[k].inner
            if not any(math.hypot(v.x - inner[0], v.y - inner[1]) < 1.6 for v in vessels):
                out[k] = 0.75
                self.inferred = getattr(self, "inferred", set()) | {(i, k)}
        return sorted((k, p) for k, p in out.items())

    def vessels(self, i):
        sim, M = self.sim, self.maps[i]
        t5 = [(m.id, sim.course.obj(m.objects[0])) for m in sim.course.module("T5")]
        out = []
        for f in M.objects(None, 0.5):
            cls = f.best_class()
            if cls not in ("vessel_orange", "vessel_black"):
                continue
            mk = f.mark()
            if not mk or mk[1] < 0.6:
                continue
            near = [(mid, o) for mid, o in t5 if math.hypot(o.x - f.x, o.y - f.y) < 2.0]
            if not near:
                continue
            mid, o = near[0]
            action = "spray" if mk[0] == "triangle" else "ball"
            out.append((mid, (o.x, o.y), action, min(1.0, f.confidence() * mk[1] / 0.8)))
        return out

    def sector(self, i):
        """Recon sector for boat i: the basin split into n vertical stripes (fixed at launch, known to every boat)."""
        W, n = self.sim.course.width, self.sim.n
        return (i * W / n, (i + 1) * W / n)

    def frontier(self, i):
        """Unexplored tiles, nearest first; each tagged with whether it lies in this boat's own recon sector.
        Own-sector tiles are worth more early, so the fleet fans out instead of exploring the same water."""
        x, y = self.sim.pos(i)
        lo, hi = self.sector(i)
        tiles = self.maps[i].unexplored_tiles()
        def key(p):
            d = math.hypot(p[0] - x, p[1] - y)
            return d - (25.0 if (lo <= p[0] < hi and self.sim.cfg.coord_sectors) else 0.0)
        tiles.sort(key=key)
        # thin out: keep tiles at least 10 m apart
        keep = []
        for p in tiles:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 100 for q in keep):
                keep.append(p)
            if len(keep) >= 6:
                break
        return [(p[0], p[1], lo <= p[0] < hi) for p in keep]

    def tracks_for_display(self, i):
        out = []
        for f in self.maps[i].fused(0.4):
            cls = f.best_class()
            if cls:
                out.append((f.x, f.y, cls, f.confidence(), f.source == "own"))
        return out
