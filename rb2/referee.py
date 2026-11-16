"""Truth-side scoring for course v2, following RoboBoat 2025 Team Handbook section 4.2 (Autonomy Challenge Scoring).

Official components (points):
  T1 Navigation channel   both gates 200
  T2 Follow the path      25 x gates - 25 x strikes (0-250); all gates in one sequence 250; correct yellow count 150
  T3 Docking              enter any unoccupied bay 100 (once); correct bay, forward, first attempt 300
  T4 Speed challenge      station-keep + start at green 250; gate in 50; circle blue 50; exit 50;
                          -25 x strikes; completion time 250 - T (0-250, T in seconds from green)
  T5 Delivery (tiered)    k-th vessel served, per type: 75 / 125 / 200; on target +25 each
  T6 Return home          100; bonus 2 x (T1-T5 total) if all tasks were attempted and the fleet returned home
  Time multiplier on T1-T5: 1.5x -> 1.0x with time remaining (handbook: 1200 s -> 0 s; here scaled to the episode).
Interpretations (documented): for a multi-boat entry each task scores once for the team and the
delivery tiers escalate with the number of vessels of that type the fleet has served; strikes = contacts with
task buoys/vessels during that task; docking is forward only (no reverse docking in the model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .course import Course


def crossed(p0, p1, a, b, forward=None):
    """Did the boat move from p0 to p1 across segment a-b (strictly between endpoints)? If `forward` (unit vector)
    is given, require the motion to be along it."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    s0 = ex * (p0[1] - a[1]) - ey * (p0[0] - a[0])
    s1 = ex * (p1[1] - a[1]) - ey * (p1[0] - a[0])
    if s0 == 0 or (s0 > 0) == (s1 > 0):
        return False
    t = s0 / (s0 - s1)
    x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
    L2 = ex * ex + ey * ey
    u = ((x - a[0]) * ex + (y - a[1]) * ey) / L2
    if not (0.0 < u < 1.0):
        return False
    if forward is not None:
        return (p1[0] - p0[0]) * forward[0] + (p1[1] - p0[1]) * forward[1] > 0
    return True


@dataclass
class BoatProgress:
    t1_gates: int = 0
    t1_done: bool = False
    t2_gate: int = 0
    t4_state: str = "idle"           # idle | holding | started | circling | returning | done | false_start
    t4_t0: float = 0.0
    t4_angle: float = 0.0
    t4_last_ang: Optional[float] = None
    dwell: Dict[str, float] = field(default_factory=dict)
    balls: int = 3
    home: bool = False
    report_t2: Optional[int] = None
    report_t4: Optional[int] = None


PTS = dict(T1=200, T2_gate=25, T2_seq=250, T2_count=150, T3_any=100, T3_correct=300, T4_start=250, T4_leg=50,
           T4_time=250, strike=25, T5_tiers=(75, 125, 200), T5_target=25, T6=100)


@dataclass
class ScoreEvent:
    t: float
    boat: int
    task: str
    points: float
    note: str = ""
    x: float = 0.0
    y: float = 0.0


class Referee:
    def __init__(self, course: Course, n_boats: int, balls: int = 3, episode_s: float = 300.0):
        self.c = course
        self.episode_s = episode_s
        self.t5_served = {"vessel_orange": 0, "vessel_black": 0}
        self.strikes = {}                          # (boat, task) -> count
        self.prog = [BoatProgress(balls=balls) for _ in range(n_boats)]
        self.events: List[ScoreEvent] = []
        self.done: Set[str] = set()              # module-level completion keys ("T1", "T2", "T3", "T4", "T5:<id>")
        self.t3_attempted = False
        self.light_green_at = None
        m4 = course.module("T4")
        self.m4 = m4[0] if m4 else None
        if self.m4:
            self.light_green_at = self.m4.data["light_delay"]   # seconds after the boat starts holding (see below)
        self.hold_started: Optional[float] = None
        self.pending = []                         # tasks finished, waiting for the boat's count report

    # --- helpers ------------------------------------------------------------------------
    def gate(self, m, k):
        r, g = m.gates[k]
        return (self.c.obj(r).x, self.c.obj(r).y), (self.c.obj(g).x, self.c.obj(g).y)

    def multiplier(self, t):
        return 1.0 + 0.5 * max(0.0, (self.episode_s - t) / self.episode_s)

    def award(self, t, i, task, pts, note="", x=0.0, y=0.0, mult=True):
        if mult and task in ("T1", "T2", "T3", "T4", "T5") and pts > 0:
            pts = round(pts * self.multiplier(t), 1)
        self.events.append(ScoreEvent(t, i, task, pts, note, x, y))

    def strike(self, i, task):
        self.strikes[(i, task)] = self.strikes.get((i, task), 0) + 1

    def finalize(self, t):
        """Return-home bonus: 2 x (T1-T5 points) if every task was attempted and the whole fleet is home."""
        attempted = {"T1", "T2", "T4"} <= self.done and self.t3_attempted and any(k.startswith("T5") for k in self.done)
        if attempted and all(p.home for p in self.prog):
            base = sum(e.points for e in self.events if e.task in ("T1", "T2", "T3", "T4", "T5"))
            self.award(t, -1, "T6", 2 * base, "bonus: all tasks attempted and fleet home", mult=False)

    def light_green(self, t) -> bool:
        return self.hold_started is not None and t - self.hold_started >= self.light_green_at

    @property
    def points(self):
        return sum(e.points for e in self.events)

    def settle(self, t, force=False):
        """Award finished counting tasks once the boat has reported (or 5 s later with no report)."""
        keep = []
        for item in self.pending:
            task, i, t_end, p = item[0], item[1], item[2], item[3]
            P = self.prog[i]
            rep = P.report_t2 if task == "T2" else P.report_t4
            if rep is None and t - t_end < 15.0 and not force:
                keep.append(item); continue
            if task == "T2":
                m = self.c.module("T2")[0]
                ok = rep == m.data["n_yellow"]
                G, Sk = len(m.gates), self.strikes.get((i, "T2"), 0)
                pts = max(0, min(250, PTS["T2_gate"] * G - PTS["strike"] * Sk)) + PTS["T2_seq"] + (PTS["T2_count"] if ok else 0)
                self.award(t, i, "T2", pts, f"{G} gates, {Sk} strikes, count {'right' if ok else 'wrong'} ({rep})", *p)
            else:
                el = item[4]
                Sk = self.strikes.get((i, "T4"), 0)
                pts = PTS["T4_start"] + 3 * PTS["T4_leg"] - PTS["strike"] * Sk + max(0.0, min(250.0, PTS["T4_time"] - el))
                self.award(t, i, "T4", round(max(0.0, pts), 1), f"run {el:.0f} s, {Sk} strikes", *p)
        self.pending = keep

    # --- per-step update ----------------------------------------------------------------
    def update(self, t, dt, i, p0, p1, psi, v):
        c, P = self.c, self.prog[i]
        # T1
        m1 = c.module("T1")[0]
        if not P.t1_done:
            a, b = self.gate(m1, P.t1_gates)
            fwd = (math.cos(m1.heading), math.sin(m1.heading))
            if crossed(p0, p1, a, b, fwd):
                P.t1_gates += 1
                if P.t1_gates == len(m1.gates):
                    P.t1_done = True
                    if "T1" not in self.done:
                        self.done.add("T1"); self.award(t, i, "T1", PTS["T1"], "both gates", *p1)
            return                                   # nothing else counts before T1
        # T2: gates in order
        m2 = c.module("T2")[0]
        if "T2" not in self.done and P.t2_gate < len(m2.gates):
            a, b = self.gate(m2, P.t2_gate)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            # forward = from the previous gate toward this one along the path
            fx, fy = _path_dir(m2.path, mid)
            if crossed(p0, p1, a, b, (fx, fy)):
                P.t2_gate += 1
                if P.t2_gate == len(m2.gates):
                    self.done.add("T2")
                    self.pending.append(("T2", i, t, p1))
        # T3: dwell in a bay
        m3 = c.module("T3")[0]
        if not self.t3_attempted:
            for k, bay in enumerate(m3.bays):
                if math.hypot(p1[0] - bay.inner[0], p1[1] - bay.inner[1]) < 1.0 and abs(v) < 0.35:
                    P.dwell[f"T3:{k}"] = P.dwell.get(f"T3:{k}", 0.0) + dt
                    if P.dwell[f"T3:{k}"] >= 2.0:
                        self.t3_attempted = True
                        ok = (bay.shape, bay.color) == (c.day_shape, c.day_color) and not bay.occupied
                        pts = (0 if bay.occupied else PTS["T3_any"]) + (PTS["T3_correct"] if ok else 0)
                        if ok:
                            self.done.add("T3")
                        self.award(t, i, "T3", pts, "correct bay, first attempt" if ok else
                                   ("occupied bay" if bay.occupied else "unoccupied but wrong bay"), *p1)
                else:
                    P.dwell[f"T3:{k}"] = 0.0
        # T4: speed challenge state machine
        if self.m4 and "T4" not in self.done:
            self._t4(t, i, P, p0, p1, v)
        # T5: deliveries
        for m in c.module("T5"):
            key = f"T5:{m.id}"
            if key in self.done:
                continue
            o = c.obj(m.objects[0])
            d = math.hypot(p1[0] - o.x, p1[1] - o.y)
            if o.cls == "vessel_orange":
                bearing = math.atan2(o.y - p1[1], o.x - p1[0])
                aim = abs((bearing - psi + math.pi) % (2 * math.pi) - math.pi) < math.radians(25)
                ok = 1.5 <= d <= 4.5 and aim and abs(v) < 0.5
                need = 3.0
            else:
                ok = d <= 2.8 and abs(v) < 0.4 and P.balls > 0      # ball launcher reach (stylized)
                need = 1.0
            P.dwell[key] = P.dwell.get(key, 0.0) + dt if ok else 0.0
            if P.dwell[key] >= need:
                self.done.add(key)
                if o.cls == "vessel_black":
                    P.balls -= 1
                k = self.t5_served[o.cls]
                self.t5_served[o.cls] += 1
                pts = PTS["T5_tiers"][min(k, 2)] + PTS["T5_target"]
                self.award(t, i, "T5", pts, ("water" if o.cls == "vessel_orange" else "ball") + f" delivery #{k + 1}", o.x, o.y)
        # T6
        m6 = c.module("T6")[0]
        if not P.home:
            a, b = (c.obj(m6.gates[0][0]).x, c.obj(m6.gates[0][0]).y), (c.obj(m6.gates[0][1]).x, c.obj(m6.gates[0][1]).y)
            if crossed(p0, p1, a, b):
                P.home = True
                self.award(t, i, "T6", PTS["T6"] / len(self.prog), "home", *p1)

    def _t4(self, t, i, P, p0, p1, v):
        m = self.m4
        hx, hy = m.path[0]
        a, b = self.gate(m, 0)
        ux, uy = math.cos(m.heading), math.sin(m.heading)
        blue = next(self.c.obj(k) for k in m.objects if self.c.obj(k).cls == "blue")
        if P.t4_state == "idle":
            if math.hypot(p1[0] - hx, p1[1] - hy) < 2.0 and abs(v) < 0.5:
                P.t4_state = "holding"
                if self.hold_started is None:
                    self.hold_started = t
        elif P.t4_state == "holding":
            if crossed(p0, p1, a, b, (ux, uy)):
                if self.light_green(t):
                    P.t4_state, P.t4_t0 = "circling", self.hold_started + self.light_green_at
                    P.t4_angle, P.t4_last_ang = 0.0, None
                else:
                    P.t4_state = "false_start"
                    self.done.add("T4"); self.award(t, i, "T4", 0, "false start (left before green)", *p1)
        elif P.t4_state == "circling":
            d = math.hypot(p1[0] - blue.x, p1[1] - blue.y)
            if d < 6.0:
                ang = math.atan2(p1[1] - blue.y, p1[0] - blue.x)
                if P.t4_last_ang is not None:
                    P.t4_angle += abs((ang - P.t4_last_ang + math.pi) % (2 * math.pi) - math.pi)
                P.t4_last_ang = ang
            if P.t4_angle >= math.radians(330) and crossed(p0, p1, a, b, (-ux, -uy)):
                el = t - P.t4_t0
                self.done.add("T4")
                P.t4_state = "done"
                self.pending.append(("T4", i, t, p1, el))


def _path_dir(path, near):
    best, k = 1e9, 0
    for j, (x, y) in enumerate(path[:-1]):
        d = (x - near[0]) ** 2 + (y - near[1]) ** 2
        if d < best:
            best, k = d, j
    (x0, y0), (x1, y1) = path[k], path[k + 1]
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    return (x1 - x0) / L, (y1 - y0) / L
