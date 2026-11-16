"""Per-task behaviour state machines (what a boat DOES once it has claimed an opportunity).

Stylization (documented): perception decides what a boat knows and chooses (task found, which bay matches, light
state, vessel class/mark, buoy counts); once a task is found, precise manoeuvre geometry (gate centres, the circle
radius, bay mouth) comes from the module. Wrong detections therefore cause wrong decisions (wrong bay, false start,
miscount, wrong delivery action) without modelling a full mapping stack.

Phases:  route(target, stop) | track(wps, stop, tail) | hold(until) | dwell(seconds, aim) | report(task)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.navigation import GuidanceState, heading_controller, los_command, plan_route


@dataclass
class Phase:
    kind: str
    target: Tuple[float, float] = (0.0, 0.0)
    wps: List[Tuple[float, float]] = field(default_factory=list)
    stop: bool = True
    seconds: float = 0.0
    aim: Optional[Tuple[float, float]] = None
    until: Optional[Callable] = None
    task: str = ""
    tail: int = 0
    vmax: float = 1.1              # speed cap while tracking (precise work is slower, the timed speed run is fast)


def _unit(a, b):
    L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    return (b[0] - a[0]) / L, (b[1] - a[1]) / L


def phases_for(sim, i, opp) -> List[Phase]:
    """Build the behaviour for boat i executing opportunity `opp` (an rb2.opportunities.Opp)."""
    c = sim.course
    kind = opp.kind
    if kind == "T1":
        m = c.module("T1")[0]
        mids = [sim.gate_mid(m, k) for k in range(len(m.gates))]
        ux, uy = math.cos(m.heading), math.sin(m.heading)
        pre = (mids[0][0] - 3.0 * ux, mids[0][1] - 3.0 * uy)
        post = (mids[-1][0] + 4.0 * ux, mids[-1][1] + 4.0 * uy)
        return [Phase("route", target=pre, stop=False), Phase("track", wps=[pre] + mids + [post], stop=False, tail=2, vmax=1.7)]
    if kind == "T2":
        m = c.module("T2")[0]
        mids = [sim.gate_mid(m, k) for k in range(len(m.gates))]
        # pass each yellow buoy on the side away from it (yellows sit between gates)
        wps = []
        for k in range(len(mids)):
            wps.append(mids[k])
            if k + 1 < len(mids):
                mx, my = (mids[k][0] + mids[k + 1][0]) / 2, (mids[k][1] + mids[k + 1][1]) / 2
                y = sim.known_near(i, (mx, my), "yellow", 1.4)
                if y is not None:
                    tx, ty = _unit(mids[k], mids[k + 1])
                    side = (y[0] - mx) * (-ty) + (y[1] - my) * tx
                    off = -0.75 if side > 0 else 0.75
                    wps.append((mx - ty * off, my + tx * off))
        ux, uy = _unit(mids[0], mids[1])
        pre = (mids[0][0] - 3.0 * ux, mids[0][1] - 3.0 * uy)
        vx, vy = _unit(mids[-2], mids[-1])
        post = (mids[-1][0] + 3.0 * vx, mids[-1][1] + 3.0 * vy)
        return [Phase("route", target=pre, stop=False), Phase("track", wps=[pre] + wps + [post], stop=False, tail=len(wps) + 1, vmax=1.35),
                Phase("report", task="T2")]
    if kind == "T3":
        bay = c.module("T3")[0].bays[opp.data["bay"]]
        return [Phase("route", target=bay.entry, stop=True), Phase("track", wps=[bay.entry, bay.inner], stop=True, tail=1, vmax=0.8),
                Phase("dwell", seconds=2.6, aim=bay.inner)]
    if kind == "T4":
        m = c.module("T4")[0]
        hold = m.path[0]
        loop = list(m.path[1:])
        return [Phase("route", target=hold, stop=True), Phase("hold", target=hold, until=lambda s, i=i: s.believes_green(i)),
                Phase("track", wps=[hold] + loop + [hold], stop=True, tail=len(loop) + 1, vmax=1.8), Phase("report", task="T4")]
    if kind == "T5":
        m = c.modules[opp.data["module"]]
        o = c.obj(m.objects[0])
        nx, ny = -math.sin(o.heading), math.cos(o.heading)          # beam direction (side of the hull)
        me = sim.pos(i)
        side = 1 if (me[0] - o.x) * nx + (me[1] - o.y) * ny >= 0 else -1
        if opp.data["action"] == "spray":
            st = (o.x + side * 3.0 * nx, o.y + side * 3.0 * ny)
            return [Phase("route", target=st, stop=True), Phase("dwell", seconds=5.0, aim=(o.x, o.y))]
        st = (o.x + side * 2.1 * nx, o.y + side * 2.1 * ny)
        return [Phase("route", target=st, stop=True), Phase("dwell", seconds=2.0, aim=(o.x, o.y))]
    if kind == "T6":
        m = c.module("T6")[0]
        mid = sim.gate_mid(m, 0)
        ux, uy = math.cos(m.heading), math.sin(m.heading)          # direction of travel through the home gate
        vx, vy = -uy, ux
        pre = (mid[0] - 3.0 * ux, mid[1] - 3.0 * uy)
        post = (mid[0] + 2.2 * ux, mid[1] + 2.2 * uy)
        # berths line up beside the gate; the first boat home takes the FARTHEST berth so later arrivals never have
        # to pass a parked boat
        k = sum(1 for j in range(sim.n) if j != i and sim.ref.prog[j].home)
        off = 3.0 + 3.5 * (sim.n - 1 - k)
        side = 1.0
        if sim.chart.clearance_at(post[0] + off * vx, post[1] + off * vy) < 1.2:
            side = -1.0
        berth = (post[0] + side * off * vx, post[1] + side * off * vy)
        return [Phase("route", target=pre, stop=False),
                Phase("track", wps=[pre, mid, post], stop=False, tail=2, vmax=1.5),
                Phase("track", wps=[post, berth], stop=True, tail=1, vmax=1.2)]
    if kind == "T3look":
        # scan the dock: pause square-on in front of each bay (nearest end first) so every banner is read head-on
        m = c.module("T3")[0]
        me = sim.pos(i)
        bays = sorted(m.bays, key=lambda b: math.hypot(b.entry[0] - me[0], b.entry[1] - me[1]))
        if math.hypot(bays[-1].entry[0] - me[0], bays[-1].entry[1] - me[1]) < math.hypot(bays[0].entry[0] - me[0], bays[0].entry[1] - me[1]):
            bays = bays[::-1]
        ends = [bays[0], bays[-1]]
        first = min(ends, key=lambda b: math.hypot(b.entry[0] - me[0], b.entry[1] - me[1]))
        order = sorted(m.bays, key=lambda b: math.hypot(b.entry[0] - first.entry[0], b.entry[1] - first.entry[1]))
        out = []
        for b in order:
            out += [Phase("route", target=b.entry, stop=True), Phase("dwell", seconds=1.6, aim=b.banner)]
        return out
    if kind == "explore":
        return [Phase("route", target=opp.xy, stop=False)]
    raise KeyError(kind)


class Executor:
    """Runs a list of phases for one boat. step() returns (psi_des, v_des, finished)."""

    def __init__(self, sim, i, opp):
        self.sim, self.i, self.opp = sim, i, opp
        self.phases = phases_for(sim, i, opp)
        self.k = 0
        self.g: Optional[GuidanceState] = None
        self.t0 = None
        self.failed = False

    def remaining_m(self):
        """Length of the rest of the plan (for a live ETA in the interface)."""
        x, y = self.sim.pos(self.i)
        pts = [(x, y)]
        if self.g is not None:
            pts += list(self.g.wps[self.g.seg + 1:])
        for ph in self.phases[self.k + 1:]:
            if ph.kind == "route":
                pts.append(ph.target)
            elif ph.kind == "track":
                pts += list(ph.wps[1:])
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    def current(self) -> Optional[Phase]:
        return self.phases[self.k] if self.k < len(self.phases) else None

    def _start(self, ph):
        sim, i = self.sim, self.i
        x, y = sim.pos(i)
        self.t0 = sim.t
        if ph.kind == "route":
            wps, _ = plan_route(sim.chart, (x, y), [ph.target])
            if wps is None:
                self.failed = True
                return
            self.g = GuidanceState(wps, 0, ph.stop)
        elif ph.kind == "track":
            wps = [(x, y)] + list(ph.wps[1:]) if math.hypot(ph.wps[0][0] - x, ph.wps[0][1] - y) < 4 else [(x, y)] + list(ph.wps)
            self.g = GuidanceState(wps, 0, ph.stop, tail_segments=ph.tail, tail_lookahead=2.2)
        else:
            self.g = None

    def step(self):
        sim, i = self.sim, self.i
        cfg = sim.dcfg
        x, y, psi, v = sim.state(i)
        if self.opp.kind == "T3look" and self.k >= 2 and hasattr(sim.know, "bay_choices"):
            if any(p > 0.7 for _, p in sim.know.bay_choices(i)):
                return psi, 0.0, True                    # the matching bay is known: stop scanning
        while True:
            ph = self.current()
            if ph is None or self.failed:
                return psi, 0.0, True
            if self.t0 is None:
                self._start(ph)
                if self.failed:
                    return psi, 0.0, True
            if ph.kind in ("route", "track"):
                psi_d, vd, d_end = los_command(self.g, x, y, psi, cfg)
                if ph.kind == "track":
                    vd = min(vd, ph.vmax)
                end_ok = d_end < (1.0 if ph.stop else 1.6) and (not ph.stop or abs(v) < 0.4)
                if end_ok or sim.t - self.t0 > 120:
                    self.k += 1; self.t0 = None
                    continue
                return psi_d, vd, False
            if ph.kind == "hold":
                if sim.t - self.t0 > 45.0:            # light never seen green: give up rather than wait forever
                    self.failed = True
                    return psi, 0.0, True
                if ph.until(sim):
                    self.k += 1; self.t0 = None
                    continue
                psi_h, v_h = sim.station_keep(i, ph.target, face=sim.light_xy())
                return psi_h, v_h, False
            if ph.kind == "dwell":
                aim = ph.aim or (x + math.cos(psi), y + math.sin(psi))
                want = math.atan2(aim[1] - y, aim[0] - x)
                err = abs((want - psi + math.pi) % (2 * math.pi) - math.pi)
                if ph.aim is not None and err > 0.3 and sim.t - self.t0 < 6.0 and not getattr(self, "_aligned", False):
                    self._dwell_start = sim.t                 # the dwell clock starts once the boat faces its aim
                else:
                    self._aligned = True
                start = getattr(self, "_dwell_start", self.t0)
                if sim.t - max(start, self.t0) >= ph.seconds:
                    self.k += 1; self.t0 = None; self._aligned = False; self._dwell_start = -1e9
                    continue
                return want, 0.0, False
            if ph.kind == "report":
                sim.report_count(i, ph.task)
                self.k += 1; self.t0 = None
                continue
            raise KeyError(ph.kind)
