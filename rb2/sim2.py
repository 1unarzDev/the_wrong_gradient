"""Course v2 simulation: RoboBoat tasks + knowledge (oracle or sensors/team map) + reused v1 physics, planner,
guidance and policies. The v1 engine in src/ is imported, never modified."""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.comms import CommTape, Message, Network
from src.navigation import control_eta, graph_eta, heading_controller, penalty_tables, wrap
from src.physics import allocate_thrust, rk4_step
from src.policies import BoatObservation, CentralObservation, Decision, TaskView, make_policy
from .policies2 import make_policy2

from .config2 import V2Config
from .central import masked_decide_all
from .executors import Executor
from .generate import GenConfig, compile_chart, generate_course
from .referee import Referee
from .rng2 import stream


@dataclass
class Opp:
    key: str                       # canonical id shared across boats, e.g. "T5:12", "T3:bay1", "explore:3,4"
    kind: str
    xy: Tuple[float, float]
    value: float
    service_s: float = 0.0
    data: Dict = field(default_factory=dict)

    @property
    def id(self) -> int:
        return zlib.crc32(self.key.encode()) & 0x7FFFFFFF


class OracleKnowledge:
    """Full information (everything on the course is known). Used for tests and as a reference."""

    def __init__(self, sim):
        self.sim = sim

    def update(self, t):
        pass

    def found(self, i, kind):
        return True

    def known_near(self, i, xy, cls, radius):
        best = None
        for o in self.sim.course.objects:
            if o.cls == cls and math.hypot(o.x - xy[0], o.y - xy[1]) < radius:
                best = (o.x, o.y)
        return best

    def believes_green(self, i):
        return self.sim.ref.light_green(self.sim.t)

    def count(self, i, task):
        m = self.sim.course.module(task)[0]
        return m.data["n_yellow"] if task == "T2" else m.data["n_black"]

    def bay_choices(self, i):
        m = self.sim.course.module("T3")[0]
        c = self.sim.course
        return [(k, 1.0 if (b.shape, b.color) == (c.day_shape, c.day_color) and not b.occupied else 0.0)
                for k, b in enumerate(m.bays)]

    def vessels(self, i):
        out = []
        for m in self.sim.course.module("T5"):
            o = self.sim.course.obj(m.objects[0])
            out.append((m.id, (o.x, o.y), "spray" if o.cls == "vessel_orange" else "ball", 1.0))
        return out

    def frontier(self, i):
        return []


class Simulation2:
    def __init__(self, cfg: V2Config, seed: int, knowledge: str = None):
        self.cfg, self.seed = cfg, seed
        # the course is the standard RoboBoat course for fleets of up to 4 boats (it never shrinks below that);
        # larger fleets get a proportionally larger basin
        self.gc = GenConfig(n_boats=max(4, cfg.n_boats), difficulty=cfg.difficulty, variant=cfg.variant)
        self.course = generate_course(seed, self.gc)
        self.course.shortcut_open = cfg.shortcut_open
        self.chart = compile_chart(self.course, self.gc)
        self.dcfg = cfg.sim_config()
        self.tables = penalty_tables(self.dcfg)
        self.policy = make_policy2(cfg.policy) or make_policy(cfg.policy)
        n = cfg.n_boats
        self.n = n
        self.dt = self.dcfg.dt
        self.n_steps = int(round(cfg.episode_s / self.dt))
        rng = stream(seed, "disturb")
        a = math.exp(-self.dt / self.dcfg.disturb_tau)
        eps = rng.standard_normal((self.n_steps + 1, 2, n))
        sig = np.array([self.dcfg.disturb_sigma_v, self.dcfg.disturb_sigma_r])[:, None]
        self.dist = np.empty_like(eps); z = eps[0] * sig
        for k in range(self.n_steps + 1):
            z = a * z + math.sqrt(1 - a * a) * sig * eps[k]; self.dist[k] = z
        sx, sy, sh = self.course.start
        xs = sx + (np.arange(n) - (n - 1) / 2) * (3.2 if n >= 3 else 5.0)   # two boats start wider apart (3.2 m deadlocked at the channel)
        self.S = np.vstack([xs, np.full(n, sy + 1.5), np.full(n, sh), np.zeros(n), np.zeros(n)]).astype(float)
        self.TL = np.zeros(n); self.TR = np.zeros(n)
        self.ref = Referee(self.course, n, cfg.balls_per_boat, cfg.episode_s)
        self.net = Network(CommTape(seed, self.dcfg))
        self.beliefs: List[Dict[int, Message]] = [dict() for _ in range(n)]
        self.claim: List[Optional[Opp]] = [None] * n
        self.claim_arrival = [math.inf] * n
        self.exec: List[Optional[Executor]] = [None] * n
        self.next_decide = [0.3 * i for i in range(n)]
        self.t, self.k = 0.0, 0
        self.energy = np.zeros(n)
        self.contacts = 0
        self.contact_set = set()
        self.events: List[Tuple[float, str, int, str]] = []
        self.cpu = []
        mode = knowledge or cfg.perception
        if mode == "oracle":
            self.know = OracleKnowledge(self)
        else:
            from .knowledge import SensorKnowledge
            self.know = SensorKnowledge(self)
        self.done_flag = False
        self.yielded: List[Dict[str, float]] = [dict() for _ in range(n)]   # boat -> {task key: until t}
        self.fails: Dict[str, int] = {}          # task key -> failed attempts (a task that fails twice is dropped)
        self.leader_next = 0.0                   # P5: next planning time of the leader boat
        self.plan: Optional[Tuple[float, Dict[int, Optional[Opp]]]] = None
        self.plan_applied = [-1.0] * n
        self.log: List[Tuple[float, int, str]] = []   # coordination log for the interface: (t, boat, text)
        self.reason: List[Optional[dict]] = [None] * n   # why each boat holds its current claim
        self.found_t: Dict[str, Tuple[float, int]] = {}  # task -> (time first found by any boat, boat)
        self.idle_s = np.zeros(n)

    # --- queries used by executors / knowledge -------------------------------------------
    def pos(self, i):
        return float(self.S[0, i]), float(self.S[1, i])

    def state(self, i):
        return tuple(float(v) for v in self.S[:4, i])

    def gate_mid(self, m, k):
        r, g = m.gates[k]
        a, b = self.course.obj(r), self.course.obj(g)
        return ((a.x + b.x) / 2, (a.y + b.y) / 2)

    def known_near(self, i, xy, cls, radius):
        return self.know.known_near(i, xy, cls, radius)

    def believes_green(self, i):
        return self.know.believes_green(i)

    def report_count(self, i, task):
        n = self.know.count(i, task)
        P = self.ref.prog[i]
        if task == "T2":
            P.report_t2 = n
        else:
            P.report_t4 = n

    def station_keep(self, i, target, face=None):
        """Hold position at `target`: creep back if pushed off (> 1 m), otherwise face `face` (e.g. the light)."""
        x, y = self.pos(i)
        d = math.hypot(target[0] - x, target[1] - y)
        if d > 1.0 or face is None:
            return math.atan2(target[1] - y, target[0] - x), (min(0.45, 0.3 * d) if d > 1.0 else 0.0)
        return math.atan2(face[1] - y, face[0] - x), 0.0

    def light_xy(self):
        m = self.course.module("T4")[0]
        for oid in m.objects:
            o = self.course.obj(oid)
            if o.cls == "light_panel":
                return (o.x, o.y)
        return None

    # --- opportunities -----------------------------------------------------------------------
    def opportunities(self, i) -> List[Opp]:
        c, P, R, K = self.course, self.ref.prog[i], self.ref, self.know
        if not P.t1_done:
            m = c.module("T1")[0]
            # T1 is every boat's gateway to scoring: one opportunity per boat so a planner never serialises the fleet
            return [Opp(f"T1#{i}", "T1", self.gate_mid(m, 0), 200.0)]
        if P.home:
            return []                                  # a boat that has returned home is finished
        out = []
        late = self.late(i)
        if "T2" not in R.done and K.found(i, "T2"):
            m = c.module("T2")[0]
            out.append(Opp("T2", "T2", self.gate_mid(m, 0), 500.0, service_s=25.0))
        if not R.t3_attempted and hasattr(K, "dock_seen") and K.found(i, "T3dock"):
            choices = K.bay_choices(i)
            if (not choices or max(p for _, p in choices) <= 0.2) and self.fails.get("T3:look", 0) < 3:
                m3 = c.module("T3")[0]
                look = (m3.x + 6.5 * math.cos(m3.heading), m3.y + 6.5 * math.sin(m3.heading))
                out.append(Opp("T3:look", "T3look", look, 120.0, service_s=3.0))
        if not R.t3_attempted and K.found(i, "T3"):
            best = max(K.bay_choices(i), key=lambda kv: kv[1], default=None)
            if best and best[1] > 0.2:
                bay = c.module("T3")[0].bays[best[0]]
                out.append(Opp(f"T3:bay{best[0]}", "T3", bay.entry, 100.0 + 300.0 * best[1], service_s=6.0, data={"bay": best[0]}))
        if "T4" not in R.done and K.found(i, "T4") and R.prog[i].t4_state in ("idle", "holding"):
            m = c.module("T4")[0]
            out.append(Opp("T4", "T4", m.path[0], 650.0, service_s=30.0))
        for mid, xy, action, conf in K.vessels(i):
            if f"T5:{mid}" in R.done or (action == "ball" and P.balls <= 0) or self.fails.get(f"T5:{mid}", 0) >= 2:
                continue
            out.append(Opp(f"T5:{mid}", "T5", xy, 150.0 * conf, service_s=4.0, data={"module": mid, "action": action}))
        for (tx, ty, own_sector) in K.frontier(i)[:6]:
            val = (120.0 if own_sector else 70.0) if self.cfg.coord_sectors else 95.0
            out.append(Opp(f"explore:{int(tx)},{int(ty)}", "explore", (tx, ty), val))
        if not P.home and (late or not [o for o in out if o.kind != "explore"]):
            m = c.module("T6")[0]
            out.append(Opp(f"T6#{i}", "T6", self.gate_mid(m, 0), 25.0 if not late else 2000.0))
        # bonus awareness: the return bonus doubles T1-T5 only if every task type was attempted, so a first attempt of
        # a type nobody has tried yet is worth more than its own points
        tried = self.tried_types()
        for o in out:
            if self.cfg.coord_bonus and o.kind in ("T2", "T3", "T4", "T5") and o.kind not in tried:
                o.value += 300.0
        Y = self.yielded[i]
        out = [o for o in out if Y.get(o.key, -1.0) < self.t]
        if late:
            out = [o for o in out if o.kind == "T6"] or out
        return out

    FOUND_TEXT = {"T2": "the path (yellow buoys)", "T3dock": "the dock", "T3": "a dock banner",
                  "T4": "the speed challenge (blue buoy / light)", "T5": "a rescue vessel and its mark"}

    def _track_found(self):
        K = self.know
        for i in range(self.n):
            for kind in ("T2", "T3dock", "T3", "T4"):
                if kind not in self.found_t and K.found(i, kind):
                    self.found_t[kind] = (self.t, i)
                    self.log.append((self.t, i, "finds " + self.FOUND_TEXT[kind]))
                    self.events.append((self.t, "found", i, kind))
            for mid, xy, action, conf in K.vessels(i):
                key = f"T5:{mid}"
                if key not in self.found_t:
                    self.found_t[key] = (self.t, i)
                    self.log.append((self.t, i, f"finds a rescue vessel ({'water' if action == 'spray' else 'ball'})"))
                    self.events.append((self.t, "found", i, key))

    def _resolve_conflicts(self):
        """Local deconfliction, sensed directly (lidar range): two boats working the SAME single-boat task.
        The boat farther from the task area yields once it is within 15 m and can see the other boat there; it
        drops the claim and avoids that task for 40 s. Policies still decide what to claim; this only stops a
        duplicate claim from turning into a queue at the task."""
        if not self.cfg.coord_deconflict:
            return
        for i in range(self.n):
            ci = self.claim[i]
            if ci is None or ci.kind in ("explore", "T1", "T6"):
                continue
            xi, yi = self.pos(i)
            di = math.hypot(xi - ci.xy[0], yi - ci.xy[1])
            if di > 15.0:
                continue
            for j in range(self.n):
                cj = self.claim[j]
                if j == i or cj is None or cj.key != ci.key:
                    continue
                xj, yj = self.pos(j)
                dj = math.hypot(xj - cj.xy[0], yj - cj.xy[1])
                if dj < di - 0.5 or (abs(dj - di) <= 0.5 and j < i):
                    self.yielded[i][ci.key] = self.t + 40.0
                    self.log.append((self.t, i, f"yields {ci.key} to boat {j + 1}"))
                    self.events.append((self.t, "yield", i, ci.key))
                    self.exec[i] = None; self.claim[i] = None; self.next_decide[i] = self.t
                    break

    def shared_opps(self, view):
        """Single-boat tasks as seen in boat `view`'s own team map (used by the P5 leader)."""
        c, R, K = self.course, self.ref, self.know
        out = []
        if "T2" not in R.done and K.found(view, "T2"):
            out.append(Opp("T2", "T2", self.gate_mid(c.module("T2")[0], 0), 500.0, service_s=25.0))
        if not R.t3_attempted and hasattr(K, "dock_seen") and K.found(view, "T3dock"):
            choices = K.bay_choices(view)
            if (not choices or max(p for _, p in choices) <= 0.2) and self.fails.get("T3:look", 0) < 3:
                m3 = c.module("T3")[0]
                out.append(Opp("T3:look", "T3look", (m3.x + 6.5 * math.cos(m3.heading), m3.y + 6.5 * math.sin(m3.heading)),
                               120.0, service_s=3.0))
        if not R.t3_attempted and K.found(view, "T3"):
            best = max(K.bay_choices(view), key=lambda kv: kv[1], default=None)
            if best and best[1] > 0.2:
                bay = c.module("T3")[0].bays[best[0]]
                out.append(Opp(f"T3:bay{best[0]}", "T3", bay.entry, 100.0 + 300.0 * best[1], service_s=6.0, data={"bay": best[0]}))
        if "T4" not in R.done and K.found(view, "T4") and all(p.t4_state in ("idle", "holding") for p in R.prog):
            out.append(Opp("T4", "T4", c.module("T4")[0].path[0], 650.0, service_s=30.0))
        for mid, xy, action, conf in K.vessels(view):
            if f"T5:{mid}" in R.done or self.fails.get(f"T5:{mid}", 0) >= 2:
                continue
            out.append(Opp(f"T5:{mid}", "T5", xy, 150.0 * conf, service_s=4.0, data={"module": mid, "action": action}))
        tried = self.tried_types()
        for o in out:
            if self.cfg.coord_bonus and o.kind not in tried:
                o.value += 300.0
        return out

    def _decide_leader(self):
        """P5 on the RoboBoat course: boat 0 runs the same masked assignment on what IT knows (its own team map and
        teammates' last received positions); each assignment reaches a boat only when that boat receives a leader
        message sent after the plan. Boats without a current assignment handle their own channel pass, search and
        return home."""
        cfg, L, lam = self.cfg, 0, self.dcfg.time_cost
        if self.t + 1e-9 >= self.leader_next:
            self.leader_next = self.t + cfg.decide_every
            tasks = self.shared_opps(L)
            boats, states, claims = [], [], []
            for k in range(self.n):
                if k == L:
                    st, cl = self.state(k), (self.claim[k].id if self.claim[k] else None)
                else:
                    m = self.beliefs[L].get(k)
                    if m is None or self.t - m.t_sent > 10.0:
                        continue
                    st, cl = (m.x, m.y, m.psi, m.v), m.claim
                boats.append(k); states.append(st); claims.append(cl)
            plan = {k: None for k in range(self.n)}
            if tasks and boats:
                cobs = CentralObservation(self.t, states, [self.taskview(o) for o in tasks], claims, self.chart, self.dcfg, self.tables)
                ds = masked_decide_all(cobs, [[True] * len(tasks) for _ in boats])
                by_id = {o.id: o for o in tasks}
                for k, d in zip(boats, ds):
                    plan[k] = by_id.get(d.task)
            self.plan = (self.t, plan)
        if self.plan is not None:
            pt, plan = self.plan
            for k in range(self.n):
                if pt <= self.plan_applied[k]:
                    continue
                if k != L:
                    m = self.beliefs[k].get(L)
                    if m is None or m.t_sent < pt:
                        continue
                self.plan_applied[k] = pt
                o, P = plan[k], self.ref.prog[k]
                if o is None or not P.t1_done or P.home or o.key in self.ref.done or self.late(k):
                    continue
                if self.claim[k] is None or self.claim[k].key != o.key:
                    x, y = self.pos(k)
                    self._claim(k, o, math.hypot(o.xy[0] - x, o.xy[1] - y) / 1.5, how="leader")
        for i in range(self.n):                      # local decisions: channel, search, home
            if self.t + 1e-9 < self.next_decide[i]:
                continue
            self.next_decide[i] = self.t + cfg.decide_every
            cl = self.claim[i]
            if self.late(i) and not self.ref.prog[i].home and (cl is None or cl.kind != "T6"):
                pass
            elif self.exec[i] is not None and cl is not None and cl.kind != "explore":
                continue
            opps = [o for o in self.opportunities(i) if o.kind in ("T1", "T6", "explore")]
            if not opps:
                continue
            x, y = self.pos(i)
            o = max(opps, key=lambda q: q.value - lam * math.hypot(q.xy[0] - x, q.xy[1] - y) / 1.5)
            self._claim(i, o, math.hypot(o.xy[0] - x, o.xy[1] - y) / 1.5, how="own")

    def tried_types(self):
        R = self.ref
        t = set()
        if "T2" in R.done: t.add("T2")
        if R.t3_attempted: t.add("T3")
        if "T4" in R.done: t.add("T4")
        if any(k.startswith("T5") for k in R.done): t.add("T5")
        for j in range(self.n):                        # a type already claimed by a teammate counts as covered
            cl = self.claim[j]
            if cl is not None and cl.kind in ("T2", "T3", "T4", "T5"):
                t.add(cl.kind)
        return t

    def home_eta(self, i):
        m = self.course.module("T6")[0]
        gx, gy = self.gate_mid(m, 0)
        x, y = self.pos(i)
        return 1.25 * math.hypot(gx - x, gy - y) / 1.6 + 8.0

    def late(self, i):
        """Time to head home: remaining time barely covers the trip (with a margin)."""
        # queue-aware: the home gate passes one boat every ~7 s, so a boat that will arrive behind k teammates
        # leaves 7k s earlier (rank by ETA among boats not yet home)
        ei = self.home_eta(i)
        ahead = sum(1 for j in range(self.n) if j != i and not self.ref.prog[j].home and self.home_eta(j) < ei)
        if not self.cfg.coord_queue_return:
            ahead = 0
        return self.t + ei + 25.0 + 7.0 * ahead > self.cfg.episode_s

    def taskview(self, o: Opp) -> TaskView:
        deadline = self.cfg.episode_s * 10 if o.kind == "T6" else self.cfg.episode_s   # going home is always worth trying
        return TaskView(o.id, o.kind, o.xy[0], o.xy[1], int(round(o.value)), 0.0, deadline,
                        (o.xy,), o.service_s, True, 0.0)

    # --- decisions ------------------------------------------------------------------------------
    def _decide(self):
        cfg = self.cfg
        if cfg.policy == "P5":
            self._decide_leader()
            return
        if self.policy.centralized:
            if self.t + 1e-9 < min(self.next_decide):
                return
            self.next_decide = [self.t + cfg.decide_every] * self.n
            opps = {i: self.opportunities(i) for i in range(self.n)}
            todo = [i for i in range(self.n) if self.exec[i] is None or self.claim[i] is None or self.claim[i].kind == "explore"]
            busy_keys = {self.claim[j].key for j in range(self.n)
                         if j not in todo and self.claim[j] is not None and self.claim[j].kind not in ("explore", "T1", "T6")}
            pool = {}
            for i in todo:
                for o in opps[i]:
                    if o.key not in busy_keys:
                        pool.setdefault(o.key, o)
            for i in todo:                              # a boat with nothing left to claim heads home
                if not any(o.key in pool and o.kind != "explore" for o in opps[i]) and not any(o.kind == "explore" for o in opps[i]) \
                        and not self.ref.prog[i].home and self.ref.prog[i].t1_done:
                    m6 = self.course.module("T6")[0]
                    pool.setdefault(f"T6#{i}", Opp(f"T6#{i}", "T6", self.gate_mid(m6, 0), 25.0))
                    opps[i] = opps[i] + [pool[f"T6#{i}"]]
            ordered = list(pool.values())
            states = [self.state(i) for i in todo]
            if not ordered or not todo:
                return
            cobs = CentralObservation(self.t, states, [self.taskview(o) for o in ordered],
                                      [self.claim[i].id if self.claim[i] else None for i in todo],
                                      self.chart, self.dcfg, self.tables)
            t0 = time.perf_counter()
            allowed = [[o.key in {x.key for x in opps[i]} for o in ordered] for i in todo]
            ds = masked_decide_all(cobs, allowed)
            self.cpu.append((time.perf_counter() - t0) / max(1, len(todo)))
            by_id = {o.id: o for o in ordered}
            for i, d in zip(todo, ds):
                o = by_id.get(d.task)
                if o is not None and o.key not in {x.key for x in opps[i]}:
                    o = None
                bids = [(e, todo[k]) for e, k in getattr(d, "bids", [])]
                self._claim(i, o, d.eta if o else math.inf, util=d.utility, how="planner", bids=bids)
            return
        for i in range(self.n):
            busy = self.exec[i] is not None and self.claim[i] is not None and self.claim[i].kind != "explore"
            if busy or self.t + 1e-9 < self.next_decide[i]:
                continue
            self.next_decide[i] = self.t + cfg.decide_every
            opps = self.opportunities(i)
            if not opps:
                continue
            obs = BoatObservation(i, self.t, self.state(i), [self.taskview(o) for o in opps],
                                  {k: (m, self.t - m.t_sent) for k, m in self.beliefs[i].items()},
                                  self.claim[i].id if self.claim[i] else None, self.chart, self.dcfg, self.tables)
            t0 = time.perf_counter()
            d = self.policy.decide(obs)
            self.cpu.append(time.perf_counter() - t0)
            o = next((x for x in opps if x.id == d.task), None)
            self._claim(i, o, d.eta if o else math.inf, util=d.utility, how="own")

    def _claim(self, i, o: Optional[Opp], eta, util=float("nan"), how="own", bids=()):
        if o is None:
            return
        if self.claim[i] is not None and self.claim[i].key == o.key and self.exec[i] is not None:
            if self.reason[i] is not None:
                self.reason[i].update(eta=eta, util=util)
            return
        self.reason[i] = dict(key=o.key, kind=o.kind, value=o.value, eta=eta, util=util, how=how, t=self.t)
        if o.kind != "explore":
            alt = ""
            if bids:
                e2, k2 = bids[0]
                alt = f"; boat {k2 + 1} would need {e2:.0f} s"
            self.log.append((self.t, i, f"{'assigned' if how == 'planner' else 'takes'} {o.key.split('#')[0]}: "
                                        f"{eta:.0f} s away, worth {o.value:.0f}{alt}"))
        self.claim[i] = o
        self.claim_arrival[i] = self.t + (eta if math.isfinite(eta) else 60.0)
        self.exec[i] = Executor(self, i, o)
        self.events.append((self.t, "claim", i, o.key))

    # --- control ---------------------------------------------------------------------------------
    def _control(self):
        cfg, W = self.dcfg, self.chart
        n = self.n
        v_des, r_des = np.zeros(n), np.zeros(n)
        X, Y, PSI, V = self.S[0], self.S[1], self.S[2], self.S[3]
        for i in range(n):
            x, y, psi, v = X[i], Y[i], PSI[i], V[i]
            if self.exec[i] is not None:
                psi_d, vd, fin = self.exec[i].step()
                if fin:
                    cl = self.claim[i]
                    if cl is not None and cl.kind == "T5" and cl.key not in self.ref.done:
                        self.fails[cl.key] = self.fails.get(cl.key, 0) + 1
                        self.log.append((self.t, i, f"misses the delivery at {cl.key} (attempt {self.fails[cl.key]})"))
                    if cl is not None and cl.kind == "T3look":
                        self.fails[cl.key] = self.fails.get(cl.key, 0) + 1
                    self.events.append((self.t, "finished", i, self.claim[i].key if self.claim[i] else ""))
                    self.exec[i] = None
                    self.claim[i] = None
                    self.next_decide[i] = self.t
            else:
                psi_d, vd = psi, 0.0
                if not self.ref.prog[i].home:
                    self.idle_s[i] += self.dt * self.dcfg.control_every
            # give way to crossing boats (sensed directly: within 16 m)
            for j in range(n):
                if j == i:
                    continue
                px, py = X[j] - x, Y[j] - y
                d = math.hypot(px, py)
                if d > 12:
                    continue
                wx = V[j] * math.cos(PSI[j]) - v * math.cos(psi); wy = V[j] * math.sin(PSI[j]) - v * math.sin(psi)
                closing = px * wx + py * wy < 0
                w2 = wx * wx + wy * wy
                tc = -(px * wx + py * wy) / w2 if w2 > 1e-6 else 0
                dc = math.hypot(px + wx * tc, py + wy * tc)
                beta = wrap(math.atan2(py, px) - psi)
                if closing and ((0 < tc < 5 and dc < 3.2) or d < 3.0):
                    if -1.4 < beta < 0.3 or (d < 3.0 and abs(beta) < 1.57):     # yield to boats ahead/right, never to followers
                        alt = psi_d - 0.55
                        if W.clearance_at(x + 4 * math.cos(alt), y + 4 * math.sin(alt)) > 1.2:
                            psi_d = alt
                        vd = min(vd, 0.5 * abs(v) + 0.15)
            gx, gy = vd * math.cos(psi_d), vd * math.sin(psi_d)
            for j in range(n):                       # separation from nearby boats (sensed directly)
                if j == i:
                    continue
                dx_, dy_ = x - X[j], y - Y[j]
                d_ = math.hypot(dx_, dy_)
                if 1e-6 < d_ < 3.6:
                    # right of way: the boat that is ahead (along my heading) is never pushed; side by side, the
                    # lower index has priority.  Only the yielding boat backs off, so narrow gates cannot deadlock.
                    ahead = -(dx_ * math.cos(psi) + dy_ * math.sin(psi)) / d_      # >0: j is in front of me
                    j_ahead_of_me = ahead > 0.35 or (abs(ahead) <= 0.35 and j < i)
                    parked_i = self.exec[i] is None and self.ref.prog[i].home
                    parked_j = self.exec[j] is None and self.ref.prog[j].home
                    if parked_i != parked_j:
                        j_ahead_of_me = parked_i         # a parked boat always gives way to a working one
                    if j_ahead_of_me or d_ < 1.9:
                        push = 0.55 * (3.6 - d_)
                        gx += push * dx_ / d_; gy += push * dy_ / d_
            clr = W.clearance_at(x, y)
            if clr < 1.1:
                ex, ey = W.clearance_grad(x, y)
                gx += 0.8 * (1.1 - clr) * ex; gy += 0.8 * (1.1 - clr) * ey
            cu, cv = self.course.current.mean(x, y)
            wx_, wy_ = gx - float(cu), gy - float(cv)
            wn = math.hypot(wx_, wy_)
            if wn > 0.05:
                psi_d = math.atan2(wy_, wx_)
                vd = wn * max(0.0, math.cos(wrap(psi_d - psi)))
            r_des[i] = heading_controller(psi_d, psi, cfg)
            v_des[i] = min(vd, cfg.cruise_frac * cfg.v_max)
        self.TL, self.TR = allocate_thrust(v_des, r_des, self.S[3], self.S[4], cfg)

    def _contacts(self, prev):
        c = self.course
        X, Y = self.S[0], self.S[1]
        br = self.cfg.beam_radius
        for i in range(self.n):
            for o in c.objects:
                if abs(o.x - X[i]) > 3 or abs(o.y - Y[i]) > 3:
                    continue
                rr = (0.95 if o.cls in ("vessel_orange", "vessel_black", "moored") else o.r) + br
                d = math.hypot(X[i] - o.x, Y[i] - o.y)
                key = (i, o.id)
                if d < rr:
                    if key not in self.contact_set:
                        self.contact_set.add(key); self.contacts += 1
                        self.events.append((self.t, "contact", i, o.cls))
                        cl = self.claim[i]
                        if cl is not None and cl.kind in ("T2", "T4") and o.module >= 0 and c.modules[o.module].kind == cl.kind:
                            self.ref.strike(i, cl.kind)
                    u = ((X[i] - o.x) / max(d, 1e-6), (Y[i] - o.y) / max(d, 1e-6))
                    X[i], Y[i] = o.x + u[0] * (rr + 0.01), o.y + u[1] * (rr + 0.01)
                    self.S[3, i] *= 0.5
                elif d > rr + 0.6:
                    self.contact_set.discard(key)
            if self.chart.is_blocked(X[i], Y[i]) or self.chart.clearance_at(X[i], Y[i]) < 0.4:
                X[i], Y[i] = prev[0, i], prev[1, i]
                self.S[3, i] *= 0.3
                key = (i, "structure")
                if key not in self.contact_set:
                    self.contact_set.add(key); self.contacts += 1
                    self.events.append((self.t, "contact", i, "structure"))
            elif self.chart.clearance_at(X[i], Y[i]) > 1.2:
                self.contact_set.discard((i, "structure"))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx, dy = X[j] - X[i], Y[j] - Y[i]
                d = math.hypot(dx, dy)
                key = ("bb", i, j)
                if d < 1.5:
                    if key not in self.contact_set:
                        self.contact_set.add(key); self.contacts += 1
                        self.events.append((self.t, "contact", i, f"boat{j}"))
                    ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
                    push = (1.5 - d) / 2 + 0.01
                    X[i] -= ux * push; Y[i] -= uy * push; X[j] += ux * push; Y[j] += uy * push
                elif d > 2.2:
                    self.contact_set.discard(key)

    def _comms(self):
        cfg = self.dcfg
        for r, msg in self.net.deliver(self.t):
            old = self.beliefs[r].get(msg.sender)
            if old is None or msg.t_sent > old.t_sent:
                self.beliefs[r][msg.sender] = msg
            if hasattr(self.know, "on_message"):
                self.know.on_message(r, msg)
        if self.k % int(round(cfg.msg_period_s / self.dt)) == 0:
            slot = int(round(self.t / cfg.msg_period_s))
            for i in range(self.n):
                x, y, psi, v = self.state(i)
                cl = self.claim[i]
                msg = self.know.make_message(i, self.t, (x, y, psi, v), cl.id if cl else None, self.claim_arrival[i]) \
                    if hasattr(self.know, "make_message") else \
                    Message(i, self.t, x, y, psi, v, cl.id if cl else None, self.claim_arrival[i], frozenset())
                self.net.broadcast(slot, msg, self.n)

    # --- main loop -------------------------------------------------------------------------------
    def step(self):
        if self.done_flag:
            return
        self._comms()
        self.know.update(self.t)
        if self.k % 10 == 0:
            self._track_found()
        if self.k % self.dcfg.control_every == 0:
            self._resolve_conflicts()
        self._decide()
        if self.k % self.dcfg.control_every == 0:
            self._control()
        prev = self.S.copy()
        self.S = rk4_step(self.S, self.TL, self.TR, self.course.current.total, self.t, self.dt, self.dcfg,
                          self.dist[self.k, 0], self.dist[self.k, 1])
        self._contacts(prev)
        for i in range(self.n):
            self.ref.update(self.t, self.dt, i, (prev[0, i], prev[1, i]), (self.S[0, i], self.S[1, i]),
                            self.S[2, i], self.S[3, i])
        self.ref.settle(self.t, force=(self.k + 1 >= self.n_steps))
        if self.k + 1 >= self.n_steps:
            self.ref.finalize(self.t)
        self.energy += (self.TL ** 2 + self.TR ** 2) * self.dt
        self.k += 1
        self.t = self.k * self.dt
        if self.k >= self.n_steps:
            self.done_flag = True

    def run(self):
        while not self.done_flag:
            self.step()
        return self.metrics()

    def metrics(self):
        cfg = self.cfg
        pts = self.ref.points
        en = float(self.energy.sum())
        by_task = {k: 0.0 for k in ("T1", "T2", "T3", "T4", "T5", "T6")}
        bonus = 0.0
        for e in self.ref.events:
            if e.note.startswith("bonus"):
                bonus += e.points
                continue
            by_task[e.task] = by_task.get(e.task, 0.0) + e.points
        need = ["T2", "T3dock", "T4"] + [f"T5:{m.id}" for m in self.course.module("T5")]
        tf = [self.found_t[k][0] for k in need if k in self.found_t]
        return dict(seed=self.seed, policy=cfg.policy, n_boats=self.n, comm_mode=cfg.comm_mode,
                    perception=cfg.perception, variant=cfg.variant, shortcut_open=cfg.shortcut_open,
                    welfare=pts - cfg.w_energy * en - cfg.w_contact * self.contacts, points=pts,
                    contacts=self.contacts, energy=en, **{f"pts_{k}": v for k, v in sorted(by_task.items())},
                    t1_boats=sum(p.t1_done for p in self.ref.prog), home_boats=sum(p.home for p in self.ref.prog),
                    t3_ok=int("T3" in self.ref.done), t4_done=int("T4" in self.ref.done),
                    t5_done=sum(1 for k in self.ref.done if k.startswith("T5")),
                    bonus=bonus, task_points=pts - bonus, t3_attempted=int(self.ref.t3_attempted),
                    idle_frac=float(self.idle_s.sum()) / (self.n * cfg.episode_s),
                    found_all_t=max(tf) if len(tf) == len(need) else math.nan, found_n=len(tf), found_need=len(need),
                    cpu_ms=1000 * float(np.mean(self.cpu)) if self.cpu else 0.0, config_hash=cfg.hash())
