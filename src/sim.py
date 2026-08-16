"""The simulation engine (shared verbatim by headless experiments and the Pygame GUI).

Per physics step (dt): deliver messages -> local sensing -> scheduled decisions ->
guidance/avoidance/reservation/control (10 Hz) -> RK4 physics -> contacts ->
task progress -> broadcasts.  All randomness comes from pre-generated exogenous tapes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .comms import CommTape, Message, Network
from .config import SimConfig
from .navigation import (GuidanceState, control_eta, corner_angles, graph_eta, heading_controller,
                         los_command, path_length, penalty_tables, plan_route, wrap)
from .physics import allocate_thrust, disturbance_tape, rk4_step
from .policies import (BoatObservation, CentralObservation, Decision, TaskView, make_policy)
from .rng import stream
from .tasks import GATE_HALF, RADIUS, VMAX_SVC, Task, generate_tasks
from .world import World, generate_world


@dataclass
class Event:
    t: float
    kind: str           # complete | collision | obstacle | redundant | expire | decide | drop | near
    boat: int = -1
    task: int = -1
    other: int = -1
    value: float = 0.0
    x: float = 0.0
    y: float = 0.0


@dataclass
class BoatMind:
    """Everything a boat legitimately knows / has decided (no hidden truth)."""
    claim: Optional[int] = None
    utility: float = 0.0
    claim_arrival: float = math.inf
    guidance: Optional[GuidanceState] = None
    beliefs: Dict[int, Message] = field(default_factory=dict)
    known_done: set = field(default_factory=set)
    next_decision: float = 0.0
    switches: int = 0
    holding: bool = False
    avoiding: bool = False
    missed: int = 0
    assign_seen: float = -1.0
    decision_t: float = 0.0
    pred: dict = field(default_factory=dict)   # predictions logged at commitment


class Simulation:
    def __init__(self, cfg: SimConfig, seed: int, world: Optional[World] = None,
                 log_steps: bool = False):
        self.cfg, self.seed = cfg, seed
        self.world = world or generate_world(cfg, seed)
        self.tasks: List[Task] = generate_tasks(self.world, cfg)
        self.world.set_point_obstacles([b for tk in self.tasks if tk.kind == "gate" for b in tk.buoys()],
                                       radius=0.9)
        self.tables = penalty_tables(cfg)
        self.policy = make_policy(cfg.policy)
        n = cfg.n_boats
        self.n = n
        self.n_steps = int(round(cfg.episode_s / cfg.dt))
        self.dist_tape = disturbance_tape(seed, n, self.n_steps + 1, cfg)
        srng = stream(seed, "sensors")
        self.sensor_tape = srng.standard_normal((self.n_steps // cfg.control_every + 2, n, 2)) \
            * cfg.sensor_noise_m
        self.net = Network(CommTape(seed, cfg))
        # start in a column off the launch pier, heading +x
        lx, ly = self.world.launch
        ys = ly + (np.arange(n) - (n - 1) / 2) * 7.0
        xs = lx + 3.0 * (np.arange(n) % 2)
        self.S = np.vstack([xs, ys, np.zeros(n), np.zeros(n), np.zeros(n)]).astype(float)
        self.TL = np.zeros(n); self.TR = np.zeros(n)
        self.minds = [BoatMind(next_decision=0.25 * i * cfg.replan_period_s / max(1, n)) for i in range(n)]
        self.t = 0.0
        self.k = 0
        self.events: List[Event] = []
        self.decisions: List[dict] = []
        self.energy = np.zeros(n)
        self.dist_travel = np.zeros(n)
        self.contact_bb = np.zeros((n, n), bool)
        self.contact_obs = np.zeros(n, bool)
        self.buoy_contact = set()
        self.near_bb = np.zeros((n, n), bool)
        self.redundant_seen = set()
        self.expired_logged = set()
        self.cpu = []
        self.ages_used = []
        self.fallbacks = 0
        self.log_steps = log_steps
        self.trace: List[np.ndarray] = []
        self._central_next = 0.0
        self.leader_assign = ()
        self.done = False
        # idle loiter points (same for all policies): quartile points on open water
        W_ = self.world
        self.idle_points = []
        for fx, fy in ((0.3, 0.3), (0.7, 0.7), (0.3, 0.7), (0.7, 0.3)):
            i0, j0 = W_.snap_free(fx * W_.width, fy * W_.height)
            self.idle_points.append(W_.cell_center(i0, j0))

    # ---------------------------------------------------------------------------------
    def task_view(self, tk: Task) -> TaskView:
        return TaskView(tk.id, tk.kind, tk.x, tk.y, tk.value, tk.release, tk.deadline,
                        tuple(tk.points), tk.service_s, tk.stop, tk.extra_len)

    def observe(self, i: int) -> BoatObservation:
        m = self.minds[i]
        x, y, psi, v, _ = self.S[:, i]
        tasks = [self.task_view(tk) for tk in self.tasks
                 if tk.release <= self.t <= tk.deadline and tk.id not in m.known_done]
        beliefs = {k: (msg, self.t - msg.t_sent) for k, msg in m.beliefs.items()}
        return BoatObservation(i, self.t, (float(x), float(y), float(psi), float(v)), tasks,
                               beliefs, m.claim, self.world, self.cfg, self.tables)

    def central_observe(self) -> CentralObservation:
        states = [tuple(float(a) for a in self.S[:4, i]) for i in range(self.n)]
        tasks = [self.task_view(tk) for tk in self.tasks if tk.active(self.t)]
        return CentralObservation(self.t, states, tasks, [m.claim for m in self.minds],
                                  self.world, self.cfg, self.tables)

    def _apply_decision(self, i: int, d: Decision):
        m = self.minds[i]
        x, y, psi, v, _ = self.S[:, i]
        if d.task != m.claim:
            if m.claim is not None:
                m.switches += 1
            m.claim = d.task
            m.decision_t = self.t
            if d.task is not None:
                tk = self.tasks[d.task]
                wps = d.wps
                if wps is None:
                    wps, _ = plan_route(self.world, (x, y), tk.points)
                st = (float(x), float(y), float(psi), float(v))
                eg = graph_eta(self.world, (x, y), self.task_view(tk), self.cfg)
                ec, _ = control_eta(self.world, st, self.task_view(tk), self.cfg, self.tables, wps=wps)
                m.guidance = GuidanceState(wps, 0, tk.stop, 2 if tk.kind == 'gate' else 0) if wps else None
                m.pred = dict(t=self.t, task=tk.id, eta_graph=eg, eta_control=ec,
                              eta_policy=d.eta, length=path_length(wps) if wps else math.nan,
                              turn=sum(corner_angles(wps)) if wps else math.nan,
                              init_err=abs(wrap(math.atan2(wps[1][1] - y, wps[1][0] - x) - psi))
                              if wps and len(wps) > 1 else 0.0)
            else:
                m.guidance = None
                m.pred = {}
        # same task: keep the committed route (identical rule for every policy); a
        # missed approach is handled in _control by replanning from the current pose.
        m.utility = d.utility
        m.claim_arrival = self.t + d.eta if d.task is not None else math.inf
        self.cpu.append(d.cpu_s)
        self.fallbacks += d.fallbacks
        if d.max_age_used:
            self.ages_used.append(d.max_age_used)
        self.events.append(Event(self.t, "decide", i, -1 if d.task is None else d.task))

    def _leader_observe(self):
        """What the leader (boat 0) knows: its own state, teammates' last messages, its task view."""
        Lm = self.minds[0]
        boats = [0] + sorted(Lm.beliefs)
        states, claims = [], []
        for b in boats:
            if b == 0:
                states.append(tuple(float(a) for a in self.S[:4, 0])); claims.append(Lm.claim)
            else:
                msg = Lm.beliefs[b]
                states.append((msg.x, msg.y, msg.psi, msg.v)); claims.append(msg.claim)
        tasks = [self.task_view(tk) for tk in self.tasks
                 if tk.release <= self.t <= tk.deadline and tk.id not in Lm.known_done]
        return boats, CentralObservation(self.t, states, tasks, claims, self.world, self.cfg, self.tables)

    def _decide(self):
        cfg = self.cfg
        if getattr(self.policy, "leader", False):
            if self.t + 1e-9 >= self._central_next:
                t0 = time.perf_counter()
                boats, cobs = self._leader_observe()
                ds = self.policy.decide_all(cobs)
                el = (time.perf_counter() - t0) / max(1, len(boats))
                self.leader_assign = tuple((b, d.task) for b, d in zip(boats, ds))
                ds[0].cpu_s = el
                self._apply_decision(0, ds[0])
                self._central_next = self.t + cfg.replan_period_s
            for i, m in enumerate(self.minds):     # a follower drops a task it knows is done
                if m.claim is not None and m.claim in m.known_done:
                    self._apply_decision(i, Decision(None, 0.0, math.inf))
            return
        if self.policy.centralized:
            if self.t + 1e-9 >= self._central_next:
                t0 = time.perf_counter()
                ds = self.policy.decide_all(self.central_observe())
                el = (time.perf_counter() - t0) / self.n
                for i, d in enumerate(ds):
                    d.cpu_s = el
                    self._apply_decision(i, d)
                self._central_next = self.t + cfg.replan_period_s
            return
        for i, m in enumerate(self.minds):
            trigger = (m.claim is not None and m.claim in m.known_done)
            if trigger or self.t + 1e-9 >= m.next_decision:
                obs = self.observe(i)
                t0 = time.perf_counter()
                d = self.policy.decide(obs)
                d.cpu_s = time.perf_counter() - t0
                self._apply_decision(i, d)
                m.next_decision = self.t + cfg.replan_period_s

    # ---------------------------------------------------------------------------------
    def _control(self):
        cfg, W = self.cfg, self.world
        n = self.n
        v_des = np.zeros(n); r_des = np.zeros(n)
        kk = min(self.k // cfg.control_every, len(self.sensor_tape) - 1)
        noise = self.sensor_tape[kk]
        X, Y, PSI, V = self.S[0], self.S[1], self.S[2], self.S[3]
        for i in range(n):
            m = self.minds[i]
            x, y, psi, v = X[i], Y[i], PSI[i], V[i]
            m.holding = m.avoiding = False
            if m.guidance is None and m.claim is None:
                ix, iy = self.idle_points[i % len(self.idle_points)]
                if math.hypot(ix - x, iy - y) > 6.0:
                    wps, _ = plan_route(W, (x, y), [(ix, iy)])
                    m.guidance = GuidanceState(wps, 0, True) if wps else None
            if m.guidance is None:
                psi_d, vd = psi, 0.0
            else:
                psi_d, vd, d_end = los_command(m.guidance, x, y, psi, cfg)
                if m.claim is not None and d_end < 0.8 and not m.guidance.stop_at_end:
                    # route exhausted without completing (missed gate): replan
                    tk = self.tasks[m.claim]
                    if tk.active(self.t):
                        wps, _ = plan_route(W, (x, y), tk.points)
                        if wps:
                            m.guidance = GuidanceState(wps, 0, tk.stop, 2 if tk.kind == 'gate' else 0)
                            m.missed += 1
                            psi_d, vd, _ = los_command(m.guidance, x, y, psi, cfg)
            # --- channel reservation (local sensing; capacity 1) -------------------
            for ch in W.channels:
                if ch.contains(x, y, pad=0.5):
                    continue
                cx, cy = ch.center
                d_i = math.hypot(cx - x, cy - y)
                if d_i > 13 or m.guidance is None:
                    continue
                # does my path pass through this channel soon?
                g = m.guidance
                through = any(ch.contains(px, py, pad=1.0)
                              for a, b in zip(g.wps[g.seg:g.seg + 3], g.wps[g.seg + 1:g.seg + 4])
                              for px, py in ((a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
                                             for s in np.linspace(0, 1, 12)))
                if not through:
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    xj, yj = X[j] + noise[j, 0], Y[j] + noise[j, 1]
                    if math.hypot(xj - x, yj - y) > cfg.sensor_range_m:
                        continue
                    inside = ch.contains(xj, yj, pad=0.5)
                    d_j = math.hypot(cx - xj, cy - yj)
                    gj = self.minds[j].guidance
                    j_wait = (self.minds[j].holding is False and d_j < d_i and d_j < 13 and gj is not None
                              and not ch.contains(xj, yj, pad=0.5)
                              and math.cos(math.atan2(cy - yj, cx - xj) - PSI[j]) > 0.3)
                    if inside or j_wait or (d_j == d_i and j < i):
                        m.holding = True
                        break
                if m.holding:
                    vd = min(vd, max(0.0, 0.35 * (d_i - 9.5)))
                    break
            # --- collision avoidance (COLREGS-flavoured, sensed positions) ----------
            vx, vy = v * math.cos(psi), v * math.sin(psi)
            for j in range(n):
                if j == i:
                    continue
                xj, yj = X[j] + noise[j, 0], Y[j] + noise[j, 1]
                px, py = xj - x, yj - y
                dist = math.hypot(px, py)
                if dist > cfg.sensor_range_m:
                    continue
                wx = V[j] * math.cos(PSI[j]) - vx
                wy = V[j] * math.sin(PSI[j]) - vy
                w2 = wx * wx + wy * wy
                tcpa = -(px * wx + py * wy) / w2 if w2 > 1e-6 else 0.0
                dcpa = math.hypot(px + wx * tcpa, py + wy * tcpa)
                beta = wrap(math.atan2(py, px) - psi)
                closing = (px * wx + py * wy) < 0
                close = dist < 2 * cfg.hull_radius_m + 2.0 and closing
                if closing and ((0 < tcpa < cfg.avoid_tcpa_s and dcpa < cfg.avoid_dcpa_m) or (close and abs(beta) < 1.6)):
                    head_on = abs(beta) < 0.3 and abs(wrap(PSI[j] - psi - math.pi)) < 0.5
                    give_way = head_on or (-1.96 < beta < -0.05) or close
                    if give_way:
                        m.avoiding = True
                        alt = psi_d - 0.6
                        ahead = (x + 5 * math.cos(alt), y + 5 * math.sin(alt))
                        if W.clearance_at(*ahead) > cfg.hull_radius_m + 0.8:
                            psi_d = alt
                        vd = min(vd, 0.45 * v + 0.2) if not close else min(vd, 0.25)
            # --- obstacle repulsion + current compensation (crab angle) ---------------
            gx, gy = vd * math.cos(psi_d), vd * math.sin(psi_d)       # desired ground velocity
            clr = W.clearance_at(x, y)
            lim = cfg.hull_radius_m + 1.3
            if clr < lim:
                ex, ey = W.clearance_grad(x, y)
                push = 0.9 * (lim - clr)
                gx += push * ex; gy += push * ey
            cu, cv = W.current.mean(x, y)                             # charted current only
            wx_, wy_ = gx - float(cu), gy - float(cv)
            wn = math.hypot(wx_, wy_)
            if wn > 0.05:
                psi_d = math.atan2(wy_, wx_)
                vd = wn * max(0.0, math.cos(wrap(psi_d - psi)))         # don't thrust sideways
            r_des[i] = heading_controller(psi_d, psi, cfg)
            v_des[i] = min(vd, cfg.cruise_frac * cfg.v_max)
        self.TL, self.TR = allocate_thrust(v_des, r_des, self.S[3], self.S[4], cfg)

    def _contacts(self, prevS):
        cfg, W = self.cfg, self.world
        R = cfg.hull_radius_m
        X, Y = self.S[0], self.S[1]
        for i in range(self.n):
            # obstacles: push out along the clearance gradient
            c = W.clearance_at(X[i], Y[i])
            if c < R or W.is_blocked(X[i], Y[i]):
                if not self.contact_obs[i]:
                    self.events.append(Event(self.t, "obstacle", i, x=X[i], y=Y[i]))
                self.contact_obs[i] = True
                X[i], Y[i] = prevS[0, i], prevS[1, i]
                self.S[3, i] *= 0.3
            elif c > R + 0.4:
                self.contact_obs[i] = False
            # gate buoys
            for tk in self.tasks:
                if tk.kind != "gate" or abs(tk.x - X[i]) > 6 or abs(tk.y - Y[i]) > 6:
                    continue
                for bx, by in tk.buoys():
                    dbx, dby = X[i] - bx, Y[i] - by
                    db = math.hypot(dbx, dby)
                    key = (i, round(bx, 2), round(by, 2))
                    if db > R + 0.9:
                        self.buoy_contact.discard(key)
                    if db < R + 0.35:
                        if key not in self.buoy_contact:
                            self.events.append(Event(self.t, "obstacle", i, x=X[i], y=Y[i]))
                        self.buoy_contact.add(key)
                        db = max(db, 1e-6)
                        X[i] = bx + dbx / db * (R + 0.36); Y[i] = by + dby / db * (R + 0.36)
                        self.S[3, i] *= 0.5
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx, dy = X[j] - X[i], Y[j] - Y[i]
                d = math.hypot(dx, dy)
                if d < 2 * R:
                    if not self.contact_bb[i, j]:
                        self.events.append(Event(self.t, "collision", i, other=j,
                                                 x=(X[i] + X[j]) / 2, y=(Y[i] + Y[j]) / 2))
                    self.contact_bb[i, j] = True
                    # separate symmetrically, damp speeds
                    ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
                    push = (2 * R - d) / 2 + 0.01
                    X[i] -= ux * push; Y[i] -= uy * push
                    X[j] += ux * push; Y[j] += uy * push
                    self.S[3, i] *= 0.5; self.S[3, j] *= 0.5
                elif d > 2 * R + 0.5:
                    self.contact_bb[i, j] = False
                if d < 2 * R + 1.5:
                    if not self.near_bb[i, j]:
                        self.events.append(Event(self.t, "near", i, other=j))
                    self.near_bb[i, j] = True
                elif d > 2 * R + 3.0:
                    self.near_bb[i, j] = False

    def _complete(self, tk: Task, i: int):
        tk.completed_by, tk.completed_at = i, self.t
        self.minds[i].known_done.add(tk.id)
        self.events.append(Event(self.t, "complete", i, tk.id, value=tk.value, x=tk.x, y=tk.y))
        m = self.minds[i]
        if m.pred and m.pred.get("task") == tk.id:
            self.decisions.append(dict(m.pred, boat=i, realized=self.t - m.pred["t"],
                                       kind=tk.kind))

    def _tasks(self, prevS):
        X, Y, PSI, V = self.S[0], self.S[1], self.S[2], self.S[3]
        dt = self.cfg.dt
        for tk in self.tasks:
            released = tk.release <= self.t
            if not released:
                continue
            if tk.completed_by is None and self.t > tk.deadline and tk.id not in self.expired_logged:
                self.expired_logged.add(tk.id)
                self.events.append(Event(self.t, "expire", task=tk.id, x=tk.x, y=tk.y))
            for i in range(self.n):
                if abs(tk.x - X[i]) > 8 or abs(tk.y - Y[i]) > 8:
                    continue
                if tk.kind == "gate":
                    nx, ny = math.cos(tk.heading), math.sin(tk.heading)
                    s0 = (prevS[0, i] - tk.x) * nx + (prevS[1, i] - tk.y) * ny
                    s1 = (X[i] - tk.x) * nx + (Y[i] - tk.y) * ny
                    lat = abs(-(X[i] - tk.x) * ny + (Y[i] - tk.y) * nx)
                    crossed = s0 < 0 <= s1 and lat < GATE_HALF
                    if crossed:
                        if tk.completed_by is None and tk.active(self.t):
                            self._complete(tk, i)
                        elif tk.completed_by is not None and tk.completed_by != i:
                            self._redundant(tk, i)
                else:
                    inside = math.hypot(tk.x - X[i], tk.y - Y[i]) < RADIUS[tk.kind]
                    if inside and abs(V[i]) < VMAX_SVC[tk.kind]:
                        if tk.completed_by is not None:
                            if tk.completed_by != i:
                                self._redundant(tk, i)
                            continue
                        if not tk.active(self.t):
                            continue
                        tk.dwell[i] = tk.dwell.get(i, 0.0) + dt
                        if tk.dwell[i] >= tk.service_s:
                            self._complete(tk, i)
                    else:
                        tk.dwell[i] = 0.0

    def _redundant(self, tk, i):
        if (tk.id, i) not in self.redundant_seen:
            self.redundant_seen.add((tk.id, i))
            self.minds[i].known_done.add(tk.id)
            self.events.append(Event(self.t, "redundant", i, tk.id, x=tk.x, y=tk.y))

    def _sense(self):
        """Local observation of task status lights within sensor range."""
        X, Y = self.S[0], self.S[1]
        for tk in self.tasks:
            if tk.completed_by is None:
                continue
            for i in range(self.n):
                if tk.id not in self.minds[i].known_done and \
                        math.hypot(tk.x - X[i], tk.y - Y[i]) < self.cfg.sensor_range_m:
                    self.minds[i].known_done.add(tk.id)

    def _comms(self):
        cfg = self.cfg
        for r, msg in self.net.deliver(self.t):
            m = self.minds[r]
            old = m.beliefs.get(msg.sender)
            if old is None or msg.t_sent > old.t_sent:
                m.beliefs[msg.sender] = msg
            m.known_done |= msg.completed
            if msg.assign and r != 0 and msg.sender == 0 and msg.t_sent > m.assign_seen:
                m.assign_seen = msg.t_sent
                given = dict(msg.assign)
                if r in given and given[r] != m.claim and given[r] not in m.known_done:
                    task = given[r]
                    if task is None:
                        self._apply_decision(r, Decision(None, 0.0, math.inf))
                    else:
                        st = tuple(float(a) for a in self.S[:4, r])
                        eta, wps = control_eta(self.world, st, self.task_view(self.tasks[task]), cfg, self.tables)
                        self._apply_decision(r, Decision(task, 0.0, eta, wps=wps))
        per = cfg.msg_period_s
        if self.k % int(round(per / cfg.dt)) == 0:
            slot = int(round(self.t / per))
            for i in range(self.n):
                mi = self.minds[i]
                x, y, psi, v, _ = self.S[:, i]
                msg = Message(i, self.t, float(x), float(y), float(psi), float(v), mi.claim,
                              mi.claim_arrival, frozenset(mi.known_done),
                              self.leader_assign if i == 0 else ())
                self.net.broadcast(slot, msg, self.n)

    # ---------------------------------------------------------------------------------
    def step(self):
        if self.done:
            return
        cfg = self.cfg
        self._comms()
        self._sense()
        self._decide()
        if self.k % cfg.control_every == 0:
            self._control()
        prev = self.S.copy()
        wv = self.dist_tape[self.k, 0]
        wr = self.dist_tape[self.k, 1]
        self.S = rk4_step(self.S, self.TL, self.TR, self.world.current.total, self.t, cfg.dt, cfg, wv, wr)
        self._contacts(prev)
        self._tasks(prev)
        self.energy += (self.TL ** 2 + self.TR ** 2) * cfg.dt
        self.dist_travel += np.hypot(self.S[0] - prev[0], self.S[1] - prev[1])
        self.t = (self.k + 1) * cfg.dt
        self.k += 1
        if self.log_steps:
            self.trace.append(self.S[:4].copy())
        if self.k >= self.n_steps:
            self.done = True

    def run(self) -> dict:
        while not self.done:
            self.step()
        return self.metrics()

    def metrics(self) -> dict:
        cfg = self.cfg
        done = [tk for tk in self.tasks if tk.completed_by is not None]
        points = float(sum(tk.value for tk in done))
        coll_bb = sum(1 for e in self.events if e.kind == "collision")
        coll_obs = sum(1 for e in self.events if e.kind == "obstacle")
        energy = float(self.energy.sum())
        welfare = points - cfg.w_energy * energy - cfg.w_collision * (coll_bb + coll_obs)
        total_value = float(sum(tk.value for tk in self.tasks))
        return dict(
            seed=self.seed, policy=cfg.policy, n_boats=self.n, comm_mode=cfg.comm_mode,
            welfare=welfare, points=points, total_value=total_value,
            points_frac=points / total_value if total_value else 0.0,
            completed=len(done), n_tasks=len(self.tasks),
            completion_rate=len(done) / len(self.tasks),
            expired=len(self.tasks) - len(done),
            redundant_visits=sum(1 for e in self.events if e.kind == "redundant"),
            switches=sum(m.switches for m in self.minds),
            makespan=max((tk.completed_at for tk in done), default=math.nan),
            distance=float(self.dist_travel.sum()), energy=energy,
            collisions_boat=coll_bb, collisions_obstacle=coll_obs,
            near_misses=sum(1 for e in self.events if e.kind == "near"),
            mean_age_used=float(np.mean(self.ages_used)) if self.ages_used else 0.0,
            cpu_ms_mean=1000 * float(np.mean(self.cpu)) if self.cpu else 0.0,
            cpu_ms_max=1000 * float(np.max(self.cpu)) if self.cpu else 0.0,
            fallbacks=self.fallbacks,
            missed_approaches=sum(m.missed for m in self.minds),
            msgs_sent=self.net.sent, msgs_delivered=self.net.delivered,
            config_hash=cfg.hash(),
        )
