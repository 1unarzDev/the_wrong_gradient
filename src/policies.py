"""Task-allocation policies and the information boundary.

Decentralised policies receive ONLY a BoatObservation (own state, public chart,
released-and-not-known-done tasks, the messages actually received with their ages).
They never see the Simulation object, other boats' true states, or true task status.

Policy family (one marginal-contribution engine, 2x2 factorial in its two ingredients):
  P0  greedy nearest task (graph distance), ignores teammates and deadlines-feasibility
  P1  selfish best response: private credit (I score only if I arrive first), graph ETA
  P2  marginal contribution (difference reward), graph ETA,   naive beliefs
  A1  marginal contribution,                     graph ETA,   age-aware beliefs   (H1 comparator)
  A2  marginal contribution,                     control ETA, naive beliefs       (H2 comparator)
  P3  marginal contribution,                     control ETA, age-aware beliefs   (candidate)
  P4  centralised rolling-horizon assignment (Hungarian), true states, control ETA
      -- a full-information REFERENCE, not an upper bound.

Marginal contribution of boat i to task j under independent success events:
    W(with i) - W(without i) = v_j [1-(1-s_i)(1-s_-i)] - v_j s_-i = v_j s_i (1 - s_-i)
    U_i(j) = v_j * s_i(j) * (1 - s_-i(j)) - lambda * ETA_i(j)
with s = Phi((deadline - predicted completion) / sigma), sigma = sigma0 + sigma_frac*ETA.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .comms import Message
from .config import SimConfig
from .navigation import PenaltyTables, control_eta, graph_eta

SQRT2 = math.sqrt(2.0)


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


@dataclass(frozen=True)
class TaskView:
    """Public, static description of a released task (no completion truth)."""
    id: int
    kind: str
    x: float
    y: float
    value: int
    release: float
    deadline: float
    points: Tuple[Tuple[float, float], ...]
    service_s: float
    stop: bool
    extra_len: float


@dataclass
class BoatObservation:
    boat: int
    t: float
    state: Tuple[float, float, float, float]          # own x, y, psi, v
    tasks: List[TaskView]                             # released, unexpired, not known done
    beliefs: Dict[int, Tuple[Message, float]]         # teammate -> (last msg, age)
    current_claim: Optional[int]
    world: object                                     # static map + CHARTED current only
    cfg: SimConfig
    tables: PenaltyTables


@dataclass
class Decision:
    task: Optional[int]
    utility: float
    eta: float                     # policy's own ETA estimate (s)
    eta_graph: float = math.nan    # both models logged for ETA-error analysis
    eta_control: float = math.nan
    wps: Optional[list] = None
    cpu_s: float = 0.0
    max_age_used: float = 0.0
    fallbacks: int = 0


class _ETACache:
    def __init__(self, obs: BoatObservation, model: str, budget_s: float = 0.25):
        self.obs, self.model, self.budget = obs, model, budget_s
        self.t0 = time.perf_counter()
        self.c: Dict[int, Tuple[float, Optional[list]]] = {}
        self.fallbacks = 0

    def __call__(self, tv: TaskView) -> Tuple[float, Optional[list]]:
        if tv.id in self.c:
            return self.c[tv.id]
        o = self.obs
        if self.model == "graph" or time.perf_counter() - self.t0 > self.budget:
            if self.model != "graph":
                self.fallbacks += 1        # compute budget exceeded -> graph fallback
            r = (graph_eta(o.world, o.state[:2], tv, o.cfg), None)
        else:
            r = control_eta(o.world, o.state, tv, o.cfg, o.tables)
        self.c[tv.id] = r
        return r


def success_prob(t_now, eta, deadline, cfg: SimConfig, extra_sigma=0.0) -> float:
    if not math.isfinite(eta):
        return 0.0
    sigma = cfg.sigma0_s + cfg.sigma_frac * eta + extra_sigma
    return norm_cdf((deadline - (t_now + eta)) / sigma)


class Policy:
    name = "base"
    centralized = False

    def decide(self, obs: BoatObservation) -> Decision:
        raise NotImplementedError

    def _choose(self, obs, utils: Dict[int, float], etas, cache) -> Decision:
        """Pick argmax utility with hysteresis toward the current claim."""
        cfg = obs.cfg
        if not utils:
            return Decision(None, 0.0, math.inf)
        best = max(utils, key=lambda j: (utils[j], -j))
        cur = obs.current_claim
        if cur in utils and utils[best] < utils[cur] + cfg.hysteresis:
            best = cur
        if utils[best] <= 0:
            return Decision(None, 0.0, math.inf)
        eta, wps = etas[best]
        return Decision(best, utils[best], eta, wps=wps, fallbacks=cache.fallbacks)


class GreedyNearest(Policy):
    name = "P0"

    def decide(self, obs):
        cache = _ETACache(obs, "graph")
        etas = {tv.id: cache(tv) for tv in obs.tasks}
        # utility = negative distance (as time); keep positive for _choose
        utils = {j: 1000.0 - e for j, (e, _) in etas.items() if math.isfinite(e)}
        d = self._choose(obs, utils, etas, cache)
        return d


class SelfishRace(Policy):
    """Private payoff: I am credited v_j only if I complete before the deadline AND
    before every teammate I believe is heading there (taken at face value)."""
    name = "P1"

    def decide(self, obs):
        cfg = obs.cfg
        cache = _ETACache(obs, "graph")
        utils, etas = {}, {}
        for tv in obs.tasks:
            eta, wps = cache(tv)
            etas[tv.id] = (eta, wps)
            if not math.isfinite(eta):
                continue
            s_i = success_prob(obs.t, eta, tv.deadline, cfg)
            my_arrival = obs.t + eta
            win = 1.0
            for k, (m, age) in obs.beliefs.items():
                if m.claim == tv.id and tv.id not in m.completed and m.claim_arrival <= my_arrival:
                    win = 0.0
            utils[tv.id] = tv.value * s_i * win - cfg.time_cost * eta
        return self._choose(obs, utils, etas, cache)


class MarginalContribution(Policy):
    def __init__(self, eta_model: str, age_aware: bool, name: str):
        self.eta_model, self.age_aware, self.name = eta_model, age_aware, name

    def others_success(self, obs, tv: TaskView) -> Tuple[float, float]:
        """P(some teammate believed to be on task completes it before the deadline)."""
        cfg = obs.cfg
        miss, max_age = 1.0, 0.0
        for k, (m, age) in obs.beliefs.items():
            if m.claim != tv.id or tv.id in m.completed:
                continue
            if (cfg.mc_priority and obs.current_claim == tv.id
                    and m.claim_arrival >= obs.t + self._my_eta.get(tv.id, math.inf)):
                continue        # I hold j and I am predicted to arrive first: they yield
            max_age = max(max_age, age)
            planned = m.claim_arrival - m.t_sent          # teammate's own ETA when sent
            if self.age_aware:
                # stale claims lose credibility; overdue claims (should be done by now,
                # yet no completion news) are probably obsolete; uncertainty grows with age
                overdue = max(0.0, obs.t - m.claim_arrival - cfg.age_grace_s)
                w = math.exp(-max(0.0, age - cfg.age_grace_s) / cfg.age_tau_s) * \
                    math.exp(-overdue / cfg.age_tau_s)
                sig = cfg.sigma0_s + cfg.sigma_frac * planned + age
            else:
                w = 1.0
                sig = cfg.sigma0_s + cfg.sigma_frac * planned
            s_k = norm_cdf((tv.deadline - m.claim_arrival) / sig) * w
            miss *= (1.0 - s_k)
        return 1.0 - miss, max_age

    def decide(self, obs):
        cfg = obs.cfg
        cache = _ETACache(obs, self.eta_model)
        utils, etas, max_age = {}, {}, 0.0
        self._my_eta = {}
        for tv in obs.tasks:
            eta, wps = cache(tv)
            etas[tv.id] = (eta, wps)
            self._my_eta[tv.id] = eta
            if not math.isfinite(eta):
                continue
            s_i = success_prob(obs.t, eta, tv.deadline, cfg)
            s_o, a = self.others_success(obs, tv)
            max_age = max(max_age, a)
            utils[tv.id] = tv.value * s_i * (1.0 - s_o) - cfg.time_cost * eta
        d = self._choose(obs, utils, etas, cache)
        d.max_age_used = max_age
        return d


@dataclass
class CentralObservation:
    """Full-information snapshot for the centralised reference (labelled as such)."""
    t: float
    states: List[Tuple[float, float, float, float]]
    tasks: List[TaskView]                 # truly active tasks
    current_claims: List[Optional[int]]
    world: object
    cfg: SimConfig
    tables: PenaltyTables


class CentralizedHungarian(Policy):
    name = "P4"
    centralized = True

    def decide_all(self, cobs: CentralObservation) -> List[Decision]:
        cfg = cobs.cfg
        n, m = len(cobs.states), len(cobs.tasks)
        t0 = time.perf_counter()
        if m == 0:
            return [Decision(None, 0.0, math.inf) for _ in range(n)]
        U = np.full((n, m), -1e3)
        E = np.full((n, m), math.inf)
        W: Dict[Tuple[int, int], list] = {}
        for i, st in enumerate(cobs.states):
            for jj, tv in enumerate(cobs.tasks):
                eta, wps = control_eta(cobs.world, st, tv, cfg, cobs.tables)
                if not math.isfinite(eta):
                    continue
                E[i, jj] = eta; W[(i, jj)] = wps
                U[i, jj] = tv.value * success_prob(cobs.t, eta, tv.deadline, cfg) - cfg.time_cost * eta
                if cobs.current_claims[i] == tv.id:
                    U[i, jj] += cfg.hysteresis
        # pad with idle columns (utility 0) so boats may stay idle
        Upad = np.hstack([U, np.zeros((n, n))])
        rows, cols = linear_sum_assignment(-Upad)
        out = [Decision(None, 0.0, math.inf) for _ in range(n)]
        cpu = (time.perf_counter() - t0) / n
        for i, jj in zip(rows, cols):
            if jj < m and U[i, jj] > 0:
                out[i] = Decision(cobs.tasks[jj].id, float(U[i, jj]), float(E[i, jj]),
                                  wps=W[(i, jj)])
            out[i].cpu_s = cpu
        return out


class LeaderAssignment(CentralizedHungarian):
    """P5 (post-hoc follow-up): the SAME centralized Hungarian solver, but run by a leader boat
    (boat 1) on what it has HEARD over the lossy radio, and its assignments reach the other boats
    only through that radio. Separates coordination structure from full information."""
    name = "P5"
    leader = True


POLICY_INFO = {
    "P0": "greedy nearest (graph distance)",
    "P1": "selfish race best response (graph ETA)",
    "P2": "marginal contribution, graph ETA, naive beliefs",
    "A1": "marginal contribution, graph ETA, age-aware beliefs",
    "A2": "marginal contribution, control ETA, naive beliefs",
    "P3": "marginal contribution, control ETA, age-aware beliefs",
    "P4": "centralised Hungarian, full information, control ETA (reference)",
    "P5": "leader-boat Hungarian on radio beliefs, commands over radio",
}


def make_policy(name: str) -> Policy:
    if name == "P0":
        return GreedyNearest()
    if name == "P1":
        return SelfishRace()
    if name == "P2":
        return MarginalContribution("graph", False, "P2")
    if name == "A1":
        return MarginalContribution("graph", True, "A1")
    if name == "A2":
        return MarginalContribution("control", False, "A2")
    if name == "P3":
        return MarginalContribution("control", True, "P3")
    if name == "P4":
        return CentralizedHungarian()
    if name == "P5":
        return LeaderAssignment()
    raise KeyError(name)
