"""Masked centralized assignment for course v2.

`src.policies.CentralizedHungarian.decide_all` (P4/P5, frozen with the v1 study) scores every boat against every task.
On course v2 some opportunities are personal (each boat's own T1 channel pass and its own return home), and a boat
may only take tasks its own knowledge supports. This wrapper applies exactly the same utility
    U = value x P(success by deadline) - time_cost x ETA  (+ hysteresis for the current claim)
with the same control-aware ETA, but sets U = -inf where a task is not allowed for that boat, then solves the same
linear assignment (with idle columns). With an all-True mask it reproduces decide_all.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.policies import Decision, control_eta, success_prob


def masked_decide_all(cobs, allowed):
    """allowed[i][jj] -> bool. Returns one Decision per boat in cobs.states."""
    cfg = cobs.cfg
    n, m = len(cobs.states), len(cobs.tasks)
    t0 = time.perf_counter()
    if m == 0:
        return [Decision(None, 0.0, math.inf) for _ in range(n)]
    U = np.full((n, m), -1e9)
    E = np.full((n, m), math.inf)
    W = {}
    for i, st in enumerate(cobs.states):
        for jj, tv in enumerate(cobs.tasks):
            if not allowed[i][jj]:
                continue
            eta, wps = control_eta(cobs.world, st, tv, cfg, cobs.tables)
            if not math.isfinite(eta):
                continue
            E[i, jj] = eta; W[(i, jj)] = wps
            U[i, jj] = tv.value * success_prob(cobs.t, eta, tv.deadline, cfg) - cfg.time_cost * eta
            if cobs.current_claims[i] == tv.id:
                U[i, jj] += cfg.hysteresis
    Upad = np.hstack([U, np.zeros((n, n))])
    rows, cols = linear_sum_assignment(-Upad)
    out = [Decision(None, 0.0, math.inf) for _ in range(n)]
    cpu = (time.perf_counter() - t0) / n
    for i, jj in zip(rows, cols):
        if jj < m and U[i, jj] > 0:
            out[i] = Decision(cobs.tasks[jj].id, float(U[i, jj]), float(E[i, jj]), wps=W[(i, jj)])
            # for the interface: every other boat's ETA to this task (who else could have taken it)
            out[i].bids = sorted((float(E[k, jj]), k) for k in range(n) if k != i and math.isfinite(E[k, jj]))
        out[i].cpu_s = cpu
    return out
