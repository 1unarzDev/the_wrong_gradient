"""Planar underactuated twin-thruster USV (simplified, NOT a validated hull model).

State per boat s = (x, y, psi, v, r).  Inputs: thruster commands T_L, T_R in [-1, 1].
    v_cmd = v_max (T_L + T_R)/2          r_cmd = r_max (T_R - T_L)/2
    x'   = v cos psi + u_c(x,y,t)
    y'   = v sin psi + v_c(x,y,t)
    psi' = r
    v'   = (v_cmd - v)/tau_v - k_d v|v| + w_v(t)
    r'   = (r_cmd - r)/tau_r + w_r(t)
Because |T| <= 1, (v_cmd/v_max) and (r_cmd/r_max) lie in the diamond |a|+|b| <= 1:
turning hard *costs* forward speed. w_v, w_r are exogenous Ornstein-Uhlenbeck
disturbances pre-generated per world (common random numbers across policies).
Integrated with classical RK4, inputs held constant over a step (zero-order hold).
"""
from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .rng import stream


def derivatives(S, TL, TR, cur_fn, t, cfg: SimConfig, wv, wr):
    x, y, psi, v, r = S
    uc, vc = cur_fn(x, y, t)
    v_cmd = cfg.v_max * 0.5 * (TL + TR)
    r_cmd = cfg.r_max * 0.5 * (TR - TL)
    return np.array([
        v * np.cos(psi) + uc,
        v * np.sin(psi) + vc,
        r,
        (v_cmd - v) / cfg.tau_v - cfg.k_drag * v * np.abs(v) + wv,
        (r_cmd - r) / cfg.tau_r + wr,
    ])


def rk4_step(S, TL, TR, cur_fn, t, dt, cfg, wv=0.0, wr=0.0):
    """S: (5, n) array. Returns new state (5, n)."""
    k1 = derivatives(S, TL, TR, cur_fn, t, cfg, wv, wr)
    k2 = derivatives(S + 0.5 * dt * k1, TL, TR, cur_fn, t + 0.5 * dt, cfg, wv, wr)
    k3 = derivatives(S + 0.5 * dt * k2, TL, TR, cur_fn, t + 0.5 * dt, cfg, wv, wr)
    k4 = derivatives(S + dt * k3, TL, TR, cur_fn, t + dt, cfg, wv, wr)
    S2 = S + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    S2[2] = (S2[2] + math.pi) % (2 * math.pi) - math.pi
    return S2


def disturbance_tape(seed: int, n_boats: int, n_steps: int, cfg: SimConfig) -> np.ndarray:
    """(n_steps, 2, n_boats) OU disturbances on surge and yaw acceleration, generated
    from the 'disturb' stream for a FIXED maximum fleet so boat k sees the same tape
    regardless of policy."""
    rng = stream(seed, "disturb")
    a = math.exp(-cfg.dt / cfg.disturb_tau)
    b = math.sqrt(1 - a * a)
    eps = rng.standard_normal((n_steps, 2, n_boats))
    out = np.empty_like(eps)
    sig = np.array([cfg.disturb_sigma_v, cfg.disturb_sigma_r])[:, None]
    z = eps[0] * sig
    for k in range(n_steps):
        z = a * z + b * sig * eps[k]
        out[k] = z
    return out


def allocate_thrust(v_des, r_des, v, r, cfg: SimConfig):
    """Invert the actuator map with yaw priority (keeps steering authority when
    saturated). Returns (TL, TR) in [-1, 1]."""
    common = np.clip(v_des / cfg.v_max + 0.25 * (v_des - v) / cfg.v_max, -1.0, 1.0)
    diff = np.clip(r_des / cfg.r_max + cfg.k_rate * (r_des - r) / cfg.r_max, -1.0, 1.0)
    room = 1.0 - np.abs(diff)
    common = np.clip(common, -room, room)
    return common - diff, common + diff
