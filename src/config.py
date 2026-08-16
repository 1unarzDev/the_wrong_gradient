"""Simulation configuration. One flat dataclass so YAML overrides are trivial and the
whole configuration can be hashed for results integrity."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class SimConfig:
    # --- fleet / episode ------------------------------------------------------------
    n_boats: int = 4
    episode_s: float = 240.0
    dt: float = 0.05                 # physics step (RK4)
    control_every: int = 2           # guidance/control at dt*control_every (10 Hz)
    replan_period_s: float = 2.0     # every boat re-decides this often (staggered)

    # --- arena (scaled with sqrt(n_boats/4) to keep density fixed) ------------------
    base_width_m: float = 200.0
    base_height_m: float = 130.0
    scale_arena_with_fleet: bool = True
    grid_res_m: float = 1.0
    n_islands: int = 5
    barrier: bool = True             # rocky spine with narrow channels (bottleneck)
    n_channels: int = 2              # channels per barrier (scaled with fleet)
    channel_width_m: float = 8.0
    current_strength: float = 0.35   # m/s peak of the mean (charted) current
    current_unknown_frac: float = 0.35  # amplitude of unmodelled time-varying part

    # --- tasks ------------------------------------------------------------------------
    tasks_per_boat: float = 3.0
    initial_task_frac: float = 0.5
    release_window_frac: float = 0.55
    deadline_min_s: float = 70.0
    deadline_max_s: float = 150.0
    value_min: int = 5
    value_max: int = 15
    task_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "gate": 0.35, "inspect": 0.25, "dock": 0.2, "sample": 0.2})

    # --- vessel (twin-thruster catamaran, ~1.8 m long; simplified model) -------------
    hull_radius_m: float = 0.9
    v_max: float = 2.2               # commanded-speed scale (m/s) at full symmetric thrust
    r_max: float = 0.9               # yaw-rate scale (rad/s) at full differential thrust
    tau_v: float = 1.6               # surge first-order lag (s)
    tau_r: float = 0.55              # yaw-rate first-order lag (s)
    k_drag: float = 0.04             # quadratic drag coefficient (1/m)
    disturb_sigma_v: float = 0.04    # OU disturbance std on surge (m/s^2 scale)
    disturb_sigma_r: float = 0.03    # OU disturbance std on yaw accel
    disturb_tau: float = 3.0

    # --- guidance & control -----------------------------------------------------------
    inflate_margin_m: float = 1.4    # planning clearance beyond hull radius
    los_lookahead_m: float = 4.5
    cruise_frac: float = 0.92        # cruise speed command as fraction of v_max
    k_heading: float = 1.8           # heading P gain -> yaw-rate command
    k_rate: float = 0.6              # yaw-rate feedback gain
    turn_slow_rad: float = 0.9       # heading error at which speed command reaches ~0
    sensor_range_m: float = 16.0     # local sensing (other boats, task status)
    sensor_noise_m: float = 0.3
    avoid_tcpa_s: float = 6.0
    avoid_dcpa_m: float = 4.5

    # --- communications ---------------------------------------------------------------
    comm_mode: str = "degraded"      # ideal | degraded | none
    msg_period_s: float = 1.0
    p_good_to_bad: float = 0.12      # Gilbert-Elliott burst loss (per broadcast slot)
    p_bad_to_good: float = 0.18
    drop_good: float = 0.15
    drop_bad: float = 0.9
    latency_min_s: float = 0.5
    latency_max_s: float = 3.0

    # --- policy ---------------------------------------------------------------------
    policy: str = "P3"               # P0 P1 P2 P3 P4 A1 A2 (see policies.py)
    time_cost: float = 0.04          # points per second of travel in utilities
    hysteresis: float = 0.6          # switch only if utility gain exceeds this
    sigma0_s: float = 4.0            # ETA uncertainty: sigma = sigma0 + sigma_frac*ETA
    sigma_frac: float = 0.12
    age_tau_s: float = 6.0           # age-aware belief decay time constant
    age_grace_s: float = 1.5
    mc_priority: bool = True         # holder counts only teammates predicted to arrive first
    graph_speed: float = 1.75        # effective speed for graph ETA (pilot-calibrated)

    # --- welfare (FROZEN in preregistration) ------------------------------------------
    w_energy: float = 0.01           # points per unit of integral (TL^2+TR^2) dt
    w_collision: float = 5.0         # points per boat-boat or boat-obstacle contact

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "SimConfig":
        return dataclasses.replace(self, **kw)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_from(overrides: Optional[Dict[str, Any]] = None) -> SimConfig:
    cfg = SimConfig()
    if overrides:
        valid = {f.name for f in dataclasses.fields(SimConfig)}
        bad = set(overrides) - valid
        if bad:
            raise KeyError(f"unknown config keys: {sorted(bad)}")
        cfg = cfg.replace(**overrides)
    return cfg
