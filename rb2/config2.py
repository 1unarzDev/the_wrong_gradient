"""v2 configuration. Wraps a src.config.SimConfig for dynamics/comms/policy knobs (reused unchanged)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from src.config import SimConfig


@dataclass
class V2Config:
    n_boats: int = 4
    episode_s: float = 300.0
    policy: str = "P3"
    comm_mode: str = "degraded"
    difficulty: str = "standard"
    variant: str = "standard"            # standard | return_rush
    shortcut_open: bool = False          # return rush: boom channel open?
    perception: str = "sensors"          # sensors | oracle (oracle = full information, for tests/baselines)
    decide_every: float = 2.0
    sense_hz: float = 5.0
    beam_radius: float = 0.55            # half-beam for buoy/structure contacts (hull radius 0.9 is for boat-boat)
    balls_per_boat: int = 3
    # RoboBoat-specific coordination layer (each can be switched off for the ablation study)
    coord_sectors: bool = True           # recon sectors: own-stripe frontier preferred
    coord_bonus: bool = True             # bonus-aware value for first attempt of an untried task type
    coord_deconflict: bool = True        # sensed local yielding on duplicate single-boat claims
    coord_queue_return: bool = True      # queue-aware return timing
    coord_dock_infer: bool = True        # bay inference by elimination / occupied-bay rule-out
    # welfare weights (same convention as v1)
    w_energy: float = 0.05               # official-points scale (a strike costs 25 in the handbook)
    w_contact: float = 25.0
    dyn: SimConfig = field(default_factory=lambda: SimConfig(episode_s=300.0, time_cost=2.0, hysteresis=30.0))

    def sim_config(self) -> SimConfig:
        return dataclasses.replace(self.dyn, n_boats=self.n_boats, episode_s=self.episode_s, policy=self.policy,
                                   comm_mode=self.comm_mode)

    def to_dict(self):
        d = dataclasses.asdict(self)
        return d

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]
