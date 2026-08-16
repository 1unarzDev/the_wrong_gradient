"""Broadcast communication with pre-generated loss/latency tapes.

Each boat broadcasts a status message every msg_period_s.  For every directed link
(sender -> receiver) and every broadcast slot the tape stores (dropped?, latency).
Loss follows a Gilbert-Elliott two-state Markov chain (bursty outages).  Tapes come
from the 'comms' stream and are generated for the whole fleet before the episode,
so every policy sees exactly the same packet fate (common random numbers).

Modes: ideal (no loss, no latency), degraded (bursty loss + latency), none (all lost).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .config import SimConfig
from .rng import stream


@dataclass(frozen=True)
class Message:
    sender: int
    t_sent: float
    x: float
    y: float
    psi: float
    v: float
    claim: Optional[int]              # task id the sender intends to service
    claim_arrival: float              # sender's own estimate of absolute completion time
    completed: FrozenSet[int]         # tasks the sender knows are done
    assign: tuple = ()                # P5 only: leader's assignment ((boat, task), ...)


class CommTape:
    def __init__(self, seed: int, cfg: SimConfig):
        self.cfg = cfg
        n = cfg.n_boats
        slots = int(math.ceil(cfg.episode_s / cfg.msg_period_s)) + 2
        rng = stream(seed, "comms")
        # always draw the same amount of randomness regardless of mode
        u_state = rng.random((slots, n, n))
        u_drop = rng.random((slots, n, n))
        u_lat = rng.random((slots, n, n))
        if cfg.comm_mode == "ideal":
            self.drop = np.zeros((slots, n, n), bool)
            self.latency = np.zeros((slots, n, n))
        elif cfg.comm_mode == "none":
            self.drop = np.ones((slots, n, n), bool)
            self.latency = np.zeros((slots, n, n))
        elif cfg.comm_mode == "degraded":
            bad = np.zeros((n, n), bool)
            drop = np.zeros((slots, n, n), bool)
            for s in range(slots):
                flip = np.where(bad, u_state[s] < cfg.p_bad_to_good, u_state[s] < cfg.p_good_to_bad)
                bad = bad ^ flip
                drop[s] = u_drop[s] < np.where(bad, cfg.drop_bad, cfg.drop_good)
            self.drop = drop
            self.latency = cfg.latency_min_s + (cfg.latency_max_s - cfg.latency_min_s) * u_lat
        else:
            raise ValueError(cfg.comm_mode)
        idx = np.arange(n)
        self.drop[:, idx, idx] = True              # no self-delivery

    def fate(self, slot: int, sender: int, receiver: int) -> Tuple[bool, float]:
        return bool(self.drop[slot, sender, receiver]), float(self.latency[slot, sender, receiver])


@dataclass
class Network:
    tape: CommTape
    in_flight: List[Tuple[float, int, Message]] = field(default_factory=list)  # (deliver_t, rcv, msg)
    sent: int = 0
    delivered: int = 0
    dropped_events: List[Tuple[float, int, int]] = field(default_factory=list)  # for visuals

    def broadcast(self, slot: int, msg: Message, n_boats: int):
        for r in range(n_boats):
            if r == msg.sender:
                continue
            self.sent += 1
            dropped, lat = self.tape.fate(slot, msg.sender, r)
            if dropped:
                self.dropped_events.append((msg.t_sent, msg.sender, r))
            else:
                self.in_flight.append((msg.t_sent + lat, r, msg))

    def deliver(self, t: float) -> List[Tuple[int, Message]]:
        out, keep = [], []
        for item in self.in_flight:
            (out if item[0] <= t + 1e-9 else keep).append(item)
        self.in_flight = keep
        self.delivered += len(out)
        return [(r, m) for _, r, m in out]
