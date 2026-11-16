"""Additional strategies for the RoboBoat course (the original P0-P5 live, frozen, in src/policies.py).

R3  Reservation-based marginal contribution (control-aware ETA).
    A teammate's claim is a *reservation* that is trusted in full until the claimant's own announced completion
    time plus a margin, and only then expires (decays). The age-aware rule of P3/A1 instead discounts a claim as
    soon as the message ages, even when the claimant is still on schedule, which under-charges for a task that is
    in fact taken (see the paper). Everything else is identical to P3/A2.
"""
from __future__ import annotations

import math

from src.policies import MarginalContribution, norm_cdf


class ReservationMC(MarginalContribution):
    def __init__(self, margin_s: float = 5.0, margin_frac: float = 0.25, tau_s: float = 6.0):
        super().__init__("control", False, "R3")
        self.margin_s, self.margin_frac, self.tau_s = margin_s, margin_frac, tau_s

    def others_success(self, obs, tv):
        cfg = obs.cfg
        miss, max_age = 1.0, 0.0
        for k, (m, age) in obs.beliefs.items():
            if m.claim != tv.id or tv.id in m.completed:
                continue
            if (cfg.mc_priority and obs.current_claim == tv.id
                    and m.claim_arrival >= obs.t + self._my_eta.get(tv.id, math.inf)):
                continue
            max_age = max(max_age, age)
            planned = m.claim_arrival - m.t_sent
            expiry = m.claim_arrival + max(self.margin_s, self.margin_frac * planned)
            w = 1.0 if obs.t <= expiry else math.exp(-(obs.t - expiry) / self.tau_s)
            sig = cfg.sigma0_s + cfg.sigma_frac * planned
            s_k = norm_cdf((tv.deadline - m.claim_arrival) / sig) * w
            miss *= (1.0 - s_k)
        return 1.0 - miss, max_age


class LeaseMC(MarginalContribution):
    """L3: heartbeat lease. Every status message re-asserts the sender's claim, so a claim stays valid as long as the
    claimant keeps talking; it decays only after a radio silence longer than `lease_s` (longer than a typical loss
    burst). Unlike P3/A1 (age and overdue discounting) and R3 (expiry at the announced arrival), lateness alone
    never weakens a claim: announced arrivals are times to REACH a task, not to finish it."""

    def __init__(self, lease_s: float = 8.0, tau_s: float = 6.0):
        super().__init__("control", False, "L3")
        self.lease_s, self.tau_s = lease_s, tau_s

    def others_success(self, obs, tv):
        cfg = obs.cfg
        miss, max_age = 1.0, 0.0
        for k, (m, age) in obs.beliefs.items():
            if m.claim != tv.id or tv.id in m.completed:
                continue
            if (cfg.mc_priority and obs.current_claim == tv.id
                    and m.claim_arrival >= obs.t + self._my_eta.get(tv.id, math.inf)):
                continue
            max_age = max(max_age, age)
            planned = m.claim_arrival - m.t_sent
            w = 1.0 if age <= self.lease_s else math.exp(-(age - self.lease_s) / self.tau_s)
            sig = cfg.sigma0_s + cfg.sigma_frac * planned
            s_k = norm_cdf((tv.deadline - m.claim_arrival) / sig) * w
            miss *= (1.0 - s_k)
        return 1.0 - miss, max_age


def make_policy2(name):
    if name == "R3":
        return ReservationMC()
    if name == "L3":
        return LeaseMC()
    return None
