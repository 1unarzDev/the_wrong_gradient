"""Mechanism check for the H1 null (exploratory, post hoc, labelled as such).

P3 drives the fleet. At every P3 decision we ALSO ask A1 (identical except graph ETA) what it
would choose from the SAME observation. If the two ETA models almost always pick the same task,
a more accurate ETA cannot change welfare much: argmax is invariant to any order-preserving
distortion of the utilities.

  python analysis/decision_agreement.py --worlds 60
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def job(seed):
    from src.config import config_from, load_yaml
    from src.policies import make_policy
    from src.sim import Simulation
    spec = load_yaml(os.path.join(ROOT, "configs/confirmatory.yaml"))
    sim = Simulation(config_from({**spec["base"], "comm_mode": "ideal", "policy": "P3"}), seed)
    shadow = make_policy("A1")
    rows = []
    real_decide = sim.policy.decide

    def capture(pol):
        box = {}
        orig = pol._choose

        def c(obs, utils, etas, cache):
            box["u"] = dict(utils)
            return orig(obs, utils, etas, cache)
        pol._choose = c
        return box

    pbox, sbox = capture(sim.policy), capture(shadow)

    def decide(obs):
        d = real_decide(obs)
        s = shadow.decide(obs)
        up, us = pbox.get("u", {}), sbox.get("u", {})
        common = sorted(set(up) & set(us))
        rho = float(np.corrcoef([up[j] for j in common], [us[j] for j in common])[0, 1]) if len(common) > 2 else np.nan
        top = sorted(up.values(), reverse=True)
        rows.append(dict(same=(d.task == s.task), n_opts=len(up), rho=rho,
                         gap=(top[0] - top[1]) if len(top) > 1 else np.nan,
                         switch_consideration=obs.current_claim is not None))
        return d
    sim.policy.decide = decide
    sim.run()
    return seed, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = list(range(20000, 20000 + a.worlds))
    with Pool(a.workers) as p:
        res = p.map(job, seeds)
    per_world = [np.mean([r["same"] for r in rows]) for _, rows in res if rows]
    allrows = [r for _, rows in res for r in rows]
    same = np.array([r["same"] for r in allrows])
    gap = np.array([r["gap"] for r in allrows], float)
    rho = np.array([r["rho"] for r in allrows], float)
    out = dict(worlds=len(seeds), decisions=len(allrows),
               agreement_overall=float(same.mean()),
               agreement_per_world_mean=float(np.mean(per_world)),
               agreement_per_world_min=float(np.min(per_world)),
               agreement_when_top2_gap_lt_1=float(same[gap < 1].mean()) if (gap < 1).any() else None,
               agreement_when_top2_gap_ge_3=float(same[gap >= 3].mean()) if (gap >= 3).any() else None,
               median_utility_correlation=float(np.nanmedian(rho)),
               note="exploratory/post hoc; decisions nested in worlds; P3 trajectory, A1 counterfactual choice")
    os.makedirs(os.path.join(ROOT, "data/exploratory"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "data/exploratory/decision_agreement.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
