"""Simulation-based power / precision analysis from PILOT data only.

For candidate N (worlds), repeatedly resample N pilot WORLDS with replacement (jointly,
so the H1 and H2 differences of a world stay together), shift each contrast's differences
so that its true mean equals a planning effect (d_z * sd), run the pre-specified
two-sided sign-flip test on each, Holm-adjust across {H1, H2}, and count rejections at
FWER 0.05.  Also reports the expected 95% CI half-width.

  python analysis/power.py --pilot data/pilot/final/episodes.csv --out data/power.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_lib import holm, sign_flip_test  # noqa: E402

H1 = dict(name="H1", a="P3", b="A1", condition="ideal")
H2 = dict(name="H2", a="P3", b="A2", condition="degraded")


def pilot_diffs(path):
    df = pd.read_csv(path)
    w = df.pivot_table(index=["condition", "seed"], columns="policy", values="welfare")
    out = {}
    for h in (H1, H2):
        x = w.loc[h["condition"]]
        out[h["name"]] = (x[h["a"]] - x[h["b"]]).sort_index()
    seeds = sorted(set(out["H1"].index) & set(out["H2"].index))
    return {k: v.loc[seeds].to_numpy() for k, v in out.items()}, seeds


def power(diffs, N, dz_plan=None, reps=600, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    n0 = len(diffs["H1"])
    rej = {"H1": 0, "H2": 0, "any": 0}
    for r in range(reps):
        idx = rng.integers(0, n0, N)
        ps = []
        for k in ("H1", "H2"):
            d = diffs[k][idx].astype(float)
            base = diffs[k]
            if dz_plan is not None:
                d = d - base.mean() + dz_plan * base.std(ddof=1)   # impose planning effect
            ps.append(sign_flip_test(d, n_perm, seed=r * 7 + len(ps)))
        adj = holm(ps)
        rej["H1"] += adj[0] < 0.05
        rej["H2"] += adj[1] < 0.05
        rej["any"] += min(adj) < 0.05
    return {k: v / reps for k, v in rej.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/pilot/final/episodes.csv")
    ap.add_argument("--out", default="data/power.json")
    ap.add_argument("--reps", type=int, default=600)
    a = ap.parse_args()
    diffs, seeds = pilot_diffs(a.pilot)
    res = dict(pilot=a.pilot, n_pilot_worlds=len(seeds), contrasts={})
    for k, d in diffs.items():
        res["contrasts"][k] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                   dz=float(d.mean() / d.std(ddof=1)),
                                   se_dz_approx=float(np.sqrt(1 / len(d) + (d.mean() / d.std(ddof=1)) ** 2 / (2 * len(d)))))
    grid = [100, 200, 240, 300, 400, 500]
    res["power_planning_dz_0.25"] = {}
    res["power_pilot_effect_as_is"] = {}
    res["ci_halfwidth_95"] = {}
    for N in grid:
        res["power_planning_dz_0.25"][N] = power(diffs, N, 0.25, reps=a.reps)
        res["power_pilot_effect_as_is"][N] = power(diffs, N, None, reps=a.reps)
        res["ci_halfwidth_95"][N] = {k: float(1.96 * d.std(ddof=1) / np.sqrt(N)) for k, d in diffs.items()}
        print(N, "dz=0.25:", res["power_planning_dz_0.25"][N], " pilot-as-is:", res["power_pilot_effect_as_is"][N],
              " CI half-width:", {k: round(v, 2) for k, v in res["ci_halfwidth_95"][N].items()}, flush=True)
    # decision rule (frozen before confirmatory data): smallest N in grid with power >= 0.90 for
    # BOTH contrasts at planning d_z = 0.25 under Holm, AND CI half-width <= 2.0 points for H1.
    ok = [N for N in grid if min(res["power_planning_dz_0.25"][N]["H1"], res["power_planning_dz_0.25"][N]["H2"]) >= 0.90
          and res["ci_halfwidth_95"][N]["H1"] <= 2.0]
    res["chosen_N"] = int(ok[0]) if ok else int(grid[-1])
    res["rule"] = ("smallest N in {100,200,240,300,400,500} with Holm-adjusted power >= 0.90 for both H1 and H2 "
                   "at planning effect d_z = 0.25, and 95% CI half-width <= 2.0 welfare points for H1")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("pilot contrasts:", json.dumps(res["contrasts"], indent=1))
    print("chosen N =", res["chosen_N"])


if __name__ == "__main__":
    main()
