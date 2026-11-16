"""Ablation of the RoboBoat-specific coordination layer on fresh courses (seeds 3000+), paired by seed.
Each variant switches off one component (or all). Exploratory. Writes data/v2_ablation/{episodes.csv,summary.json}."""
import csv, json, math, os, sys
from multiprocessing import Pool

import numpy as np

from rb2.config2 import V2Config
from rb2.sim2 import Simulation2

VARIANTS = {"full": {}, "no_sectors": {"coord_sectors": False}, "no_bonus": {"coord_bonus": False},
            "no_deconflict": {"coord_deconflict": False}, "no_queue_return": {"coord_queue_return": False},
            "no_dock_infer": {"coord_dock_infer": False},
            "none": {"coord_sectors": False, "coord_bonus": False, "coord_deconflict": False,
                     "coord_queue_return": False, "coord_dock_infer": False}}
POLS = ["P3", "P4"]
KEYS = ["points", "task_points", "bonus", "home_boats", "t3_attempted", "t3_ok", "t4_done", "t5_done", "idle_frac", "contacts"]


def run(a):
    var, pol, seed = a
    m = Simulation2(V2Config(policy=pol, comm_mode="degraded", n_boats=int(os.environ.get("NBOATS", 4)), **VARIANTS[var]), seed).run()
    return {"variant": var, "policy": pol, "seed": seed, **{k: m[k] for k in KEYS}}


def signflip(d, rng, B=20000):
    perm = (rng.choice([-1, 1], (B, len(d))) * d).mean(1)
    return float((np.abs(perm) >= abs(d.mean()) - 1e-12).mean())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = os.environ.get("OUTDIR", "data/v2_ablation"); os.makedirs(out, exist_ok=True)
    jobs = [(v, p, s) for v in VARIANTS for p in POLS for s in range(3000, 3000 + n)]
    with Pool(int(os.environ.get("JOBS", 16))) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(f"{out}/episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rng = np.random.default_rng(7)
    summ = {}
    for p in POLS:
        full = {r["seed"]: r for r in rows if r["policy"] == p and r["variant"] == "full"}
        for v in VARIANTS:
            R = {r["seed"]: r for r in rows if r["policy"] == p and r["variant"] == v}
            e = {}
            for k in ("points", "task_points"):
                d = np.array([full[s][k] - R[s][k] for s in full])
                bs = rng.choice(d, (10000, len(d))).mean(1)
                e[k] = dict(mean=float(np.mean([R[s][k] for s in R])), full_minus=float(d.mean()),
                            ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            p=signflip(d, rng) if v != "full" else 1.0)
            e["bonus_rate"] = float(np.mean([R[s]["bonus"] > 0 for s in R]))
            e["home"] = float(np.mean([R[s]["home_boats"] for s in R]))
            e["t3_attempted"] = float(np.mean([R[s]["t3_attempted"] for s in R]))
            summ[f"{p}/{v}"] = e
            print(p, v, f"points {e['points']['mean']:.0f}  full-minus {e['points']['full_minus']:+.0f} {e['points']['ci']} p={e['points']['p']:.3f}"
                  f"  task {e['task_points']['full_minus']:+.0f} p={e['task_points']['p']:.3f} bonus {e['bonus_rate']:.2f}")
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
