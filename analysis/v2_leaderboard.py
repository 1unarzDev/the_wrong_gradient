"""Exploratory v2 leaderboard: every strategy on the same procedurally generated courses (paired by seed),
sensor perception, degraded radio. Writes data/v2_bench/episodes.csv and leaderboard.json (used by the game panel).
Exploratory and descriptive; not part of the preregistered study."""
import csv, json, math, os, sys
from multiprocessing import Pool

import numpy as np

from rb2.config2 import V2Config
from rb2.sim2 import Simulation2

POLS = ["P0", "P1", "P2", "A1", "A2", "P3", "P4"]
KEYS = ["points", "task_points", "bonus", "welfare", "pts_T1", "pts_T2", "pts_T3", "pts_T4", "pts_T5", "pts_T6", "contacts",
        "home_boats", "t3_attempted", "t3_ok", "t4_done", "t5_done", "idle_frac", "found_all_t", "found_n", "found_need"]


def run(a):
    pol, seed, comm = a
    m = Simulation2(V2Config(policy=pol, comm_mode=comm, n_boats=int(os.environ.get("NBOATS", 4))), seed).run()
    return {"policy": pol, "seed": seed, "comm": comm, **{k: m.get(k, math.nan) for k in KEYS}}


def boot_ci(x, B=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    comms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["degraded"]
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    outdir = sys.argv[4] if len(sys.argv) > 4 else "data/v2_bench"
    os.makedirs(outdir, exist_ok=True)
    jobs = [(p, s, c) for c in comms for s in range(base, base + n) for p in POLS]
    with Pool(int(os.environ.get("JOBS", 16))) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(f"{outdir}/episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = {}
    for c in comms:
        out[c] = {}
        for p in POLS:
            R = [r for r in rows if r["policy"] == p and r["comm"] == c]
            pts = [r["points"] for r in R]
            out[c][p] = dict(n=len(R), points=float(np.mean(pts)), ci=boot_ci(pts),
                             task_points=float(np.mean([r["task_points"] for r in R])),
                             bonus_rate=float(np.mean([r["bonus"] > 0 for r in R])),
                             home=float(np.mean([r["home_boats"] for r in R])),
                             idle=float(np.mean([r["idle_frac"] for r in R])),
                             contacts=float(np.mean([r["contacts"] for r in R])))
    json.dump(out, open(f"{outdir}/leaderboard.json", "w"), indent=1)
    for c in comms:
        print(c)
        for p, d in sorted(out[c].items(), key=lambda kv: -kv[1]["points"]):
            print(f"  {p}: {d['points']:.0f} [{d['ci'][0]:.0f},{d['ci'][1]:.0f}] task {d['task_points']:.0f} bonus {d['bonus_rate']:.2f} home {d['home']:.2f} idle {d['idle']:.2f}")
