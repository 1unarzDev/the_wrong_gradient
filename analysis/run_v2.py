"""General runner for RoboBoat-course experiments (paired by course seed; common random numbers).

  PYTHONPATH=. python analysis/run_v2.py --out data/X --pols P3,R3 --comms degraded,ideal --seeds 2000:2200 \
      [--nboats 4] [--episode 300] [--cfg '{"coord_bonus": false}'] [--dyn '{"p_good_to_bad": 0.25}'] [--jobs 16]
Writes <out>/episodes.csv and <out>/config.json."""
import argparse, csv, dataclasses, json, math, os
from multiprocessing import Pool

from rb2.config2 import V2Config
from rb2.sim2 import Simulation2

KEYS = ["points", "task_points", "bonus", "welfare", "pts_T1", "pts_T2", "pts_T3", "pts_T4", "pts_T5", "pts_T6", "contacts",
        "home_boats", "t3_attempted", "t3_ok", "t4_done", "t5_done", "idle_frac", "found_all_t", "found_n", "found_need"]
A = None


def run(job):
    pol, seed, comm = job
    cfg = V2Config(policy=pol, comm_mode=comm, n_boats=A.nboats, episode_s=A.episode, **json.loads(A.cfg))
    if A.dyn != "{}":
        cfg = dataclasses.replace(cfg, dyn=dataclasses.replace(cfg.dyn, **json.loads(A.dyn)))
    s = Simulation2(cfg, seed)
    m = s.run()
    yields = sum(1 for e in s.events if e[1] == "yield")
    return {"policy": pol, "seed": seed, "comm": comm, **{k: m.get(k, math.nan) for k in KEYS}, "yields": yields}


def init(a):
    global A
    A = a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--pols", required=True)
    ap.add_argument("--comms", default="degraded"); ap.add_argument("--seeds", default="2000:2200")
    ap.add_argument("--nboats", type=int, default=4); ap.add_argument("--episode", type=float, default=300.0)
    ap.add_argument("--cfg", default="{}"); ap.add_argument("--dyn", default="{}"); ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    jobs = [(p, s, c) for c in a.comms.split(",") for s in range(lo, hi) for p in a.pols.split(",")]
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    with Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), "episodes to", a.out)
