import os, sys, numpy as np
from multiprocessing import Pool
from rb2.config2 import V2Config; from rb2.sim2 import Simulation2
def f(a):
    pol, seed = a
    s = Simulation2(V2Config(policy=pol, comm_mode="degraded", n_boats=int(os.environ.get("NBOATS", 4))), seed); s.run()
    ylds = sum(1 for e in s.events if e[1] == "yield")
    # duplicate claim-seconds: time two boats hold the same non-personal claim
    return pol, seed, ylds
if __name__ == "__main__":
    jobs = [(p, s) for p in ["P3", "A2", "P1", "P4"] for s in range(2000, 2030)]
    with Pool(6) as pool: R = pool.map(f, jobs)
    for p in ["P3", "A2", "P1", "P4"]:
        y = [r[2] for r in R if r[0] == p]; print(p, "yields/run %.2f" % np.mean(y))
