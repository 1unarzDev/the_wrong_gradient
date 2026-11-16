"""Statistics for the revision-round experiments, as fixed in data/analysis_plan_round3.md.
Writes data/round3_stats.json and data/round3_stats.md."""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from stats_lib import holm, summarize_contrast  # noqa: E402

D = lambda p: pd.read_csv(os.path.join(ROOT, p))


def contrast(df, a, b, comm, outcome, seed_col="seed"):
    A = df[(df.policy == a) & (df.comm == comm)].set_index(seed_col)[outcome]
    B = df[(df.policy == b) & (df.comm == comm)].set_index(seed_col)[outcome]
    idx = A.index.intersection(B.index)
    r = summarize_contrast((A.loc[idx] - B.loc[idx]).values)
    r.update(contrast=f"{a}-{b}", comm=comm, outcome=outcome, n=len(idx))
    return r


def family(df, pairs, outcome, comm="degraded"):
    rs = [contrast(df, a, b, comm, outcome) for a, b in pairs]
    for r, q in zip(rs, holm([r["p_perm"] for r in rs])):
        r["p_holm"] = float(q)
    return rs


out, md = {}, ["# Revision-round statistics (plan: data/analysis_plan_round3.md)", ""]
pairs = [("R3", "P3"), ("R3", "A2"), ("P4", "P5"), ("P5", "P3")]
for n, base, new, lease in [(4, "data/v2_replication/episodes.csv", "data/v2_new_policies_n4/episodes.csv", "data/v2_lease_n4/episodes.csv"),
                            (2, "data/v2_two_boats/episodes.csv", "data/v2_new_policies_n2/episodes.csv", "data/v2_lease_n2/episodes.csv")]:
    df = pd.concat([D(base), D(new), D(lease)], ignore_index=True)
    for outcome in ("task_points", "points"):     # exploratory follow-up family (designed after the R3 result)
        out[f"n{n}_lease_{outcome}"] = family(df, [("L3", "P3"), ("L3", "A2")], outcome)
        out[f"n{n}_lease_ideal_{outcome}"] = family(df, [("L3", "P3"), ("L3", "A2")], outcome, comm="ideal")
    for outcome in ("task_points", "points"):
        out[f"n{n}_{outcome}"] = family(df, pairs, outcome)
    desc = df.groupby(["comm", "policy"]).agg(points=("points", "mean"), task=("task_points", "mean"),
                                             bonus=("bonus", lambda x: float(np.mean(x > 0)))).round(2)
    out[f"n{n}_desc"] = json.loads(desc.reset_index().to_json(orient="records"))
    md += [f"## {n} boats", desc.to_string(), ""]
    if "yields" in df.columns:
        yl = df[df.policy.isin(["P5", "R3", "L3"])].groupby(["comm", "policy"]).yields.mean().round(2)
        out[f"n{n}_yields"] = {f"{c}/{p}": float(v) for (c, p), v in yl.items()}
        md += ["yields per run (new policies):", yl.to_string(), ""]
# robustness: unadjusted, descriptive
for var in ("harsh_radio", "slow_boats"):
    df = D(f"data/v2_robust/{var}/episodes.csv")
    out[f"robust_{var}"] = [contrast(df, a, b, "degraded", "task_points") for a, b in [("P4", "P3"), ("A2", "P3"), ("R3", "P3")]]
    out[f"robust_{var}_desc"] = json.loads(df.groupby("policy").agg(task=("task_points", "mean"), yields=("yields", "mean"),
                                                                    bonus=("bonus", lambda x: float(np.mean(x > 0)))).round(2).reset_index().to_json(orient="records"))
# single boat: fleet-size comparison point (descriptive)
one = D("data/v2_one_boat/episodes.csv")
out["n1_desc"] = json.loads(one.groupby("policy").agg(points=("points", "mean"), task=("task_points", "mean"),
                            bonus=("bonus", lambda x: float(np.mean(x > 0))), home=("home_boats", "mean")).round(3).reset_index().to_json(orient="records"))


def fmt(r):
    return (f"| {r['contrast']} | {r.get('outcome','')} | {r['mean']:+.0f} [{r['ci_mean'][0]:+.0f}, {r['ci_mean'][1]:+.0f}] | "
            f"{r['dz']:+.2f} | {r['p_perm']:.4f} | {r.get('p_holm', float('nan')):.4f} | {100 * r['frac_positive']:.0f}% |")


for k, v in out.items():
    if isinstance(v, list) and v and "contrast" in v[0]:
        md += [f"### {k}", "| contrast | outcome | mean [95% CI, unadjusted] | d_z | p | p_Holm | wins |", "|---|---|---|---|---|---|---|"] + [fmt(r) for r in v] + [""]
    elif isinstance(v, list):
        md += [f"### {k}", pd.DataFrame(v).to_string(), ""]
json.dump(out, open(os.path.join(ROOT, "data/round3_stats.json"), "w"), indent=1, default=float)
open(os.path.join(ROOT, "data/round3_stats.md"), "w").write("\n".join(md))
print("\n".join(md))
