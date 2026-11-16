"""Stats for the v2 (RoboBoat course) replication, following data/v2_replication/analysis_plan.md.
Writes data/v2_replication/stats.json and stats.md."""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from stats_lib import holm, summarize_contrast  # noqa: E402

FAMILY = [("P3", "A1", "ideal"), ("P3", "A2", "degraded"), ("P4", "P3", "ideal"), ("P4", "P3", "degraded"),
          ("P1", "P3", "ideal"), ("P1", "P3", "degraded")]

ep = pd.read_csv(os.path.join(ROOT, os.environ.get("V2DIR", "data/v2_replication"), "episodes.csv"))
out, lines = {}, ["# v2 replication statistics (200 fresh courses; exploratory, family fixed in advance)", ""]
for outcome in ("points", "task_points"):
    res = []
    for a, b, c in FAMILY:
        A = ep[(ep.policy == a) & (ep.comm == c)].set_index("seed")[outcome]
        B = ep[(ep.policy == b) & (ep.comm == c)].set_index("seed")[outcome]
        d = (A - B.reindex(A.index)).values
        res.append(dict(contrast=f"{a}-{b}", comm=c, **summarize_contrast(d)))
    adj = holm([r["p_perm"] for r in res])
    for r, q in zip(res, adj):
        r["p_holm"] = float(q)
    out[outcome] = res
    lines += [f"## {outcome}", "", "| contrast | radio | mean [95% CI] | d_z [95% CI] | Holm p | wins |", "|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['contrast']} | {r['comm']} | {r['mean']:+.0f} [{r['ci_mean'][0]:+.0f}, {r['ci_mean'][1]:+.0f}] | "
                     f"{r['dz']:+.2f} [{r['ci_dz'][0]:+.2f}, {r['ci_dz'][1]:+.2f}] | {r['p_holm']:.3g} | {100 * r['frac_positive']:.0f}% |")
    lines.append("")
desc = ep.groupby(["comm", "policy"]).agg(points=("points", "mean"), task=("task_points", "mean"),
                                         bonus_rate=("bonus", lambda x: float(np.mean(x > 0))),
                                         home=("home_boats", "mean"), t3=("t3_attempted", "mean"),
                                         t4=("t4_done", "mean"), t5=("t5_done", "mean"), idle=("idle_frac", "mean"),
                                         contacts=("contacts", "mean"), found_all=("found_all_t", "mean"))
out["descriptive"] = json.loads(desc.reset_index().to_json(orient="records"))
lines += ["## descriptive", "", desc.round(2).to_string()]
json.dump(out, open(os.path.join(ROOT, os.environ.get("V2DIR", "data/v2_replication"), "stats.json"), "w"), indent=1)
open(os.path.join(ROOT, os.environ.get("V2DIR", "data/v2_replication"), "stats.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
