"""Return rush (course v2): does the physical course reproduce Braess's paradox?

Regenerates every number and the figure from simulation (no hand-entered numbers):

  .venv/bin/python analysis/return_rush.py            # full pipeline (cached stages are reused)
  .venv/bin/python analysis/return_rush.py --force    # recompute everything
  .venv/bin/python analysis/return_rush.py --figure   # only redraw the figure from data/return_rush/results.json

Stages (outputs in data/return_rush/):
  1. sweep.json     calibration (isolated runs, k = 1..4, 6 seeds) on a grid of course knobs (west/east lane lengths,
                    no-wake speed); exact-game prediction for each; screening measurement (16 seeds) of the predicted
                    closed-NE and open-NE profiles.
  2. selection      primary geometry = largest PREDICTED paradox among geometries whose calibrated game predicts one
                    (a priori rule). Secondary = largest SCREENED measured paradox among them (post hoc selection on
                    the screening seeds; confirmed on fresh seeds). Either must still predict a paradox after the
                    16-seed final calibration, otherwise the next one in its ranking is tried.
  3. calib_*.json   final calibration of the selected geometries (16 seeds) -> Fraction costs -> RouteGame prediction.
  4. measure_*.json conditions a/b/c with common random numbers on seeds 0..99 (n = 4 boats).
  5. profiles_*.json  all 15 route-count profiles x 30 seeds: the measured profile-level game (true NE/OPT) and the
                    separability check (per-profile measured vs predicted team time).
  6. results.json   summary + paired contrasts (analysis/stats_lib.summarize_contrast), figure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

from rb2.braess2 import (ROUTES, RushGeom, aggregate_calibration, all_count_profiles, calib_job,  # noqa: E402
                         ckey, counts_team_cost, measure_job, predict, profile_game)
from stats_lib import bootstrap_ci, summarize_contrast  # noqa: E402

DATA = os.path.join(ROOT, "data", "return_rush")
FIG = os.path.join(ROOT, "figures")
N = 4
SWEEP_CAL_SEEDS = list(range(500, 506))
FINAL_CAL_SEEDS = list(range(500, 516))
SCREEN_SEEDS = list(range(1000, 1016))
MEASURE_SEEDS = list(range(0, 100))
PROFILE_SEEDS = list(range(2000, 2030))
GRID = [dict(lane_len=lw, lane_len_e=le, v_open=v)
        for lw in (3.0, 6.0) for le in (6.0, 10.0, 14.0) for v in (0.36, 0.40, 0.44, 0.48, 0.52, 0.56)]
PROCS = min(18, os.cpu_count() or 1)


# ------------------------------------------------------------------------------------------------
def jdump(obj, name):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(obj, f, indent=1, default=_enc)


def jload(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _enc(o):
    if isinstance(o, Fraction):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def gname(g: dict) -> str:
    if g.get("congestion") == "slow_zone":
        return f"SZ_a{g['alpha']:g}_v{g['v_open']:g}"
    return f"Lw{g['lane_len']:g}_Le{g['lane_len_e']:g}_v{g['v_open']:g}"


def pred_summary(P):
    """JSON-friendly prediction summary."""
    def fr(x):
        return dict(exact=str(x), value=float(x))
    return dict(
        costs_fraction=P["costs_fraction"],
        closed=dict(ne_counts=P["closed"]["ne_counts"], worst_ne=fr(P["closed"]["worst_ne"]),
                    opt=fr(P["closed"]["opt"]), poa=fr(P["closed"]["poa"])),
        open=dict(ne_counts=P["open"]["ne_counts"], worst_ne=fr(P["open"]["worst_ne"]),
                  best_ne=fr(P["open"]["best_ne"]), opt=fr(P["open"]["opt"]), opt_counts=P["open"]["opt_counts"],
                  poa=fr(P["open"]["poa"])),
        tolled=dict(ne_counts=P["tolled"]["ne_counts"], worst_true_cost=fr(P["tolled"]["worst_ne_true_cost"])),
        profiles=P["profiles"], team_cost={k: fr(v) for k, v in P["team_cost"].items()},
        paradox=P["paradox"], strong_paradox=P["strong_paradox"],
        ratio_open_closed=float(P["team_cost"]["open"] / P["team_cost"]["closed"]))


def calibrate(pool, geom: RushGeom, seeds, kmax=N):
    jobs = [(geom, r, k, s) for r in ROUTES for k in range(1, kmax + 1) for s in seeds]
    rows = pool.map(calib_job, jobs, chunksize=2)
    costs, sds, metered = aggregate_calibration(rows, kmax)
    return dict(costs=costs, sd=sds, metered_bt=metered, n_seeds=len(seeds),
                unfinished=int(sum(not r["finished"] for r in rows)),
                contacts=dict(boat=int(sum(r["contacts"]["boat"] for r in rows)),
                              structure=int(sum(r["contacts"]["structure"] for r in rows))))


def measure(pool, geom: RushGeom, conds: dict, seeds):
    """conds: name -> (counts, shortcut_open). Same seeds (start jitter, disturbance tapes) in every condition."""
    jobs = [(geom, c, s, op) for (c, op) in conds.values() for s in seeds]
    rows = pool.map(measure_job, jobs, chunksize=2)
    out, k = {}, 0
    for name in conds:
        out[name] = rows[k:k + len(seeds)]
        k += len(seeds)
    return out


# ------------------------------------------------------------------------------------------------
def stage_sweep(pool, force):
    sw = None if force else jload("sweep.json")
    if sw is not None:
        return sw
    t0 = time.time()
    entries = []
    for g in GRID:
        geom = RushGeom(**g)
        cal = calibrate(pool, geom, SWEEP_CAL_SEEDS)
        P = predict(cal["costs"], N)
        entries.append(dict(geom=g, name=gname(g), calibration=cal, prediction=pred_summary(P)))
        print(f"  sweep {gname(g)}: paradox={P['paradox']} profiles={P['profiles']} "
              f"C={ {k: float(v) for k, v in P['team_cost'].items()} }", flush=True)
    # screening: measured closed-NE vs open-NE profile on 16 seeds for every geometry where the two differ
    for e in entries:
        pr = e["prediction"]
        if SLOWZONE or pr["profiles"]["closed"] == pr["profiles"]["open"]:
            continue
        geom = RushGeom(**e["geom"])
        m = measure(pool, geom, {"closed": (pr["profiles"]["closed"], False), "open": (pr["profiles"]["open"], True)},
                    SCREEN_SEEDS)
        tc = np.array([r["total"] for r in m["closed"]]); to = np.array([r["total"] for r in m["open"]])
        e["screen"] = dict(closed_mean=float(tc.mean()), open_mean=float(to.mean()), diff_mean=float((to - tc).mean()),
                           diff_ci=bootstrap_ci(to - tc, B=5000), n=len(tc))
        print(f"  screen {e['name']}: open-closed = {e['screen']['diff_mean']:.1f} s  CI {e['screen']['diff_ci']}",
              flush=True)
    sw = dict(grid=GRID, entries=entries, seconds=time.time() - t0)
    jdump(sw, "sweep.json")
    return sw


def select(sw):
    cands = [e for e in sw["entries"] if e["prediction"]["paradox"]]
    if SLOWZONE:
        return select_slowzone(cands), []
    primary_rank = sorted(cands, key=lambda e: (-e["prediction"]["ratio_open_closed"], e["name"]))
    screened = [e for e in cands if "screen" in e]
    secondary_rank = sorted(screened, key=lambda e: (-e["screen"]["diff_mean"] / e["screen"]["closed_mean"], e["name"]))
    return primary_rank, secondary_rank


def stage_geometry(pool, entry_list, tag, force):
    """Final calibration for the first geometry in entry_list whose 16-seed calibration still predicts a paradox,
    then the measured conditions and the profile-level game."""
    cached = None if force else jload(f"geometry_{tag}.json")
    if cached is not None:
        return cached
    chosen = None
    tried = []
    for e in entry_list:
        geom = RushGeom(**e["geom"])
        cal = calibrate(pool, geom, FINAL_CAL_SEEDS)
        P = predict(cal["costs"], N)
        tried.append(dict(name=e["name"], paradox=P["paradox"]))
        print(f"  [{tag}] final calibration {e['name']}: paradox={P['paradox']} profiles={P['profiles']}", flush=True)
        if P["paradox"]:
            chosen = (e, geom, cal, P)
            break
    if chosen is None:
        out = dict(tag=tag, calibration_failed=True, tried=tried)
        jdump(out, f"geometry_{tag}.json")
        return out
    e, geom, cal, P = chosen
    pr = P["profiles"]
    conds = {"a_closed_selfish": (pr["closed"], False), "b_open_selfish": (pr["open"], True),
             "c_open_tolled": (pr["tolled"], True)}
    t0 = time.time()
    m = measure(pool, geom, conds, MEASURE_SEEDS)
    print(f"  [{tag}] measured 3 x {len(MEASURE_SEEDS)} runs in {time.time() - t0:.0f} s", flush=True)
    # profile-level game: every count profile with the shortcut open, and the no-shortcut profiles with it closed
    t0 = time.time()
    pc = {}
    for c in all_count_profiles(N, True):
        pc[("open",) + ckey(c)] = (c, True)
    for c in all_count_profiles(N, False):
        pc[("closed",) + ckey(c)] = (c, False)
    mp = measure(pool, geom, pc, PROFILE_SEEDS)
    print(f"  [{tag}] profile game {len(pc)} x {len(PROFILE_SEEDS)} runs in {time.time() - t0:.0f} s", flush=True)
    prof_rows = {"|".join(map(str, k)): v for k, v in mp.items()}
    out = dict(tag=tag, geom=e["geom"], name=e["name"], tried=tried, calibration=cal, prediction=pred_summary(P),
               measured={k: v for k, v in m.items()}, profile_runs=prof_rows,
               team_cost_pred_by_profile={"|".join(map(str, ckey(c))): float(counts_team_cost(cal["costs"], c))
                                          for c in all_count_profiles(N, True)})
    jdump(out, f"geometry_{tag}.json")
    return out


def analyze_geometry(G):
    if G.get("calibration_failed"):
        return dict(calibration_failed=True, tried=G["tried"])
    M = G["measured"]
    tot = {k: np.array([r["total"] for r in v]) for k, v in M.items()}
    mk = {k: np.array([r["makespan"] for r in v]) for k, v in M.items()}
    for k, v in M.items():
        assert [r["seed"] for r in v] == MEASURE_SEEDS
    cond = {}
    for k in M:
        cond[k] = dict(counts=M[k][0]["counts"], total_mean=float(tot[k].mean()), total_ci=bootstrap_ci(tot[k]),
                       makespan_mean=float(mk[k].mean()), makespan_ci=bootstrap_ci(mk[k]),
                       unfinished=int(sum(not r["finished"] for r in M[k])),
                       boat_contacts_per_run=float(np.mean([r["contacts"]["boat"] for r in M[k]])),
                       structure_contacts_per_run=float(np.mean([r["contacts"]["structure"] for r in M[k]])),
                       max_in_channel=max((max(r["max_in_channel"].values(), default=0) for r in M[k]), default=0),
                       max_in_zone={z: max(r.get("max_in_zone", {}).get(z, 0) for r in M[k])
                                    for z in M[k][0].get("max_in_zone", {})})
    contrasts = {}
    for a, b in (("b_open_selfish", "a_closed_selfish"), ("c_open_tolled", "b_open_selfish"),
                 ("c_open_tolled", "a_closed_selfish")):
        for metric, D in (("total", tot), ("makespan", mk)):
            d = D[a] - D[b]
            if np.all(d == 0):
                contrasts[f"{a}-{b}:{metric}"] = dict(n=len(d), mean=0.0, identical_profiles=True)
                continue
            contrasts[f"{a}-{b}:{metric}"] = summarize_contrast(d, n_perm=20_000)
    # profile-level measured game
    PR = G["profile_runs"]
    table_open, table_closed = {}, {}
    team_meas, team_ci = {}, {}
    for key, rows in PR.items():
        parts = key.split("|")
        cond_, p = parts[0], tuple(int(x) for x in parts[1:])
        per = {}
        for r in ROUTES:
            vals = [row["per_route"][r] for row in rows if r in row["per_route"]]
            if vals:
                per[r] = float(np.mean(vals))
        (table_open if cond_ == "open" else table_closed)[p] = per
        tt = np.array([row["total"] for row in rows])
        team_meas[key] = float(tt.mean()); team_ci[key] = bootstrap_ci(tt, B=5000)
    tg_open = profile_game(table_open, N, True)
    tg_closed = profile_game(table_closed, N, False)
    true_paradox = bool(tg_open["equilibria"] and tg_closed["equilibria"]
                        and tg_open["worst_eq"] > tg_closed["worst_eq"] and tg_open["worst_eq"] > tg_open["optimum"])
    pred = G["prediction"]
    pa = cond["a_closed_selfish"]["total_mean"]; pb = cond["b_open_selfish"]["total_mean"]
    dkey = "b_open_selfish-a_closed_selfish:total"
    ci = contrasts[dkey].get("ci_mean", (0.0, 0.0))
    measured_paradox = bool(contrasts[dkey]["mean"] > 0 and ci[0] > 0)
    measured_reverse = bool(contrasts[dkey]["mean"] < 0 and ci[1] < 0)
    sep = [dict(profile=k, predicted=G["team_cost_pred_by_profile"][k.split("|", 1)[1]], measured=team_meas[k],
                ci=team_ci[k]) for k in PR if k.startswith("open|")]
    return dict(name=G["name"], geom=G["geom"], tried=G["tried"], calibration=G["calibration"], prediction=pred,
                conditions=cond, contrasts=contrasts, measured_paradox=measured_paradox,
                measured_reverse=measured_reverse,
                prediction_matches=bool(measured_paradox == pred["paradox"]),
                profile_game=dict(
                    open=dict(equilibria=tg_open["equilibria"], eq_costs=tg_open["eq_costs"],
                              optimum=tg_open["optimum"], optima=tg_open["optima"]),
                    closed=dict(equilibria=tg_closed["equilibria"], eq_costs=tg_closed["eq_costs"],
                                optimum=tg_closed["optimum"], optima=tg_closed["optima"]),
                    true_paradox=true_paradox),
                separability=sep, n_seeds=len(MEASURE_SEEDS))


# ------------------------------------------------------------------------------------------------
def style_tokens():
    try:   # rb2/viz/style.py holds the v2 tokens (needs pygame); fall back to the same literal values
        from rb2.viz import style as S
        return dict(paper=S.PAPER, ink1=S.INK1, ink2=S.INK2, ink3=S.INK3, ink4=S.INK4, accent=S.FLEET[0],
                    water=S.WATER)
    except Exception:
        return dict(paper=(244, 246, 248), ink1=(36, 44, 54), ink2=(82, 94, 108), ink3=(126, 138, 151),
                    ink4=(178, 189, 199), accent=(88, 100, 170), water=(223, 231, 237))


def _verdict(A):
    if A["measured_paradox"]:
        return "paradox reproduced"
    return "reversed: shortcut helps" if A["measured_reverse"] else "not reproduced"


def _panel_team(ax, A, T, title):
    names = [("a_closed_selfish", "closed\nselfish"), ("b_open_selfish", "open\nselfish"),
             ("c_open_tolled", "open\ntolled")]
    pred = A["prediction"]["team_cost"]
    pk = {"a_closed_selfish": "closed", "b_open_selfish": "open", "c_open_tolled": "tolled"}
    for x, (k, _) in enumerate(names):
        c = A["conditions"][k]
        ax.errorbar(x + 0.13, c["total_mean"],
                    yerr=[[c["total_mean"] - c["total_ci"][0]], [c["total_ci"][1] - c["total_mean"]]],
                    fmt="o", color=T["ink1"], ms=5, capsize=3, label="measured, mean ± 95% CI" if x == 0 else None)
        ax.plot([x - 0.13], [pred[pk[k]]["value"]], marker="D", ms=5, color=T["accent"], ls="none",
                label="exact-game prediction" if x == 0 else None)
        cnt = c["counts"]
        ax.annotate(f"W{cnt['west']} E{cnt['east']} X{cnt['shortcut']}", (x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color=T["ink3"])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in names]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("team time  Σ S→T  [s]"); ax.set_title(title, loc="left")
    ax.legend(fontsize=6.8, loc="upper left")


def make_figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = {k: tuple(c / 255 for c in v) for k, v in style_tokens().items()}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": T["ink3"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["ink2"], "ytick.color": T["ink2"], "axes.grid": True, "grid.color": T["ink4"] + (0.35,),
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
        "axes.titleweight": "bold", "axes.titlecolor": T["ink1"], "legend.frameon": False,
        "figure.facecolor": T["paper"], "axes.facecolor": T["paper"], "savefig.facecolor": T["paper"]})
    os.makedirs(FIG, exist_ok=True)
    A = R["primary"]
    if A.get("calibration_failed"):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "calibration failed: no geometry's calibrated game predicts a paradox", ha="center")
        fig.savefig(os.path.join(FIG, "return_rush_results.png"), dpi=200)
        return
    B = R.get("secondary")
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.6))
    # (a) calibrated edge costs (primary)
    ax = axs[0, 0]
    cal = A["calibration"]
    ks = np.arange(1, len(cal["costs"]["SA"]) + 1)
    lab = {"SA": "S→A channel", "BT": "B→T channel", "AT": "A→T open water", "SB": "S→B open water",
           "AB": "A→B shortcut"}
    for e, col, mk in (("SA", T["ink1"], "o"), ("BT", T["ink1"], "s"), ("AT", T["ink3"], "o"), ("SB", T["ink3"], "s"),
                       ("AB", T["accent"], "D")):
        ax.errorbar(ks, cal["costs"][e], yerr=cal["sd"][e], color=col, lw=1.6, marker=mk, ms=4, capsize=2,
                    label=lab[e])
    ax.plot(ks, cal["metered_bt"], color=T["ink2"], lw=1, ls=(0, (1, 1.5)), marker="s", ms=3.5, mfc=T["paper"],
            label="B→T when reached via the shortcut")
    ax.set_xticks(ks); ax.set_xlabel("boats on the edge, k"); ax.set_ylabel("mean edge time  c_e(k)  [s]")
    ax.set_title(f"a  Calibrated edge costs, primary\n    {A['name']} (16 seeds, ± sd)", loc="left")
    ax.legend(fontsize=6.6, loc="center right")
    # (b) team time (primary)
    _panel_team(axs[0, 1], A, T, f"b  Primary (a priori choice)\n    measured: {_verdict(A)}")
    # (c) paired contrasts (primary, and secondary if present)
    ax = axs[0, 2]
    rows = [("b_open_selfish-a_closed_selfish", "open − closed\n(Braess test)"),
            ("c_open_tolled-b_open_selfish", "tolled − open"),
            ("c_open_tolled-a_closed_selfish", "tolled − closed")]
    sets = [("primary", A, T["ink1"], -0.16)] + ([("secondary", B, T["ink3"], 0.16)] if B and not B.get("calibration_failed") else [])
    for y, (key, _) in enumerate(rows):
        for nm, G, col, dy in sets:
            c = G["contrasts"][f"{key}:total"]
            ci = c.get("ci_mean", (0.0, 0.0))
            ax.errorbar(c["mean"], y + dy, xerr=[[c["mean"] - ci[0]], [ci[1] - c["mean"]]], fmt="o", color=col,
                        ms=4.5, capsize=3, label=f"{nm} measured" if y == 0 else None)
            if key.startswith("b_"):
                pr = G["prediction"]["team_cost"]
                ax.plot([pr["open"]["value"] - pr["closed"]["value"]], [y + dy], marker="D", ms=5, ls="none",
                        color=T["accent"], label="predicted" if nm == "primary" else None)
    ax.axvline(0, color=T["ink2"], lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows]); ax.invert_yaxis()
    ax.set_xlabel("paired difference in team time [s]  (mean, 95% BCa CI)")
    ax.set_title(f"c  Paired contrasts (common random numbers)\n    n = {A['n_seeds']} seeds per condition", loc="left")
    ax.legend(fontsize=6.8, loc="lower right")
    # (d) separability (primary)
    ax = axs[1, 0]
    sep = A["separability"]
    pv = np.array([x["predicted"] for x in sep]); mv = np.array([x["measured"] for x in sep])
    lo = min(pv.min(), mv.min()) - 5; hi = max(pv.max(), mv.max()) + 5
    ax.plot([lo, hi], [lo, hi], color=T["ink4"], lw=1)
    nx = np.array([int(x["profile"].split("|")[3]) for x in sep])
    ax.scatter(pv[nx == 0], mv[nx == 0], s=18, color=T["ink3"], label="no shortcut users", zorder=3)
    ax.scatter(pv[nx > 0], mv[nx > 0], s=18, color=T["ink1"], label="≥ 1 shortcut user", zorder=3)
    for x, p_, m_ in zip(sep, pv, mv):
        prof = x["profile"].split("|", 1)[1]
        if prof in ("0|0|4", "2|2|0", "1|1|2"):
            ax.annotate("W{} E{} X{}".format(*prof.split("|")), (p_, m_), xytext=(4, -9), textcoords="offset points",
                        fontsize=6.3, color=T["ink2"])
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("separable exact game: predicted team time [s]"); ax.set_ylabel("measured team time [s]")
    ax.set_title("d  Separability check, primary\n    (all 15 profiles × 30 seeds)", loc="left")
    ax.legend(fontsize=6.8, loc="upper left")
    # (e) sweep: predicted vs screened open - closed
    ax = axs[1, 1]
    pts = [x for x in R["sweep_summary"] if x.get("screen")]
    for x in pts:
        pd_ = x["pred_diff"]; md = x["screen"]["diff_mean"]; ci = x["screen"]["diff_ci"]
        col = T["ink1"] if x["paradox"] else T["ink4"]
        ax.errorbar(pd_, md, yerr=[[md - ci[0]], [ci[1] - md]], fmt="o", ms=4, color=col, capsize=0, lw=0.8)
    for G, txt in ((A, "primary"), (B, "secondary")):
        if G and not G.get("calibration_failed"):
            x = next(q for q in pts if q["name"] == G["name"])
            ax.annotate(txt, (x["pred_diff"], x["screen"]["diff_mean"]), xytext=(6, 4), textcoords="offset points",
                        fontsize=6.8, color=T["accent"])
            ax.scatter([x["pred_diff"]], [x["screen"]["diff_mean"]], s=70, facecolors="none", edgecolors=T["accent"],
                       lw=1.3, zorder=4)
    ax.axhline(0, color=T["ink2"], lw=0.8); ax.axvline(0, color=T["ink2"], lw=0.8)
    ax.fill_between([0, ax.get_xlim()[1]], 0, ax.get_ylim()[1], color=T["water"], alpha=0.6, zorder=0, lw=0)
    ax.text(0.98, 0.97, "predicted & measured paradox", transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=T["ink2"])
    ax.set_xlabel("predicted open − closed team time [s]"); ax.set_ylabel("screened measured open − closed [s]")
    ax.set_title("e  Knob sweep, 26 geometries\n    (16 screening seeds each)", loc="left")
    ax.scatter([], [], color=T["ink1"], s=16, label="game predicts a paradox")
    ax.scatter([], [], color=T["ink4"], s=16, label="game predicts none")
    ax.legend(fontsize=6.6, loc="lower right")
    # (f) secondary
    ax = axs[1, 2]
    if B and not B.get("calibration_failed"):
        _panel_team(ax, B, T, f"f  Secondary {B['name']} (post hoc, fresh seeds)\n    measured: {_verdict(B)}")
    else:
        ax.axis("off")
    fig.suptitle("Return rush, n = 4 boats: does the physical course reproduce Braess's paradox?",
                 fontsize=11, color=T["ink1"], x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "return_rush_results.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/return_rush_results.png")


# ------------------------------------------------------------------------------------------------
# Experiment 2: road-like "slow zone" congestion (rb2.braess2 RushGeom(congestion="slow_zone"))
# ------------------------------------------------------------------------------------------------
SLOWZONE = False
SLOWZONE_GRID = [dict(congestion="slow_zone", alpha=al, v_open=v)
                 for al in (0.25, 0.5, 0.75, 1.0) for v in (0.3, 0.35, 0.4, 0.45, 0.5, 0.6)]
# A-PRIORI SELECTION RULE (fixed before any measurement of conditions a/b/c):
#   among the grid geometries whose 6-seed calibrated game predicts a paradox, prefer those whose selfish open
#   equilibrium is "everyone takes the shortcut" (the textbook Braess profile); within that set (or, if it is empty,
#   among all predicted paradoxes) take the largest predicted ratio C_open / C_closed. The 16-seed final calibration
#   must still predict a paradox, otherwise the next geometry in this ranking is used.
SLOWZONE_RULE = ("largest predicted C_open/C_closed among predicted paradoxes whose selfish open NE is all-shortcut "
                 "(fallback: among all predicted paradoxes); must survive the 16-seed recalibration")


def select_slowzone(cands):
    allx = [e for e in cands if e["prediction"]["profiles"]["open"]["shortcut"] == N]
    rest = [e for e in cands if e not in allx]
    key = lambda e: (-e["prediction"]["ratio_open_closed"], e["name"])  # noqa: E731
    return sorted(allx, key=key) + sorted(rest, key=key)


def make_figure_slowzone(R):
    """Two panels in the style of fig14_return_rush: (a) conditions, prediction diamonds vs measured dots with CIs;
    (b) separability, all 15 profiles: predicted vs measured team time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from make_figures import BLUE, GRID as GRIDC, MUTED, save
    A = R["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    conds = [("closed", "a_closed_selfish", "shortcut closed\nselfish"), ("open", "b_open_selfish", "shortcut open\nselfish"),
             ("tolled", "c_open_tolled", "shortcut open\nmarginal-cost tolls")]
    for k, (pk, mk, lab) in enumerate(conds):
        pred = A["prediction"]["team_cost"][pk]["value"]
        m = A["conditions"][mk]
        ax.plot([k - 0.12, k + 0.12], [pred, m["total_mean"]], color="#bbbbbb", lw=1.0, zorder=1)
        ax.plot(k - 0.12, pred, "D", color=MUTED, ms=5, label="exact game prediction" if k == 0 else None)
        ax.plot([k + 0.12] * 2, m["total_ci"], color=BLUE, lw=1.4)
        ax.plot(k + 0.12, m["total_mean"], "o", color=BLUE, ms=5, label="measured (95% CI)" if k == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in conds], fontsize=7)
    ax.set_xlim(-0.5, 2.5)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 1, hi + (hi - lo) * 0.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
    ax.set_ylabel("team time to home (s)", fontsize=7.5)
    ax.set_title("Slow-zone congestion, geometry chosen in advance", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax = axes[1]
    sep = A["separability"]
    pv = np.array([x["predicted"] for x in sep]); mv = np.array([x["measured"] for x in sep])
    ci = np.array([x["ci"] for x in sep])
    lo = min(pv.min(), ci.min()) - 4; hi = max(pv.max(), ci.max()) + 4
    ax.plot([lo, hi], [lo, hi], color="#bbbbbb", lw=1.0, zorder=1)
    for p_, m_, c_ in zip(pv, mv, ci):
        ax.plot([p_, p_], c_, color=BLUE, lw=1.2)
    ax.plot(pv, mv, "o", color=BLUE, ms=4)
    fit = A["separability_fit"]
    ax.text(0.04, 0.96, f"15 route profiles\nmean |error| {fit['mae']:.1f} s, max {fit['max_abs']:.1f} s\n"
            f"R² = {fit['r2']:.2f}", transform=ax.transAxes, va="top", fontsize=6.8, color="#555555")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted team time (s)", fontsize=7.5); ax.set_ylabel("measured team time (s)", fontsize=7.5)
    ax.set_title("Separability check", loc="left", fontsize=8.5)
    for a_ in axes:
        a_.grid(color=GRIDC); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig16_return_rush_slowzone")


def separability_fit(sep):
    pv = np.array([x["predicted"] for x in sep]); mv = np.array([x["measured"] for x in sep])
    err = mv - pv
    return dict(mae=float(np.mean(np.abs(err))), max_abs=float(np.max(np.abs(err))), bias=float(err.mean()),
                r2=float(1 - np.sum(err ** 2) / np.sum((mv - mv.mean()) ** 2)),
                corr=float(np.corrcoef(pv, mv)[0, 1]))


def enable_slowzone():
    global SLOWZONE, DATA, GRID, PROCS
    SLOWZONE = True
    DATA = os.path.join(ROOT, "data", "return_rush_slowzone")
    GRID = SLOWZONE_GRID
    PROCS = min(6, os.cpu_count() or 1)      # a background replication shares the machine


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--slowzone", action="store_true", help="experiment 2: slow-zone congestion (separate outputs)")
    a = ap.parse_args()
    if a.slowzone:
        enable_slowzone()
    fig_fn = make_figure_slowzone if a.slowzone else make_figure
    if a.figure:
        fig_fn(jload("results.json"))
        return
    t0 = time.time()
    with Pool(PROCS) as pool:
        sw = stage_sweep(pool, a.force)
        primary_rank, secondary_rank = select(sw)
        print("predicted-paradox geometries:", [e["name"] for e in primary_rank])
        R = dict(sweep_summary=[dict(name=e["name"], geom=e["geom"], paradox=e["prediction"]["paradox"],
                                     strong=e["prediction"]["strong_paradox"],
                                     ratio=e["prediction"]["ratio_open_closed"],
                                     pred_diff=e["prediction"]["team_cost"]["open"]["value"]
                                     - e["prediction"]["team_cost"]["closed"]["value"],
                                     profiles=e["prediction"]["profiles"], screen=e.get("screen"))
                                for e in sw["entries"]])
        if not primary_rank:
            R["primary"] = dict(calibration_failed=True, tried=[e["name"] for e in sw["entries"]])
        else:
            Gp = stage_geometry(pool, primary_rank, "primary", a.force)
            R["primary"] = analyze_geometry(Gp)
            if SLOWZONE:
                if not R["primary"].get("calibration_failed"):
                    R["primary"]["separability_fit"] = separability_fit(R["primary"]["separability"])
                R["selection_rule"] = SLOWZONE_RULE
                secondary_rank = []
            rest = [e for e in secondary_rank if e["name"] != Gp.get("name")]
            if rest:
                Gs = stage_geometry(pool, rest, "secondary", a.force)
                R["secondary"] = analyze_geometry(Gs)
    R["seeds"] = dict(sweep_calibration=SWEEP_CAL_SEEDS, final_calibration=FINAL_CAL_SEEDS, screening=SCREEN_SEEDS,
                      measurement=MEASURE_SEEDS, profile_game=PROFILE_SEEDS)
    R["runtime_s"] = time.time() - t0
    jdump(R, "results.json")
    fig_fn(R)
    for tag in ("primary", "secondary"):
        if tag in R and not R[tag].get("calibration_failed"):
            A = R[tag]
            print(f"\n[{tag}] {A['name']}  predicted paradox={A['prediction']['paradox']}  "
                  f"measured paradox={A['measured_paradox']}  reverse={A['measured_reverse']}  "
                  f"true-game paradox={A['profile_game']['true_paradox']}")
            for k, c in A["conditions"].items():
                print(f"   {k:18s} {c['counts']}  team {c['total_mean']:.1f} {np.round(c['total_ci'], 1)}  "
                      f"makespan {c['makespan_mean']:.1f}")
            for k, c in A["contrasts"].items():
                print(f"   {k:55s} mean {c['mean']:+.1f}  CI {np.round(c.get('ci_mean', (0, 0)), 1)}")
    print(f"total runtime {R['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
