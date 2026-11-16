"""Why the separable congestion game fails physically on the return-rush course: cross-route coupling.

  .venv/bin/python analysis/coupling.py            # all parts (cached in data/return_rush_coupling/)
  .venv/bin/python analysis/coupling.py --force    # recompute
  .venv/bin/python analysis/coupling.py --figures  # redraw figures only

Parts
  A  arrival-spacing injection: 4 boats reach the B->T single-file channel with controlled inter-arrival spacing
     Delta (released one by one at the same spot upstream of B); mean B->T delay vs Delta, for three service
     intervals of the channel (east channel speed limit). Deterministic-queue reference: mean delay =
     (1/4) sum_j max(0, j (h - Delta)) = 1.5 max(0, h - Delta) for 4 boats with service interval h.
  B  service interval (headway) of a single-file channel vs its speed limit and length (exits of a saturated queue).
  C  exploratory transition sweep over ONE parameter: the east channel speed limit (west fixed), which sets
     rho = h_east / h_west. Per point: coupling index kappa, separable-model error for the all-shortcut profile and
     over all 15 profiles, predicted vs measured Braess sign. The two end points are re-measured on fresh seeds.
  D  potential structure of the MEASURED games (queue = experiment 1 primary, slow zone = experiment 2 primary) and
     of the separable calibrated games: Monderer-Shapley 4-cycle circulations with a seed bootstrap, and an explicit
     improvement cycle.

Coupling index (k = 4): kappa = 1 - (c_BT|via shortcut(4) - c_BT|via shortcut(1)) / (c_BT|bunched(4) - c_BT|bunched(1)).
kappa = 0: B->T congestion depends only on its own load (separable); kappa = 1: upstream metering removes it entirely.
(Each arrival path's own k = 1 time is subtracted, so the fixed turn-in cost of the shortcut path is not counted as
coupling.)
"""
from __future__ import annotations

import argparse
import itertools
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

from rb2.braess2 import (ROUTES, RushGeom, aggregate_calibration, all_count_profiles, ckey,  # noqa: E402
                         counts_team_cost, edge_times, get_course, make_game, measure_job, predict, route_label,
                         run_scenario)
from src.game_exact import four_cycle_sums  # noqa: E402
from stats_lib import bootstrap_ci  # noqa: E402

DATA = os.path.join(ROOT, "data", "return_rush_coupling")
PROCS = min(8, os.cpu_count() or 1)
N = 4

# Part A / B
INJ_GEOM = dict(lane_len=6.0, lane_len_e=3.0, v_open=None)
INJ_SPEEDS = [None, 1.0, 0.6]
INJ_DELTAS = list(range(0, 21))
INJ_SEEDS = list(range(100, 106))
HW_SPEEDS = [None, 1.4, 1.0, 0.8, 0.6, 0.45]
HW_LENGTHS = [3.0, 6.0, 10.0, 14.0]
# Part C (west channel 6 m, east channel 3 m, open-water no-wake 0.4 m/s; parameter: east channel speed limit)
SWEEP_BASE = dict(lane_len=6.0, lane_len_e=3.0, v_open=0.4)
# The first pass used [None, 1.4, 1.0, 0.8, 0.6, 0.45] (rho 0.8-1.4); kappa only fell to about 0.5, so the grid was
# extended to slower downstream channels to reach the separable side. All points are reported.
SWEEP_SPEEDS = [None, 1.4, 1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.14]
# calibration and profile measurement use the SAME seeds (common random numbers): a profile in which all boats take
# one route is then literally the calibration run, so its separable-model error is exactly 0 and every remaining error
# is structural (cross-route coupling) plus seed noise, which the joint seed bootstrap quantifies.
SWEEP_CAL_SEEDS = SWEEP_PROF_SEEDS = list(range(3000, 3010))
VALID_CAL_SEEDS = VALID_PROF_SEEDS = list(range(4000, 4030))
BOOT = 2000


def jdump(o, name):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(o, f, indent=1, default=lambda x: str(x) if isinstance(x, Fraction) else float(x))


def jload(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------
# Part A + B: injection with controlled spacing
# ------------------------------------------------------------------------------------------------
def inject_job(args):
    """k boats released at the same spot 8 m upstream of B (alternating 2.4 m lanes), every `delta` seconds, at
    1.9 m/s heading south; delta = 0 means a bunched 2x2 formation. Returns per-boat B->T times, arrival times at
    B and the exit headways of the east channel."""
    gd, delta, seed, k = args
    geom = RushGeom(**gd)
    rc = get_course(geom)
    B = rc.nodes["B"]; ce = rc.x1 + geom.lane_width / 2; v0 = 1.9
    if delta == 0:
        st = [(ce + 2.4 * (j % 2), B[1] + 8.0 + 3.4 * (j // 2), -math.pi / 2, v0) for j in range(k)]; rel = None
    else:
        st = [(ce + 2.4 * (j % 2), B[1] + 8.0, -math.pi / 2, v0) for j in range(k)]; rel = [j * delta for j in range(k)]
    res = run_scenario(geom, ["east"] * k, seed, False, starts=st, lanes=False, release=rel)
    bt = [e[("B", "T")] for e in edge_times(res)]
    arr = sorted(nt["B"] for nt in res["node_times"])
    ex = [t for t, c, e, i in res["channel_log"] if e == "exit" and c == "east"]
    return dict(delta=delta, seed=seed, k=k, bt=bt, arrivals=[a - arr[0] for a in arr],
                headways=list(np.diff(ex)), finished=res["all_finished"], contacts=res["contacts"])


def part_a(pool):
    out = {}
    for v in INJ_SPEEDS:
        gd = dict(INJ_GEOM, lane_speed_e=v)
        free = pool.map(inject_job, [(gd, 0, s, 1) for s in INJ_SEEDS])
        rows = pool.map(inject_job, [(gd, d, s, N) for d in INJ_DELTAS for s in INJ_SEEDS], chunksize=2)
        c1 = {r["seed"]: r["bt"][0] for r in free}
        h = float(np.mean([x for r in rows if r["delta"] == 0 for x in r["headways"]]))
        curve = []
        for d in INJ_DELTAS:
            rr = [r for r in rows if r["delta"] == d]
            delay = np.array([np.mean(r["bt"]) - c1[r["seed"]] for r in rr])
            sp = np.array([np.mean(np.diff(r["arrivals"])) for r in rr])
            curve.append(dict(delta=d, delay=float(delay.mean()), delay_ci=[float(delay.min()), float(delay.max())],
                              spacing_at_B=float(sp.mean()), theory=1.5 * max(0.0, h - d)))
        out[str(v)] = dict(lane_speed_e=v, free_flow=float(np.mean(list(c1.values()))), service_interval=h,
                           curve=curve, unfinished=int(sum(not r["finished"] for r in rows)))
        print(f"  A: speed limit {v}: h = {h:.1f} s, delay(0) = {curve[0]['delay']:.1f} s", flush=True)
    return out


def part_b(pool):
    rows = []
    for v in HW_SPEEDS:
        rows.append(("speed", v, dict(INJ_GEOM, lane_speed_e=v)))
    for L in HW_LENGTHS:
        rows.append(("length", L, dict(INJ_GEOM, lane_len_e=L)))
    res = pool.map(inject_job, [(gd, 0, s, N) for _, _, gd in rows for s in INJ_SEEDS[:3]])
    out = []
    for k, (kind, val, gd) in enumerate(rows):
        rr = res[k * 3:(k + 1) * 3]
        free = inject_job((gd, 0, INJ_SEEDS[0], 1))["bt"][0]
        out.append(dict(kind=kind, value=val, headway=float(np.mean([x for r in rr for x in r["headways"]])),
                        single_boat_bt=free))
    return out


# ------------------------------------------------------------------------------------------------
# Part C: transition sweep
# ------------------------------------------------------------------------------------------------
def calib_hw_job(args):
    gd, route, k, seed = args
    geom = RushGeom(**gd)
    res = run_scenario(geom, [route] * k, seed, shortcut_open=(route == "shortcut"))
    et = edge_times(res)
    edges = {}
    for e in {e for d in et for e in d}:
        edges[e] = float(np.mean([d[e] for d in et if e in d]))
    hw = {}
    for ch in ("west", "east"):
        ex = [t for t, c, ev, i in res["channel_log"] if ev == "exit" and c == ch]
        hw[ch] = list(np.diff(ex))
    return dict(route=route, k=k, seed=seed, edges=edges, finished=res["all_finished"], contacts=res["contacts"],
                headways=hw)


def evaluate_point(pool, gd, cal_seeds, prof_seeds):
    jobs = [(gd, r, k, s) for r in ROUTES for k in range(1, N + 1) for s in cal_seeds]
    rows = pool.map(calib_hw_job, jobs, chunksize=2)
    costs, sds, via = aggregate_calibration(rows, N)
    h_w = float(np.mean([x for r in rows if r["route"] == "west" and r["k"] == N for x in r["headways"]["west"]]))
    h_e = float(np.mean([x for r in rows if r["route"] == "east" and r["k"] == N for x in r["headways"]["east"]]))
    bunched = costs["BT"]
    kappa = 1 - (via[-1] - via[0]) / (bunched[-1] - bunched[0])
    kappa_simple = 1 - (via[-1] - bunched[0]) / (bunched[-1] - bunched[0])
    P = predict(costs, N)
    geom = RushGeom(**gd)
    conds = {("open",) + ckey(c): (c, True) for c in all_count_profiles(N, True)}
    conds[("closedNE",)] = (P["profiles"]["closed"], False)
    jobs = [(geom, c, s, op) for (c, op) in conds.values() for s in prof_seeds]
    mrows = pool.map(measure_job, jobs, chunksize=2)
    table, team, team_rows = {}, {}, {}
    for idx, key in enumerate(conds):
        rr = mrows[idx * len(prof_seeds):(idx + 1) * len(prof_seeds)]
        team["|".join(map(str, key))] = float(np.mean([r["total"] for r in rr]))
        team_rows["|".join(map(str, key))] = [r["total"] for r in rr]
        if key[0] == "open":
            table["|".join(map(str, key[1:]))] = {r: [row["per_route"][r] for row in rr if r in row["per_route"]]
                                                 for r in ROUTES}
    pred = {"|".join(map(str, ckey(c))): float(counts_team_cost(costs, c)) for c in all_count_profiles(N, True)}
    err = {p: team["open|" + p] - pred[p] for p in pred}
    open_ne = "|".join(map(str, ckey(P["profiles"]["open"])))
    pred_braess = float(P["team_cost"]["open"] - P["team_cost"]["closed"])
    d_meas = np.array(team_rows["open|" + open_ne]) - np.array(team_rows["closedNE"])
    meas_braess = float(d_meas.mean())
    ci = bootstrap_ci(d_meas, B=5000) if np.any(d_meas != d_meas[0]) else (meas_braess, meas_braess)
    boot = bootstrap_point(rows, mrows, list(conds), prof_seeds)
    return dict(geom=gd, h_w=h_w, h_e=h_e, rho=h_e / h_w, kappa=kappa, kappa_simple=kappa_simple, boot=boot,
                costs=costs, via_bt=via, prediction=dict(profiles=P["profiles"], paradox=P["paradox"],
                                                         team_cost={k: float(v) for k, v in P["team_cost"].items()}),
                pred_team=pred, meas_team={p: team["open|" + p] for p in pred}, errors=err,
                err_all_shortcut=err["0|0|4"], mae=float(np.mean(np.abs(list(err.values())))),
                pred_braess=pred_braess, meas_braess=meas_braess, meas_braess_ci=list(ci),
                sign_agree=bool(np.sign(round(pred_braess, 1)) == np.sign(meas_braess) if ci[0] * ci[1] > 0
                                else round(pred_braess, 1) == 0),
                table=table, unfinished=int(sum(not r["finished"] for r in rows) + sum(not r["finished"] for r in mrows)))


LOADS = {"SA": lambda c: c[0] + c[2], "AT": lambda c: c[0], "SB": lambda c: c[1], "BT": lambda c: c[1] + c[2],
         "AB": lambda c: c[2]}
SRC = {"SA": "west", "AT": "west", "SB": "east", "BT": "east", "AB": "shortcut"}


def bootstrap_point(rows, mrows, cond_keys, seeds, B=2000):
    """Joint seed bootstrap (same seed indices for calibration and measurement) of the separable-model errors,
    the all-shortcut error, the 15-profile MAE and kappa. Float costs (no 0.1 s rounding)."""
    S = len(seeds); sidx = {s: i for i, s in enumerate(seeds)}
    C = {e: np.full((N, S), np.nan) for e in LOADS}
    via = np.full((N, S), np.nan)
    for r in rows:
        for e, src in SRC.items():
            key = (e[0], e[1])
            if r["route"] == src and key in r["edges"]:
                C[e][r["k"] - 1, sidx[r["seed"]]] = r["edges"][key]
        if r["route"] == "shortcut" and ("B", "T") in r["edges"]:
            via[r["k"] - 1, sidx[r["seed"]]] = r["edges"][("B", "T")]
    profs = [k for k in cond_keys if k[0] == "open"]
    M = np.full((len(profs), S), np.nan)
    for i, key in enumerate(profs):
        rr = mrows[cond_keys.index(key) * S:(cond_keys.index(key) + 1) * S]
        for row in rr:
            M[i, sidx[row["seed"]]] = row["total"]

    def stats(ix):
        c = {e: np.nanmean(C[e][:, ix], axis=1) for e in C}
        v = np.nanmean(via[:, ix], axis=1)
        pred = np.array([sum(LOADS[e](k[1:]) * c[e][LOADS[e](k[1:]) - 1] for e in c if LOADS[e](k[1:]) > 0)
                         for k in profs])
        err = np.nanmean(M[:, ix], axis=1) - pred
        kap = 1 - (v[-1] - v[0]) / (c["BT"][-1] - c["BT"][0])
        return err, kap
    err0, kap0 = stats(np.arange(S))
    rng = np.random.default_rng(3)
    E, K = [], []
    for _ in range(B):
        e_, k_ = stats(rng.integers(0, S, S)); E.append(e_); K.append(k_)
    E = np.array(E); K = np.array(K)
    ax = profs.index(("open", 0, 0, N))
    mixed = [i for i, k in enumerate(profs) if max(k[1:]) < N]      # profiles with >= 2 routes in use

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(err={"|".join(map(str, k[1:])): float(x) for k, x in zip(profs, err0)},
                err_ci={"|".join(map(str, k[1:])): ci(E[:, i]) for i, k in enumerate(profs)},
                err_all_shortcut=float(err0[ax]), err_all_shortcut_ci=ci(E[:, ax]),
                mae=float(np.mean(np.abs(err0))), mae_ci=ci(np.mean(np.abs(E), axis=1)),
                mae_mixed=float(np.mean(np.abs(err0[mixed]))), mae_mixed_ci=ci(np.mean(np.abs(E[:, mixed]), axis=1)),
                kappa=float(kap0), kappa_ci=ci(K),
                n_sig_profiles=int(sum(1 for i in range(len(profs)) if ci(E[:, i])[0] > 0 or ci(E[:, i])[1] < 0)))


def part_c(pool, force):
    sw = None if force else jload("sweep.json")
    done = {str(p["lane_speed_e"]): p for p in (sw or {}).get("points", [])}
    if sw is None or any(str(v) not in done for v in SWEEP_SPEEDS):
        pts = []
        for v in SWEEP_SPEEDS:
            if str(v) in done:
                pts.append(done[str(v)]); continue
            t0 = time.time()
            r = evaluate_point(pool, dict(SWEEP_BASE, lane_speed_e=v), SWEEP_CAL_SEEDS, SWEEP_PROF_SEEDS)
            r["lane_speed_e"] = v
            pts.append(r)
            print(f"  C: v_e={v}: rho={r['rho']:.2f} kappa={r['boot']['kappa']:.2f} {np.round(r['boot']['kappa_ci'], 2)} "
                  f"errX={r['boot']['err_all_shortcut']:+.1f} {np.round(r['boot']['err_all_shortcut_ci'], 1)} "
                  f"MAEmix={r['boot']['mae_mixed']:.1f} {np.round(r['boot']['mae_mixed_ci'], 1)} "
                  f"err(all-X, 0.1s-rounded game)={r['err_all_shortcut']:+.1f} "
                  f"MAE={r['mae']:.1f} braess pred {r['pred_braess']:+.1f} meas {r['meas_braess']:+.1f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        sw = dict(points=pts)
        jdump(sw, "sweep.json")
    va = None if force else jload("validation.json")
    if va is None or [v["lane_speed_e"] for v in va] != [SWEEP_SPEEDS[0], SWEEP_SPEEDS[-1]]:
        va = []
        for v in (SWEEP_SPEEDS[0], SWEEP_SPEEDS[-1]):     # one point on each side, chosen by position, fresh seeds
            r = evaluate_point(pool, dict(SWEEP_BASE, lane_speed_e=v), VALID_CAL_SEEDS, VALID_PROF_SEEDS)
            r["lane_speed_e"] = v
            va.append(r)
            print(f"  C validation v_e={v}: rho={r['rho']:.2f} kappa={r['kappa']:.2f} "
                  f"err(all-X)={r['err_all_shortcut']:+.1f} MAE={r['mae']:.1f}", flush=True)
        jdump(va, "validation.json")
    return sw, va


# ------------------------------------------------------------------------------------------------
# Part D: circulation and improvement cycles
# ------------------------------------------------------------------------------------------------
ROUTE_IDX = {"west": 0, "east": 1, "shortcut": 2}


def tables_from_profile_runs(prof_runs):
    """prof_runs: {"open|w|e|s": [row per seed]} -> per-seed cost arrays {(w,e,s): {route: np.array(seeds)}}."""
    T = {}
    for key, rows in prof_runs.items():
        parts = key.split("|")
        if parts[0] != "open":
            continue
        p = tuple(int(x) for x in parts[1:])
        T[p] = {r: np.array([row["per_route"][r] for row in rows]) for r in ROUTES if r in rows[0]["per_route"]}
    return T


def _counts(a):
    return (a.count(0), a.count(1), a.count(2))


def cycle_design():
    """Every Monderer-Shapley 4-cycle of the labeled 4-boat, 3-route game, in the enumeration order of
    src.game_exact.four_cycle_sums. Each circulation is a signed sum of 8 cost entries (counts, route); returns the
    cycles, the entry list and the (n_cycles x n_entries) signed incidence matrix, so I = M @ costs."""
    entries, eidx, cycles, rows = [], {}, [], []

    def e(a, i):
        key = (_counts(a), ROUTES[a[i]])
        if key not in eidx:
            eidx[key] = len(entries); entries.append(key)
        return eidx[key]
    for a in itertools.product(range(3), repeat=N):
        for i, j in itertools.combinations(range(N), 2):
            for x in range(3):
                if x == a[i]:
                    continue
                for y in range(3):
                    if y == a[j]:
                        continue
                    p0 = tuple(a); p1 = list(a); p1[i] = x; p1 = tuple(p1)
                    p2 = list(p1); p2[j] = y; p2 = tuple(p2); p3 = list(a); p3[j] = y; p3 = tuple(p3)
                    row = {}
                    for (p, q, k) in ((p0, p1, i), (p1, p2, j), (p2, p3, i), (p3, p0, j)):
                        row[e(q, k)] = row.get(e(q, k), 0) + 1
                        row[e(p, k)] = row.get(e(p, k), 0) - 1
                    cycles.append((p0, p1, p2, p3)); rows.append(row)
    M = np.zeros((len(rows), len(entries)))
    for r, row in enumerate(rows):
        for c, v in row.items():
            M[r, c] = v
    return cycles, entries, M


def circulation_report(T, seeds_n):
    cycles, entries, M = cycle_design()
    E = np.array([T[p][r] for (p, r) in entries])               # entries x seeds
    I = M @ E.mean(axis=1)
    # cross-check against the reference implementation in src.game_exact
    mean_tab = {p: {r: float(v.mean()) for r, v in d.items()} for p, d in T.items()}
    ref = [float(x) for _, x in four_cycle_sums([3] * N, lambda a, i: mean_tab[_counts(a)][ROUTES[a[i]]])]
    assert np.allclose(ref, I, atol=1e-9)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, seeds_n, (BOOT, seeds_n))              # joint seed resampling (common random numbers)
    boots = (M @ E[:, idx].mean(axis=2)).T                       # BOOT x cycles
    se = boots.std(axis=0, ddof=1)
    z = np.where(se > 1e-12, I / np.maximum(se, 1e-12), 0.0)
    noise_max = float(np.percentile(np.max(np.abs(boots - I[None, :]), axis=1), 95))
    k = int(np.argmax(np.abs(I)))

    def nm(a):
        return "(" + ",".join({"west": "W", "east": "E", "shortcut": "X"}[ROUTES[x]] for x in a) + ")"
    steps = []
    seq = list(cycles[k]) + [cycles[k][0]]
    for p, q in zip(seq, seq[1:]):
        i = next(m for m in range(N) if p[m] != q[m])
        steps.append(dict(frm=nm(p), to=nm(q), mover=i, cost_before=mean_tab[_counts(p)][ROUTES[p[i]]],
                          cost_after=mean_tab[_counts(q)][ROUTES[q[i]]]))
    return dict(n_cycles=len(I), max_abs=float(np.max(np.abs(I))), median_abs=float(np.median(np.abs(I))),
                frac_nonzero_structural=float(np.mean(np.abs(I) > 1e-9)),
                frac_z_gt2=float(np.mean(np.abs(z) > 2)), frac_z_gt3=float(np.mean(np.abs(z) > 3)),
                max_abs_ci=[float(np.percentile(np.max(np.abs(boots), axis=1), 2.5)),
                            float(np.percentile(np.max(np.abs(boots), axis=1), 97.5))],
                noise_max95=noise_max,
                example=dict(cycle=[nm(x) for x in cycles[k]], circulation=float(I[k]), se=float(se[k]), steps=steps),
                abs_sorted=sorted(np.abs(I).tolist()), z_abs_sorted=sorted(np.abs(z).tolist()))


def improvement_cycle(T, seeds_n, allowed=None):
    """Shortest directed cycle in the route-count graph where every step is a strict improvement for the mover
    (costs = seed means). Lifted to labeled boats; each step's improvement gets a seed-bootstrap CI."""
    mt = {p: {r: float(v.mean()) for r, v in d.items()} for p, d in T.items()}
    succ = {}
    for p in mt:
        succ[p] = []
        for r in ROUTES:
            if p[ROUTE_IDX[r]] == 0:
                continue
            for r2 in ROUTES:
                if r2 == r:
                    continue
                q = list(p); q[ROUTE_IDX[r]] -= 1; q[ROUTE_IDX[r2]] += 1; q = tuple(q)
                if mt[q][r2] < mt[p][r] and (allowed is None or allowed[(p, r, r2)]):
                    succ[p].append((q, r, r2))
    best = None
    for start in sorted(mt):                             # BFS: shortest cycle through `start`
        parent = {start: None}
        queue = [start]
        found = None
        while queue and found is None:
            nq = []
            for u in queue:
                for (q, r, r2) in succ[u]:
                    if q == start:
                        found = (u, r, r2); break
                    if q not in parent:
                        parent[q] = (u, r, r2); nq.append(q)
                if found:
                    break
            queue = nq
        if found is None:
            continue
        # walk parents from found[0] back to start
        chain = []
        u = found[0]
        while u != start:
            pu, r, r2 = parent[u]
            chain.append((pu, u, r, r2)); u = pu
        chain = chain[::-1] + [(found[0], start, found[1], found[2])]
        if best is None or len(chain) < len(best):
            best = chain
    if best is None:
        return None
    rng = np.random.default_rng(1)
    idxs = rng.integers(0, seeds_n, (BOOT, seeds_n))
    steps = []
    for (p, q, r, r2) in best:
        before, after = mt[p][r], mt[q][r2]
        gains = T[p][r][idxs].mean(axis=1) - T[q][r2][idxs].mean(axis=1)
        steps.append(dict(frm=list(p), to=list(q), route_from=r, route_to=r2, cost_before=before, cost_after=after,
                          gain=before - after, gain_ci=[float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
                          p_gain_positive=float(np.mean(gains > 0))))
    labeled = lift_cycle(best[0][0], steps)
    return dict(count_cycle=[list(b[0]) for b in best] + [list(best[0][0])], steps=steps, labeled_cycle=labeled,
                labeled_length=len(labeled) - 1, all_steps_significant=bool(all(s["gain_ci"][0] > 0 for s in steps)))


def lift_cycle(p0, steps):
    """Shortest closed walk of LABELED profiles (boat 1..4 -> route) that follows the count-level improvement cycle
    (any boat on the step's origin route may be the mover). Costs depend only on counts, so each step of the walk is
    the same strict improvement for its mover; the walk is a genuine finite improvement cycle of the 4-player game."""
    start = tuple(r for r in ROUTES for _ in range(p0[ROUTE_IDX[r]]))
    L = len(steps)
    parent = {(start, 0): None}
    queue = [(start, 0)]
    while queue:
        nq = []
        for (lab, k) in queue:
            st = steps[k % L]
            for i in range(N):
                if lab[i] != st["route_from"]:
                    continue
                nl = list(lab); nl[i] = st["route_to"]; node = (tuple(nl), (k + 1) % L)
                if node == (start, 0):
                    path = [node, (lab, k)]
                    cur = (lab, k)
                    while parent[cur] is not None:
                        cur = parent[cur]; path.append(cur)
                    return [list(x[0]) for x in path[::-1]]
                if node not in parent:
                    parent[node] = (lab, k); nq.append(node)
        queue = nq
    return None


def significant_improvement_cycle(T, seeds_n):
    """Same search restricted to moves whose improvement is significant (bootstrap 2.5% quantile > 0)."""
    rng = np.random.default_rng(2)
    idxs = rng.integers(0, seeds_n, (BOOT, seeds_n))
    Tb = {}
    for p, d in T.items():
        Tb[p] = {r: v[idxs].mean(axis=1) for r, v in d.items()}
    sig = {}
    for p in T:
        for r in T[p]:
            for r2 in ROUTES:
                if r2 == r:
                    continue
                q = list(p); q[ROUTE_IDX[r]] -= 1; q[ROUTE_IDX[r2]] += 1; q = tuple(q)
                g = Tb[p][r] - Tb[q][r2]
                sig[(p, r, r2)] = np.percentile(g, 2.5) > 0
    T2 = {p: {r: v for r, v in d.items()} for p, d in T.items()}
    return improvement_cycle(T2, seeds_n, allowed=sig)


def separable_cost_fn(costs):
    g = make_game(costs, True, N)
    lab = {route_label(g, r): r for r in range(len(g.routes))}

    def cost_counts(c, r):
        prof = [lab[x] for x in ROUTES for _ in range(c[ROUTE_IDX[x]])]
        return g.private_cost(prof, prof.index(lab[r]))
    return cost_counts


def separable_circulation(costs):
    """Exact (Fraction) circulations of the separable calibrated game: all zero by Rosenthal's potential."""
    cf = separable_cost_fn(costs)
    I = [x for _, x in four_cycle_sums([3] * N, lambda a, i: cf(_counts(a), ROUTES[a[i]]))]
    return dict(n_cycles=len(I), max_abs=str(max(abs(x) for x in I)), all_zero=bool(all(x == 0 for x in I)))


def part_d():
    out = {}
    srcs = {"queue": os.path.join(ROOT, "data", "return_rush", "geometry_primary.json"),
            "slow_zone": os.path.join(ROOT, "data", "return_rush_slowzone", "geometry_primary.json")}
    for name, path in srcs.items():
        G = json.load(open(path))
        T = tables_from_profile_runs(G["profile_runs"])
        n = len(next(iter(G["profile_runs"].values())))
        rep = circulation_report(T, n)
        rep["improvement_cycle"] = improvement_cycle(T, n)
        rep["significant_improvement_cycle"] = significant_improvement_cycle(T, n)
        rep["pure_ne"] = G.get("profile_game_open_equilibria")
        rep["separable"] = separable_circulation(G["calibration"]["costs"])
        rep["geometry"] = G["name"]; rep["seeds"] = n
        out[name] = rep
        ic = rep["improvement_cycle"]
        print(f"  D {name}: max|I| = {rep['max_abs']:.1f} s (95% noise max {rep['noise_max95']:.1f}), "
              f"|z|>3: {rep['frac_z_gt3']:.2f}; separable all zero: {rep['separable']['all_zero']}; "
              f"improvement cycle: {None if ic is None else ic['count_cycle']}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------
# Figures (style of figures/fig14_return_rush)
# ------------------------------------------------------------------------------------------------
def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures import BLUE, GRID, INK2, MUTED, ORANGE, save

    def tidy(ax):
        ax.grid(color=GRID); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5)

    # ---- fig18: (a) delay vs spacing, (b) kappa vs rho, (c) model error vs rho
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axs.ravel()
    ax = axes[0]
    shades = {"None": BLUE, "1.0": "#6e9fdc", "0.6": "#a9c6ec"}
    for key, A in R["A"].items():
        d = [c["delta"] for c in A["curve"]]; y = [c["delay"] for c in A["curve"]]
        th = [c["theory"] for c in A["curve"]]
        ax.plot(d, th, color="#bbbbbb", lw=1.0, zorder=1)
        ax.plot(d, y, "o", color=shades[key], ms=3.2, label=f"h = {A['service_interval']:.0f} s")
    ax.plot([], [], color="#bbbbbb", lw=1.0, label="deterministic queue")
    ax.set_xlabel("arrival spacing Δ at B (s)", fontsize=7.5)
    ax.set_ylabel("mean B→T delay, 4 boats (s)", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    tidy(ax)
    pts = R["C"]["points"]; val = R["C"]["validation"]
    rho = np.array([p["rho"] for p in pts])

    def eb(ax, x, y, lo_hi, **kw):
        y = np.asarray(y); lo_hi = np.asarray(lo_hi)
        ax.errorbar(x, y, yerr=[y - lo_hi[:, 0], lo_hi[:, 1] - y], capsize=0, elinewidth=1.0, **kw)
    ax = axes[1]
    ax.axvline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=0)
    eb(ax, rho, [p["boot"]["kappa"] for p in pts], [p["boot"]["kappa_ci"] for p in pts], fmt="o-", color=BLUE,
       ms=4, lw=1.0, label="exploratory sweep")
    eb(ax, [v["rho"] for v in val], [v["boot"]["kappa"] for v in val], [v["boot"]["kappa_ci"] for v in val],
       fmt="D", mfc="white", mec=INK2, color=INK2, ms=5.5, label="fresh-seed check")
    ax.set_xlabel("ρ = downstream / upstream service interval", fontsize=7.5)
    ax.set_ylabel("coupling index κ", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    tidy(ax)
    ax = axes[2]
    ax.axvline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=0)
    eb(ax, rho, [p["boot"]["err_all_shortcut"] for p in pts], [p["boot"]["err_all_shortcut_ci"] for p in pts],
       fmt="o-", color=BLUE, ms=4, lw=1.0, label="all-shortcut profile")
    eb(ax, rho, [p["boot"]["mae_mixed"] for p in pts], [p["boot"]["mae_mixed_ci"] for p in pts],
       fmt="s-", color=ORANGE, ms=3.5, lw=1.0, label="mean |error|, mixed profiles")
    for v in val:
        eb(ax, [v["rho"]], [v["boot"]["err_all_shortcut"]], [v["boot"]["err_all_shortcut_ci"]], fmt="D",
           mfc="white", mec=INK2, color=INK2, ms=5.5)
        eb(ax, [v["rho"]], [v["boot"]["mae_mixed"]], [v["boot"]["mae_mixed_ci"]], fmt="D", mfc="white", mec=INK2,
           color=INK2, ms=5.5)
    ax.set_xlabel("ρ = downstream / upstream service interval", fontsize=7.5)
    ax.set_ylabel("measured − separable prediction (s)", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    tidy(ax)
    ax = axes[3]
    ax.axvline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=0)
    for k, p in enumerate(pts):
        ax.plot([p["rho"], p["rho"]], [p["pred_braess"], p["meas_braess"]], color="#bbbbbb", lw=1.0, zorder=1)
    ax.plot(rho, [p["pred_braess"] for p in pts], "D", color=MUTED, ms=4.5, label="separable game prediction")
    eb(ax, rho, [p["meas_braess"] for p in pts], [p["meas_braess_ci"] for p in pts], fmt="o", color=BLUE, ms=4,
       label="measured (95% CI)")
    ax.set_xlabel("ρ = downstream / upstream service interval", fontsize=7.5)
    ax.set_ylabel("team time, open NE − closed NE (s)", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    tidy(ax)
    for a_, l in zip(axes, "abcd"):
        a_.text(-0.16, 1.03, l, transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.tight_layout()
    save(fig, "fig18_coupling")

    # ---- fig19: circulation magnitudes vs bootstrap noise
    D = R["D"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, key, lab in zip(axes, ["queue", "slow_zone"], ["single-file queues", "slow zones"]):
        d = D[key]
        a = np.array(d["abs_sorted"]); q = np.arange(1, len(a) + 1) / len(a)
        ax.plot(a, q, color=BLUE, lw=1.4, label="measured game")
        ax.axvline(d["noise_max95"], color=MUTED, lw=1.0, label="95% bootstrap noise (max)")
        ax.axvline(0, color=INK2, lw=1.4, label="separable game (exactly 0)")
        ax.set_xlabel(f"|circulation| around a 4-cycle (s), {lab}", fontsize=7.5)
        ax.set_ylabel("fraction of 4-cycles", fontsize=7.5)
        tidy(ax)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    for a_, l in zip(axes, "ab"):
        a_.text(-0.15, 1.04, l, transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.tight_layout()
    save(fig, "fig19_circulation")


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.figures:
        figures(jload("results.json")); return
    t0 = time.time()
    R = {}
    with Pool(PROCS) as pool:
        R["A"] = (None if a.force else jload("injection.json")) or part_a(pool)
        jdump(R["A"], "injection.json")
        R["B"] = (None if a.force else jload("service_interval.json")) or part_b(pool)
        jdump(R["B"], "service_interval.json")
        sw, va = part_c(pool, a.force)
    R["C"] = dict(points=[{k: v for k, v in p.items() if k != "table"} for p in sw["points"]],
                  validation=[{k: v for k, v in p.items() if k != "table"} for p in va])
    R["D"] = part_d()
    R["runtime_s"] = time.time() - t0
    jdump(R, "results.json")
    figures(R)
    print(f"total {R['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
