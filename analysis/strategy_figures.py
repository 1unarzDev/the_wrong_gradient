"""Figures that explain WHICH strategy did better and WHY (confirmatory data only).

  python analysis/strategy_figures.py            # all
  python analysis/strategy_figures.py arch       # just the architecture diagram

The time-course figure re-simulates 200 confirmatory worlds deterministically to recover
completion timestamps and ASSERTS that each reproduced welfare equals the CSV value."""
import math
import os
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, save  # noqa: E402

NAMES = {"P0": "Greedy nearest", "P1": "Selfish race", "P2": "Marginal, graph ETA",
         "A1": "Marginal, graph ETA, age-aware", "A2": "Marginal, control ETA",
         "P3": "Marginal, control ETA, age-aware", "P4": "Centralized assignment"}
EP = os.path.join(ROOT, "data/confirmatory/episodes.csv")


def load():
    return pd.read_csv(EP)


def fig_league(df):
    """Ranked mean welfare, both comms; the centralized reference is the only coloured mark."""
    order = df[df.condition == "ideal"].groupby("policy").welfare.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for k, pol in enumerate(order):
        for cond, mk, dy in (("ideal", "o", 0.13), ("degraded", "s", -0.13)):
            v = df[(df.condition == cond) & (df.policy == pol)].welfare
            m, h = v.mean(), 1.96 * v.std(ddof=1) / math.sqrt(len(v))
            col = BLUE if pol == "P4" else INK2
            ax.plot([m - h, m + h], [k + dy, k + dy], color=col, lw=2, solid_capstyle="round")
            ax.plot(m, k + dy, mk, color=col, ms=6, mfc=col if cond == "ideal" else "white", mew=1.5)
        ax.text(df[(df.policy == pol)].welfare.mean() + 3.2, k, "", va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{p}  {NAMES[p]}" for p in order], fontsize=8.5)
    ax.set_xlabel("mean team welfare W per world (95% CI, 400 worlds)")
    ax.plot([], [], "o", color=INK2, label="ideal comms"); ax.plot([], [], "s", color=INK2, mfc="white", label="degraded comms")
    ax.legend(loc="lower right", fontsize=8)
    # annotate the gap that matters
    p4 = df[(df.condition == "ideal") & (df.policy == "P4")].welfare.mean()
    best_dec = df[(df.condition == "ideal") & (df.policy == "P3")].welfare.mean()   # tested contrast P4 − P3
    y = len(order) - 1
    ax.annotate("", xy=(p4, y + 0.42), xytext=(best_dec, y + 0.42),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
    ax.text((p4 + best_dec) / 2, y + 0.55, f"P4 − P3 = +{p4 - best_dec:.1f}", ha="center", color=BLUE, fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_title("Centralized assignment leads; decentralized variants are indistinguishable", loc="left", fontsize=9.5)
    save(fig, "fig7_league")


def fig_head_to_head(df):
    pols = ["P4", "P1", "A1", "P2", "P3", "A2", "P0"]
    x = df[df.condition == "ideal"].pivot_table(index="seed", columns="policy", values="welfare")
    M = np.full((len(pols), len(pols)), np.nan)
    for i, a in enumerate(pols):
        for j, b in enumerate(pols):
            if i != j:
                d = x[a] - x[b]
                M[i, j] = (d > 0).mean() + 0.5 * (d == 0).mean()
    from make_figures import DIVERGING
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(M, cmap=DIVERGING.reversed(), vmin=0.0, vmax=1.0)
    for i in range(len(pols)):
        for j in range(len(pols)):
            if i != j:
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j] - 0.5) > 0.3 else INK)
    ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols)
    ax.set_yticks(range(len(pols))); ax.set_yticklabels([f"{p} {NAMES[p].split(',')[0]}" for p in pols], fontsize=8)
    ax.set_xlabel("opponent"); ax.grid(False)
    ax.set_title("% of the 400 worlds in which the row strategy\nbeats the column strategy (ideal comms; ties count ½)", loc="left", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.045); cb.set_label("win rate")
    save(fig, "fig8_head_to_head")


def fig_mechanism(df):
    """Why the centralized reference wins: less churn, fewer expired tasks."""
    pols = ["P0", "P1", "P2", "A1", "A2", "P3", "P4"]
    d = df[df.condition == "degraded"]
    stats = [("switches", "target switches per episode"), ("expired", "tasks that expired"),
             ("redundant_visits", "redundant visits")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    for ax, (col, lab) in zip(axs, stats):
        g = d.groupby("policy")[col]
        m = g.mean().reindex(pols); se = (g.std(ddof=1) / np.sqrt(g.count())).reindex(pols)
        colors = [BLUE if p == "P4" else (MUTED if p == "P0" else INK2) for p in pols]
        ax.barh(range(len(pols)), m.values, xerr=1.96 * se.values, color=colors, height=0.62,
                error_kw=dict(ecolor=INK2, lw=0.8))
        ax.set_title(lab, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axs[0].set_yticks(range(len(pols))); axs[0].set_yticklabels(pols)
    fig.suptitle("Mechanism (degraded comms): a single coordinator re-plans far less and lets fewer deadlines pass",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig9_mechanism")


def _timecourse_job(a):
    seed, pol = a
    from src.config import config_from, load_yaml
    from src.sim import Simulation
    spec = load_yaml(os.path.join(ROOT, "configs/confirmatory.yaml"))
    cfg = config_from({**spec["base"], "comm_mode": "ideal", "policy": pol})
    sim = Simulation(cfg, seed)
    m = sim.run()
    comp = [(e.t, e.value) for e in sim.events if e.kind == "complete"]
    return seed, pol, m["welfare"], comp


def fig_timecourse(df, n_worlds=200):
    pols = ["P0", "P2", "P3", "P4"]
    seeds = sorted(df.seed.unique())[:n_worlds]
    with Pool(int(os.environ.get("WORKERS", "16"))) as pool:
        res = pool.map(_timecourse_job, [(s, p) for s in seeds for p in pols])
    ref = df[df.condition == "ideal"].set_index(["seed", "policy"]).welfare
    bad = [(s, p) for s, p, w, _ in res if abs(w - ref.loc[(s, p)]) > 1e-9]
    assert not bad, f"reproduction mismatch: {bad[:5]}"
    print(f"reproduced {len(res)} confirmatory episodes exactly")
    tgrid = np.arange(0, 241, 2.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    style = {"P4": (BLUE, "-"), "P3": (INK, "-"), "P2": (INK2, "--"), "P0": (MUTED, ":")}
    for pol in pols:
        curves = []
        for s, p, w, comp in res:
            if p != pol:
                continue
            ts = np.array([c[0] for c in comp]); vs = np.array([c[1] for c in comp])
            curves.append([vs[ts <= t].sum() for t in tgrid])
        C = np.array(curves, float)
        m = C.mean(0); h = 1.96 * C.std(0, ddof=1) / math.sqrt(len(C))
        col, ls = style[pol]
        ax.plot(tgrid, m, ls, color=col, lw=2)
        ax.fill_between(tgrid, m - h, m + h, color=col, alpha=0.12, lw=0)
        dy = {"P2": -2.2, "P3": 2.2}.get(pol, 0)
        ax.text(242, m[-1] + dy, f"{pol} {NAMES[pol]}", va="center", fontsize=7.5, color=col)
    ax.set_xlim(0, 330); ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xlabel("mission time (s)"); ax.set_ylabel("task points collected")
    ax.set_title(f"Points over the mission ({len(seeds)} reproduced worlds, ideal comms, 95% bands)", loc="left", fontsize=9.5)
    save(fig, "fig10_timecourse")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 104); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, edge=INK2, fill="#f7f8f9", tcol=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.1))
        ax.text(x + 1.2, y + h - 2.2, title, fontsize=8, fontweight="bold", color=tcol, va="top")
        ax.text(x + 1.2, y + h - 6.0, body, fontsize=6.6, color=INK2, va="top", linespacing=1.35)

    def arrow(a, b, col=INK2, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=col, lw=1.0, linestyle=ls))

    box(1, 44, 29, 14, "Seeded exogenous tapes", "map · tasks · currents\ndisturbances · sensor noise\npacket loss & latency\n(named RNG streams)", edge=MUTED)
    box(1, 23, 29, 17, "World", "procedural course, islands,\nbarrier + one-lane channels,\ncharted + unknown current,\ntasks with deadlines")
    box(1, 2, 29, 17, "Physics", "twin-thruster USV, RK4 Δt=0.05 s\nv̇ = (v_cmd − v)/τ_v − k v|v|\nṙ = (r_cmd − r)/τ_r\n|v|/v_max + |r|/r_max ≤ 1")
    box(36, 2, 30, 17, "Guidance & control (10 Hz)", "LOS pursuit + speed schedule\nyaw-rate control, crab angle\ncollision avoidance (give way)\nchannel reservation (cap. 1)")
    box(36, 23, 30, 17, "Planner + ETA models", "Dijkstra on inflated 1 m grid\nLOS shortcutting\ngraph ETA = d / v̄\ncontrol ETA = rollout-calibrated")
    box(72, 23, 31, 17, "Policy (every 2 s)", "P0 greedy · P1 selfish race\nP2/A1/A2/P3 marginal contribution\n  U = v·s_i·(1 − s_−i) − λ·ETA\nP4 centralized Hungarian (ref.)", edge=BLUE, tcol=BLUE)
    box(72, 44, 31, 14, "Observation (info boundary)", "own state · public task list\nmessages actually received + age\nno teammate truth, no future", edge=BLUE, tcol=BLUE)
    box(36, 44, 30, 14, "Comms", "1 Hz broadcast: pose, claim,\npredicted finish, known-done\nGilbert–Elliott loss, 0.5–3 s delay")
    box(72, 2, 31, 17, "Outputs", "episodes.csv (1 row = 1 world×policy)\nETA log, manifest + hashes\n→ paired tests, Holm, bootstrap\nGUI reads the SAME engine", edge=MUTED)
    arrow((15.5, 44), (15.5, 40)); arrow((15.5, 23), (15.5, 19))
    arrow((30, 10), (36, 10)); arrow((51, 23), (51, 19))
    arrow((72, 31), (66, 31)); arrow((87.5, 44), (87.5, 40))
    arrow((66, 51), (72, 51)); arrow((30, 51), (36, 51), MUTED, "--")
    arrow((30, 31), (36, 31)); arrow((66, 10), (72, 10), MUTED, "--")
    ax.set_title("Simulator architecture: one engine, a strict information boundary, common random numbers",
                 loc="left", fontsize=9.5)
    save(fig, "fig0_architecture")


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "league", "h2h", "mech", "time"]
    df = load() if set(which) - {"arch"} else None
    fns = dict(arch=lambda: fig_architecture(), league=lambda: fig_league(df), h2h=lambda: fig_head_to_head(df),
               mech=lambda: fig_mechanism(df), time=lambda: fig_timecourse(df))
    for w in which:
        fns[w]()
