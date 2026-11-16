"""Regenerate every paper figure from data files and the exact game (no hand-entered numbers).

  python analysis/make_figures.py
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# validated reference palette (light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#eb6834"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------
def fig_braess():
    """(a) harbour network, (b) one edge: staircase (potential) vs rectangle (team cost) split into marginal-cost
    L-strips, (c) best-response trajectories of Phi and C."""
    from src.game_exact import summarize_braess
    from matplotlib.patches import Rectangle, Polygon
    s = summarize_braess(4)
    fig = plt.figure(figsize=(7.4, 2.6))
    ax = fig.add_axes([0.0, 0.08, 0.27, 0.80])
    sx = fig.add_axes([0.33, 0.30, 0.24, 0.58])
    bx = fig.add_axes([0.66, 0.20, 0.33, 0.68])
    # (a) network
    pos = {"S": (0, 0), "A": (1, 0.8), "B": (1, -0.8), "T": (2, 0)}
    lab = {("S", "A"): "2k", ("A", "T"): "10", ("S", "B"): "10", ("B", "T"): "2k", ("A", "B"): "1"}
    off = {("S", "A"): (-0.17, 0.14), ("A", "T"): (0.17, 0.14), ("S", "B"): (-0.17, -0.14), ("B", "T"): (0.17, -0.14),
           ("A", "B"): (0.2, 0.0)}
    for (u, v), l in lab.items():
        (x0, y0), (x1, y1) = pos[u], pos[v]
        cong = l == "2k"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2 if cong else 1.3,
                                    color=INK if cong else (VIOLET if l == "1" else MUTED), shrinkA=14, shrinkB=14))
        mx, my = (x0 + x1) / 2 + off[(u, v)][0], (y0 + y1) / 2 + off[(u, v)][1]
        ax.text(mx, my, l, ha="center", va="center", fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for n_, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.11, fc="white", ec=INK, lw=1.3, zorder=3))
        ax.text(x, y, n_, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    ax.text(1.0, -1.18, "minutes per boat when k boats use the edge", fontsize=7.2, color=INK2, ha="center", va="top")
    ax.set_xlim(-0.25, 2.25); ax.set_ylim(-1.4, 1.05); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) The harbour network", loc="left", fontsize=9)
    # (b) staircase vs rectangle on one canal, n = 3 boats, c(k) = 2k
    c = lambda k: 2 * k
    n = 3
    cols = ["#d9e6f5", "#b3cdeb", "#8db4e1"]
    for k in range(1, n + 1):          # L-strip m(k): own column k up to c(k) + band c(k-1)..c(k) over columns 1..k-1
        sx.add_patch(Rectangle((k - 1, 0), 1, c(k), fc=cols[k - 1], ec="white", lw=0.8))
        if k > 1:
            sx.add_patch(Rectangle((0, c(k - 1)), k - 1, c(k) - c(k - 1), fc=cols[k - 1], ec="white", lw=0.8, hatch="////", alpha=1.0))
    stair = [(0, 0)] + [(x, c(k)) for k in range(1, n + 1) for x in (k - 1, k)] + [(n, 0)]
    sx.add_patch(Polygon(stair, closed=True, fc="none", ec=BLUE, lw=1.6))
    sx.add_patch(Rectangle((0, 0), n, c(n), fc="none", ec=ORANGE, lw=1.6))
    for k in range(1, n + 1):
        sx.text(k - 0.5, c(k - 1) / 2 if k > 1 else c(k) / 2, f"c({k})", ha="center", va="center", fontsize=7, color=INK)
    sx.set_xlim(0, n + 0.05); sx.set_ylim(0, c(n) + 0.6)
    sx.set_xticks([0.5, 1.5, 2.5]); sx.set_xticklabels(["boat 1", "2", "3"], fontsize=7)
    sx.set_yticks([0, 2, 4, 6]); sx.tick_params(axis="y", labelsize=7)
    sx.grid(False)
    sx.set_ylabel("minutes", fontsize=7.5)
    sx.set_title("(b) One canal, three boats", loc="left", fontsize=9)
    for k in range(2, n + 1):
        sx.text((k - 1) / 2, (c(k - 1) + c(k)) / 2, f"m({k})", ha="center", va="center", fontsize=6.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    sx.text(0.0, -0.22, "staircase (blue) = Rosenthal share 2+4+6 = 12\nrectangle (orange) = fleet share 3·6 = 18\n"
            "L-strips m(1)+m(2)+m(3) = 2+6+10 = 18", transform=sx.transAxes, fontsize=6.6, color=INK2, va="top")
    # (c) trajectories
    tr, tm = s["trace_selfish"], s["trace_marginal"]
    xs = list(range(len(tr)))
    bx.plot(xs, [float(t.phi) for t in tr], color=BLUE, marker="o", ms=4.5, label=r"potential $\Phi$")
    bx.plot(xs, [float(t.team_cost) for t in tr], color=ORANGE, marker="o", ms=4.5, label=r"fleet cost $C$")
    xm = list(range(len(tr) - 1, len(tr) - 1 + len(tm)))
    bx.plot(xm, [float(t.team_cost) for t in tm], color=ORANGE, ls="--", marker="s", ms=3.8, label=r"$C$ under tolls ($=\Phi_m$)")
    A1 = s["shortcut"]
    bx.axhline(float(A1.optimum_cost), color=MUTED, lw=1, ls=":")
    bx.text(7.9, float(A1.optimum_cost) - 2.8, "optimum", color=INK2, fontsize=7, ha="right")
    bx.set_xlabel("best-response move", fontsize=7.5); bx.set_ylabel("boat-minutes", fontsize=7.5)
    bx.tick_params(labelsize=7)
    bx.set_title(r"(c) Selfish moves lower $\Phi$, raise $C$", loc="left", fontsize=9)
    bx.set_ylim(38, 72)
    bx.legend(loc="lower right", fontsize=6.6, bbox_to_anchor=(1.0, 0.0))
    save(fig, "fig1_braess")


def fig_trajectories(seed=20007):
    from src.config import config_from, load_yaml
    from src.sim import Simulation
    spec = load_yaml(os.path.join(ROOT, "configs/confirmatory.yaml"))
    cfg = config_from({**spec["base"], "comm_mode": "degraded", "policy": "P3"})
    sim = Simulation(cfg, seed, log_steps=True)
    ages = []
    while not sim.done:
        sim.step()
        if sim.k % 10 == 0:
            row = [sim.t]
            for i in range(sim.n):
                bel = sim.minds[i].beliefs
                row.append(np.mean([sim.t - m.t_sent for m in bel.values()]) if bel else np.nan)
            ages.append(row)
    W = sim.world
    tr = np.array(sim.trace)              # (T, 4, n)
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.0, 0.3, 1.0, 0.68])
    ax.imshow(W.blocked, origin="lower", extent=(0, W.width, 0, W.height), cmap=LinearSegmentedColormap.from_list(
        "m", ["#f4f8fb", "#9b958c"]), interpolation="nearest")
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    for i in range(sim.n):
        ax.plot(tr[:, 0, i], tr[:, 1, i], color=cols[i], lw=1.3, label=f"B{i + 1}")
        ax.plot(tr[0, 0, i], tr[0, 1, i], "o", color=cols[i], ms=4)
    mark = {"gate": "s", "inspect": "o", "dock": "D", "sample": "^"}
    for tk in sim.tasks:
        done = tk.completed_by is not None
        ax.plot(tk.x, tk.y, mark[tk.kind], ms=7, mfc=cols[tk.completed_by] if done else "white",
                mec=INK, mew=1.0)
        ax.text(tk.x + 2.2, tk.y + 2.2, f"{tk.value}", fontsize=6.5, color=INK2)
    for ch in W.channels:
        ax.add_patch(plt.Rectangle((ch.x0, ch.y0), ch.x1 - ch.x0, ch.y1 - ch.y0, fill=False, ec=RED, lw=1, ls="--"))
    ax.set_xlim(0, W.width); ax.set_ylim(0, W.height); ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower right", ncol=4, fontsize=7.5, frameon=True, framealpha=0.9)
    m = sim.metrics()
    ax.set_title(f"(a) P3, degraded comms, world {seed}: W = {m['welfare']:.1f}, {m['completed']}/{m['n_tasks']} tasks\n"
                 "markers: task (filled = completed, in completing boat's colour); dashed box = 1-lane channel",
                 loc="center", fontsize=8.5)
    bx = fig.add_axes([0.08, 0.06, 0.9, 0.18])
    A = np.array(ages)
    for i in range(sim.n):
        bx.plot(A[:, 0], A[:, 1 + i], color=cols[i], lw=1.1)
    bx.set_ylabel("mean msg\nage (s)"); bx.set_xlabel("time (s)")
    bx.set_title("(b) age of the newest message each boat holds from its teammates (bursty loss)", loc="left", fontsize=8.5)
    save(fig, "fig2_trajectories")


def fig_effects():
    p = os.path.join(ROOT, "data/confirmatory/results.json")
    if not os.path.exists(p):
        print("skip effects (no results)"); return
    r = json.load(open(p))
    df = pd.read_csv(os.path.join(ROOT, "data/confirmatory/episodes.csv"))
    rows = [(f"{k}: {v['a']}−{v['b']} ({v['condition']})", v, True) for k, v in r["primary"].items()]
    rows += [(f"{s['name']} ({s['condition']})" if s['name'].split(':')[0] not in s['name'][:3] else s['name'], s, False)
             for s in r["secondary"]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for k, (lab, s, prim) in enumerate(rows):
        y = len(rows) - k
        lo, hi = s["ci_mean"]
        ax.plot([lo, hi], [y, y], color=BLUE if prim else MUTED, lw=2.4 if prim else 1.6, solid_capstyle="round")
        ax.plot(s["mean"], y, "o", color=BLUE if prim else INK2, ms=7 if prim else 5, mec="white", mew=1.2)
        txt = lab if prim else f"{s['name']} [{s['condition']}]"
        ax.text(-0.02, y, txt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7,
                color=INK if prim else INK2, fontweight="bold" if prim else "normal")
        if prim:
            ax.text(hi + 0.6, y, f"Holm p={s['p_holm']:.2g}", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([]); ax.set_xlabel("difference in welfare W, 95% CI\n(bold: preregistered)")
    ax.set_title("(a) paired contrasts", loc="left")
    w = df.pivot_table(index=["condition", "seed"], columns="policy", values="welfare")
    d1 = (w.loc["ideal"]["P3"] - w.loc["ideal"]["A1"]).to_numpy()
    d2 = (w.loc["degraded"]["P3"] - w.loc["degraded"]["A2"]).to_numpy()
    bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 41)
    bx.hist(d1, bins=bins, color=BLUE, alpha=0.75, label="H1: P3−A1 (ideal)")
    bx.hist(d2, bins=bins, color=ORANGE, alpha=0.6, label="H2: P3−A2 (degraded)")
    bx.axvline(0, color=INK2, lw=1)
    bx.set_xlabel("per-world paired difference"); bx.set_ylabel("worlds")
    bx.set_title(f"(b) per-world differences, N={len(d1)}", loc="left")
    bx.legend(fontsize=7)
    fig.subplots_adjust(left=0.33, wspace=0.35)
    save(fig, "fig3_effects")


def fig_policies():
    p = os.path.join(ROOT, "data/confirmatory/episodes.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    order = ["P0", "P1", "P2", "A1", "A2", "P3", "P4"]
    names = {"P0": "P0 greedy", "P1": "P1 selfish race", "P2": "P2 MC graph/naive", "A1": "A1 MC graph/age",
             "A2": "A2 MC control/naive", "P3": "P3 MC control/age", "P4": "P4 centralized (ref.)"}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, cond, col in zip(axs, ["ideal", "degraded"], [BLUE, ORANGE]):
        for k, pol in enumerate(order):
            v = df[(df.condition == cond) & (df.policy == pol)].welfare
            m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
            ax.plot([m - 1.96 * se, m + 1.96 * se], [k, k], color=col, lw=2, solid_capstyle="round")
            ax.plot(m, k, "o", color=col, ms=6, mec="white", mew=1)
            ax.text(m + 1.96 * se + 0.8, k, f"{m:.1f}", va="center", fontsize=7, color=INK2)
        ax.set_title(f"{cond} communication", loc="left")
        ax.set_xlabel("mean welfare W (95% CI)")
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([names[p] for p in order])
    save(fig, "fig6_policies")


def fig_eta():
    p = os.path.join(ROOT, "data/confirmatory/eta_log.csv")
    if not os.path.exists(p):
        return
    e = pd.read_csv(p)
    e = e[e.condition == "ideal"]
    e["complex"] = e.turn + e.init_err
    bins = [0, 1, 2, 3, 4, 20]
    e["bin"] = pd.cut(e.complex, bins, labels=["0–1", "1–2", "2–3", "3–4", ">4"])
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    xs = np.arange(5)
    for off, col, key, lab in ((-0.17, BLUE, "eta_graph", "graph ETA (calibrated)"),
                               (0.17, ORANGE, "eta_control", "control-aware ETA")):
        g = e.groupby("bin", observed=False).apply(lambda x: (x[key] - x.realized).to_numpy(), include_groups=False)
        ax.boxplot([g[b] for b in g.index], positions=xs + off, widths=0.3, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.45, edgecolor=col), medianprops=dict(color=INK),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        ax.plot([], [], color=col, lw=6, alpha=0.5, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    counts = e.bin.value_counts().reindex(["0–1", "1–2", "2–3", "3–4", ">4"]).to_numpy()
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(["0–1", "1–2", "2–3", "3–4", ">4"], counts)])
    ax.set_xlabel("turn complexity: Σ corner angles + initial heading error (rad)")
    ax.set_ylabel("predicted − realized (s)")
    ax.set_title("ETA error at commitment (ideal comms, all policies)", loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "fig5_eta_error")


def fig_heatmaps():
    have_f = os.path.exists(os.path.join(ROOT, "data/factorial/episodes.csv"))
    have_s = os.path.exists(os.path.join(ROOT, "data/scale/episodes.csv"))
    if not (have_f or have_s):
        print("skip heatmaps"); return
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    if have_f:
        f = pd.read_csv(os.path.join(ROOT, "data/factorial/episodes.csv"))
        w = f.pivot_table(index=["condition", "seed"], columns="policy", values="welfare")
        for ax, (a, b, title) in zip(axs[:2], [("P3", "A1", "P3−A1 (ETA model)"), ("P3", "A2", "P3−A2 (belief model)")]):
            M = np.zeros((2, 4)); SE = np.zeros((2, 4))
            cols = ["barrier\nideal", "barrier\ndegraded", "open\nideal", "open\ndegraded"]
            for i, ag in enumerate(["agile", "sluggish"]):
                for j, (bb, cc) in enumerate([("barrier", "ideal"), ("barrier", "degraded"), ("open", "ideal"), ("open", "degraded")]):
                    x = w.loc[f"{ag}_{bb}_{cc}"]
                    d = x[a] - x[b]
                    M[i, j], SE[i, j] = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
            lim = max(4, np.abs(M).max())
            ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            for i in range(2):
                for j in range(4):
                    ax.text(j, i, f"{M[i, j]:+.1f}\n±{1.96 * SE[i, j]:.1f}", ha="center", va="center", fontsize=7, color=INK)
            ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=6.5)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["agile", "sluggish"])
            ax.grid(False)
            ax.set_title(title, loc="left", fontsize=8.5)
    if have_s:
        s = pd.read_csv(os.path.join(ROOT, "data/scale/episodes.csv"))
        s["n"] = s.condition.str.extract(r"n(\d+)_").astype(int)
        s["comm"] = s.condition.str.extract(r"_(\w+)$")
        ax = axs[2]
        for pol, col in (("P4", BLUE), ("P3", ORANGE), ("P2", INK2)):
            for comm, ls in (("ideal", "-"), ("degraded", "--")):
                g = s[(s.policy == pol) & (s.comm == comm)].groupby("n")
                y = g.points_frac.mean()
                ax.plot(y.index, y.values, ls, color=col, marker="o", ms=3.5, lw=1.5,
                        label=pol if comm == "ideal" else None)
        ax.set_xscale("log", base=2); ax.set_xticks([3, 4, 6, 8, 12, 16]); ax.set_xticklabels([3, 4, 6, 8, 12, 16])
        ax.set_xlabel("fleet size (density held fixed)"); ax.set_ylabel("share of task points")
        ax.set_title("scaling (solid ideal, dashed degraded)", loc="left", fontsize=8.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_interactions_scale")


if __name__ == "__main__":
    which = sys.argv[1:] or ["braess", "traj", "effects", "policies", "eta", "heat"]
    fns = dict(braess=fig_braess, traj=fig_trajectories, effects=fig_effects, policies=fig_policies,
               eta=fig_eta, heat=fig_heatmaps)
    for w in which:
        fns[w]()
