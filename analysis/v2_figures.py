"""Figures for the RoboBoat-course (v2) part of the paper.

  python analysis/v2_figures.py arch | results | ablation | all
"""
import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, VIOLET, save  # noqa: E402

NAMES = {"P0": "Greedy nearest (P0)", "P1": "Selfish race (P1)", "P2": "Team-aware, graph ETA (P2)",
         "A1": "Team-aware, graph ETA, age-aware (A1)", "A2": "Team-aware, dynamics ETA (A2)",
         "P3": "Team-aware, dynamics ETA, age-aware (P3)", "P4": "Central assignment (P4)"}


def fig_arch():
    """3 x 2 loop with straight, orthogonal arrows; text sized for a full-width figure."""
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
    W, H = 25, 13.5
    X = [1.5, 35, 63.5]
    YT, YB = 22.5, 2.5

    def box(c, y, title, body, edge=INK2, tcol=INK):
        x = X[c]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.25,rounding_size=0.9", fc="#f7f8f9", ec=edge, lw=1.0))
        ax.text(x + 1.0, y + H - 1.4, title, fontsize=8.4, fontweight="bold", color=tcol, va="top")
        ax.text(x + 1.0, y + H - 4.9, body, fontsize=7.2, color=INK2, va="top", linespacing=1.35)

    def arrow(a, b, col=INK2):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=col, lw=1.0, shrinkA=0, shrinkB=0))

    ax.add_patch(FancyBboxPatch((X[1] - 2.6, 0.6), 100 - X[1] + 1.8, 36.8, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="#eef4fb", ec=BLUE, lw=0.8, ls=(0, (4, 3))))
    ax.text(X[1] - 1.8, 38.0, "Decisions use only what the boat's sensors and radio report", fontsize=7.2, color=BLUE, va="bottom")
    box(0, YT, "World and referee", "generated course, currents\ntwin-thruster boats (RK4)\n2025 task scoring (Table 1)")
    box(1, YT, "Sensors", "camera: 90°, 18 m, 5 Hz\nlidar: 360°, 20 m, 10 Hz\nclass errors grow with range", edge=BLUE, tcol=BLUE)
    box(2, YT, "Team map", "detections fused into objects\nwith class probabilities;\nteammates' maps merged", edge=BLUE, tcol=BLUE)
    box(2, YB, "Task opportunities", "tasks the map supports,\nshaped by the RoboBoat\nlayer (Section 4)", edge=ORANGE, tcol=ORANGE)
    box(1, YB, "Policy (every 2 s)", "P0 greedy, P1 selfish race\nP2 A1 A2 P3 R3 L3 team-aware\nP4 central oracle, P5 leader", edge=BLUE, tcol=BLUE)
    box(0, YB, "Task execution", "per-task state machines,\nline-of-sight steering,\ngive-way and separation")
    # radio: same height as the team map, inside the boundary
    rx = X[2] + W + 2.2
    ax.add_patch(FancyBboxPatch((rx, YT), 5.6, H, boxstyle="round,pad=0.25,rounding_size=0.9", fc="#f7f8f9", ec=INK2, lw=1.0))
    ax.text(rx + 2.8, YT + H / 2, "radio\n1 Hz\nlossy", fontsize=6.8, color=INK2, ha="center", va="center", linespacing=1.3)
    ym, yb = YT + H / 2, YB + H / 2
    arrow((X[0] + W + 0.4, ym), (X[1] - 0.4, ym), BLUE)
    arrow((X[1] + W + 0.4, ym), (X[2] - 0.4, ym), BLUE)
    arrow((X[2] + W + 0.3, ym + 2), (rx - 0.3, ym + 2)); arrow((rx - 0.3, ym - 2), (X[2] + W + 0.3, ym - 2))
    arrow((X[2] + W / 2, YT - 0.4), (X[2] + W / 2, YB + H + 0.4), ORANGE)
    arrow((X[2] - 0.4, yb), (X[1] + W + 0.4, yb), ORANGE)
    arrow((X[1] - 0.4, yb), (X[0] + W + 0.4, yb))
    arrow((X[0] + W / 2, YB + H + 0.4), (X[0] + W / 2, YT - 0.4))
    ax.text(X[0] + W / 2 + 1.0, (YB + H + YT) / 2, "thrust", fontsize=6.8, color=INK2, va="center")
    save(fig, "fig11_v2_architecture")


def fig_results():
    """Task points (mean, 95% CI) per strategy, both radios (filled vs hollow, neutral ink); bonus rate column."""
    ep = pd.read_csv(os.path.join(ROOT, "data/v2_replication/episodes.csv"))
    pols = ["P0", "P1", "P2", "A1", "A2", "P3", "P4"]
    order = ep[ep.comm == "degraded"].groupby("policy").task_points.mean().reindex(pols).sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    rng = np.random.default_rng(0)
    X0 = 2150
    for comm, off, mfc, lab in [("degraded", 0.15, INK, "unreliable radio"), ("ideal", -0.15, "white", "perfect radio")]:
        d = ep[ep.comm == comm]
        for k, p in enumerate(order):
            r = d[d.policy == p].task_points.values
            bs = rng.choice(r, (4000, len(r))).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            if r.mean() < X0:                      # greedy is far off-scale: arrow + value at the left edge
                ax.annotate(f"{r.mean():.0f}", xy=(X0 + 2, k + off), xytext=(X0 + 45, k + off), fontsize=7, color=INK2,
                            va="center", arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
                continue
            ax.plot([lo, hi], [k + off] * 2, color=INK, lw=1.1)
            ax.plot(r.mean(), k + off, "o", mfc=mfc, mec=INK, mew=1.0, ms=4.8, label=lab if k == len(order) - 1 else None)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[p] for p in order], fontsize=7.5)
    ax.set_xlim(X0, 2560)
    ax.set_xlabel("task points per run (mean and 95% CI; return bonus excluded)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(1.02, 1.02, "bonus earned\nunreliable / perfect", transform=ax.transAxes, fontsize=7, color=INK2, va="bottom", ha="left")
    for k, p in enumerate(order):
        b1 = (ep[(ep.comm == "degraded") & (ep.policy == p)].bonus > 0).mean()
        b2 = (ep[(ep.comm == "ideal") & (ep.policy == p)].bonus > 0).mean()
        ax.text(1.04, k, f"{100 * b1:.0f}%  /  {100 * b2:.0f}%", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=INK2, va="center", ha="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.subplots_adjust(left=0.32, right=0.83, top=0.86, bottom=0.18)
    save(fig, "fig12_v2_league")


def fig_ablation():
    S = json.load(open(os.path.join(ROOT, "data/v2_ablation/summary.json")))
    lab = {"no_sectors": "recon sectors", "no_bonus": "bonus-aware values", "no_deconflict": "sensed deconfliction",
           "no_queue_return": "queue-aware return", "no_dock_infer": "bay inference", "none": "all five"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, key, title in zip(axes, ["points", "task_points"], ["Total points", "Task points (no bonus)"]):
        for j, (pol, col, off) in enumerate([("P4", ORANGE, 0.15), ("P3", INK, -0.15)]):
            for k, v in enumerate(lab):
                e = S[f"{pol}/{v}"][key]
                if pol == "P4" and v == "no_deconflict":
                    ax.text(0, k + off, "  not used by P4", fontsize=6.5, color=MUTED, va="center", ha="left")
                    if k == 0:
                        ax.plot([], [], "o", color=col, label="central assignment (P4)")
                    continue
                ax.plot(e["ci"], [k + off] * 2, color=col, lw=1.2)
                ax.plot(e["full_minus"], k + off, "o", color=col, ms=4,
                        label=("team-aware fleet (P3)" if pol == "P3" else "central assignment (P4)") if k == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axhline(4.5, color=GRID, lw=1.0)
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(list(lab.values()), fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("points lost when switched off", fontsize=7.5)
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig13_v2_ablation")


def fig_rush():
    r = json.load(open(os.path.join(ROOT, "data/return_rush/results.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title in zip(axes, ["primary", "secondary"],
                              ["Geometry chosen in advance", "Geometry chosen after screening (fresh seeds)"]):
        g = r[key]
        conds = [("closed", "a_closed_selfish", "shortcut closed\nselfish"), ("open", "b_open_selfish", "shortcut open\nselfish"),
                 ("tolled", "c_open_tolled", "shortcut open\nmarginal-cost tolls")]
        for k, (pk, mk, lab) in enumerate(conds):
            pred = g["prediction"]["team_cost"][pk]["value"]
            m = g["conditions"][mk]
            ax.plot([k - 0.12, k + 0.12], [pred, m["total_mean"]], color="#bbbbbb", lw=1.0, zorder=1)
            ax.plot(k - 0.12, pred, "D", color=MUTED, ms=5, label="exact game prediction" if k == 0 else None)
            ax.plot([k + 0.12] * 2, m["total_ci"], color=BLUE, lw=1.4)
            ax.plot(k + 0.12, m["total_mean"], "o", color=BLUE, ms=5, label="measured (95% CI)" if k == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in conds], fontsize=7)
        ax.set_xlim(-0.5, 2.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 1, hi + (hi - lo) * 0.18)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
        ax.set_ylabel("team time to home (s)", fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig14_return_rush")


def fig_fleet():
    """Task points per strategy for 2 vs 4 boats on the same 200 courses (unreliable radio)."""
    e4 = pd.read_csv(os.path.join(ROOT, "data/v2_replication/episodes.csv"))
    e2 = pd.read_csv(os.path.join(ROOT, "data/v2_two_boats/episodes.csv"))
    pols = ["P1", "P2", "A1", "A2", "P3", "P4"]
    order = e4[e4.comm == "degraded"].groupby("policy").task_points.mean().reindex(pols).sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, ep, title in zip(axes, [e2, e4], ["Two boats", "Four boats"]):
        d = ep[ep.comm == "degraded"]
        for k, p in enumerate(order):
            r = d[d.policy == p].task_points.values
            bs = rng.choice(r, (4000, len(r))).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            col = ORANGE if p == "P4" else INK
            ax.plot([lo, hi], [k, k], color=col, lw=1.2)
            ax.plot(r.mean(), k, "o", color=col, ms=4.8)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlim(2000, 2560)
        ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("task points per run, unreliable radio", fontsize=7.5)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([NAMES[p] for p in order], fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig17_fleet_size")


def fig_physical():
    """(a,b) exact-game prediction vs measured team time, single-file queues and slow zones;
    (c) four-cycle circulations of the measured games against seed noise and the separable game's exact zero."""
    q = json.load(open(os.path.join(ROOT, "data/return_rush/results.json")))["primary"]
    z = json.load(open(os.path.join(ROOT, "data/return_rush_slowzone/results.json")))["primary"]
    C = json.load(open(os.path.join(ROOT, "data/return_rush_coupling/results.json")))["D"]
    fig = plt.figure(figsize=(7.4, 2.7))
    axes = [fig.add_axes([0.075, 0.24, 0.26, 0.62]), fig.add_axes([0.40, 0.24, 0.26, 0.62]), fig.add_axes([0.745, 0.24, 0.245, 0.62])]
    conds = [("closed", "a_closed_selfish", "closed"), ("open", "b_open_selfish", "open"), ("tolled", "c_open_tolled", "open\n+ tolls")]
    for ax, g, title in [(axes[0], q, "(a) Single-file channels"), (axes[1], z, "(b) Slow zones")]:
        for k, (pk, mk, lab) in enumerate(conds):
            pred = g["prediction"]["team_cost"][pk]["value"]; m = g["conditions"][mk]
            ax.plot([k - 0.13, k + 0.13], [pred, m["total_mean"]], color="#bbbbbb", lw=1.0, zorder=1)
            ax.plot(k - 0.13, pred, "D", color=MUTED, ms=5, label="game prediction" if k == 0 else None)
            ax.plot([k + 0.13] * 2, m["total_ci"], color=BLUE, lw=1.4)
            ax.plot(k + 0.13, m["total_mean"], "o", color=BLUE, ms=5, label="measured" if k == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in conds], fontsize=7.5)
        ax.set_xlim(-0.5, 2.5); ax.tick_params(axis="y", labelsize=7.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 2, hi + (hi - lo) * 0.12)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("fleet time to home (s)", fontsize=7.5)
    axes[0].legend(fontsize=6.8, frameon=False, loc="lower left")
    ax = axes[2]
    for key, col, lab in [("queue", BLUE, "single-file"), ("slow_zone", INK2, "slow zones")]:
        v = np.array(C[key]["abs_sorted"]); y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=col, lw=1.5, label=lab)
        ax.axvline(C[key]["noise_max95"], color=col, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(0, color=INK, lw=1.6)
    ax.text(0.6, 0.06, "separable\ngame: all 0", fontsize=6.5, color=INK, va="bottom")
    ax.text(8.5, 0.20, "dashed: largest\ncirculation from\nseed noise (95%)", fontsize=6.2, color=INK2, va="top")
    ax.set_xlabel("|circulation| around a four-cycle (s)", fontsize=7.5)
    ax.set_ylabel("fraction of 1,944 cycles", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title("(c) Loops that do not close", loc="left", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, loc="center right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "fig_physical")


def fig_league():
    """Task points per strategy (unreliable radio) for 2 and 4 boats on the same 200 courses, with duplicate
    claims abandoned per run (4 boats) and the single-boat reference."""
    def load(paths):
        return pd.concat([pd.read_csv(os.path.join(ROOT, p_)) for p_ in paths if os.path.exists(os.path.join(ROOT, p_))],
                         ignore_index=True)
    e4 = load(["data/v2_replication/episodes.csv", "data/v2_new_policies_n4/episodes.csv", "data/v2_lease_n4/episodes.csv"])
    e2 = load(["data/v2_two_boats/episodes.csv", "data/v2_new_policies_n2/episodes.csv", "data/v2_lease_n2/episodes.csv"])
    y4 = load(["data/v2_yields_n4/episodes.csv", "data/v2_new_policies_n4/episodes.csv", "data/v2_lease_n4/episodes.csv"])
    one = pd.read_csv(os.path.join(ROOT, "data/v2_one_boat/episodes.csv"))
    names = dict(NAMES, P5="Leader on radio (P5)", R3="Reservations (R3)", L3="Heartbeat lease (L3)")
    pols = ["P5", "A1", "P3", "R3", "L3", "P2", "A2", "P1", "P4"]
    e4d = e4[e4.comm == "degraded"]
    order = e4d.groupby("policy").task_points.mean().reindex(pols).sort_values().index.tolist()
    fig = plt.figure(figsize=(7.4, 3.0))
    a2 = fig.add_axes([0.30, 0.17, 0.27, 0.72]); a4 = fig.add_axes([0.62, 0.17, 0.27, 0.72])
    rng = np.random.default_rng(0)
    ref = one[one.comm == "degraded"].groupby("policy").task_points.mean().get("A2", np.nan)
    for ax, ep, title in [(a2, e2, "Two boats"), (a4, e4, "Four boats")]:
        d = ep[ep.comm == "degraded"]
        for k, p in enumerate(order):
            r = d[d.policy == p].task_points.values
            if not len(r):
                continue
            bs = rng.choice(r, (4000, len(r))).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            col = ORANGE if p == "P4" else (VIOLET if p in ("R3", "L3", "P5") else INK)
            ax.plot([lo, hi], [k, k], color=col, lw=1.2)
            ax.plot(r.mean(), k, "o", color=col, ms=4.8)
        ax.axvline(ref, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.set_xlim(1350, 2560)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[p] for p in order] if ax is a2 else [], fontsize=7.3)
        ax.tick_params(axis="x", labelsize=7.3)
        ax.grid(axis="y", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("task points per run", fontsize=7.5)
    a2.text(ref + 15, len(order) - 1, "one boat", fontsize=6.5, color=INK2, va="center")
    yd = y4[y4.comm == "degraded"].groupby("policy").yields.mean()
    a4.text(1.04, len(order) - 0.35, "duplicates\nper run", transform=a4.get_yaxis_transform(), fontsize=6.6, color=INK2, va="bottom")
    for k, p in enumerate(order):
        if p in yd:
            a4.text(1.08, k, f"{yd[p]:.1f}", transform=a4.get_yaxis_transform(), fontsize=7.3, color=INK2, va="center")
    save(fig, "fig_league")


if __name__ == "__main__":
    w = sys.argv[1:] or ["all"]
    if "arch" in w or "all" in w: fig_arch()
    if "results" in w or "all" in w: fig_results()
    if "ablation" in w or "all" in w: fig_ablation()
    if "rush" in w or "all" in w: fig_rush()
    if "fleet" in w or "all" in w: fig_fleet()
    if "physical" in w or "all" in w: fig_physical()
    if "league" in w or "all" in w: fig_league()


