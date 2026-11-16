"""Figure: one real decision on a RoboBoat course, made three ways (selfish argmax, joint assignment,
marginal contribution). Uses the simulator's own utilities at seed 11, t = 36 s (central-assignment run)."""
import argparse, math, os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
import numpy as np, pygame  # noqa: E402
import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, save  # noqa: E402
from rb2.viz.app2 import Game  # noqa: E402
from rb2.viz import style as S  # noqa: E402
from src.policies import control_eta, success_prob  # noqa: E402

SEED, T = 11, 36.0
FLEET = ["#%02x%02x%02x" % c for c in S.FLEET[:4]]


def build():
    a = argparse.Namespace(seed=SEED, policy="P4", fleet=4, comm="degraded", variant="standard", shortcut=False,
                           perception="sensors", episode=300.0, width=1700, height=1150, record=None, seconds=0, ghost=None)
    g = Game(a)
    g.ghosts = []; g.show_phantoms = False; g.follow = False
    g.show_paths = g.show_cones = g.show_det = g.show_lidar = False
    while g.sim.t < T:
        g.step(1 / 30); g.consume_events()
    g.popups = []
    sim = g.sim
    lam = sim.dcfg.time_cost
    # columns: shared tasks + one personal "search" column per boat (its best frontier tile)
    shared = {}
    for i in range(sim.n):
        for o in sim.opportunities(i):
            if o.kind not in ("explore", "T1", "T6"):
                shared.setdefault(o.key, o)
    cols = list(shared.values())
    names = {"T2": "Follow\nthe path", "T4": "Speed\nchallenge", "T3:look": "Read dock\nbanners"}
    U = np.full((sim.n, len(cols) + 1), np.nan); E = np.full_like(U, np.nan); V = np.full_like(U, np.nan)
    search_xy = []
    for i in range(sim.n):
        own = {o.key: o for o in sim.opportunities(i)}
        for j, o in enumerate(cols):
            if o.key in own:
                tv = sim.taskview(o); e, _ = control_eta(sim.chart, sim.state(i), tv, sim.dcfg, sim.tables)
                U[i, j] = o.value * success_prob(sim.t, e, tv.deadline, sim.dcfg) - lam * e; E[i, j] = e; V[i, j] = o.value
        best = None
        for o in own.values():
            if o.kind == "explore":
                tv = sim.taskview(o); e, _ = control_eta(sim.chart, sim.state(i), tv, sim.dcfg, sim.tables)
                u = o.value * success_prob(sim.t, e, tv.deadline, sim.dcfg) - lam * e
                if best is None or u > best[0]:
                    best = (u, e, o)
        U[i, -1], E[i, -1], V[i, -1] = best[0], best[1], best[2].value
        search_xy.append(best[2].xy)
    # selfish argmax (each boat alone) and joint assignment (search is personal, so it never conflicts)
    alone = [int(np.nanargmax(U[i])) for i in range(sim.n)]
    big = np.full((sim.n, len(cols) + sim.n), -1e9)
    big[:, :len(cols)] = np.nan_to_num(U[:, :len(cols)], nan=-1e9)
    for i in range(sim.n):
        big[i, len(cols) + i] = U[i, -1]
    r, c = linear_sum_assignment(-big)
    joint = [int(cc) if cc < len(cols) else len(cols) for cc in c]
    # render map
    v = g.cam.view(); R = g.cam.rect
    g.draw()
    img = pygame.surfarray.array3d(g.screen.subsurface(R)).transpose(1, 0, 2).copy()
    P = lambda x, y: (v.p(x, y)[0] - R.x, v.p(x, y)[1] - R.y)
    boats = [P(*sim.pos(i)) for i in range(sim.n)]
    tx = [P(*o.xy) for o in cols]
    sx = [P(*q) for q in search_xy]
    labels = [names.get(o.key, o.key) for o in cols] + ["Search own\nsector"]
    return img, boats, tx, sx, U, E, V, alone, joint, labels, lam, cols


def main():
    img, boats, tx, sx, U, E, V, alone, joint, labels, lam, cols = build()
    n, m = U.shape
    allp = np.array(boats + tx + sx)
    x0, y0 = allp.min(0) - 70; x1, y1 = allp.max(0) + 70
    fig = plt.figure(figsize=(7.4, 4.9))

    def panel(rect, choice, title):
        ax = fig.add_axes(rect)
        ax.imshow(img); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
        for j, q in enumerate(tx):
            ax.text(q[0], q[1] - 26, labels[j].replace("\n", " "), fontsize=6, ha="center", color=INK, va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
            ax.plot(*q, "s", ms=5, mfc="white", mec=INK, mew=0.8)
        same = {}
        for i in range(len(boats)):
            same.setdefault(choice[i] if choice[i] < len(tx) else ("s", i), []).append(i)
        for i, b in enumerate(boats):
            j = choice[i]
            tgt = tx[j] if j < len(tx) else sx[i]
            grp = same[j if j < len(tx) else ("s", i)]
            k = grp.index(i); nk = len(grp)
            if nk > 1:                                   # fan the arrowheads side by side under a shared target
                order = sorted(grp, key=lambda q: boats[q][0])
                kk = order.index(i)
                tgt = (tgt[0] + (kk - (nk - 1) / 2) * 24, tgt[1] + 16)
            ax.add_patch(FancyArrowPatch(b, tgt, arrowstyle="-|>", mutation_scale=8, color=FLEET[i], lw=1.4,
                                         shrinkA=9, shrinkB=3 if nk > 1 else 7, connectionstyle="arc3,rad=0.12"))
            ax.text(b[0], b[1] + 22, f"boat {i + 1}", fontsize=5.8, color=FLEET[i], ha="center", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
            if j == len(tx):
                ax.plot(*sx[i], "o", ms=4.5, mfc="white", mec=FLEET[i], mew=0.9)
                near = any(abs(sx[i][0] - q[0]) < 90 and abs(sx[i][1] - (q[1] - 26)) < 30 for q in tx)
                dx_, dy_, va_ = (16, 26, "top") if near else (9, -8, "bottom")
                ax.text(sx[i][0] + dx_, sx[i][1] + dy_, "search", fontsize=5.6, color=FLEET[i], ha="left", va=va_,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
        ax.set_title(title, fontsize=7.6, loc="left", color=INK)
        return ax

    panel([0.0, 0.47, 0.49, 0.49], alone, "(a) Each boat picks its own best task")
    panel([0.51, 0.47, 0.49, 0.49], joint, "(b) Joint assignment (central planner)")

    # (c) utility matrix
    ax = fig.add_axes([0.076, 0.0, 0.66, 0.47]); ax.axis("off")      # centres the matrix body (not its row labels) on the page
    ax.set_xlim(-1.6, m); ax.set_ylim(n + 0.5, -1.0)
    for j, lab in enumerate(labels):
        ax.text(j + 0.5, -0.08, lab, ha="center", va="bottom", fontsize=6.2, color=INK)
    for i in range(n):
        ax.text(-0.15, i + 0.5, f"boat {i + 1}", ha="right", va="center", fontsize=6.4, color=FLEET[i])
        for j in range(m):
            u = U[i, j]
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc="#f4f5f6" if np.isfinite(u) else "white", ec=GRID, lw=0.6))
            if np.isfinite(u):
                ax.text(j + 0.5, i + 0.42, f"{u:.0f}", ha="center", va="center", fontsize=7, color=INK)
                ax.text(j + 0.5, i + 0.75, f"{V[i, j]:.0f} − {lam:g}·{E[i, j]:.0f}", ha="center", va="center", fontsize=4.9, color=MUTED)
            else:
                ax.text(j + 0.5, i + 0.5, "not known", ha="center", va="center", fontsize=4.9, color=MUTED)
            if alone[i] == j:
                ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc="none", ec=MUTED, lw=1.0, ls=(0, (3, 2))))
            if joint[i] == j:
                ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fc="none", ec=ORANGE, lw=1.5))
    ax.text(m / 2, -0.85, "(c) Utility of each task for each boat:  U = value − λ·ETA,  λ = 2 points/s", fontsize=7.6, color=INK, fontweight="bold", ha="center")
    ax.text(m / 2, n + 0.15, "Dashed: each boat’s own best.  Orange: the joint assignment.",
            fontsize=6.2, color=INK2, va="top", ha="center")

    save(fig, "fig15_decision")
    print("alone", alone, "joint", joint)


if __name__ == "__main__":
    main()
