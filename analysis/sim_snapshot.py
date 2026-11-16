"""Annotated snapshot of the RoboBoat simulator for the paper (fig15_sim_snapshot).
Renders a real frame with the game renderer (whole course, no follow camera), then adds technical callouts
and a strip with the update rate of every loop."""
import math
import os
import sys
import argparse

os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import BLUE, INK, INK2, MUTED, ORANGE, save  # noqa: E402
from rb2.viz.app2 import Game, carrot  # noqa: E402

SEED, POL, T_SNAP, FOCUS = 11, "P4", 64.0, 2


def render():
    a = argparse.Namespace(seed=SEED, policy=POL, fleet=4, comm="degraded", variant="standard", shortcut=False,
                           perception="sensors", episode=300.0, width=2000, height=1500, record=None, seconds=0, ghost=None)
    g = Game(a)
    g.ghosts = []; g.show_phantoms = False; g.follow = False; g.focus = FOCUS
    while g.sim.t < T_SNAP:
        sdt = g.step(1 / 30); g.update_lidar(sdt); g.consume_events()
    g.popups = []
    fx, fy = g.sim.pos(FOCUS)
    g.cam.zoom = 2.3
    hw, hh = g.cam.rect.w / 2 / g.cam.s, g.cam.rect.h / 2 / g.cam.s
    c0 = g.sim.course
    g.cam.cx = min(max(fx - 4, hw), c0.width - hw); g.cam.cy = min(max(fy + 2, hh), c0.height - hh)
    g.draw()
    R = g.cam.rect
    v = g.cam.view()
    world = pygame.surfarray.array3d(g.screen.subsurface(R)).transpose(1, 0, 2).copy()
    P = lambda x, y: (v.p(x, y)[0] - R.x, v.p(x, y)[1] - R.y)
    sim, c = g.sim, g.sim.course
    tm = pygame.Surface((520, 420)); tm.fill((250, 251, 252))
    g.draw_teammap(tm, pygame.Rect(10, 6, 500, 400))
    teammap = pygame.surfarray.array3d(tm).transpose(1, 0, 2).copy()
    pts = {}
    pts["boats"] = [P(*sim.pos(i)) for i in range(sim.n)]
    ex = sim.exec[FOCUS]
    x, y = sim.pos(FOCUS)
    cr = carrot(ex.g, x, y, sim.dcfg.los_lookahead_m) if ex is not None and ex.g is not None else None
    pts["carrot"] = P(*cr[0]) if cr else None
    psi = sim.state(FOCUS)[2]
    pts["cone_tip"] = P(x + 11 * math.cos(psi), y + 11 * math.sin(psi))
    pts["path"] = None
    if ex is not None and ex.g is not None and len(ex.g.wps) > ex.g.seg + 2:
        q = ex.g.wps[-1]; pts["path"] = P(*q)
        if cr and math.hypot(P(*q)[0] - P(*cr[0])[0], P(*q)[1] - P(*cr[0])[1]) < 60:
            q2 = ex.g.wps[min(len(ex.g.wps) - 1, ex.g.seg + 1)]
            mq = ((q2[0] + x) / 2, (q2[1] + y) / 2); pts["path"] = P(*mq)
    tr = g.sim.know.maps[FOCUS].tracker
    fr = [f for f in tr.confirmed() if g.sim.t - f.last_t <= 0.6 and f.source == "own"]
    fr.sort(key=lambda f: math.hypot(f.x - x, f.y - y))
    pts["det"] = P(fr[0].x, fr[0].y) if fr else None
    m = {k: c.module(k)[0] for k in ("T1", "T2", "T3", "T4", "T6")}
    pts["T1"] = P(*sim.gate_mid(m["T1"], 1)); pts["T2"] = P(*sim.gate_mid(m["T2"], len(m["T2"].gates) // 2))
    pts["T3"] = P(m["T3"].x, m["T3"].y); pts["T4"] = P(*sim.light_xy()); pts["T6"] = P(*sim.gate_mid(m["T6"], 0))
    t5 = [c.obj(mm.objects[0]) for mm in c.module("T5")]
    pts["T5"] = P(t5[0].x, t5[0].y)
    pts["sector"] = P(sim.know.sector(0)[1], c.height * 0.93)
    jobs = []
    for i in range(sim.n):
        cl = sim.claim[i]
        jobs.append("idle" if cl is None else cl.kind)
    return world, teammap, pts, jobs, sim.t


def figure():
    world, teammap, pts, jobs, t = render()
    H, W = world.shape[:2]
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_axes([0.0, 0.25, 0.64, 0.70])
    ax.imshow(world); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    inside = lambda q: q is not None and 0 <= q[0] < W and 0 <= q[1] < H
    b = pts["boats"]
    items = [
        (pts["cone_tip"], "Camera: 90° field of view, 18 m, 5 Hz. Detection probability falls with range and off-axis angle; class confusions (red/orange, triangle/plus) grow with range; occlusion by ray casting.", BLUE),
        (b[FOCUS], "Lidar: 360°, 20 m, 10 Hz, positions only. Its returns are the dots that fade within a third of a second.", BLUE),
        (pts["det"], "Detection: tracker confirms after repeated or camera+lidar hits; label = most likely class and its confidence.", BLUE),
        (pts["path"], "Planned path from the executor: grid planner, then task-specific waypoints (gates, loop, bay entry).", INK2),
        (pts["carrot"], "Line-of-sight guidance: steer toward a point 4.5 m ahead on the path (2.2 m on precise segments); speed drops with heading error.", INK2),
        (pts["T2"], "T2 path: Catmull-Rom spline, gates every few metres; yellow buoys are counted from de-duplicated tracks.", INK2),
        (pts["T4"], "T4 start light: the boat holds position facing it and starts only when its camera reports green.", INK2),
        (pts["T3"], "T3 dock: banners readable within 7 m from the front; an occupied bay is ruled out; the last bay is inferred.", INK2),
        (pts["T5"], "T5 vessel: spray for 3 s (triangle) or drop a ball (plus); delivery tiers 75/125/200.", INK2),
        (pts["T1"], "T1 channel: every boat must pass both gates before anything it does scores.", INK2),
        (pts["T6"], "T6 home gate (3.2 m): queue-aware return timing; the first boat home takes the farthest berth.", INK2),
    ]
    shown = [(q, txt, col) for q, txt, col in items if inside(q)]
    fb = b[FOCUS]
    placed = []
    for k, (q, txt, col) in enumerate(shown, 1):
        mx, my = q
        dx, dy = q[0] - fb[0], q[1] - fb[1]
        d = math.hypot(dx, dy)
        if d < 90:                                   # near the focused boat: push the marker outward
            ang = math.atan2(dy, dx) if d > 5 else -2.4 + 0.9 * k
            mx, my = fb[0] + 95 * math.cos(ang), fb[1] + 95 * math.sin(ang)
        for (px, py) in placed:                      # avoid marker overlap
            if math.hypot(mx - px, my - py) < 30:
                my += 34
        placed.append((mx, my))
        if (mx, my) != tuple(q):
            ax.plot([q[0], mx], [q[1], my], color=col, lw=0.6, zorder=5)
            ax.plot(*q, "o", ms=2.2, color=col, zorder=6)
        ax.plot(mx, my, "o", ms=9, mfc="white", mec=col, mew=1.0, zorder=6)
        ax.text(mx, my, str(k), fontsize=5.6, ha="center", va="center", color=col, zorder=7, fontweight="bold")
    ax.set_title(f"One frame of the simulator, t = {t:.0f} s (central-assignment fleet, unreliable radio)",
                 fontsize=7.5, loc="left", color=INK)
    kx = fig.add_axes([0.655, 0.25, 0.34, 0.70]); kx.axis("off")
    yk = 1.0
    import textwrap
    for k, (q, txt, col) in enumerate(shown, 1):
        lines = textwrap.wrap(txt, 52)
        kx.text(0.0, yk, f"{k}", fontsize=6.3, color=col, fontweight="bold", va="top")
        kx.text(0.06, yk, "\n".join(lines), fontsize=6.0, color=INK2, va="top", linespacing=1.25)
        yk -= 0.027 * len(lines) + 0.018
    kx.text(0.0, yk - 0.005, "Thin coloured lines: each boat's planned path. Every boat broadcasts pose, claim,\nETA and a map digest at 1 Hz over a Gilbert–Elliott radio (burst loss, 0.5–3 s\nlatency).",
            fontsize=6.0, color=INK2, va="top", linespacing=1.25)
    ai = fig.add_axes([0.70, 0.19, 0.26, 0.17])
    ai.imshow(teammap); ai.axis("off")
    ai.set_title("Team map of boat 1: filled = seen itself, ring = heard by radio,\ntinted stripe = its recon sector", fontsize=5.6, color=INK2)
    sx = fig.add_axes([0.16, 0.05, 0.46, 0.15])
    loops = [("physics (RK4)", 20.0), ("guidance + control", 10.0), ("lidar", 10.0), ("camera", 5.0),
             ("radio broadcast", 1.0), ("policy decision", 0.5)]
    for k, (name, hz) in enumerate(loops):
        ts = np.arange(0, 2.0001, 1 / hz)
        sx.vlines(ts, k - 0.32, k + 0.32, color=BLUE if name in ("lidar", "camera") else INK2, lw=0.7)
        sx.text(-0.05, k, f"{name}  {hz:g} Hz", ha="right", va="center", fontsize=6)
    sx.set_xlim(0, 2.0); sx.set_ylim(len(loops) - 0.4, -0.6)
    sx.set_yticks([]); sx.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); sx.tick_params(labelsize=6)
    sx.set_xlabel("seconds (one 2 s decision cycle)", fontsize=6)
    for sp in ("top", "right", "left"):
        sx.spines[sp].set_visible(False)
    save(fig, "fig15_sim_snapshot")


if __name__ == "__main__":
    figure()
