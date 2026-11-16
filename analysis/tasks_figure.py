"""Paper version of the six-task gallery: one clean panel per task, cropped to water, few labels, PDF."""
import math, os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
import numpy as np, pygame  # noqa: E402
import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import INK, save  # noqa: E402
from rb2.generate import generate_course, GenConfig  # noqa: E402
from rb2.course import TASK_NAMES  # noqa: E402
from rb2.viz import style as S  # noqa: E402
from rb2.viz.gallery import View, draw_course, draw_sensors  # noqa: E402

pygame.init()
_dl = S.detection_label
S.detection_label = lambda surf, rect, name, conf, size=10, water=S.WATER: _dl(surf, rect, name, conf, 17, water)
PW, PH = 440, 357


def view_for(c, cx, cy, R, rect):
    hx, hy = R * PW / PH, R
    cx = min(max(cx, hx), c.width - hx); cy = min(max(cy, hy), c.height - hy)
    return View(rect, cx - hx, cy - hy, cx + hx, cy + hy)


def panel(c, kind):
    surf = pygame.Surface((PW, PH)); surf.fill(S.PAPER)
    rect = pygame.Rect(0, 0, PW, PH)
    ms = c.module(kind); m = ms[0]
    if kind == "T5":
        vs = [c.obj(mm.objects[0]) for mm in ms]
        pair = [next(o for o in vs if o.cls == "vessel_orange"), next(o for o in vs if o.cls == "vessel_black")]
        for j, o in enumerate(pair):
            sub = pygame.Rect(j * PW // 2, 0, PW // 2 - 2, PH)
            vv = View(sub, o.x - 5.2, o.y - 8.4, o.x + 5.2, o.y + 8.4)
            surf.set_clip(sub); draw_course(surf, vv, c)
            bx, by = o.x - 3.4 * math.cos(o.heading + math.pi / 2), o.y - 3.4 * math.sin(o.heading + math.pi / 2)
            psi = math.atan2(o.y - by, o.x - bx)
            A0, A1 = vv.p(bx, by), vv.p(o.x, o.y)
            if o.cls == "vessel_orange":
                pts = [(A0[0] + (A1[0] - A0[0]) * f, A0[1] + (A1[1] - A0[1]) * f - 11 * math.sin(math.pi * f)) for f in np.linspace(0, 1, 20)]
                lay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                S.aa_polyline(lay, (*S.SEM["blue"][0], 170), pts, 2.0); surf.blit(lay, (0, 0))
            else:
                S.aa_circle(surf, S.MARK, (A1[0] + 1, A1[1]), 5, S.INK2)
            draw_sensors(surf, vv, c, bx, by, psi, labels=False, cam_range=9)
            surf.set_clip(None)
            S.aa_line(surf, S.FRAME, (sub.right + 1, 0), (sub.right + 1, PH), 1)
            lab = "spray (triangle)" if j == 0 else "ball (plus)"
            surf.blit(S.font(17).render(lab, True, S.INK2), (sub.x + 12, PH - 28))
        return surf
    R = {"T1": 11, "T2": 15, "T3": 9, "T4": 13, "T6": 8}[kind]
    if kind == "T1":
        cx, cy = m.x, m.y - 2.0
    elif kind == "T6":
        cx, cy = m.x, m.y + 2.0
    else:
        cx, cy = m.x, m.y
    v = view_for(c, cx, cy, R, rect)
    surf.set_clip(rect); draw_course(surf, v, c, light_green=(kind == "T4"))
    if kind == "T1":
        draw_sensors(surf, v, c, m.x, m.y - 9.0, math.pi / 2, max_labels=2)
    elif kind == "T2":
        px, py = m.path[2]; draw_sensors(surf, v, c, px, py, math.atan2(m.path[3][1] - py, m.path[3][0] - px), max_labels=1)
    elif kind == "T3":
        b = m.bays[0]
        draw_sensors(surf, v, c, b.entry[0] + 2.5 * math.cos(b.heading + math.pi / 2), b.entry[1] + 2.5 * math.sin(b.heading + math.pi / 2), b.heading, max_labels=1)
        S.banner(surf, (20, 22), c.day_shape, c.day_color, 18)
        surf.blit(S.font(17).render("banner of the day", True, S.INK2), (36, 11))
    elif kind == "T4":
        hx, hy = m.path[0]; draw_sensors(surf, v, c, hx, hy, m.heading, max_labels=1)
    else:
        draw_sensors(surf, v, c, m.x, m.y + 6, -math.pi / 2, cam_range=10, max_labels=1)
    surf.set_clip(None)
    return surf


def main():
    c = generate_course(3, GenConfig(difficulty="hard"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15))
    for ax, kind in zip(axes.flat, ["T1", "T2", "T3", "T4", "T5", "T6"]):
        img = pygame.surfarray.array3d(panel(c, kind)).transpose(1, 0, 2)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#b8c3cc"); sp.set_linewidth(0.6)
        ax.set_title(f"{kind}  {TASK_NAMES[kind]}", fontsize=8, loc="left", color=INK, fontweight="bold", pad=3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.04, hspace=0.16)
    save(fig, "fig_tasks")


if __name__ == "__main__":
    main()
