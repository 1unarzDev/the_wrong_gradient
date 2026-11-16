"""v2 visual system: restrained palette, anti-aliased primitives and sensor overlays.

Restrained, mostly monochrome: pale handbook water, one blue-grey ink at four strengths, muted semantic
colours that still read, neutral boats identified by a small muted deck colour (no numbers), no dashed lines.
State is shown with opacity and weight, never with dash patterns. All primitives are antialiased."""
from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw as gfx

# ---- tokens -----------------------------------------------------------------------------
PAPER = (244, 246, 248)
WATER = (223, 231, 237)
WATER_EDGE = (210, 220, 228)
FRAME = (170, 184, 196)
DOCK, DOCK_LINE = (204, 197, 183), (180, 172, 156)
MOORED = (186, 196, 205)
BOOM, BOOM_FLOAT = (196, 164, 84), (176, 144, 64)
INK1, INK2, INK3, INK4 = (36, 44, 54), (82, 94, 108), (126, 138, 151), (178, 189, 199)
CHIP_BG, CHIP_BORDER = (250, 251, 252), (214, 221, 228)
SEM = {  # fill, rim
    "red": ((200, 84, 76), (156, 58, 52)), "green": ((64, 146, 102), (40, 108, 72)),
    "yellow": ((226, 188, 72), (168, 132, 36)), "black": ((54, 60, 68), (30, 34, 40)),
    "blue": ((62, 108, 176), (40, 78, 136)), "orange": ((218, 128, 64), (170, 92, 40)),
}
LIGHT_OFF, MARK = (88, 96, 106), (246, 247, 248)
SENSE = (52, 62, 76)            # sensing ink: quiet slate (teal clashed with the teal boat)
HULL, HULL_OUTLINE = (240, 243, 246), (120, 132, 146)
FLEET = [(88, 100, 170), (50, 136, 140), (136, 86, 140), (150, 112, 70),
         (186, 104, 128), (98, 112, 128), (118, 132, 74), (82, 140, 180)]

CLASS_SEM = {"red_tall": "red", "red": "red", "green_tall": "green", "green": "green", "yellow": "yellow",
             "black": "black", "blue": "blue", "vessel_orange": "orange", "vessel_black": "black"}
CLASS_LABEL = {"red_tall": "red", "red": "red", "green_tall": "green", "green": "green", "yellow": "yellow",
               "black": "black", "blue": "blue", "vessel_orange": "vessel orange", "vessel_black": "vessel black",
               "light_panel": "light", "marker": "marker"}

_FONT = {}


def font(size, bold=False):
    from src.viz import theme as T
    return T.font(size, bold)


# ---- antialiased primitives --------------------------------------------------------------
def aa_circle(surf, col, c, r, outline=None, ow=1):
    x, y, r = int(round(c[0])), int(round(c[1])), max(1, int(round(r)))
    gfx.filled_circle(surf, x, y, r, col)
    gfx.aacircle(surf, x, y, r, outline or col)
    if outline and ow > 1:
        gfx.aacircle(surf, x, y, max(1, r - 1), outline)


def aa_poly(surf, col, pts, outline=None):
    P = [(int(round(a)), int(round(b))) for a, b in pts]
    gfx.filled_polygon(surf, P, col)
    gfx.aapolygon(surf, P, outline or col)


def aa_line(surf, col, a, b, width=1.0):
    """Antialiased thick line as a thin quad."""
    if width <= 1.01:
        pygame.draw.aaline(surf, col, a, b)
        return
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    aa_poly(surf, col, [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)])


def aa_polyline(surf, col, pts, width=1.0, closed=False):
    """Smooth stroke: one filled ribbon polygon (no per-vertex beads) with antialiased edges."""
    P = [tuple(map(float, p)) for p in pts] + ([tuple(pts[0])] if closed else [])
    if len(P) < 2:
        return
    if width <= 1.01:
        pygame.draw.aalines(surf, col, False, P)
        return
    A = np.array(P)
    d = np.gradient(A, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    left, right = A + n * width / 2, A - n * width / 2
    poly = [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]
    gfx.filled_polygon(surf, [(int(round(x)), int(round(y))) for x, y in poly], col)
    pygame.draw.aalines(surf, col, False, [tuple(p) for p in left])
    pygame.draw.aalines(surf, col, False, [tuple(p) for p in right])


def chaikin(pts, it=2, closed=False):
    P = [tuple(p) for p in pts]
    for _ in range(it):
        Q = [] if closed else [P[0]]
        seq = list(zip(P, P[1:] + ([P[0]] if closed else [])))
        for (ax, ay), (bx, by) in seq:
            Q += [(0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by), (0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by)]
        if not closed:
            Q.append(P[-1])
        P = Q
    return P


def chevrons(surf, col, pts, every=90.0, arm=6.0, width=1.5):
    acc = 0.0
    nxt = every * 0.5
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6:
            continue
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        while acc + L >= nxt:
            f = (nxt - acc) / L
            px, py = a[0] + ux * L * f, a[1] + uy * L * f
            for s in (-1, 1):
                aa_line(surf, col, (px, py), (px - ux * arm + s * uy * arm * 0.8, py - uy * arm - s * ux * arm * 0.8), width)
            nxt += every
        acc += L


def sector(surf, col, c, r0, r1, a0, a1, n=24):
    """Filled annular sector (screen angles, y down handled by caller)."""
    A = np.linspace(a0, a1, n)
    outer = [(c[0] + r1 * math.cos(a), c[1] - r1 * math.sin(a)) for a in A]
    inner = [(c[0] + r0 * math.cos(a), c[1] - r0 * math.sin(a)) for a in A[::-1]]
    aa_poly(surf, col, outer + inner)


# ---- objects -------------------------------------------------------------------------------
def boat(surf, pos, psi, idx=0, length=22, focus=False, alpha=255):
    """Neutral twin-hull ASV; identity = small muted deck pad (split pad for boats 9-16). Bow points along psi."""
    L = float(length); B = L * 0.62
    pw = B * 0.26
    c, s = math.cos(psi), math.sin(psi)

    def P(a, b):
        return (pos[0] + a * c - b * s, pos[1] - (a * s + b * c))
    acc = FLEET[idx % 8]
    tint = tuple(int(a + (255 - a) * 0.72) for a in acc)               # same hue as the deck, much lighter
    outline = INK1 if focus else tuple(int(a * 0.72) for a in acc)
    for side in (-1, 1):
        yc = side * (B / 2 - pw / 2)
        hull = [P(-0.5 * L, yc - pw / 2), P(0.25 * L, yc - pw / 2), P(0.5 * L, yc), P(0.25 * L, yc + pw / 2), P(-0.5 * L, yc + pw / 2)]
        aa_poly(surf, tint + ((alpha,) if alpha < 255 else ()), hull, outline)
    deck = [P(-0.25 * L, -B / 2 + pw), P(0.25 * L, -B / 2 + pw), P(0.25 * L, B / 2 - pw), P(-0.25 * L, B / 2 - pw)]
    if idx < 8:
        aa_poly(surf, acc, deck, outline)
    else:
        aa_poly(surf, tint, deck, outline)
        aa_poly(surf, acc, [P(0.0, -B / 2 + pw), P(0.25 * L, -B / 2 + pw), P(0.25 * L, B / 2 - pw), P(0.0, B / 2 - pw)])


def vessel(surf, pos, heading, cls, mark, length=20):
    L = float(length); B = L * 0.4
    fill, rim = SEM["orange" if cls == "vessel_orange" else "black"]
    c, s = math.cos(heading), math.sin(heading)

    def P(a, b):
        return (pos[0] + a * c - b * s, pos[1] - (a * s + b * c))
    aa_poly(surf, fill, [P(-0.5 * L, -B / 2), P(0.15 * L, -B / 2), P(0.5 * L, 0), P(0.15 * L, B / 2), P(-0.5 * L, B / 2)], rim)
    mx, my = P(0.1 * L, 0)
    k = 0.7 * B
    if mark == "triangle":
        aa_poly(surf, MARK, [P(0.1 * L + 0.55 * k, 0), P(0.1 * L - 0.45 * k, -0.5 * k), P(0.1 * L - 0.45 * k, 0.5 * k)])
    elif mark == "plus":
        w = 0.18 * B
        aa_line(surf, MARK, P(0.1 * L - k / 2, 0), P(0.1 * L + k / 2, 0), w)
        aa_line(surf, MARK, P(0.1 * L, -k / 2), P(0.1 * L, k / 2), w)


def moored(surf, pos, heading, length=18):
    L = float(length); B = L * 0.38
    c, s = math.cos(heading), math.sin(heading)

    def P(a, b):
        return (pos[0] + a * c - b * s, pos[1] - (a * s + b * c))
    aa_poly(surf, MOORED, [P(-0.5 * L, -B / 2), P(0.2 * L, -B / 2), P(0.5 * L, 0), P(0.2 * L, B / 2), P(-0.5 * L, B / 2)])


def buoy(surf, pos, cls, d_px):
    sem = CLASS_SEM.get(cls)
    if sem is None:
        return
    fill, rim = SEM[sem]
    tall = cls.endswith("_tall") or cls == "blue"
    r = d_px / 2
    aa_circle(surf, fill, pos, r, rim)
    if tall:
        aa_circle(surf, MARK, pos, max(1.5, r * 0.35))


def light_panel(surf, pos, green, scale=1.0):
    w, h = 7 * scale, 15 * scale
    r = pygame.Rect(0, 0, w, h); r.center = (int(pos[0]), int(pos[1]))
    pygame.draw.rect(surf, SEM["black"][0], r, border_radius=max(1, int(2 * scale)))
    top, bot = (r.centerx, r.y + h * 0.28), (r.centerx, r.y + h * 0.72)
    aa_circle(surf, LIGHT_OFF if green else SEM["red"][0], top, 1.6 * scale)
    aa_circle(surf, SEM["green"][0] if green else LIGHT_OFF, bot, 1.6 * scale)


def banner(surf, pos, shape, color, size=14):
    r = pygame.Rect(0, 0, size, size); r.center = (int(pos[0]), int(pos[1]))
    pygame.draw.rect(surf, MARK, r, border_radius=2)
    pygame.draw.rect(surf, CHIP_BORDER, r, 1, border_radius=2)
    col = SEM[color][0]; x, y = r.center; k = size * 0.32
    if shape == "circle":
        aa_circle(surf, col, (x, y), k)
    elif shape == "triangle":
        aa_poly(surf, col, [(x, y - k), (x + k, y + k * 0.8), (x - k, y + k * 0.8)])
    elif shape == "square":
        aa_poly(surf, col, [(x - k * 0.85, y - k * 0.85), (x + k * 0.85, y - k * 0.85), (x + k * 0.85, y + k * 0.85), (x - k * 0.85, y + k * 0.85)])
    else:
        aa_line(surf, col, (x - k, y), (x + k, y), k * 0.5); aa_line(surf, col, (x, y - k), (x, y + k), k * 0.5)


# ---- overlays --------------------------------------------------------------------------------
def camera_cone(layer, c, psi, rng_px, fov):
    """FOV wedge that fades with distance (14% -> 0%), radial edges only, no outer arc."""
    rings = 8
    for k in range(rings):
        a = int(255 * 0.14 * (1 - k / rings))
        sector(layer, (*INK1, a), c, rng_px * k / rings, rng_px * (k + 1) / rings, psi - fov / 2, psi + fov / 2)
    for sgn in (-1, 1):
        e = (c[0] + rng_px * math.cos(psi + sgn * fov / 2), c[1] - rng_px * math.sin(psi + sgn * fov / 2))
        aa_line(layer, (*INK1, 56), c, e, 1)
    arc = [(c[0] + rng_px * math.cos(a), c[1] - rng_px * math.sin(a)) for a in np.linspace(psi - fov / 2, psi + fov / 2, 40)]
    pygame.draw.aalines(layer, (*INK1, 56), False, arc)


def brackets(surf, obj_rect, col, alpha=255, width=1.0):
    x0, y0, x1, y1 = obj_rect.left, obj_rect.top, obj_rect.right, obj_rect.bottom
    arm = max(3, min(8, 0.3 * min(obj_rect.w, obj_rect.h)))
    c = (*col, alpha)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        aa_line(surf, c, (cx, cy), (cx + sx * arm, cy), width)
        aa_line(surf, c, (cx, cy), (cx, cy + sy * arm), width)


def detection_label(surf, rect, name, conf, size=10, water=WATER):
    a = font(size).render(name, True, INK2)
    b = font(size).render(f" {conf}", True, INK3)
    w, h = a.get_width() + b.get_width(), max(a.get_height(), b.get_height())
    x, y = rect.left, rect.top - h - 2
    knock = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA); knock.fill((*water, 204))
    surf.blit(knock, (x - 1, y - 1)); surf.blit(a, (x, y)); surf.blit(b, (x + a.get_width(), y))


def lidar_outline(layer, c, pts_px):
    P = chaikin(pts_px, 1, closed=True)
    aa_polyline(layer, (*INK2, 115), P, 1.0, closed=True)


def lidar_tick(layer, center, rim_r, from_ang):
    """One short arc on the face of an object that the lidar hits (from_ang = screen angle toward the boat)."""
    A = np.linspace(from_ang - 0.5, from_ang + 0.5, 8)
    r = rim_r + 2
    P = [(center[0] + r * math.cos(a), center[1] - r * math.sin(a)) for a in A]
    aa_polyline(layer, (*INK1, 178), P, 1.5)
