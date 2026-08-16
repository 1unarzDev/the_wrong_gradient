"""Chart rendering and the few restrained effects (visual only; no randomness)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pygame
from scipy import ndimage

from . import theme as T


def _offset_poly(poly, d):
    c = poly.mean(0)
    v = poly - c
    r = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return c + v * np.maximum(r + d, 0.2) / r


def _upsample(field, ph, pw):
    zy, zx = ph / field.shape[0], pw / field.shape[1]
    up = ndimage.zoom(field, (zy, zx), order=1)[:ph, :pw]
    if up.shape != (ph, pw):
        up = np.pad(up, ((0, ph - up.shape[0]), (0, pw - up.shape[1])), mode="edge")
    return up


def render_static(world, scale: float) -> pygame.Surface:
    """Bake the course at `scale` px/m: flat water with a soft shallow band, sandy islands
    with vegetation, a stone barrier shaded by height, wooden piers."""
    W, H = world.width, world.height
    pw, ph = int(W * scale), int(H * scale)
    blocked = world.blocked.copy()
    ny, nx = blocked.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cxs, cys = (xx + 0.5) * world.res, (yy + 0.5) * world.res
    for bx, by, br in getattr(world, "point_obstacles", []):
        blocked[(cxs - bx) ** 2 + (cys - by) ** 2 <= (br + 0.5 * world.res) ** 2] = False
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    rock_only = np.zeros_like(blocked)
    if world.channels:
        rock_only = blocked & (np.abs(cxs - world.barrier_x) < 8.0)
        rock_only[0, :] = rock_only[-1, :] = False
    land_only = blocked & ~rock_only
    clr = np.flipud(ndimage.distance_transform_edt(~land_only) * world.res)
    up = _upsample(ndimage.gaussian_filter(clr, 0.8), ph, pw)
    # water: flat, with a narrow linear shallow band (~2 m) at the shore; rock walls get no band
    t = np.clip((up - 0.5) / 2.2, 0, 1)[..., None]
    arr = (np.array(T.SHALLOW) * (1 - t) + np.array(T.WATER) * t)
    # stone barrier: rock cells only (near the barrier line), shaded by height
    rock = blocked.copy()
    rock[:, :] = False
    if world.channels:
        band = np.abs(cxs - world.barrier_x) < 8.0
        rock = blocked & band
        rock[0, :] = rock[-1, :] = False
    rin = np.flipud(ndimage.distance_transform_edt(rock) * world.res)
    rup = _upsample(ndimage.gaussian_filter(rin, 0.8), ph, pw)
    rmask = rup > 0.45
    arr[rmask] = T.ROCK
    arr[rup > 1.6] = T.ROCK_HI
    redge = rmask ^ (np.roll(rmask, 1, 0) & np.roll(rmask, 1, 1) & np.roll(rmask, -1, 0) & np.roll(rmask, -1, 1))
    arr[redge] = T.ROCK_EDGE
    surf = pygame.Surface((pw, ph))
    pygame.surfarray.blit_array(surf, np.clip(arr, 0, 255).astype(np.uint8).transpose(1, 0, 2))

    def P(poly):
        return [(float(x * scale), float((H - y) * scale)) for x, y in poly]

    for isl in world.islands:
        pygame.draw.polygon(surf, T.LAND_EDGE, P(_offset_poly(isl, 0.5)))
        pygame.draw.polygon(surf, T.SAND, P(isl))
        pygame.draw.polygon(surf, T.LAND, P(_offset_poly(isl, -1.8)))
    for pr in world.piers:
        pygame.draw.polygon(surf, T.PIER_DARK, P(_offset_poly(pr, 0.15)))
        pygame.draw.polygon(surf, T.PIER, P(pr))
        a_, b_, c_, d_ = pr
        for f in np.linspace(0.15, 0.85, 5):
            pygame.draw.line(surf, T.PIER_DARK, P([a_ + (b_ - a_) * f])[0], P([d_ + (c_ - d_) * f])[0], 1)
    lx, ly = world.launch
    r = pygame.Rect(0, 0, 14 * scale, 34 * scale)
    r.center = (lx * scale, (H - ly) * scale)
    dashed_rect(surf, T.WAKE, r, 6)
    img = T.font(11).render("Start", True, T.WAKE)
    surf.blit(img, (r.x + 4, r.y - img.get_height() - 2))
    return surf


def dashed_rect(surf, col, r, dash=6):
    for (a, b) in ((r.topleft, r.topright), (r.topright, r.bottomright),
                   (r.bottomright, r.bottomleft), (r.bottomleft, r.topleft)):
        dashed_line(surf, col, a, b, dash, dash)


def dashed_line(surf, col, a, b, on=6, off=5, width=1, phase=0.0):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L < 1:
        return
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    d = -(phase % (on + off))
    while d < L:
        s0, s1 = max(0.0, d), min(L, d + on)
        if s1 > s0:
            p0 = (a[0] + ux * s0, a[1] + uy * s0); p1 = (a[0] + ux * s1, a[1] + uy * s1)
            if width == 1:
                pygame.draw.aaline(surf, col, p0, p1)
            else:
                pygame.draw.line(surf, col, p0, p1, width)
        d += on + off


def thick_polyline(surf, col, pts, width):
    """Round-joined thick polyline for clean route strokes."""
    if len(pts) < 2:
        return
    for a, b in zip(pts, pts[1:]):
        pygame.draw.line(surf, col, a, b, width)
    for p in pts:
        pygame.draw.circle(surf, col, p, width / 2)


def ribbon(surf, col, pts, width):
    """Anti-aliased thick stroke: filled offset polygon with smooth borders and round caps."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    h = width / 2
    left, right = P + n * h, P - n * h
    poly = [tuple(p) for p in left] + [tuple(p) for p in right[::-1]]
    pygame.draw.polygon(surf, col, poly)
    pygame.draw.aalines(surf, col, False, [tuple(p) for p in left])
    pygame.draw.aalines(surf, col, False, [tuple(p) for p in right])
    for e in (P[0], P[-1]):
        pygame.draw.circle(surf, col, tuple(e), h)


def _xf(pos, psi, pts):
    c, s_ = math.cos(psi), math.sin(psi)
    return [(pos[0] + x * c - y * s_, pos[1] - (x * s_ + y * c)) for x, y in pts]


def hull_points(pos, psi, length):
    """Outline of the whole boat (used for belief ghosts)."""
    L, Wd = length, length * 0.62
    return _xf(pos, psi, [(0.5 * L, 0.3 * Wd), (0.5 * L, -0.3 * Wd), (-0.5 * L, -0.5 * Wd), (-0.5 * L, 0.5 * Wd)])


def draw_boat(dst, pos, psi, color, length_px):
    """Simple twin-hull catamaran: two white hulls, a coloured deck between them."""
    L, Wd = length_px, length_px * 0.62
    hw = Wd * 0.26
    for side in (-1, 1):
        yc = side * (Wd / 2 - hw / 2)
        hull = [(0.5 * L, yc), (0.3 * L, yc + hw / 2), (-0.5 * L, yc + hw / 2), (-0.5 * L, yc - hw / 2), (0.3 * L, yc - hw / 2)]
        pts = _xf(pos, psi, hull)
        pygame.draw.polygon(dst, T.HULL, pts)
        pygame.draw.aalines(dst, T.HULL_EDGE, True, pts)
    deck = [(0.28 * L, Wd / 2 - hw * 0.6), (0.28 * L, -Wd / 2 + hw * 0.6), (-0.36 * L, -Wd / 2 + hw * 0.6), (-0.36 * L, Wd / 2 - hw * 0.6)]
    pts = _xf(pos, psi, deck)
    pygame.draw.polygon(dst, color, pts)
    pygame.draw.aalines(dst, T.shade(color, 0.6), True, pts)


def ghost_boat(dst, pos, psi, color, length_px, dot=None):
    """Phantom catamaran: same silhouette as a real boat, 1.5 px outline, faint fill,
    and a dot in the colour of the matching real boat (dst must be SRCALPHA)."""
    L, Wd = length_px * 0.9, length_px * 0.9 * 0.62
    hw = Wd * 0.26
    parts = []
    for side in (-1, 1):
        yc = side * (Wd / 2 - hw / 2)
        parts.append([(0.5 * L, yc), (0.3 * L, yc + hw / 2), (-0.5 * L, yc + hw / 2), (-0.5 * L, yc - hw / 2), (0.3 * L, yc - hw / 2)])
    parts.append([(0.28 * L, Wd / 2 - hw * 0.6), (0.28 * L, -Wd / 2 + hw * 0.6), (-0.36 * L, -Wd / 2 + hw * 0.6), (-0.36 * L, Wd / 2 - hw * 0.6)])
    for poly in parts:
        pts = _xf(pos, psi, poly)
        pygame.draw.polygon(dst, (*color, 28), pts)
        pygame.draw.polygon(dst, (*color, 230), pts, 2)
    if dot is not None:
        pygame.draw.rect(dst, dot, pygame.Rect(int(pos[0]) - 2, int(pos[1]) - 2, 5, 5))


def v_wake(dst, pos, psi, length_px, alpha=110):
    """Short static V-wake behind a moving boat (shape, not a history trace)."""
    L, Wd = length_px, length_px * 0.62
    for side in (-1, 1):
        stern = _xf(pos, psi, [(-0.5 * L, side * Wd * 0.37)])[0]
        ang = psi + math.pi + side * math.radians(14)
        end = (stern[0] + 18 * math.cos(ang), stern[1] - 18 * math.sin(ang))
        pygame.draw.aaline(dst, (150, 195, 230, alpha), stern, end)


def pennant(dst, base, color, hollow=False, dx=0):
    """Planted flag on a completed task: 20 px pole, 13x9 flag, dark base dot."""
    x, y = base[0] + dx, base[1]
    pygame.draw.circle(dst, (12, 22, 40), (x, y), 2)
    pygame.draw.line(dst, (240, 244, 248), (x, y), (x, y - 20), 2)
    flag = [(x + 1, y - 20), (x + 14, y - 15.5), (x + 1, y - 11)]
    if hollow:
        pygame.draw.polygon(dst, (*color, 40), flag)
        pygame.draw.polygon(dst, (*color, 240), flag, 2)
    else:
        pygame.draw.polygon(dst, color, flag)


@dataclass
class Effect:
    x: float
    y: float
    t0: float
    life: float
    kind: str           # ring | label
    color: tuple
    text: str = ""
    r_max: float = 6.0  # metres


class Effects:
    """Only two effects exist: a hairline ring that expands once, and a label that fades."""

    def __init__(self):
        self.items: List[Effect] = []

    def ring(self, x, y, t, color, r_max=6.0, life=0.9):
        self.items.append(Effect(x, y, t, life, "ring", color, r_max=r_max))

    def label(self, x, y, t, text, color, life=1.6):
        self.items.append(Effect(x, y, t, life, "label", color, text))

    def draw(self, dst, cam, t):
        keep = []
        for e in self.items:
            f = (t - e.t0) / e.life
            if f >= 1 or f < 0:
                continue
            keep.append(e)
            X, Y = cam.to_screen(e.x, e.y)
            ease = 1 - (1 - f) ** 3
            if e.kind == "ring":
                r = max(2, int((1.5 + e.r_max * ease) * cam.scale))
                pygame.draw.circle(dst, (*e.color, int(230 * (1 - f))), (X, Y), r, 2)
            else:
                img = T.font(14, True).render(e.text, True, e.color)
                img.set_alpha(int(255 * min(1.0, 3 * (1 - f))))
                dst.blit(img, (X - img.get_width() / 2, Y - 22 - 14 * ease))
        self.items = keep
