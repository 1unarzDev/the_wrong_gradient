"""Visual tokens: a quiet hydrographic-chart look. The only saturated elements on screen
are the boats and their plans. Visual-only: nothing here touches simulation state."""
import os

import pygame

# --- map (a calm RoboBoat course seen from above) ---------------------------------
WATER = (19, 96, 164)          # open water: saturated deep blue
SHALLOW = (46, 132, 184)       # narrow shallow band near land (no glow)
CONTOUR = (214, 244, 242)      # white surf line at the coast
GRID = WATER                   # no graticule
SAND = (246, 226, 172)
LAND = (70, 158, 84)           # lush vegetation
LAND_EDGE = (222, 201, 148)    # sand edge
ROCK = (150, 138, 124)
ROCK_EDGE = (146, 135, 120)
ROCK_HI = (186, 174, 156)
PIER = (176, 128, 76)
PIER_DARK = (110, 82, 52)
CURRENT = (78, 144, 170)
HULL = (238, 242, 244)
HULL_EDGE = (150, 162, 170)
WAKE = (220, 238, 245)

# --- HUD (dark, quiet) ---------------------------------------------------------------
PAPER = (13, 24, 40)           # panel / chips background (navy)
INK = (234, 240, 244)          # primary text
INK2 = (150, 176, 200)         # secondary text
MUTED = (104, 122, 134)
RULE = (40, 60, 84)            # hairlines
ACCENT = (64, 204, 255)        # the one UI accent
WARN = (255, 184, 77)
BAD = (255, 107, 94)
GOOD = (110, 214, 140)
RED_BUOY = (232, 72, 60)
GREEN_BUOY = (46, 186, 98)

# boat deck colours: chosen to read on blue water; fixed order
BOAT_COLORS = [
    (255, 140, 60), (255, 214, 80), (240, 96, 150), (150, 226, 110),
    (186, 150, 255), (90, 220, 230), (255, 255, 255), (255, 110, 90),
    (255, 140, 60), (255, 214, 80), (240, 96, 150), (150, 226, 110),
    (186, 150, 255), (90, 220, 230), (255, 255, 255), (255, 110, 90),
]
TASK_LABEL = {"gate": "Gate", "inspect": "Inspect", "dock": "Dock", "sample": "Sample"}

_FONT_CACHE = {}
_FAMILIES = ["adwaitasans", "notosans", "dejavusans"]


def _font_path(bold):
    for fam in _FAMILIES:
        p = pygame.font.match_font(fam, bold=bold)
        if p:
            return p
    try:   # portable fallback: matplotlib ships DejaVu Sans
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        return os.path.join(d, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    except Exception:
        return None


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.Font(_font_path(bold), size)
    return _FONT_CACHE[key]


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)

OUTER_LAND = (46, 90, 64)
