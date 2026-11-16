"""Course v2: a stylized interpretation of the RoboBoat 2025 autonomy challenge (open water).

Truth model only (what is physically on the water). Units: metres; x right, y up.
Sizes follow the RoboNation 2025 handbook (section 3.2) where given:
  Sur-Mark tall buoys 18 in (0.46 m) diameter; Polyform A-0 0.20 m; A-2 0.37 m;
  speed gate 6-10 ft (1.8-3.0 m); blue buoy 40-100 ft (12-30 m) from the gate;
  Jet-Dock cubes 20 in (0.51 m); banners 24 in.
Point values are fictional (documented in the paper)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# object classes (what a perfect classifier would say)
CLASSES = ("red_tall", "green_tall", "red", "green", "yellow", "black", "blue",
           "vessel_orange", "vessel_black", "marker", "light_panel", "moored", "boom_float")
BUOY_RADIUS = {"red_tall": 0.23, "green_tall": 0.23, "red": 0.10, "green": 0.10, "yellow": 0.10,
               "black": 0.18, "blue": 0.18, "marker": 0.30, "light_panel": 0.25}
SHAPES = ("circle", "triangle", "square", "plus")
COLORS = ("blue", "green", "red")

TASK_NAMES_RR = {"RR": "Return rush"}
TASK_NAMES = {"T1": "Navigation channel", "T2": "Follow the path", "T3": "Docking",
              "T4": "Speed challenge", "T5": "Rescue delivery", "T6": "Return home"}


@dataclass
class Obj:
    id: int
    cls: str
    x: float
    y: float
    r: float
    module: int = -1
    heading: float = 0.0          # vessels: hull heading
    length: float = 0.0           # vessels: hull length (m)
    mark: str = ""                # vessels: "triangle" (orange, spray) or "plus" (black, ball)


@dataclass
class Bay:
    """One docking bay between two tines."""
    entry: Tuple[float, float]    # point just outside the bay mouth
    inner: Tuple[float, float]    # berth point inside the bay
    heading: float                # direction pointing INTO the bay
    shape: str
    color: str
    occupied: bool
    banner: Tuple[float, float]   # banner position (on the dock spine)


@dataclass
class Module:
    id: int
    kind: str                     # T1..T6
    x: float                      # anchor
    y: float
    heading: float
    radius: float                 # footprint radius for placement
    value: int
    objects: List[int] = field(default_factory=list)
    gates: List[Tuple[int, int]] = field(default_factory=list)   # (red id, green id) in passing order
    path: List[Tuple[float, float]] = field(default_factory=list)  # T2 spline samples / T4 loop
    bays: List[Bay] = field(default_factory=list)                  # T3
    cubes: List[Tuple[float, float, float, float]] = field(default_factory=list)  # T3 cube rects x,y,w,h (world)
    data: Dict = field(default_factory=dict)                       # kind-specific (counts, light delay, ...)


@dataclass
class Course:
    seed: int
    width: float
    height: float
    start: Tuple[float, float, float]          # x, y, heading of the launch point
    objects: List[Obj]
    modules: List[Module]
    day_shape: str
    day_color: str
    current: object = None                     # src.world.CurrentField
    difficulty: str = "standard"
    variant: str = "standard"                  # standard | return_rush
    booms: List[List[Tuple[float, float]]] = field(default_factory=list)   # containment boom polylines (obstacles)
    boom_gates: List[List[Tuple[float, float]]] = field(default_factory=list)  # closable boom sections (shortcut)
    shortcut_open: bool = False
    piers: List[Tuple[float, float, float, float]] = field(default_factory=list)  # marina cubes x, y, size, heading
    rush_nodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # S, A, B, T of the Braess network

    def module(self, kind):
        return [m for m in self.modules if m.kind == kind]

    def obj(self, i) -> Obj:
        return self.objects[i]
