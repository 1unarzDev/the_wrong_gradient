"""Mission tasks (fictional point values; NOT official RoboBoat scoring).

Types and completion rules (a task scores ONCE, to the first boat completing it):
  gate    : cross the segment between a red/green buoy pair in the marked direction
  inspect : stay within 3.0 m of the target for 3 s (any speed < 1.2 m/s)
  dock    : be within 2.0 m of the berth with speed < 0.4 m/s for 2 s
  sample  : stay within 2.5 m of the station with speed < 0.6 m/s for 5 s
A task is available from `release` until `deadline`; completion after the deadline
scores nothing (the task expires).  Redundant visits = a boat arriving to service a
task that is already completed or being serviced by another boat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import SimConfig
from .rng import stream
from .world import World

SERVICE = {"gate": 0.0, "inspect": 3.0, "dock": 2.0, "sample": 5.0}
RADIUS = {"gate": 0.0, "inspect": 3.0, "dock": 2.0, "sample": 2.5}
VMAX_SVC = {"gate": 99.0, "inspect": 1.2, "dock": 0.4, "sample": 0.6}
GATE_HALF = 3.6            # half width between buoys (m)
GATE_LEAD = 8.0            # entry/exit points this far before/after the gate


@dataclass
class Task:
    id: int
    kind: str
    x: float
    y: float
    heading: float              # gate normal (direction of travel) / dock approach
    value: int
    release: float
    deadline: float
    points: List[Tuple[float, float]] = field(default_factory=list)  # waypoints to service
    # dynamic state (truth, owned by the simulator)
    completed_by: Optional[int] = None
    completed_at: Optional[float] = None
    dwell: dict = field(default_factory=dict)       # boat -> seconds accumulated

    @property
    def service_s(self) -> float:
        return SERVICE[self.kind]

    @property
    def stop(self) -> bool:
        return self.kind != "gate"          # every dwell task requires slowing down

    @property
    def extra_len(self) -> float:
        """Path length beyond points[0] that the task itself requires."""
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:]))

    def buoys(self):
        """(red, green) buoy positions for gates."""
        nx, ny = -math.sin(self.heading), math.cos(self.heading)
        return ((self.x + GATE_HALF * nx, self.y + GATE_HALF * ny),
                (self.x - GATE_HALF * nx, self.y - GATE_HALF * ny))

    def active(self, t) -> bool:
        return self.release <= t <= self.deadline and self.completed_by is None


def generate_tasks(world: World, cfg: SimConfig) -> List[Task]:
    rng = stream(world.seed, "tasks")
    n = int(round(cfg.tasks_per_boat * cfg.n_boats))
    kinds = list(cfg.task_type_weights)
    p = np.array([cfg.task_type_weights[k] for k in kinds], float); p /= p.sum()
    tasks: List[Task] = []
    used_docks = set()
    n_init = int(round(cfg.initial_task_frac * n))
    for tid in range(n):
        kind = str(rng.choice(kinds, p=p))
        release = 0.0 if tid < n_init else float(rng.uniform(5, cfg.release_window_frac * cfg.episode_s))
        deadline = release + float(rng.uniform(cfg.deadline_min_s, cfg.deadline_max_s))
        value = int(rng.integers(cfg.value_min, cfg.value_max + 1))
        free_docks = [k for k in range(len(world.dock_sites)) if k not in used_docks]
        if kind == "dock" and not free_docks:
            kind = "sample"
        # Placement constraints are relaxed ONLY if the original 400 attempts fail, so every world
        # that could be generated before this fallback existed is bit-identical (same RNG draws).
        placed = False
        for spacing, chan_pad, tries in ((11.0, 10.0, 400), (8.0, 6.0, 400), (5.0, 3.0, 800)):
            for _ in range(tries):
                if kind == "dock":
                    k = int(rng.choice(free_docks))
                    x, y, hd = world.dock_sites[k]
                    pts = [(x, y)]
                    ok = True
                else:
                    x, y = world.random_water_point(rng, min_clear=5.0 if kind != "gate" else 6.5,
                                                    x_range=(25.0, world.width - 8.0))
                    hd = float(rng.uniform(-math.pi, math.pi))
                    if kind == "gate":
                        dx, dy = math.cos(hd), math.sin(hd)
                        a = (x - GATE_LEAD * dx, y - GATE_LEAD * dy)
                        b = (x + GATE_LEAD * dx, y + GATE_LEAD * dy)
                        pts = [a, (x, y), b]
                        ok = (world.clearance_at(*a) >= 3.0 and world.clearance_at(*b) >= 3.0
                              and world.segment_free(a, b, 2.4) and world.reachable(*a))
                    else:
                        pts = [(x, y)]
                        ok = True
                    ok = ok and all(not ch.contains(x, y, pad=chan_pad) for ch in world.channels)
                if ok and all(math.hypot(x - t.x, y - t.y) > spacing for t in tasks):
                    placed = True
                    break
            if placed:
                break
            if kind == "dock":                 # all dock berths crowded: fall back to a sample task
                kind = "sample"
        if not placed:
            raise RuntimeError("task placement failed")
        if kind == "dock":
            used_docks.add(k)
        tasks.append(Task(tid, kind, float(x), float(y), float(hd), value, release, deadline,
                          [(float(a), float(b)) for a, b in pts]))
    return tasks
