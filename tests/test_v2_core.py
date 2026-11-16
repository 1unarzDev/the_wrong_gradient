"""Course v2 core: generator validity, determinism, referee rules, oracle episodes."""
import math

import numpy as np
import pytest

from rb2.config2 import V2Config
from rb2.course import COLORS, SHAPES
from rb2.generate import GenConfig, compile_chart, entry_points, generate_course, max_curvature, DIFFICULTY
from rb2.referee import Referee, crossed
from rb2.sim2 import Simulation2


@pytest.mark.parametrize("diff", ["standard", "hard"])
def test_generator_constraints_many_seeds(diff):
    for seed in range(250):
        c = generate_course(seed, GenConfig(difficulty=diff))
        kinds = [m.kind for m in c.modules]
        assert {"T1", "T2", "T3", "T4", "T6"} <= set(kinds) and kinds.count("T5") >= 2
        # module footprints do not overlap (dock may touch the edge)
        ms = [m for m in c.modules]
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                d = math.hypot(ms[a].x - ms[b].x, ms[a].y - ms[b].y)
                assert d > 0.9 * (ms[a].radius + ms[b].radius), (seed, ms[a].kind, ms[b].kind)
        # T1 straight ahead of the launch, T6 beside it
        m1, m6 = c.module("T1")[0], c.module("T6")[0]
        assert m1.y < 20 and m6.y < 10
        # spline curvature limit, gates spaced along the path
        m2 = c.module("T2")[0]
        assert max_curvature(m2.path) <= DIFFICULTY[diff]["curv_max"] + 1e-9
        # dock: at least one EMPTY bay matching the day's banner
        bays = c.module("T3")[0].bays
        assert any((b.shape, b.color) == (c.day_shape, c.day_color) and not b.occupied for b in bays)
        # speed challenge geometry within the handbook ranges
        m4 = c.module("T4")[0]
        assert 2.4 <= m4.data["gate_width"] <= 3.0 and 12.2 <= m4.data["buoy_dist"] <= 18.0


def test_every_entry_reachable_and_deterministic():
    for seed in range(40):
        c1 = generate_course(seed); c2 = generate_course(seed)
        assert [(o.cls, o.x, o.y) for o in c1.objects] == [(o.cls, o.x, o.y) for o in c2.objects]
        ch = compile_chart(c1, GenConfig())
        assert all(ch.reachable(*p) for m in c1.modules for p in entry_points(m))


def test_return_rush_layout():
    for seed in range(20):
        c = generate_course(seed, GenConfig(variant="return_rush"))
        N = c.rush_nodes
        assert N["A"][0] < N["S"][0] < N["B"][0] and N["T"][1] < N["A"][1] < N["S"][1]
        assert len(c.booms) == 4 and len(c.boom_gates) == 2 and len(c.piers) > 20


def test_crossing_logic():
    a, b = (0.0, -1.0), (0.0, 1.0)
    assert crossed((-1, 0), (1, 0), a, b, (1, 0))
    assert not crossed((-1, 0), (1, 0), a, b, (-1, 0))          # wrong direction
    assert not crossed((-1, 2), (1, 2), a, b)                     # outside the buoys
    assert not crossed((-1, 0), (-0.5, 0), a, b)


def test_t1_precedence_and_bay_rule():
    c = generate_course(5)
    ref = Referee(c, 2)
    m4 = c.module("T5")[0]
    o = c.obj(m4.objects[0])
    # boat 0 sits on a delivery target before doing T1: nothing may score
    for k in range(200):
        ref.update(k * 0.05, 0.05, 0, (o.x + 3.0, o.y), (o.x + 3.0, o.y), math.pi, 0.0)
    assert ref.points == 0
    # wrong bay: no 300-point 'correct bay' award (at most the 100 for an unoccupied bay), attempt consumed
    ref.prog[1].t1_done = True
    m3 = c.module("T3")[0]
    wrong = next((k for k, b in enumerate(m3.bays) if (b.shape, b.color) != (c.day_shape, c.day_color)), None)
    if wrong is not None:
        bay = m3.bays[wrong]
        for k in range(60):
            ref.update(10 + k * 0.05, 0.05, 1, bay.inner, bay.inner, 0.0, 0.0)
        assert ref.t3_attempted and "T3" not in ref.done
        assert all(e.points <= 100 * 1.5 for e in ref.events if e.task == "T3")


@pytest.mark.parametrize("policy", ["P1", "P3", "P4"])
def test_oracle_episode_runs_and_scores(policy):
    s = Simulation2(V2Config(policy=policy, perception="oracle", episode_s=240), 11)
    m = s.run()
    assert m["t1_boats"] >= 3 and m["points"] > 400
    tasks = [e.task for e in s.ref.events]
    assert tasks.index("T1") == 0                                   # nothing scores before the channel
