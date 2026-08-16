"""Engineering / integrity tests for the dynamic simulator."""
import math
import os

import numpy as np
import pytest

from src.comms import CommTape
from src.config import SimConfig
from src.navigation import penalty_tables, rollout
from src.physics import rk4_step
from src.policies import make_policy
from src.sim import Simulation

SHORT = dict(episode_s=60.0)


def run_short(policy="P3", seed=7, **kw):
    s = Simulation(SimConfig(policy=policy, **{**SHORT, **kw}), seed)
    m = s.run()
    return s, m


def test_deterministic_replay():
    a, ma = run_short()
    b, mb = run_short()
    assert ma == {**mb, "cpu_ms_mean": ma["cpu_ms_mean"], "cpu_ms_max": ma["cpu_ms_max"]}
    np.testing.assert_array_equal(a.S, b.S)
    assert [(e.t, e.kind, e.boat, e.task) for e in a.events] == [(e.t, e.kind, e.boat, e.task) for e in b.events]


def test_common_random_numbers_across_policies():
    """World, tasks, disturbances and packet tapes must not depend on the policy."""
    sims = [Simulation(SimConfig(policy=p, **SHORT), 11) for p in ("P0", "P3", "P4")]
    ref = sims[0]
    for s in sims[1:]:
        np.testing.assert_array_equal(s.world.blocked, ref.world.blocked)
        assert [(t.kind, t.x, t.y, t.value, t.release, t.deadline) for t in s.tasks] == \
               [(t.kind, t.x, t.y, t.value, t.release, t.deadline) for t in ref.tasks]
        np.testing.assert_array_equal(s.dist_tape, ref.dist_tape)
        np.testing.assert_array_equal(s.net.tape.drop, ref.net.tape.drop)
        np.testing.assert_array_equal(s.sensor_tape, ref.sensor_tape)


def test_comm_modes_share_randomness_draws():
    cfg = SimConfig()
    ideal = CommTape(5, cfg.replace(comm_mode="ideal"))
    none = CommTape(5, cfg.replace(comm_mode="none"))
    deg = CommTape(5, cfg.replace(comm_mode="degraded"))
    n = cfg.n_boats
    off = ~np.eye(n, dtype=bool)
    assert not ideal.drop[:, off].any() and none.drop[:, off].all()
    rate = deg.drop[:, off].mean()
    assert 0.2 < rate < 0.8            # bursty, substantial loss


def test_gui_rendering_does_not_change_simulation():
    """The visualizer only reads state: stepping with rendering == headless."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # force headless even if the session sets wayland
    import pygame
    from src.viz.mission_view import MissionView, draw_panel
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    a, _ = run_short(seed=3)
    b = Simulation(SimConfig(policy="P3", **SHORT), 3)
    view = MissionView(b, pygame.Rect(0, 0, 600, 600))
    while not b.done:
        for _ in range(10):
            b.step()
        view.update(1 / 60, 0.5)
        view.draw(screen)
        draw_panel(screen, pygame.Rect(600, 0, 300, 600), view)
    np.testing.assert_array_equal(a.S, b.S)
    pygame.quit()


def test_rk4_dt_convergence_open_loop():
    cfg = SimConfig()
    zero = lambda x, y, t: (0 * x, 0 * y)
    out = []
    for dt in (0.05, 0.025, 0.0125):
        S = np.zeros((5, 1))
        for k in range(int(round(20 / dt))):
            S = rk4_step(S, np.array([0.9]), np.array([0.5]), zero, k * dt, dt, cfg)
        out.append(S[:2, 0].copy())
    assert np.linalg.norm(out[0] - out[2]) < 1e-3       # metres after 20 s of turning
    assert np.linalg.norm(out[1] - out[2]) < 1e-4


def test_closed_loop_dt_convergence():
    cfg = SimConfig()
    wps = [(0, 0), (30, 0), (30, 30), (0, 30)]
    t1 = rollout(wps, 0, 0, 0, 0, cfg, dt=0.1)
    t2 = rollout(wps, 0, 0, 0, 0, cfg, dt=0.025)
    assert abs(t1 - t2) / t2 < 0.03


def test_physical_bounds_and_scoring():
    s = Simulation(SimConfig(policy="P1", episode_s=120.0), 21)
    vmax = s.cfg.v_max + 2 * s.cfg.current_strength
    while not s.done:
        s.step()
        assert np.all(np.abs(s.S[3]) <= s.cfg.v_max * 1.05)
        assert np.all(np.abs(s.S[4]) <= s.cfg.r_max * 1.2)
        for i in range(s.n):
            assert not s.world.is_blocked(s.S[0, i], s.S[1, i])
    m = s.metrics()
    comp = [e for e in s.events if e.kind == "complete"]
    assert len({e.task for e in comp}) == len(comp)              # each task scored once
    assert m["points"] == sum(e.value for e in comp)
    for e in comp:
        tk = s.tasks[e.task]
        assert tk.release <= e.t <= tk.deadline
    assert m["welfare"] == pytest.approx(m["points"] - s.cfg.w_energy * m["energy"]
                                         - s.cfg.w_collision * (m["collisions_boat"] + m["collisions_obstacle"]))


def test_information_boundary():
    """A decentralised decision must depend only on the observation: changing hidden truth
    (teammates' true positions, true task completion) must not change it."""
    s = Simulation(SimConfig(policy="P3", **SHORT), 9)
    for _ in range(300):
        s.step()
    obs = s.observe(0)
    pol = make_policy("P3")
    d1 = pol.decide(obs)
    # perturb hidden truth
    s.S[0, 1:] += 25.0
    for tk in s.tasks:
        if tk.completed_by is None and tk.id not in s.minds[0].known_done:
            tk.completed_by = 3
    obs2 = s.observe(0)
    assert [t.id for t in obs2.tasks] == [t.id for t in obs.tasks]
    assert obs2.beliefs.keys() == obs.beliefs.keys()
    d2 = pol.decide(obs2)
    assert d1.task == d2.task and d1.utility == pytest.approx(d2.utility)
    # the observation carries no reference to the simulator or true states
    for v in vars(obs).values():
        assert not isinstance(v, Simulation)


def test_penalty_tables_monotone():
    T = penalty_tables(SimConfig())
    assert T.turn_pen[-1] > T.turn_pen[0]
    assert T.init(math.pi, 0) > T.init(0, T.v_ss)
    assert 1.0 < T.v_ss < SimConfig().v_max


def test_leader_assignment_only_travels_over_radio():
    """P5: with the radio off, followers can never receive an assignment (only the leader acts)."""
    s = Simulation(SimConfig(policy="P5", comm_mode="none", **SHORT), 4)
    s.run()
    followers_claimed = [e for e in s.events if e.kind == "decide" and e.boat != 0 and e.task >= 0]
    assert not followers_claimed
    s2 = Simulation(SimConfig(policy="P5", comm_mode="ideal", **SHORT), 4)
    s2.run()
    assert any(e.kind == "decide" and e.boat != 0 and e.task >= 0 for e in s2.events)
