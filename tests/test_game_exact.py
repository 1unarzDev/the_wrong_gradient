"""Exhaustive checks of every exact claim made about the static games."""
import itertools
import random
from fractions import Fraction as F

import pytest

from src.game_exact import (Edge, RouteGame, TaskGame, analyze, best_response_dynamics,
                            braess_game, demo_task_game, find_paradox_move,
                            is_exact_potential_game, matching_pennies_cost,
                            summarize_braess, verify_exact_potential,
                            verify_marginal_telescoping)


def test_braess_numbers():
    s = summarize_braess(4)
    A0, A1, A1m = s["no_shortcut"], s["shortcut"], s["shortcut_marginal"]
    assert A0.optimum_cost == 56 and A0.worst_eq_cost == 56 and A0.best_eq_cost == 56
    assert A1.optimum_cost == 56                      # the shortcut does not help OPT
    assert A1.worst_eq_cost == 68 and A1.best_eq_cost == 68
    assert A1.price_of_anarchy == F(17, 14)
    g = s["game"]
    sabt = g.routes.index((("S", "A"), ("A", "B"), ("B", "T")))
    assert A1.equilibria == [(sabt,) * 4]             # unique NE: everybody takes shortcut
    # marginal-cost incentives: every NE of the tolled game is socially optimal HERE
    assert A1m.worst_eq_cost == 56
    # Braess paradox: adding an edge raised the equilibrium cost
    assert A1.worst_eq_cost > A0.worst_eq_cost


def test_braess_trace_paradox():
    s = summarize_braess(4)
    tr = s["trace_selfish"]
    phis = [st.phi for st in tr]
    Cs = [st.team_cost for st in tr]
    assert phis == [52, 49, 48, 45, 44]
    assert Cs == [56, 57, 62, 63, 68]
    assert all(b < a for a, b in zip(phis, phis[1:]))     # Phi strictly decreasing
    assert all(b > a for a, b in zip(Cs, Cs[1:]))         # C strictly increasing
    p = s["paradox_step"]
    assert p.t == 1 and p.mover_old_cost == 14 and p.mover_new_cost == 11
    # marginal dynamics: Phi_m == C at every step and ends at the optimum
    trm = s["trace_marginal"]
    assert all(st.phi == st.team_cost for st in trm)
    assert trm[-1].team_cost == 56


def test_potential_identity_braess():
    for shortcut in (False, True):
        g = braess_game(shortcut, 4)
        assert verify_exact_potential(g) > 0
        assert verify_exact_potential(g, g.marginal_cost) > 0
        assert verify_marginal_telescoping(g) == len(g.routes) ** 4


def random_congestion_game(rng, n_agents):
    nodes = ["S", "A", "B", "C", "T"]
    edges = []
    for u, v in itertools.combinations(nodes, 2):       # DAG in this node order
        if rng.random() < 0.7 or (u, v) == ("S", "T"):
            base = F(rng.randint(0, 6))
            slope = F(rng.randint(0, 4))
            quad = F(rng.randint(0, 2))
            edges.append(Edge(u, v, (lambda b, s, q: (lambda k: b + s * k + q * k * k))(base, slope, quad)))
    return RouteGame(edges, "S", "T", n_agents)


@pytest.mark.parametrize("seed", range(12))
def test_random_games_potential_and_telescoping(seed):
    rng = random.Random(seed)
    g = random_congestion_game(rng, rng.choice([2, 3]))
    verify_exact_potential(g)                      # Rosenthal: exact potential
    verify_exact_potential(g, g.marginal_cost)     # tolled game is also exact potential
    verify_marginal_telescoping(g)                 # ... and its potential is C
    # any NE of the selfish game exists (Rosenthal) and BR dynamics reaches one
    A = analyze(g)
    assert len(A.equilibria) >= 1
    tr = best_response_dynamics(g, [0] * g.n_agents)
    assert g.is_nash(tr[-1].profile)
    # strict improvements in the marginal-cost game strictly lower C
    trm = best_response_dynamics(g, [0] * g.n_agents, g.marginal_cost)
    Cs = [st.team_cost for st in trm]
    assert all(b < a for a, b in zip(Cs, Cs[1:]))
    # PoA >= 1 by definition of the optimum
    assert A.price_of_anarchy >= 1


def test_four_cycle_characterization():
    g = braess_game(True, 3)
    nA = [len(g.routes)] * 3
    assert is_exact_potential_game(nA, lambda a, i: g.private_cost(a, i))
    assert not is_exact_potential_game([2, 2], matching_pennies_cost)


def test_task_game_difference_reward_is_exact_potential():
    tg = demo_task_game()
    assert is_exact_potential_game(tg.n_actions, tg.diff_cost)
    # explicit identity: delta u_i == delta W for every unilateral deviation
    for a in tg.profiles():
        for i in range(tg.n):
            for x in range(tg.m + 1):
                b = list(a); b[i] = x; b = tuple(b)
                assert tg.difference_reward(b, i) - tg.difference_reward(a, i) == \
                    tg.welfare(b) - tg.welfare(a)


@pytest.mark.parametrize("seed", range(8))
def test_random_task_games(seed):
    rng = random.Random(100 + seed)
    n, m = rng.choice([2, 3]), rng.choice([2, 3])
    tg = TaskGame(value=[F(rng.randint(1, 10)) for _ in range(m)],
                  q=[[F(rng.randint(1, 10), 10) for _ in range(m)] for _ in range(n)],
                  travel=[[F(rng.randint(0, 3)) for _ in range(m)] for _ in range(n)])
    assert is_exact_potential_game(tg.n_actions, tg.diff_cost)
    opt = max(tg.welfare(a) for a in tg.profiles())
    eqs = tg.nash_equilibria(tg.difference_reward)
    assert eqs, "potential game must have a pure NE"
    assert opt in [tg.welfare(a) for a in eqs]     # the welfare maximizer is always a NE


def test_task_game_demo_numbers():
    tg = demo_task_game()
    opt = max(tg.profiles(), key=tg.welfare)
    assert opt == (0, 1, 2) and tg.welfare(opt) == F(62, 5)
    assert tg.nash_equilibria(tg.selfish_reward) == [(0, 0, 0)]
    assert tg.welfare((0, 0, 0)) == F(347, 50)
    eqs = tg.nash_equilibria(tg.difference_reward)
    assert set(eqs) == {(0, 1, 2), (1, 0, 0)}       # a suboptimal (local) equilibrium exists
    assert tg.welfare((1, 0, 0)) == F(56, 5)


def externality_terms(g, a, i, r):
    """ΔC = Δc_i + (delay imposed on boats joined) − (relief given to boats left)."""
    P, Q = set(g.routes[a[i]]), set(g.routes[r])
    load = g.loads(a)
    imposed = sum((load[e] * (g.c(e, load[e] + 1) - g.c(e, load[e])) for e in Q - P), F(0))
    relief = sum(((load[e] - 1) * (g.c(e, load[e]) - g.c(e, load[e] - 1)) for e in P - Q), F(0))
    return imposed, relief


@pytest.mark.parametrize("seed", list(range(6)) + ["braess"])
def test_externality_decomposition(seed):
    g = braess_game(True, 4) if seed == "braess" else random_congestion_game(random.Random(500 + seed), 3)
    for a in g.profiles():
        for i in range(g.n_agents):
            for r in range(len(g.routes)):
                if r == a[i]:
                    continue
                b = list(a); b[i] = r
                dC = g.team_cost(b) - g.team_cost(a)
                dci = g.private_cost(b, i) - g.private_cost(a, i)
                imp, rel = externality_terms(g, a, i, r)
                assert dC == dci + imp - rel


def test_externality_first_braess_move():
    g = braess_game(True, 4)
    sat, sbt, sabt = (g.routes.index(r) for r in ((("S", "A"), ("A", "T")), (("S", "B"), ("B", "T")),
                                                  (("S", "A"), ("A", "B"), ("B", "T"))))
    imp, rel = externality_terms(g, (sat, sat, sbt, sbt), 0, sabt)
    assert (imp, rel) == (4, 0)          # joins B→T with 2 boats: 2·(c(3)−c(2)) = 2·2
