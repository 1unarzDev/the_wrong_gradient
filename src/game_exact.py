"""Exact finite games: atomic congestion (route) games, Braess paradox, marginal-cost
incentives, difference-reward task games, and the Monderer-Shapley 4-cycle test.

Everything here is exhaustive and uses exact rational arithmetic (fractions.Fraction),
so the identities that the paper states as theorems are *checked*, not approximated.

Notation (matches the paper and STUDY_NOTES):
    n_e(a)  number of agents whose route uses edge e in profile a
    c_i(a)  = sum_{e in a_i} c_e(n_e(a))                 private (selfish) cost
    Phi(a)  = sum_e sum_{k=1}^{n_e(a)} c_e(k)             Rosenthal potential
    C(a)    = sum_e n_e(a) c_e(n_e(a)) = sum_i c_i(a)     team (social) cost
    m_e(k)  = k c_e(k) - (k-1) c_e(k-1)                   marginal social cost
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Num = Fraction
CostFn = Callable[[int], Fraction]


# ----------------------------------------------------------------------------------
# Atomic congestion (route) game on a directed graph
# ----------------------------------------------------------------------------------
@dataclass
class Edge:
    u: str
    v: str
    cost: CostFn            # c_e(k) for k >= 1 agents on the edge
    label: str = ""         # human-readable formula, e.g. "2k" or "10"

    @property
    def key(self) -> Tuple[str, str]:
        return (self.u, self.v)


@dataclass
class RouteGame:
    """Symmetric atomic congestion game: every agent travels source -> sink."""
    edges: List[Edge]
    source: str
    sink: str
    n_agents: int
    routes: List[Tuple[Tuple[str, str], ...]] = field(default_factory=list)

    def __post_init__(self):
        self._edge = {e.key: e for e in self.edges}
        if not self.routes:
            self.routes = enumerate_simple_paths(self.edges, self.source, self.sink)
        if not self.routes:
            raise ValueError("no source-sink route")

    # -- basic quantities --------------------------------------------------------
    def c(self, e: Tuple[str, str], k: int) -> Fraction:
        if k <= 0:
            return Fraction(0)
        return Fraction(self._edge[e].cost(k))

    def loads(self, a: Sequence[int]) -> Dict[Tuple[str, str], int]:
        n = {e.key: 0 for e in self.edges}
        for r in a:
            for e in self.routes[r]:
                n[e] += 1
        return n

    def private_cost(self, a: Sequence[int], i: int, edge_cost=None) -> Fraction:
        edge_cost = edge_cost or self.c
        n = self.loads(a)
        return sum((edge_cost(e, n[e]) for e in self.routes[a[i]]), Fraction(0))

    def private_costs(self, a: Sequence[int], edge_cost=None) -> List[Fraction]:
        return [self.private_cost(a, i, edge_cost) for i in range(len(a))]

    def team_cost(self, a: Sequence[int]) -> Fraction:
        n = self.loads(a)
        return sum((k * self.c(e, k) for e, k in n.items()), Fraction(0))

    def potential(self, a: Sequence[int], edge_cost=None) -> Fraction:
        """Rosenthal potential for the cost function `edge_cost` (default c_e)."""
        edge_cost = edge_cost or self.c
        n = self.loads(a)
        return sum((sum((edge_cost(e, k) for k in range(1, ne + 1)), Fraction(0))
                    for e, ne in n.items()), Fraction(0))

    def marginal_cost(self, e: Tuple[str, str], k: int) -> Fraction:
        """m_e(k) = k c_e(k) - (k-1) c_e(k-1); equals c_e(k) + toll (k-1)(c_e(k)-c_e(k-1))."""
        if k <= 0:
            return Fraction(0)
        return k * self.c(e, k) - (k - 1) * self.c(e, k - 1)

    def toll(self, e: Tuple[str, str], k: int) -> Fraction:
        return self.marginal_cost(e, k) - self.c(e, k)

    # -- profiles ----------------------------------------------------------------
    def profiles(self) -> Iterable[Tuple[int, ...]]:
        return itertools.product(range(len(self.routes)), repeat=self.n_agents)

    def best_response(self, a: Sequence[int], i: int, edge_cost=None,
                      ) -> Tuple[int, Fraction]:
        """Lowest-cost route for agent i holding others fixed (ties -> current route,
        then lowest index). Returns (route, cost)."""
        cur = self.private_cost(a, i, edge_cost)
        best_r, best_c = a[i], cur
        for r in range(len(self.routes)):
            b = list(a); b[i] = r
            cr = self.private_cost(b, i, edge_cost)
            if cr < best_c:
                best_r, best_c = r, cr
        return best_r, best_c

    def is_nash(self, a: Sequence[int], edge_cost=None) -> bool:
        for i in range(len(a)):
            r, cr = self.best_response(a, i, edge_cost)
            if cr < self.private_cost(a, i, edge_cost):
                return False
        return True

    def route_name(self, r: int) -> str:
        path = self.routes[r]
        return "".join([path[0][0]] + [e[1] for e in path])


def enumerate_simple_paths(edges: List[Edge], s: str, t: str):
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for e in edges:
        adj.setdefault(e.u, []).append(e.key)
    out = []

    def dfs(node, seen, path):
        if node == t:
            out.append(tuple(path)); return
        for e in adj.get(node, []):
            if e[1] not in seen:
                dfs(e[1], seen | {e[1]}, path + [e])
    dfs(s, {s}, [])
    return sorted(out, key=lambda p: (len(p), p))


# ----------------------------------------------------------------------------------
# Exhaustive analysis
# ----------------------------------------------------------------------------------
@dataclass
class GameAnalysis:
    equilibria: List[Tuple[int, ...]]
    optimum_cost: Fraction
    optima: List[Tuple[int, ...]]
    worst_eq_cost: Fraction
    best_eq_cost: Fraction

    @property
    def price_of_anarchy(self) -> Fraction:
        return self.worst_eq_cost / self.optimum_cost

    @property
    def price_of_stability(self) -> Fraction:
        return self.best_eq_cost / self.optimum_cost


def analyze(game: RouteGame, edge_cost=None) -> GameAnalysis:
    """Enumerate all |R|^n profiles: pure NE (w.r.t. edge_cost, default selfish c_e)
    and the social optimum of the TRUE team cost C."""
    eqs, costs = [], {}
    for a in game.profiles():
        costs[a] = game.team_cost(a)
        if game.is_nash(a, edge_cost):
            eqs.append(a)
    opt = min(costs.values())
    optima = [a for a, c in costs.items() if c == opt]
    eq_costs = [costs[a] for a in eqs]
    return GameAnalysis(eqs, opt, optima, max(eq_costs), min(eq_costs))


def verify_exact_potential(game: RouteGame, edge_cost=None) -> int:
    """Assert Phi(b)-Phi(a) == c_i(b)-c_i(a) for EVERY profile a and EVERY unilateral
    deviation b of every agent i. Returns the number of deviations checked."""
    checked = 0
    R = len(game.routes)
    for a in game.profiles():
        pa = game.potential(a, edge_cost)
        for i in range(game.n_agents):
            ca = game.private_cost(a, i, edge_cost)
            for r in range(R):
                if r == a[i]:
                    continue
                b = list(a); b[i] = r
                dphi = game.potential(b, edge_cost) - pa
                dci = game.private_cost(b, i, edge_cost) - ca
                assert dphi == dci, (a, i, r, dphi, dci)
                checked += 1
    return checked


def verify_marginal_telescoping(game: RouteGame) -> int:
    """Assert that the Rosenthal potential of the marginal-cost game equals the team
    cost C(a) for every profile a (telescoping sum). Returns profiles checked."""
    n = 0
    for a in game.profiles():
        assert game.potential(a, game.marginal_cost) == game.team_cost(a), a
        n += 1
    return n


@dataclass
class BRStep:
    t: int
    profile: Tuple[int, ...]
    mover: Optional[int]
    old_route: Optional[int]
    new_route: Optional[int]
    mover_old_cost: Optional[Fraction]   # in the game being played (selfish or marginal)
    mover_new_cost: Optional[Fraction]
    phi: Fraction                         # potential of the game being played
    team_cost: Fraction                   # TRUE team cost C


def best_response_dynamics(game: RouteGame, start: Sequence[int], edge_cost=None,
                           order: Optional[Sequence[int]] = None, max_rounds: int = 100,
                           ) -> List[BRStep]:
    """Sequential (one agent at a time) best-response dynamics, round-robin order.
    Terminates at a pure NE because Phi strictly decreases each move (finite game)."""
    a = list(start)
    order = list(order) if order is not None else list(range(game.n_agents))
    trace = [BRStep(0, tuple(a), None, None, None, None, None,
                    game.potential(a, edge_cost), game.team_cost(a))]
    t = 0
    for _ in range(max_rounds):
        moved = False
        for i in order:
            old_c = game.private_cost(a, i, edge_cost)
            r, new_c = game.best_response(a, i, edge_cost)
            if new_c < old_c:
                old_r = a[i]; a[i] = r; t += 1; moved = True
                trace.append(BRStep(t, tuple(a), i, old_r, r, old_c, new_c,
                                    game.potential(a, edge_cost), game.team_cost(a)))
        if not moved:
            return trace
    raise RuntimeError("best-response dynamics did not converge")


# ----------------------------------------------------------------------------------
# The Braess instance used throughout (boat-themed, fictional numbers)
# ----------------------------------------------------------------------------------
F = Fraction


def braess_game(with_shortcut: bool = True, n_agents: int = 4) -> RouteGame:
    """Dock S -> inspection area T for n boats. Travel times in minutes.

    S->A narrow canal  c(k)=2k   (congestible: boats queue/slow each other)
    A->T open lake     c(k)=10
    S->B open lake     c(k)=10
    B->T narrow canal  c(k)=2k
    A->B new shortcut gate c(k)=1 (only if with_shortcut)
    """
    edges = [
        Edge("S", "A", lambda k: F(2 * k), "2k"),
        Edge("A", "T", lambda k: F(10), "10"),
        Edge("S", "B", lambda k: F(10), "10"),
        Edge("B", "T", lambda k: F(2 * k), "2k"),
    ]
    if with_shortcut:
        edges.append(Edge("A", "B", lambda k: F(1), "1"))
    return RouteGame(edges, "S", "T", n_agents)


def find_paradox_move(game: RouteGame, trace: List[BRStep]) -> Optional[BRStep]:
    """First step where the mover strictly improves privately but C strictly rises."""
    for prev, step in zip(trace, trace[1:]):
        if step.mover_new_cost < step.mover_old_cost and step.team_cost > prev.team_cost:
            return step
    return None


# ----------------------------------------------------------------------------------
# General finite games: the Monderer-Shapley 4-cycle test
# ----------------------------------------------------------------------------------
def four_cycle_sums(n_actions: Sequence[int], cost: Callable[[Tuple[int, ...], int], Fraction]):
    """For every pair of players i<j, profile a, and alternative actions x!=a_i, y!=a_j,
    compute the circulation around the closed 4-cycle
        a -> (x, a_-i) -> (x, y, a_-ij) -> (a_i, y, a_-ij) -> a
    I(gamma) = sum over steps of [cost of mover after - cost of mover before].
    Monderer & Shapley (1996): the game is an exact potential game iff I == 0 for all
    such 4-cycles. Yields (cycle, I)."""
    N = len(n_actions)
    for a in itertools.product(*[range(m) for m in n_actions]):
        for i, j in itertools.combinations(range(N), 2):
            for x in range(n_actions[i]):
                if x == a[i]:
                    continue
                for y in range(n_actions[j]):
                    if y == a[j]:
                        continue
                    p0 = list(a)
                    p1 = list(a); p1[i] = x
                    p2 = list(p1); p2[j] = y
                    p3 = list(a); p3[j] = y
                    seq = [(p0, p1, i), (p1, p2, j), (p2, p3, i), (p3, p0, j)]
                    I = sum((cost(tuple(q), k) - cost(tuple(p), k) for p, q, k in seq),
                            Fraction(0))
                    yield (tuple(p0), tuple(p1), tuple(p2), tuple(p3)), I


def is_exact_potential_game(n_actions, cost) -> bool:
    return all(I == 0 for _, I in four_cycle_sums(n_actions, cost))


def matching_pennies_cost(a: Tuple[int, ...], i: int) -> Fraction:
    """Classic non-potential game. Player 0 wants to match, player 1 to mismatch.
    Cost = -payoff."""
    match = a[0] == a[1]
    payoff = (1 if match else -1) if i == 0 else (-1 if match else 1)
    return Fraction(-payoff)


# ----------------------------------------------------------------------------------
# Task-allocation game with difference rewards
# ----------------------------------------------------------------------------------
@dataclass
class TaskGame:
    """n boats each choose one task (or IDLE = -1 encoded as index m).
    value[j] points for task j; q[i][j] success probability of boat i on task j
    (e.g. chance of beating the deadline); travel[i][j] cost (points) of boat i going
    to task j. A task is scored at most once:
        W(a) = sum_j value_j * (1 - prod_{i: a_i=j} (1 - q_ij)) - sum_i travel_i,a_i
    """
    value: List[Fraction]
    q: List[List[Fraction]]
    travel: List[List[Fraction]]

    @property
    def n(self) -> int:
        return len(self.q)

    @property
    def m(self) -> int:
        return len(self.value)

    @property
    def n_actions(self) -> List[int]:
        return [self.m + 1] * self.n          # last action = idle

    def welfare(self, a: Tuple[int, ...]) -> Fraction:
        W = Fraction(0)
        for j in range(self.m):
            miss = Fraction(1)
            for i, ai in enumerate(a):
                if ai == j:
                    miss *= (1 - self.q[i][j])
            W += self.value[j] * (1 - miss)
        for i, ai in enumerate(a):
            if ai < self.m:
                W -= self.travel[i][ai]
        return W

    def difference_reward(self, a: Tuple[int, ...], i: int) -> Fraction:
        """u_i(a) = W(a) - W(idle_i, a_-i): agent i's marginal contribution."""
        b = list(a); b[i] = self.m
        return self.welfare(a) - self.welfare(tuple(b))

    def selfish_reward(self, a: Tuple[int, ...], i: int) -> Fraction:
        """'Full credit' private payoff: i counts the whole task value it might win,
        ignoring that a teammate on the same task makes its effort redundant."""
        if a[i] == self.m:
            return Fraction(0)
        j = a[i]
        return self.value[j] * self.q[i][j] - self.travel[i][j]

    # cost = -payoff wrappers for the 4-cycle test
    def diff_cost(self, a, i):
        return -self.difference_reward(a, i)

    def selfish_cost(self, a, i):
        return -self.selfish_reward(a, i)

    def profiles(self):
        return itertools.product(*[range(k) for k in self.n_actions])

    def nash_equilibria(self, payoff) -> List[Tuple[int, ...]]:
        eqs = []
        for a in self.profiles():
            ok = True
            for i in range(self.n):
                cur = payoff(a, i)
                for x in range(self.m + 1):
                    b = list(a); b[i] = x
                    if payoff(tuple(b), i) > cur:
                        ok = False; break
                if not ok:
                    break
            if ok:
                eqs.append(a)
        return eqs


def demo_task_game() -> TaskGame:
    """3 boats, 2 tasks: one big 'docking' task (10 pts) that everybody likes and a
    smaller 'buoy inspection' task (6 pts). Fictional numbers."""
    return TaskGame(
        value=[F(10), F(6)],
        q=[[F(9, 10), F(8, 10)],
           [F(8, 10), F(9, 10)],
           [F(7, 10), F(6, 10)]],
        travel=[[F(1), F(1)],
                [F(1), F(1)],
                [F(1), F(2)]],
    )


def summarize_braess(n_agents: int = 4) -> dict:
    """All numbers for the paper/notes are produced here and nowhere else."""
    g0, g1 = braess_game(False, n_agents), braess_game(True, n_agents)
    A0, A1 = analyze(g0), analyze(g1)
    A1m = analyze(g1, g1.marginal_cost)
    # Start at the pre-shortcut optimum (2 via A, 2 via B) and open the shortcut.
    sat = g1.routes.index((("S", "A"), ("A", "T")))
    sbt = g1.routes.index((("S", "B"), ("B", "T")))
    start = [sat] * (n_agents // 2) + [sbt] * (n_agents - n_agents // 2)
    tr_self = best_response_dynamics(g1, start)
    tr_marg = best_response_dynamics(g1, tr_self[-1].profile, g1.marginal_cost)
    return dict(no_shortcut=A0, shortcut=A1, shortcut_marginal=A1m,
                trace_selfish=tr_self, trace_marginal=tr_marg,
                paradox_step=find_paradox_move(g1, tr_self), game=g1, game0=g0)


if __name__ == "__main__":
    s = summarize_braess()
    g = s["game"]
    def names(ps): return [tuple(g.route_name(r) for r in p) for p in ps]
    print("routes:", [g.route_name(r) for r in range(len(g.routes))])
    for k in ("no_shortcut", "shortcut", "shortcut_marginal"):
        A = s[k]
        gg = s["game0"] if k == "no_shortcut" else g
        print(f"\n[{k}] #NE={len(A.equilibria)} optimum C={A.optimum_cost} "
              f"best-NE C={A.best_eq_cost} worst-NE C={A.worst_eq_cost} "
              f"PoA={A.price_of_anarchy} ({float(A.price_of_anarchy):.4f})")
        print("  NE:", sorted(set(tuple(sorted(x)) for x in
                                 [tuple(gg.route_name(r) for r in p) for p in A.equilibria])))
    print("\nselfish best-response trace (start: 2-2 split, shortcut opened):")
    for st in s["trace_selfish"]:
        mv = "" if st.mover is None else (f"boat {st.mover+1}: {g.route_name(st.old_route)}->"
              f"{g.route_name(st.new_route)} private {st.mover_old_cost}->{st.mover_new_cost}")
        print(f"  t={st.t} Phi={st.phi} C={st.team_cost} {mv}")
    print("\nmarginal-cost best-response trace from the selfish NE:")
    for st in s["trace_marginal"]:
        mv = "" if st.mover is None else (f"boat {st.mover+1}: {g.route_name(st.old_route)}->"
              f"{g.route_name(st.new_route)} marginal {st.mover_old_cost}->{st.mover_new_cost}")
        print(f"  t={st.t} Phi_m={st.phi} C={st.team_cost} {mv}")
    print("\npotential identity deviations checked:", verify_exact_potential(g))
    print("telescoping profiles checked:", verify_marginal_telescoping(g))
    tg = demo_task_game()
    print("\ntask game: difference reward exact potential?",
          is_exact_potential_game(tg.n_actions, tg.diff_cost))
    print("matching pennies exact potential?", is_exact_potential_game([2, 2], matching_pennies_cost))
