"""Return rush (Braess on the v2 course): fast checks of the scenario runner and the calibrated exact game."""
import json
import math
import os
from fractions import Fraction

import pytest

from rb2.braess2 import (CALIBRATED, CALIBRATED_GEOM, ROUTES, RushGeom, assign_routes, edge_times, get_course,
                         make_game, predict, profile_game, run_scenario, start_slots)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = RushGeom(**CALIBRATED_GEOM)


def test_scenario_deterministic_for_a_seed():
    a = run_scenario(GEOM, ["west", "east"], 3, shortcut_open=False)
    b = run_scenario(GEOM, ["west", "east"], 3, shortcut_open=False)
    assert a["times"] == b["times"] and a["node_times"] == b["node_times"]
    c = run_scenario(GEOM, ["west", "east"], 4, shortcut_open=False)
    assert c["times"] != a["times"]                      # the seed matters (start jitter + disturbances)


def test_start_jitter_is_common_across_conditions():
    rc = get_course(GEOM)
    s1, s2 = start_slots(rc, 4, 11), start_slots(rc, 4, 11)
    assert s1 == s2
    r1 = assign_routes({"west": 2, "east": 2, "shortcut": 0}, s1)
    r2 = assign_routes({"west": 0, "east": 1, "shortcut": 3}, s1)
    assert sorted(r1) == ["east", "east", "west", "west"] and sorted(r2) == ["east", "shortcut", "shortcut", "shortcut"]


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_reaches_T_in_node_order(route):
    res = run_scenario(GEOM, [route], 1, shortcut_open=(route == "shortcut"))
    assert res["all_finished"]
    nt = res["node_times"][0]
    nodes = {"west": "SAT", "east": "SBT", "shortcut": "SABT"}[route]
    ts = [nt[n] for n in nodes]
    assert ts == sorted(ts) and ts[-1] == res["times"][0] < 200
    assert res["contacts"]["structure"] == 0
    assert all(v > 0 for v in edge_times(res)[0].values())


def test_shortcut_route_rejected_when_closed():
    with pytest.raises(ValueError):
        run_scenario(GEOM, ["shortcut"], 1, shortcut_open=False)


def test_channel_discipline_one_boat_at_a_time():
    res = run_scenario(GEOM, ["shortcut", "shortcut", "west", "east"], 5, shortcut_open=True)
    assert res["all_finished"]
    assert res["max_in_channel"]["west"] <= 1 and res["max_in_channel"]["east"] <= 1
    for ch in ("west", "east"):
        inside = None
        for t, c, ev, i in res["channel_log"]:
            if c != ch:
                continue
            if ev == "grant":
                assert inside is None, (ch, t, i)
                inside = i
            elif ev in ("exit", "revoke"):
                assert inside == i
                inside = None
        assert inside is None
    # queueing is real: the west channel admits 3 boats in sequence, so their S->A times are spread out
    sa = sorted(e[("S", "A")] for e in edge_times(res) if ("S", "A") in e)
    assert sa[-1] - sa[0] > 8.0


def test_calibrated_game_predicts_a_paradox():
    P = predict(CALIBRATED, 4)
    assert P["paradox"]
    assert P["open"]["worst_ne"] > P["open"]["opt"]
    assert P["open"]["worst_ne"] > P["closed"]["worst_ne"]
    # the tolled (marginal-cost) game's equilibrium is never worse than the selfish one
    assert P["tolled"]["worst_ne_true_cost"] <= P["open"]["worst_ne"]
    # exact rationals: the costs are rounded to 0.1 s
    g = make_game(CALIBRATED, True, 4)
    assert all(isinstance(g.c(e.key, 2), Fraction) and (g.c(e.key, 2) * 10).denominator == 1 for e in g.edges)


def test_calibration_constant_matches_results_file():
    p = os.path.join(ROOT, "data", "return_rush", "results.json")
    if not os.path.exists(p):
        pytest.skip("run analysis/return_rush.py first")
    R = json.load(open(p))
    A = R["primary"]
    assert A["geom"] == {k: CALIBRATED_GEOM[k] for k in A["geom"]}
    for e, vals in CALIBRATED.items():
        assert [round(v, 6) for v in vals] == [round(v, 6) for v in A["calibration"]["costs"][e]]


def test_profile_game_on_textbook_braess():
    # route-level costs of the classic 2k / 10 / 1 instance give the textbook NE and optimum
    c = {"SA": lambda k: 2 * k, "BT": lambda k: 2 * k, "AT": lambda k: 10, "SB": lambda k: 10, "AB": lambda k: 1}
    table = {}
    for w in range(5):
        for s in range(5 - w):
            e = 4 - w - s
            na, nb = w + s, e + s
            table[(w, e, s)] = {"west": c["SA"](na) + 10, "east": 10 + c["BT"](nb), "shortcut": 2 * na + 1 + 2 * nb}
    G1, G0 = profile_game(table, 4, True), profile_game(table, 4, False)
    assert G1["equilibria"] == [(0, 0, 4)] and math.isclose(G1["worst_eq"], 68)
    assert G0["equilibria"] == [(2, 2, 0)] and math.isclose(G0["worst_eq"], 56)


# ---- experiment 2: road-like slow-zone congestion -------------------------------------------------
SZ = RushGeom(congestion="slow_zone", alpha=0.5, v_open=0.5)


def test_slow_zone_has_no_walls_and_no_single_file():
    rc = get_course(SZ)
    assert rc.channels == {} and rc.course.piers == [] and set(rc.czones) == {"west", "east"}
    res = run_scenario(SZ, ["shortcut"] * 4, 2, shortcut_open=True)
    assert res["all_finished"] and res["contacts"]["structure"] == 0 and res["channel_log"] == []
    assert res["max_in_zone"]["west"] == 4 and res["max_in_zone"]["east"] >= 2   # boats share the zone (abreast)


def test_slow_zone_congestion_slows_everyone_and_is_deterministic():
    one = run_scenario(SZ, ["west"], 2, shortcut_open=False)
    four = run_scenario(SZ, ["west"] * 4, 2, shortcut_open=False)
    sa1 = edge_times(one)[0][("S", "A")]
    sa4 = [e[("S", "A")] for e in edge_times(four)]
    # road-like: every boat is slower with 4 in the zone, and nobody waits in a queue
    assert min(sa4) > sa1 + 5.0 and max(sa4) - min(sa4) < 15.0
    assert run_scenario(SZ, ["west"] * 4, 2, shortcut_open=False)["times"] == four["times"]


# ---- coupling study: potential structure ------------------------------------------------------------
def test_separable_calibrated_game_has_zero_circulation():
    """Monderer-Shapley: the separable (Rosenthal) calibrated game is an exact potential game, so the movers' cost
    changes around every closed 4-cycle of unilateral deviations sum to exactly 0 (Fractions)."""
    import itertools
    from src.game_exact import four_cycle_sums
    g = make_game(CALIBRATED, True, 4)
    I = [x for _, x in four_cycle_sums([len(g.routes)] * 4, lambda a, i: g.private_cost(a, i))]
    assert len(I) == 1944 and all(x == 0 for x in I)
    # a non-separable cost (B->T cheaper for shortcut users than for east-route users) breaks it
    lab = {g.route_name(r): r for r in range(len(g.routes))}

    def coupled(a, i):
        c = g.private_cost(a, i)
        if a[i] == lab["SABT"]:
            c -= Fraction(sum(1 for r in a if r == lab["SBT"]), 1)
        return c
    assert any(x != 0 for _, x in four_cycle_sums([len(g.routes)] * 4, coupled))
