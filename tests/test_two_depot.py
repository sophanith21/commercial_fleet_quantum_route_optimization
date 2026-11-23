import numpy as np
import pytest

from fleetopt.exact import brute_force_qubo
from fleetopt.two_depot import (
    D1, D2, brute_force_plans, build_tour_qubo, decode_tour, distance_matrix, encode_tour,
    energy, greedy_plan, split_routes, total_distance,
)

DESTS = ["A", "B", "C"]
REPORT_COORDS = {D1: (0, 0), D2: (20, 0), "A": (2, 4), "B": (18, 4), "C": (16, -3)}


@pytest.fixture
def report():
    d = distance_matrix(REPORT_COORDS)
    Q, variables, offset, A = build_tour_qubo(d, DESTS)
    return d, Q, variables, offset, A


def random_instance(seed, n):
    rng = np.random.default_rng(seed)
    names = [D1, D2, *[f"S{k}" for k in range(n)]]
    return distance_matrix({k: tuple(rng.uniform(0, 100, 2)) for k in names}), names[2:]


def test_report_qubo_size(report):
    _, Q, variables, _, A = report
    assert len(variables) == 16
    assert A == pytest.approx(36.88, abs=0.01)
    diag = sum(1 for u, v in Q if u == v)
    off = sum(1 for u, v in Q if u != v)
    assert diag == 16
    print(f"\noff-diagonal couplings: {off} (report says 84)")


def test_energy_matches_distance_on_every_plan(report):
    d, Q, _, offset, _ = report
    for dist, tour in brute_force_plans(d, DESTS):
        assert energy(Q, encode_tour(tour)) + offset == pytest.approx(dist)


def test_report_optimum(report):
    d, Q, variables, offset, A = report
    plans = brute_force_plans(d, DESTS)
    assert len(plans) == 24
    assert plans[0][0] == pytest.approx(25.70, abs=0.01)
    assert plans[-1][0] == pytest.approx(78.88, abs=0.01)

    x, e = brute_force_qubo(Q, variables)       # all 2^16 bit strings
    tour = decode_tour(x)
    assert tour is not None
    assert e + offset == pytest.approx(plans[0][0])
    r1, r2 = split_routes(tour)
    assert r1 == [D1, "A", D1]
    assert set(r2) == {D2, "B", "C"}


def test_exactly_24_feasible_and_gap_to_invalid(report):
    d, Q, variables, offset, _ = report
    n = len(variables)
    feasible, best_invalid = 0, np.inf
    for bits in range(1 << n):
        x = {v: (bits >> k) & 1 for k, v in enumerate(variables)}
        e = energy(Q, x) + offset
        if decode_tour(x) is None:
            best_invalid = min(best_invalid, e)
        else:
            feasible += 1
    assert feasible == 24
    print(f"\nlowest invalid energy: {best_invalid:.2f} (report says 83.23)")
    assert best_invalid > 25.70


@pytest.mark.parametrize("seed", range(20))
def test_qubo_ground_state_is_optimal_random(seed):
    d, dests = random_instance(seed, 3)
    Q, variables, offset, _ = build_tour_qubo(d, dests)
    x, e = brute_force_qubo(Q, variables)
    tour = decode_tour(x)
    assert tour is not None
    assert total_distance(d, tour) == pytest.approx(brute_force_plans(d, dests)[0][0])


def test_greedy_never_beats_optimum():
    for seed in range(50):
        d, dests = random_instance(seed, 4)
        assert total_distance(d, greedy_plan(d, dests)) >= brute_force_plans(d, dests)[0][0] - 1e-9


# ---------------------------------------------------------------------------------------------
# Helpers and edge cases
# ---------------------------------------------------------------------------------------------

def test_report_max_cost_and_greedy(report):
    from fleetopt.two_depot import max_tour_cost
    d = report[0]
    assert max_tour_cost(d, DESTS) == pytest.approx(18.44, abs=0.01)
    tour = greedy_plan(d, DESTS)
    assert sorted(tour) == sorted([D1, D2, *DESTS])
    assert total_distance(d, tour) == pytest.approx(25.70, abs=0.01)   # greedy is optimal here


def test_idle_van_routes():
    r1, r2 = split_routes([D1, D2, "A", "B", "C"])
    assert r1 == [D1, D1] and r2 == [D2, "A", "B", "C", D2]
    r1, r2 = split_routes([D1, "A", "B", "C", D2])
    assert r1 == [D1, "A", "B", "C", D1] and r2 == [D2, D2]


def test_decode_rejects_invalid_bit_strings():
    x = encode_tour([D1, "A", D2, "B", "C"])
    assert decode_tour(x) == [D1, "A", D2, "B", "C"]
    x[("A", 2)] = 1                                # A in two positions, position 2 doubly filled
    assert decode_tour(x) is None
    x = {k: 0 for k in x}                           # nothing placed
    assert decode_tour(x) is None


def test_symmetric_tours_cost_the_same(report):
    d = report[0]
    assert total_distance(d, [D1, "A", D2, "B", "C"]) == pytest.approx(
        total_distance(d, [D1, "A", D2, "C", "B"]))


# ---------------------------------------------------------------------------------------------
# Heuristic solvers on the QUBO (normalised: B = 1 / max c, A = 1, see two_depot_penalty_sweep)
# ---------------------------------------------------------------------------------------------

def normalised_qubo(d, dests, factor=1.0):
    from fleetopt.two_depot import max_tour_cost
    mc = max_tour_cost(d, dests)
    Q, variables, offset, _ = build_tour_qubo(d, dests, A=factor, B=1.0 / mc)
    return Q, variables, offset, mc


@pytest.mark.parametrize("n,seed", [(3, 0), (3, 1), (4, 0), (4, 1), (5, 0)])
def test_simulated_annealing_finds_optimum(n, seed):
    from fleetopt.heuristic import simulated_annealing
    d, dests = random_instance(seed, n)
    Q, variables, offset, mc = normalised_qubo(d, dests)
    x, e = simulated_annealing(Q, variables, seed=seed)
    tour = decode_tour(x)
    opt = brute_force_plans(d, dests)[0][0]
    assert tour is not None
    assert total_distance(d, tour) == pytest.approx(opt)
    assert (e + offset) * mc == pytest.approx(opt)      # returned energy is the real distance


@pytest.mark.parametrize("n,seed", [(3, 0), (4, 0), (5, 0), (5, 1)])
def test_tabu_finds_optimum(n, seed):
    pytest.importorskip("dwave.samplers")
    from fleetopt.heuristic import tabu
    d, dests = random_instance(seed, n)
    Q, variables, offset, mc = normalised_qubo(d, dests)
    x, e = tabu(Q, variables, seed=seed)
    tour = decode_tour(x)
    assert tour is not None
    assert total_distance(d, tour) == pytest.approx(brute_force_plans(d, dests)[0][0])


def test_heuristics_on_report_example(report):
    from fleetopt.heuristic import simulated_annealing
    d = report[0]
    Q, variables, _, _ = normalised_qubo(d, DESTS)
    tour = decode_tour(simulated_annealing(Q, variables, seed=0)[0])
    assert split_routes(tour)[0] == [D1, "A", D1]


def test_too_small_penalty_breaks_ground_state():
    """A = 0.25 max c: on these layouts the lowest-energy bit string is not a valid plan."""
    broken = 0
    for seed in range(5):
        d, dests = random_instance(seed, 3)
        Q, variables, _, _ = normalised_qubo(d, dests, factor=0.25)
        x, _ = brute_force_qubo(Q, variables)
        broken += decode_tour(x) is None
    assert broken == 5


# ---------------------------------------------------------------------------------------------
# Road-network layouts (SUMO)
# ---------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_sumo_layout_qubo_matches_brute_force():
    import os
    from fleetopt.problem import DEFAULT_NET, load_road_graph
    from fleetopt.two_depot import sumo_layout
    pytest.importorskip("sumolib")
    if not os.path.exists(DEFAULT_NET):
        pytest.skip("SUMO network file not found")
    G = load_road_graph()
    for seed in range(3):
        d, dests, nodes = sumo_layout(G, seed, 3)
        assert len(set(nodes.values())) == 5                     # five distinct junctions
        Q, variables, offset, mc = normalised_qubo(d, dests)
        x, e = brute_force_qubo(Q, variables)
        tour = decode_tour(x)
        opt = brute_force_plans(d, dests)[0][0]
        assert tour is not None
        assert total_distance(d, tour) == pytest.approx(opt)
        assert (e + offset) * mc == pytest.approx(opt)


@pytest.mark.slow
def test_layout_0_is_the_notebook_scenario():
    """Layout 0 (3 stops) reproduces preprocessing_new_problem.ipynb: same travel times, 847.5 s plan."""
    import os
    from fleetopt.problem import DEFAULT_NET, load_road_graph
    from fleetopt.two_depot import road_layout
    pytest.importorskip("sumolib")
    if not os.path.exists(DEFAULT_NET):
        pytest.skip("SUMO network file not found")
    d, dests, nodes = road_layout(load_road_graph(), 0, 3)
    assert dests == ["A", "B", "C"]
    assert d[D1, "A"] == pytest.approx(79.968, abs=1e-3)
    assert d[D1, D2] == pytest.approx(144.834, abs=1e-3)
    assert d["A", "B"] == pytest.approx(169.694, abs=1e-3)
    best, tour = brute_force_plans(d, dests)[0]
    assert best == pytest.approx(847.50, abs=0.01)
    assert tour == [D1, "A", "B", D2, "C"]
