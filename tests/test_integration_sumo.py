import os

import pytest

from fleetopt.evaluate import decode, is_feasible, objective
from fleetopt.exact import brute_force_qubo, enumerate_feasible
from fleetopt.heuristic import greedy, simulated_annealing
from fleetopt.qubo import build_qubo, qubo_offset

NET = os.path.join(os.path.dirname(__file__), "..", "SUMO_GEN_SCENARIO", "osm.net.xml.gz")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sumo_problem():
    pytest.importorskip("sumolib")
    if not os.path.exists(NET):
        pytest.skip("SUMO network file not found")
    from fleetopt.problem import build_problem_from_sumo
    return build_problem_from_sumo(NET, num_vans=3, num_destinations=3, k_paths=2, seed=42)


def test_scenario_shape(sumo_problem):
    assert len(sumo_problem.routes) == 6
    _, variables = build_qubo(sumo_problem)
    assert len(variables) == 18


def test_all_solvers_feasible_and_exact_agree(sumo_problem):
    p = sumo_problem
    Q, variables = build_qubo(p)
    _, opt = enumerate_feasible(p)

    x_bf, e_bf = brute_force_qubo(Q, variables)
    assert is_feasible(decode(x_bf), p)
    assert e_bf + qubo_offset(p) == pytest.approx(opt)

    pytest.importorskip("pulp")
    from fleetopt.exact import milp
    _, milp_val = milp(p)
    assert milp_val == pytest.approx(opt, rel=1e-9, abs=1e-12)  # near-tied paths: needs gap=0

    g = greedy(p)
    assert is_feasible(g, p) and objective(g, p) >= opt - 1e-9

    x_sa, _ = simulated_annealing(Q, variables, num_reads=10, num_sweeps=200, seed=0)
    assert is_feasible(decode(x_sa), p)
    assert objective(decode(x_sa), p) == pytest.approx(opt)
