"""End-to-end check on the real SUMO scenario (marked slow; skipped without sumolib).

Builds the same instance as preprocessing.ipynb (seed 42, 3 vans, 3 destinations, 2 paths) and
checks that every solver returns a feasible plan with the exact optimum. This instance matters
because its two D3 paths are near-ties, which the random instances do not exercise.
"""

import os

import pytest

from conftest import requires_milp
from fleetopt.evaluate import decode, is_feasible, objective
from fleetopt.exact import brute_force_qubo, enumerate_feasible, milp
from fleetopt.heuristic import greedy, simulated_annealing
from fleetopt.qubo import build_qubo, qubo_offset

NET = os.path.join(os.path.dirname(__file__), "..", "SUMO_GEN_SCENARIO", "osm.net.xml.gz")

pytestmark = pytest.mark.slow  # applies to every test in this file


@pytest.fixture(scope="module")
def sumo_problem():
    """Loaded once for the whole module: reading the network takes about a second."""
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

    g = greedy(p)
    assert is_feasible(g, p) and objective(g, p) >= opt - 1e-9

    x_sa, _ = simulated_annealing(Q, variables, num_reads=10, num_sweeps=200, seed=0)
    assert is_feasible(decode(x_sa), p)
    assert objective(decode(x_sa), p) == pytest.approx(opt)


@requires_milp
def test_milp_resolves_near_tied_paths(sumo_problem):
    """Regression: with the solver's default MIP gap (1e-4) the MILP returned 1.799103 instead of
    the optimum 1.799024. Exact agreement needs gapRel = gapAbs = 0 (see exact._milp_solver)."""
    _, opt = enumerate_feasible(sumo_problem)
    _, milp_val = milp(sumo_problem)
    assert milp_val == pytest.approx(opt, rel=1e-9, abs=1e-12)
