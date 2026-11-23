from dataclasses import replace

import pytest

from conftest import TINY_OPT_ROUTES, TINY_OPT_VALUE, requires_milp
from fleetopt.evaluate import decode, is_feasible, objective
from fleetopt.exact import brute_force_qubo, enumerate_feasible, milp
from fleetopt.heuristic import greedy, simulated_annealing
from fleetopt.qubo import build_qubo, qubo_offset


def routes_of(chosen):
    """Drop the van from each selected variable: {(d, p), ...}."""
    return {(d, p) for _, d, p in chosen}


# ---------- brute force and enumeration ----------


def test_enumerate_feasible_tiny(tiny):
    chosen, val = enumerate_feasible(tiny)
    assert val == pytest.approx(TINY_OPT_VALUE)
    assert routes_of(chosen) == TINY_OPT_ROUTES
    assert is_feasible(chosen, tiny)


def test_brute_force_tiny(tiny):
    Q, variables = build_qubo(tiny)
    x, e = brute_force_qubo(Q, variables)
    chosen = decode(x)
    assert is_feasible(chosen, tiny)
    assert routes_of(chosen) == TINY_OPT_ROUTES
    assert e + qubo_offset(tiny) == pytest.approx(TINY_OPT_VALUE)


@pytest.mark.parametrize("seed", range(20))
def test_two_exact_solvers_agree(seed, random_problem):
    """Compare objective values, not assignments: vans are interchangeable, so ties are expected."""
    p = random_problem(seed)
    A = 10
    Q, variables = build_qubo(p, A=A)
    x, e = brute_force_qubo(Q, variables)
    _, opt = enumerate_feasible(p)
    assert e + qubo_offset(p, A) == pytest.approx(opt)
    assert objective(decode(x), p) == pytest.approx(opt)


def test_more_vans_than_destinations(more_vans_than_dests):
    p = more_vans_than_dests
    Q, variables = build_qubo(p)
    x, e = brute_force_qubo(Q, variables)
    chosen = decode(x)
    assert is_feasible(chosen, p)
    assert len({v for v, _, _ in chosen}) == len(p.paths) < len(p.vans)
    assert e + qubo_offset(p) == pytest.approx(enumerate_feasible(p)[1])


def test_brute_force_refuses_large_instances(random_problem):
    p = random_problem(0, num_vans=5, num_destinations=5, k_paths=2)  # 50 variables
    Q, variables = build_qubo(p)
    with pytest.raises(ValueError):
        brute_force_qubo(Q, variables)


def test_matches_dimod_exact_solver(random_problem):
    dimod = pytest.importorskip("dimod")
    p = random_problem(1, num_vans=2, num_destinations=2, k_paths=2)
    Q, variables = build_qubo(p)
    ref = dimod.ExactSolver().sample_qubo(Q).first.energy
    _, e = brute_force_qubo(Q, variables)
    assert e == pytest.approx(ref)


# ---------- MILP (PuLP + HiGHS) ----------

@requires_milp
def test_milp_tiny(tiny):
    chosen, val = milp(tiny)
    assert is_feasible(chosen, tiny)
    assert val == pytest.approx(TINY_OPT_VALUE)
    assert routes_of(chosen) == TINY_OPT_ROUTES


@requires_milp
@pytest.mark.parametrize("seed", range(20))
def test_milp_matches_enumeration(seed, random_problem):
    p = random_problem(seed)
    chosen, val = milp(p)
    assert is_feasible(chosen, p)
    assert val == pytest.approx(objective(chosen, p))
    assert val == pytest.approx(enumerate_feasible(p)[1])


@requires_milp
def test_milp_more_vans_than_destinations(more_vans_than_dests):
    p = more_vans_than_dests
    chosen, val = milp(p)
    assert is_feasible(chosen, p)
    assert val == pytest.approx(enumerate_feasible(p)[1])


@requires_milp
@pytest.mark.parametrize("seed", range(3))
def test_milp_on_instance_too_large_for_brute_force(seed, random_problem):
    """75 variables: brute force refuses; MILP must still be feasible and a lower bound on heuristics."""
    p = random_problem(seed, num_vans=5, num_destinations=5, k_paths=3)
    Q, variables = build_qubo(p)
    assert len(variables) > 24
    chosen, opt = milp(p)
    assert is_feasible(chosen, p)
    assert objective(greedy(p), p) >= opt - 1e-9
    x, _ = simulated_annealing(Q, variables, num_reads=5, num_sweeps=200, seed=seed)
    sa = decode(x)
    if is_feasible(sa, p):
        assert objective(sa, p) >= opt - 1e-9


@requires_milp
def test_milp_rejects_negative_lambda(tiny):
    with pytest.raises(ValueError):
        milp(replace(tiny, lam_cong=-0.1))
