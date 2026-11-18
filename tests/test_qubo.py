"""The QUBO model itself: structure, and agreement with the independently computed objective.

The most important test is test_energy_matches_objective_on_every_feasible_state: if QUBO energy
plus offset equals `objective()` on every valid plan, the cost and congestion terms are encoded
correctly. The penalty tests then check that invalid plans are pushed above the optimum.
"""

import itertools
from dataclasses import replace

import pytest

from fleetopt.evaluate import decode, encode, is_feasible, objective, qubo_energy
from fleetopt.exact import brute_force_qubo
from fleetopt.qubo import build_qubo, qubo_offset


def all_feasible(problem):
    """Yield every valid plan as a chosen set (same enumeration as exact.enumerate_feasible)."""
    dests = list(problem.paths)
    for vans in itertools.permutations(problem.vans, len(dests)):
        for ps in itertools.product(*(problem.paths[d] for d in dests)):
            yield set(zip(vans, dests, ps))


def test_variable_count(random_problem):
    p = random_problem(0)
    _, variables = build_qubo(p)
    assert len(variables) == len(p.vans) * sum(len(ps) for ps in p.paths.values())
    assert len(set(variables)) == len(variables)


def test_keys_are_upper_triangular(random_problem):
    Q, variables = build_qubo(random_problem(0))
    idx = {v: i for i, v in enumerate(variables)}
    for a, b in Q:
        assert idx[a] <= idx[b]
        if a != b:
            assert (b, a) not in Q


@pytest.mark.parametrize("seed", range(10))
def test_energy_matches_objective_on_every_feasible_state(seed, random_problem):
    """The key check: QUBO energy + offset == directly computed objective."""
    p = random_problem(seed)
    A = 10
    Q, variables = build_qubo(p, A=A, B=10)
    for chosen in all_feasible(p):
        x = encode(chosen, variables)
        assert qubo_energy(Q, x) + qubo_offset(p, A) == pytest.approx(objective(chosen, p))


def test_tiny_energy_by_hand(tiny):
    Q, variables = build_qubo(tiny, A=10, B=10)
    x = encode({(1, "A", 0), (2, "B", 1)}, variables)
    assert qubo_energy(Q, x) + qubo_offset(tiny, 10) == pytest.approx(1.25)


@pytest.mark.parametrize("seed", range(5))
def test_strong_penalties_make_ground_state_feasible(seed, random_problem):
    p = random_problem(seed)
    Q, variables = build_qubo(p, A=10, B=10)
    x, _ = brute_force_qubo(Q, variables)
    assert is_feasible(decode(x), p)


def test_weak_penalties_make_ground_state_infeasible(tiny):
    """If the penalties are near zero the solver should drop constraints: proves they matter."""
    Q, variables = build_qubo(tiny, A=0.01, B=0.01)
    x, _ = brute_force_qubo(Q, variables)
    assert not is_feasible(decode(x), tiny)


def test_infeasible_states_cost_more_than_optimum(tiny):
    A = B = 10
    Q, variables = build_qubo(tiny, A=A, B=B)
    off = qubo_offset(tiny, A)
    best_feasible = min(objective(c, tiny) for c in all_feasible(tiny))
    for bits in itertools.product([0, 1], repeat=len(variables)):
        x = dict(zip(variables, bits))
        if not is_feasible(decode(x), tiny):
            assert qubo_energy(Q, x) + off > best_feasible


def test_zero_lambda_equals_zero_overlap(random_problem):
    p = random_problem(3)
    Q_lam0, _ = build_qubo(replace(p, lam_cong=0.0))
    Q_s0, _ = build_qubo(replace(p, S={k: 0 for k in p.S}))
    assert Q_lam0.keys() == Q_s0.keys()
    for k in Q_lam0:
        assert Q_lam0[k] == pytest.approx(Q_s0[k])
