from dataclasses import replace

import pytest

from fleetopt.evaluate import decode, is_feasible
from fleetopt.exact import brute_force_qubo, enumerate_feasible
from fleetopt.qubo import build_qubo, penalty_bounds, qubo_offset, recommended_penalties


def ground_state(problem, A, B):
    """Exact QUBO minimiser for the given penalties: (chosen set, energy + offset)."""
    Q, variables = build_qubo(problem, A=A, B=B)
    x, e = brute_force_qubo(Q, variables)
    return decode(x), e + qubo_offset(problem, A)


def test_tiny_bound_by_hand(tiny):
    # d=A: min(0.5+0.2, 0.6+0.2) = 0.70 ; d=B: min(0.7+0.2, 0.75+0.0) = 0.75  -> A_min = 0.75
    A_min, B_min = penalty_bounds(tiny)
    assert A_min == pytest.approx(0.75)
    assert B_min == 0.0


def test_tiny_bound_is_tight(tiny):
    """Just below A_min, serving only route A0 (0.5 + A) beats the optimum (1.25)."""
    A_min, _ = penalty_bounds(tiny)
    below, _ = ground_state(tiny, A=A_min - 0.01, B=1)
    above, val = ground_state(tiny, A=A_min + 0.01, B=1)
    assert not is_feasible(below, tiny)
    assert is_feasible(above, tiny)
    assert val == pytest.approx(1.25)


@pytest.mark.parametrize("seed", range(15))
def test_ground_state_feasible_just_above_bound(seed, random_problem):
    """A = 1.01 * A_min and a tiny B must still give the true optimum."""
    p = random_problem(seed)
    A_min, _ = penalty_bounds(p)
    chosen, val = ground_state(p, A=1.01 * A_min, B=0.01)
    assert is_feasible(chosen, p)
    assert val == pytest.approx(enumerate_feasible(p)[1])


def test_more_vans_than_destinations_bound(more_vans_than_dests):
    p = more_vans_than_dests
    A_min, _ = penalty_bounds(p)
    chosen, val = ground_state(p, A=1.01 * A_min, B=0.01)
    assert is_feasible(chosen, p)
    assert val == pytest.approx(enumerate_feasible(p)[1])


def test_bound_far_below_old_default(random_problem):
    """Documents how loose the old A = B = 10 was: A_min <= 1 + lam * (|D| - 1)."""
    p = random_problem(0)
    A_min, _ = penalty_bounds(p)
    assert A_min <= 1 + p.lam_cong * (len(p.paths) - 1) + 1e-12
    assert A_min < 10


def test_zero_congestion_bound_is_max_min_cost(random_problem):
    p = replace(random_problem(2), lam_cong=0.0)
    A_min, _ = penalty_bounds(p)
    assert A_min == pytest.approx(max(min(p.C_norm[(d, q)] for q in ps) for d, ps in p.paths.items()))


def test_recommended_penalties(random_problem):
    p = random_problem(0)
    A_min, _ = penalty_bounds(p)
    assert recommended_penalties(p, factor=2) == pytest.approx((2 * A_min, 2 * A_min))
    assert recommended_penalties(p, factor=2, B=0.5)[1] == 0.5
    with pytest.raises(ValueError):
        recommended_penalties(p, factor=1.0)


def test_bound_rejects_too_few_vans(random_problem):
    p = random_problem(0)
    with pytest.raises(ValueError):
        penalty_bounds(replace(p, vans=p.vans[:1]))
