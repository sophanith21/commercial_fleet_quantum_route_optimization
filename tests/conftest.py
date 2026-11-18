"""Shared fixtures and helpers for the test suite.

Fixtures:
    tiny                  hand-checkable 2x2x2 instance with a known optimum (TINY_OPT_*)
    more_vans_than_dests  3 vans / 2 destinations: one van must stay idle
    random_problem        factory random_problem(seed, **kwargs) -> Problem (18 vars by default)

Markers:
    requires_milp         skip a test unless PuLP and a MILP backend (HiGHS or CBC) are installed
"""

import pytest

from fleetopt.problem import Problem, make_random_problem


def _milp_available() -> bool:
    try:
        import pulp
        from fleetopt.exact import _milp_solver
        _milp_solver(pulp, msg=False, time_limit=None)
        return True
    except (ImportError, RuntimeError):
        return False


requires_milp = pytest.mark.skipif(not _milp_available(),
                                   reason="needs pulp + a MILP backend (pip install pulp highspy)")


@pytest.fixture
def tiny():
    """2 vans, 2 destinations, 2 paths each; optimum worked out by hand.

    Cross-destination max overlap = 10, lam = 0.2:
        (A0, B0): 0.5 + 0.70 + 0.2 = 1.40
        (A0, B1): 0.5 + 0.75 + 0.0 = 1.25   <- optimum
        (A1, B0): 0.6 + 0.70 + 0.2 = 1.50
        (A1, B1): 0.6 + 0.75 + 0.0 = 1.35
    """
    return Problem(
        vans=[1, 2],
        paths={"A": [0, 1], "B": [0, 1]},
        C_norm={("A", 0): 0.5, ("A", 1): 0.6, ("B", 0): 0.7, ("B", 1): 0.75},
        S={
            (("A", 0), ("B", 0)): 10, (("A", 0), ("B", 1)): 0,
            (("A", 1), ("B", 0)): 10, (("A", 1), ("B", 1)): 0,
            (("A", 0), ("A", 1)): 5, (("B", 0), ("B", 1)): 5,
        },
        lam_cong=0.2,
    )


TINY_OPT_VALUE = 1.25
TINY_OPT_ROUTES = {("A", 0), ("B", 1)}


@pytest.fixture
def more_vans_than_dests():
    """3 vans, 2 destinations: one van must stay idle."""
    return make_random_problem(seed=7, num_vans=3, num_destinations=2, k_paths=2)


@pytest.fixture
def random_problem():
    """Factory: random_problem(seed, **kwargs) -> Problem (18 variables by default)."""
    return make_random_problem
