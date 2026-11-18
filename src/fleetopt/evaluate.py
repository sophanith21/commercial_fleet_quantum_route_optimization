"""Scoring and checking solutions.

Two independent ways to score a plan are provided on purpose:

  * qubo_energy(Q, x)        - what the QUBO says (penalties included, constant offset dropped)
  * objective(chosen, prob)  - the real cost, computed straight from the problem data

For a feasible plan they must agree (up to qubo_offset). The tests rely on that agreement to
catch mistakes in the QUBO construction, so `objective` must never be written in terms of Q.

Two solution formats are used:
  x       dict {var: 0 or 1} over *all* variables  (what QUBO solvers return)
  chosen  set {(v, d, p), ...} of selected variables (what humans and `objective` read)
`decode` / `encode` convert between them.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from .problem import Problem


def qubo_energy(Q: dict, x: dict) -> float:
    """E(x) = sum Q_ij x_i x_j  (offset not included)."""
    return sum(c * x[a] * x[b] for (a, b), c in Q.items())


def qubo_matrix(Q: dict, variables: list) -> np.ndarray:
    """Dense upper-triangular matrix M with  E(x) = x^T M x,  rows/cols in `variables` order."""
    idx = {v: i for i, v in enumerate(variables)}
    M = np.zeros((len(variables), len(variables)))
    for (a, b), c in Q.items():
        i, j = idx[a], idx[b]
        M[min(i, j), max(i, j)] += c
    return M


def decode(x: dict) -> set:
    """Bit dict -> set of selected variables."""
    return {var for var, bit in x.items() if bit}


def encode(chosen, variables: list) -> dict:
    """Set of selected variables -> bit dict over all `variables`."""
    chosen = set(chosen)
    return {v: int(v in chosen) for v in variables}


def is_feasible(chosen, problem: Problem) -> bool:
    """True iff every destination is served exactly once and no van is used twice."""
    dest_counts = Counter(d for _, d, _ in chosen)
    van_counts = Counter(v for v, _, _ in chosen)
    every_dest_served = set(dest_counts) == set(problem.paths)
    each_dest_once = all(n == 1 for n in dest_counts.values())
    each_van_at_most_once = all(n == 1 for n in van_counts.values())
    return every_dest_served and each_dest_once and each_van_at_most_once


def objective(chosen, problem: Problem) -> float:
    """Real cost of a plan: total normalised travel time + congestion between every route pair.

    Computed only from the Problem (never from Q). Meaningful for feasible plans; for infeasible
    ones it simply scores whatever routes are selected.
    """
    routes = [(d, p) for _, d, p in chosen]
    travel = sum(problem.C_norm[r] for r in routes)
    congestion = sum(problem.congestion(a, b) for a, b in combinations(routes, 2))
    return travel + congestion
