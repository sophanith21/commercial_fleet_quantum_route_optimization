"""Solution evaluation, independent of the QUBO so the two can be cross-checked."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from .problem import Problem


def qubo_energy(Q: dict, x: dict) -> float:
    """Energy of a 0/1 assignment x (var -> bit) under Q (offset not included)."""
    return sum(c * x[a] * x[b] for (a, b), c in Q.items())


def qubo_matrix(Q: dict, variables: list) -> np.ndarray:
    """Upper-triangular matrix M with E(x) = x^T M x."""
    idx = {v: i for i, v in enumerate(variables)}
    M = np.zeros((len(variables), len(variables)))
    for (a, b), c in Q.items():
        i, j = idx[a], idx[b]
        M[min(i, j), max(i, j)] += c
    return M


def decode(x: dict) -> set:
    """Selected variables {(van, dest, path), ...}."""
    return {var for var, bit in x.items() if bit}


def encode(chosen, variables: list) -> dict:
    chosen = set(chosen)
    return {v: int(v in chosen) for v in variables}


def is_feasible(chosen, problem: Problem) -> bool:
    """Each destination served exactly once; each van used at most once."""
    dest_counts = Counter(d for _, d, _ in chosen)
    van_counts = Counter(v for v, _, _ in chosen)
    return (
        set(dest_counts) == set(problem.paths)
        and all(n == 1 for n in dest_counts.values())
        and all(n == 1 for n in van_counts.values())
    )


def objective(chosen, problem: Problem) -> float:
    """True cost (travel + congestion), computed directly from the problem, not from Q."""
    routes = [(d, p) for _, d, p in chosen]
    cost = sum(problem.C_norm[r] for r in routes)
    cost += sum(problem.congestion(a, b) for a, b in combinations(routes, 2))
    return cost
