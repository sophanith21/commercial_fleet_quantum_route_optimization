"""QUBO helper shared by the solvers."""

from __future__ import annotations

import numpy as np


def qubo_matrix(Q: dict, variables: list) -> np.ndarray:
    """Dense upper-triangular matrix M with  E(x) = x^T M x,  rows/cols in `variables` order."""
    idx = {v: i for i, v in enumerate(variables)}
    M = np.zeros((len(variables), len(variables)))
    for (a, b), c in Q.items():
        i, j = idx[a], idx[b]
        M[min(i, j), max(i, j)] += c
    return M
