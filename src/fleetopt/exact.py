"""Exact QUBO solver: scores every bit string, so its answer is the true QUBO minimum.

Used as ground truth for the two-depot QUBO at 3 stops (16 variables, 65,536 states).
"""

from __future__ import annotations

import numpy as np

from .evaluate import qubo_matrix

MAX_BRUTE_FORCE_VARS = 24  # 2^24 = 16.7M states: a few seconds; every extra var doubles it


def brute_force_qubo(Q: dict, variables: list, chunk: int = 1 << 16):
    """Minimise x^T M x over every bitstring. Returns (x_dict, energy) without the offset.

    Integer k in [0, 2^n) encodes the state: bit i of k is the value of variables[i]. States are
    scored `chunk` at a time with numpy so memory stays bounded (chunk x n floats).
    """
    n = len(variables)
    if n > MAX_BRUTE_FORCE_VARS:
        raise ValueError(f"{n} variables is too many for brute force (max {MAX_BRUTE_FORCE_VARS}).")

    M = qubo_matrix(Q, variables)
    bit_positions = np.arange(n)
    best_energy, best_k = np.inf, 0

    for start in range(0, 1 << n, chunk):
        ks = np.arange(start, min(start + chunk, 1 << n))
        X = ((ks[:, None] >> bit_positions) & 1).astype(float)  # one row per state
        energies = np.einsum("ij,ij->i", X @ M, X)               # row-wise x^T M x
        i = int(energies.argmin())
        if energies[i] < best_energy:
            best_energy, best_k = float(energies[i]), int(ks[i])

    x = {v: (best_k >> i) & 1 for i, v in enumerate(variables)}
    return x, best_energy
