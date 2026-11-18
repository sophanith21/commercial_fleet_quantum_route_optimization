"""Heuristic classical solvers."""

from __future__ import annotations

import numpy as np

from .evaluate import qubo_matrix
from .problem import Problem


def greedy(problem: Problem) -> set:
    """Constructive heuristic, always feasible.

    Destinations are handled from most to least expensive; each gets a free van and the
    path with the lowest travel cost plus congestion against routes already chosen.
    """
    order = sorted(problem.paths, key=lambda d: min(problem.C_norm[(d, p)] for p in problem.paths[d]),
                   reverse=True)
    free_vans = list(problem.vans)
    chosen, used_routes = set(), []
    for d in order:
        if not free_vans:
            raise ValueError("More destinations than vans.")
        best_p = min(
            problem.paths[d],
            key=lambda p: problem.C_norm[(d, p)]
            + sum(problem.congestion((d, p), r) for r in used_routes),
        )
        chosen.add((free_vans.pop(0), d, best_p))
        used_routes.append((d, best_p))
    return chosen


def simulated_annealing(
    Q: dict,
    variables: list,
    num_reads: int = 50,
    num_sweeps: int = 500,
    beta_range: tuple = (0.1, 10.0),
    seed: int | None = None,
):
    """Single-bit-flip simulated annealing on the QUBO. Returns (x_dict, energy) without offset."""
    rng = np.random.default_rng(seed)
    M = qubo_matrix(Q, variables)
    h = np.diag(M).copy()            # linear terms
    J = M + M.T
    np.fill_diagonal(J, 0.0)         # symmetric couplings
    n = len(variables)
    betas = np.geomspace(beta_range[0], beta_range[1], num_sweeps)

    best_x, best_e = None, np.inf
    for _ in range(num_reads):
        x = rng.integers(0, 2, n).astype(float)
        field = J @ x
        e = float(x @ M @ x)
        for beta in betas:
            for i in rng.permutation(n):
                delta = (1 - 2 * x[i]) * (h[i] + field[i])
                if delta <= 0 or rng.random() < np.exp(-beta * delta):
                    step = 1 - 2 * x[i]
                    x[i] += step
                    field += step * J[:, i]
                    e += delta
            if e < best_e - 1e-12:
                best_e, best_x = e, x.copy()
    return {v: int(best_x[i]) for i, v in enumerate(variables)}, float(best_e)


def tabu(Q: dict, variables: list, num_reads: int = 10, seed: int | None = None, timeout_ms: int = 100):
    """Tabu search via dwave-samplers (optional dependency). Returns (x_dict, energy)."""
    from dwave.samplers import TabuSampler

    sampleset = TabuSampler().sample_qubo(Q, num_reads=num_reads, seed=seed, timeout=timeout_ms)
    best = sampleset.first
    return {v: int(best.sample[v]) for v in variables}, float(best.energy)
