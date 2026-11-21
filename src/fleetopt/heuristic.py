from __future__ import annotations

import numpy as np

from .evaluate import qubo_matrix
from .problem import Problem


def greedy(problem: Problem) -> set:

    def cheapest(d):
        return min(problem.C_norm[(d, p)] for p in problem.paths[d])

    order = sorted(problem.paths, key=cheapest, reverse=True)
    free_vans = list(problem.vans)
    chosen, used_routes = set(), []

    for d in order:
        if not free_vans:
            raise ValueError("More destinations than vans.")

        def added_cost(p):
            return problem.C_norm[(d, p)] + sum(problem.congestion((d, p), r) for r in used_routes)

        best_p = min(problem.paths[d], key=added_cost)
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

    rng = np.random.default_rng(seed)
    M = qubo_matrix(Q, variables)
    h = np.diag(M).copy()            # linear terms
    J = M + M.T                      # symmetric couplings
    np.fill_diagonal(J, 0.0)         # without the diagonal
    n = len(variables)
    betas = np.geomspace(beta_range[0], beta_range[1], num_sweeps)

    best_x, best_energy = None, np.inf
    for _ in range(num_reads):
        x = rng.integers(0, 2, n).astype(float)
        local_field = J @ x
        energy = float(x @ M @ x)

        for beta in betas:
            for i in rng.permutation(n):
                direction = 1 - 2 * x[i]                   # +1 if flipping 0->1, -1 if 1->0
                delta = direction * (h[i] + local_field[i])
                if delta <= 0 or rng.random() < np.exp(-beta * delta):
                    x[i] += direction
                    local_field += direction * J[:, i]
                    energy += delta
            # Track the best state at the end of each sweep (tolerance avoids float churn)
            if energy < best_energy - 1e-12:
                best_energy, best_x = energy, x.copy()

    return {v: int(best_x[i]) for i, v in enumerate(variables)}, float(best_energy)


def tabu(Q: dict, variables: list, num_reads: int = 10, seed: int | None = None,
         timeout_ms: int = 100):
    from dwave.samplers import TabuSampler

    sampleset = TabuSampler().sample_qubo(Q, num_reads=num_reads, seed=seed, timeout=timeout_ms)
    best = sampleset.first  # lowest-energy sample
    return {v: int(best.sample[v]) for v in variables}, float(best.energy)
