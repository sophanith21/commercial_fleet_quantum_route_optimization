"""Heuristic classical solvers: fast, but with no optimality guarantee.

  greedy               builds a plan directly from the Problem; always feasible, often sub-optimal
  simulated_annealing  QUBO search (own implementation): the closest classical analogue of
                       quantum annealing, so it is the main baseline for the quantum solver
  tabu                 QUBO local search from dwave-samplers (optional dependency)

The QUBO solvers can return infeasible bitstrings if the search fails; always check the
result with `evaluate.is_feasible`.
"""

from __future__ import annotations

import numpy as np

from .evaluate import qubo_matrix
from .problem import Problem


def greedy(problem: Problem) -> set:
    """Constructive heuristic. Returns a feasible chosen set {(v, d, p), ...}.

    1. Order destinations from most to least expensive (by their cheapest path): the long trips
       dominate the cost, so they get first pick of uncongested roads.
    2. For each one, take the path with the lowest  travel cost + congestion with the routes
       already chosen, and assign the next free van (vans are interchangeable).

    Decisions are never revisited, which is why it can miss the optimum.
    """
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
    """Single-bit-flip simulated annealing on the QUBO. Returns (x_dict, energy) without offset.

    Each read starts from a random bitstring and does `num_sweeps` sweeps. A sweep visits every
    bit once in random order and proposes flipping it:
        energy goes down (delta <= 0)  -> always accept
        energy goes up                 -> accept with probability exp(-beta * delta)
    beta (inverse temperature) rises geometrically over the sweeps: early on uphill moves are
    common (exploration), at the end almost none are (settling into a minimum). The best state
    seen across all reads is returned. Same seed -> same result.

    Cheap flip energy: write E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j with h = diag(M) and
    J = M + M^T off the diagonal. Flipping bit i changes E by
        delta_i = (1 - 2 x_i) * (h_i + sum_j J_ij x_j)
    The sum (the "local field") is kept up to date incrementally, so each proposal is O(1)
    and each accepted flip is O(n), instead of recomputing E from scratch.
    """
    rng = np.random.default_rng(seed)
    M = qubo_matrix(Q, variables)
    h = np.diag(M).copy()            # linear terms
    J = M + M.T                      # symmetric couplings...
    np.fill_diagonal(J, 0.0)         # ...without the diagonal
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
    """Tabu search via dwave-samplers. Returns (x_dict, energy) without offset.

    Tabu search is greedy local search that forbids ("tabu") recently flipped bits for a while,
    so it can climb out of local minima without cycling. Each read runs for `timeout_ms`, so
    runtime is roughly num_reads * timeout_ms regardless of problem size.
    """
    from dwave.samplers import TabuSampler

    sampleset = TabuSampler().sample_qubo(Q, num_reads=num_reads, seed=seed, timeout=timeout_ms)
    best = sampleset.first  # lowest-energy sample
    return {v: int(best.sample[v]) for v in variables}, float(best.energy)
