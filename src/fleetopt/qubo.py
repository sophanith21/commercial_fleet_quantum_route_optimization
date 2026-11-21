from __future__ import annotations

from itertools import combinations

from .problem import Problem

# A = B = DEFAULT_PENALTY_FACTOR * A_min. Chosen by scripts/penalty_sweep.py: it gave the best
# simulated-annealing success rate at every problem size tried (the old fixed A = B = 10 was
# 6-14x larger than needed and found the optimum far less often).
DEFAULT_PENALTY_FACTOR = 1.1


# ---------------------------------------------------------------------------------------------
# Penalty weights
# ---------------------------------------------------------------------------------------------

def penalty_bounds(problem: Problem) -> tuple[float, float]:

    if len(problem.vans) < len(problem.paths):
        raise ValueError("Penalty bound assumes at least as many vans as destinations.")
    if problem.lam_cong < 0 or min(problem.C_norm.values()) < 0:
        raise ValueError("Penalty bound assumes non-negative costs and lam_cong.")

    def worst_add_cost(d, p):
        """Most that adding route (d, p) can raise the cost, whatever else is selected."""
        worst_congestion = sum(
            max(problem.congestion((d, p), (d2, p2)) for p2 in ps2)
            for d2, ps2 in problem.paths.items() if d2 != d
        )
        return problem.C_norm[(d, p)] + worst_congestion

    A_min = max(min(worst_add_cost(d, p) for p in ps) for d, ps in problem.paths.items())
    return A_min, 0.0


def recommended_penalties(problem: Problem, factor: float = DEFAULT_PENALTY_FACTOR,
                          B: float | None = None) -> tuple[float, float]:
    """Return (A, B) with A = factor * A_min. B defaults to A (best in the penalty sweep)."""
    if factor <= 1:
        raise ValueError("factor must be > 1 (A must strictly exceed A_min).")
    A_min, _ = penalty_bounds(problem)
    A = factor * A_min
    return A, (A if B is None else B)


def qubo_offset(problem: Problem, A: float | None = None) -> float:
    """The constant dropped from H_A: A per destination.

    For any feasible x:  qubo_energy(Q, x) + qubo_offset(problem, A) == objective(x).
    Pass the same A used in build_qubo; A=None uses the same default as build_qubo.
    """
    if A is None:
        A, _ = recommended_penalties(problem)
    return A * len(problem.paths)


# ---------------------------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------------------------

def build_qubo(problem: Problem, A: float | None = None, B: float | None = None,
               verbose: bool = False):
    if A is None or B is None:
        A_default, B_default = recommended_penalties(problem)
        A = A_default if A is None else A
        B = B_default if B is None else B

    vans, paths = problem.vans, problem.paths
    variables = [(v, d, p) for v in vans for d, ps in paths.items() for p in ps]
    index = {var: i for i, var in enumerate(variables)}
    Q: dict = {}

    def add(a, b, coefficient):
        """Accumulate into Q, always storing the pair in (lower index, higher index) order."""
        if index[a] > index[b]:
            a, b = b, a
        Q[(a, b)] = Q.get((a, b), 0.0) + coefficient

    # H_cost: linear travel cost. It does not depend on the van.
    for v, d, p in variables:
        add((v, d, p), (v, d, p), problem.C_norm[(d, p)])

    # H_cong: quadratic congestion between routes to different destinations. It applies to every
    # van pair (v1, v2): congestion depends on the roads used, not on which van drives them.
    for route_a, route_b in combinations(problem.routes, 2):
        c = problem.congestion(route_a, route_b)
        if c == 0:
            continue
        for v1 in vans:
            for v2 in vans:
                add((v1, *route_a), (v2, *route_b), c)

    # H_A: each destination exactly once  ->  -A on the diagonal, +2A between every pair of
    # variables that serve the same destination (any van, any path).
    for d in paths:
        dest_vars = [(v, d, p) for v in vans for p in paths[d]]
        for var in dest_vars:
            add(var, var, -A)
        for a, b in combinations(dest_vars, 2):
            add(a, b, 2 * A)

    # H_B: each van at most once  ->  +B between every pair of variables of the same van.
    # ("At most" rather than "exactly" because there may be more vans than destinations,
    # so no linear term is needed.)
    for v in vans:
        van_vars = [(v, d, p) for d in paths for p in paths[d]]
        for a, b in combinations(van_vars, 2):
            add(a, b, B)

    if verbose:
        _print_qubo(problem, Q, variables, A, B)
    return Q, variables


def _print_qubo(problem: Problem, Q: dict, variables: list, A: float, B: float) -> None:
    """Human-readable dump, with variables named x{van}_D{i}_P{path+1} as in the notebook."""
    labels = {d: f"D{i + 1}" for i, d in enumerate(problem.paths)}
    name = {var: f"x{var[0]}_{labels[var[1]]}_P{var[2] + 1}" for var in variables}
    print("========== QUBO MODEL ==========")
    print(f"Vans: {len(problem.vans)}  Destinations: {len(problem.paths)}  Variables: {len(variables)}")
    print(f"lam_cong={problem.lam_cong}  A={A:.4f}  B={B:.4f}  Coefficients: {len(Q)}")
    for (a, b), c in Q.items():
        print(f"Q[{name[a]}, {name[b]}] = {c:.4f}")
