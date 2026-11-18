"""QUBO builder (refactored from preprocessing.ipynb: no globals, optional printing)."""

from __future__ import annotations

from itertools import combinations

from .problem import Problem


DEFAULT_PENALTY_FACTOR = 1.1  # chosen by scripts/penalty_sweep.py: best SA success at every size


def qubo_offset(problem: Problem, A: float | None = None) -> float:
    """Constant dropped from H_A = A * sum_d (sum x - 1)^2.

    For any feasible x:  qubo_energy(Q, x) + qubo_offset(problem, A) == objective(x).
    A=None uses the same default as build_qubo.
    """
    if A is None:
        A, _ = recommended_penalties(problem, DEFAULT_PENALTY_FACTOR)
    return A * len(problem.paths)


def penalty_bounds(problem: Problem) -> tuple[float, float]:
    """Return (A_min, B_min): any A > A_min and B > B_min makes every QUBO ground state feasible.

    Assumes len(vans) >= len(destinations) and non-negative costs / lam_cong. Argument:
      1. Over-serving a destination only adds non-negative cost and A-penalty, so a minimiser
         serves each destination at most once.
      2. With <= |D| <= |V| selected variables, vans can be reassigned to be distinct without
         changing travel or congestion cost (neither depends on the van), removing all B-penalty.
         So any B > 0 rules out van sharing:  B_min = 0 (strict).
      3. Serving a missing destination d with its best path p lowers the A-penalty by A and
         raises cost by at most  C_norm[d,p] + sum_{d' != d} max_{p'} congestion((d,p),(d',p')).
         So A > max_d min_p of that quantity rules out unserved destinations.
    The bound is tight in the worst case (see tests/test_penalties.py::test_tiny_bound_is_tight).
    """
    if len(problem.vans) < len(problem.paths):
        raise ValueError("Penalty bound assumes at least as many vans as destinations.")
    if problem.lam_cong < 0 or min(problem.C_norm.values()) < 0:
        raise ValueError("Penalty bound assumes non-negative costs and lam_cong.")

    def worst_add_cost(d, p):
        return problem.C_norm[(d, p)] + sum(
            max(problem.congestion((d, p), (d2, p2)) for p2 in ps2)
            for d2, ps2 in problem.paths.items() if d2 != d
        )

    A_min = max(min(worst_add_cost(d, p) for p in ps) for d, ps in problem.paths.items())
    return A_min, 0.0


def recommended_penalties(problem: Problem, factor: float = 1.5, B: float | None = None):
    """A = factor * A_min (factor > 1 for a safety margin); B defaults to A."""
    if factor <= 1:
        raise ValueError("factor must be > 1 (A must strictly exceed A_min).")
    A_min, _ = penalty_bounds(problem)
    A = factor * A_min
    return A, (A if B is None else B)


def build_qubo(problem: Problem, A: float | None = None, B: float | None = None, verbose: bool = False):
    """Return (Q, variables). Q maps (var_i, var_j) with index(var_i) <= index(var_j) to a coefficient.

    A/B default to 1.1 * A_min (see penalty_bounds); pass explicit values to override.

    Variables are (van, destination, path). Terms:
      H_cost = sum C_norm[d,p] x[v,d,p]
      H_cong = sum over route pairs of different destinations, all van pairs, of lam * S_norm
      H_A    = A * sum_d (sum_{v,p} x[v,d,p] - 1)^2   (constant dropped)
      H_B    = B * sum_v sum_{pairs} x_i x_j           (each van serves at most one destination)
    """
    if A is None or B is None:
        A_def, B_def = recommended_penalties(problem, DEFAULT_PENALTY_FACTOR)
        A = A_def if A is None else A
        B = B_def if B is None else B

    vans, paths = problem.vans, problem.paths
    variables =[(v, d, p) for v in vans for d, ps in paths.items() for p in ps]
    index = {var: i for i, var in enumerate(variables)}
    Q: dict = {}

    def add(a, b, c):
        if index[a] > index[b]:
            a, b = b, a
        Q[(a, b)] = Q.get((a, b), 0.0) + c

    # Travel cost
    for v, d, p in variables:
        add((v, d, p), (v, d, p), problem.C_norm[(d, p)])

    # Congestion (only between routes of different destinations)
    for ra, rb in combinations(problem.routes, 2):
        c = problem.congestion(ra, rb)
        if c == 0:
            continue
        for v1 in vans:
            for v2 in vans:
                add((v1, *ra), (v2, *rb), c)

    # Each destination exactly once
    for d in paths:
        dvars = [(v, d, p) for v in vans for p in paths[d]]
        for var in dvars:
            add(var, var, -A)
        for a, b in combinations(dvars, 2):
            add(a, b, 2 * A)

    # Each van at most one destination
    for v in vans:
        vvars = [(v, d, p) for d in paths for p in paths[d]]
        for a, b in combinations(vvars, 2):
            add(a, b, B)

    if verbose:
        _print_qubo(problem, Q, variables, A, B)
    return Q, variables


def _print_qubo(problem, Q, variables, A, B):
    labels = {d: f"D{i + 1}" for i, d in enumerate(problem.paths)}
    name = {var: f"x{var[0]}_{labels[var[1]]}_P{var[2] + 1}" for var in variables}
    print("========== QUBO MODEL ==========")
    print(f"Vans: {len(problem.vans)}  Destinations: {len(problem.paths)}  Variables: {len(variables)}")
    print(f"lam_cong={problem.lam_cong}  A={A}  B={B}  Coefficients: {len(Q)}")
    for (a, b), c in Q.items():
        print(f"Q[{name[a]}, {name[b]}] = {c:.4f}")
