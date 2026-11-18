"""Exact classical solvers (ground truth for small instances)."""

from __future__ import annotations

import itertools

import numpy as np

from .evaluate import objective, qubo_matrix
from .problem import Problem

MAX_BRUTE_FORCE_VARS = 24


def brute_force_qubo(Q: dict, variables: list, chunk: int = 1 << 16):
    """Minimise x^T M x over all 2^n bitstrings. Returns (x_dict, energy) without offset."""
    n = len(variables)
    if n > MAX_BRUTE_FORCE_VARS:
        raise ValueError(f"{n} variables is too many for brute force (max {MAX_BRUTE_FORCE_VARS}).")
    M = qubo_matrix(Q, variables)
    bits = np.arange(n)
    best_e, best_k = np.inf, 0
    for start in range(0, 1 << n, chunk):
        ks = np.arange(start, min(start + chunk, 1 << n))
        X = ((ks[:, None] >> bits) & 1).astype(float)
        E = np.einsum("ij,ij->i", X @ M, X)
        i = int(E.argmin())
        if E[i] < best_e:
            best_e, best_k = float(E[i]), int(ks[i])
    x = {v: (best_k >> i) & 1 for i, v in enumerate(variables)}
    return x, best_e


def _milp_solver(pulp, msg, time_limit):
    """First available open-source MILP backend: HiGHS (pip install highspy), then CBC.

    gapRel/gapAbs = 0: the default MIP gap (1e-4 in HiGHS) is loose enough to return a
    sub-optimal plan when candidate paths are near-ties (e.g. the SUMO D3 paths).
    """
    for factory in (getattr(pulp, "HiGHS", None), getattr(pulp, "PULP_CBC_CMD", None)):
        if factory is None:
            continue
        solver = factory(msg=msg, timeLimit=time_limit, gapRel=0, gapAbs=0)
        if solver.available():
            return solver
    raise RuntimeError("No MILP solver available: pip install highspy")


def milp(problem: Problem, time_limit: float | None = 60, msg: bool = False):
    """Exact MILP (PuLP + CBC) for instances too large to enumerate. Returns (chosen_set, value).

    y[v,d,p] = 1  if van v drives path p to destination d
    r[d,p]   = sum_v y[v,d,p]             (route used, implied binary)
    z[a,b]  >= r[a] + r[b] - 1            (both routes used; linearises r[a]*r[b])

    min  sum C_norm[a] r[a] + sum congestion(a,b) z[a,b]
    s.t. sum_{v,p} y[v,d,p] == 1  for each d   (destination served exactly once)
         sum_{d,p} y[v,d,p] <= 1  for each v   (van serves at most one destination)

    The z linearisation is exact because every congestion coefficient is >= 0, so the
    minimiser pushes z down to max(0, r[a] + r[b] - 1).
    """
    import pulp

    if problem.lam_cong < 0:
        raise ValueError("MILP linearisation assumes lam_cong >= 0.")

    model = pulp.LpProblem("fleet_assignment", pulp.LpMinimize)

    def new_var(name, **kw):          # PuLP >= 4 creates variables on the model
        if hasattr(model, "add_variable"):
            return model.add_variable(name, **kw)
        return pulp.LpVariable(name, **kw)

    keys = [(v, d, p) for v in problem.vans for d, ps in problem.paths.items() for p in ps]
    y = {k: new_var(f"y_{i}", cat="Binary") for i, k in enumerate(keys)}
    r = {a: pulp.lpSum(y[(v, *a)] for v in problem.vans) for a in problem.routes}

    pairs = [(a, b) for a, b in itertools.combinations(problem.routes, 2)
             if problem.congestion(a, b) > 0]
    z = {ab: new_var(f"z_{i}", lowBound=0) for i, ab in enumerate(pairs)}

    model += (pulp.lpSum(problem.C_norm[a] * r[a] for a in problem.routes)
              + pulp.lpSum(problem.congestion(a, b) * z[a, b] for a, b in pairs))
    for d, ps in problem.paths.items():
        model += pulp.lpSum(y[(v, d, p)] for v in problem.vans for p in ps) == 1
    for v in problem.vans:
        model += pulp.lpSum(y[(v, d, p)] for d, ps in problem.paths.items() for p in ps) <= 1
    for a, b in pairs:
        model += z[a, b] >= r[a] + r[b] - 1

    result = model.solve(_milp_solver(pulp, msg, time_limit))
    status = getattr(result, "status", result)          # PuLP 4: stats object; older: int code
    optimal = (status == pulp.LpSolveStatus.Optimal if hasattr(pulp, "LpSolveStatus")
               else pulp.LpStatus[status] == "Optimal")
    if not optimal:
        raise RuntimeError(f"MILP not solved to optimality: {status}")

    chosen = {k for k, var in y.items() if var.value() > 0.5}
    return chosen, objective(chosen, problem)


def enumerate_feasible(problem: Problem):
    """Enumerate only valid assignments (no QUBO). Returns (chosen_set, objective_value)."""
    dests = list(problem.paths)
    best_set, best_val = None, np.inf
    for van_order in itertools.permutations(problem.vans, len(dests)):
        for ps in itertools.product(*(problem.paths[d] for d in dests)):
            chosen = set(zip(van_order, dests, ps))
            val = objective(chosen, problem)
            if val < best_val:
                best_set, best_val = chosen, val
    return best_set, best_val
