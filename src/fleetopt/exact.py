"""Exact classical solvers: guaranteed optimal, used as ground truth.

Three independent methods, so each one can check the others:

  solver               searches                       practical limit
  -------------------  -----------------------------  -----------------------------------------
  brute_force_qubo     all 2^n bitstrings of the QUBO  ~24 variables (memory/time double per var)
  enumerate_feasible   only valid plans, no QUBO       ~50k plans (grows factorially with vans)
  milp                 MILP solver (branch and bound)  well beyond the instance sizes used here

brute_force_qubo checks the *QUBO*; enumerate_feasible and milp check the *problem*. If they
agree, the QUBO is a faithful encoding.
"""

from __future__ import annotations

import itertools

import numpy as np

from .evaluate import objective, qubo_matrix
from .problem import Problem

MAX_BRUTE_FORCE_VARS = 24  # 2^24 = 16.7M states: a few seconds; every extra var doubles it


# ---------------------------------------------------------------------------------------------
# QUBO brute force
# ---------------------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------------------
# Enumeration of valid plans
# ---------------------------------------------------------------------------------------------

def enumerate_feasible(problem: Problem):
    """Score every valid plan directly with `objective`. Returns (chosen_set, value).

    A valid plan = an ordered choice of distinct vans for the destinations (permutations)
    x one path per destination (product). Count: P(|V|, |D|) * prod_d k_d.
    """
    dests = list(problem.paths)
    best_set, best_val = None, np.inf
    for van_order in itertools.permutations(problem.vans, len(dests)):
        for path_choice in itertools.product(*(problem.paths[d] for d in dests)):
            chosen = set(zip(van_order, dests, path_choice))
            val = objective(chosen, problem)
            if val < best_val:
                best_set, best_val = chosen, val
    return best_set, best_val


# ---------------------------------------------------------------------------------------------
# Mixed-integer linear program
# ---------------------------------------------------------------------------------------------

def _milp_solver(pulp, msg: bool, time_limit: float | None):
    """First available open-source MILP backend: HiGHS (pip install highspy), then CBC.

    gapRel/gapAbs = 0 forces a proven optimum. The HiGHS default relative gap (1e-4) returned a
    sub-optimal plan on the SUMO scenario, whose two D3 paths differ by less than that.
    """
    for factory in (getattr(pulp, "HiGHS", None), getattr(pulp, "PULP_CBC_CMD", None)):
        if factory is None:
            continue
        solver = factory(msg=msg, timeLimit=time_limit, gapRel=0, gapAbs=0)
        if solver.available():
            return solver
    raise RuntimeError("No MILP solver available: pip install highspy")


def milp(problem: Problem, time_limit: float | None = 60, msg: bool = False):
    """Exact MILP via PuLP (HiGHS, falling back to CBC). Returns (chosen_set, value).

    Variables:
        y[v,d,p] in {0,1}  van v drives path p to destination d
        r[d,p]   = sum_v y[v,d,p]          route used (an expression, binary by the constraints)
        z[a,b]   >= 0                      "routes a and b both used", replaces the product r_a*r_b

    Model:
        min  sum_a C_norm[a] r[a] + sum_{a,b} congestion(a,b) z[a,b]
        s.t. sum_{v,p} y[v,d,p] == 1     for each destination d   (served exactly once)
             sum_{d,p} y[v,d,p] <= 1     for each van v           (at most one destination)
             z[a,b] >= r[a] + r[b] - 1   for each congested pair  (z = 1 when both are used)

    The z trick is exact because congestion(a,b) >= 0: minimisation pushes z down to
    max(0, r[a] + r[b] - 1), which equals r[a]*r[b] for binary r. This is why lam_cong < 0 is
    rejected. Unlike the QUBO, constraints are enforced directly, so no penalty weights.
    """
    import pulp

    if problem.lam_cong < 0:
        raise ValueError("MILP linearisation assumes lam_cong >= 0.")

    model = pulp.LpProblem("fleet_assignment", pulp.LpMinimize)

    def new_var(name, **kw):
        """PuLP >= 4 creates variables on the model; older versions use LpVariable directly."""
        if hasattr(model, "add_variable"):
            return model.add_variable(name, **kw)
        return pulp.LpVariable(name, **kw)

    keys = [(v, d, p) for v in problem.vans for d, ps in problem.paths.items() for p in ps]
    y = {k: new_var(f"y_{i}", cat="Binary") for i, k in enumerate(keys)}
    r = {route: pulp.lpSum(y[(v, *route)] for v in problem.vans) for route in problem.routes}

    # Only pairs with positive congestion need a z variable
    pairs = [(a, b) for a, b in itertools.combinations(problem.routes, 2)
             if problem.congestion(a, b) > 0]
    z = {ab: new_var(f"z_{i}", lowBound=0) for i, ab in enumerate(pairs)}

    # Objective
    model += (pulp.lpSum(problem.C_norm[a] * r[a] for a in problem.routes)
              + pulp.lpSum(problem.congestion(a, b) * z[a, b] for a, b in pairs))
    # Constraints
    for d, ps in problem.paths.items():
        model += pulp.lpSum(y[(v, d, p)] for v in problem.vans for p in ps) == 1
    for v in problem.vans:
        model += pulp.lpSum(y[(v, d, p)] for d, ps in problem.paths.items() for p in ps) <= 1
    for a, b in pairs:
        model += z[a, b] >= r[a] + r[b] - 1

    result = model.solve(_milp_solver(pulp, msg, time_limit))
    # PuLP >= 4 returns a stats object with .status; older versions return an int status code
    status = getattr(result, "status", result)
    if hasattr(pulp, "LpSolveStatus"):
        optimal = status == pulp.LpSolveStatus.Optimal
    else:
        optimal = pulp.LpStatus[status] == "Optimal"
    if not optimal:
        raise RuntimeError(f"MILP not solved to optimality: {status}")

    chosen = {k for k, var in y.items() if var.value() > 0.5}  # > 0.5 guards float round-off
    return chosen, objective(chosen, problem)
