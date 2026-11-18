"""Commercial fleet route optimisation: QUBO model plus classical baselines.

Typical use:

    from fleetopt import build_problem_from_sumo, build_qubo, objective
    from fleetopt.exact import milp
    from fleetopt.heuristic import simulated_annealing
    from fleetopt.evaluate import decode

    problem = build_problem_from_sumo()          # or make_random_problem(seed)
    Q, variables = build_qubo(problem)           # penalties default to 1.1 * A_min
    x, _ = simulated_annealing(Q, variables, seed=0)
    print(objective(decode(x), problem), "vs optimum", milp(problem)[1])

Modules:
    problem    Problem dataclass and instance builders (SUMO / random)
    qubo       QUBO construction and penalty-weight selection
    evaluate   scoring (QUBO energy and true objective) and feasibility checks
    exact      brute force, enumeration and MILP (ground truth)
    heuristic  greedy, simulated annealing, tabu
"""

from .evaluate import decode, is_feasible, objective, qubo_energy
from .problem import Problem, build_problem_from_sumo, make_random_problem
from .qubo import build_qubo, penalty_bounds, qubo_offset, recommended_penalties

__all__ = [
    "Problem",
    "make_random_problem",
    "build_problem_from_sumo",
    "build_qubo",
    "qubo_offset",
    "penalty_bounds",
    "recommended_penalties",
    "qubo_energy",
    "decode",
    "is_feasible",
    "objective",
]
