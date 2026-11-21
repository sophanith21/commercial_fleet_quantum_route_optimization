from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

# Make `fleetopt` importable when the script is run directly (no `pip install -e .` needed)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.evaluate import decode, is_feasible, objective  # noqa: E402
from fleetopt.exact import brute_force_qubo, enumerate_feasible, milp  # noqa: E402
from fleetopt.heuristic import greedy, simulated_annealing, tabu  # noqa: E402
from fleetopt.problem import make_random_problem  # noqa: E402
from fleetopt.qubo import DEFAULT_PENALTY_FACTOR, build_qubo, recommended_penalties  # noqa: E402

# Skip limits for the exact solvers that do not scale (well below their hard limits, to keep a
# default run to about a minute)
BF_MAX_VARS = 20          # brute force: 2^20 ~ 1M states
ENUM_MAX_PLANS = 50_000   # enumeration: ~7 s at 29k plans (5x5x3)

DEFAULT_SIZES = ["3x3x2", "3x3x3", "4x4x2", "4x4x3", "5x5x2", "5x5x3", "6x6x3"]
# MILP must run first: its optimum is the reference for every gap
SOLVERS = ["milp", "enumerate_feasible", "brute_force_qubo", "greedy", "simulated_annealing", "tabu"]
FIELDS = ["source", "vans", "dests", "k_paths", "seed", "n_vars", "solver",
          "status", "feasible", "objective", "gap", "time_s"]
OPTIMAL_TOL = 1e-9        # relative gap at or below this counts as optimal


def parse_size(s: str) -> tuple[int, int, int]:
    """'5x5x3' -> (vans, destinations, k_paths)."""
    v, d, k = (int(t) for t in s.lower().split("x"))
    return v, d, k


def num_plans(problem) -> int:
    """Number of valid plans: ordered van choices P(|V|, |D|) times path choices."""
    v, d = len(problem.vans), len(problem.paths)
    return math.perm(v, d) * math.prod(len(ps) for ps in problem.paths.values())


def has_tabu() -> bool:
    """True if the optional dwave-samplers package is installed."""
    try:
        import dwave.samplers  # noqa: F401
        return True
    except ImportError:
        return False


def run_solvers(problem, args):
    """Run every solver on one instance. Yields (solver, status, chosen_or_None, seconds).

    status is "ok", or "skipped (...)" with chosen = seconds = None.
    """
    t0 = time.perf_counter()
    if args.penalty_factor is None:
        A = B = 10.0                                   # old fixed default, for comparison
    else:
        A, B = recommended_penalties(problem, args.penalty_factor)
    Q, variables = build_qubo(problem, A=A, B=B)
    qubo_time = time.perf_counter() - t0
    n = len(variables)

    def timed(fn):
        """Call fn() and return (result, elapsed seconds)."""
        t = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t

    for name in SOLVERS:
        if name == "milp":
            (chosen, _), dt = timed(lambda: milp(problem, time_limit=args.milp_time_limit))
        elif name == "enumerate_feasible":
            if num_plans(problem) > ENUM_MAX_PLANS:
                yield name, f"skipped (>{ENUM_MAX_PLANS} plans)", None, None
                continue
            (chosen, _), dt = timed(lambda: enumerate_feasible(problem))
        elif name == "brute_force_qubo":
            if n > BF_MAX_VARS:
                yield name, f"skipped (>{BF_MAX_VARS} vars)", None, None
                continue
            (x, _), dt = timed(lambda: brute_force_qubo(Q, variables))
            chosen, dt = decode(x), dt + qubo_time
        elif name == "greedy":
            chosen, dt = timed(lambda: greedy(problem))
        elif name == "simulated_annealing":
            (x, _), dt = timed(lambda: simulated_annealing(
                Q, variables, num_reads=args.sa_reads, num_sweeps=args.sa_sweeps, seed=args.solver_seed))
            chosen, dt = decode(x), dt + qubo_time
        elif name == "tabu":
            if not has_tabu():
                yield name, "skipped (dwave-samplers missing)", None, None
                continue
            (x, _), dt = timed(lambda: tabu(Q, variables, num_reads=args.tabu_reads, seed=args.solver_seed))
            chosen, dt = decode(x), dt + qubo_time
        yield name, "ok", chosen, dt


def benchmark_instance(problem, source, size, seed, args, writer, rows):
    """Run all solvers on one instance; write one CSV row per solver and collect it in `rows`.

    gap = (objective - MILP optimum) / MILP optimum, filled only for feasible results.
    """
    v, d, k = size
    n_vars = len(problem.vans) * sum(len(ps) for ps in problem.paths.values())
    results = list(run_solvers(problem, args))
    milp_chosen = next(chosen for solver, _, chosen, _ in results if solver == "milp")
    ref = objective(milp_chosen, problem)
    for solver, status, chosen, dt in results:
        row = dict(source=source, vans=v, dests=d, k_paths=k, seed=seed, n_vars=n_vars,
                   solver=solver, status=status, feasible="", objective="", gap="", time_s="")
        if chosen is not None:
            feas = is_feasible(chosen, problem)
            row.update(feasible=feas, time_s=f"{dt:.4f}")
            if feas:
                val = objective(chosen, problem)
                row.update(objective=f"{val:.10f}", gap=f"{(val - ref) / ref:.3e}")
        writer.writerow(row)
        rows.append(row)


def summarise(rows):
    """Print one line per (instance size, solver), aggregated over seeds."""
    groups = defaultdict(list)  # insertion-ordered, so output follows the run order
    for r in rows:
        groups[(r["source"], r["vans"], r["dests"], r["k_paths"], r["n_vars"], r["solver"])].append(r)

    head = f"{'source':6} {'size':7} {'vars':>4}  {'solver':20} {'runs':>4} {'feasible':>8} " \
           f"{'optimal':>7} {'mean gap':>10} {'max gap':>9} {'mean time':>10}"
    print("\n" + head + "\n" + "-" * len(head))
    last = None
    for (src, v, d, k, n, solver), rs in groups.items():
        if last is not None and (src, v, d, k) != last:
            print()  # blank line between instance sizes
        last = (src, v, d, k)
        ran = [r for r in rs if r["status"] == "ok"]
        size = f"{v}x{d}x{k}"
        if not ran:
            print(f"{src:6} {size:7} {n:>4}  {solver:20} {rs[0]['status']}")
            continue
        feas = [r for r in ran if r["feasible"] is True]
        gaps = [float(r["gap"]) for r in feas]
        opt = sum(g <= OPTIMAL_TOL for g in gaps)
        times = [float(r["time_s"]) for r in ran]
        mean_gap = f"{sum(gaps) / len(gaps):.4%}" if gaps else "-"
        max_gap = f"{max(gaps):.4%}" if gaps else "-"
        print(f"{src:6} {size:7} {n:>4}  {solver:20} {len(ran):>4} {len(feas):>4}/{len(ran):<3} "
              f"{opt:>3}/{len(ran):<3} {mean_gap:>10} {max_gap:>9} {sum(times) / len(times):>9.4f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--seeds", type=int, default=5, help="random instances per size")
    ap.add_argument("--sumo", action="store_true", help="also benchmark the real SUMO scenario")
    ap.add_argument("--sumo-sizes", nargs="+", default=["3x3x2", "3x3x3", "4x4x2", "4x4x3", "5x5x2", "5x5x3"],
                    help="SUMO sizes; the map is reloaded per size (about 1 s each)")
    ap.add_argument("--sa-reads", type=int, default=10, help="SA restarts per instance")
    ap.add_argument("--sa-sweeps", type=int, default=300, help="SA sweeps per restart")
    ap.add_argument("--tabu-reads", type=int, default=5)
    ap.add_argument("--solver-seed", type=int, default=0, help="seed for SA and tabu")
    ap.add_argument("--milp-time-limit", type=float, default=60, help="seconds per MILP solve")
    ap.add_argument("--penalty-factor", type=float, default=DEFAULT_PENALTY_FACTOR,
                    help="A = B = factor * A_min (default %(default)s)")
    ap.add_argument("--old-penalties", dest="penalty_factor", action="store_const", const=None,
                    help="use the old fixed A = B = 10 instead")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "benchmark.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        # Synthetic instances: seeds 0..N-1 per size
        for s in args.sizes:
            size = parse_size(s)
            for seed in range(args.seeds):
                p = make_random_problem(seed, num_vans=size[0], num_destinations=size[1], k_paths=size[2])
                benchmark_instance(p, "random", size, seed, args, writer, rows)
            print(f"random {s}: {args.seeds} instances done", flush=True)

        # Real map: one instance per size, seed 42 as in the notebook
        if args.sumo:
            from fleetopt.problem import build_problem_from_sumo  # needs sumolib
            net = os.path.join(ROOT, "SUMO_GEN_SCENARIO", "osm.net.xml.gz")
            for s in args.sumo_sizes:
                size = parse_size(s)
                p = build_problem_from_sumo(net, num_vans=size[0], num_destinations=size[1],
                                            k_paths=size[2], seed=42)
                benchmark_instance(p, "sumo", size, 42, args, writer, rows)
                print(f"sumo   {s}: done", flush=True)

    summarise(rows)
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
