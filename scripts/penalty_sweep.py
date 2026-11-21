from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

# Make `fleetopt` importable when the script is run directly (no `pip install -e .` needed)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.evaluate import decode, is_feasible, objective  # noqa: E402
from fleetopt.exact import milp  # noqa: E402
from fleetopt.heuristic import simulated_annealing  # noqa: E402
from fleetopt.problem import make_random_problem  # noqa: E402
from fleetopt.qubo import build_qubo, penalty_bounds  # noqa: E402

FIELDS = ["vans", "dests", "k_paths", "seed", "n_vars", "A_min", "setting", "A", "B",
          "runs", "p_feasible", "p_optimal", "mean_gap_feasible"]
OPTIMAL_TOL = 1e-9  # relative gap at or below this counts as optimal


@dataclass
class Tally:
    """Running totals for one (size, setting) over all seeds and runs."""
    runs: int = 0
    feasible: int = 0
    optimal: int = 0
    gap_sum: float = 0.0   # over feasible runs only


def settings(A_min, factors, b_modes):
    """Yield (label, A, B) for every factor x B-mode combination, plus the old A = B = 10."""
    for f in factors:
        A = f * A_min
        for mode in b_modes:
            B = A if mode == "A" else float(mode) * A_min
            yield f"A={f:g}xA_min, B={'A' if mode == 'A' else mode + 'xA_min'}", A, B
    yield "old default A=B=10", 10.0, 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["3x3x2", "4x4x3", "5x5x3", "6x6x3"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=10, help="independent single-read SA runs per setting")
    ap.add_argument("--factors", nargs="+", type=float, default=[1.1, 1.5, 2, 3, 5])
    ap.add_argument("--b-modes", nargs="+", default=["A", "0.5"],
                    help="'A' means B=A; a number m means B = m * A_min")
    ap.add_argument("--sa-sweeps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "penalty_sweep.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tallies = defaultdict(Tally)  # (size, setting label) -> Tally
    a_mins = []

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for size in args.sizes:
            v, d, k = (int(t) for t in size.lower().split("x"))
            for seed in range(args.seeds):
                problem = make_random_problem(seed, num_vans=v, num_destinations=d, k_paths=k)
                _, opt = milp(problem)                   # exact reference for the gap
                A_min, _ = penalty_bounds(problem)
                a_mins.append(A_min)

                for label, A, B in settings(A_min, args.factors, args.b_modes):
                    Q, variables = build_qubo(problem, A=A, B=B)

                    # Independent single-read SA runs: the fraction that succeed estimates the
                    # per-read success probability, the usual metric for annealers.
                    feasible = optimal = 0
                    gaps = []
                    for run_seed in range(args.runs):
                        x, _ = simulated_annealing(Q, variables, num_reads=1,
                                                   num_sweeps=args.sa_sweeps, seed=run_seed)
                        chosen = decode(x)
                        if is_feasible(chosen, problem):
                            feasible += 1
                            gap = (objective(chosen, problem) - opt) / opt
                            gaps.append(gap)
                            optimal += gap <= OPTIMAL_TOL

                    writer.writerow(dict(
                        vans=v, dests=d, k_paths=k, seed=seed, n_vars=len(variables),
                        A_min=f"{A_min:.4f}", setting=label, A=f"{A:.4f}", B=f"{B:.4f}",
                        runs=args.runs, p_feasible=feasible / args.runs,
                        p_optimal=optimal / args.runs,
                        mean_gap_feasible=f"{sum(gaps) / len(gaps):.4e}" if gaps else ""))

                    t = tallies[(size, label)]
                    t.runs += args.runs
                    t.feasible += feasible
                    t.optimal += optimal
                    t.gap_sum += sum(gaps)
            print(f"{size}: done", flush=True)

    # Summary: one block per size, one line per penalty setting
    print(f"\nA_min over all instances: min {min(a_mins):.3f}, max {max(a_mins):.3f} "
          f"(old default A = 10 is {10 / max(a_mins):.0f}-{10 / min(a_mins):.0f}x larger)")
    head = f"{'size':7} {'setting':30} {'P(feasible)':>11} {'P(optimal)':>10} {'mean gap':>9}"
    last_size = None
    for (size, label), t in tallies.items():
        if size != last_size:
            print("\n" + head + "\n" + "-" * len(head))
            last_size = size
        gap = f"{t.gap_sum / t.feasible:.2%}" if t.feasible else "-"
        print(f"{size:7} {label:30} {t.feasible / t.runs:>11.0%} {t.optimal / t.runs:>10.0%} {gap:>9}")
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
