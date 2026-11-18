"""Penalty-weight sweep: how A and B affect simulated-annealing success.

Every A tested is a multiple of the provable bound A_min (fleetopt.qubo.penalty_bounds), so every
setting has a feasible ground state; what changes is how easy that state is to find. The old
fixed default A = B = 10 is included for comparison.

For each (instance, A, B) SA runs --runs times with num_reads=1, giving a per-read
probability of returning a feasible plan and of returning the exact (MILP) optimum.

Usage:
    python scripts/penalty_sweep.py
    python scripts/penalty_sweep.py --sizes 5x5x3 6x6x3 --seeds 10 --runs 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.evaluate import decode, is_feasible, objective  # noqa: E402
from fleetopt.exact import milp  # noqa: E402
from fleetopt.heuristic import simulated_annealing  # noqa: E402
from fleetopt.problem import make_random_problem  # noqa: E402
from fleetopt.qubo import build_qubo, penalty_bounds  # noqa: E402

FIELDS = ["vans", "dests", "k_paths", "seed", "n_vars", "A_min", "setting", "A", "B",
          "runs", "p_feasible", "p_optimal", "mean_gap_feasible"]


def settings(A_min, factors, b_modes):
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
    agg = defaultdict(lambda: [0, 0, 0, 0.0])  # setting -> [runs, feasible, optimal, gap_sum]
    a_mins = []

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in args.sizes:
            v, d, k = (int(t) for t in s.lower().split("x"))
            for seed in range(args.seeds):
                p = make_random_problem(seed, num_vans=v, num_destinations=d, k_paths=k)
                _, opt = milp(p)
                A_min, _ = penalty_bounds(p)
                a_mins.append(A_min)
                for name, A, B in settings(A_min, args.factors, args.b_modes):
                    Q, variables = build_qubo(p, A=A, B=B)
                    feas = optimal = 0
                    gaps = []
                    for r in range(args.runs):
                        x, _ = simulated_annealing(Q, variables, num_reads=1,
                                                   num_sweeps=args.sa_sweeps, seed=r)
                        c = decode(x)
                        if is_feasible(c, p):
                            feas += 1
                            g = (objective(c, p) - opt) / opt
                            gaps.append(g)
                            optimal += g <= 1e-9
                    w.writerow(dict(vans=v, dests=d, k_paths=k, seed=seed, n_vars=len(variables),
                                    A_min=f"{A_min:.4f}", setting=name, A=f"{A:.4f}", B=f"{B:.4f}",
                                    runs=args.runs, p_feasible=feas / args.runs,
                                    p_optimal=optimal / args.runs,
                                    mean_gap_feasible=f"{sum(gaps) / len(gaps):.4e}" if gaps else ""))
                    a = agg[(s, name)]
                    a[0] += args.runs; a[1] += feas; a[2] += optimal; a[3] += sum(gaps)
            print(f"{s}: done", flush=True)

    print(f"\nA_min over all instances: min {min(a_mins):.3f}, max {max(a_mins):.3f} "
          f"(old default A = 10 is {10 / max(a_mins):.0f}-{10 / min(a_mins):.0f}x larger)")
    head = f"{'size':7} {'setting':30} {'P(feasible)':>11} {'P(optimal)':>10} {'mean gap':>9}"
    last = None
    for (s, name), (runs, feas, opt, gsum) in agg.items():
        if s != last:
            print("\n" + head + "\n" + "-" * len(head))
            last = s
        gap = f"{gsum / feas:.2%}" if feas else "-"
        print(f"{s:7} {name:30} {feas / runs:>11.0%} {opt / runs:>10.0%} {gap:>9}")
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
