"""Penalty-weight sweep for the two-depot merged-tour QUBO (fleetopt.two_depot).

Two questions per penalty setting:
  1. Ground state: is the lowest-energy bit string a valid plan, and is it the optimum?
     Checked exactly by enumerating all 2^16 states, so only for 3 destinations.
  2. Annealing: how often does a single simulated-annealing read return a valid plan / the optimum?

Layouts: real junctions of the SUMO road network (default) or --source random.

Settings: A = f * max c for each factor f. Two scalings of the same QUBO are compared:
  raw         B = 1, energies in distance units (the report's convention; A = 2*max c is f = 2)
  normalised  B = 1 / max c, so the largest travel cost is 1 and A = f
The two scalings have the same ground state, but simulated annealing's temperature schedule
(beta 0.1 -> 10) is absolute, so they anneal very differently.

Usage:
    python scripts/two_depot_penalty_sweep.py
    python scripts/two_depot_penalty_sweep.py --dests 3 4 5 --seeds 5 --runs 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.exact import brute_force_qubo  # noqa: E402
from fleetopt.heuristic import simulated_annealing  # noqa: E402
from fleetopt.two_depot import (  # noqa: E402
    brute_force_plans, build_tour_qubo, decode_tour, max_tour_cost, layout_source, total_distance,
)

FIELDS = ["source", "dests", "seed", "n_vars", "max_c", "scaling", "factor", "A", "B",
          "ground_feasible", "ground_optimal", "runs", "p_feasible", "p_optimal", "mean_gap_feasible"]
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dests", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10, help="independent single-read SA runs per setting")
    ap.add_argument("--factors", nargs="+", type=float, default=[0.5, 0.75, 1, 1.5, 2, 3])
    ap.add_argument("--scalings", nargs="+", default=["raw", "normalised"])
    ap.add_argument("--sa-sweeps", type=int, default=500)
    ap.add_argument("--source", choices=["sumo", "random"], default="sumo",
                    help="sumo = real junctions + travel times on the SUMO road network (default); "
                         "random = points in a 100 x 100 square, straight-line distances")
    ap.add_argument("--net", default=None, help="SUMO .net.xml.gz (default: SUMO_GEN_SCENARIO)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "two_depot_penalty_sweep.csv"))
    args = ap.parse_args()
    make_layout = layout_source(args.source, args.net)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tally = defaultdict(lambda: [0, 0, 0, 0, 0])  # key -> [runs, feasible, optimal, ground_ok, n_inst]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n in args.dests:
            for seed in range(args.seeds):
                d, dests = make_layout(seed, n)
                opt = brute_force_plans(d, dests)[0][0]
                mc = max_tour_cost(d, dests)
                for scaling in args.scalings:
                    B = 1.0 if scaling == "raw" else 1.0 / mc
                    for fac in args.factors:
                        A = fac * mc * B
                        Q, variables, offset, _ = build_tour_qubo(d, dests, A=A, B=B)

                        g_feas = g_opt = ""
                        if len(variables) <= 16:
                            x, _ = brute_force_qubo(Q, variables)
                            t = decode_tour(x)
                            g_feas = int(t is not None)
                            g_opt = int(t is not None and total_distance(d, t) - opt <= TOL)

                        feas = optm = 0
                        gaps = []
                        for r in range(args.runs):
                            x, _ = simulated_annealing(Q, variables, num_reads=1,
                                                       num_sweeps=args.sa_sweeps, seed=r)
                            t = decode_tour(x)
                            if t is not None:
                                feas += 1
                                gap = (total_distance(d, t) - opt) / opt
                                gaps.append(gap)
                                optm += gap <= TOL
                        writer.writerow(dict(
                            source=args.source, dests=n, seed=seed, n_vars=len(variables), max_c=f"{mc:.3f}",
                            scaling=scaling, factor=fac, A=f"{A:.4f}", B=f"{B:.5f}",
                            ground_feasible=g_feas, ground_optimal=g_opt, runs=args.runs,
                            p_feasible=feas / args.runs, p_optimal=optm / args.runs,
                            mean_gap_feasible=f"{sum(gaps) / len(gaps):.4f}" if gaps else ""))
                        t_ = tally[(n, scaling, fac)]
                        t_[0] += args.runs
                        t_[1] += feas
                        t_[2] += optm
                        t_[3] += g_opt == 1
                        t_[4] += 1
            print(f"{n} destinations: done", flush=True)

    head = f"{'dests':5} {'scaling':10} {'A/max c':>7} {'ground=opt':>10} {'P(feasible)':>11} {'P(optimal)':>10}"
    last = None
    for (n, scaling, fac), (runs, feas, optm, gok, ninst) in tally.items():
        if (n, scaling) != last:
            print("\n" + head + "\n" + "-" * len(head))
            last = (n, scaling)
        ground = f"{gok}/{ninst}" if n == 3 else "-"
        print(f"{n:5} {scaling:10} {fac:>7g} {ground:>10} {feas / runs:>11.0%} {optm / runs:>10.0%}")
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
