from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from math import factorial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.exact import brute_force_qubo  # noqa: E402
from fleetopt.heuristic import simulated_annealing, tabu  # noqa: E402
from fleetopt.two_depot import (  # noqa: E402
    brute_force_plans, build_tour_qubo, decode_tour, greedy_plan, max_tour_cost, layout_source,
    total_distance,
)

FIELDS = ["source", "dests", "seed", "optimum", "optimum_s", "greedy", "greedy_s", "arbitrary_mean", "worst",
          "bf_qubo", "bf_qubo_s", "tabu", "tabu_s", "sa", "sa_s"]
MAX_BRUTE_VARS = 16   # 2^16 bit strings; 2^25 (4 stops) is already too slow
TOL = 1e-6


def timed(f, *args, **kwargs):
    t0 = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t0


def solve_qubo(solver, d, dests, factor, seed=None):
    """Run solver on the normalised QUBO. Returns (distance or None if invalid, seconds)."""
    mc = max_tour_cost(d, dests)
    Q, variables, _, _ = build_tour_qubo(d, dests, A=factor, B=1.0 / mc)
    t0 = time.perf_counter()
    x, _ = solver(Q, variables) if seed is None else solver(Q, variables, seed=seed)
    secs = time.perf_counter() - t0
    tour = decode_tour(x)
    return (total_distance(d, tour) if tour else None), secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dests", nargs="+", type=int, default=[3])
    ap.add_argument("--layouts", type=int, default=3)
    ap.add_argument("--qubo-layouts", type=int, default=3)
    ap.add_argument("--factor", type=float, default=1.0, help="A / max c for the QUBO solvers")
    ap.add_argument("--source", choices=["sumo", "random"], default="sumo",
                    help="sumo = real junctions + travel times on the SUMO road network (default); "
                         "random = points in a 100 x 100 square, straight-line distances")
    ap.add_argument("--net", default=None, help="SUMO .net.xml.gz (default: SUMO_GEN_SCENARIO)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "two_depot_benchmark.csv"))
    args = ap.parse_args()
    make_layout = layout_source(args.source, args.net)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n in args.dests:
            for seed in range(args.layouts):
                d, dests = make_layout(seed, n)
                plans, opt_s = timed(brute_force_plans, d, dests)
                g, g_s = timed(greedy_plan, d, dests)
                row = dict(source=args.source, dests=n, seed=seed, optimum=plans[0][0], optimum_s=opt_s,
                           greedy=total_distance(d, g), greedy_s=g_s,
                           arbitrary_mean=sum(p[0] for p in plans) / len(plans),
                           worst=plans[-1][0], bf_qubo="", bf_qubo_s="", tabu="", tabu_s="", sa="", sa_s="")
                if seed < args.qubo_layouts:
                    if (n + 1) ** 2 <= MAX_BRUTE_VARS:
                        row["bf_qubo"], row["bf_qubo_s"] = solve_qubo(brute_force_qubo, d, dests, args.factor)
                    row["tabu"], row["tabu_s"] = solve_qubo(tabu, d, dests, args.factor, seed)
                    row["sa"], row["sa_s"] = solve_qubo(simulated_annealing, d, dests, args.factor, seed)
                writer.writerow({k: ("invalid" if v is None else (f"{v:.3e}" if k.endswith("_s") else f"{v:.4f}")
                                     if isinstance(v, float) else v)
                                 for k, v in row.items()})
                rows.append(row)
            print(f"{n} destinations: done", flush=True)

    for n in args.dests:
        rs = [r for r in rows if r["dests"] == n]
        gap = [(r["greedy"] - r["optimum"]) / r["optimum"] for r in rs]
        arb = [(r["arbitrary_mean"] - r["optimum"]) / r["optimum"] for r in rs]
        print(f"\n{n} destinations, {len(rs)} layouts ({factorial(n + 1)} plans each)")
        print(f"  greedy  mean gap {sum(gap) / len(gap):+.0%}   worst {max(gap):+.0%}   "
              f"optimal on {sum(g <= TOL for g in gap)}/{len(gap)}")
        print(f"  arbitrary plan mean gap {sum(arb) / len(arb):+.0%}")
        for name in ("tabu", "sa"):
            qs = [r for r in rs if r[name] != ""]
            if not qs:
                continue
            valid = [r for r in qs if r[name] is not None]
            hits = sum(r[name] - r["optimum"] <= TOL for r in valid)
            secs = sum(r[name + "_s"] for r in qs) / len(qs)
            print(f"  QUBO {name:4} optimal on {hits}/{len(qs)}, valid {len(valid)}/{len(qs)}, "
                  f"{secs:.2f} s per layout")
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
