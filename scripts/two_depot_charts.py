from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fleetopt.evaluate import qubo_matrix  # noqa: E402
from fleetopt.heuristic import simulated_annealing, tabu  # noqa: E402
from fleetopt.two_depot import (  # noqa: E402
    D1, D2, brute_force_plans, build_tour_qubo, decode_tour, distance_matrix, greedy_plan,
    max_tour_cost, random_coords, split_routes, total_distance,
)

REPORT_COORDS = {D1: (0, 0), D2: (20, 0), "A": (2, 4), "B": (18, 4), "C": (16, -3)}

# Palette (validated categorical slots): vans = slots 1-2, QUBO scalings = slots 3 and 7.
VAN = ["#2a78d6", "#eb6834"]
RAW, NORM = "#1baf7a", "#6250d6"
INK, INK2, INK3, RULE, BAD = "#141a23", "#4a5464", "#788294", "#d9dfe7", "#d03b3b"
TOL = 1e-6

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK3, "ytick.color": INK3, "axes.grid": True, "grid.color": RULE,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def read_csv(path):
    if not os.path.exists(path):
        sys.exit(f"Missing {path}; run the sweep / benchmark script first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------------
def energy_landscape(out):
    d = distance_matrix(REPORT_COORDS)
    dests = ["A", "B", "C"]
    Q, variables, offset, A = build_tour_qubo(d, dests)
    M = qubo_matrix(Q, variables)
    n = len(variables)
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
    E = np.einsum("si,ij,sj->s", states, M, states) + offset
    valid = np.array([decode_tour(dict(zip(variables, s.astype(int)))) is not None for s in states])
    opt, best_bad = E[valid].min(), E[~valid].min()

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, np.percentile(E, 99.5), 90)
    ax.hist(E[~valid], bins=bins, color=INK3, alpha=0.55, label=f"invalid ({(~valid).sum():,} bit strings)")
    ax.hist(E[valid], bins=bins, color=VAN[0], label=f"valid plans ({valid.sum()})")
    ax.set_yscale("log")
    ax.axvline(opt, color=VAN[0], lw=1.5, label=f"optimum {opt:.2f}")
    ax.axvline(best_bad, color=BAD, lw=1.5, ls="--", label=f"lowest invalid {best_bad:.2f}")
    ax.set_xlabel("H + constant (distance units for valid plans)")
    ax.set_ylabel("bit strings (log)")
    ax.set_title(f"Report example: all 2^16 bit strings, A = {A:.2f}, B = 1")
    ax.legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)
    return dict(optimum=opt, lowest_invalid=best_bad, valid=int(valid.sum()))


# ------------------------------------------------------------------------------------------------
def penalty_sweep(rows, out):
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    for r in rows:
        a = agg[(int(r["dests"]), r["scaling"], float(r["factor"]))]
        a[0] += 1
        a[1] += float(r["p_feasible"])
        a[2] += float(r["p_optimal"])
        if r["ground_optimal"] != "":
            a[3] += int(r["ground_optimal"])
            a[4] += 1
    ns = sorted({k[0] for k in agg})
    # Unsafe A: ground state not always a valid optimum (checked exactly at 3 stops)
    unsafe = max((f for (n, s, f), a in agg.items() if a[4] and a[3] < a[4]), default=None)

    fig, axes = plt.subplots(2, len(ns), figsize=(2.6 * len(ns) + 0.6, 5.2), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for c, n in enumerate(ns):
        for r_, (metric, label) in enumerate([(2, "P(optimal plan)"), (1, "P(valid plan)")]):
            ax = axes[r_, c]
            if unsafe is not None:
                ax.axvspan(0, unsafe, color=BAD, alpha=0.08, lw=0)
            ax.axvline(2, color=INK3, lw=0.8, ls=":")
            for sc, col, ls in [("raw", RAW, "--"), ("normalised", NORM, "-")]:
                pts = sorted((f, a[metric] / a[0]) for (n_, s, f), a in agg.items() if n_ == n and s == sc)
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=col, ls=ls, lw=2, marker="o", ms=4.5, mec="white", mew=0.8)
            ax.set_ylim(-0.04, 1.04)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if r_ == 0:
                ax.set_title(f"{n} stops · {(n + 1) ** 2} qubits")
            if c == 0:
                ax.set_ylabel(label)
            if r_ == 1:
                ax.set_xlabel("A / max c")
    handles = [Line2D([], [], color=NORM, lw=2, marker="o", label="normalised QUBO (B = 1 / max c)"),
               Line2D([], [], color=RAW, lw=2, ls="--", marker="o", label="raw QUBO, as in report (B = 1)"),
               Line2D([], [], color=INK3, lw=0.8, ls=":", label="report default A = 2 max c"),
               matplotlib.patches.Patch(color=BAD, alpha=0.15, label="A too small: ground state can be invalid")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Single simulated-annealing read: success vs penalty weight (5 layouts × 10 reads per point)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def benchmark(rows, out):
    ns = sorted({int(r["dests"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]))

    ax = axes[0]
    data, labels, pos = [], [], []
    for k, n in enumerate(ns):
        rs = [r for r in rows if int(r["dests"]) == n]
        opt = np.array([float(r["optimum"]) for r in rs])
        g = (np.array([float(r["greedy"]) for r in rs]) - opt) / opt
        a = (np.array([float(r["arbitrary_mean"]) for r in rs]) - opt) / opt
        data += [g, a]
        labels += [f"greedy\n{n} stops", f"random plan\n{n} stops"]
        pos += [k * 2.6, k * 2.6 + 1]
    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=True, showmeans=True,
                    meanprops=dict(marker="D", ms=5, mfc=INK, mec="white"),
                    medianprops=dict(color=INK, lw=1.5), flierprops=dict(ms=3, mec=INK3, alpha=0.6))
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor=[VAN[0], INK3][i % 2], alpha=0.35 if i % 2 else 0.5, edgecolor=INK2)
    ax.set_xticks(pos, [f"{lab}\nmean {v.mean():+.0%}" for lab, v in zip(labels, data)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("extra distance vs optimum")
    ax.set_title(f"How far from optimal ({len(data[0])} random layouts per size)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    methods = [("greedy", "Greedy", INK3), ("tabu", "QUBO · tabu", NORM), ("sa", "QUBO · SA", VAN[0])]
    w = 0.26
    for j, (key, name, col) in enumerate(methods):
        vals = []
        for n in ns:
            rs = [r for r in rows if int(r["dests"]) == n]
            if key != "greedy":
                rs = [r for r in rs if r[key] != ""]
            hits = sum(r[key] not in ("", "invalid") and float(r[key]) - float(r["optimum"]) <= TOL for r in rs)
            vals.append(hits / len(rs))
        xs = np.arange(len(ns)) + (j - 1) * w
        ax.bar(xs, vals, width=w - 0.03, color=col, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
    n_q = sum(1 for r in rows if int(r["dests"]) == ns[0] and r["tabu"] != "")
    ax.set_xticks(np.arange(len(ns)), [f"{n} stops" for n in ns])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Optimum found (QUBO on {n_q} layouts)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def draw_plan(ax, coords, d, tour, title, opt):
    for k, r in enumerate(split_routes(tour) if tour else []):
        if len(r) <= 2:
            continue
        pts = np.array([coords[n] for n in r])
        ax.plot(pts[:, 0], pts[:, 1], color=VAN[k], lw=2, zorder=2, solid_joinstyle="round")
        for a, b in zip(pts, pts[1:]):
            ax.annotate("", xy=a + 0.58 * (b - a), xytext=a + 0.42 * (b - a), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=VAN[k], lw=1.5, mutation_scale=11))
    owner = {}
    if tour:
        r1, r2 = split_routes(tour)
        owner = {**{n: 0 for n in r1}, **{n: 1 for n in r2}}
    for n, (x, y) in coords.items():
        if n in (D1, D2):
            ax.scatter(x, y, marker="s", s=150, color=VAN[n == D2], edgecolor="white", lw=1.5, zorder=4)
            ax.annotate(n, (x, y), ha="center", va="center", color="white", fontsize=7.5,
                        fontweight="bold", zorder=5)
        else:
            ax.scatter(x, y, s=120, color="white", edgecolor=VAN[owner[n]] if n in owner else INK3,
                       lw=2, zorder=4)
            ax.annotate(n, (x, y), ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)
    if tour:
        dist = total_distance(d, tour)
        gap = (dist - opt) / opt
        sub = "optimal" if gap < TOL else f"{gap:+.0%} vs optimal"
        ax.set_title(f"{title}\n{dist:.1f} · {sub}", fontsize=9.5)
    else:
        ax.set_title(f"{title}\ninvalid bit string", fontsize=9.5, color=BAD)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(RULE)


def tours(out, seed_6=1):
    rowsets = []
    d = distance_matrix(REPORT_COORDS)
    dests = ["A", "B", "C"]
    plans = brute_force_plans(d, dests)
    mc = max_tour_cost(d, dests)
    Q, v, _, _ = build_tour_qubo(d, dests, A=1.0, B=1 / mc)
    rowsets.append(("Report example (3 stops, 16 qubits)", REPORT_COORDS, d, plans[0][0], [
        ("Optimal (brute force)", plans[0][1]),
        ("QUBO · SA, normalised", decode_tour(simulated_annealing(Q, v, seed=0)[0])),
        ("Greedy rule", greedy_plan(d, dests)),
        ("Worst valid plan", plans[-1][1]),
    ]))

    coords = random_coords(seed_6, 6)
    d = distance_matrix(coords)
    dests = [k for k in coords if k not in (D1, D2)]
    plans = brute_force_plans(d, dests)
    mc = max_tour_cost(d, dests)
    Qn, vn, _, _ = build_tour_qubo(d, dests, A=1.0, B=1 / mc)
    Qr, vr, _, _ = build_tour_qubo(d, dests)                  # report weights: B = 1, A = 2 max c
    rowsets.append((f"Random layout, 6 stops (49 qubits, seed {seed_6})", coords, d, plans[0][0], [
        ("Optimal (brute force)", plans[0][1]),
        ("QUBO · tabu, normalised", decode_tour(tabu(Qn, vn, seed=0)[0])),
        ("QUBO · SA, raw weights", decode_tour(simulated_annealing(Qr, vr, seed=0)[0])),
        ("Greedy rule", greedy_plan(d, dests)),
    ]))

    fig, axes = plt.subplots(len(rowsets), 4, figsize=(12, 3.3 * len(rowsets)))
    for row, (label, coords, d, opt, panels) in zip(axes, rowsets):
        for ax, (title, tour) in zip(row, panels):
            draw_plan(ax, coords, d, tour, title, opt)
        row[0].set_ylabel(label, fontsize=9.5, color=INK2, labelpad=8)
    handles = [Line2D([], [], color=VAN[0], lw=2, label="Van 1 (from D1)"),
               Line2D([], [], color=VAN[1], lw=2, label="Van 2 (from D2)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    p = lambda name: os.path.join(args.outdir, name)  # noqa: E731

    info = energy_landscape(p("two_depot_energy_landscape.png"))
    print(f"energy landscape: {info}")
    penalty_sweep(read_csv(os.path.join(ROOT, "results", "two_depot_penalty_sweep.csv")),
                  p("two_depot_penalty_sweep.png"))
    benchmark(read_csv(os.path.join(ROOT, "results", "two_depot_benchmark.csv")),
              p("two_depot_benchmark.png"))
    tours(p("two_depot_tours.png"))
    print(f"Figures written to {args.outdir}")


if __name__ == "__main__":
    main()
