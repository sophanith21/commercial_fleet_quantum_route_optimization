"""Evaluation charts for the two-depot merged-tour QUBO -> results/figures/*.png

Default output:
  two_depot_solvers.png           every solver side by side: % of layouts where it found the best
                                  route (with average extra distance) and average run time

With --all, also:
  two_depot_energy_landscape.png  report example: energy of all 2^16 bit strings, valid vs invalid
  two_depot_penalty_sweep.png     SA success vs penalty weight A, raw vs normalised QUBO
  two_depot_benchmark.png         extra distance of greedy / arbitrary plans vs optimum; solver hit rates
  two_depot_tours.png             what the tours look like: optimal, greedy and QUBO plans on a map

Reads results/two_depot_benchmark.csv (and results/two_depot_penalty_sweep.csv with --all); run
scripts/two_depot_benchmark.py (and scripts/two_depot_penalty_sweep.py) first.

Usage:
    python scripts/two_depot_charts.py
    python scripts/two_depot_charts.py --all
"""

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
    max_tour_cost, split_routes, total_distance,
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


def layout_word(rows):
    """'road-network layouts' or 'random layouts', from the CSV's source column."""
    return "road-network layouts" if rows and rows[0].get("source") == "sumo" else "random layouts"


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
    ax.set_title(f"How far from optimal ({len(data[0])} {layout_word(rows)} per size)")
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


def draw_road_plan(ax, G, nodes, d, tour, title, opt, segs, bbox):
    """One plan on the real road network: grey roads, each van's legs along its shortest road path."""
    import networkx as nx
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs, colors=RULE, linewidths=0.4, zorder=1))
    owner = {}
    for k, r in enumerate(split_routes(tour) if tour else []):
        owner.update({n: k for n in r[1:-1]})
        for a, b in zip(r, r[1:]):
            if a == b:
                continue
            path = nx.shortest_path(G, nodes[a], nodes[b], weight="weight")
            xy = np.array([G.nodes[n]["pos"] for n in path])
            ax.plot(xy[:, 0], xy[:, 1], color=VAN[k], lw=2.2 if k == 0 else 1.6, zorder=2 + k,
                    solid_joinstyle="round", solid_capstyle="round")
            m = len(xy) // 2
            if m > 0:
                ax.annotate("", xy=xy[m], xytext=xy[m - 1], zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=VAN[k], lw=1.2, mutation_scale=11))
    for name, node in nodes.items():
        x, y = G.nodes[node]["pos"]
        if name in (D1, D2):
            ax.scatter(x, y, marker="s", s=150, color=VAN[name == D2], edgecolor="white", lw=1.5, zorder=6)
            ax.annotate(name, (x, y), ha="center", va="center", color="white", fontsize=7,
                        fontweight="bold", zorder=7)
        else:
            ax.scatter(x, y, s=130, color="white", edgecolor=VAN[owner[name]] if name in owner else INK3,
                       lw=2, zorder=6)
            ax.annotate(name, (x, y), ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=7)
    if tour:
        dist = total_distance(d, tour)
        gap = (dist - opt) / opt
        sub = "optimal" if gap < TOL else f"{gap:+.0%} vs optimal"
        ax.set_title(f"{title}\n{dist:,.0f} s driving · {sub}", fontsize=9.5)
    else:
        ax.set_title(f"{title}\ninvalid bit string", fontsize=9.5, color=BAD)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(RULE)


def tours(out, net=None, seeds=((3, 0), (3, 1), (3, 2))):
    """Plans on real road-network layouts (same sampler as the benchmark), drawn on the roads."""
    import networkx as nx

    from fleetopt.problem import DEFAULT_NET, load_road_graph
    from fleetopt.two_depot import road_layout

    G = load_road_graph(net or DEFAULT_NET)
    segs = [(G.nodes[u]["pos"], G.nodes[v]["pos"]) for u, v in G.edges]
    rowsets = []
    for n, seed in seeds:
        d, dests, nodes = road_layout(G, seed, n)
        plans = brute_force_plans(d, dests)
        mc = max_tour_cost(d, dests)
        Qn, vn, _, _ = build_tour_qubo(d, dests, A=1.0, B=1 / mc)
        Qr, vr, _, _ = build_tour_qubo(d, dests)              # report weights: B = 1, A = 2 max c
        panels = [
            ("Optimal (brute force)", plans[0][1]),
            ("QUBO · tabu, normalised", decode_tour(tabu(Qn, vn, seed=0)[0])),
            ("QUBO · SA, raw weights", decode_tour(simulated_annealing(Qr, vr, seed=0)[0])),
            ("Greedy rule", greedy_plan(d, dests)),
        ]
        # Frame: every road the plans use, plus a margin
        pts = []
        for _, t in panels:
            for r in (split_routes(t) if t else []):
                for a, b in zip(r, r[1:]):
                    if a != b:
                        pts += [G.nodes[x]["pos"] for x in nx.shortest_path(G, nodes[a], nodes[b], weight="weight")]
        pts = np.array(pts)
        pad = 0.06 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        bbox = (pts[:, 0].min() - pad, pts[:, 0].max() + pad, pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        rowsets.append(("Notebook scenario" if seed == 0 and n == 3 else f"{n} stops · layout {seed}",
                        nodes, d, plans[0][0], panels, bbox))

    fig, axes = plt.subplots(len(rowsets), 4, figsize=(13, 2.9 * len(rowsets)))
    axes = np.atleast_2d(axes)
    for row, (label, nodes, d, opt, panels, bbox) in zip(axes, rowsets):
        for ax, (title, tour) in zip(row, panels):
            draw_road_plan(ax, G, nodes, d, tour, title, opt, segs, bbox)
        row[0].set_ylabel(label, fontsize=10, color=INK, fontweight="bold", labelpad=8)
    handles = [Line2D([], [], color=VAN[0], lw=2, label="Van 1 (from D1)"),
               Line2D([], [], color=VAN[1], lw=2, label="Van 2 (from D2)"),
               Line2D([], [], color=RULE, lw=1, label="SUMO road network")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Plans on the real road network (cost = free-flow travel time)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
SOLVERS = [  # (csv column, label, time column)
    ("optimum", "Brute force\n(checks every route)", "optimum_s"),
    ("bf_qubo", "Brute force on QUBO\n(checks every bit string)", "bf_qubo_s"),
    ("tabu", "QUBO · tabu search", "tabu_s"),
    ("sa", "QUBO · simulated annealing", "sa_s"),
    ("greedy", "Greedy rule", "greedy_s"),
]
SIZE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def fmt_time(sec):
    if sec < 1e-3:
        return f"{sec * 1e6:.0f} µs"
    if sec < 1:
        return f"{sec * 1e3:.1f} ms"
    return f"{sec:.2f} s"


def fmt_tick(sec):
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6)):
        if sec >= scale * 0.999:
            return f"{sec / scale:g} {unit}"
    return f"{sec * 1e9:g} ns"


def solver_stats(rows):
    """{(n, solver): (share optimal, mean extra distance, mean seconds)} on the layouts where
    the QUBO solvers ran, so every solver is scored on the same layouts."""
    stats = {}
    for n in sorted({int(r["dests"]) for r in rows}):
        rs = [r for r in rows if int(r["dests"]) == n and r["tabu"] != ""]
        for key, _, tkey in SOLVERS:
            done = [r for r in rs if r.get(key, "") != ""]
            if not done:
                continue
            gaps = [(float(r[key]) - float(r["optimum"])) / float(r["optimum"])
                    for r in done if r[key] != "invalid"]
            hits = sum(g <= TOL for g in gaps)
            secs = sum(float(r[tkey]) for r in done) / len(done)
            stats[(n, key)] = (hits / len(done), sum(gaps) / len(gaps) if gaps else None, secs, len(done))
    return stats


def solver_comparison(rows, out):
    stats = solver_stats(rows)
    ns = sorted({n for n, _ in stats})
    n_layouts = max(v[3] for v in stats.values())
    fig, (ax_ok, ax_t) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True,
                                      gridspec_kw=dict(width_ratios=[1.15, 1]))
    h = 0.8 / len(ns)
    missing = []
    ys = np.arange(len(SOLVERS))[::-1]
    for k, n in enumerate(ns):
        col = SIZE_COLORS[k % len(SIZE_COLORS)]
        off = (len(ns) - 1) / 2 * h - k * h
        for y, (key, _, _) in zip(ys, SOLVERS):
            yy = y + off
            if (n, key) not in stats:
                msg = f"not possible: 2^{(n + 1) ** 2} bit strings"
                ax_ok.text(0.01, yy, msg, va="center", fontsize=8.5, color=INK3, style="italic")
                missing.append((yy, msg))
                continue
            share, gap, secs, _ = stats[(n, key)]
            ax_ok.barh(yy, share, height=h * 0.9, color=col)
            label = f"{share:.0%}"
            if gap is not None and gap > TOL:
                label += f"  (avg {gap:+.1%} longer)" if gap < 0.01 else f"  (avg {gap:+.0%} longer)"
            ax_ok.text(share + 0.015, yy, label, va="center", fontsize=8.5, color=INK)
            ax_t.barh(yy, secs, height=h * 0.9, color=col)
            ax_t.text(secs * 1.25, yy, fmt_time(secs), va="center", fontsize=8.5, color=INK)

    ax_ok.set_yticks(ys, [lab for _, lab, _ in SOLVERS])
    ax_ok.set_xlim(0, 1.45)
    ax_ok.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_ok.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_ok.set_title("Found the best route", loc="left")
    ax_ok.set_xlabel(f"share of {n_layouts} {layout_word(rows)}")
    ax_ok.grid(axis="y", visible=False)

    ax_t.set_xscale("log")
    tmin = min(v[2] for v in stats.values())
    tmax = max(v[2] for v in stats.values())
    ax_t.set_xlim(tmin / 3, tmax * 20)
    ax_t.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tick(v)))
    for yy, msg in missing:
        ax_t.text(tmin / 2.5, yy, "–", va="center", fontsize=8.5, color=INK3)
    ax_t.set_title("Average time per layout (log scale)", loc="left")
    ax_t.set_xlabel("wall-clock time on this laptop")
    ax_t.grid(axis="y", visible=False)
    ax_t.tick_params(axis="y", length=0)

    handles = [matplotlib.patches.Patch(color=SIZE_COLORS[k % len(SIZE_COLORS)],
                                        label=f"{n} stops ({(n + 1) ** 2} qubits)")
               for k, n in enumerate(ns)]
    fig.legend(handles=handles, loc="upper center", ncol=len(ns), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "figures"))
    ap.add_argument("--sweep-csv", default=os.path.join(ROOT, "results", "two_depot_penalty_sweep.csv"))
    ap.add_argument("--benchmark-csv", default=os.path.join(ROOT, "results", "two_depot_benchmark.csv"))
    ap.add_argument("--net", default=None, help="SUMO network for the tours figure")
    ap.add_argument("--all", action="store_true", help="also write the four detailed figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    p = lambda name: os.path.join(args.outdir, name)  # noqa: E731

    bench = read_csv(args.benchmark_csv)
    stats = solver_comparison(bench, p("two_depot_solvers.png"))
    for (n, key), (share, gap, secs, cnt) in stats.items():
        print(f"{n} stops  {key:8} best route {share:4.0%} of {cnt}  avg extra "
              f"{'-' if gap is None else f'{gap:+.1%}':>6}  {fmt_time(secs)}")
    if not args.all:
        print(f"Figure written to {args.outdir}")
        return

    info = energy_landscape(p("two_depot_energy_landscape.png"))
    print(f"energy landscape: {info}")
    penalty_sweep(read_csv(args.sweep_csv),
                  p("two_depot_penalty_sweep.png"))
    benchmark(bench, p("two_depot_benchmark.png"))
    tours(p("two_depot_tours.png"), args.net)
    print(f"Figures written to {args.outdir}")


if __name__ == "__main__":
    main()
