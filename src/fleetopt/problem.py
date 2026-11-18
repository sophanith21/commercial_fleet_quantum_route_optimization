"""Problem definition for congestion-aware multi-van route assignment.

The problem in one paragraph
----------------------------
A depot has a fleet of vans and a set of destinations. Each destination has a few candidate
paths (from Yen's k-shortest-paths on the road graph). We must pick, for every destination,
one van and one path, such that

  * every destination is served exactly once,
  * no van serves more than one destination,

while minimising total travel time plus a congestion penalty for routes that share roads.

This module only *describes* an instance (the `Problem` dataclass) and knows how to build one,
either from the real SUMO map or synthetically. The QUBO lives in `qubo.py`, scoring in
`evaluate.py`, and the solvers in `exact.py` / `heuristic.py`.

Naming used throughout the package
----------------------------------
  v      van id                       e.g. 1, 2, 3
  d      destination id               SUMO node id, or "D1", "D2", ... for synthetic data
  p      path index for a destination 0 .. k-1 (0 = shortest)
  route  (d, p)                       one candidate path to one destination
  var    (v, d, p)                    one binary decision: "van v drives route (d, p)"
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations, islice


@dataclass(frozen=True)
class Problem:
    """Everything the QUBO builder and the solvers need.

    Attributes:
        vans:     van ids, e.g. [1, 2, 3].
        paths:    destination -> list of candidate path indices, e.g. {"D1": [0, 1], ...}.
        C_norm:   route (d, p) -> travel time divided by the largest travel time, so in (0, 1].
                  Normalising keeps the cost term on the same scale as the congestion term.
        S:        pair of routes -> number of road edges they share. Keys are stored in one
                  order only; use `overlap()` to look up a pair in either order.
        lam_cong: weight of the congestion term relative to travel cost.
        meta:     free-form extras (depot, graph, raw costs...). Not used by the solvers and
                  excluded from equality comparison.
    """

    vans: list
    paths: dict
    C_norm: dict
    S: dict
    lam_cong: float = 0.2
    meta: dict = field(default_factory=dict, compare=False)

    # cached_property stores its result in the instance __dict__, which works on a frozen
    # dataclass (it bypasses __setattr__). Safe because the fields are never mutated.

    @cached_property
    def routes(self) -> list:
        """All candidate routes [(d, p), ...], in a fixed order (destination, then path)."""
        return [(d, p) for d, ps in self.paths.items() for p in ps]

    @cached_property
    def _max_cross_overlap(self) -> int:
        """Largest overlap between routes of *different* destinations: the congestion normaliser.

        Same-destination pairs are excluded because two paths to the same destination can never
        both be used in a feasible plan, so their overlap is irrelevant.
        """
        return max(
            (self.overlap(a, b) for a, b in combinations(self.routes, 2) if a[0] != b[0]),
            default=0,
        )

    def overlap(self, a: tuple, b: tuple) -> int:
        """Shared-edge count of routes a and b, regardless of key order (0 if unknown)."""
        return self.S.get((a, b), self.S.get((b, a), 0))

    def congestion(self, a: tuple, b: tuple) -> float:
        """Congestion cost of using routes a and b at the same time.

        lam_cong * overlap / max_overlap, so the worst-overlapping pair costs exactly lam_cong.
        Zero for two routes to the same destination (never used together) or if no routes
        overlap at all.
        """
        if a[0] == b[0] or self._max_cross_overlap == 0:
            return 0.0
        return self.lam_cong * self.overlap(a, b) / self._max_cross_overlap


# ---------------------------------------------------------------------------------------------
# Synthetic instances (tests and benchmarks; no SUMO needed)
# ---------------------------------------------------------------------------------------------

def make_random_problem(
    seed: int,
    num_vans: int = 3,
    num_destinations: int = 3,
    k_paths: int = 2,
    lam_cong: float = 0.2,
    max_overlap: int = 20,
) -> Problem:
    """Seeded random instance: same seed and arguments always give the same Problem.

    Travel times are drawn uniformly from 10-200 s (a realistic range for this map) and then
    normalised; overlaps are random integers in [0, max_overlap] for every route pair.
    """
    if num_destinations > num_vans:
        raise ValueError("Need at least as many vans as destinations.")
    rng = random.Random(seed)  # local RNG: does not disturb the global `random` state

    dests = [f"D{i + 1}" for i in range(num_destinations)]
    paths = {d: list(range(k_paths)) for d in dests}

    raw = {(d, p): rng.uniform(10, 200) for d in dests for p in paths[d]}
    c_max = max(raw.values())
    C_norm = {r: c / c_max for r, c in raw.items()}

    S = {(a, b): rng.randint(0, max_overlap) for a, b in combinations(list(C_norm), 2)}

    return Problem(
        vans=list(range(1, num_vans + 1)),
        paths=paths,
        C_norm=C_norm,
        S=S,
        lam_cong=lam_cong,
        meta={"seed": seed},
    )


# ---------------------------------------------------------------------------------------------
# Real instance from the SUMO scenario (same pipeline as preprocessing.ipynb)
# ---------------------------------------------------------------------------------------------

def build_problem_from_sumo(
    net_path: str = os.path.join("SUMO_GEN_SCENARIO", "osm.net.xml.gz"),
    num_vans: int = 3,
    num_destinations: int = 3,
    min_distance: float = 1000.0,
    k_paths: int = 2,
    skip_hops: int = 2,
    seed: int = 42,
    lam_cong: float = 0.2,
) -> Problem:
    """Build a Problem from the SUMO road network.

    Args:
        net_path:         SUMO .net.xml(.gz) file.
        min_distance:     destinations must be at least this many metres (straight line) from
                          the depot, so routes are long enough to overlap meaningfully.
        k_paths:          candidate paths per destination (Yen's algorithm).
        skip_hops:        the first edges leaving the depot are shared by *every* route; they are
                          dropped from the overlap count so they don't swamp the congestion signal.
        seed:             controls depot and destination sampling (42 reproduces the notebook).

    Raw travel times, node paths and the graph are kept in `meta` for plotting / reporting.
    """
    import networkx as nx
    import sumolib

    net = sumolib.net.readNet(net_path)

    # 1. Road graph: one directed edge per drivable SUMO edge, weighted by free-flow time ------
    G = nx.DiGraph()
    for edge in net.getEdges():
        # Skip internal junction connectors and roads a delivery van may not use
        if edge.isSpecial() or not edge.allows("delivery"):
            continue
        speed = edge.getSpeed()
        if speed <= 0:
            continue
        u, v = edge.getFromNode().getID(), edge.getToNode().getID()
        travel_time = edge.getLength() / speed  # seconds
        # DiGraph holds one edge per (u, v): keep the faster of parallel edges
        if G.has_edge(u, v) and G[u][v]["weight"] <= travel_time:
            continue
        G.add_edge(u, v, edge_id=edge.getID(), length=edge.getLength(), weight=travel_time)

    # Keep the largest strongly connected component so every node can reach every other node
    core = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(core).copy()
    for n in G.nodes:
        G.nodes[n]["pos"] = net.getNode(n).getCoord()

    # 2. Depot and destinations -----------------------------------------------------------------
    # Sorted node list + seeded RNG = reproducible choice. Destinations are restricted to the
    # right of the depot and at least min_distance away (same rule as the notebook).
    rng = random.Random(seed)
    nodes = sorted(G.nodes)
    depot = rng.choice(nodes)
    dx, dy = G.nodes[depot]["pos"]

    def far_right(n):
        x, y = G.nodes[n]["pos"]
        return n != depot and x > dx and math.hypot(x - dx, y - dy) >= min_distance

    eligible = [n for n in nodes if far_right(n)]
    if len(eligible) < num_destinations:
        raise ValueError("Not enough eligible destination nodes.")
    destinations = rng.sample(eligible, num_destinations)

    # 3. Candidate paths, their costs and their edge sets -----------------------------------------
    paths, C, E, node_paths = {}, {}, {}, {}
    for d in destinations:
        routes = list(islice(nx.shortest_simple_paths(G, depot, d, weight="weight"), k_paths))
        if not routes:
            raise ValueError(f"No path to {d}")
        paths[d] = list(range(len(routes)))
        for p, route in enumerate(routes):
            legs = list(zip(route, route[1:]))
            node_paths[d, p] = route
            C[d, p] = sum(G[u][v]["weight"] for u, v in legs)
            E[d, p] = {G[u][v]["edge_id"] for u, v in legs[skip_hops:]}

    # 4. Pairwise overlaps and normalised costs ---------------------------------------------------
    S = {(a, b): len(E[a] & E[b]) for a, b in combinations(list(C), 2)}
    c_max = max(C.values())

    return Problem(
        vans=list(range(1, num_vans + 1)),
        paths=paths,
        C_norm={r: c / c_max for r, c in C.items()},
        S=S,
        lam_cong=lam_cong,
        meta={"depot": depot, "destinations": destinations, "C": C,
              "node_paths": node_paths, "graph": G},
    )
