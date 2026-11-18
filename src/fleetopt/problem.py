"""Problem definition, built either from the SUMO scenario or synthetically."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from itertools import combinations, islice

Route = tuple  # (destination, path_index)


@dataclass(frozen=True)
class Problem:
    """Everything the QUBO builder and the solvers need.

    vans:    van ids, e.g. [1, 2, 3]
    paths:   destination -> list of candidate path indices
    C_norm:  (d, p) -> normalised travel cost in [0, 1]
    S:       ((d, p), (d', p')) -> shared-edge count (either key order)
    lam_cong: weight of the congestion term
    """

    vans: list
    paths: dict
    C_norm: dict
    S: dict
    lam_cong: float = 0.2
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def routes(self) -> list:
        return [(d, p) for d, ps in self.paths.items() for p in ps]

    def overlap(self, a: Route, b: Route) -> int:
        return self.S.get((a, b), self.S.get((b, a), 0))

    def max_cross_overlap(self) -> int:
        """Largest overlap between routes of *different* destinations (the normaliser)."""
        return max(
            (self.overlap(a, b) for a, b in combinations(self.routes, 2) if a[0] != b[0]),
            default=0,
        )

    def congestion(self, a: Route, b: Route) -> float:
        """Normalised congestion cost of using routes a and b together."""
        if a[0] == b[0]:
            return 0.0
        m = self.max_cross_overlap()
        return 0.0 if m == 0 else self.lam_cong * self.overlap(a, b) / m


def make_random_problem(
    seed: int,
    num_vans: int = 3,
    num_destinations: int = 3,
    k_paths: int = 2,
    lam_cong: float = 0.2,
    max_overlap: int = 20,
) -> Problem:
    """Seeded synthetic instance for property tests and benchmarks."""
    if num_destinations > num_vans:
        raise ValueError("Need at least as many vans as destinations.")
    rng = random.Random(seed)
    dests = [f"D{i + 1}" for i in range(num_destinations)]
    paths = {d: list(range(k_paths)) for d in dests}
    raw = {(d, p): rng.uniform(10, 200) for d in dests for p in paths[d]}
    c_max = max(raw.values())
    C_norm = {r: c / c_max for r, c in raw.items()}
    routes = list(C_norm)
    S = {(a, b): rng.randint(0, max_overlap) for a, b in combinations(routes, 2)}
    return Problem(
        vans=list(range(1, num_vans + 1)),
        paths=paths,
        C_norm=C_norm,
        S=S,
        lam_cong=lam_cong,
        meta={"seed": seed},
    )


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
    """Same pipeline as preprocessing.ipynb, packaged as a function."""
    import networkx as nx
    import sumolib  # optional dependency

    net = sumolib.net.readNet(net_path)

    G = nx.DiGraph()
    for edge in net.getEdges():
        if edge.isSpecial() or not edge.allows("delivery"):
            continue
        speed = edge.getSpeed()
        if speed <= 0:
            continue
        u, v = edge.getFromNode().getID(), edge.getToNode().getID()
        tt = edge.getLength() / speed
        if G.has_edge(u, v) and G[u][v]["weight"] <= tt:
            continue
        G.add_edge(u, v, edge_id=edge.getID(), length=edge.getLength(), weight=tt)

    core = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(core).copy()
    for n in G.nodes:
        G.nodes[n]["pos"] = net.getNode(n).getCoord()

    rng = random.Random(seed)
    nodes = sorted(G.nodes)
    depot = rng.choice(nodes)
    dx, dy = G.nodes[depot]["pos"]
    eligible = [
        n
        for n in nodes
        if n != depot
        and G.nodes[n]["pos"][0] > dx
        and math.hypot(G.nodes[n]["pos"][0] - dx, G.nodes[n]["pos"][1] - dy) >= min_distance
    ]
    if len(eligible) < num_destinations:
        raise ValueError("Not enough eligible destination nodes.")
    destinations = rng.sample(eligible, num_destinations)

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
