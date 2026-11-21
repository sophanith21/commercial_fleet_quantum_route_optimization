from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

D1, D2 = "D1", "D2"


def distance_matrix(coords: dict) -> dict:
    """Euclidean distances {(a, b): d} for every ordered pair of named points."""
    return {(a, b): float(np.hypot(*np.subtract(coords[a], coords[b])))
            for a in coords for b in coords if a != b}


def tour_cost(d: dict, i, j) -> float:
    """Cost c(i, j) of tour edge i -> j (see module docstring)."""
    if j == D2:
        return 0.0 if i == D1 else d[i, D1]
    if j == D1:
        return 0.0 if i == D2 else d[i, D2]
    return d[i, j]


def max_tour_cost(d: dict, destinations: list) -> float:
    """Largest tour-edge cost max c(i, j): the scale for the penalty weight A."""
    nodes = [D1, D2, *destinations]
    return max(tour_cost(d, i, j) for i in nodes for j in nodes if i != j)


def random_layout(seed: int, n: int, size: float = 100.0):
    """(distance dict, destination names) for two depots and n stops uniform in a size x size square."""
    return distance_matrix(random_coords(seed, n, size)), [f"S{k + 1}" for k in range(n)]


def random_coords(seed: int, n: int, size: float = 100.0) -> dict:
    rng = np.random.default_rng(seed)
    names = [D1, D2, *[f"S{k + 1}" for k in range(n)]]
    return {k: tuple(float(c) for c in rng.uniform(0, size, 2)) for k in names}


def sumo_layout(G, seed: int, n: int, min_separation: float = 500.0):
    """Two depots and n stops on real junctions of the road graph G (fleetopt.problem.load_road_graph).

    Junctions are drawn uniformly (seeded) from the whole network, at least `min_separation`
    metres apart in a straight line so no two points sit on the same block. Costs are directed
    shortest-path travel times in seconds, so d[a, b] and d[b, a] can differ.
    Returns (d, destination names, {name: junction id}).
    """
    import math
    import random

    import networkx as nx

    rng = random.Random(seed)
    pool = sorted(G.nodes)
    picked = []
    for _ in range(100_000):
        if len(picked) == n + 2:
            break
        cand = rng.choice(pool)
        pos = G.nodes[cand]["pos"]
        if all(math.dist(pos, G.nodes[q]["pos"]) >= min_separation for q in picked):
            picked.append(cand)
    else:
        raise ValueError("Could not place the points; lower min_separation.")
    labels = [D1, D2, *[f"S{k + 1}" for k in range(n)]]
    nodes = dict(zip(labels, picked))
    d = {}
    for a, na in nodes.items():
        times = nx.single_source_dijkstra_path_length(G, na, weight="weight")
        for b, nb in nodes.items():
            if a != b:
                d[a, b] = float(times[nb])
    return d, labels[2:], nodes


def _travel_times(G, nodes: dict) -> dict:
    import networkx as nx

    d = {}
    for a, na in nodes.items():
        times = nx.single_source_dijkstra_path_length(G, na, weight="weight")
        for b, nb in nodes.items():
            if a != b:
                d[a, b] = float(times[nb])
    return d


def notebook_layout(G):
    """The scenario of preprocessing_new_problem.ipynb: D1, D2 and stops A, B, C.

    Same rules and random sequence as the notebook (seed 42): two random depots; A and B east
    of D1 and at least 1,000 m from it; C moved into the area x 2,500-5,000 m, y 2,500-6,000 m,
    at least 500 m from D2. Returns (d, destinations, {name: junction id}) like sumo_layout.
    """
    import math
    import random

    rng = random.Random(42)                  # same sequence as the notebook's random.seed(42)
    nodes = sorted(G.nodes)
    pos = lambda n: G.nodes[n]["pos"]  # noqa: E731
    depot1 = rng.choice(nodes)
    depot2 = rng.choice([n for n in nodes if n != depot1])
    dx, dy = pos(depot1)
    eligible = [n for n in nodes if n not in (depot1, depot2) and pos(n)[0] > dx
                and math.dist(pos(n), (dx, dy)) >= 1000]
    stops = rng.sample(eligible, 3)
    c_candidates = [n for n in nodes if n not in (depot1, depot2, *stops[:2])
                    and 2500 <= pos(n)[0] <= 5000 and 2500 <= pos(n)[1] <= 6000
                    and math.dist(pos(n), pos(depot2)) >= 500]
    stops[2] = rng.choice(c_candidates)
    named = {D1: depot1, D2: depot2, "A": stops[0], "B": stops[1], "C": stops[2]}
    return _travel_times(G, named), ["A", "B", "C"], named


def road_layout(G, seed: int, n: int):
    """Layout `seed` on the road network: seed 0 with 3 stops is the notebook's scenario
    (notebook_layout); every other one is sumo_layout(G, seed, n)."""
    if seed == 0 and n == 3:
        return notebook_layout(G)
    return sumo_layout(G, seed, n)


def layout_source(source: str = "sumo", net_path: str | None = None):
    """Return make(seed, n) -> (d, destinations) for 'sumo' (road network) or 'random' layouts.

    The road graph is loaded once, here, so the returned function is cheap to call.
    """
    if source == "random":
        return lambda seed, n: random_layout(seed, n)
    if source != "sumo":
        raise ValueError(f"unknown layout source {source!r}")
    from .problem import DEFAULT_NET, load_road_graph
    G = load_road_graph(net_path or DEFAULT_NET)
    return lambda seed, n: road_layout(G, seed, n)[:2]


def build_tour_qubo(d: dict, destinations: list, A: float | None = None, B: float = 1.0):
    """Return (Q, variables, offset, A). A defaults to 2 * B * max c(i, j) as in the report."""
    nodes = [D2, *destinations]
    P = len(nodes)                      # positions 1..P
    positions = range(1, P + 1)
    variables = [(i, p) for i in nodes for p in positions]
    index = {v: k for k, v in enumerate(variables)}
    if A is None:
        A = 2 * B * max_tour_cost(d, destinations)

    Q: dict = {}

    def add(u, v, w):
        key = (u, v) if index[u] <= index[v] else (v, u)
        Q[key] = Q.get(key, 0.0) + w

    # Penalties: (1 - sum x)^2 = 1 - sum x + 2 sum_{a<b} x_a x_b, for every row and column.
    groups = [[(i, p) for p in positions] for i in nodes] + \
             [[(i, p) for i in nodes] for p in positions]
    for group in groups:
        for v in group:
            add(v, v, -A)
        for u, v in combinations(group, 2):
            add(u, v, 2 * A)

    # Travel cost.
    for i in nodes:
        add((i, 1), (i, 1), B * tour_cost(d, D1, i))
        add((i, P), (i, P), B * tour_cost(d, i, D1))
    for p in range(1, P):
        for i in nodes:
            for j in nodes:
                if i != j:
                    add((i, p), (j, p + 1), B * tour_cost(d, i, j))

    Q = {k: w for k, w in Q.items() if w != 0}
    return Q, variables, 2 * P * A, A


def energy(Q: dict, x: dict) -> float:
    return sum(w * x[u] * x[v] for (u, v), w in Q.items())


def decode_tour(x: dict) -> list | None:
    """Tour [D1, ...] from a bit assignment, or None if it is not a permutation."""
    nodes = sorted({i for i, _ in x})
    P = len(nodes)
    tour = [D1]
    for p in range(1, P + 1):
        here = [i for i in nodes if x[i, p]]
        if len(here) != 1:
            return None
        tour.append(here[0])
    return tour if len(set(tour)) == P + 1 else None


def encode_tour(tour: list) -> dict:
    nodes = tour[1:]
    return {(i, p): int(tour[p] == i) for i in nodes for p in range(1, len(tour))}


def split_routes(tour: list) -> tuple[list, list]:
    """(van 1 route, van 2 route), each starting and ending at its own depot."""
    k = tour.index(D2)
    return [D1, *tour[1:k], D1], [D2, *tour[k + 1:], D2]


def route_length(d: dict, route: list) -> float:
    return sum(d[a, b] for a, b in zip(route, route[1:]) if a != b)


def total_distance(d: dict, tour: list) -> float:
    """Real driving distance of both vans, computed from the routes (independent of the QUBO)."""
    r1, r2 = split_routes(tour)
    return route_length(d, r1) + route_length(d, r2)


def all_tours(destinations: list):
    """Every valid plan as a tour: (n+1)! of them (24 for 3 destinations)."""
    for perm in permutations([D2, *destinations]):
        yield [D1, *perm]


def brute_force_plans(d: dict, destinations: list) -> list[tuple[float, list]]:
    """All plans sorted by total distance."""
    return sorted(((total_distance(d, t), t) for t in all_tours(destinations)),
                  key=lambda r: r[0])


def greedy_plan(d: dict, destinations: list) -> list:
    """Report's greedy rule: each stop to its nearer depot, then nearest-unvisited ordering."""
    def order(depot, stops):
        route, here, left = [], depot, list(stops)
        while left:
            here = min(left, key=lambda s: d[here, s])
            route.append(here)
            left.remove(here)
        return route

    v1 = [s for s in destinations if d[D1, s] <= d[D2, s]]
    v2 = [s for s in destinations if s not in v1]
    return [D1, *order(D1, v1), D2, *order(D2, v2)]
