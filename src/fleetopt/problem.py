"""The SUMO road network as a directed graph (used for road-network layouts)."""

from __future__ import annotations

import os

DEFAULT_NET = os.path.join(os.path.dirname(__file__), "..", "..", "SUMO_GEN_SCENARIO", "osm.net.xml.gz")


def load_road_graph(net_path: str = DEFAULT_NET):
    """Directed road graph of the SUMO network, as used by the notebooks.

    One edge per road a delivery van may use, weighted by free-flow travel time (s); parallel
    edges keep the faster one; reduced to the largest strongly connected component so every
    junction can reach every other. Node attribute "pos" = (x, y) in metres.
    """
    import networkx as nx
    import sumolib

    net = sumolib.net.readNet(net_path)
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

    core = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(core).copy()
    for n in G.nodes:
        G.nodes[n]["pos"] = net.getNode(n).getCoord()
    return G
