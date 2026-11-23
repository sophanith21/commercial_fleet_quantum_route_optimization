"""Two-depot, two-van route optimisation as a QUBO, with classical baselines.

Typical use:

    from fleetopt.two_depot import (build_tour_qubo, decode_tour, layout_source,
                                    max_tour_cost, total_distance)
    from fleetopt.heuristic import tabu

    d, stops = layout_source("sumo")(seed=0, n=3)    # 3 stops on the SUMO road network
    Q, variables, offset, A = build_tour_qubo(d, stops, A=1.0, B=1 / max_tour_cost(d, stops))
    x, _ = tabu(Q, variables, seed=0)
    print(decode_tour(x), total_distance(d, decode_tour(x)), "s")

Modules:
    two_depot  merged-tour QUBO, layouts (road network or random), decoding, brute force, greedy
    problem    SUMO road network as a directed graph (load_road_graph)
    exact      brute force over every QUBO bit string
    heuristic  simulated annealing, tabu search
    evaluate   QUBO matrix helper
"""
