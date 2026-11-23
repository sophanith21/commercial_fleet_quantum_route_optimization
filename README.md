# commercial_fleet_quantum_route_optimization

Two-depot, two-van delivery routing written as a QUBO for quantum annealing, tested on a real
road network (SUMO scenario) against exact and heuristic classical solvers.

**Problem.** Van 1 is based at depot D1 and van 2 at depot D2. Together they must serve n stops
exactly once at minimum total driving time; each van returns to its own depot and may serve no
stops.

**Formulation** (from `QUBO_Two_Van_Routing_Report.pdf`). Both routes are written as one loop,
D1 → (van 1's stops) → D2 → (van 2's stops) → D1, so the position of D2 decides which van serves
which stop. Variables x(i, p) = 1 if node i (D2 or a stop) is at tour position p; D1 is fixed at
position 0, giving (n + 1)² qubits (16 for 3 stops). An edge entering a depot is charged as the
previous van driving home: c(i, D2) = d(i, D1) and c(i, D1) = d(i, D2).

    H = A Σ_i (1 − Σ_p x_ip)² + A Σ_p (1 − Σ_i x_ip)² + B · H_travel

## Layout

| Path | What it does |
|---|---|
| `preprocessing_new_problem.ipynb` | Walk-through on the SUMO network: road graph, depots and stops, travel-time table, QUBO, exact solve, map of the plan |
| `src/fleetopt/two_depot.py` | `build_tour_qubo`, tour decoding, layouts (`sumo_layout`, `random_layout`, `layout_source`), brute force over plans, greedy rule |
| `src/fleetopt/problem.py` | `load_road_graph`: the SUMO network as a directed graph weighted by free-flow travel time |
| `src/fleetopt/exact.py` | `brute_force_qubo`: scores every QUBO bit string (up to 24 variables) |
| `src/fleetopt/heuristic.py` | `simulated_annealing` (own implementation) and `tabu` (dwave-samplers) |
| `src/fleetopt/evaluate.py` | `qubo_matrix` helper |
| `scripts/two_depot_penalty_sweep.py` | Penalty weight A against annealing success, raw against normalised QUBO → `results/two_depot_penalty_sweep.csv` |
| `scripts/two_depot_benchmark.py` | Every solver's result and run time against the optimum → `results/two_depot_benchmark.csv` |
| `scripts/two_depot_charts.py` | Solver comparison figure (`--all`: plus energy landscape, penalty sweep, benchmark and tours) → `results/figures/` |
| `tests/` | pytest suite: QUBO correctness and solvers, scripts, notebook |
| `SUMO_GEN_SCENARIO/` | SUMO network and scenario files |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[sumo,dwave,test]"   # Linux/macOS: .venv/bin/python
```

## Usage

```bash
python -m pytest                            # all 47 tests (about 50 s)
python -m pytest -m "not slow"              # skip the tests that need the SUMO network
python scripts/two_depot_penalty_sweep.py   # -> results/two_depot_penalty_sweep.csv
python scripts/two_depot_benchmark.py       # -> results/two_depot_benchmark.csv
python scripts/two_depot_charts.py --all    # -> results/figures/*.png
```

Layouts are real junctions of the SUMO road network by default: two depots and n stops drawn at
random, at least 500 m apart, with directed shortest-path travel times in seconds as costs. Layout 0
with 3 stops is the notebook's scenario (D1, D2, A, B, C; `notebook_layout`). Both scripts default
to 3 layouts; `--layouts` / `--seeds` change that, and `--source random` uses
straight-line points in a 100 × 100 square instead.

```python
from fleetopt.two_depot import build_tour_qubo, decode_tour, layout_source, max_tour_cost, total_distance
from fleetopt.heuristic import tabu

d, stops = layout_source("sumo")(0, 3)                  # layout 0 = the notebook's scenario
Q, variables, offset, A = build_tour_qubo(d, stops, A=1.0, B=1 / max_tour_cost(d, stops))
tour = decode_tour(tabu(Q, variables, seed=0)[0])
print(tour, total_distance(d, tour), "s")               # 847.5 s, the brute-force optimum
```

## Key results

All on 3 road-network layouts per size (layout 0 is the notebook's scenario); full write-up in the
project report.

- **The report's QUBO is correct.** Its worked example reproduces exactly (16 qubits, 84
  couplings, 24 valid plans, optimum 25.70, lowest invalid bit string 83.23), and on every layout
  tested the QUBO's lowest-energy state is the brute-force optimum.
- **Normalise the QUBO and use A = 1 × max c.** With the report's weights (B = 1, A = 2 × max c),
  a single annealing run found the optimum 0–3% of the time. Dividing every cost by the largest
  one fixes this without changing the optimum. A ≤ 0.75 × max c can make the lowest-energy state
  an invalid or wrong plan (at 0.75 it was right on only 1 of 3 layouts).

  | Stops | Report's weights | Normalised, A = 0.75 max c | Normalised, A = 1 max c |
  |---|---|---|---|
  | 3 | 0% | 33% | 87% |
  | 4 | 0% | 50% | 53% |
  | 5 | 3% | 17% | 13% |

  Share of single simulated-annealing runs that returned the optimal plan.

- **Solver comparison, 3 stops** (`results/figures/two_depot_solvers.png`):

  | Solver | Best route found | Average time |
  |---|---|---|
  | Brute force, every route | 3 of 3 | 64 µs |
  | Brute force, every QUBO bit string | 3 of 3 | 11 ms |
  | QUBO, tabu search | 3 of 3 | 1.1 s |
  | QUBO, simulated annealing (50 runs) | 3 of 3 | 0.5 s |
  | Greedy rule | 1 of 3 (0%, 6% and 3% longer) | 11 µs |

- **At these sizes brute force wins on speed.** There are only (n + 1)! plans (24 at 3 stops), so
  listing them all is about 8,000–17,000× faster than the QUBO solvers. The QUBO is worth testing
  where listing is impossible, against strong classical routers such as OR-Tools or LKH.
