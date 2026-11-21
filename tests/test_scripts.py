import csv
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
NET = os.path.join(ROOT, "SUMO_GEN_SCENARIO", "osm.net.xml.gz")


def run(script, *args):
    res = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert res.returncode == 0, res.stderr
    return res.stdout


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def sweep_csv(tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep") / "sweep.csv"
    stdout = run("two_depot_penalty_sweep.py", "--dests", 3, 4, "--seeds", 1, "--runs", 2,
                 "--factors", 0.25, 1, "--source", "random", "--out", out)
    assert "P(optimal)" in stdout
    return out


@pytest.fixture(scope="module")
def benchmark_csv(tmp_path_factory):
    pytest.importorskip("dwave.samplers")
    out = tmp_path_factory.mktemp("bench") / "bench.csv"
    stdout = run("two_depot_benchmark.py", "--dests", 3, 4, "--layouts", 3, "--qubo-layouts", 1,
                 "--source", "random", "--out", out)
    assert "greedy" in stdout
    return out


def test_penalty_sweep_output(sweep_csv):
    rs = rows(sweep_csv)
    assert len(rs) == 2 * 2 * 2                      # dests x scalings x factors (1 seed)
    for r in rs:
        assert 0 <= float(r["p_feasible"]) <= 1 and 0 <= float(r["p_optimal"]) <= float(r["p_feasible"])
    # The exact ground-state check only runs at 3 stops (16 variables)
    three = [r for r in rs if r["dests"] == "3"]
    assert all(r["ground_feasible"] in ("0", "1") for r in three)
    assert all(r["ground_feasible"] == "" for r in rs if r["dests"] == "4")
    # A = 1 max c always has a valid optimal ground state; A = 0.25 does not on this layout
    assert all(r["ground_optimal"] == "1" for r in three if float(r["factor"]) == 1)


def test_benchmark_output(benchmark_csv):
    rs = rows(benchmark_csv)
    assert len(rs) == 2 * 3
    for r in rs:
        opt = float(r["optimum"])
        assert float(r["greedy"]) >= opt - 1e-6
        assert float(r["arbitrary_mean"]) >= opt - 1e-6
        assert float(r["worst"]) >= float(r["arbitrary_mean"]) - 1e-6
    qubo = [r for r in rs if r["tabu"] != ""]
    assert len(qubo) == 2                             # first layout of each size
    for r in qubo:
        assert r["tabu"] != "invalid" and r["sa"] != "invalid"


def needs_sumo():
    pytest.importorskip("sumolib")
    if not os.path.exists(NET):
        pytest.skip("SUMO network file not found")


@pytest.mark.slow
def test_benchmark_on_road_network(tmp_path):
    """Default source: real junctions of the SUMO network, directed travel times in seconds."""
    needs_sumo()
    pytest.importorskip("dwave.samplers")
    out = tmp_path / "bench.csv"
    run("two_depot_benchmark.py", "--dests", 3, "--layouts", 2, "--qubo-layouts", 1, "--out", out)
    rs = rows(out)
    assert len(rs) == 2 and all(r["source"] == "sumo" for r in rs)
    assert all(float(r["greedy"]) >= float(r["optimum"]) - 1e-6 for r in rs)
    assert float(rs[0]["tabu"]) == pytest.approx(float(rs[0]["optimum"]), abs=1e-3)


@pytest.mark.slow
def test_charts(sweep_csv, benchmark_csv, tmp_path):
    pytest.importorskip("matplotlib")
    needs_sumo()                                       # the tours figure is drawn on the road network
    run("two_depot_charts.py", "--outdir", tmp_path, "--sweep-csv", sweep_csv,
        "--benchmark-csv", benchmark_csv, "--all")
    for name in ["two_depot_solvers.png", "two_depot_energy_landscape.png", "two_depot_penalty_sweep.png",
                 "two_depot_benchmark.png", "two_depot_tours.png"]:
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 10_000, name
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
