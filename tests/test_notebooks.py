import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "SUMO_GEN_SCENARIO", "osm.net.xml.gz")

pytestmark = [
    pytest.mark.slow,
    # Windows-only notice from pyzmq about the asyncio event loop; harmless
    pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning"),
]

# Appended to the two-depot notebook: its plan must be the brute-force optimum, and its QUBO must
# match fleetopt.two_depot term for term.
TWO_DEPOT_CHECK = """
import sys
sys.path.insert(0, "src")
from fleetopt.two_depot import brute_force_plans, build_tour_qubo, total_distance
assert tour is not None, "notebook QUBO returned an invalid plan"
best = brute_force_plans(d, dest_labels)[0][0]
assert abs(total_distance(d, tour) - best) < 1e-6, (total_distance(d, tour), best)
Q_pkg, vars_pkg, _, A_pkg = build_tour_qubo(d, dest_labels)
assert abs(A - A_pkg) < 1e-9
assert set(Q) == set(Q_pkg) and all(abs(Q[k] - Q_pkg[k]) < 1e-9 for k in Q)
print("CHECK OK", round(best, 2))
"""


def execute(name, extra_cell=None):
    for mod in ("nbclient", "nbformat", "ipykernel", "sumolib", "matplotlib"):
        pytest.importorskip(mod)
    if not os.path.exists(NET):
        pytest.skip("SUMO network file not found")
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(os.path.join(ROOT, name), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith("%"))
    if extra_cell:
        nb.cells.append(nbformat.v4.new_code_cell(extra_cell))
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": ROOT}}).execute()
    return nb


def outputs_text(nb):
    out = []
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            out.append(o.get("text", "") if "text" in o else str(o.get("data", {}).get("text/plain", "")))
    return "\n".join(out)


def test_two_depot_notebook_runs_and_is_optimal():
    nb = execute("preprocessing_new_problem.ipynb", TWO_DEPOT_CHECK)
    assert "CHECK OK" in outputs_text(nb)
