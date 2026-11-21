import pytest

from conftest import TINY_OPT_VALUE
from fleetopt.evaluate import decode, is_feasible, objective, qubo_energy
from fleetopt.exact import enumerate_feasible
from fleetopt.heuristic import greedy, simulated_annealing, tabu
from fleetopt.qubo import build_qubo

SA_KW = dict(num_reads=10, num_sweeps=200)  # small budget: keeps the suite fast


@pytest.mark.parametrize("seed", range(20))
def test_greedy_always_feasible_and_never_beats_exact(seed, random_problem):
    p = random_problem(seed)
    chosen = greedy(p)
    _, opt = enumerate_feasible(p)
    assert is_feasible(chosen, p)
    assert objective(chosen, p) >= opt - 1e-9


def test_greedy_gap_report(random_problem):
    """Greedy is not optimal; record its gap and guard against regressions."""
    gaps = []
    for seed in range(20):
        p = random_problem(seed)
        _, opt = enumerate_feasible(p)
        gaps.append((objective(greedy(p), p) - opt) / opt)
    assert max(gaps) < 0.5, f"greedy gap regressed: {max(gaps):.2%}"


# ---------- simulated annealing ----------

def test_sa_tiny_optimal(tiny):
    Q, variables = build_qubo(tiny)
    x, _ = simulated_annealing(Q, variables, seed=0, **SA_KW)
    chosen = decode(x)
    assert is_feasible(chosen, tiny)
    assert objective(chosen, tiny) == pytest.approx(TINY_OPT_VALUE)


def test_sa_is_reproducible_with_seed(random_problem):
    Q, variables = build_qubo(random_problem(0))
    assert simulated_annealing(Q, variables, seed=123, **SA_KW) == \
        simulated_annealing(Q, variables, seed=123, **SA_KW)


def test_sa_reported_energy_is_consistent(random_problem):
    """SA tracks energy incrementally; it must match a from-scratch recomputation."""
    Q, variables = build_qubo(random_problem(2))
    x, e = simulated_annealing(Q, variables, seed=1, **SA_KW)
    assert e == pytest.approx(qubo_energy(Q, x))


@pytest.mark.parametrize("seed", range(10))
def test_sa_matches_exact(seed, random_problem):
    p = random_problem(seed)
    Q, variables = build_qubo(p)
    _, opt = enumerate_feasible(p)
    x, _ = simulated_annealing(Q, variables, seed=seed, **SA_KW)
    chosen = decode(x)
    assert is_feasible(chosen, p)
    assert objective(chosen, p) == pytest.approx(opt)


def test_sa_success_rate(random_problem):
    """Single-read success probability over fixed seeds (deterministic, so not flaky)."""
    p = random_problem(4)
    Q, variables = build_qubo(p)
    _, opt = enumerate_feasible(p)
    hits = 0
    for s in range(30):
        x, _ = simulated_annealing(Q, variables, num_reads=1, num_sweeps=200, seed=s)
        c = decode(x)
        hits += is_feasible(c, p) and objective(c, p) == pytest.approx(opt)
    assert hits / 30 >= 0.8



@pytest.mark.parametrize("seed", range(5))
def test_tabu_matches_exact(seed, random_problem):
    pytest.importorskip("dwave.samplers")
    p = random_problem(seed)
    Q, variables = build_qubo(p)
    _, opt = enumerate_feasible(p)
    x, _ = tabu(Q, variables, num_reads=5, seed=seed)
    chosen = decode(x)
    assert is_feasible(chosen, p)
    assert objective(chosen, p) == pytest.approx(opt)
