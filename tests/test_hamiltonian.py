import numpy as np
import pytest

from fleetopt.hamiltonian import (
    bitstring_to_assignment,
    ising_to_cost_hamiltonian,
    qubo_to_cost_hamiltonian,
    qubo_to_ising,
)
from fleetopt.two_depot import (
    D1,
    D2,
    brute_force_plans,
    build_tour_qubo,
    decode_tour,
    distance_matrix,
    encode_tour,
    energy,
    max_tour_cost,
    random_layout,
    total_distance,
)

REPORT_COORDS = {D1: (0, 0), D2: (20, 0), "A": (2, 4), "B": (18, 4), "C": (16, -3)}
REPORT_DESTS = ["A", "B", "C"]


def _pauli_expectation(hamiltonian, bitstring: str) -> float:
    """Exact diagonal eigenvalue <bitstring|H|bitstring> for a diagonal Z/I SparsePauliOp."""
    total = 0.0
    for pauli, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        sign = 1.0
        for p, b in zip(pauli, bitstring):
            if p == "Z" and b == "1":
                sign = -sign
        total += float(coeff.real) * sign
    return total


def test_qubo_to_ising_matches_all_bitstrings_2_stops():
    d, dests = random_layout(seed=0, n=2)
    Q, variables, offset, _ = build_tour_qubo(d, dests)
    h, J, ising_offset = qubo_to_ising(Q, variables, offset)

    n = len(variables)  # 9 qubits -> 512 states
    for bits in range(1 << n):
        x = {v: (bits >> k) & 1 for k, v in enumerate(variables)}
        s = {v: 1 - 2 * x[v] for v in variables}
        e_qubo = energy(Q, x) + offset
        e_ising = (
            ising_offset
            + sum(h[v] * s[v] for v in variables)
            + sum(w * s[u] * s[v] for (u, v), w in J.items())
        )
        assert e_ising == pytest.approx(e_qubo, abs=1e-9)


def test_cost_hamiltonian_diagonal_matches_qubo_all_states():
    pytest.importorskip("qiskit")
    d, dests = random_layout(seed=1, n=2)
    Q, variables, offset, _ = build_tour_qubo(d, dests)
    H = qubo_to_cost_hamiltonian(Q, variables, offset)

    assert H.num_qubits == len(variables)
    diag = np.real(H.to_matrix().diagonal())

    n = len(variables)
    for basis_idx, eigval in enumerate(diag):
        bitstring = format(basis_idx, f"0{n}b")
        x = bitstring_to_assignment(bitstring, variables)
        assert eigval == pytest.approx(energy(Q, x) + offset, abs=1e-8)


def test_cost_hamiltonian_matches_report_tours_raw_and_normalised():
    pytest.importorskip("qiskit")
    d = distance_matrix(REPORT_COORDS)
    mc = max_tour_cost(d, REPORT_DESTS)
    plans = brute_force_plans(d, REPORT_DESTS)

    for normalised in (False, True):
        if normalised:
            Q, variables, offset, _ = build_tour_qubo(d, REPORT_DESTS, A=1.0, B=1.0 / mc)
            scale = 1.0 / mc
        else:
            Q, variables, offset, _ = build_tour_qubo(d, REPORT_DESTS)
            scale = 1.0

        h, J, ising_offset = qubo_to_ising(Q, variables, offset)
        H = ising_to_cost_hamiltonian(h, J, ising_offset, variables)
        assert H.num_qubits == 16

        for dist, tour in plans:
            x = encode_tour(tour)
            # Little-endian bitstring: rightmost bit is variables[0]
            bitstring = "".join(str(x[v]) for v in reversed(variables))
            assert bitstring_to_assignment(bitstring, variables) == x
            assert decode_tour(bitstring_to_assignment(bitstring, variables)) == tour
            assert _pauli_expectation(H, bitstring) == pytest.approx(dist * scale, abs=1e-8)
            assert total_distance(d, tour) == pytest.approx(dist)


def test_bitstring_to_assignment_invalid_length():
    with pytest.raises(ValueError):
        bitstring_to_assignment("010", [("D2", 1), ("A", 1)])
