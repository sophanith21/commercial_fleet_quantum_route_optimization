"""Convert a QUBO to an Ising model and a Qiskit SparsePauliOp cost Hamiltonian.

Mapping convention (matches Qiskit's little-endian computational basis):
    x_k = (1 - Z_k) / 2   so   |0> (Z = +1) -> x = 0,   |1> (Z = -1) -> x = 1.

Qubit k (variables[k]) sits at position (n - 1 - k) in Qiskit Pauli strings and
measurement bitstrings (rightmost character is qubit 0).
"""
from __future__ import annotations


def qubo_to_ising(Q: dict, variables: list, offset: float = 0.0) -> tuple[dict, dict, float]:
    """Convert QUBO sum_{(u,v)} Q[u,v] x_u x_v + offset to Ising (h, J, ising_offset).

    Returns (h, J, ising_offset) for
        H(s) = ising_offset + sum_i h[i] s_i + sum_{(i,j)} J[i,j] s_i s_j,
    where s_i in {+1, -1} corresponds to Z_i and x_i = (1 - s_i) / 2.
    """
    index = {v: i for i, v in enumerate(variables)}
    h: dict = {v: 0.0 for v in variables}
    J: dict = {}
    ising_offset = float(offset)

    for (v1, v2), coeff in Q.items():
        if coeff == 0:
            continue
        if v1 == v2:
            h[v1] -= coeff / 2.0
            ising_offset += coeff / 2.0
        else:
            key = (v1, v2) if index[v1] <= index[v2] else (v2, v1)
            J[key] = J.get(key, 0.0) + coeff / 4.0
            h[v1] -= coeff / 4.0
            h[v2] -= coeff / 4.0
            ising_offset += coeff / 4.0

    J = {k: w for k, w in J.items() if w != 0.0}
    return h, J, ising_offset


def ising_to_cost_hamiltonian(h: dict, J: dict, ising_offset: float, variables: list):
    """Build a Qiskit SparsePauliOp cost Hamiltonian from Ising (h, J, ising_offset).

    Requires qiskit to be installed.
    """
    from qiskit.quantum_info import SparsePauliOp

    qubit_index = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    pauli_terms: list[tuple[str, float]] = []

    if ising_offset != 0.0:
        pauli_terms.append(("I" * n, float(ising_offset)))

    for v, coeff in h.items():
        if coeff == 0.0:
            continue
        pauli = ["I"] * n
        pauli[n - 1 - qubit_index[v]] = "Z"
        pauli_terms.append(("".join(pauli), float(coeff)))

    for (v1, v2), coeff in J.items():
        if coeff == 0.0:
            continue
        pauli = ["I"] * n
        pauli[n - 1 - qubit_index[v1]] = "Z"
        pauli[n - 1 - qubit_index[v2]] = "Z"
        pauli_terms.append(("".join(pauli), float(coeff)))

    if not pauli_terms:
        pauli_terms.append(("I" * n, 0.0))

    return SparsePauliOp.from_list(pauli_terms)


def qubo_to_cost_hamiltonian(Q: dict, variables: list, offset: float = 0.0):
    """Convenience wrapper: convert a QUBO directly to a Qiskit SparsePauliOp."""
    h, J, ising_offset = qubo_to_ising(Q, variables, offset)
    return ising_to_cost_hamiltonian(h, J, ising_offset, variables)


def bitstring_to_assignment(bitstring: str, variables: list) -> dict:
    """Convert a Qiskit measurement bitstring (little-endian) to {variable: 0 or 1}.

    In Qiskit, bitstring[-1] is qubit 0 (variables[0]) and bitstring[0] is qubit n-1.
    """
    n = len(variables)
    if len(bitstring) != n:
        raise ValueError(f"Expected bitstring of length {n}, got {len(bitstring)}")
    return {var: int(bitstring[n - 1 - idx]) for idx, var in enumerate(variables)}
