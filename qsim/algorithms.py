"""Classic quantum algorithms implemented on top of qsim.QuantumCircuit."""
from __future__ import annotations

import math

import numpy as np

from .circuit import QuantumCircuit, ry, rz


def bell_state(seed: int | None = None) -> QuantumCircuit:
    """Prepare (|00> + |11>)/sqrt(2)."""
    qc = QuantumCircuit(2, seed)
    qc.h(0).cx(0, 1)
    return qc


def ghz_state(n: int, seed: int | None = None) -> QuantumCircuit:
    """Prepare (|0...0> + |1...1>)/sqrt(2)."""
    qc = QuantumCircuit(n, seed)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc


def deutsch_jozsa(n: int, oracle: str = "balanced", secret: int | None = None,
                  seed: int | None = None) -> str:
    """Decide whether f: {0,1}^n -> {0,1} is constant or balanced.

    oracle: 'constant0', 'constant1', or 'balanced' (f(x) = s.x mod 2, s != 0).
    Returns 'constant' or 'balanced' using a single oracle call.
    """
    qc = QuantumCircuit(n + 1, seed)
    anc = n
    qc.x(anc)
    for q in range(n + 1):
        qc.h(q)

    if oracle == "constant1":
        qc.x(anc)
    elif oracle == "balanced":
        s = secret if secret is not None else (1 << n) - 1
        if not 0 < s < (1 << n):
            raise ValueError("secret must be a nonzero n-bit integer")
        bits = format(s, f"0{n}b")
        for i, b in enumerate(bits):
            if b == "1":
                qc.cx(i, anc)
    elif oracle != "constant0":
        raise ValueError(f"unknown oracle {oracle!r}")

    for q in range(n):
        qc.h(q)
    result = qc.sample(shots=1, qubits=list(range(n)))
    bitstring = next(iter(result))
    return "constant" if set(bitstring) == {"0"} else "balanced"


def grover(n: int, marked: int, iterations: int | None = None,
           seed: int | None = None) -> QuantumCircuit:
    """Grover search for a single marked item among 2^n."""
    if not 0 <= marked < 2**n:
        raise ValueError("marked out of range")
    if iterations is None:
        iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(2**n))))
    qc = QuantumCircuit(n, seed)
    qubits = list(range(n))
    for q in qubits:
        qc.h(q)

    bits = format(marked, f"0{n}b")
    zeros = [i for i, b in enumerate(bits) if b == "0"]

    for _ in range(iterations):
        # Oracle: phase flip on |marked>
        for q in zeros:
            qc.x(q)
        qc.mcz(qubits)
        for q in zeros:
            qc.x(q)
        # Diffusion: inversion about the mean
        for q in qubits:
            qc.h(q)
            qc.x(q)
        qc.mcz(qubits)
        for q in qubits:
            qc.x(q)
            qc.h(q)
    return qc


def qft(qc: QuantumCircuit, qubits: list[int] | None = None) -> QuantumCircuit:
    """Apply the quantum Fourier transform in place (qubits[0] = MSB)."""
    qs = list(range(qc.n)) if qubits is None else qubits
    m = len(qs)
    for j in range(m):
        qc.h(qs[j])
        for k in range(j + 1, m):
            qc.cp(qs[k], qs[j], math.pi / 2 ** (k - j))
    for i in range(m // 2):
        qc.swap(qs[i], qs[m - 1 - i])
    return qc


def teleportation(theta: float, phi: float, seed: int | None = None):
    """Teleport the state ry(theta)rz(phi)|0> from qubit 0 to qubit 2.

    Returns (fidelity, (m0, m1)).
    """
    qc = QuantumCircuit(3, seed)
    prep = ry(theta) @ rz(phi)
    qc.apply(prep, 0)
    original = prep @ np.array([1, 0], dtype=complex)

    qc.h(1).cx(1, 2)          # shared Bell pair
    qc.cx(0, 1).h(0)          # Bell-basis rotation
    m0 = qc.measure(0)
    m1 = qc.measure(1)
    if m1:
        qc.x(2)
    if m0:
        qc.z(2)

    base = m0 * 4 + m1 * 2
    received = qc.state[[base, base + 1]]
    fidelity = float(abs(np.vdot(original, received)) ** 2)
    return fidelity, (m0, m1)
