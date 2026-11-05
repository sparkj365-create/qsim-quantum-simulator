import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsim import QuantumCircuit
from qsim.algorithms import (bell_state, deutsch_jozsa, ghz_state, grover, qft,
                             teleportation)


def test_hadamard_superposition():
    qc = QuantumCircuit(1)
    qc.h(0)
    assert np.allclose(qc.probabilities(), [0.5, 0.5])


def test_cnot_control_ordering():
    qc = QuantumCircuit(3)
    qc.x(2).cx(2, 0)          # control above target index
    assert qc.state[0b101] == 1
    qc = QuantumCircuit(3)
    qc.x(0).cx(0, 2)          # control below target index
    assert qc.state[0b101] == 1


def test_toffoli():
    qc = QuantumCircuit(3)
    qc.x(0).x(1).ccx(0, 1, 2)
    assert qc.state[0b111] == 1


def test_swap():
    qc = QuantumCircuit(2)
    qc.x(0).swap(0, 1)
    assert qc.state[0b01] == 1


def test_bell_state_correlated():
    qc = bell_state(seed=1)
    assert np.allclose(qc.probabilities(), [0.5, 0, 0, 0.5])
    counts = qc.sample(500)
    assert set(counts) <= {"00", "11"}


def test_ghz():
    qc = ghz_state(4)
    p = qc.probabilities()
    assert np.isclose(p[0], 0.5) and np.isclose(p[-1], 0.5)


def test_measurement_collapse():
    qc = bell_state(seed=3)
    a = qc.measure(0)
    b = qc.measure(1)
    assert a == b


def test_deutsch_jozsa():
    for n in (1, 3, 5):
        assert deutsch_jozsa(n, "constant0") == "constant"
        assert deutsch_jozsa(n, "constant1") == "constant"
        assert deutsch_jozsa(n, "balanced") == "balanced"
        assert deutsch_jozsa(n, "balanced", secret=1) == "balanced"


def test_grover_finds_marked():
    for n, marked in [(3, 5), (4, 11), (5, 0), (6, 42)]:
        qc = grover(n, marked)
        p = qc.probabilities()
        assert int(np.argmax(p)) == marked
        assert p[marked] > 0.9


def test_qft_matches_dft_matrix():
    n = 3
    N = 2**n
    for x in range(N):
        qc = QuantumCircuit(n)
        for i, b in enumerate(format(x, f"0{n}b")):
            if b == "1":
                qc.x(i)
        qft(qc)
        expected = np.exp(2j * np.pi * x * np.arange(N) / N) / math.sqrt(N)
        assert np.allclose(qc.state, expected)


def test_teleportation():
    rng = np.random.default_rng(0)
    for k in range(20):
        theta, phi = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
        fid, _ = teleportation(theta, phi, seed=k)
        assert fid > 1 - 1e-9


def test_normalisation_preserved():
    qc = QuantumCircuit(4, seed=0)
    qc.h(0).cx(0, 1).rx(2, 0.7).cp(1, 3, 1.1).ry(3, 0.4)
    assert np.isclose(np.linalg.norm(qc.state), 1.0)
