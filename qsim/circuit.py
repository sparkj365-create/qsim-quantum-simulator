"""A small state-vector quantum circuit simulator built on NumPy.

Qubit convention: qubit 0 is the most significant bit in basis labels,
so the state |q0 q1 ... q(n-1)> has index q0*2^(n-1) + ... + q(n-1).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

# ---- Single-qubit gates -------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def phase_gate(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


class QuantumCircuit:
    def __init__(self, n_qubits: int, seed: int | None = None):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n = n_qubits
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0
        self.classical: dict[int, int] = {}

    # ---- Core gate application ------------------------------------------
    def _check(self, *qubits: int) -> None:
        for q in qubits:
            if not 0 <= q < self.n:
                raise IndexError(f"qubit {q} out of range for {self.n} qubits")
        if len(set(qubits)) != len(qubits):
            raise ValueError("qubits must be distinct")

    def apply(self, gate: np.ndarray, target: int, controls: tuple[int, ...] = ()):
        """Apply a 2x2 gate to `target`, optionally controlled on `controls`."""
        self._check(target, *controls)
        psi = self.state.reshape([2] * self.n)
        # Move target axis to the front.
        psi = np.moveaxis(psi, target, 0)
        if not controls:
            psi = np.tensordot(gate, psi, axes=([1], [0]))
        else:
            # After moving target to axis 0, a control at original index c
            # sits at c+1 if c < target, else at c.
            ctrl_axes = [c + 1 if c < target else c for c in controls]
            idx = [slice(None)] * self.n
            for ax in ctrl_axes:
                idx[ax] = 1
            idx = tuple(idx)
            sub = psi[idx]  # axis 0 is still the target axis
            psi = psi.copy()
            psi[idx] = np.tensordot(gate, sub, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, target)
        self.state = np.ascontiguousarray(psi).reshape(-1)
        return self

    # ---- Gate helpers ------------------------------------------------------
    def h(self, q): return self.apply(H, q)
    def x(self, q): return self.apply(X, q)
    def y(self, q): return self.apply(Y, q)
    def z(self, q): return self.apply(Z, q)
    def s(self, q): return self.apply(S, q)
    def t(self, q): return self.apply(T, q)
    def rx(self, q, theta): return self.apply(rx(theta), q)
    def ry(self, q, theta): return self.apply(ry(theta), q)
    def rz(self, q, theta): return self.apply(rz(theta), q)
    def p(self, q, theta): return self.apply(phase_gate(theta), q)

    def cx(self, control, target): return self.apply(X, target, (control,))
    def cz(self, control, target): return self.apply(Z, target, (control,))
    def cp(self, control, target, theta):
        return self.apply(phase_gate(theta), target, (control,))
    def ccx(self, c1, c2, target): return self.apply(X, target, (c1, c2))

    def mcz(self, qubits: list[int]):
        """Multi-controlled Z over all given qubits (last one is the target)."""
        *ctrls, tgt = qubits
        return self.apply(Z, tgt, tuple(ctrls))

    def swap(self, a, b):
        return self.cx(a, b).cx(b, a).cx(a, b)

    # ---- Measurement -------------------------------------------------------
    def probabilities(self, qubits: list[int] | None = None) -> np.ndarray:
        """Probability distribution over the given qubits (default: all)."""
        probs = np.abs(self.state) ** 2
        if qubits is None:
            return probs
        self._check(*qubits)
        probs = probs.reshape([2] * self.n)
        other = tuple(i for i in range(self.n) if i not in qubits)
        marg = probs.sum(axis=other) if other else probs
        # Reorder axes to follow the order of `qubits`.
        kept_sorted = sorted(qubits)
        perm = [kept_sorted.index(q) for q in qubits]
        marg = np.transpose(marg, perm)
        return marg.reshape(-1)

    def measure(self, qubit: int, cbit: int | None = None) -> int:
        """Projective measurement of one qubit; collapses the state."""
        self._check(qubit)
        psi = self.state.reshape([2] * self.n)
        p1 = float(np.sum(np.abs(np.take(psi, 1, axis=qubit)) ** 2))
        outcome = int(self.rng.random() < p1)
        mask = np.zeros(2, dtype=complex)
        mask[outcome] = 1.0
        shape = [1] * self.n
        shape[qubit] = 2
        psi = psi * mask.reshape(shape)
        self.state = psi.reshape(-1)
        self.state /= np.linalg.norm(self.state)
        self.classical[qubit if cbit is None else cbit] = outcome
        return outcome

    def sample(self, shots: int = 1024, qubits: list[int] | None = None) -> Counter:
        """Sample bitstrings without collapsing the state."""
        qs = list(range(self.n)) if qubits is None else qubits
        probs = self.probabilities(qs)
        probs = probs / probs.sum()
        draws = self.rng.choice(len(probs), size=shots, p=probs)
        width = len(qs)
        return Counter(format(int(d), f"0{width}b") for d in draws)

    def statevector_str(self, threshold: float = 1e-9) -> str:
        parts = []
        for i, amp in enumerate(self.state):
            if abs(amp) > threshold:
                parts.append(f"({amp.real:+.3f}{amp.imag:+.3f}j)|{i:0{self.n}b}>")
        return " + ".join(parts)
