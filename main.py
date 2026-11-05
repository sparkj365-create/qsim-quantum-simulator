"""Demo runner for the qsim quantum circuit simulator."""
import math

from qsim.algorithms import (bell_state, deutsch_jozsa, ghz_state, grover, qft,
                             teleportation)
from qsim import QuantumCircuit


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("Bell state")
    qc = bell_state(seed=42)
    print("state:", qc.statevector_str())
    print("1000 shots:", dict(sorted(qc.sample(1000).items())))

    section("GHZ state (4 qubits)")
    print("500 shots:", dict(sorted(ghz_state(4, seed=1).sample(500).items())))

    section("Deutsch-Jozsa (n=4)")
    for oracle in ("constant0", "constant1", "balanced"):
        print(f"{oracle:10s} ->", deutsch_jozsa(4, oracle))

    section("Grover search (n=5, marked=21)")
    qc = grover(5, 21, seed=7)
    probs = qc.probabilities()
    print(f"P(marked) = {probs[21]:.4f}")
    print("100 shots:", dict(qc.sample(100).most_common(3)))

    section("Quantum Fourier Transform (3 qubits, input |5>)")
    qc = QuantumCircuit(3)
    qc.x(0).x(2)  # |101> = 5
    qft(qc)
    for i, a in enumerate(qc.state):
        print(f"  |{i:03b}>  |amp|={abs(a):.4f}  phase={math.degrees(math.atan2(a.imag, a.real)):7.1f} deg")

    section("Quantum teleportation")
    fid, (m0, m1) = teleportation(theta=1.1, phi=0.6, seed=5)
    print(f"measured bits = ({m0}, {m1}), fidelity = {fid:.6f}")


if __name__ == "__main__":
    main()
