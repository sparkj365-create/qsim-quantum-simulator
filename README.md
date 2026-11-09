# qsim: Quantum Circuit Simulator in Python

[![CI](https://github.com/sparkj365-create/qsim-quantum-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkj365-create/qsim-quantum-simulator/actions)

A state-vector quantum circuit simulator written from scratch with NumPy, with classic quantum algorithms built on top.

Persian version: README.fa.md

## Features

- Gates: H, X, Y, Z, S, T, Rx, Ry, Rz, phase, CNOT, CZ, controlled-phase, Toffoli, SWAP, multi-controlled Z
- Measurement with state collapse, plus shot sampling
- Algorithms: Bell and GHZ states, Deutsch-Jozsa, Grover search, Quantum Fourier Transform, quantum teleportation

## Install

```bash
git clone https://github.com/sparkj365-create/qsim-quantum-simulator.git
cd qsim-quantum-simulator
pip install -r requirements.txt
```

## Quick start

```python
from qsim import QuantumCircuit

qc = QuantumCircuit(2, seed=1)
qc.h(0).cx(0, 1)          # Bell state
print(qc.sample(1000))    # only 00 and 11 appear
```

## Run the demo

```bash
python main.py
```

Sample output (excerpt):

```text
=== Grover search (n=5, marked=21) ===
P(marked) = 0.9992

=== Quantum teleportation ===
measured bits = (0, 0), fidelity = 1.000000
```

## Project structure

```text
qsim/
  circuit.py      QuantumCircuit: gates, measurement, sampling
  algorithms.py   Bell, GHZ, Deutsch-Jozsa, Grover, QFT, teleportation
tests/            pytest suite
main.py           demo of all algorithms
```

## Qubit convention

Qubit 0 is the most significant bit: the state of q0 q1 ... q(n-1) has index q0*2^(n-1) + ... + q(n-1).

## Tests

```bash
python -m pytest -q tests
```

CI runs the suite on Python 3.10, 3.11 and 3.12.

## License

MIT
