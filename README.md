# N-Qubit Quantum Computer Simulator

A physics-based simulation of an N-qubit quantum computer modelled on nuclear spins coupled by Ising interactions. Quantum gates are implemented as timed electromagnetic pulses (π and π/2 pulses) acting on the spin chain. The default 3-qubit configuration demonstrates **quantum teleportation** end-to-end.

> Based on the author's undergraduate professional project — see [`files/PAP-I.pdf`](./files/PAP-I.pdf) for the full theoretical derivation.

---

## Features

- **Physical fidelity** — Hamiltonian evolution solved with 4th-order Runge-Kutta; no gate-level abstractions
- **N-qubit support** — 2 to 5 qubits; algorithm auto-selected per qubit count
- **Interactive web app** — real-time parameter tuning, state probability plots, animated Bloch spheres, and a drag-and-drop circuit builder
- **Custom circuits** — compose arbitrary π / π/2 pulse sequences and export results as PDF
- **Qiskit integration** — exploratory notebooks comparing the physical simulation against Qiskit-Aer

---

## Physical Model

| Quantity | Description |
|---|---|
| Larmor frequency | `w[i] = (ζ₀ + i·ζ) / W₀` |
| Ising coupling | `J[i] = J / 10^i` (nearest / next-nearest neighbours) |
| π pulse duration | `pp = π / U` |
| π/2 pulse duration | `hpp = pp / 2` |
| State indexing | 1-based, matching the reference paper |

Qubit probabilities are tracked as `D[j] = |cx[j]|² + |cy[j]|²` in the interaction picture; Bloch vectors are recovered by tracing out the reduced density matrix at each time step.

---

## Project Structure

```
.
├── app.py                      # Dash web application (primary interface)
├── pdf_report.py               # PDF export for circuit simulation results
├── quantum_computer_src/
│   ├── qc_simulation.py        # RK4 simulation engine & pulse scheduler
│   ├── qc_circuit.py           # Allowed transitions & energy computation
│   └── qc_utils.py             # Pure utility functions (states, energies, selection rules)
├── notebooks/                  # Exploratory Jupyter notebooks
│   └── qc_utils.py             # Mirror of quantum_computer_src/qc_utils.py
├── files/
│   └── PAP-I.pdf               # Theoretical reference (undergraduate thesis)
├── results/                    # Simulation output artifacts
├── requirements.txt
└── README.md
```

---

## Setup

Python 3.8 and a project-level virtualenv are required.

```bash
# Create and activate the virtualenv
python3.8 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running

### Web App (recommended)

```bash
source venv/bin/activate
python app.py
# → http://localhost:8050/
```

The sidebar exposes all physical parameters. Click **Run Simulation** to compute, then explore:

- **State Probabilities** tab — time-evolution of all 2ᴺ basis state amplitudes
- **Bloch Spheres** tab — per-qubit Bloch vectors with a time slider
- **Circuit Builder** tab — compose custom pulse sequences and run them independently

### Legacy Simulation Script

```bash
source venv/bin/activate
cd quantum_computer_src
python qc_simulation.py
```

### Notebooks

```bash
source venv/bin/activate
jupyter lab
```

---

## Architecture

### `qc_simulation.py` — Simulation Engine

| Function | Purpose |
|---|---|
| `run_simulation(...)` | RK4 integration; returns `T`, `D`, `cx`, `cy`, `E`, `states` |
| `compute_bloch_vectors(cx, cy, n_qubits, t_idx)` | Reduced density matrix → `(⟨σx⟩, ⟨σy⟩, ⟨σz⟩)` per qubit |
| `_build_pulse_schedule(...)` | Resonant pulse sequence for the selected algorithm |

### `qc_utils.py` — Utilities

| Function | Purpose |
|---|---|
| `get_all_states(n)` | All 2ᴺ binary state vectors |
| `get_selection_rules(n)` | Allowed transitions (Hamming distance = 1) |
| `get_energy(state, ...)` | Energy eigenvalue under Larmor + Ising Hamiltonian |

### Algorithms by Qubit Count

| N | Algorithm |
|---|---|
| 2 | Entanglement Demo |
| 3 | Quantum Teleportation |
| 4 | Multi-Qubit Cascade |
| 5 | Multi-Qubit Cascade |

---

## Reference

D. H. Díaz-Mota, *Simulación de una computadora cuántica de N qubits basada en espines nucleares* (undergraduate thesis), 2022. [`files/PAP-I.pdf`](./files/PAP-I.pdf)
