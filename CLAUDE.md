# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A physical simulation of an N-qubit quantum computer based on nuclear spin (Ising model) physics. The simulation models qubits as nuclear spins coupled by Ising interactions and drives state transitions using timed electromagnetic pulses (π and π/2 pulses). The main demo implements quantum teleportation with 3 qubits.

## Environment Setup

All work uses the project-level virtualenv (`venv/`, Python 3.8). Activate before running anything:

```bash
source venv/bin/activate
```

Key dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `qiskit`, `qiskit-aer`, `jupyterlab`, `dash`, `plotly`.

## Running

**Web app** (primary interface — run from project root):

```bash
python app.py
# → open http://localhost:8050
```

**Simulation script** (legacy, must run from `quantum_computer_src/`):

```bash
cd quantum_computer_src
python qc_simulation.py
```

**Notebooks:**

```bash
jupyter lab
```

## Architecture

### `quantum_computer_src/`

- **`qc_utils.py`** — Pure utility functions with no global state:
  - `get_all_states(n_qubits)` — generates all 2^N binary state vectors
  - `get_num_states(n_qubits)` — returns 2^N
  - `get_selection_rules(n_qubits)` — builds an (2^N × k) array of allowed transitions; two states are connected if their Hamming distance is 1 (single spin flip)
  - `get_energy(estado, larmor, ising, n_qubits)` — computes the energy eigenvalue for a state under Larmor + Ising Hamiltonian

- **`qc_simulation.py`** — Simulation engine exposing a public API:
  - `run_simulation(n_qubits, zeta, initial_zeta, j_interaction, step_size, initial_amplitudes)` — runs RK4 integration and returns a dict with `T`, `D`, `cx`, `cy`, `E`, `states`
  - `compute_bloch_vectors(cx_data, cy_data, n_qubits, time_idx)` — computes (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) Bloch vector for each qubit at a given time step by tracing out the reduced density matrix
  - `_build_pulse_schedule(n_qubits, hpp, pp, W_kj)` — returns the resonant pulse sequence for the chosen algorithm (quantum teleportation for N=3; cascade sequences for N=2,4,5)
  - The interaction-picture amplitudes `vx`/`vy` are stored per state; `cx`/`cy` are obtained by rotating back by energy phase; `D[j] = cx[j]² + cy[j]²` is the probability of state j

### `app.py` (project root)

Dash web application. The sidebar exposes physical parameters (N qubits, ζ, J, step size, initial state preset); clicking **Run** stores results in a `dcc.Store` and updates two tabs:
- **State Probabilities** — Plotly line chart, one trace per 2^N state plus a normalisation check
- **Bloch Spheres** — N interactive 3D Bloch sphere figures (one per qubit) driven by a time slider; Bloch vectors are recomputed from `cx`/`cy` on every slider change without re-running the simulation

### `notebooks/`

Jupyter notebooks for exploratory work and visualization. `notebooks/qc_utils.py` is an exact copy of `quantum_computer_src/qc_utils.py` — keep them in sync when modifying utilities.

### Physical model

- Qubits are nuclear spins with Larmor frequencies `w[i] = (INITIAL_ZETA + i * ZETA) / W0`
- Ising coupling strengths decay as `J[i] = j_interaction / 10^i` for nearest/next-nearest neighbors
- Gate operations are π pulses (full inversion) and π/2 pulses (superposition creation), parameterized by `pp = π/U` and `hpp = pp/2`
- State indexing is 1-based throughout (matches the physics literature reference in `files/PAP-I.pdf`)
