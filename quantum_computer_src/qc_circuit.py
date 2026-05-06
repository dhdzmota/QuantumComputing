import numpy as np
from numpy import pi
from qc_utils import get_all_states, get_energy, get_selection_rules

_W0 = 1.0 / pi


def compute_energies(n_qubits, zeta, initial_zeta, j_interaction):
    """Compute energy eigenvalues for all 2^N states using the Larmor+Ising Hamiltonian."""
    w = [(initial_zeta + qb * zeta) / _W0 for qb in range(n_qubits)]
    J = [j_interaction / (10 ** ii) for ii in range(n_qubits - 1)] if n_qubits > 1 else []
    all_s = get_all_states(n_qubits)
    return [get_energy(all_s[ii], w, J, n_qubits) for ii in range(2 ** n_qubits)]


def get_allowed_transitions(n_qubits, energies):
    """
    Return all allowed single-spin-flip transitions as a list of dicts.

    A transition is allowed iff the two states differ by exactly one bit
    (Hamming distance = 1), matching the physical selection rules.

    Each dict contains:
      k, j          — 1-based state indices (k < j, i.e. k is lower energy)
      energy_diff   — E[j-1] - E[k-1]  (positive for k < j)
      resonant_freq — ω = ΔE / W0  (the driving frequency that resonates this transition)
      label_k, label_j — ket strings e.g. '|010⟩'
    """
    rules = get_selection_rules(n_qubits)
    all_states = get_all_states(n_qubits)
    num_states = 2 ** n_qubits
    seen = set()
    out = []
    for i in range(num_states):
        for j1 in rules[i]:
            pair = (min(i + 1, j1), max(i + 1, j1))
            if pair in seen:
                continue
            seen.add(pair)
            k, j = pair
            de = float(energies[j - 1] - energies[k - 1])
            lk = '|' + ''.join(str(b) for b in all_states[k - 1]) + '⟩'
            lj = '|' + ''.join(str(b) for b in all_states[j - 1]) + '⟩'
            out.append({
                'k': k, 'j': j,
                'energy_diff': de,
                'resonant_freq': de / _W0,
                'label_k': lk,
                'label_j': lj,
            })
    return out


def gates_to_pulse_schedule(gates, W_kj_func, hpp, pp):
    """
    Convert a gate list into a pulse schedule for run_simulation.

    Each gate: {'type': 'pi'|'pi_half', 'k': int, 'j': int}
      - 'pi_half' → duration hpp  (π/2 pulse: creates superposition)
      - 'pi'      → duration pp   (π   pulse: full population inversion)

    The resonant frequency for gate (k, j) with k < j is W_kj(j, k), the
    convention used by the simulation engine (higher-energy state index first
    yields a positive driving frequency that satisfies the RWA resonance condition).

    Returns [(t_start, t_end, w_mag), ...] ending with a zero-amplitude sentinel
    pulse so that run_simulation has a well-defined final time.
    """
    pulses = []
    t = 0.0
    for gate in gates:
        dur = hpp if gate['type'] == 'pi_half' else pp
        wm = W_kj_func(gate['j'], gate['k'])   # W(higher, lower) → positive ω
        pulses.append((t, t + dur, wm))
        t += dur
    pulses.append((t, t + hpp, 0.0))            # sentinel marks end of sequence
    return pulses
