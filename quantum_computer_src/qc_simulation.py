import numpy as np
from numpy import sin, cos, sqrt, pi
from qc_utils import get_all_states, get_num_states, get_energy, get_selection_rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_pulse_schedule(n_qubits, hpp, pp, W_kj):
    """Return list of (t_start, t_end, w_mag) pulses for the chosen algorithm."""
    if n_qubits == 3:
        # Quantum teleportation (3-qubit)
        t = [0, hpp, 2*hpp, 4*hpp, 6*hpp, 7*hpp, 8*hpp, 9*hpp, 10*hpp, 11*hpp]
        wm = [W_kj(3,1), W_kj(7,5), W_kj(4,3), W_kj(6,5),
              W_kj(5,1), W_kj(6,2), W_kj(8,4), W_kj(7,3), 0.0]
    elif n_qubits == 2:
        t = [0, hpp, hpp+pp, 2*hpp+pp, 3*hpp+pp]
        wm = [W_kj(3,1), W_kj(4,3), W_kj(2,1), 0.0]
    elif n_qubits == 4:
        t = [0, hpp, hpp+pp, 2*hpp+pp, 3*hpp+pp, 4*hpp+pp]
        wm = [W_kj(2,1), W_kj(3,1), W_kj(5,1), W_kj(9,1), 0.0]
    elif n_qubits == 5:
        t = [0, hpp, hpp+pp, 2*hpp+pp, 3*hpp+pp, 3*hpp+2*pp, 4*hpp+2*pp]
        wm = [W_kj(2,1), W_kj(3,1), W_kj(5,1), W_kj(9,1), W_kj(17,1), 0.0]
    else:
        t = [0, 2*hpp, 2*hpp+pp]
        wm = [W_kj(2,1), 0.0]
    return [(t[i], t[i+1], wm[i]) for i in range(len(wm))]


def compute_bloch_vectors(cx_data, cy_data, n_qubits, time_idx):
    """
    Compute Bloch vectors for each qubit at time_idx.

    Derives the single-qubit reduced density matrix by tracing out the rest,
    then returns (bx, by, bz) = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩) per qubit.
    """
    num_s = 2 ** n_qubits
    psi = np.array([cx_data[i][time_idx] + 1j * cy_data[i][time_idx]
                    for i in range(num_s)])
    vectors = []
    for k in range(n_qubits):
        rho_00, rho_01 = 0.0, 0.0 + 0j
        for i in range(num_s):
            if not ((i >> (n_qubits - 1 - k)) & 1):       # bit k = 0
                j = i | (1 << (n_qubits - 1 - k))         # partner with bit k = 1
                rho_00 += abs(psi[i]) ** 2
                rho_01 += np.conj(psi[i]) * psi[j]
        bx = 2.0 * float(np.real(rho_01))
        by = -2.0 * float(np.imag(rho_01))
        bz = float(2.0 * rho_00 - 1.0)
        vectors.append((bx, by, bz))
    return vectors


def run_simulation(n_qubits=3, zeta=100, initial_zeta=100, j_interaction=10,
                   step_size=0.01, initial_amplitudes=None, custom_gates=None):
    """
    Run the N-qubit spin-chain simulation via 4th-order Runge-Kutta.

    Returns a dict:
        T        – time array (list)
        D        – 2-D list [state][time] of state probabilities
        cx, cy   – 2-D lists of complex amplitude components (interaction picture)
        E        – list of energy eigenvalues
        states   – ket labels  e.g. ['|000⟩', '|001⟩', ...]
        n_qubits – number of qubits
    """
    W0  = 1.0 / pi
    U   = 0.1 / W0
    PHI = 0.0

    w = [(initial_zeta + qb * zeta) / W0 for qb in range(n_qubits)]
    J = [j_interaction / (10 ** ii) for ii in range(n_qubits - 1)] if n_qubits > 1 else []

    all_s  = get_all_states(n_qubits=n_qubits)
    num_s  = 2 ** n_qubits
    E      = [get_energy(all_s[ii], w, J, n_qubits) for ii in range(num_s)]
    r      = get_selection_rules(n_qubits)
    E_arr  = np.array(E)

    W_kj = lambda k, j: (E[k - 1] - E[j - 1]) / W0

    pp_val = pi / U
    hpp    = pp_val / 2

    if custom_gates:
        from qc_circuit import gates_to_pulse_schedule
        pulses = gates_to_pulse_schedule(custom_gates, W_kj, hpp, pp_val)
    else:
        pulses = _build_pulse_schedule(n_qubits, hpp, pp_val, W_kj)
    tf     = pulses[-1][1]

    def w_mag_at(t_val):
        for ts, te, wm in pulses:
            if t_val <= te:
                return wm
        return 0.0

    n  = int(tf / step_size)
    vx = np.zeros((num_s, n))
    vy = np.zeros((num_s, n))
    cx = np.zeros((num_s, n))
    cy = np.zeros((num_s, n))
    D  = np.zeros((num_s, n))
    T  = np.zeros(n)
    K  = np.zeros((5, num_s))
    L  = np.zeros((5, num_s))

    if initial_amplitudes is not None:
        for idx, amp in enumerate(list(initial_amplitudes)[:num_s]):
            vx[idx, 0] = float(amp)
    elif n_qubits == 3:
        vx[0, 0] = sqrt(2.0 / 10.0)
        vx[4, 0] = sqrt(8.0 / 10.0)
    else:
        vx[0, 0] = 1.0

    def _diff(t_val, step_i, j_st, m_st, wm_val):
        ALPHA = 0.0 if m_st == 1 else (0.5 if m_st in (2, 3) else 1.0)
        tv    = t_val + step_size * ALPHA
        pos   = j_st + 1
        sx = sy = 0.0
        for ri in r[j_st]:
            if pos < ri:
                A1 = 1
                B1 = W_kj(pos, ri) * tv + wm_val * tv + PHI
            elif pos > ri:
                A1 = 0
                B1 = W_kj(ri, pos) * tv + wm_val * tv + PHI
            else:
                continue
            AK = ALPHA * K[m_st - 1, ri - 1]
            AL = ALPHA * L[m_st - 1, ri - 1]
            Xr = vx[ri - 1, step_i]
            Yr = vy[ri - 1, step_i]
            sb, cb = sin(B1), cos(B1)
            p = (-1) ** A1
            sx += (U / 2) * (p * sb * (Xr + AK) - cb * (Yr + AL))
            sy += (U / 2) * (cb * (Xr + AK) + p * sb * (Yr + AL))
        return sx, sy

    for i in range(n - 1):
        T[i + 1] = T[i] + step_size
        if T[i + 1] > tf:
            break
        wm = w_mag_at(T[i + 1])
        for m in range(1, 5):
            for j in range(num_s):
                dx, dy = _diff(T[i], i, j, m, wm)
                K[m, j] = step_size * dx
                L[m, j] = step_size * dy
        vx[:, i + 1] = vx[:, i] + (K[1] + 2*K[2] + 2*K[3] + K[4]) / 6
        vy[:, i + 1] = vy[:, i] + (L[1] + 2*L[2] + 2*L[3] + L[4]) / 6
        cos_E = np.cos(E_arr * T[i])
        sin_E = np.sin(E_arr * T[i])
        cx[:, i] = vx[:, i] * cos_E - vy[:, i] * sin_E
        cy[:, i] = vx[:, i] * sin_E + vy[:, i] * cos_E
        D[:,  i] = cx[:, i] ** 2 + cy[:, i] ** 2

    cos_E = np.cos(E_arr * T[-1])
    sin_E = np.sin(E_arr * T[-1])
    cx[:, -1] = vx[:, -1] * cos_E - vy[:, -1] * sin_E
    cy[:, -1] = vx[:, -1] * sin_E + vy[:, -1] * cos_E
    D[:,  -1]  = cx[:, -1] ** 2 + cy[:, -1] ** 2

    labels = ['|' + ''.join(str(b) for b in all_s[ii]) + '⟩'
              for ii in range(num_s)]

    return {
        'T':        T.tolist(),
        'D':        D.tolist(),
        'cx':       cx.tolist(),
        'cy':       cy.tolist(),
        'E':        E,
        'states':   labels,
        'n_qubits': n_qubits,
    }


# ---------------------------------------------------------------------------
# Legacy script entry-point (kept for direct execution)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    QUBITS_NB   = 3
    ZETA        = 100
    INITIAL_ZETA = 100
    J_INT       = 10
    STEP_SIZE   = 0.01

    res = run_simulation(
        n_qubits=QUBITS_NB,
        zeta=ZETA,
        initial_zeta=INITIAL_ZETA,
        j_interaction=J_INT,
        step_size=STEP_SIZE,
    )

    T      = res['T']
    D      = res['D']
    states = res['states']
    N_norm = [sum(d[i] for d in D) for i in range(len(T))]

    plt.figure(figsize=(20, 8))
    plt.plot(T, N_norm, label='Norm', color='k', linewidth=1)
    for idx, (d, label) in enumerate(zip(D, states)):
        plt.plot(T, d, label=label)
    plt.xlabel('Time (τ)')
    plt.ylabel('Probability')
    plt.title(f'Quantum Teleportation – {QUBITS_NB} Qubits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
