import numpy as np

def get_num_states(n_qubits):
    """
    Function that generates the count of the total amount of states.

    Parameters:
        n_qubits: Number of qubits.
    Returns: 
        int: An int that shows the total amount of states.
    """
    num_states = 2 ** n_qubits
    return num_states


def get_all_states(n_qubits):
    """
    Generate all possible binary representations of states for a given number of bits.
    For a given quantum system this represents a linear combination of all possible 
    states in superposition.
    
    Parameters:
        n_qubits (int): Number of qubits for binary representation.
    Returns:
        numpy.ndarray: An array containing all possible binary states.
    """
    # Create a range from 0 to 2^n and convert to binary using bitwise operations
    num_states = get_num_states(n_qubits=n_qubits)
    possible_binary_states = [
        [(i >> j) & 1 for j in range(n_qubits-1, -1, -1)] for i in range(num_states)
    ]
    possible_binary_states = np.array(possible_binary_states)
    return possible_binary_states


def get_selection_rules(n_qubits):
    """
    Generate selection rules for N bits based on transitions.

    Parameters:
        n (int): Number of bits.

    Returns:
        numpy.ndarray: A 2D array representing the selection rules.
    """
    num_states = get_num_states(n_qubits=n_qubits)
    states = np.arange(1, num_states + 1)  # State indices (1-based indexing)
    binary_states = get_all_states(n_qubits=n_qubits)

    # Precompute Hamming distances
    hamming_distances = np.sum(np.abs(binary_states[:, None] - binary_states), axis=2)

    # Identify transitions with Hamming distance of 1
    rules = [states[np.where(hamming_distances[i] == 1)[0]].tolist() for i in range(num_states)]
    return np.array(rules)

    
def get_energy(estado, larmor, ising, n_qubits):
    """
    Compute the energy of a system governed by Larmor and Ising interactions.

    Parameters:
        estado (list or array): State of the system (e.g., [0, 1, 1]).
        larmor (list or array): Larmor frequencies for each spin.
        ising (list): Ising interaction coefficients.
        n_qubits (int): Number of qubits.

    Returns:
        float: Energy of the system.
    """
    # Ensure inputs are NumPy arrays
    estado = estado[::-1]
    estado = np.array(estado, dtype=int)
    larmor = np.array(larmor, dtype=float)

    # Calculate the non-interaction energy term
    energy_NI = np.dot((-1) ** estado, larmor)

    # Calculate the interaction energy term
    energy_I = 0
    for j in range(1, n_qubits):  # j represents the interaction distance
        energy_I += np.sum(ising[j - 1] * (-1) ** (estado[:-j] + estado[j:]))

    # Calculate total energy
    energy_states = -0.5 * (energy_NI + 0.5 * energy_I)
    return energy_states