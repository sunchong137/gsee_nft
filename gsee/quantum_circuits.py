import numpy as np
from scipy import linalg as sl
import random
from gsee import gates


def control_time_evolve_1qubit(input_state_vector, hamiltonian, dft_order, energy_rescalor=None, id="X", energy_bound=np.pi/3):
    '''
    Eq. (1) in the paper.
    The circuit to perform the controlled time evolution and Hadamard test.
    Args:
        input_state_vector : 1D array. Initial quantum state (does not include the ancilla qubit).
        hamiltonian        : 2D array. Single-qubit Hamiltonian.
        dft_order          : Integer. Sampled from [-d, d] (j in the paper).
        energy_rescalor    : float. Rescaling factor of the Hamiltonian spectrum (tau in the paper).
        id                 : String. If id == "X": the real part of exp(-i j tau H) is measured;
                                     If id == "Y": the imaginary part of exp(-i j tau H) is measured.   
        energy_bound       : the maximum of the energy spectrum.                             
    Return:
        The entangled state of the ancilla and the quantum state.
    '''
    if energy_rescalor is None:
        energy_rescalor = rescale_hamiltonian_spectrum(hamiltonian, bound=energy_bound)

    # apply Hadamard gate onto ancilla first
    ancilla = np.array([1, 0]) # initialize the ancilla to be |0>
    ancilla = np.dot(gates.Hd, ancilla)
    full_state = np.kron(ancilla, input_state_vector)

    # apply the controlled time evolution
    expH = sl.expm(-1.j * dft_order * energy_rescalor * hamiltonian) # exponential of a matrix
    c_expH = control_operator_1qubit(expH)
    full_state = np.dot(c_expH, full_state)

    # apply W gate to ancilla
    W = gates.I
    if id == 'Y':
        W = gates.Sdag
    full_W = np.kron(W, gates.I)
    full_state = np.dot(full_W, full_state)

    # apply Hadamard gate to ancilla
    full_Hd = np.kron(gates.Hd, gates.I)
    full_state = np.dot(full_Hd, full_state)

    return full_state
    
def measure_ancilla(full_state_vector):
    '''
    Do a measurement of the ancilla qubit.
    Args:
        full_state_vector : 1d array, the two-qubit state of | ancilla, state>
    Returns:
        int: 0 or 1, measurement of the ancilla qubit

    '''
    l_state = full_state_vector.shape[-1]
    l_half = int(l_state / 2)
    assert abs(l_half - l_state/2) < 1e-10 # must be even   
    p0 = np.sum(np.abs(full_state_vector[:l_half]) ** 2)

    # mimic the collapsing
    a = random.uniform(0, 1)
    if a < p0:
        return 0
    else:
        return 1
    
def measure_Xj_1qubit(input_state_vector, hamiltonian, dft_order, energy_rescalor=None, energy_bound=np.pi/3):
    """
    Measure the real part of Tr[\rho exp(-i j tau H)]
    One qubit case.
    Args:
        input_state_vector : vector, initial state.
        hamiltonian        : matrix, Hamiltonian.
        dft_order          : int, parameter sampled from [-d, d].
        energy_rescalor    : float, rescaling factor of the Hamiltonian (tau)
    returns:
        An int number to be either 1 or -1.
    """
    if energy_rescalor is None:
        energy_rescalor = rescale_hamiltonian_spectrum(hamiltonian, bound=energy_bound)

    full_state_vector = control_time_evolve_1qubit(
        input_state_vector, hamiltonian, energy_rescalor, dft_order, id="X"
    )
    ancilla_output = measure_ancilla(full_state_vector)
    Xj = -1.0 * (2 * ancilla_output - 1)  # 0 -> 1, 1 -> -1
    return Xj


def measure_Yj_1qubit(input_state_vector, hamiltonian, j_val, energy_rescalor=None, energy_bound=np.pi/3):
    """
    Measure the imaginary part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    """
    if energy_rescalor is None:
        energy_rescalor = rescale_hamiltonian_spectrum(hamiltonian, bound=energy_bound)

    full_state_vector = control_time_evolve_1qubit(
        input_state_vector, hamiltonian, energy_rescalor, j_val, id="Y"
    )
    ancilla_output = measure_ancilla(full_state_vector)
    Yj = -1.0 * (2 * ancilla_output - 1)  # 0 -> 1, 1 -> -1
    return Yj


def control_operator_1qubit(gate_1qubit):
    '''
    Given a single qubit operator op, return controlled-op.
    For single qubit, the formula is simply 
    [[I , 0 ], [0, gate]]
    '''
    c_gate = np.eye(4) + 1.j * np.zeros((4, 4))
    c_gate[2:, 2:] = gate_1qubit

    return c_gate

def rescale_hamiltonian_spectrum(hamiltonian, bound=np.pi/3):
    """
    Rescaling the hamiltonian, returns the rescaling factor tau.
    Suppose we can diagonalize the Hamiltonian.
    Args:
        hamiltonian : 2D array, the matrix representation of the Hamiltonian
        bound       : the targeted upper limit of the spectrum of the Hamiltonian
    Returns:
        Float (tau in the paper).
    """
    energies, _ = np.linalg.eigh(hamiltonian)
    energy_rescaling_factor = bound / max(abs(energies[0]), abs(energies[-1]))

    return energy_rescaling_factor

if __name__ == "__main__":

    pass