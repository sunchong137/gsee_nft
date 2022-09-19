import numpy as np
from scipy import linalg as sl
import random
from gsee import helpers
from gsee import gates


def main_circuit_1q(state, ham, j, tau=None, id="X"):
    '''n circuit for the one qubit case.
    Main circuit. Eq.(1) in the paper
    Args:
        ham  : hamiltonian
        ham: Hamiltonian, 2x2 array
        j  : integer
         tau: rescaling factor
        id  : measure the real part or imaginary part.
    Return:
        The entangled state of the ancilla qubit and state qubit.
    '''
    if tau is None:
        tau = helpers.rescale_hamiltonian_spectrum(ham)

    # apply Hadamard gate onto ancilla first
    ancilla = np.array([1, 0])
    ancilla = np.dot(gates.Hd, ancilla)
    full_state = np.kron(ancilla, state)

    # apply the controlled time evolution
    expH = sl.expm(-1.j * j * tau * ham) # exponential of a matrix
    c_expH = control_op_1q(expH)
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
    
def measure_ancilla(full_state):
    '''
    Do a measurement of the ancilla qubit.
    Args:
        full_state: 1d array, the two-qubit state of | ancilla, state>
    Returns:
        int: 0 or 1, measurement of the ancilla qubit

    '''
    l_state = full_state.shape[-1]
    l_half = int(l_state / 2)
    assert abs(l_half - l_state/2) < 1e-10 # must be even   
    p0 = np.sum(np.abs(full_state[:l_half]) **2 )

    # props = np.array([p0, 1 - p0])
    # outs = np.array([0, 1])
    # a = np.random.choice(outs, p=props)
    # return a
    # NOTE: for some reason np.random.choice() is much slower than my code.

    # mimic the collapsing
    a =  random.uniform(0, 1)
    if a < p0:
        return 0
    else:
        return 1
    
def measure_Xj_1q(input_state_vector, hamiltonian, j_val, energy_rescalor=None):
    """
    Measure the real part of Tr[\rho exp(-i j tau H)]
    One qubit case.
    Args:
        input_state_vector: vector, initial state
        hamiltonian: matrix, Hamiltonian
        j_val: int, parameter sampled
        energy_rescalor: float, rescaling factor of the Hamiltonian (tau)
    returns:
        An int number to be either 1 or -1.
    """
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_spectrum(hamiltonian)

    full_state_vector = main_circuit_1q(
        input_state_vector, hamiltonian, energy_rescalor, j_val, id="X"
    )
    ancilla_output = measure_ancilla(full_state_vector)
    Xj = -1.0 * (2 * ancilla_output - 1)  # 0 -> 1, 1 -> -1
    return Xj


def measure_Yj_1q(input_state_vector, hamiltonian, j_val, energy_rescalor=None):
    """
    Measure the imaginary part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    """
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_spectrum(hamiltonian)

    full_state_vector = main_circuit_1q(
        input_state_vector, hamiltonian, energy_rescalor, j_val, id="Y"
    )
    ancilla_output = measure_ancilla(full_state_vector)
    Yj = -1.0 * (2 * ancilla_output - 1)  # 0 -> 1, 1 -> -1
    return Yj


def control_op_1q(op):
    '''
    Given a single qubit operator op, return controlled-op.
    For single qubit, the formula is simply 
    [[I , 0 ], [0, op]]
    '''
    c_op = np.eye(4) + 1.j * np.zeros((4, 4))
    c_op[2:, 2:] = op

    return c_op


if __name__ == "__main__":

    # run control_op_1q
    cnot = control_op_1q(gates.X)
    print(cnot)
    
    # run main_circuit_1qubit
    state = np.random.rand(2)
    state /= np.linalg.norm(state)
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    ew, ev = np.linalg.eigh(ham)
    tau = np.pi/(3 * max(abs(ew[0]), abs(ew[1])))
    j = 1
    W = gates.I
    full_state = main_circuit_1q(state, ham, tau, j, W)
    print(full_state)

    # run measure_ancilla()
    props = measure_ancilla(full_state)
    print(props)