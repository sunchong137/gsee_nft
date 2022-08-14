import numpy as np
from scipy import linalg as sl
import random
import helpers

# Define single-qubit gates
X = np.array([[0., 1.], [1., 0.]])
Y = 1.j * np.array([[0., -1.], [1., 0.]])
Z = np.array([[1., 0.], [0., -1.]])
I  = np.eye(2) # identity gate
Hd = np.array([[1., 1.], [1., -1.]])/np.sqrt(2) # Hadamard gate
S = np.array([[1., 0.], [0., 1.j]]) # phase gate
Sdag = np.array([[1., 0.], [0., -1.j]]) # S^\dag

####### Main circuit #######
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
        tau = helpers.rescale_ham_slow(ham)

    # apply Hadamard gate onto ancilla first
    ancilla = np.array([1, 0])
    ancilla = np.dot(Hd, ancilla)
    full_state = np.kron(ancilla, state)

    # apply the controlled time evolution
    expH = sl.expm(-1.j * j * tau * ham) # exponential of a matrix
    c_expH = control_op_1q(expH)
    full_state = np.dot(c_expH, full_state)

    # apply W gate to ancilla
    W = I
    if id == 'Y':
        W = Sdag
    full_W = np.kron(W, I)
    full_state = np.dot(full_W, full_state)

    # apply Hadamard gate to ancilla
    full_Hd = np.kron(Hd, I)
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
    cnot = control_op_1q(X)
    print(cnot)
    
    # run main_circuit_1qubit
    state = np.random.rand(2)
    state /= np.linalg.norm(state)
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    ew, ev = np.linalg.eigh(ham)
    tau = np.pi/(3 * max(abs(ew[0]), abs(ew[1])))
    j = 1
    W = I
    full_state = main_circuit_1q(state, ham, tau, j, W)
    print(full_state)

    # run measure_ancilla()
    props = measure_ancilla(full_state)
    print(props)