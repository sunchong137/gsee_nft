import numpy as np
from scipy import linalg as sl
import random

# Define single-qubit gates
X = np.array([[0., 1.], [1., 0.]])
Y = 1.j * np.array([[0., -1.], [1., 0.]])
Z = np.array([[1., 0.], [0., -1.]])
I  = np.eye(2) # identity gate
Hd = np.array([[1., 1.], [1., -1.]])/np.sqrt(2) # Hadamard gate
S = np.array([[1., 0.], [0., 1.j]]) # phase gate
Sdag = np.array([[1., 0.], [0., -1.j]]) # S^\dag

####### Main circuit #######
def main_circuit_1qubit(state, ham, tau, j, id="X"):
    '''n circuit for the one qubit case.
    Main circuit. Eq.(1) in the paper
    Args:
        ham  : hamiltonian
        ham: Hamiltonian, 2x2 array
        tau: rescaling factor
        j  : a number I am not sure what it is # TODO
        id  : measure the real part or imaginary part.
    Return:
        The entangled state of the ancilla qubit and state qubit.
    '''
    # apply Hadamard gate onto ancilla first
    ancilla = np.array([1, 0])
    ancilla = np.dot(Hd, ancilla)
    full_state = np.kron(ancilla, state)

    # apply the controlled time evolution
    expH = sl.expm(-1.j * j * tau * ham) # exponential of a matrix
    c_expH = control_time_evolve_1qubit(expH)
    full_state = np.dot(c_expH, full_state)

    # apply W gate to ancilla
    if id == 'X':
        W = I
    elif id == 'Y':
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
    Since we have the full state information, we will give the 
    exact probablities of getting state |0> and |1>
    '''
    prop_0 = np.linalg.norm(full_state[0])**2 + np.linalg.norm(full_state[1])**2
    prop_1 = np.linalg.norm(full_state[2])**2 + np.linalg.norm(full_state[3])**2

    # mimic the collapsing
    a =  random.uniform(0, 1)
    if a < prop_0:
        return 0
    else:
        return 1

    #return np.array([prop_0, prop_1])


def control_time_evolve_1qubit(op):
    '''
    Given a single qubit operator op, return controlled-op.
    For single qubit, the formula is simply 
    [[I , 0 ], [0, op]]
    '''
    c_op = np.eye(4) + 1.j * np.zeros((4, 4))
    c_op[2:, 2:] = op

    return c_op


if __name__ == "__main__":

    # run control_time_evolve_1qubit
    cnot = control_time_evolve_1qubit(X)
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
    full_state = main_circuit_1qubit(state, ham, tau, j, W)
    print(full_state)

    # run measure_ancilla()
    props = measure_ancilla(full_state)
    print(props)