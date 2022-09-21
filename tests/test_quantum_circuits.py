from gsee import quantum_circuits
from gsee import gates
import numpy as np
import time


class TestQuantumCircuits():
    
    def test_control_time_evolve_1qubit(self):
        # generate a random Hamiltonian
        ham = np.zeros((2, 2))
        ham[0, 0] = 1
        ham[1, 1] = -1
        # generate a random initial state
        state = np.array([1., -1.])
        state /= np.linalg.norm(state)
        j = 1
        full_state = quantum_circuits.control_time_evolve_1qubit(state, ham, j, id="X", energy_bound=np.pi/3.)
        ref_state = np.array([ 0.53033009-0.30618622j, 
                              -0.53033009-0.30618622j,  
                               0.1767767 +0.30618622j,
                              -0.1767767 +0.30618622j])
        assert np.linalg.norm(full_state - ref_state) < 1e-6
        
        full_state = quantum_circuits.control_time_evolve_1qubit(state, ham, j, id="Y", energy_bound=np.pi/3.)
        ref_state = np.array([0.04736717-0.1767767j,
                             -0.65973961+0.1767767j,
                              0.65973961+0.1767767j,
                             -0.04736717-0.1767767j])
        assert np.linalg.norm(full_state - ref_state) < 1e-6

    def test_measure_ancilla(self):

        # simple test
        state = np.array([1.+1.j, 1-1.j, 0, 0])
        state /= np.linalg.norm(state)
        outcome = quantum_circuits.measure_ancilla(state)
        assert abs(outcome - 0) < 1e-10

        # complicated one
        state_r = np.random.rand(4)
        state_i = np.random.rand(4)
        state = state_r + 1.j * state_i
        state /= np.linalg.norm(state)
        prop_0 = np.linalg.norm(state[0])**2 + np.linalg.norm(state[1])**2

        # repeat the sampling
        N0, N1 = 0, 0
        Ns = 100000
        for i in range(Ns):
            out = quantum_circuits.measure_ancilla(state)
            if abs(out - 1) < 1e-10:
                N1 += 1
            elif abs(out) < 1e-10:
                N0 += 1
            else:
                raise ValueError("The measurement outcome should be 0 or 1!")

        p0 = 1. * N0 / Ns
        assert np.abs(p0 - prop_0) < 1e-2


    def test_measure_Xj_1q(self):
        # generate a random Hamiltonian
        ham = np.random.rand(2, 2)
        ham = 0.5 * (ham + ham.T)
        # generate a random initial state
        state = np.random.rand(2)
        state /= np.linalg.norm(state)
        j = 2
        Ns = 100
        X = []
        for i in range(Ns):
            X.append(quantum_circuits.measure_Xj_1q(state, ham, j))
        print(X)


    def test_measure_Yj_1q(self):
        # generate a random Hamiltonian
        ham = np.random.rand(2, 2)
        ham = 0.5 * (ham + ham.T)
        # generate a random initial state
        state = np.random.rand(2)
        state /= np.linalg.norm(state)
        j = 2
        Ns = 100
        Y = []
        for i in range(Ns):
            Y.append(quantum_circuits.measure_Yj_1q(state, ham, j))
        print(Y)


    def test_control_op_1q(self):
        cx = quantum_circuits.control_op_1q(gates.X)
        diff = np.linalg.norm(cx - gates.CNOT)
        assert diff < 1e-15

   

        
    def test_rescale_hamiltonian_spectrum(self):
        bound = np.pi / 3 
        ham = np.zeros((2, 2))
        ham[0, 0] = -1
        ham[1, 1] = 0.5
        tau = quantum_circuits.rescale_hamiltonian_spectrum(ham)
        ref_tau = bound / 1
        assert abs(tau - ref_tau) < 1e-10

if __name__ == "__main__":
        obj = TestQuantumCircuits()
        obj.test_measure_ancilla()