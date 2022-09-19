from gsee.quantum_circuits import *
from gsee import gates
import numpy as np
import time


class TestQuantumCircuits():

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
            X.append(measure_Xj_1q(state, ham, j))
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
            Y.append(measure_Yj_1q(state, ham, j))
        print(Y)


    def test_control_op_1q(self):
        cx = control_op_1q(gates.X)
        diff = np.linalg.norm(cx - gates.CNOT)
        assert diff < 1e-15

    def test_measure_ancilla(self):
        state_r = np.random.rand(4)
        state_i = np.random.rand(4)
        state = state_r + 1.j * state_i
        state /= np.linalg.norm(state)
        prop_0 = np.linalg.norm(state[0])**2 + np.linalg.norm(state[1])**2
        prop_1 = np.linalg.norm(state[2])**2 + np.linalg.norm(state[3])**2
        assert (prop_0 + prop_1) - 1 < 1e-10

        # repeat the sampling
        N0, N1 = 0, 0
        Ns = 100000
        t1 = time.time()
        for i in range(Ns):
            out = measure_ancilla(state)
            if abs(out - 1) < 1e-10:
                N1 += 1
            elif abs(out) < 1e-10:
                N0 += 1
            else:
                raise ValueError("The measurement outcome should be 0 or 1!")
        t2 = time.time()
        print("time used to draw {} samples: {}".format(Ns, t2 - t1))
        p0 = 1. * N0 / Ns
        p1 = 1. * N1 / Ns

        assert np.abs(p0 - prop_0) < 1e-2
        print("Diff: {}".format(np.abs(p0 - prop_0)))

    def test_control_time_evolve_1q(self):
        # generate a random Hamiltonian
        ham = np.zeros((2, 2))
        ham = 0.5 * (ham + ham.T)
        # generate a random initial state
        state = np.random.rand(2)
        state /= np.linalg.norm(state)
        j = 2
        full_state = control_time_evolve_1q(state, ham, j)
        print(full_state) 
        assert abs(np.linalg.norm(full_state) - 1) < 1e-10
        
    def test_rescale_hamiltonian_spectrum(self):
        bound = np.pi / 3 
        ham = np.zeros((2, 2))
        ham[0, 0] = -1
        ham[1, 1] = 0.5
        tau = rescale_hamiltonian_spectrum(ham)
        ref_tau = bound / 1
        assert abs(tau - ref_tau) < 1e-10

if __name__ == "__main__":
        obj = TestQuantumCircuits()
        #obj.test_control_time_evolve_1q()
