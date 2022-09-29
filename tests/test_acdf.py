from gsee import acdf
from gsee.helpers import *
from gsee.quantum_circuits import *
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi

class TestACDF():

    def test_classical_sampler(self):
        # Test the sampling procedure
        num_samples = 20000
        max_dft_order = 2 # [-2, -1, 0, 1, 2]
        rescaled_energy_acc = 0.1
        nmesh = 5000
        dft_order_range = np.arange(-max_dft_order, max_dft_order+1, 1)

        
        # Test if DFT coefficients of the heaviside are not given
        dft_coeffs, dft_orders = acdf.classical_sampler(num_samples, max_dft_order, rescaled_energy_acc, nmesh=nmesh)
        abs_dft_coeffs = np.abs(dft_coeffs)
        n_orders = len(dft_order_range)
        counts = np.zeros(n_orders)
        for i in range(n_orders):
            counts[i] = np.count_nonzero(dft_orders == dft_order_range[i])
        diff = np.linalg.norm(counts / num_samples - abs_dft_coeffs / np.sum(abs_dft_coeffs))
        assert diff < 5e-2
        
        
    def test_eval_acdf_single_sample(self):
        x = np.linspace(-pi / 3, pi / 3, 100)
        F_tot = 2.0
        j = 2
        Xj = 1.0
        Yj = -1.0
        Zj = Xj + 1.0j * Yj
        ang_j = 0.1
        G = acdf.eval_acdf_single_sample(x, j, Zj, ang_j)
        print(G.shape)


    def test_acdf_kernel(self):
        Ns = 10000
        d = 40
        delt = 0.2
        nmesh = 500
        max_x = pi / 2
        x = np.linspace(-max_x, max_x, nmesh)
        # generate a random Hamiltonian
        ham = np.random.rand(2, 2)
        ham = 0.5 * (ham + ham.T)

        # rescale hamiltonian
        tau = rescale_hamiltonian_spectrum(ham, bound=np.pi/3)
        ham *=  tau

        ew, ev = np.linalg.eigh(ham)
        state = ev[0] + np.random.rand(2) * 0.5  # make a good initial guess
        state /= np.linalg.norm(state)
        print("eigenvalues - E1: {}; E2: {}".format(ew[0], ew[1]))
        p0 = np.dot(ev[0].T, state) ** 2
        print("The overlap between the initial state and the ground state: {}".format(p0))
        # generate a random initial state
        Fj, j_samp = acdf.classical_sampler(Ns, d, delt, nmesh=nmesh)
        Z_samp = acdf.quantum_sampler(j_samp, state, ham, energy_rescalor=tau)

        G_bar = acdf.acdf_kernel(d, Fj, j_samp, Z_samp, energy_grid=x, nmesh=nmesh)


if __name__ == "__main__":
    obj = TestACDF()
    obj.test_classical_sampler()