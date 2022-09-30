from gsee import acdf
from gsee import helpers
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
        x = np.array([-0.1, 0., 0.1])
        j = 2
        Zj = 1 + 1.0j
        ang_j = 0.1
        G = acdf.eval_acdf_single_sample(x, j, Zj, ang_j)
        ref = np.array([1.094837581924854+0.8951707486311975j, 
                        0.8951707486311975+1.094837581924854j,
                        0.6598162824642664+1.2508566957869456j])
        diff = np.linalg.norm(G - ref)
        assert  diff < 1e-10


    def test_acdf_kernel(self):
        num_samples = 3
        max_dft_order = 1
        dft_filter_coeffs = np.array([0.1*np.exp(1.j), 
                                      0.5*np.exp(0.2j),
                                      1.0*np.exp(-0.1j)])
        dft_orders_sample = np.array([0, -1, 1])
        ham_evo_sample = np.array([1.+1.j, 1.-1.j, -1.-1.j])
        energy_grid = np.array([-0.1, 0.1])
        abs_dft = np.array([0.1, 0.5, 1.0])
        angle_sample = np.array([0.2, 1., -0.1])
        f_tot = np.sum(abs_dft)
        
        G_ref_1 = np.sum(ham_evo_sample * np.exp(1.j * (angle_sample - dft_orders_sample * 0.1))) * f_tot / 3
        G_ref_2 = np.sum(ham_evo_sample * np.exp(1.j * (angle_sample + dft_orders_sample * 0.1))) * f_tot / 3
        G_ref = np.array([G_ref_1, G_ref_2])
        
        G = acdf.acdf_kernel(max_dft_order,
                             dft_filter_coeffs, 
                             dft_orders_sample, 
                             ham_evo_sample, 
                             energy_grid=energy_grid)
        
    
        diff = np.linalg.norm(G - G_ref)
        assert diff < 1e-8
        
    def test_gen_acdf_exact(self):
        
        max_dft_order = 40
        dft_order_range = np.arange(-max_dft_order, max_dft_order+1, 1)
        rescaled_energy_acc = 0.01
        # Set the Hamiltonian
        # Generate a random real one-qubit Hamiltonian
        # hamiltonian = np.random.rand(2, 2)
        hamiltonian = np.array([[1, 0], [0, -0.7]])
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)

        # Solve the Hamiltonian
        ew, ev = np.linalg.eigh(hamiltonian)
        energy_rescalor = np.pi / 3

        # generate the initial guess based on the ground state ev[:0]
        # state = ev[:, 0] + np.random.rand(2) * init_guess_noise
        state = np.array([0.5, 0.5])
        state /= np.linalg.norm(state)
        
        dft_coeffs = helpers.dft_coeffs_approx_heaviside(
        max_dft_order, rescaled_energy_acc, dft_order_range, nmesh=2000
    )
        
        
        dft_coeffs = helpers._eval_dft_coeffs_heaviside(dft_order_range)
        
        G = acdf.gen_acdf_exact(dft_coeffs, state, hamiltonian, max_dft_order, energy_rescalor, nmesh=1000)
        
        G = np.abs(G)
        print(max(G))
#
        energy_grid = np.linspace(-np.pi/2, np.pi/2, 1000, endpoint=True)
        plt.plot(energy_grid, G/np.sqrt(2*np.pi))
        plt.show()
        
if __name__ == "__main__":
    obj = TestACDF()
    obj.test_gen_acdf_exact()