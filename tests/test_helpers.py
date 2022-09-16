import numpy as np
from gsee import helpers
from matplotlib import pyplot as plt
from scipy.special import eval_chebyt
from scipy import integrate
import time

class TestHelpers():


    def test_eval_dft_coeffs_slow(self):
        # test on sin and cosine functions
        w = 1.0
        grids = np.linspace(-np.pi, np.pi, 1000)
        def sin(x):
            return np.sin(w * x)
        fx = sin(grids)
        def dft_sin(k):
            if k == w:
                tmp1 = np.pi
            else:
                tmp1 = np.sin(np.pi * (w - k)) / (w - k)
            if k == -w:
                tmp2 = np.pi
            else:
                tmp2 = np.sin(np.pi*(w + k)) /(w + k)
            return -1.j / np.sqrt(2 * np.pi) * (tmp1 - tmp2)
        dft_grids = np.array([-2*w, -w, 0, w, -2*w])
        dft_f = helpers._eval_dft_coeffs_slow(fx, dft_grids, grids)
        dft_reference = []
        for k in dft_grids:
            dft_reference.append(dft_sin(k))
        
        diff = np.linalg.norm(dft_f - dft_reference)
        assert np.linalg.norm(diff) < 1e-5

    def test_eval_smear_dirac(self):
        nmesh = 200
        d = 20
        delt = 0.2
        x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
        M = helpers._eval_smear_dirac(d, delt, nmesh)
        
        # make sure that the integration is 1
        area = integrate.simpson(M, x)
        assert(abs(area - 1) < 1e-5)
                
    def test_dft_coeffs_smear_dirac(self):
        nmesh = 10
        d = 20
        delt = 0.2
        k = np.arange(-nmesh,nmesh+1,1)
        Mk = helpers._eval_dft_coeffs_smear_dirac(d, delt, k, nmesh)
        print(Mk)

    def test_dft_coeffs_heaviside(self):
        k = np.arange(-4, 5)
        Hk = helpers.dft_coeffs_heaviside(k)
        print(Hk)
        Hks = []
        for i in k:
            Hks.append(helpers.dft_coeffs_heaviside(i))
        Hks = np.asarray(Hks)
        print(np.linalg.norm(Hks - Hk))
        print(k)

    def test_dft_coeffs_approx_heaviside(self):
        nmesh = 10
        d = 20
        delt = 0.2
        k = np.arange(-nmesh,nmesh+1,1)
        Fk = helpers.dft_coeffs_approx_heaviside(d, delt, k, nmesh)
        print(Fk)
        

    def test_approx_heaviside_from_dft(self):
        nmesh = 40
        d = 20
        delt = 0.2
        x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
        F = helpers.approx_heaviside_from_dft(d, delt, x)
        print(F)
        plt.plot(x, F)
        # plt.show()

    def test_approx_heaviside_from_convol(self):
        nmesh = 40
        d = 20
        delt = 0.2
        x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
        F = helpers.approx_heaviside_from_convol(d, delt, nmesh)
        plt.plot(x, F)
        # plt.show()

    def compare_aheaviside(self):
        nmesh = 80
        d = 40
        delt = 0.2
        x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
        F1 = helpers.approx_heaviside_from_convol(d, delt, nmesh)*(2*np.pi/(nmesh+1))
        F2 = helpers.approx_heaviside_from_dft(d, delt, x)
        plt.plot(x, F1, label="convol")
        plt.plot(x, F2, '--', label="FT")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$F_{d, \delta}$")
        plt.legend()
        #plt.savefig("figures/aheaviside_compare.png", dpi=300)

    def test_rescale_hamiltonian_spectrum(self):
        ham = np.random.rand(4,4)
        tau = helpers.rescale_hamiltonian_spectrum(ham)
        print(tau)

if __name__ == "__main__":
    obj = TestHelpers()
    obj.test_eval_dft_coeffs_slow()