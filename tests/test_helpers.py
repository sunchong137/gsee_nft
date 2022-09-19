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
        
        fx = sin(grids)
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
        M = helpers._eval_smear_dirac(d, delt, grids=x)
        # make sure that the integration is 1
        area = integrate.simpson(M, x)
        assert(abs(area - 1) < 1e-5)
                

    def test_eval_dft_coeffs_heaviside(self):
        dft_grids = np.arange(-3, 4)
        Hk0 = helpers._eval_dft_coeffs_heaviside(dft_grids)
        # compare to the result from _eval_dft_coeffs_slow
        nmesh = 30000
        grids = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
        heaviside = helpers._gen_heaviside(nmesh=nmesh)
        Hk_slow = helpers._eval_dft_coeffs_slow(heaviside, dft_grids, grids)
        diff = np.linalg.norm(Hk0 - Hk_slow)
        assert diff < 1e-4
          
    
    def test_dft_coeffs_approx_heaviside(self):
        k = 0
        nmesh = 1000
        d = 20
        delt = 0.2
        ah_dft = helpers.dft_coeffs_approx_heaviside(d, delt, k, nmesh)
        assert abs(ah_dft - np.sqrt(np.pi/2)) < 1e-10
 
        k = 4
        ah_dft = helpers.dft_coeffs_approx_heaviside(d, delt, k, nmesh)
        assert abs(ah_dft) < 1e-10
        
        k = 1
        ah_dft = helpers.dft_coeffs_approx_heaviside(d, delt, k, nmesh)
        ah_ref = 5.536436314976699e-18-0.795265159938173j
        assert np.linalg.norm(ah_dft - ah_ref) < 1e-10
        


if __name__ == "__main__":
    # to run a specific test
    obj = TestHelpers()
    obj.test_dft_coeffs_approx_heaviside()