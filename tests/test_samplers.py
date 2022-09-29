from random import sample
import unittest
import numpy as np
from gsee import samplers
from scipy import linalg as sl
from gsee import quantum_circuits

class TestSamplers():
    def test_dft_order_sampler(self):
        
        num_samples = 20000
        max_dft_order = 2
        dft_order_range = np.arange(-max_dft_order, max_dft_order+1, 1)
        
        dft_coeffs = np.array([1.j+1, 0.5j+1, 0, 2.0, 0.4j])
        abs_dft_coeffs = np.array([np.sqrt(2), np.sqrt(1 + 0.25), 0, 2, 0.4])
        dft_orders = samplers.dft_order_sampler(num_samples, max_dft_order, dft_coeffs)
        
        n_orders = len(dft_order_range)
        counts = np.zeros(n_orders)
        for i in range(n_orders):
            counts[i] = np.count_nonzero(dft_orders == dft_order_range[i])
        diff = np.linalg.norm(counts / num_samples - abs_dft_coeffs / np.sum(abs_dft_coeffs))
        assert diff < 5e-2
        
if __name__ == "__main__":
    obj = TestSamplers()
    obj.test_dft_order_sampler()