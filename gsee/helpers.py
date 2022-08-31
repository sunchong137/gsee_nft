from operator import mod
from tkinter.font import names
import numpy as np
from scipy.special import eval_chebyt
from scipy import integrate
from scipy import fft
from scipy import signal

pi = np.pi

def dft_coeffs_approx_heaviside(d, delt, k, nmesh=100):
    '''
    Evaluate the Fourier coefficients the approximate heaviside function.
    '''
    Mk = dft_coeffs_smear_dirac(d, delt, k, nmesh)
    Hk = dft_coeffs_heaviside(k)
    Fk = np.sqrt(2 * pi) * Mk * Hk
    return Fk

def approx_heaviside_from_dft(d, delt, x):
    '''
    Evaluate the approximate Heaviside at x.
    '''
    k = np.arange(-d, d+1)
    Fk = dft_coeffs_approx_heaviside(d, delt, k, nmesh=200)
    lk = 2 * d + 1
    lx = len(x)
    F = np.dot(Fk, np.exp(1.j * np.kron(k, x).reshape(lk, lx))) / np.sqrt(2 * pi)
    # try: # x is an array
    #     lx = len(x)
    #     F = np.dot(Fk, np.exp(1.j * np.kron(k, x))) / np.sqrt(2 * pi)
    #except:
    #    F = np.dot(Fk, np.exp(1.j * k * x)) / np.sqrt(2 * pi)
    return F

def approx_heaviside_from_convol(d, delt, nmesh=200):
    x = np.linspace(-pi, pi, nmesh+1, endpoint=True)
    M = eval_smear_dirac(d, delt, nmesh)
    H = heaviside(nmesh)
    F = np.convolve(M, H, mode="same")
    #F = signal.fftconvolve(M, H, mode="same")
    return F

def eval_chebyshev_slow(x, d):
    '''
    Evaluating the d-th order of Chebyshev polynomial of the first kind 
    based on the recursive relation. 
    Compare to scipy.special.eval_chebyt()
    Args:
        x: 1d array of N points at which the polynomials are evaluated.
        d: up to what order to return.
    Return:
        1d array of N points of T(x) values.
    '''
    lx = len(x)
    t0 = np.ones(lx)
    t1 = np.copy(x)
    if d < 2:
        if d == 1:
            return t1
        else:
            return t0
    
    for i in range(d-1):
        t = 2 * x * t1 - t0 
        t0 = t1.copy()
        t1 = t.copy()
    return t

def eval_smear_dirac(d, delt, nmesh, thr=1e-5):
    '''
    Evaluate smear Dirac delta function in Lemma 5 at [-pi, pi].
    Args:
        d: degree of Chebyshev polynomials
        delt: parameter
        nmesh: number of mesh points (assume uniform grid for now)
    Returns:
        1d array of the function values.
    '''
    assert abs(delt - pi) > thr and abs(delt + pi) > thr # no overflow!
    x = np.linspace(-pi, pi, nmesh+1, endpoint=True)
    x_n = 1 + 2 * (np.cos(x) - np.cos(delt)) / (1 + np.cos(delt))
    M = eval_chebyt(d, x_n)
    #M /= (np.sum(M[:-1]) * 2 * pi / nmesh) # normalization, the integration is done
    M /= integrate.simpson(M, x)

    return M

def heaviside(nmesh):
    '''
    Heaviside function from [-pi, pi]
    '''
    H = np.ones(nmesh + 1)
    H[:int(nmesh//2)] *= 0

    return H

def eval_dft_coeffs_slow(y, k, x=None):
    '''
    Slow version of discrete Fourier transform. 
    Compare to scipy.fft.fft()
    Args:
        x: original space points in [-pi, pi]
        y: f(x)
        k: reciprical space point(s), 1d array
    Returns:
        Fourier coefficients of f(x) at k.
    '''
    lx = len(y)
    if x is None:
        x = np.linspace(-pi, pi, lx)
    try: # k is provided as an array
        lk = len(k)
        #lx = len(x)
        n = np.exp(-1.j * np.einsum('i, j -> ij', k, x))
        n = np.einsum('i, ji -> ji', y, n)
        #x_n = np.kron(np.ones(lk), x).reshape(lk, lx)
        yk = integrate.simpson(n, x) / np.sqrt(2 * pi)
        #print("k is an array!")
        
    except:
        #print("k is given as a number!")
        n = y * np.exp(-1.j * k * x)
        yk = integrate.simpson(n, x) / np.sqrt(2 * pi)
    
    return yk

def dft_coeffs_smear_dirac(d, delt, k, nmesh, thr=1e-5):
    '''
    Fourier transform of the smear dirac function.
    '''
    Mx = eval_smear_dirac(d, delt, nmesh, thr=thr)
    Mk = eval_dft_coeffs_slow(Mx, k)
    return Mk

def dft_coeffs_heaviside(k):
    '''
    FFT of the periodic Heaviside function.
    Args:
        k: int or array of int.
    Returns:
        complex number or array of complex number.
    '''
    try:
        lk = len(k)
        Hk = np.zeros(lk)
        Hk = -2.j * (k % 2) / (np.sqrt(2 * pi) * k) # complain when k = 0
        Hk[np.where(k==0)] = np.sqrt(pi / 2.)
    # try:
    #     lk = len(k)
    #     Hk = np.zeros(lk)
    #     for i in range(lk):
    #         if k[i] == 0:
    #             Hk[i] = np.sqrt(pi / 2.)
    #         elif k[i] % 2 == 0:
    #             Hk[i] = 0
    #         else:
    #             Hk[i] = (-2.j / (np.sqrt(2 * pi) * k[i]))    

    except:
        if k == 0:
            Hk = np.sqrt(pi / 2.)
        elif k % 2 == 0:
            Hk = 0
        else:
            Hk = (-2.j / (np.sqrt(2 * pi) * k))

    return Hk

def rescale_hamiltonian_slow(ham, bound=pi/3):
    '''
    Rescaling the hamiltonian, returns the rescaling factor tau.
    Suppose we can diagonalize the Hamiltonian.
    Args:
        ham - the array of the Hamiltonian
        bound - the upper limit of tau * ||ham||
    Returns:
        a double number - tau
    '''
    ew, _ = np.linalg.eigh(ham)
    tau = bound / max(abs(ew[0]), abs(ew[-1]))

    return tau


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    print("done")


